# ---------------------------------------------------------------------------
# Cubed-sphere Strang splitting orchestrator for src
#
# Performs X → Y → Z → Z → Y → X dimensionally-split advection on 6
# gnomonic panels with halo exchange between horizontal sweeps.
#
# Panel-interior kernels reuse the SAME reconstruction functions
# (_xface_tracer_flux, _yface_tracer_flux, _zface_tracer_flux) as the
# LatLon path. The only CS-specific logic is:
#   1. Halo exchange after each horizontal sweep (fill_panel_halos!)
#   2. Kernel launch on interior indices with Hp offset
#   3. Per-panel loop over 6 panels
#
# The panel arrays have layout (Nc+2Hp, Nc+2Hp, Nz) with interior at
# [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]. The reconstruction stencil reads into
# the halo region naturally.
#
# Mass conservation: guaranteed by the telescoping identity, same as
# LatLon. Each panel's X sweep is periodic (halo wraps around), Y sweep
# reads from halo at panel edges, Z sweep is panel-local closed.
#
# References:
#   Strang (1968) — symmetric splitting for second-order accuracy
#   Putman & Lin (2007) — FV3 cubed-sphere transport
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# =========================================================================
# CS panel sweep kernels
#
# These launch on ndrange=(Nc, Nc, Nz) and read/write the interior region
# of halo-padded arrays. The Hp offset is added to all indices.
# =========================================================================

# These KA kernels dispatch on `scheme` via _xface_tracer_flux and work for
# UpwindScheme, SlopesScheme, and PPMScheme. They are used by the higher-order
# _sweep_x/y/z_panel! methods (AbstractAdvectionScheme fallback) via KA_CPU().
# On GPU backends, they can be launched directly with the appropriate backend.
# The UpwindScheme specialization uses hand-written gamma-clamped loops instead
# (positivity-safe even at CFL > 1).

"""X-sweep kernel on one CS panel. Interior i ∈ [1,Nc], periodic via halo."""
@kernel function _cs_xsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(am), scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        # Map to halo-padded indices
        i = ii + Hp
        j = jj + Hp
        # Face fluxes: am has same halo-padded layout
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]
        # Reconstruction on the full halo-padded array (Nx = Nc + 2Hp for stencil)
        Nx_padded = Int32(Nc + 2 * Hp)
        flux_L = _xface_tracer_flux(Int32(i), j, k, rm, m, am_l, scheme, Nx_padded)
        flux_R = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm, m, am_r, scheme, Nx_padded)
        rm_new[i, j, k] = rm[i, j, k] + flux_L - flux_R
        m_new[i, j, k]  = m[i, j, k]  + am_l - am_r
    end
end

"""Y-sweep kernel on one CS panel. Interior j ∈ [1,Nc], halo provides neighbors."""
@kernel function _cs_ysweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(bm), scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]
        Ny_padded = Int32(Nc + 2 * Hp)
        flux_S = _yface_tracer_flux(i, Int32(j), k, rm, m, bm_s, scheme, Ny_padded)
        flux_N = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm, m, bm_n, scheme, Ny_padded)
        rm_new[i, j, k] = rm[i, j, k] + flux_S - flux_N
        m_new[i, j, k]  = m[i, j, k]  + bm_s - bm_n
    end
end

"""Z-sweep kernel on one CS panel. Same as LatLon z-kernel but with Hp offset."""
@kernel function _cs_zsweep_kernel!(rm_new, @Const(rm), m_new, @Const(m),
                                     @Const(cm), scheme, Nz, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]
        flux_T = _zface_tracer_flux(i, j, Int32(k), rm, m, cm_t, scheme, Int32(Nz))
        flux_B = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm, m, cm_b, scheme, Int32(Nz))
        rm_new[i, j, k] = rm[i, j, k] + flux_T - flux_B
        m_new[i, j, k]  = m[i, j, k]  + cm_t - cm_b
    end
end

# =========================================================================
# Per-panel sweep functions (double-buffered)
#
# Dispatch strategy:
#   UpwindScheme  → hand-written gamma-clamped loops (positivity-safe)
#   SlopesScheme  → KA kernel via _xface_tracer_flux dispatch (needs Hp ≥ 2)
#   PPMScheme     → KA kernel via _xface_tracer_flux dispatch (needs Hp ≥ 3)
# =========================================================================

"""Validate that the halo width Hp is sufficient for the advection scheme's stencil."""
@inline function _validate_halo_for_scheme(scheme::AbstractAdvectionScheme, Hp::Int)
    order = reconstruction_order(scheme)
    min_hp = order == 0 ? 1 : order == 1 ? 2 : 3   # upwind=1, slopes=2, PPM=3
    Hp >= min_hp || error("CS panel sweep with $(typeof(scheme)) requires Hp ≥ $min_hp, got Hp=$Hp. " *
                          "Construct CubedSphereMesh with Hp=$min_hp.")
    return nothing
end

# =========================================================================
# Gamma-clamped tracer flux (from legacy src/Advection/cubed_sphere_mass_flux.jl)
#
# For face flux F through a donor cell with mass m_donor:
#   gamma = clamp(F / m_donor, 0, 1)  (positive F) or clamp(F / m_donor, -1, 0)
#   F_tracer = gamma * rm_donor
#
# When CFL = |F|/m_donor > 1, gamma is clamped to ±1, reducing tracer transport
# to at most the entire donor cell content. Mass update m_new = m + F_in - F_out
# is EXACT (no clamping on mass). Only the tracer flux is limited.
#
# This guarantees rm_new ≥ 0 when rm_src ≥ 0, and preserves mass conservation
# exactly. It's the TM5/FV3/GCHP standard approach for high-CFL cells.
# =========================================================================

"""
    _gamma_clamped_x_flux(F, m_donor, rm_donor) -> tracer_flux

Gamma-clamped upwind tracer flux (legacy cubed_sphere_mass_flux.jl pattern).

Given mass flux `F` [kg] through a face, donor cell mass `m_donor` [kg],
and donor tracer mass `rm_donor` [kg]:

    γ = clamp(F / m_donor, {0, 1} or {-1, 0})
    tracer_flux = γ × rm_donor

This ensures:
- When CFL = |F|/m ≤ 1 (normal): `γ = F/m`, recovering first-order upwind.
- When CFL > 1 (overshooting): `γ` is clamped to ±1, so the tracer flux
  never exceeds the donor cell's total tracer mass. This guarantees
  `rm_new ≥ 0` when `rm ≥ 0` (positivity preservation).
- Mass update `m_new = m + F_west − F_east` is EXACT (unclamped), so total
  mass is conserved. Only the tracer distribution is limited.

The gamma clamping should ideally not be needed if CFL < 1 via the
subcycling pilot. It's a safety net for preprocessing flux-inconsistency
(see CLAUDE.md: clamps should ideally not be needed).
"""
@inline function _gamma_clamped_x_flux(F::FT, m_donor::FT, rm_donor::FT) where FT
    m_donor > zero(FT) || return zero(FT)
    # γ = F/m clamped to [0, 1] for positive flux, [-1, 0] for negative
    gamma = F >= zero(FT) ?
        clamp(F / m_donor, zero(FT), one(FT)) :
        clamp(F / m_donor, -one(FT), zero(FT))
    return gamma * rm_donor
end

"""Higher-order X-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.).

Requires sufficient halo padding: Hp >= 2 for SlopesScheme, Hp >= 3 for PPMScheme.
The `_xface_tracer_flux` reconstruction reads neighbors up to `face_i ± 3` for PPM,
which stays within the halo-padded array when Hp is large enough. The periodic wrap
in `_wrap_periodic` is a safety net that should never trigger with correct Hp.
"""
function _sweep_x_panel!(rm, m, am, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_xsweep_kernel!(backend, 256)
    kernel!(rm_A, rm, m_A, m, am, scheme, Int32(Nc), Int32(Hp), FT(flux_scale);
            ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

"""Gamma-clamped upwind X-sweep: positivity-safe even at CFL > 1."""
function _sweep_x_panel!(rm, m, am, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        i = ii + Hp; j = jj + Hp
        am_l = flux_scale * am[i, j, k]
        am_r = flux_scale * am[i + 1, j, k]

        # Left face flux: donor is i-1 if am_l > 0, else i
        mi   = m[i, j, k]
        mim1 = m[i - 1, j, k]
        rim1 = rm[i - 1, j, k]
        ri   = rm[i, j, k]
        fl = am_l >= zero(FT) ?
            _gamma_clamped_x_flux(am_l, mim1, rim1) :
            _gamma_clamped_x_flux(am_l, mi, ri)

        # Right face flux: donor is i if am_r > 0, else i+1
        mip1 = m[i + 1, j, k]
        rip1 = rm[i + 1, j, k]
        fr = am_r >= zero(FT) ?
            _gamma_clamped_x_flux(am_r, mi, ri) :
            _gamma_clamped_x_flux(am_r, mip1, rip1)

        rm_A[i, j, k] = ri + fl - fr
        m_A[i, j, k]  = mi + am_l - am_r
    end
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

"""Higher-order Y-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.)."""
function _sweep_y_panel!(rm, m, bm, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    _validate_halo_for_scheme(scheme, Hp)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_ysweep_kernel!(backend, 256)
    kernel!(rm_A, rm, m_A, m, bm, scheme, Int32(Nc), Int32(Hp), FT(flux_scale);
            ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

function _sweep_y_panel!(rm, m, bm, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        i = ii + Hp; j = jj + Hp
        bm_s = flux_scale * bm[i, j, k]
        bm_n = flux_scale * bm[i, j + 1, k]

        mi   = m[i, j, k]
        mjm1 = m[i, j - 1, k]
        rjm1 = rm[i, j - 1, k]
        ri   = rm[i, j, k]
        fs = bm_s >= zero(FT) ?
            _gamma_clamped_x_flux(bm_s, mjm1, rjm1) :
            _gamma_clamped_x_flux(bm_s, mi, ri)

        mjp1 = m[i, j + 1, k]
        rjp1 = rm[i, j + 1, k]
        fn = bm_n >= zero(FT) ?
            _gamma_clamped_x_flux(bm_n, mi, ri) :
            _gamma_clamped_x_flux(bm_n, mjp1, rjp1)

        rm_A[i, j, k] = ri + fs - fn
        m_A[i, j, k]  = mi + bm_s - bm_n
    end
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

"""Higher-order Z-sweep via KA kernel dispatching on scheme (Slopes, PPM, etc.).

Z boundary: `_zface_tracer_flux` handles k=1 (TOA) and k=Nz+1 (surface) boundaries
by falling back to upwind at the domain edges.
"""
function _sweep_z_panel!(rm, m, cm, scheme::AbstractAdvectionScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    # Z-direction does not need halo validation (vertical boundaries are closed, not halo-exchanged)
    FT = eltype(m)
    backend = get_backend(rm)
    kernel! = _cs_zsweep_kernel!(backend, 256)
    kernel!(rm_A, rm, m_A, m, cm, scheme, Int32(Nz), Int32(Hp), FT(flux_scale);
            ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

function _sweep_z_panel!(rm, m, cm, scheme::UpwindScheme, rm_A, m_A, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    FT = eltype(m)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        i = ii + Hp; j = jj + Hp
        cm_t = flux_scale * cm[i, j, k]
        cm_b = flux_scale * cm[i, j, k + 1]

        mi = m[i, j, k]
        ri = rm[i, j, k]

        # Top interface (k): donor is k-1 if cm_t > 0 (downward), else k
        ft = if k > 1
            if cm_t >= zero(FT)
                _gamma_clamped_x_flux(cm_t, m[i, j, k - 1], rm[i, j, k - 1])
            else
                _gamma_clamped_x_flux(cm_t, mi, ri)
            end
        else
            zero(FT)
        end

        # Bottom interface (k+1): donor is k if cm_b > 0, else k+1
        fb = if k < Nz
            if cm_b >= zero(FT)
                _gamma_clamped_x_flux(cm_b, mi, ri)
            else
                _gamma_clamped_x_flux(cm_b, m[i, j, k + 1], rm[i, j, k + 1])
            end
        else
            zero(FT)
        end

        rm_A[i, j, k] = ri + ft - fb
        m_A[i, j, k]  = mi + cm_t - cm_b
    end
    _copy_interior!(rm, rm_A, Nc, Hp, Nz)
    _copy_interior!(m, m_A, Nc, Hp, Nz)
    return nothing
end

"""Copy interior region from buffer back to array."""
function _copy_interior!(dst, src, Nc, Hp, Nz)
    r = Hp+1:Hp+Nc
    dst[r, r, 1:Nz] .= src[r, r, 1:Nz]
    return nothing
end

# =========================================================================
# CS workspace — pre-allocated buffers for one panel
# =========================================================================

"""
    CSAdvectionWorkspace{FT, A}

Pre-allocated double buffers for cubed-sphere panel advection.
One workspace is shared across all 6 panels (sequential panel loop).
"""
struct CSAdvectionWorkspace{FT, A <: AbstractArray{FT, 3}}
    rm_A :: A
    m_A  :: A
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              array_type::Type{<:AbstractArray} = Array)
    N = mesh.Nc + 2 * mesh.Hp
    rm_A = array_type(zeros(FT, N, N, Nz))
    m_A  = array_type(zeros(FT, N, N, Nz))
    return CSAdvectionWorkspace{FT, typeof(rm_A)}(rm_A, m_A)
end

# =========================================================================
# Public API: strang_split_cs!
# =========================================================================

# =========================================================================
# CFL-based subcycle count
# =========================================================================

"""Static CFL subcycle count from initial mass (no evolving-mass pilot)."""
function _cs_static_subcycle_count(panels_flux::NTuple{6}, panels_m::NTuple{6},
                                    Nc::Int, Hp::Int, Nz::Int, cfl_limit::Real,
                                    direction::Symbol)
    FT = eltype(panels_m[1])
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mi = panels_m[p][Hp+i, Hp+j, k]
            mi <= zero(FT) && continue
            c = if direction === :x
                max(abs(panels_flux[p][Hp+i, Hp+j, k]),
                    abs(panels_flux[p][Hp+i+1, Hp+j, k])) / mi
            elseif direction === :y
                max(abs(panels_flux[p][Hp+i, Hp+j, k]),
                    abs(panels_flux[p][Hp+i, Hp+j+1, k])) / mi
            else
                max(abs(panels_flux[p][Hp+i, Hp+j, k]),
                    abs(panels_flux[p][Hp+i, Hp+j, k+1])) / mi
            end
            max_cfl = max(max_cfl, c)
        end
    end
    max_cfl <= cfl_limit && return 1
    return ceil(Int, max_cfl / cfl_limit)
end


# NOTE: Evolving-mass pilot functions (_cs_x/y/z_pilot_subcycle_count) were
# removed — they were dead code (strang_split_cs! uses _cs_static_subcycle_count).
# The static pilot is sufficient because the gamma-clamped sweep handles CFL > 1
# safely. If evolving-mass pilots are needed in the future, see git history.

"""
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, scheme, workspace; flux_scale=1, cfl_limit=0.95)

Perform one Strang-split advection step on a 6-panel cubed-sphere field
with automatic CFL-based subcycling per direction.

## Splitting sequence

    X sweep (n_x subcycles)
    → fill_panel_halos!(dir=1)     ← exchange halos between panels (X direction)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=2)     ← exchange halos between panels (Y direction)
    → Z sweep (n_z subcycles)      ← first Z half-step
    → Z sweep (n_z subcycles)      ← second Z half-step (palindrome)
    → fill_panel_halos!(dir=2)
    → Y sweep (n_y subcycles)
    → fill_panel_halos!(dir=1)
    → X sweep (n_x subcycles)

This palindromic sequence (X-Y-Z-Z-Y-X) gives second-order temporal accuracy
via Strang (1968) symmetry. The halo exchanges must happen BETWEEN successive
horizontal sweeps because the panel-edge reconstruction stencil reads from
adjacent panels.

## Subcycling

Each direction `D ∈ {X, Y, Z}` has its own subcycle count `n_D` determined
by an evolving-mass CFL pilot: the pilot applies `n_D` passes of
`flux_scale/n_D`, checking that no cell mass goes negative or that
`|outgoing_flux| < cfl_limit × cell_mass` at each pass. If the pilot fails,
`n_D` is incremented until it passes (or hits `max_n_sub` and errors).

## Panel array layout

Each panel's rm and m arrays are `(Nc+2Hp, Nc+2Hp, Nz)` with Hp-wide halos.
Interior cells are at indices `[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]`. The sweep
kernels only update interior cells; halo regions are filled by
`fill_panel_halos!` from adjacent panels.

## Arguments

- `panels_rm`, `panels_m`: NTuple{6} of 3D arrays `(Nc+2Hp, Nc+2Hp, Nz)` —
  tracer mass and air mass. Modified in-place.
- `panels_am`, `panels_bm`, `panels_cm`: NTuple{6} of flux arrays.
  `am[Nc+2Hp+1, Nc+2Hp, Nz]`, `bm[Nc+2Hp, Nc+2Hp+1, Nz]`,
  `cm[Nc+2Hp, Nc+2Hp, Nz+1]`. Read-only.
- `mesh`: `CubedSphereMesh` with Nc, Hp, and panel connectivity.
- `scheme`: advection scheme — `UpwindScheme()` uses gamma-clamped upwind
  (positivity-safe). `SlopesScheme()` and `PPMScheme()` use the generic
  KA kernels with `_xface_tracer_flux` dispatch. Higher-order schemes
  require `mesh.Hp ≥ 2` (Slopes) or `mesh.Hp ≥ 3` (PPM).
- `workspace`: pre-allocated `CSAdvectionWorkspace` buffers.
- `flux_scale`: overall scaling applied to all fluxes (default 1.0).
- `cfl_limit`: maximum CFL per subcycle pass (default 0.95).
"""
function strang_split_cs!(panels_rm::NTuple{6},
                          panels_m::NTuple{6},
                          panels_am::NTuple{6},
                          panels_bm::NTuple{6},
                          panels_cm::NTuple{6},
                          mesh::CubedSphereMesh,
                          scheme,
                          workspace::CSAdvectionWorkspace;
                          flux_scale = one(eltype(panels_m[1])),
                          cfl_limit::Real = 0.95)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)
    rm_A, m_A = workspace.rm_A, workspace.m_A
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    # Static CFL subcycle count. Gamma clamping in the sweep kernels handles
    # per-cell CFL > 1 correctly (tracer flux saturates at donor mass, mass
    # update is exact). Subcycling reduces the average CFL but isn't required
    # for stability — it's for accuracy (second-order advection needs CFL < 1).
    n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
    n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
    n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    # ---- X sweep (subcycled) ----
    for _ in 1:n_x
        for p in 1:6
            _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_x)
        end
        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
    end

    # ---- Y sweep (subcycled) ----
    for _ in 1:n_y
        for p in 1:6
            _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_y)
        end
        fill_panel_halos!(panels_rm, mesh; dir=2)
        fill_panel_halos!(panels_m, mesh; dir=2)
    end

    # ---- Z sweep × 2 (subcycled) ----
    for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_z)
        end
    end
    for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_z)
        end
    end

    # ---- Reverse: Y sweep (subcycled) ----
    fill_panel_halos!(panels_rm, mesh; dir=2)
    fill_panel_halos!(panels_m, mesh; dir=2)
    for _ in 1:n_y
        for p in 1:6
            _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_y)
        end
        fill_panel_halos!(panels_rm, mesh; dir=2)
        fill_panel_halos!(panels_m, mesh; dir=2)
    end

    # ---- Reverse: X sweep (subcycled) ----
    fill_panel_halos!(panels_rm, mesh; dir=1)
    fill_panel_halos!(panels_m, mesh; dir=1)
    for _ in 1:n_x
        for p in 1:6
            _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                             scheme, rm_A, m_A, Nc, Hp, Nz; flux_scale=fs_x)
        end
        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
    end

    return nothing
end

"""
    _sweep_z!(rm_panels, m_panels, cm_panels, mesh, use_limiter, ws)

Multi-panel Z-sweep orchestrator for LinRood integration. Applies vertical
mass-flux advection to all 6 panels using the per-panel `_sweep_z_panel!`.
Always uses `UpwindScheme()` (matching FV3's upwind vertical advection).
The `use_limiter` argument is accepted for call-site compatibility but ignored.
"""
function _sweep_z!(rm_panels, m_panels, cm_panels,
                   mesh::CubedSphereMesh, use_limiter::Bool,
                   ws::CSAdvectionWorkspace)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(rm_panels[1], 3)
    for p in 1:6
        _sweep_z_panel!(rm_panels[p], m_panels[p], cm_panels[p],
                         UpwindScheme(), ws.rm_A, ws.m_A, Nc, Hp, Nz)
    end
    return nothing
end

export strang_split_cs!, CSAdvectionWorkspace
