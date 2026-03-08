# ---------------------------------------------------------------------------
# Cubed-Sphere Mass-Flux Advection
#
# Panel-local Russell-Lerner slopes advection for CubedSphereGrid, following
# the same TM5-faithful scheme as the lat-lon version but with:
#   - 2D per-panel cell area (not 1D by latitude)
#   - Halo-based boundary conditions (no periodic wrap or pole logic)
#   - fill_panel_halos! between directional sweeps
#
# Data layout: each panel has interior (Nc × Nc × Nz) and halos of width Hp.
# Tracer mass and air mass arrays are (Nc+2Hp) × (Nc+2Hp) × Nz with halos.
# Mass flux arrays (am, bm) are interior-only: (Nc+1 × Nc × Nz) and (Nc × Nc+1 × Nz).
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend
using Printf: @sprintf

# ---- helpers ---------------------------------------------------------------

"""Move a CPU vector/matrix to the same device as `ref`."""
function _to_device_2d(cpu_data, ref::AbstractArray)
    ArrayType = typeof(similar(ref, eltype(cpu_data), size(cpu_data)))
    return ArrayType(cpu_data)
end

# ---- Geometry Cache --------------------------------------------------------

"""
$(TYPEDEF)

Device-side geometry cache for cubed-sphere mass-flux advection.

Unlike the lat-lon `GridGeometryCache` (which stores 1-D area/dy vectors),
this stores per-panel 2-D cell area and metric arrays.

$(FIELDS)
"""
struct CubedSphereGeometryCache{FT, A2 <: AbstractMatrix{FT}, A1 <: AbstractVector{FT}}
    "per-panel cell areas [m²], NTuple of Nc×Nc matrices"
    area  :: NTuple{6, A2}
    "per-panel Δx at cell centers [m], NTuple of Nc×Nc matrices"
    dx    :: NTuple{6, A2}
    "per-panel Δy at cell centers [m], NTuple of Nc×Nc matrices"
    dy    :: NTuple{6, A2}
    "vertical B-ratio for mass-flux closure, length Nz"
    bt    :: A1
    gravity :: FT
    Nc :: Int
    Nz :: Int
    Hp :: Int
end

"""
$(SIGNATURES)

Build a [`CubedSphereGeometryCache`](@ref) from a `CubedSphereGrid`.
`ref_array` determines device placement (CPU or GPU).
"""
function build_geometry_cache(grid::CubedSphereGrid{FT},
                              ref_array::AbstractArray{FT}) where FT
    Nc = grid.Nc
    Nz = grid.Nz
    Hp = grid.Hp
    g  = FT(grid.gravity)

    area_panels = ntuple(6) do p
        a_cpu = Matrix{FT}(undef, Nc, Nc)
        @inbounds for j in 1:Nc, i in 1:Nc
            a_cpu[i, j] = FT(grid.Aᶜ[p][i, j])
        end
        _to_device_2d(a_cpu, ref_array)
    end

    dx_panels = ntuple(6) do p
        d_cpu = Matrix{FT}(undef, Nc, Nc)
        @inbounds for j in 1:Nc, i in 1:Nc
            d_cpu[i, j] = FT(grid.Δxᶜ[p][i, j])
        end
        _to_device_2d(d_cpu, ref_array)
    end

    dy_panels = ntuple(6) do p
        d_cpu = Matrix{FT}(undef, Nc, Nc)
        @inbounds for j in 1:Nc, i in 1:Nc
            d_cpu[i, j] = FT(grid.Δyᶜ[p][i, j])
        end
        _to_device_2d(d_cpu, ref_array)
    end

    vc = grid.vertical
    ΔB_cpu = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        ΔB_cpu[k] = FT(vc.B[k + 1] - vc.B[k])
    end
    ΔB_total = FT(vc.B[Nz + 1] - vc.B[1])
    bt_cpu = abs(ΔB_total) > eps(FT) ? ΔB_cpu ./ ΔB_total : zeros(FT, Nz)

    bt = typeof(ref_array) <: Array ? bt_cpu : _to_device_2d(bt_cpu, ref_array)

    return CubedSphereGeometryCache{FT, typeof(area_panels[1]), typeof(bt)}(
        area_panels, dx_panels, dy_panels, bt, g, Nc, Nz, Hp)
end

# ---- Air Mass Kernel -------------------------------------------------------

@kernel function _air_mass_cs_kernel!(m_out, @Const(delp), @Const(area), g, Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds m_out[Hp + i, Hp + j, k] = delp[Hp + i, Hp + j, k] * area[i, j] / g
end

"""
$(SIGNATURES)

Compute air mass for a single cubed-sphere panel from pressure thickness.
`m` and `delp` are haloed arrays (Nc+2Hp × Nc+2Hp × Nz).
`area` is an interior-only array (Nc × Nc).
"""
function compute_air_mass_panel!(m::AbstractArray{FT,3},
                                 delp::AbstractArray{FT,3},
                                 area::AbstractMatrix{FT},
                                 g::FT, Nc::Int, Nz::Int, Hp::Int) where FT
    backend = get_backend(m)
    k! = _air_mass_cs_kernel!(backend, 256)
    k!(m, delp, area, g, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---- Mass Fixer: preserve mixing ratio across air mass reset ----------------

@kernel function _mass_fixer_kernel!(rm, @Const(m_ref), @Const(m_evolved), Hp)
    i, j, k = @index(Global, NTuple)
    ii = Hp + i
    jj = Hp + j
    @inbounds begin
        m_e = m_evolved[ii, jj, k]
        rm[ii, jj, k] = if m_e > zero(eltype(rm))
            (rm[ii, jj, k] / m_e) * m_ref[ii, jj, k]
        else
            zero(eltype(rm))
        end
    end
end

"""
$(SIGNATURES)

Correct tracer mass `rm` to preserve mixing ratio `q = rm/m` when air mass is
reset from `m_evolved` (post-advection) back to `m_ref` (DELP-derived).

Sets `rm[i,j,k] = (rm[i,j,k] / m_evolved[i,j,k]) * m_ref[i,j,k]` for interior cells.
"""
function apply_mass_fixer!(rm::AbstractArray{FT,3},
                            m_ref::AbstractArray{FT,3},
                            m_evolved::AbstractArray{FT,3},
                            Nc::Int, Nz::Int, Hp::Int) where FT
    backend = get_backend(rm)
    k! = _mass_fixer_kernel!(backend, 256)
    k!(rm, m_ref, m_evolved, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---- Vertical Mass-Flux Closure --------------------------------------------

@kernel function _cm_column_cs_kernel!(cm, @Const(am), @Const(bm), @Const(bt), Nc, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)
    @inbounds begin
        pit = zero(FT)
        for k in 1:Nz
            pit += am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
        end
        acc = zero(FT)
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
            acc += conv_k - bt[k] * pit
            cm[i, j, k + 1] = acc
        end
    end
end

"""
$(SIGNATURES)

Compute vertical mass flux `cm` from horizontal convergence of `am` (X mass flux)
and `bm` (Y mass flux) for a single panel, ensuring column mass conservation.

- `cm`: (Nc, Nc, Nz+1) — vertical flux at level interfaces
- `am`: (Nc+1, Nc, Nz) — X-face mass flux
- `bm`: (Nc, Nc+1, Nz) — Y-face mass flux
- `bt`: (Nz,) — B-ratio for sigma correction
"""
function compute_cm_panel!(cm::AbstractArray{FT,3},
                           am::AbstractArray{FT,3},
                           bm::AbstractArray{FT,3},
                           bt::AbstractVector{FT},
                           Nc::Int, Nz::Int) where FT
    backend = get_backend(cm)
    fill!(cm, zero(FT))
    k! = _cm_column_cs_kernel!(backend, 256)
    k!(cm, am, bm, bt, Nc, Nz; ndrange=(Nc, Nc))
    synchronize(backend)
    return nothing
end

# ---- Pressure-Fixer Vertical Mass-Flux Closure -----------------------------

@kernel function _cm_pressure_fixer_kernel!(cm, @Const(am), @Const(bm), @Const(bt),
        @Const(delp_curr), @Const(delp_next), @Const(area), g, n_sub_x2, Hp, Nc, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)
    @inbounds begin
        inv_g_n = one(FT) / (g * FT(n_sub_x2))
        a = area[i, j]
        pit = zero(FT)
        dp_sum = zero(FT)
        for k in 1:Nz
            pit    += am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
            dp_sum += (delp_next[Hp + i, Hp + j, k] - delp_curr[Hp + i, Hp + j, k]) * a * inv_g_n
        end
        residual = pit - dp_sum
        acc = zero(FT)
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
            dp_k   = (delp_next[Hp + i, Hp + j, k] - delp_curr[Hp + i, Hp + j, k]) * a * inv_g_n
            acc += conv_k - bt[k] * residual - dp_k
            cm[i, j, k + 1] = acc
        end
    end
end

"""
$(SIGNATURES)

Compute vertical mass flux `cm` using pressure-fixer formulation: incorporates the
pressure tendency (DELP_next - DELP_current) so that air mass evolves toward the
next window's target across sub-steps.  Falls back to standard bt-only closure
when `delp_next === nothing`.
"""
function compute_cm_pressure_fixer_panel!(cm::AbstractArray{FT,3},
                                          am::AbstractArray{FT,3},
                                          bm::AbstractArray{FT,3},
                                          bt::AbstractVector{FT},
                                          delp_curr::AbstractArray{FT,3},
                                          delp_next::AbstractArray{FT,3},
                                          area::AbstractMatrix{FT},
                                          g::FT, n_sub::Int,
                                          Nc::Int, Nz::Int, Hp::Int) where FT
    backend = get_backend(cm)
    fill!(cm, zero(FT))
    n_sub_x2 = 2 * n_sub
    k! = _cm_pressure_fixer_kernel!(backend, 256)
    k!(cm, am, bm, bt, delp_curr, delp_next, area, g, n_sub_x2, Hp, Nc, Nz;
       ndrange=(Nc, Nc))
    synchronize(backend)
    return nothing
end

# ---- Per-Sub-Step Air Mass Increment ----------------------------------------

@kernel function _dm_per_sub_kernel!(dm, @Const(delp_curr), @Const(delp_next),
                                      @Const(area), inv_g_n, Hp)
    i, j, k = @index(Global, NTuple)
    FT = eltype(dm)
    @inbounds dm[Hp + i, Hp + j, k] =
        (delp_next[Hp + i, Hp + j, k] - delp_curr[Hp + i, Hp + j, k]) *
        area[i, j] * inv_g_n
end

"""
$(SIGNATURES)

Compute the air-mass increment per sub-step for the pressure-fixer m_ref evolution:
`dm[i,j,k] = (DELP_next - DELP_curr) * area / (g * n_sub)`.
"""
function compute_dm_per_sub_panel!(dm::AbstractArray{FT,3},
                                    delp_curr::AbstractArray{FT,3},
                                    delp_next::AbstractArray{FT,3},
                                    area::AbstractMatrix{FT},
                                    g::FT, n_sub::Int,
                                    Nc::Int, Nz::Int, Hp::Int) where FT
    backend = get_backend(dm)
    inv_g_n = FT(1) / (g * FT(n_sub))
    k! = _dm_per_sub_kernel!(backend, 256)
    k!(dm, delp_curr, delp_next, area, inv_g_n, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return nothing
end

# ---- Panel-Local X-Advection Kernel ----------------------------------------

@kernel function _massflux_x_cs_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am), Hp, Nc, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii  = Hp + i
        jj  = Hp + j
        FT  = eltype(rm)

        c_imm = rm[ii - 2, jj, k] / m[ii - 2, jj, k]
        c_im  = rm[ii - 1, jj, k] / m[ii - 1, jj, k]
        c_i   = rm[ii,     jj, k] / m[ii,     jj, k]
        c_ip  = rm[ii + 1, jj, k] / m[ii + 1, jj, k]
        c_ipp = rm[ii + 2, jj, k] / m[ii + 2, jj, k]

        # Slope at i-1
        sc_im = (c_i - c_imm) / 2
        if use_limiter
            sc_im = minmod_device(sc_im, 2 * (c_i - c_im), 2 * (c_im - c_imm))
        end
        sx_im = m[ii - 1, jj, k] * sc_im
        if use_limiter
            sx_im = max(min(sx_im, rm[ii - 1, jj, k]), -rm[ii - 1, jj, k])
        end

        # Slope at i
        sc_i = (c_ip - c_im) / 2
        if use_limiter
            sc_i = minmod_device(sc_i, 2 * (c_ip - c_i), 2 * (c_i - c_im))
        end
        sx_i = m[ii, jj, k] * sc_i
        if use_limiter
            sx_i = max(min(sx_i, rm[ii, jj, k]), -rm[ii, jj, k])
        end

        # Slope at i+1
        sc_ip = (c_ipp - c_i) / 2
        if use_limiter
            sc_ip = minmod_device(sc_ip, 2 * (c_ipp - c_ip), 2 * (c_ip - c_i))
        end
        sx_ip = m[ii + 1, jj, k] * sc_ip
        if use_limiter
            sx_ip = max(min(sx_ip, rm[ii + 1, jj, k]), -rm[ii + 1, jj, k])
        end

        # Flux at left face (face i in am, which is interior-indexed)
        am_l = am[i, j, k]
        flux_left = if am_l >= zero(FT)
            alpha = am_l / m[ii - 1, jj, k]
            alpha * (rm[ii - 1, jj, k] + (one(FT) - alpha) * sx_im)
        else
            alpha = am_l / m[ii, jj, k]
            alpha * (rm[ii, jj, k] - (one(FT) + alpha) * sx_i)
        end

        # Flux at right face (face i+1 in am)
        am_r = am[i + 1, j, k]
        flux_right = if am_r >= zero(FT)
            alpha = am_r / m[ii, jj, k]
            alpha * (rm[ii, jj, k] + (one(FT) - alpha) * sx_i)
        else
            alpha = am_r / m[ii + 1, jj, k]
            alpha * (rm[ii + 1, jj, k] - (one(FT) + alpha) * sx_ip)
        end

        rm_new[ii, jj, k] = rm[ii, jj, k] + flux_left - flux_right
        m_new[ii, jj, k]  = m[ii, jj, k]  + am[i, j, k] - am[i + 1, j, k]
    end
end

# ---- Panel-Local Y-Advection Kernel ----------------------------------------

@kernel function _massflux_y_cs_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm), Hp, Nc, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        c_jmm = rm[ii, jj - 2, k] / m[ii, jj - 2, k]
        c_jm  = rm[ii, jj - 1, k] / m[ii, jj - 1, k]
        c_j   = rm[ii, jj,     k] / m[ii, jj,     k]
        c_jp  = rm[ii, jj + 1, k] / m[ii, jj + 1, k]
        c_jpp = rm[ii, jj + 2, k] / m[ii, jj + 2, k]

        # Slope at j-1
        sc_jm = (c_j - c_jmm) / 2
        if use_limiter
            sc_jm = minmod_device(sc_jm, 2 * (c_j - c_jm), 2 * (c_jm - c_jmm))
        end
        sy_jm = m[ii, jj - 1, k] * sc_jm
        if use_limiter
            sy_jm = max(min(sy_jm, rm[ii, jj - 1, k]), -rm[ii, jj - 1, k])
        end

        # Slope at j
        sc_j = (c_jp - c_jm) / 2
        if use_limiter
            sc_j = minmod_device(sc_j, 2 * (c_jp - c_j), 2 * (c_j - c_jm))
        end
        sy_j = m[ii, jj, k] * sc_j
        if use_limiter
            sy_j = max(min(sy_j, rm[ii, jj, k]), -rm[ii, jj, k])
        end

        # Slope at j+1
        sc_jp = (c_jpp - c_j) / 2
        if use_limiter
            sc_jp = minmod_device(sc_jp, 2 * (c_jpp - c_jp), 2 * (c_jp - c_j))
        end
        sy_jp = m[ii, jj + 1, k] * sc_jp
        if use_limiter
            sy_jp = max(min(sy_jp, rm[ii, jj + 1, k]), -rm[ii, jj + 1, k])
        end

        # Flux at south face (face j in bm)
        bm_s = bm[i, j, k]
        flux_south = if bm_s >= zero(FT)
            beta = bm_s / m[ii, jj - 1, k]
            beta * (rm[ii, jj - 1, k] + (one(FT) - beta) * sy_jm)
        else
            beta = bm_s / m[ii, jj, k]
            beta * (rm[ii, jj, k] - (one(FT) + beta) * sy_j)
        end

        # Flux at north face (face j+1 in bm)
        bm_n = bm[i, j + 1, k]
        flux_north = if bm_n >= zero(FT)
            beta = bm_n / m[ii, jj, k]
            beta * (rm[ii, jj, k] + (one(FT) - beta) * sy_j)
        else
            beta = bm_n / m[ii, jj + 1, k]
            beta * (rm[ii, jj + 1, k] - (one(FT) + beta) * sy_jp)
        end

        rm_new[ii, jj, k] = rm[ii, jj, k] + flux_south - flux_north
        m_new[ii, jj, k]  = m[ii, jj, k]  + bm[i, j, k] - bm[i, j + 1, k]
    end
end

# ---- Panel-Local Z-Advection (reuses lat-lon kernel) -----------------------

@kernel function _massflux_z_cs_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(cm), Hp, Nz, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        sz_k = if k > 1 && k < Nz
            ckm = rm[ii, jj, k - 1] / m[ii, jj, k - 1]
            ck  = rm[ii, jj, k]     / m[ii, jj, k]
            ckp = rm[ii, jj, k + 1] / m[ii, jj, k + 1]
            sc = (ckp - ckm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ckp - ck), 2 * (ck - ckm)); end
            s = m[ii, jj, k] * sc
            if use_limiter; s = max(min(s, rm[ii, jj, k]), -rm[ii, jj, k]); end
            s
        else
            zero(FT)
        end

        sz_km = if k > 2 && k - 1 < Nz
            ckmm = rm[ii, jj, k - 2] / m[ii, jj, k - 2]
            ckm  = rm[ii, jj, k - 1] / m[ii, jj, k - 1]
            ck   = rm[ii, jj, k]     / m[ii, jj, k]
            sc = (ck - ckmm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ck - ckm), 2 * (ckm - ckmm)); end
            s = m[ii, jj, k - 1] * sc
            if use_limiter; s = max(min(s, rm[ii, jj, k - 1]), -rm[ii, jj, k - 1]); end
            s
        else
            zero(FT)
        end

        sz_kp = if k < Nz - 1 && k + 1 > 1
            ck   = rm[ii, jj, k]     / m[ii, jj, k]
            ckp  = rm[ii, jj, k + 1] / m[ii, jj, k + 1]
            ckpp = rm[ii, jj, k + 2] / m[ii, jj, k + 2]
            sc = (ckpp - ck) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ckpp - ckp), 2 * (ckp - ck)); end
            s = m[ii, jj, k + 1] * sc
            if use_limiter; s = max(min(s, rm[ii, jj, k + 1]), -rm[ii, jj, k + 1]); end
            s
        else
            zero(FT)
        end

        flux_top = if k > 1
            cm_t = cm[i, j, k]
            if cm_t > zero(FT)
                gamma = cm_t / m[ii, jj, k - 1]
                gamma * (rm[ii, jj, k - 1] + (one(FT) - gamma) * sz_km)
            elseif cm_t < zero(FT)
                gamma = cm_t / m[ii, jj, k]
                gamma * (rm[ii, jj, k] - (one(FT) + gamma) * sz_k)
            else
                zero(FT)
            end
        else
            zero(FT)
        end

        flux_bot = if k < Nz
            cm_b = cm[i, j, k + 1]
            if cm_b > zero(FT)
                gamma = cm_b / m[ii, jj, k]
                gamma * (rm[ii, jj, k] + (one(FT) - gamma) * sz_k)
            elseif cm_b < zero(FT)
                gamma = cm_b / m[ii, jj, k + 1]
                gamma * (rm[ii, jj, k + 1] - (one(FT) + gamma) * sz_kp)
            else
                zero(FT)
            end
        else
            zero(FT)
        end

        rm_new[ii, jj, k] = rm[ii, jj, k] + flux_top - flux_bot
        m_new[ii, jj, k]  = m[ii, jj, k]  + cm[i, j, k] - cm[i, j, k + 1]
    end
end

# ---- Column-Sequential Z-Advection Kernel ---------------------------------
#
# Processes levels sequentially within each (i,j) column, preventing
# negative mass by clamping gamma (= cm/m) to [-1, 1].  This is necessary
# when the per-face Z-CFL exceeds 0.5, because two-sided outflow from a
# single cell can drain more mass than the cell contains.
#
# The mass update m_new = m + cm[k] - cm[k+1] is exact regardless of gamma
# clamping.  Only the TRACER flux uses gamma, so clamping reduces the tracer
# transport to first-order upwind at extreme-CFL cells while keeping the
# mass budget perfectly balanced.

@kernel function _massflux_z_cs_column_kernel!(rm, m,
        @Const(rm_src), @Const(m_src), @Const(cm), Hp, Nz, use_limiter)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        # Process levels top to bottom.
        # All slope and flux computations read from rm_src/m_src (the ORIGINAL
        # values copied before this kernel launch).  Updates are written to
        # rm/m.  This guarantees that the tracer flux at each interface is
        # computed identically by both adjacent levels → exact flux telescoping
        # → exact mass conservation.
        for k in 1:Nz
            # ── Slopes for the second-order correction ──────────────────
            # Slope at k (needs k-1 and k+1)
            sz_k = if k > 1 && k < Nz
                _m = m_src[ii, jj, k - 1]; ckm = _m > zero(FT) ? rm_src[ii, jj, k - 1] / _m : zero(FT)
                _m = m_src[ii, jj, k];     ck  = _m > zero(FT) ? rm_src[ii, jj, k]     / _m : zero(FT)
                _m = m_src[ii, jj, k + 1]; ckp = _m > zero(FT) ? rm_src[ii, jj, k + 1] / _m : zero(FT)
                sc = (ckp - ckm) / FT(2)
                if use_limiter
                    sc = minmod_device(sc, FT(2) * (ckp - ck), FT(2) * (ck - ckm))
                end
                s = m_src[ii, jj, k] * sc
                if use_limiter
                    s = max(min(s, rm_src[ii, jj, k]), -rm_src[ii, jj, k])
                end
                s
            else
                zero(FT)
            end

            # Slope at k-1 (needs k-2 and k)
            sz_km = if k > 2 && k - 1 < Nz
                _m = m_src[ii, jj, k - 2]; ckmm = _m > zero(FT) ? rm_src[ii, jj, k - 2] / _m : zero(FT)
                _m = m_src[ii, jj, k - 1]; ckm  = _m > zero(FT) ? rm_src[ii, jj, k - 1] / _m : zero(FT)
                _m = m_src[ii, jj, k];     ck   = _m > zero(FT) ? rm_src[ii, jj, k]     / _m : zero(FT)
                sc = (ck - ckmm) / FT(2)
                if use_limiter
                    sc = minmod_device(sc, FT(2) * (ck - ckm), FT(2) * (ckm - ckmm))
                end
                s = m_src[ii, jj, k - 1] * sc
                if use_limiter
                    s = max(min(s, rm_src[ii, jj, k - 1]), -rm_src[ii, jj, k - 1])
                end
                s
            else
                zero(FT)
            end

            # Slope at k+1 (needs k and k+2)
            sz_kp = if k < Nz - 1 && k + 1 > 1
                _m = m_src[ii, jj, k];     ck   = _m > zero(FT) ? rm_src[ii, jj, k]     / _m : zero(FT)
                _m = m_src[ii, jj, k + 1]; ckp  = _m > zero(FT) ? rm_src[ii, jj, k + 1] / _m : zero(FT)
                _m = m_src[ii, jj, k + 2]; ckpp = _m > zero(FT) ? rm_src[ii, jj, k + 2] / _m : zero(FT)
                sc = (ckpp - ck) / FT(2)
                if use_limiter
                    sc = minmod_device(sc, FT(2) * (ckpp - ckp), FT(2) * (ckp - ck))
                end
                s = m_src[ii, jj, k + 1] * sc
                if use_limiter
                    s = max(min(s, rm_src[ii, jj, k + 1]), -rm_src[ii, jj, k + 1])
                end
                s
            else
                zero(FT)
            end

            # ── Tracer flux at top interface (k) ────────────────────────
            flux_top = if k > 1
                cm_t = cm[i, j, k]
                if cm_t > zero(FT)
                    md = m_src[ii, jj, k - 1]
                    gamma = md > zero(FT) ? clamp(cm_t / md, zero(FT), one(FT)) : zero(FT)
                    gamma * (rm_src[ii, jj, k - 1] + (one(FT) - gamma) * sz_km)
                elseif cm_t < zero(FT)
                    md = m_src[ii, jj, k]
                    gamma = md > zero(FT) ? clamp(cm_t / md, -one(FT), zero(FT)) : zero(FT)
                    gamma * (rm_src[ii, jj, k] - (one(FT) + gamma) * sz_k)
                else
                    zero(FT)
                end
            else
                zero(FT)
            end

            # ── Tracer flux at bottom interface (k+1) ───────────────────
            flux_bot = if k < Nz
                cm_b = cm[i, j, k + 1]
                if cm_b > zero(FT)
                    md = m_src[ii, jj, k]
                    gamma = md > zero(FT) ? clamp(cm_b / md, zero(FT), one(FT)) : zero(FT)
                    gamma * (rm_src[ii, jj, k] + (one(FT) - gamma) * sz_k)
                elseif cm_b < zero(FT)
                    md = m_src[ii, jj, k + 1]
                    gamma = md > zero(FT) ? clamp(cm_b / md, -one(FT), zero(FT)) : zero(FT)
                    gamma * (rm_src[ii, jj, k + 1] - (one(FT) + gamma) * sz_kp)
                else
                    zero(FT)
                end
            else
                zero(FT)
            end

            # ── Update rm and m from ORIGINAL values (exact conservation) ─
            rm[ii, jj, k] = rm_src[ii, jj, k] + flux_top - flux_bot
            m[ii, jj, k]  = m_src[ii, jj, k]  + cm[i, j, k] - cm[i, j, k + 1]
        end
    end
end

"""
$(SIGNATURES)

Column-sequential Z-direction mass-flux advection for a single cubed-sphere
panel.  Reads slopes and fluxes from `rm_src`/`m_src` (original values) and
writes updates to `rm`/`m`, ensuring exact flux telescoping and mass
conservation.  Gamma is clamped to [-1,1] to prevent negative-mass
instabilities at high vertical CFL.
"""
function advect_z_cs_panel_column!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                   rm_src::AbstractArray{FT,3}, m_src::AbstractArray{FT,3},
                                   cm::AbstractArray{FT,3},
                                   Hp::Int, Nc::Int, Nz::Int,
                                   use_limiter::Bool) where FT
    backend = get_backend(rm)
    k! = _massflux_z_cs_column_kernel!(backend, 256)
    k!(rm, m, rm_src, m_src, cm, Hp, Nz, use_limiter; ndrange=(Nc, Nc))
    synchronize(backend)
    return nothing
end

# ---- Panel-Level Advection Functions ---------------------------------------

"""
$(SIGNATURES)

X-direction mass-flux advection for a single cubed-sphere panel.
`rm`, `m`, `rm_buf`, `m_buf` are haloed (Nc+2Hp × Nc+2Hp × Nz).
`am` is interior-only (Nc+1 × Nc × Nz).
"""
function advect_x_cs_panel!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                            am::AbstractArray{FT,3},
                            rm_buf::AbstractArray{FT,3}, m_buf::AbstractArray{FT,3},
                            Hp::Int, Nc::Int, use_limiter::Bool) where FT
    backend = get_backend(rm)
    k! = _massflux_x_cs_kernel!(backend, 256)
    k!(rm_buf, rm, m_buf, m, am, Hp, Nc, use_limiter; ndrange=(Nc, Nc, size(rm, 3)))
    synchronize(backend)
    _copy_interior!(rm, rm_buf, Hp, Nc, size(rm, 3))
    _copy_interior!(m, m_buf, Hp, Nc, size(rm, 3))
    return nothing
end

"""
$(SIGNATURES)

Y-direction mass-flux advection for a single cubed-sphere panel.
"""
function advect_y_cs_panel!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                            bm::AbstractArray{FT,3},
                            rm_buf::AbstractArray{FT,3}, m_buf::AbstractArray{FT,3},
                            Hp::Int, Nc::Int, use_limiter::Bool) where FT
    backend = get_backend(rm)
    k! = _massflux_y_cs_kernel!(backend, 256)
    k!(rm_buf, rm, m_buf, m, bm, Hp, Nc, use_limiter; ndrange=(Nc, Nc, size(rm, 3)))
    synchronize(backend)
    _copy_interior!(rm, rm_buf, Hp, Nc, size(rm, 3))
    _copy_interior!(m, m_buf, Hp, Nc, size(rm, 3))
    return nothing
end

"""
$(SIGNATURES)

Z-direction mass-flux advection for a single cubed-sphere panel.
`cm` is interior-only (Nc × Nc × Nz+1).
"""
function advect_z_cs_panel!(rm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                            cm::AbstractArray{FT,3},
                            rm_buf::AbstractArray{FT,3}, m_buf::AbstractArray{FT,3},
                            Hp::Int, Nc::Int, Nz::Int, use_limiter::Bool) where FT
    backend = get_backend(rm)
    k! = _massflux_z_cs_kernel!(backend, 256)
    k!(rm_buf, rm, m_buf, m, cm, Hp, Nz, use_limiter; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, rm_buf, Hp, Nc, Nz)
    _copy_interior!(m, m_buf, Hp, Nc, Nz)
    return nothing
end

@kernel function _copy_interior_kernel!(dst, @Const(src), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[Hp + i, Hp + j, k] = src[Hp + i, Hp + j, k]
end

function _copy_interior!(dst, src, Hp, Nc, Nz)
    backend = get_backend(dst)
    k! = _copy_interior_kernel!(backend, 256)
    k!(dst, src, Hp; ndrange=(Nc, Nc, Nz))
    synchronize(backend)
end

# ---- CFL Kernels for Cubed-Sphere -----------------------------------------
#
# Per-face Courant numbers for CFL-adaptive subcycling.
# Unlike the lat-lon kernels, these account for the halo offset Hp between
# the interior-only flux arrays (am/bm) and the haloed mass array (m).

@kernel function _cfl_x_cs_kernel!(cfl, @Const(am), @Const(m), Hp)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        md = am[i, j, k] >= zero(FT) ? m[ii - 1, jj, k] : m[ii, jj, k]
        cfl[i, j, k] = md > zero(FT) ? abs(am[i, j, k]) / md : zero(FT)
    end
end

@kernel function _cfl_y_cs_kernel!(cfl, @Const(bm), @Const(m), Hp)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        md = bm[i, j, k] >= zero(FT) ? m[ii, jj - 1, k] : m[ii, jj, k]
        cfl[i, j, k] = md > zero(FT) ? abs(bm[i, j, k]) / md : zero(FT)
    end
end

@kernel function _cfl_z_cs_kernel!(cfl, @Const(cm), @Const(m), Hp, Nz)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        if k >= 2 && k <= Nz
            md = cm[i, j, k] > zero(FT) ? m[ii, jj, k - 1] : m[ii, jj, k]
            cfl[i, j, k] = md > zero(FT) ? abs(cm[i, j, k]) / md : zero(FT)
        else
            cfl[i, j, k] = zero(FT)
        end
    end
end

"""Maximum per-face CFL in x-direction for one CS panel."""
function max_cfl_x_cs(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                      cfl_arr::AbstractArray{FT,3}, Hp::Int) where FT
    backend = get_backend(m)
    fill!(cfl_arr, zero(FT))
    k! = _cfl_x_cs_kernel!(backend, 256)
    k!(cfl_arr, am, m, Hp; ndrange=size(am))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

"""Maximum per-face CFL in y-direction for one CS panel."""
function max_cfl_y_cs(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                      cfl_arr::AbstractArray{FT,3}, Hp::Int) where FT
    backend = get_backend(m)
    fill!(cfl_arr, zero(FT))
    k! = _cfl_y_cs_kernel!(backend, 256)
    k!(cfl_arr, bm, m, Hp; ndrange=size(bm))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

"""Maximum per-face CFL in z-direction for one CS panel."""
function max_cfl_z_cs(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                      cfl_arr::AbstractArray{FT,3}, Hp::Int, Nz::Int) where FT
    backend = get_backend(m)
    fill!(cfl_arr, zero(FT))
    k! = _cfl_z_cs_kernel!(backend, 256)
    k!(cfl_arr, cm, m, Hp, Nz; ndrange=size(cm))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

# ---- Workspace for Cubed-Sphere Mass-Flux Advection ------------------------

"""
$(TYPEDEF)

Pre-allocated buffers for cubed-sphere mass-flux advection.
One set of haloed buffers is reused across all panels (sequential processing).
CFL buffers are sized to the largest flux array per direction.

$(FIELDS)
"""
struct CubedSphereMassFluxWorkspace{FT, A3 <: AbstractArray{FT,3}}
    "tracer mass buffer (haloed, Nc+2Hp × Nc+2Hp × Nz)"
    rm_buf :: A3
    "air mass buffer (haloed, Nc+2Hp × Nc+2Hp × Nz)"
    m_buf  :: A3
    "CFL scratch for x-direction (Nc+1 × Nc × Nz)"
    cfl_x  :: A3
    "CFL scratch for y-direction (Nc × Nc+1 × Nz)"
    cfl_y  :: A3
    "CFL scratch for z-direction (Nc × Nc × Nz+1)"
    cfl_z  :: A3
end

"""
$(SIGNATURES)

Allocate workspace buffers for cubed-sphere mass-flux advection.
`ref_panel` is a haloed panel array whose size and device type are matched.
`Nc` is the number of interior cells per panel edge (required so CFL scratch
arrays are sized exactly to the flux dimensions — oversized arrays caused
`maximum()` to read uninitialized GPU memory, producing garbage CFL values).
"""
function allocate_cs_massflux_workspace(ref_panel::AbstractArray{FT,3}, Nc::Int) where FT
    Nz = size(ref_panel, 3)
    # CFL buffers match the exact flux array dimensions:
    #   x-flux: (Nc+1, Nc, Nz),  y-flux: (Nc, Nc+1, Nz),  z-flux: (Nc, Nc, Nz+1)
    cfl_x = similar(ref_panel, FT, Nc + 1, Nc, Nz)
    cfl_y = similar(ref_panel, FT, Nc, Nc + 1, Nz)
    cfl_z = similar(ref_panel, FT, Nc, Nc, Nz + 1)
    CubedSphereMassFluxWorkspace{FT, typeof(ref_panel)}(
        similar(ref_panel),
        similar(ref_panel),
        cfl_x, cfl_y, cfl_z,
    )
end

# ---- Strang Split for CubedSphereGrid -------------------------------------

"""
$(SIGNATURES)

Perform a full Strang-split mass-flux advection step (X-Y-Z-Z-Y-X) on a
`CubedSphereGrid` with CFL-adaptive subcycling per direction.

Each directional sweep computes the maximum per-face CFL across all 6 panels.
When CFL exceeds `cfl_limit`, the sweep is subcycled: fluxes are divided by
the subcycle count and the advection kernel is applied that many times.

Arguments:
- `rm_panels`: NTuple{6, Array} of tracer mass (haloed)
- `m_panels`: NTuple{6, Array} of air mass (haloed)
- `am_panels`: NTuple{6, Array} of X mass flux (interior, Nc+1 × Nc × Nz)
- `bm_panels`: NTuple{6, Array} of Y mass flux (interior, Nc × Nc+1 × Nz)
- `cm_panels`: NTuple{6, Array} of Z mass flux (interior, Nc × Nc × Nz+1)
- `grid`: CubedSphereGrid
- `use_limiter`: enable minmod slope limiter
- `ws`: CubedSphereMassFluxWorkspace
- `cfl_limit`: maximum allowed CFL per sweep (default 0.95)
"""
function strang_split_massflux!(rm_panels::NTuple{6},
                                m_panels::NTuple{6},
                                am_panels::NTuple{6},
                                bm_panels::NTuple{6},
                                cm_panels::NTuple{6},
                                grid::CubedSphereGrid{FT},
                                use_limiter::Bool,
                                ws::CubedSphereMassFluxWorkspace;
                                cfl_limit::FT = FT(0.95)) where FT
    # X → Y → Z → Z → Y → X (Strang splitting)
    _sweep_x!(rm_panels, m_panels, am_panels, grid, use_limiter, ws; cfl_limit)
    _sweep_y!(rm_panels, m_panels, bm_panels, grid, use_limiter, ws; cfl_limit)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, use_limiter, ws)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, use_limiter, ws)
    _sweep_y!(rm_panels, m_panels, bm_panels, grid, use_limiter, ws; cfl_limit)
    _sweep_x!(rm_panels, m_panels, am_panels, grid, use_limiter, ws; cfl_limit)
    return nothing
end

function _sweep_x!(rm_panels, m_panels, am_panels, grid, use_limiter, ws;
                   cfl_limit = eltype(ws.rm_buf)(0.95))
    FT = eltype(ws.rm_buf)
    Hp = grid.Hp
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    # Per-face CFL across all panels
    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_x_cs(am_panels[p], m_panels[p], ws.cfl_x, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))
    if n_sub > 100
        @warn "Extreme CFL subcycling in x-sweep — likely uninitialized data or bad fluxes" max_cfl n_sub
        n_sub = 100
    end

    if n_sub > 1
        inv = FT(1) / FT(n_sub)
        for p in 1:6; am_panels[p] .*= inv; end
    end
    for _ in 1:n_sub
        for p in 1:6
            advect_x_cs_panel!(rm_panels[p], m_panels[p], am_panels[p],
                               ws.rm_buf, ws.m_buf, grid.Hp, grid.Nc, use_limiter)
        end
    end
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; am_panels[p] .*= fwd; end
    end
    return n_sub
end

function _sweep_y!(rm_panels, m_panels, bm_panels, grid, use_limiter, ws;
                   cfl_limit = eltype(ws.rm_buf)(0.95))
    FT = eltype(ws.rm_buf)
    Hp = grid.Hp
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_y_cs(bm_panels[p], m_panels[p], ws.cfl_y, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))
    if n_sub > 100
        @warn "Extreme CFL subcycling in y-sweep — likely uninitialized data or bad fluxes" max_cfl n_sub
        n_sub = 100
    end

    if n_sub > 1
        inv = FT(1) / FT(n_sub)
        for p in 1:6; bm_panels[p] .*= inv; end
    end
    for _ in 1:n_sub
        for p in 1:6
            advect_y_cs_panel!(rm_panels[p], m_panels[p], bm_panels[p],
                               ws.rm_buf, ws.m_buf, grid.Hp, grid.Nc, use_limiter)
        end
    end
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; bm_panels[p] .*= fwd; end
    end
    return n_sub
end

function _sweep_z!(rm_panels, m_panels, cm_panels, grid, use_limiter, ws)
    Hp = grid.Hp
    Nc = grid.Nc
    Nz = grid.Nz
    # Column-sequential kernel: processes k=1..Nz sequentially per column,
    # clamping gamma to [-1,1] to prevent negative mass.
    # We copy rm/m to workspace buffers before each panel so the kernel reads
    # from ORIGINAL values — this guarantees exact flux telescoping.
    for p in 1:6
        copyto!(ws.rm_buf, rm_panels[p])
        copyto!(ws.m_buf, m_panels[p])
        advect_z_cs_panel_column!(rm_panels[p], m_panels[p],
                                  ws.rm_buf, ws.m_buf,
                                  cm_panels[p], Hp, Nc, Nz, use_limiter)
    end
    return 1
end
