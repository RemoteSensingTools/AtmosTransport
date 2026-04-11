# ---------------------------------------------------------------------------
# Cubed-sphere Strang splitting orchestrator for src_v2
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

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend,
    CPU as KA_CPU

# =========================================================================
# CS panel sweep kernels
#
# These launch on ndrange=(Nc, Nc, Nz) and read/write the interior region
# of halo-padded arrays. The Hp offset is added to all indices.
# =========================================================================

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
# =========================================================================

"""Run X-sweep on one panel's interior with double buffering."""
function _sweep_x_panel!(rm, m, am, scheme, rm_buf, m_buf, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    backend = get_backend(rm)
    if backend isa KA_CPU
        # CPU: direct loop over interior
        @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
            i = ii + Hp; j = jj + Hp
            am_l = flux_scale * am[i, j, k]
            am_r = flux_scale * am[i + 1, j, k]
            Nx_padded = Int32(Nc + 2Hp)
            fl = _xface_tracer_flux(Int32(i), j, k, rm, m, am_l, scheme, Nx_padded)
            fr = _xface_tracer_flux(Int32(i) + Int32(1), j, k, rm, m, am_r, scheme, Nx_padded)
            rm_buf[i, j, k] = rm[i, j, k] + fl - fr
            m_buf[i, j, k]  = m[i, j, k]  + am_l - am_r
        end
    else
        k! = _cs_xsweep_kernel!(backend, 256)
        k!(rm_buf, rm, m_buf, m, am, scheme, Nc, Hp, flux_scale; ndrange=(Nc, Nc, Nz))
        synchronize(backend)
    end
    # Copy buffer back to interior
    _copy_interior!(rm, rm_buf, Nc, Hp, Nz)
    _copy_interior!(m, m_buf, Nc, Hp, Nz)
    return nothing
end

"""Run Y-sweep on one panel's interior with double buffering."""
function _sweep_y_panel!(rm, m, bm, scheme, rm_buf, m_buf, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    backend = get_backend(rm)
    if backend isa KA_CPU
        @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
            i = ii + Hp; j = jj + Hp
            bm_s = flux_scale * bm[i, j, k]
            bm_n = flux_scale * bm[i, j + 1, k]
            Ny_padded = Int32(Nc + 2Hp)
            fs = _yface_tracer_flux(i, Int32(j), k, rm, m, bm_s, scheme, Ny_padded)
            fn = _yface_tracer_flux(i, Int32(j) + Int32(1), k, rm, m, bm_n, scheme, Ny_padded)
            rm_buf[i, j, k] = rm[i, j, k] + fs - fn
            m_buf[i, j, k]  = m[i, j, k]  + bm_s - bm_n
        end
    else
        k! = _cs_ysweep_kernel!(backend, 256)
        k!(rm_buf, rm, m_buf, m, bm, scheme, Nc, Hp, flux_scale; ndrange=(Nc, Nc, Nz))
        synchronize(backend)
    end
    _copy_interior!(rm, rm_buf, Nc, Hp, Nz)
    _copy_interior!(m, m_buf, Nc, Hp, Nz)
    return nothing
end

"""Run Z-sweep on one panel's interior with double buffering."""
function _sweep_z_panel!(rm, m, cm, scheme, rm_buf, m_buf, Nc, Hp, Nz;
                         flux_scale = one(eltype(m)))
    backend = get_backend(rm)
    if backend isa KA_CPU
        @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
            i = ii + Hp; j = jj + Hp
            cm_t = flux_scale * cm[i, j, k]
            cm_b = flux_scale * cm[i, j, k + 1]
            ft = _zface_tracer_flux(i, j, Int32(k), rm, m, cm_t, scheme, Int32(Nz))
            fb = _zface_tracer_flux(i, j, Int32(k) + Int32(1), rm, m, cm_b, scheme, Int32(Nz))
            rm_buf[i, j, k] = rm[i, j, k] + ft - fb
            m_buf[i, j, k]  = m[i, j, k]  + cm_t - cm_b
        end
    else
        k! = _cs_zsweep_kernel!(backend, 256)
        k!(rm_buf, rm, m_buf, m, cm, scheme, Nz, Hp, flux_scale; ndrange=(Nc, Nc, Nz))
        synchronize(backend)
    end
    _copy_interior!(rm, rm_buf, Nc, Hp, Nz)
    _copy_interior!(m, m_buf, Nc, Hp, Nz)
    return nothing
end

"""Copy interior region from buffer back to array."""
function _copy_interior!(dst, src, Nc, Hp, Nz)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        i = ii + Hp; j = jj + Hp
        dst[i, j, k] = src[i, j, k]
    end
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
    rm_buf :: A
    m_buf  :: A
end

function CSAdvectionWorkspace(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64)
    N = mesh.Nc + 2 * mesh.Hp
    rm_buf = zeros(FT, N, N, Nz)
    m_buf  = zeros(FT, N, N, Nz)
    return CSAdvectionWorkspace{FT, typeof(rm_buf)}(rm_buf, m_buf)
end

# =========================================================================
# Public API: strang_split_cs!
# =========================================================================

# =========================================================================
# CFL-based subcycle count
# =========================================================================

"""
    _cs_x_pilot_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_limit) -> Int

Evolving-mass pilot for X-direction subcycling (TM5-style).

Copies panel mass to pilot buffers, simulates the mass evolution through
subcycles, and checks per-face CFL on evolved mass. Increases n_sub until
no face exceeds cfl_limit at any subcycle step.

This is the CS equivalent of the LatLon `_x_subcycling_pass_count` in
StrangSplitting.jl, which also simulates evolving mass.
"""
function _cs_x_pilot_subcycle_count(panels_am::NTuple{6}, panels_m::NTuple{6},
                                     Nc::Int, Hp::Int, Nz::Int,
                                     cfl_limit::Real; max_n_sub::Int = 1024)
    FT = eltype(panels_m[1])

    # Initial estimate from static CFL
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mi = panels_m[p][Hp+i, Hp+j, k]
            mi <= zero(FT) && continue
            c = max(abs(panels_am[p][Hp+i, Hp+j, k]),
                    abs(panels_am[p][Hp+i+1, Hp+j, k])) / mi
            max_cfl = max(max_cfl, c)
        end
    end
    max_cfl <= cfl_limit && return 1
    n_sub = ceil(Int, max_cfl / cfl_limit)

    # Pilot: simulate mass evolution and verify CFL stays within limit
    mx = ntuple(p -> copy(panels_m[p]), 6)
    while true
        # Reset pilot to initial mass
        for p in 1:6
            copyto!(mx[p], panels_m[p])
        end
        flux_scale = inv(FT(n_sub))
        cfl_ok = true

        for pass in 1:n_sub
            # Check CFL on evolved mass for this pass
            @inbounds for p in 1:6
                cfl_ok || break
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    mi = mx[p][Hp+i, Hp+j, k]
                    mi <= zero(FT) && (cfl_ok = false; break)
                    am_l = abs(flux_scale * panels_am[p][Hp+i, Hp+j, k])
                    am_r = abs(flux_scale * panels_am[p][Hp+i+1, Hp+j, k])
                    if max(am_l, am_r) >= cfl_limit * mi
                        cfl_ok = false
                        break
                    end
                end
            end
            cfl_ok || break

            # Evolve pilot mass (only if not last pass — saves work)
            if pass < n_sub
                @inbounds for p in 1:6
                    for k in 1:Nz, j in 1:Nc, i in 1:Nc
                        ii = Hp + i; jj = Hp + j
                        mx[p][ii, jj, k] += flux_scale * panels_am[p][ii, jj, k] -
                                              flux_scale * panels_am[p][ii+1, jj, k]
                    end
                end
            end
        end

        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || error("CS X subcycling exceeded max_n_sub=$max_n_sub")
    end
end

"""Same evolving-mass pilot for Y direction."""
function _cs_y_pilot_subcycle_count(panels_bm::NTuple{6}, panels_m::NTuple{6},
                                     Nc::Int, Hp::Int, Nz::Int,
                                     cfl_limit::Real; max_n_sub::Int = 1024)
    FT = eltype(panels_m[1])
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mi = panels_m[p][Hp+i, Hp+j, k]
            mi <= zero(FT) && continue
            c = max(abs(panels_bm[p][Hp+i, Hp+j, k]),
                    abs(panels_bm[p][Hp+i, Hp+j+1, k])) / mi
            max_cfl = max(max_cfl, c)
        end
    end
    max_cfl <= cfl_limit && return 1
    n_sub = ceil(Int, max_cfl / cfl_limit)

    mx = ntuple(p -> copy(panels_m[p]), 6)
    while true
        for p in 1:6; copyto!(mx[p], panels_m[p]); end
        flux_scale = inv(FT(n_sub))
        cfl_ok = true
        for pass in 1:n_sub
            @inbounds for p in 1:6
                cfl_ok || break
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    mi = mx[p][Hp+i, Hp+j, k]
                    mi <= zero(FT) && (cfl_ok = false; break)
                    if max(abs(flux_scale * panels_bm[p][Hp+i, Hp+j, k]),
                           abs(flux_scale * panels_bm[p][Hp+i, Hp+j+1, k])) >= cfl_limit * mi
                        cfl_ok = false; break
                    end
                end
            end
            cfl_ok || break
            if pass < n_sub
                @inbounds for p in 1:6
                    for k in 1:Nz, j in 1:Nc, i in 1:Nc
                        ii = Hp + i; jj = Hp + j
                        mx[p][ii, jj, k] += flux_scale * panels_bm[p][ii, jj, k] -
                                              flux_scale * panels_bm[p][ii, jj+1, k]
                    end
                end
            end
        end
        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || error("CS Y subcycling exceeded max_n_sub=$max_n_sub")
    end
end

"""Same evolving-mass pilot for Z direction."""
function _cs_z_pilot_subcycle_count(panels_cm::NTuple{6}, panels_m::NTuple{6},
                                     Nc::Int, Hp::Int, Nz::Int,
                                     cfl_limit::Real; max_n_sub::Int = 1024)
    FT = eltype(panels_m[1])
    max_cfl = zero(FT)
    @inbounds for p in 1:6
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mi = panels_m[p][Hp+i, Hp+j, k]
            mi <= zero(FT) && continue
            c = max(abs(panels_cm[p][Hp+i, Hp+j, k]),
                    abs(panels_cm[p][Hp+i, Hp+j, k+1])) / mi
            max_cfl = max(max_cfl, c)
        end
    end
    max_cfl <= cfl_limit && return 1
    n_sub = ceil(Int, max_cfl / cfl_limit)

    mx = ntuple(p -> copy(panels_m[p]), 6)
    while true
        for p in 1:6; copyto!(mx[p], panels_m[p]); end
        flux_scale = inv(FT(n_sub))
        cfl_ok = true
        for pass in 1:n_sub
            @inbounds for p in 1:6
                cfl_ok || break
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    mi = mx[p][Hp+i, Hp+j, k]
                    mi <= zero(FT) && (cfl_ok = false; break)
                    if max(abs(flux_scale * panels_cm[p][Hp+i, Hp+j, k]),
                           abs(flux_scale * panels_cm[p][Hp+i, Hp+j, k+1])) >= cfl_limit * mi
                        cfl_ok = false; break
                    end
                end
            end
            cfl_ok || break
            if pass < n_sub
                @inbounds for p in 1:6
                    for k in 1:Nz, j in 1:Nc, i in 1:Nc
                        ii = Hp + i; jj = Hp + j
                        mx[p][ii, jj, k] += flux_scale * panels_cm[p][ii, jj, k] -
                                              flux_scale * panels_cm[p][ii, jj, k+1]
                    end
                end
            end
        end
        cfl_ok && return n_sub
        n_sub += 1
        n_sub <= max_n_sub || error("CS Z subcycling exceeded max_n_sub=$max_n_sub")
    end
end

"""
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, scheme, workspace; flux_scale=1, cfl_limit=0.95)

Perform one Strang-split advection step on a 6-panel cubed-sphere field
with automatic CFL-based subcycling per direction.

Splitting sequence: X → halo → Y → halo → Z → Z → halo → Y → halo → X

Each direction is subcycled independently: if max CFL for X exceeds
`cfl_limit`, the X sweep is repeated n_sub times with flux_scale/n_sub.
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
    rm_buf, m_buf = workspace.rm_buf, workspace.m_buf
    FT = eltype(panels_m[1])
    fs = convert(FT, flux_scale)
    cfl_ft = convert(FT, cfl_limit)

    # Compute subcycle counts per direction using evolving-mass pilot
    n_x = _cs_x_pilot_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft)
    n_y = _cs_y_pilot_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft)
    n_z = _cs_z_pilot_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft)

    fs_x = fs / FT(n_x)
    fs_y = fs / FT(n_y)
    fs_z = fs / FT(n_z)

    # ---- X sweep (subcycled) ----
    for _ in 1:n_x
        for p in 1:6
            _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_x)
        end
        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
    end

    # ---- Y sweep (subcycled) ----
    for _ in 1:n_y
        for p in 1:6
            _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_y)
        end
        fill_panel_halos!(panels_rm, mesh; dir=2)
        fill_panel_halos!(panels_m, mesh; dir=2)
    end

    # ---- Z sweep × 2 (subcycled) ----
    for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_z)
        end
    end
    for _ in 1:n_z
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_z)
        end
    end

    # ---- Reverse: Y sweep (subcycled) ----
    fill_panel_halos!(panels_rm, mesh; dir=2)
    fill_panel_halos!(panels_m, mesh; dir=2)
    for _ in 1:n_y
        for p in 1:6
            _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_y)
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
                             scheme, rm_buf, m_buf, Nc, Hp, Nz; flux_scale=fs_x)
        end
        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
    end

    return nothing
end

export strang_split_cs!, CSAdvectionWorkspace
