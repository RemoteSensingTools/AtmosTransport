# ---------------------------------------------------------------------------
# Lat-Lon Mass-Flux PPM Advection (Putman & Lin 2007)
#
# PPM counterparts of the Russell-Lerner kernels in mass_flux_advection.jl.
# X uses periodic wrapping; Y falls back to upwind at pole boundaries;
# Z reuses the existing Russell-Lerner kernel (PPM not needed vertically).
#
# Dispatch on Val{ORD} for compile-time kernel specialization (ORD=4,5,6,7).
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# =====================================================================
# X-direction PPM kernel  (periodic boundary)
# =====================================================================

"""Periodic index wrapping: 1-based, domain [1, N]."""
@inline _wrap(i, N) = mod1(i, N)

@kernel function _massflux_x_ppm_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am), Nx, ::Val{ORD}
) where ORD
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rm)

        # 7-point stencil with periodic wrapping
        i3m = _wrap(i - 3, Nx)
        imm = _wrap(i - 2, Nx)
        im  = _wrap(i - 1, Nx)
        ip  = _wrap(i + 1, Nx)
        ipp = _wrap(i + 2, Nx)
        i3p = _wrap(i + 3, Nx)

        c_i3m = _safe_mixing_ratio(rm[i3m, j, k], m[i3m, j, k])
        c_imm = _safe_mixing_ratio(rm[imm, j, k], m[imm, j, k])
        c_im  = _safe_mixing_ratio(rm[im,  j, k], m[im,  j, k])
        c_i   = _safe_mixing_ratio(rm[i,   j, k], m[i,   j, k])
        c_ip  = _safe_mixing_ratio(rm[ip,  j, k], m[ip,  j, k])
        c_ipp = _safe_mixing_ratio(rm[ipp, j, k], m[ipp, j, k])
        c_i3p = _safe_mixing_ratio(rm[i3p, j, k], m[i3p, j, k])

        # PPM edge values for cells i-1, i, i+1
        _,     q_R_im = _ppm_edge_values(c_i3m, c_imm, c_im,  c_ip,  c_ipp,  Val(ORD))
        q_L_i, q_R_i  = _ppm_edge_values(c_imm, c_im,  c_i,   c_ip,  c_ipp,  Val(ORD))
        q_L_ip, _     = _ppm_edge_values(c_im,  c_i,   c_ip,  c_ipp, c_i3p,  Val(ORD))

        # Flux at left face
        am_l = am[i, j, k]
        flux_left = if am_l >= zero(FT)
            alpha = am_l / m[im, j, k]
            alpha * (rm[im, j, k] + (one(FT) - alpha) * m[im, j, k] * (q_R_im - c_im))
        else
            alpha = am_l / m[i, j, k]
            alpha * (rm[i, j, k] - (one(FT) + alpha) * m[i, j, k] * (c_i - q_L_i))
        end

        # Flux at right face
        am_r = am[_wrap(i + 1, Nx + 1), j, k]
        flux_right = if am_r >= zero(FT)
            alpha = am_r / m[i, j, k]
            alpha * (rm[i, j, k] + (one(FT) - alpha) * m[i, j, k] * (q_R_i - c_i))
        else
            alpha = am_r / m[ip, j, k]
            alpha * (rm[ip, j, k] - (one(FT) + alpha) * m[ip, j, k] * (c_ip - q_L_ip))
        end

        rm_new[i, j, k] = rm[i, j, k] + flux_left - flux_right
        m_new[i, j, k]  = m[i, j, k]  + am[i, j, k] - am[_wrap(i + 1, Nx + 1), j, k]
    end
end

# =====================================================================
# Y-direction PPM kernel  (poles → upwind fallback)
# =====================================================================

@kernel function _massflux_y_ppm_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm), Ny, ::Val{ORD}
) where ORD
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin

        # --- PPM edge values for cell j (center) ---
        q_L_j, q_R_j = if j >= 3 && j <= Ny - 2
            c_jmm = _safe_mixing_ratio(rm[i, j - 2, k], m[i, j - 2, k])
            c_jm  = _safe_mixing_ratio(rm[i, j - 1, k], m[i, j - 1, k])
            c_j   = _safe_mixing_ratio(rm[i, j,     k], m[i, j,     k])
            c_jp  = _safe_mixing_ratio(rm[i, j + 1, k], m[i, j + 1, k])
            c_jpp = _safe_mixing_ratio(rm[i, j + 2, k], m[i, j + 2, k])
            _ppm_edge_values(c_jmm, c_jm, c_j, c_jp, c_jpp, Val(ORD))
        else
            c_j = _safe_mixing_ratio(rm[i, j, k], m[i, j, k])
            (c_j, c_j)   # upwind fallback at boundaries
        end

        # --- PPM right edge of cell j-1 (for south flux, upwind from south) ---
        q_R_jm = if j >= 4 && j - 1 <= Ny - 2
            c_jmm2 = _safe_mixing_ratio(rm[i, j - 3, k], m[i, j - 3, k])
            c_jmm  = _safe_mixing_ratio(rm[i, j - 2, k], m[i, j - 2, k])
            c_jm   = _safe_mixing_ratio(rm[i, j - 1, k], m[i, j - 1, k])
            c_j_   = _safe_mixing_ratio(rm[i, j,     k], m[i, j,     k])
            c_jp_  = _safe_mixing_ratio(rm[i, j + 1, k], m[i, j + 1, k])
            _, qR = _ppm_edge_values(c_jmm2, c_jmm, c_jm, c_j_, c_jp_, Val(ORD))
            qR
        else
            _safe_mixing_ratio(rm[i, max(j - 1, 1), k], m[i, max(j - 1, 1), k])
        end

        # --- PPM left edge of cell j+1 (for north flux, upwind from north) ---
        q_L_jp = if j + 1 >= 3 && j <= Ny - 3
            c_jm_  = _safe_mixing_ratio(rm[i, j,     k], m[i, j,     k])
            c_j_   = _safe_mixing_ratio(rm[i, j + 1, k], m[i, j + 1, k])
            c_jp_  = _safe_mixing_ratio(rm[i, j + 2, k], m[i, j + 2, k])
            c_jpp_ = _safe_mixing_ratio(rm[i, j + 3, k], m[i, j + 3, k])
            c_jm2  = _safe_mixing_ratio(rm[i, j - 1, k], m[i, j - 1, k])
            qL, _ = _ppm_edge_values(c_jm2, c_jm_, c_j_, c_jp_, c_jpp_, Val(ORD))
            qL
        else
            _safe_mixing_ratio(rm[i, min(j + 1, Ny), k], m[i, min(j + 1, Ny), k])
        end

        # --- Flux at south face (j) ---
        flux_s = if j > 1
            bm_s = bm[i, j, k]
            if bm_s >= zero(FT)
                beta = bm_s / m[i, j - 1, k]
                c_jm = _safe_mixing_ratio(rm[i, j - 1, k], m[i, j - 1, k])
                beta * (rm[i, j - 1, k] + (one(FT) - beta) * m[i, j - 1, k] * (q_R_jm - c_jm))
            else
                beta = bm_s / m[i, j, k]
                c_j = _safe_mixing_ratio(rm[i, j, k], m[i, j, k])
                beta * (rm[i, j, k] - (one(FT) + beta) * m[i, j, k] * (c_j - q_L_j))
            end
        else
            zero(FT)
        end

        # --- Flux at north face (j+1) ---
        flux_n = if j < Ny
            bm_n = bm[i, j + 1, k]
            if bm_n >= zero(FT)
                beta = bm_n / m[i, j, k]
                c_j = _safe_mixing_ratio(rm[i, j, k], m[i, j, k])
                beta * (rm[i, j, k] + (one(FT) - beta) * m[i, j, k] * (q_R_j - c_j))
            else
                beta = bm_n / m[i, j + 1, k]
                c_jp = _safe_mixing_ratio(rm[i, j + 1, k], m[i, j + 1, k])
                beta * (rm[i, j + 1, k] - (one(FT) + beta) * m[i, j + 1, k] * (c_jp - q_L_jp))
            end
        else
            zero(FT)
        end

        rm_new[i, j, k] = rm[i, j, k] + flux_s - flux_n
        m_new[i, j, k]  = m[i, j, k]  + bm[i, j, k] - bm[i, j + 1, k]
    end
end

# =====================================================================
# Kernel launch wrappers
# =====================================================================

function advect_x_massflux_ppm!(rm_tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am::AbstractArray{FT,3},
                                 grid,
                                 ::Val{ORD},
                                 rm_buf::AbstractArray{FT,3},
                                 m_buf::AbstractArray{FT,3}) where {FT, ORD}
    backend = get_backend(m)
    Nx = grid.Nx
    k! = _massflux_x_ppm_kernel!(backend, 256)
    for (_, rm) in pairs(rm_tracers)
        k!(rm_buf, rm, m_buf, m, am, Nx, Val(ORD); ndrange=size(m))
        synchronize(backend)
        copyto!(rm, rm_buf)
    end
    copyto!(m, m_buf)
    return nothing
end

function advect_y_massflux_ppm!(rm_tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 bm::AbstractArray{FT,3},
                                 grid,
                                 ::Val{ORD},
                                 rm_buf::AbstractArray{FT,3},
                                 m_buf::AbstractArray{FT,3}) where {FT, ORD}
    backend = get_backend(m)
    Ny = grid.Ny
    k! = _massflux_y_ppm_kernel!(backend, 256)
    for (_, rm) in pairs(rm_tracers)
        k!(rm_buf, rm, m_buf, m, bm, Ny, Val(ORD); ndrange=size(m))
        synchronize(backend)
        copyto!(rm, rm_buf)
    end
    copyto!(m, m_buf)
    return nothing
end

# =====================================================================
# Subcycled wrappers with CFL control
# =====================================================================

function advect_x_massflux_ppm_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                           grid, ::Val{ORD},
                                           ws::MassFluxWorkspace{FT};
                                           cfl_limit = FT(0.95)) where {FT, ORD}
    cfl = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    if n_sub > 1
        ws.cfl_x .= am ./ FT(n_sub)
        am_eff = ws.cfl_x
    else
        am_eff = am
    end
    for _ in 1:n_sub
        advect_x_massflux_ppm!(rm_tracers, m, am_eff, grid, Val(ORD),
                                ws.rm_buf, ws.m_buf)
    end
    return n_sub
end

function advect_y_massflux_ppm_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                           grid, ::Val{ORD},
                                           ws::MassFluxWorkspace{FT};
                                           cfl_limit = FT(0.95)) where {FT, ORD}
    cfl = max_cfl_massflux_y(bm, m, ws.cfl_y)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    if n_sub > 1
        ws.cfl_y .= bm ./ FT(n_sub)
        bm_eff = ws.cfl_y
    else
        bm_eff = bm
    end
    for _ in 1:n_sub
        advect_y_massflux_ppm!(rm_tracers, m, bm_eff, grid, Val(ORD),
                                ws.rm_buf, ws.m_buf)
    end
    return n_sub
end

# =====================================================================
# Strang-split wrapper  (X → Y → Z → Z → Y → X)
#
# Z reuses the existing Russell-Lerner kernel (PPM not needed vertically).
# =====================================================================

"""
    strang_split_massflux_ppm!(tracers, m, am, bm, cm, grid::LatitudeLongitudeGrid,
                                ::Val{ORD}, ws)

Strang-split mass-flux advection using Putman & Lin PPM for horizontal
directions and Russell-Lerner for vertical.
"""
function strang_split_massflux_ppm!(tracers::NamedTuple,
                                     m::AbstractArray{FT,3},
                                     am, bm, cm,
                                     grid::LatitudeLongitudeGrid,
                                     ::Val{ORD},
                                     ws::MassFluxWorkspace{FT};
                                     cfl_limit::FT = FT(0.95)) where {FT, ORD}
    # Convert concentration → tracer mass using pre-allocated ws.rm
    ws.rm .= m .* first(values(tracers))
    rm_tracers = NamedTuple{keys(tracers)}((ws.rm,))

    # X → Y → Z → Z → Y → X  (PPM horizontal, Russell-Lerner vertical)
    advect_x_massflux_ppm_subcycled!(rm_tracers, m, am, grid, Val(ORD), ws; cfl_limit)
    advect_y_massflux_ppm_subcycled!(rm_tracers, m, bm, grid, Val(ORD), ws; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, true, ws; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, true, ws; cfl_limit)
    advect_y_massflux_ppm_subcycled!(rm_tracers, m, bm, grid, Val(ORD), ws; cfl_limit)
    advect_x_massflux_ppm_subcycled!(rm_tracers, m, am, grid, Val(ORD), ws; cfl_limit)

    first(values(tracers)) .= ws.rm ./ m
    return nothing
end
