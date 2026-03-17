# ---------------------------------------------------------------------------
# Prather (1986) Second-Order Moments advection — Cubed-Sphere port
#
# Adapts the lat-lon Prather implementation to cubed-sphere grids.
# Key differences from LL:
#   - Hp-offset indexing (ii = Hp + i) instead of periodic wrap
#   - Panel halos filled via fill_panel_halos! before each horizontal sweep
#   - Slopes stored as NTuple{6} of haloed panel arrays
#   - Z-sweep is column-sequential (reads from src buffers)
#
# The flux formulas and moment update equations are IDENTICAL to LL Prather.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Workspace for Prather advection on cubed-sphere grids.
Holds prognostic slopes (6 panels each, haloed) plus shared double-buffer arrays.

$(FIELDS)
"""
struct CSPratherWorkspace{FT, A3h <: AbstractArray{FT,3}}
    "x-slope panels (haloed, persist across timesteps)"
    rxm     :: NTuple{6, A3h}
    "y-slope panels (haloed, persist across timesteps)"
    rym     :: NTuple{6, A3h}
    "z-slope panels (haloed, persist across timesteps)"
    rzm     :: NTuple{6, A3h}
    "double-buffer for rm (single panel, shared)"
    rm_buf  :: A3h
    "double-buffer for m (single panel, shared)"
    m_buf   :: A3h
    "double-buffer for rxm"
    rxm_buf :: A3h
    "double-buffer for rym"
    rym_buf :: A3h
    "double-buffer for rzm"
    rzm_buf :: A3h
end

"""
    allocate_cs_prather_workspace(grid, arch) → CSPratherWorkspace

Allocate slope panels and buffers for one tracer on a cubed-sphere grid.
"""
function allocate_cs_prather_workspace(grid::CubedSphereGrid{FT}, arch) where FT
    AT = array_type(arch)
    N = grid.Nc + 2 * grid.Hp
    Nz = grid.Nz
    rxm = ntuple(_ -> AT(zeros(FT, N, N, Nz)), 6)
    rym = ntuple(_ -> AT(zeros(FT, N, N, Nz)), 6)
    rzm = ntuple(_ -> AT(zeros(FT, N, N, Nz)), 6)
    ref = AT(zeros(FT, N, N, Nz))
    CSPratherWorkspace{FT, typeof(ref)}(
        rxm, rym, rzm,
        similar(ref), similar(ref), similar(ref), similar(ref), similar(ref))
end

"""
    allocate_cs_prather_workspaces(tracers, grid, arch) → NamedTuple

Allocate one CSPratherWorkspace per tracer, keyed by tracer name.
"""
function allocate_cs_prather_workspaces(tracers::NamedTuple, grid::CubedSphereGrid, arch)
    names = keys(tracers)
    workspaces = Tuple(allocate_cs_prather_workspace(grid, arch) for _ in names)
    return NamedTuple{names}(workspaces)
end

# ---------------------------------------------------------------------------
# X-direction kernel (adapted from _prather_x_kernel! with Hp-offset indexing)
# ---------------------------------------------------------------------------

@kernel function _prather_x_cs_kernel!(
    rm_new, m_new, rxm_new, rym_new, rzm_new,
    @Const(rm), @Const(m), @Const(rxm), @Const(rym), @Const(rzm),
    @Const(am), Hp, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)

        # Neighbors (from halos — no periodic wrap needed)
        rm_i  = rm[ii, jj, k];     m_i  = m[ii, jj, k]
        rm_im = rm[ii-1, jj, k];   m_im = m[ii-1, jj, k]
        rm_ip = rm[ii+1, jj, k];   m_ip = m[ii+1, jj, k]

        rxm_i  = rxm[ii, jj, k]
        rxm_im = rxm[ii-1, jj, k]
        rxm_ip = rxm[ii+1, jj, k]

        rym_i  = rym[ii, jj, k];  rym_im = rym[ii-1, jj, k];  rym_ip = rym[ii+1, jj, k]
        rzm_i  = rzm[ii, jj, k];  rzm_im = rzm[ii-1, jj, k];  rzm_ip = rzm[ii+1, jj, k]

        # Limiter on along-slopes before flux computation
        if use_limiter
            rxm_im = max(min(rxm_im, rm_im), -rm_im)
            rxm_i  = max(min(rxm_i,  rm_i),  -rm_i)
            rxm_ip = max(min(rxm_ip, rm_ip), -rm_ip)
        end

        # --- Left face flux (am[i, j, k], interior-indexed) ---
        am_l = am[i, j, k]
        if am_l >= zero(FT)
            alpha = m_im > eps(FT) * 100 ? am_l / m_im : zero(FT)
            f_l  = alpha * (rm_im + (one(FT) - alpha) * rxm_im)
            pf_l = am_l * (alpha * alpha * rxm_im - FT(3) * f_l)
            fy_l = alpha * rym_im
            fz_l = alpha * rzm_im
        else
            alpha = m_i > eps(FT) * 100 ? am_l / m_i : zero(FT)
            f_l  = alpha * (rm_i - (one(FT) + alpha) * rxm_i)
            pf_l = am_l * (alpha * alpha * rxm_i - FT(3) * f_l)
            fy_l = alpha * rym_i
            fz_l = alpha * rzm_i
        end

        # --- Right face flux (am[i+1, j, k]) ---
        am_r = am[i + 1, j, k]
        if am_r >= zero(FT)
            alpha = m_i > eps(FT) * 100 ? am_r / m_i : zero(FT)
            f_r  = alpha * (rm_i + (one(FT) - alpha) * rxm_i)
            pf_r = am_r * (alpha * alpha * rxm_i - FT(3) * f_r)
            fy_r = alpha * rym_i
            fz_r = alpha * rzm_i
        else
            alpha = m_ip > eps(FT) * 100 ? am_r / m_ip : zero(FT)
            f_r  = alpha * (rm_ip - (one(FT) + alpha) * rxm_ip)
            pf_r = am_r * (alpha * alpha * rxm_ip - FT(3) * f_r)
            fy_r = alpha * rym_ip
            fz_r = alpha * rzm_ip
        end

        # --- Updates ---
        rm_new_val = rm_i + f_l - f_r
        m_new_val  = m_i + am_l - am_r
        m_safe     = max(m_new_val, eps(FT) * 100)

        # Along-slope update (Prather Eq. 21)
        rxm_new_val = (rxm_i + (pf_l - pf_r)
                       - (am_l - am_r) * rxm_i
                       + FT(3) * ((am_l + am_r) * rm_i - (f_l + f_r) * m_i)
                      ) / m_safe

        # Cross-slope update (passive transport)
        rym_new_val = rym_i + fy_l - fy_r
        rzm_new_val = rzm_i + fz_l - fz_r

        if use_limiter
            rxm_new_val = max(min(rxm_new_val, rm_new_val), -rm_new_val)
        end

        rm_new[ii, jj, k]  = rm_new_val
        m_new[ii, jj, k]   = m_new_val
        rxm_new[ii, jj, k] = rxm_new_val
        rym_new[ii, jj, k] = rym_new_val
        rzm_new[ii, jj, k] = rzm_new_val
    end
end

# ---------------------------------------------------------------------------
# Y-direction kernel
# ---------------------------------------------------------------------------

@kernel function _prather_y_cs_kernel!(
    rm_new, m_new, rxm_new, rym_new, rzm_new,
    @Const(rm), @Const(m), @Const(rxm), @Const(rym), @Const(rzm),
    @Const(bm), Hp, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)

        rm_i  = rm[ii, jj, k];     m_i  = m[ii, jj, k]
        rm_jm = rm[ii, jj-1, k];   m_jm = m[ii, jj-1, k]
        rm_jp = rm[ii, jj+1, k];   m_jp = m[ii, jj+1, k]

        # Along-slope = rym; cross-slopes = rxm, rzm
        rym_i  = rym[ii, jj, k]
        rym_jm = rym[ii, jj-1, k]
        rym_jp = rym[ii, jj+1, k]

        rxm_i  = rxm[ii, jj, k];  rxm_jm = rxm[ii, jj-1, k];  rxm_jp = rxm[ii, jj+1, k]
        rzm_i  = rzm[ii, jj, k];  rzm_jm = rzm[ii, jj-1, k];  rzm_jp = rzm[ii, jj+1, k]

        if use_limiter
            rym_jm = max(min(rym_jm, rm_jm), -rm_jm)
            rym_i  = max(min(rym_i,  rm_i),  -rm_i)
            rym_jp = max(min(rym_jp, rm_jp), -rm_jp)
        end

        # --- South face flux (bm[i, j, k]) ---
        bm_s = bm[i, j, k]
        if bm_s >= zero(FT)
            alpha = m_jm > eps(FT) * 100 ? bm_s / m_jm : zero(FT)
            f_s  = alpha * (rm_jm + (one(FT) - alpha) * rym_jm)
            pf_s = bm_s * (alpha * alpha * rym_jm - FT(3) * f_s)
            fx_s = alpha * rxm_jm
            fz_s = alpha * rzm_jm
        else
            alpha = m_i > eps(FT) * 100 ? bm_s / m_i : zero(FT)
            f_s  = alpha * (rm_i - (one(FT) + alpha) * rym_i)
            pf_s = bm_s * (alpha * alpha * rym_i - FT(3) * f_s)
            fx_s = alpha * rxm_i
            fz_s = alpha * rzm_i
        end

        # --- North face flux (bm[i, j+1, k]) ---
        bm_n = bm[i, j + 1, k]
        if bm_n >= zero(FT)
            alpha = m_i > eps(FT) * 100 ? bm_n / m_i : zero(FT)
            f_n  = alpha * (rm_i + (one(FT) - alpha) * rym_i)
            pf_n = bm_n * (alpha * alpha * rym_i - FT(3) * f_n)
            fx_n = alpha * rxm_i
            fz_n = alpha * rzm_i
        else
            alpha = m_jp > eps(FT) * 100 ? bm_n / m_jp : zero(FT)
            f_n  = alpha * (rm_jp - (one(FT) + alpha) * rym_jp)
            pf_n = bm_n * (alpha * alpha * rym_jp - FT(3) * f_n)
            fx_n = alpha * rxm_jp
            fz_n = alpha * rzm_jp
        end

        rm_new_val = rm_i + f_s - f_n
        m_new_val  = m_i + bm_s - bm_n
        m_safe     = max(m_new_val, eps(FT) * 100)

        rym_new_val = (rym_i + (pf_s - pf_n)
                       - (bm_s - bm_n) * rym_i
                       + FT(3) * ((bm_s + bm_n) * rm_i - (f_s + f_n) * m_i)
                      ) / m_safe

        rxm_new_val = rxm_i + fx_s - fx_n
        rzm_new_val = rzm_i + fz_s - fz_n

        if use_limiter
            rym_new_val = max(min(rym_new_val, rm_new_val), -rm_new_val)
        end

        rm_new[ii, jj, k]  = rm_new_val
        m_new[ii, jj, k]   = m_new_val
        rxm_new[ii, jj, k] = rxm_new_val
        rym_new[ii, jj, k] = rym_new_val
        rzm_new[ii, jj, k] = rzm_new_val
    end
end

# ---------------------------------------------------------------------------
# Z-direction kernel (column-sequential, reads from src buffers)
# ---------------------------------------------------------------------------

@kernel function _prather_z_cs_kernel!(
    rm, m, rxm_p, rym_p, rzm_p,
    @Const(rm_src), @Const(m_src), @Const(rxm_src), @Const(rym_src), @Const(rzm_src),
    @Const(cm), Hp, Nz, use_limiter
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)

        for k in 1:Nz
            rm_i  = rm_src[ii, jj, k];  m_i  = m_src[ii, jj, k]
            rzm_i = rzm_src[ii, jj, k]
            rxm_i = rxm_src[ii, jj, k]
            rym_i = rym_src[ii, jj, k]

            rzm_lim = use_limiter ? max(min(rzm_i, rm_i), -rm_i) : rzm_i

            # --- Top face (cm[i, j, k]) ---
            cm_t = cm[i, j, k]
            if k == 1
                f_t = zero(FT);  pf_t = zero(FT);  fx_t = zero(FT);  fy_t = zero(FT)
            elseif cm_t >= zero(FT)
                rm_km  = rm_src[ii, jj, k-1];  m_km = m_src[ii, jj, k-1]
                rzm_km = rzm_src[ii, jj, k-1]
                if use_limiter; rzm_km = max(min(rzm_km, rm_km), -rm_km); end
                alpha  = m_km > eps(FT) * 100 ? cm_t / m_km : zero(FT)
                f_t    = alpha * (rm_km + (one(FT) - alpha) * rzm_km)
                pf_t   = cm_t * (alpha * alpha * rzm_km - FT(3) * f_t)
                fx_t   = alpha * rxm_src[ii, jj, k-1]
                fy_t   = alpha * rym_src[ii, jj, k-1]
            else
                alpha  = m_i > eps(FT) * 100 ? cm_t / m_i : zero(FT)
                f_t    = alpha * (rm_i - (one(FT) + alpha) * rzm_lim)
                pf_t   = cm_t * (alpha * alpha * rzm_lim - FT(3) * f_t)
                fx_t   = alpha * rxm_i
                fy_t   = alpha * rym_i
            end

            # --- Bottom face (cm[i, j, k+1]) ---
            cm_b = cm[i, j, k + 1]
            if k == Nz
                f_b = zero(FT);  pf_b = zero(FT);  fx_b = zero(FT);  fy_b = zero(FT)
            elseif cm_b >= zero(FT)
                alpha  = m_i > eps(FT) * 100 ? cm_b / m_i : zero(FT)
                f_b    = alpha * (rm_i + (one(FT) - alpha) * rzm_lim)
                pf_b   = cm_b * (alpha * alpha * rzm_lim - FT(3) * f_b)
                fx_b   = alpha * rxm_i
                fy_b   = alpha * rym_i
            else
                rm_kp  = rm_src[ii, jj, k+1];  m_kp = m_src[ii, jj, k+1]
                rzm_kp = rzm_src[ii, jj, k+1]
                if use_limiter; rzm_kp = max(min(rzm_kp, rm_kp), -rm_kp); end
                alpha  = m_kp > eps(FT) * 100 ? cm_b / m_kp : zero(FT)
                f_b    = alpha * (rm_kp - (one(FT) + alpha) * rzm_kp)
                pf_b   = cm_b * (alpha * alpha * rzm_kp - FT(3) * f_b)
                fx_b   = alpha * rxm_src[ii, jj, k+1]
                fy_b   = alpha * rym_src[ii, jj, k+1]
            end

            rm_new_val = rm_i + f_t - f_b
            m_new_val  = m_i + cm_t - cm_b
            m_safe     = max(m_new_val, eps(FT) * 100)

            rzm_new_val = (rzm_i + (pf_t - pf_b)
                           - (cm_t - cm_b) * rzm_i
                           + FT(3) * ((cm_t + cm_b) * rm_i - (f_t + f_b) * m_i)
                          ) / m_safe

            rxm_new_val = rxm_i + fx_t - fx_b
            rym_new_val = rym_i + fy_t - fy_b

            if use_limiter
                rzm_new_val = max(min(rzm_new_val, rm_new_val), -rm_new_val)
            end

            rm[ii, jj, k]    = rm_new_val
            m[ii, jj, k]     = m_new_val
            rxm_p[ii, jj, k] = rxm_new_val
            rym_p[ii, jj, k] = rym_new_val
            rzm_p[ii, jj, k] = rzm_new_val
        end
    end
end

# ---------------------------------------------------------------------------
# Panel-level advection functions
# ---------------------------------------------------------------------------

function _advect_x_cs_prather_panel!(rm, m, am, pw, rxm_p, rym_p, rzm_p,
                                      Hp, Nc, Nz, use_limiter)
    backend = get_backend(rm)
    k! = _prather_x_cs_kernel!(backend, 256)
    k!(pw.rm_buf, pw.m_buf, pw.rxm_buf, pw.rym_buf, pw.rzm_buf,
       rm, m, rxm_p, rym_p, rzm_p, am, Hp, use_limiter;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, pw.rm_buf, Hp, Nc, Nz)
    _copy_interior!(m, pw.m_buf, Hp, Nc, Nz)
    _copy_interior!(rxm_p, pw.rxm_buf, Hp, Nc, Nz)
    _copy_interior!(rym_p, pw.rym_buf, Hp, Nc, Nz)
    _copy_interior!(rzm_p, pw.rzm_buf, Hp, Nc, Nz)
end

function _advect_y_cs_prather_panel!(rm, m, bm, pw, rxm_p, rym_p, rzm_p,
                                      Hp, Nc, Nz, use_limiter)
    backend = get_backend(rm)
    k! = _prather_y_cs_kernel!(backend, 256)
    k!(pw.rm_buf, pw.m_buf, pw.rxm_buf, pw.rym_buf, pw.rzm_buf,
       rm, m, rxm_p, rym_p, rzm_p, bm, Hp, use_limiter;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    _copy_interior!(rm, pw.rm_buf, Hp, Nc, Nz)
    _copy_interior!(m, pw.m_buf, Hp, Nc, Nz)
    _copy_interior!(rxm_p, pw.rxm_buf, Hp, Nc, Nz)
    _copy_interior!(rym_p, pw.rym_buf, Hp, Nc, Nz)
    _copy_interior!(rzm_p, pw.rzm_buf, Hp, Nc, Nz)
end

function _advect_z_cs_prather_panel!(rm, m, rxm_p, rym_p, rzm_p,
                                      rm_buf, m_buf, rxm_buf, rym_buf, rzm_buf,
                                      cm, Hp, Nc, Nz, use_limiter)
    backend = get_backend(rm)
    k! = _prather_z_cs_kernel!(backend, 256)
    k!(rm, m, rxm_p, rym_p, rzm_p,
       rm_buf, m_buf, rxm_buf, rym_buf, rzm_buf,
       cm, Hp, Nz, use_limiter; ndrange=(Nc, Nc))
    synchronize(backend)
end

# ---------------------------------------------------------------------------
# Sweep functions (X/Y with halo exchange, Z column-local)
# ---------------------------------------------------------------------------

function _sweep_x_prather!(rm_panels, m_panels, am_panels, grid, pw_cs, use_limiter,
                             cfl_ws; cfl_limit=eltype(pw_cs.rm_buf)(0.95))
    FT = eltype(pw_cs.rm_buf)
    Hp, Nc, Nz = grid.Hp, grid.Nc, grid.Nz

    # Fill halos for rm, m, and all slopes
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)
    fill_panel_halos!(pw_cs.rxm, grid)
    fill_panel_halos!(pw_cs.rym, grid)
    fill_panel_halos!(pw_cs.rzm, grid)

    # CFL-adaptive subcycling
    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_x_cs(am_panels[p], m_panels[p], cfl_ws, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))
    if n_sub > 100
        @warn "Extreme CFL in Prather x-sweep" max_cfl n_sub
        n_sub = 100
    end

    if n_sub > 1
        inv = FT(1) / FT(n_sub)
        for p in 1:6; am_panels[p] .*= inv; end
    end
    for _ in 1:n_sub
        for p in 1:6
            _advect_x_cs_prather_panel!(rm_panels[p], m_panels[p], am_panels[p],
                                         pw_cs, pw_cs.rxm[p], pw_cs.rym[p], pw_cs.rzm[p],
                                         Hp, Nc, Nz, use_limiter)
        end
    end
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; am_panels[p] .*= fwd; end
    end
    return n_sub
end

function _sweep_y_prather!(rm_panels, m_panels, bm_panels, grid, pw_cs, use_limiter,
                             cfl_ws; cfl_limit=eltype(pw_cs.rm_buf)(0.95))
    FT = eltype(pw_cs.rm_buf)
    Hp, Nc, Nz = grid.Hp, grid.Nc, grid.Nz

    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)
    fill_panel_halos!(pw_cs.rxm, grid)
    fill_panel_halos!(pw_cs.rym, grid)
    fill_panel_halos!(pw_cs.rzm, grid)

    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_y_cs(bm_panels[p], m_panels[p], cfl_ws, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))
    if n_sub > 100
        @warn "Extreme CFL in Prather y-sweep" max_cfl n_sub
        n_sub = 100
    end

    if n_sub > 1
        inv = FT(1) / FT(n_sub)
        for p in 1:6; bm_panels[p] .*= inv; end
    end
    for _ in 1:n_sub
        for p in 1:6
            _advect_y_cs_prather_panel!(rm_panels[p], m_panels[p], bm_panels[p],
                                         pw_cs, pw_cs.rxm[p], pw_cs.rym[p], pw_cs.rzm[p],
                                         Hp, Nc, Nz, use_limiter)
        end
    end
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; bm_panels[p] .*= fwd; end
    end
    return n_sub
end

function _sweep_z_prather!(rm_panels, m_panels, cm_panels, grid, pw_cs, use_limiter)
    Hp, Nc, Nz = grid.Hp, grid.Nc, grid.Nz
    # No fill_panel_halos! needed — Z is column-local
    for p in 1:6
        # Copy originals to buffers (flux telescoping guarantee)
        copyto!(pw_cs.rm_buf, rm_panels[p])
        copyto!(pw_cs.m_buf, m_panels[p])
        copyto!(pw_cs.rxm_buf, pw_cs.rxm[p])
        copyto!(pw_cs.rym_buf, pw_cs.rym[p])
        copyto!(pw_cs.rzm_buf, pw_cs.rzm[p])

        _advect_z_cs_prather_panel!(rm_panels[p], m_panels[p],
                                     pw_cs.rxm[p], pw_cs.rym[p], pw_cs.rzm[p],
                                     pw_cs.rm_buf, pw_cs.m_buf,
                                     pw_cs.rxm_buf, pw_cs.rym_buf, pw_cs.rzm_buf,
                                     cm_panels[p], Hp, Nc, Nz, use_limiter)
    end
end

# ---------------------------------------------------------------------------
# Main entry point: Strang splitting X → Y → Z → Z → Y → X
# ---------------------------------------------------------------------------

"""
    strang_split_prather_cs!(rm_panels, m_panels, am, bm, cm, grid, pw_cs, use_limiter;
                              cfl_limit=0.95, cfl_ws=nothing)

Cubed-sphere Prather advection with Strang splitting and prognostic slopes.
`pw_cs` is a `CSPratherWorkspace` for the tracer being advected.
`cfl_ws` is a CFL scratch array from `CubedSphereMassFluxWorkspace`.
"""
function strang_split_prather_cs!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                                   am_panels, bm_panels, cm_panels,
                                   grid::CubedSphereGrid, pw_cs::CSPratherWorkspace,
                                   use_limiter::Bool;
                                   cfl_limit=eltype(pw_cs.rm_buf)(0.95),
                                   cfl_ws_x=nothing, cfl_ws_y=nothing)
    _sweep_x_prather!(rm_panels, m_panels, am_panels, grid, pw_cs, use_limiter,
                       cfl_ws_x; cfl_limit)
    _sweep_y_prather!(rm_panels, m_panels, bm_panels, grid, pw_cs, use_limiter,
                       cfl_ws_y; cfl_limit)
    _sweep_z_prather!(rm_panels, m_panels, cm_panels, grid, pw_cs, use_limiter)
    _sweep_z_prather!(rm_panels, m_panels, cm_panels, grid, pw_cs, use_limiter)
    _sweep_y_prather!(rm_panels, m_panels, bm_panels, grid, pw_cs, use_limiter,
                       cfl_ws_y; cfl_limit)
    _sweep_x_prather!(rm_panels, m_panels, am_panels, grid, pw_cs, use_limiter,
                       cfl_ws_x; cfl_limit)
    return nothing
end
