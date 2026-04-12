# ---------------------------------------------------------------------------
# Prather (1986) Second-Order Moments advection — GPU-accelerated
#
# Implements "first-order Prather": 4 moments per tracer (rm, rxm, rym, rzm).
# Slopes are PROGNOSTIC — they persist between timesteps and are updated with
# second-moment fluxes (pf). This preserves tracer gradients much better and
# significantly reduces numerical diffusion compared to diagnostic slopes.
#
# Reference: Prather, M.J. (1986). "Numerical advection by conservation of
#   second-order moments." J. Geophys. Res., 91, 6671-6681.
# TM5 implementation: deps/tm5/base/src/advectx.F90, lines 660-717.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

export PratherAdvection, PratherWorkspace, allocate_prather_workspace
export strang_split_prather!, allocate_prather_workspaces

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Prather (1986) second-order moments advection scheme.
Carries prognostic slopes (rxm, rym, rzm) alongside tracer mass.

$(FIELDS)
"""
struct PratherAdvection <: AbstractAdvectionScheme
    "apply positivity limits on slopes (|rxm| ≤ rm)"
    use_limiter :: Bool
end
PratherAdvection() = PratherAdvection(true)

"""
$(TYPEDEF)

Workspace holding prognostic slope fields for one tracer + double-buffer outputs.

$(FIELDS)
"""
struct PratherWorkspace{FT, A3 <: AbstractArray{FT,3}}
    "x-slope of tracer mass"
    rxm     :: A3
    "y-slope of tracer mass"
    rym     :: A3
    "z-slope of tracer mass"
    rzm     :: A3
    "double-buffer output for rm"
    rm_buf  :: A3
    "double-buffer output for air mass"
    m_buf   :: A3
    "double-buffer output for rxm"
    rxm_buf :: A3
    "double-buffer output for rym"
    rym_buf :: A3
    "double-buffer output for rzm"
    rzm_buf :: A3
end

"""
    allocate_prather_workspace(m::AbstractArray{FT,3}) → PratherWorkspace{FT}

Allocate slope and buffer arrays matching the shape of air mass `m`.
"""
function allocate_prather_workspace(m::AbstractArray{FT,3}) where FT
    sz = size(m)
    AT = typeof(m)
    make = () -> similar(m)
    PratherWorkspace{FT, AT}(
        fill!(make(), zero(FT)),  # rxm — initialized to zero
        fill!(make(), zero(FT)),  # rym
        fill!(make(), zero(FT)),  # rzm
        make(),                    # rm_buf
        make(),                    # m_buf
        make(),                    # rxm_buf
        make(),                    # rym_buf
        make(),                    # rzm_buf
    )
end

# ---------------------------------------------------------------------------
# X-direction kernel
# ---------------------------------------------------------------------------

@kernel function _prather_x_kernel!(
    rm_new, m_new, rxm_new, rym_new, rzm_new,
    @Const(rm), @Const(m), @Const(rxm), @Const(rym), @Const(rzm),
    @Const(am), Nx, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rm)

        # Periodic neighbors
        im = i == 1  ? Nx : i - 1
        ip = i == Nx ? 1  : i + 1

        # Local values
        rm_i  = rm[i,  j, k];  m_i  = m[i,  j, k]
        rm_im = rm[im, j, k];  m_im = m[im, j, k]
        rm_ip = rm[ip, j, k];  m_ip = m[ip, j, k]

        rxm_i  = rxm[i,  j, k]
        rxm_im = rxm[im, j, k]
        rxm_ip = rxm[ip, j, k]

        rym_i  = rym[i,  j, k];  rym_im = rym[im, j, k];  rym_ip = rym[ip, j, k]
        rzm_i  = rzm[i,  j, k];  rzm_im = rzm[im, j, k];  rzm_ip = rzm[ip, j, k]

        # Apply positivity limiter on slopes before flux computation
        if use_limiter
            rxm_im = max(min(rxm_im, rm_im), -rm_im)
            rxm_i  = max(min(rxm_i,  rm_i),  -rm_i)
            rxm_ip = max(min(rxm_ip, rm_ip), -rm_ip)
        end

        # --- Left face flux (am[i, j, k]) ---
        am_l = am[i, j, k]
        if am_l >= zero(FT)
            # Donor = cell im
            alpha = m_im > eps(FT) * 100 ? am_l / m_im : zero(FT)
            f_l  = alpha * (rm_im + (one(FT) - alpha) * rxm_im)
            pf_l = am_l * (alpha * alpha * rxm_im - FT(3) * f_l)
            fy_l = alpha * rym_im
            fz_l = alpha * rzm_im
        else
            # Donor = cell i
            alpha = m_i > eps(FT) * 100 ? am_l / m_i : zero(FT)
            f_l  = alpha * (rm_i - (one(FT) + alpha) * rxm_i)
            pf_l = am_l * (alpha * alpha * rxm_i - FT(3) * f_l)
            fy_l = alpha * rym_i
            fz_l = alpha * rzm_i
        end

        # --- Right face flux (am[i+1, j, k]) ---
        am_r = am[i + 1, j, k]
        if am_r >= zero(FT)
            # Donor = cell i
            alpha = m_i > eps(FT) * 100 ? am_r / m_i : zero(FT)
            f_r  = alpha * (rm_i + (one(FT) - alpha) * rxm_i)
            pf_r = am_r * (alpha * alpha * rxm_i - FT(3) * f_r)
            fy_r = alpha * rym_i
            fz_r = alpha * rzm_i
        else
            # Donor = cell ip
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

        # Slope update (Prather Eq. 21 / TM5 advectx.F90:707-710)
        rxm_new_val = (rxm_i + (pf_l - pf_r)
                       - (am_l - am_r) * rxm_i
                       + FT(3) * ((am_l + am_r) * rm_i - (f_l + f_r) * m_i)
                      ) / m_safe

        # Cross-slope update (passive transport)
        rym_new_val = rym_i + fy_l - fy_r
        rzm_new_val = rzm_i + fz_l - fz_r

        # Apply limiter on new slope
        if use_limiter
            rxm_new_val = max(min(rxm_new_val, rm_new_val), -rm_new_val)
        end

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = m_new_val
        rxm_new[i, j, k] = rxm_new_val
        rym_new[i, j, k] = rym_new_val
        rzm_new[i, j, k] = rzm_new_val
    end
end

# ---------------------------------------------------------------------------
# Y-direction kernel
# ---------------------------------------------------------------------------

@kernel function _prather_y_kernel!(
    rm_new, m_new, rxm_new, rym_new, rzm_new,
    @Const(rm), @Const(m), @Const(rxm), @Const(rym), @Const(rzm),
    @Const(bm), Ny, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rm)

        # For Y: rym is the "along" slope, rxm/rzm are cross-slopes
        rm_i  = rm[i, j, k];  m_i  = m[i, j, k]
        rym_i = rym[i, j, k]
        rxm_i = rxm[i, j, k]
        rzm_i = rzm[i, j, k]

        # Apply limiter on along-slope before flux computation
        rym_lim = use_limiter ? max(min(rym_i, rm_i), -rm_i) : rym_i

        # --- South face (bm[i, j, k]) ---
        am_s = bm[i, j, k]
        if j == 1
            # South pole: no flux
            f_s  = zero(FT)
            pf_s = zero(FT)
            fx_s = zero(FT)
            fz_s = zero(FT)
        elseif am_s >= zero(FT)
            # Donor = cell j-1
            rm_jm  = rm[i, j-1, k];  m_jm = m[i, j-1, k]
            rym_jm = rym[i, j-1, k]
            if use_limiter; rym_jm = max(min(rym_jm, rm_jm), -rm_jm); end
            alpha  = m_jm > eps(FT) * 100 ? am_s / m_jm : zero(FT)
            f_s    = alpha * (rm_jm + (one(FT) - alpha) * rym_jm)
            pf_s   = am_s * (alpha * alpha * rym_jm - FT(3) * f_s)
            fx_s   = alpha * rxm[i, j-1, k]
            fz_s   = alpha * rzm[i, j-1, k]
        else
            # Donor = cell j
            alpha  = m_i > eps(FT) * 100 ? am_s / m_i : zero(FT)
            f_s    = alpha * (rm_i - (one(FT) + alpha) * rym_lim)
            pf_s   = am_s * (alpha * alpha * rym_lim - FT(3) * f_s)
            fx_s   = alpha * rxm_i
            fz_s   = alpha * rzm_i
        end

        # --- North face (bm[i, j+1, k]) ---
        am_n = bm[i, j + 1, k]
        if j == Ny
            # North pole: no flux
            f_n  = zero(FT)
            pf_n = zero(FT)
            fx_n = zero(FT)
            fz_n = zero(FT)
        elseif am_n >= zero(FT)
            # Donor = cell j
            alpha  = m_i > eps(FT) * 100 ? am_n / m_i : zero(FT)
            f_n    = alpha * (rm_i + (one(FT) - alpha) * rym_lim)
            pf_n   = am_n * (alpha * alpha * rym_lim - FT(3) * f_n)
            fx_n   = alpha * rxm_i
            fz_n   = alpha * rzm_i
        else
            # Donor = cell j+1
            rm_jp  = rm[i, j+1, k];  m_jp = m[i, j+1, k]
            rym_jp = rym[i, j+1, k]
            if use_limiter; rym_jp = max(min(rym_jp, rm_jp), -rm_jp); end
            alpha  = m_jp > eps(FT) * 100 ? am_n / m_jp : zero(FT)
            f_n    = alpha * (rm_jp - (one(FT) + alpha) * rym_jp)
            pf_n   = am_n * (alpha * alpha * rym_jp - FT(3) * f_n)
            fx_n   = alpha * rxm[i, j+1, k]
            fz_n   = alpha * rzm[i, j+1, k]
        end

        # --- Updates ---
        rm_new_val = rm_i + f_s - f_n
        m_new_val  = m_i + am_s - am_n
        m_safe     = max(m_new_val, eps(FT) * 100)

        rym_new_val = (rym_i + (pf_s - pf_n)
                       - (am_s - am_n) * rym_i
                       + FT(3) * ((am_s + am_n) * rm_i - (f_s + f_n) * m_i)
                      ) / m_safe

        rxm_new_val = rxm_i + fx_s - fx_n
        rzm_new_val = rzm_i + fz_s - fz_n

        if use_limiter
            rym_new_val = max(min(rym_new_val, rm_new_val), -rm_new_val)
        end

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = m_new_val
        rxm_new[i, j, k] = rxm_new_val
        rym_new[i, j, k] = rym_new_val
        rzm_new[i, j, k] = rzm_new_val
    end
end

# ---------------------------------------------------------------------------
# Z-direction kernel
# ---------------------------------------------------------------------------

@kernel function _prather_z_kernel!(
    rm_new, m_new, rxm_new, rym_new, rzm_new,
    @Const(rm), @Const(m), @Const(rxm), @Const(rym), @Const(rzm),
    @Const(cm), Nz, use_limiter
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rm)

        # For Z: rzm is the "along" slope, rxm/rym are cross-slopes
        rm_i  = rm[i, j, k];  m_i  = m[i, j, k]
        rzm_i = rzm[i, j, k]
        rxm_i = rxm[i, j, k]
        rym_i = rym[i, j, k]

        # Apply limiter on along-slope
        rzm_lim = use_limiter ? max(min(rzm_i, rm_i), -rm_i) : rzm_i

        # --- Top face (cm[i, j, k]) ---
        cm_t = cm[i, j, k]
        if k == 1
            # TOA: no flux (cm[i,j,1] = 0 by construction)
            f_t  = zero(FT)
            pf_t = zero(FT)
            fx_t = zero(FT)
            fy_t = zero(FT)
        elseif cm_t >= zero(FT)
            # Donor = cell k-1 (downward flux = positive cm means mass moves down)
            rm_km  = rm[i, j, k-1];  m_km = m[i, j, k-1]
            rzm_km = rzm[i, j, k-1]
            if use_limiter; rzm_km = max(min(rzm_km, rm_km), -rm_km); end
            alpha  = m_km > eps(FT) * 100 ? cm_t / m_km : zero(FT)
            f_t    = alpha * (rm_km + (one(FT) - alpha) * rzm_km)
            pf_t   = cm_t * (alpha * alpha * rzm_km - FT(3) * f_t)
            fx_t   = alpha * rxm[i, j, k-1]
            fy_t   = alpha * rym[i, j, k-1]
        else
            # Donor = cell k
            alpha  = m_i > eps(FT) * 100 ? cm_t / m_i : zero(FT)
            f_t    = alpha * (rm_i - (one(FT) + alpha) * rzm_lim)
            pf_t   = cm_t * (alpha * alpha * rzm_lim - FT(3) * f_t)
            fx_t   = alpha * rxm_i
            fy_t   = alpha * rym_i
        end

        # --- Bottom face (cm[i, j, k+1]) ---
        cm_b = cm[i, j, k + 1]
        if k == Nz
            # Surface: no flux (cm[i,j,Nz+1] = 0 by construction)
            f_b  = zero(FT)
            pf_b = zero(FT)
            fx_b = zero(FT)
            fy_b = zero(FT)
        elseif cm_b >= zero(FT)
            # Donor = cell k
            alpha  = m_i > eps(FT) * 100 ? cm_b / m_i : zero(FT)
            f_b    = alpha * (rm_i + (one(FT) - alpha) * rzm_lim)
            pf_b   = cm_b * (alpha * alpha * rzm_lim - FT(3) * f_b)
            fx_b   = alpha * rxm_i
            fy_b   = alpha * rym_i
        else
            # Donor = cell k+1
            rm_kp  = rm[i, j, k+1];  m_kp = m[i, j, k+1]
            rzm_kp = rzm[i, j, k+1]
            if use_limiter; rzm_kp = max(min(rzm_kp, rm_kp), -rm_kp); end
            alpha  = m_kp > eps(FT) * 100 ? cm_b / m_kp : zero(FT)
            f_b    = alpha * (rm_kp - (one(FT) + alpha) * rzm_kp)
            pf_b   = cm_b * (alpha * alpha * rzm_kp - FT(3) * f_b)
            fx_b   = alpha * rxm[i, j, k+1]
            fy_b   = alpha * rym[i, j, k+1]
        end

        # --- Updates ---
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

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = m_new_val
        rxm_new[i, j, k] = rxm_new_val
        rym_new[i, j, k] = rym_new_val
        rzm_new[i, j, k] = rzm_new_val
    end
end

# ---------------------------------------------------------------------------
# Directional advection functions (GPU launch + copyback)
# ---------------------------------------------------------------------------

function _advect_x_prather!(rm, m, am, pw::PratherWorkspace{FT}, Nx, use_limiter) where FT
    backend = get_backend(rm)
    k! = _prather_x_kernel!(backend, 256)
    k!(pw.rm_buf, pw.m_buf, pw.rxm_buf, pw.rym_buf, pw.rzm_buf,
       rm, m, pw.rxm, pw.rym, pw.rzm, am, Int32(Nx), use_limiter;
       ndrange=size(rm))
    synchronize(backend)
    copyto!(rm, pw.rm_buf)
    copyto!(m, pw.m_buf)
    copyto!(pw.rxm, pw.rxm_buf)
    copyto!(pw.rym, pw.rym_buf)
    copyto!(pw.rzm, pw.rzm_buf)
end

function _advect_y_prather!(rm, m, bm, pw::PratherWorkspace{FT}, Ny, use_limiter) where FT
    backend = get_backend(rm)
    k! = _prather_y_kernel!(backend, 256)
    k!(pw.rm_buf, pw.m_buf, pw.rxm_buf, pw.rym_buf, pw.rzm_buf,
       rm, m, pw.rxm, pw.rym, pw.rzm, bm, Int32(Ny), use_limiter;
       ndrange=size(rm))
    synchronize(backend)
    copyto!(rm, pw.rm_buf)
    copyto!(m, pw.m_buf)
    copyto!(pw.rxm, pw.rxm_buf)
    copyto!(pw.rym, pw.rym_buf)
    copyto!(pw.rzm, pw.rzm_buf)
end

function _advect_z_prather!(rm, m, cm, pw::PratherWorkspace{FT}, Nz, use_limiter) where FT
    backend = get_backend(rm)
    k! = _prather_z_kernel!(backend, 256)
    k!(pw.rm_buf, pw.m_buf, pw.rxm_buf, pw.rym_buf, pw.rzm_buf,
       rm, m, pw.rxm, pw.rym, pw.rzm, cm, Int32(Nz), use_limiter;
       ndrange=size(rm))
    synchronize(backend)
    copyto!(rm, pw.rm_buf)
    copyto!(m, pw.m_buf)
    copyto!(pw.rxm, pw.rxm_buf)
    copyto!(pw.rym, pw.rym_buf)
    copyto!(pw.rzm, pw.rzm_buf)
end

# ---------------------------------------------------------------------------
# Strang splitting: X → Y → Z → Z → Y → X
# ---------------------------------------------------------------------------

"""
    strang_split_prather!(tracers, m, am, bm, cm, grid, pw_dict, use_limiter)

Full Strang-split advection using Prather (1986) prognostic slopes.
`pw_dict` is a NamedTuple of `PratherWorkspace` keyed by tracer name.
Tracers are tracer mass `rm = c × m` (TM5-style prognostic variable).
"""
function strang_split_prather!(tracers::NamedTuple,
                                m::AbstractArray{FT,3},
                                am, bm, cm,
                                grid::LatitudeLongitudeGrid,
                                pw_dict::NamedTuple,
                                use_limiter::Bool) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    n_tr = length(tracers)
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (idx, (name, rm_tracer)) in enumerate(pairs(tracers))
        if idx > 1
            copyto!(m, m_save)
        end

        pw = pw_dict[name]

        rm = pw.rm_buf
        copyto!(rm, rm_tracer)

        # Strang splitting: X → Y → Z → Z → Y → X
        _advect_x_prather!(rm, m, am, pw, Nx, use_limiter)
        _advect_y_prather!(rm, m, bm, pw, Ny, use_limiter)
        _advect_z_prather!(rm, m, cm, pw, Nz, use_limiter)
        _advect_z_prather!(rm, m, cm, pw, Nz, use_limiter)
        _advect_y_prather!(rm, m, bm, pw, Ny, use_limiter)
        _advect_x_prather!(rm, m, am, pw, Nx, use_limiter)

        copyto!(rm_tracer, rm)
    end
    return nothing
end

"""
    allocate_prather_workspaces(tracers, m) → NamedTuple of PratherWorkspace

Allocate one PratherWorkspace per tracer, keyed by tracer name.
Call once before the time loop.
"""
function allocate_prather_workspaces(tracers::NamedTuple, m::AbstractArray{FT,3}) where FT
    names = keys(tracers)
    workspaces = Tuple(allocate_prather_workspace(m) for _ in names)
    return NamedTuple{names}(workspaces)
end
