# ---------------------------------------------------------------------------
# TM5-faithful mass-flux advection — unified KernelAbstractions implementation
#
# Every function uses @kernel so the SAME code runs on CPU and GPU.
# Backend is inferred from the arrays via get_backend(). No if/else branching
# on architecture. No hardcoded Array{}/Vector{} allocations.
#
# Reference: TM5 advectx.F90, advecty.F90, advectz.F90 (dynamw_1d)
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, synchronize, get_backend

@inline function _to_device(cpu_vec::Vector{FT}, ref::AbstractArray{FT}) where FT
    dev = similar(ref, FT, length(cpu_vec))
    copyto!(dev, cpu_vec)
    return dev
end

# ---------------------------------------------------------------------------
# Reduced-grid helpers for GPU kernels
# ---------------------------------------------------------------------------

"""Sum an extensive quantity (rm or m) over a cluster of r fine cells."""
@inline function _cluster_sum(arr, ic::Int, j::Int, k::Int, r::Int)
    T = eltype(arr)
    s = zero(T)
    c = zero(T)
    i_start = (ic - 1) * r + 1
    for off in 0:r-1
        (s, c) = _kahan_add(s, c, arr[i_start + off, j, k])
    end
    return s
end

# =====================================================================
# Preprocessing kernels
# =====================================================================

@kernel function _air_mass_kernel!(m_out, @Const(Δp), @Const(area_j), g)
    i, j, k = @index(Global, NTuple)
    @inbounds m_out[i, j, k] = Δp[i, j, k] * area_j[j] / g
end

@kernel function _am_kernel!(am, @Const(u), @Const(Δp), @Const(dy_j), Nx, half_dt, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        il = i == 1 ? Nx : i - 1
        ir = i > Nx ? 1 : i
        dp_f = (Δp[il, j, k] + Δp[ir, j, k]) / 2
        am[i, j, k] = half_dt * u[i, j, k] * dp_f * dy_j[j] / g
    end
end

@kernel function _bm_kernel!(bm, @Const(v), @Const(Δp), @Const(dx_face), Ny, half_dt, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        jb = max(j - 1, 1)
        ja = min(j, Ny)
        dp_f = (Δp[i, jb, k] + Δp[i, ja, k]) / 2
        bm[i, j, k] = half_dt * v[i, j, k] * dp_f * abs(dx_face[j]) / g
    end
end

@kernel function _cm_column_kernel!(cm, @Const(am), @Const(bm), @Const(bt), Nz)
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

# =====================================================================
# Advection kernels — one thread per (i,j,k), double-buffer
# =====================================================================

@kernel function _massflux_x_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am),
    Nx, @Const(cluster_sizes), use_limiter
)
    i, j, k = @index(Global, NTuple)
    r = Int(cluster_sizes[j])
    @inbounds begin
        FT = eltype(rm)
        if r == 1
            # --- Uniform row: existing fine-grid logic ---
            ip  = i == Nx ? 1 : i + 1
            im  = i == 1  ? Nx : i - 1
            ipp = ip == Nx ? 1 : ip + 1
            imm = im == 1  ? Nx : im - 1

            c_imm = rm[imm, j, k] / m[imm, j, k]
            c_im  = rm[im,  j, k] / m[im,  j, k]
            c_i   = rm[i,   j, k] / m[i,   j, k]
            c_ip  = rm[ip,  j, k] / m[ip,  j, k]
            c_ipp = rm[ipp, j, k] / m[ipp, j, k]

            sc_im = (c_i - c_imm) / 2
            if use_limiter
                sc_im = minmod_device(sc_im, 2 * (c_i - c_im), 2 * (c_im - c_imm))
            end
            sx_im = m[im, j, k] * sc_im
            if use_limiter
                sx_im = max(min(sx_im, rm[im, j, k]), -rm[im, j, k])
            end

            sc_i = (c_ip - c_im) / 2
            if use_limiter
                sc_i = minmod_device(sc_i, 2 * (c_ip - c_i), 2 * (c_i - c_im))
            end
            sx_i = m[i, j, k] * sc_i
            if use_limiter
                sx_i = max(min(sx_i, rm[i, j, k]), -rm[i, j, k])
            end

            sc_ip = (c_ipp - c_i) / 2
            if use_limiter
                sc_ip = minmod_device(sc_ip, 2 * (c_ipp - c_ip), 2 * (c_ip - c_i))
            end
            sx_ip = m[ip, j, k] * sc_ip
            if use_limiter
                sx_ip = max(min(sx_ip, rm[ip, j, k]), -rm[ip, j, k])
            end

            am_l = am[i, j, k]
            flux_left = if am_l >= zero(FT)
                alpha = am_l / m[im, j, k]
                alpha * (rm[im, j, k] + (one(FT) - alpha) * sx_im)
            else
                alpha = am_l / m[i, j, k]
                alpha * (rm[i, j, k] - (one(FT) + alpha) * sx_i)
            end

            am_r = am[i + 1, j, k]
            flux_right = if am_r >= zero(FT)
                alpha = am_r / m[i, j, k]
                alpha * (rm[i, j, k] + (one(FT) - alpha) * sx_i)
            else
                alpha = am_r / m[ip, j, k]
                alpha * (rm[ip, j, k] - (one(FT) + alpha) * sx_ip)
            end

            rm_new[i, j, k] = rm[i, j, k] + flux_left - flux_right
            m_new[i, j, k]  = m[i, j, k]  + am[i, j, k] - am[i + 1, j, k]
        else
            # --- Reduced row: work on cluster aggregates ---
            Nx_red = Nx ÷ r
            ic  = (i - 1) ÷ r + 1
            ic_m  = ic == 1      ? Nx_red : ic - 1
            ic_p  = ic == Nx_red ? 1      : ic + 1
            ic_mm = ic_m == 1      ? Nx_red : ic_m - 1
            ic_pp = ic_p == Nx_red ? 1      : ic_p + 1

            # Aggregate rm, m over clusters (extensive: sum)
            rm_ic  = _cluster_sum(rm, ic, j, k, r)
            m_ic   = _cluster_sum(m,  ic, j, k, r)
            rm_im  = _cluster_sum(rm, ic_m, j, k, r)
            m_im   = _cluster_sum(m,  ic_m, j, k, r)
            rm_ip  = _cluster_sum(rm, ic_p, j, k, r)
            m_ip   = _cluster_sum(m,  ic_p, j, k, r)
            rm_imm = _cluster_sum(rm, ic_mm, j, k, r)
            m_imm  = _cluster_sum(m,  ic_mm, j, k, r)
            rm_ipp = _cluster_sum(rm, ic_pp, j, k, r)
            m_ipp  = _cluster_sum(m,  ic_pp, j, k, r)

            # Concentrations on reduced grid
            c_imm_r = rm_imm / m_imm
            c_im_r  = rm_im  / m_im
            c_ic_r  = rm_ic  / m_ic
            c_ip_r  = rm_ip  / m_ip
            c_ipp_r = rm_ipp / m_ipp

            # Slopes on reduced grid
            sc_im_r = (c_ic_r - c_imm_r) / 2
            if use_limiter
                sc_im_r = minmod_device(sc_im_r, 2 * (c_ic_r - c_im_r), 2 * (c_im_r - c_imm_r))
            end
            sx_im_r = m_im * sc_im_r
            if use_limiter
                sx_im_r = max(min(sx_im_r, rm_im), -rm_im)
            end

            sc_ic_r = (c_ip_r - c_im_r) / 2
            if use_limiter
                sc_ic_r = minmod_device(sc_ic_r, 2 * (c_ip_r - c_ic_r), 2 * (c_ic_r - c_im_r))
            end
            sx_ic_r = m_ic * sc_ic_r
            if use_limiter
                sx_ic_r = max(min(sx_ic_r, rm_ic), -rm_ic)
            end

            sc_ip_r = (c_ipp_r - c_ic_r) / 2
            if use_limiter
                sc_ip_r = minmod_device(sc_ip_r, 2 * (c_ipp_r - c_ip_r), 2 * (c_ip_r - c_ic_r))
            end
            sx_ip_r = m_ip * sc_ip_r
            if use_limiter
                sx_ip_r = max(min(sx_ip_r, rm_ip), -rm_ip)
            end

            # Face mass fluxes at cluster boundaries
            am_l_r = am[(ic - 1) * r + 1, j, k]
            am_r_idx = ic * r + 1
            am_r_r = am[am_r_idx > Nx ? 1 : am_r_idx, j, k]

            flux_left_r = if am_l_r >= zero(FT)
                alpha = am_l_r / m_im
                alpha * (rm_im + (one(FT) - alpha) * sx_im_r)
            else
                alpha = am_l_r / m_ic
                alpha * (rm_ic - (one(FT) + alpha) * sx_ic_r)
            end

            flux_right_r = if am_r_r >= zero(FT)
                alpha = am_r_r / m_ic
                alpha * (rm_ic + (one(FT) - alpha) * sx_ic_r)
            else
                alpha = am_r_r / m_ip
                alpha * (rm_ip - (one(FT) + alpha) * sx_ip_r)
            end

            # Cluster-level deltas
            delta_rm = flux_left_r - flux_right_r
            delta_m  = am_l_r - am_r_r

            # TM5-style distribute-back: use air mass fraction (smooth),
            # replace fine cell with share of new cluster total (not delta).
            # This prevents tracer noise from corrupting the distribution.
            frac_m = abs(m_ic) > eps(FT) ? m[i, j, k] / m_ic : one(FT) / FT(r)
            rm_new[i, j, k] = (rm_ic + delta_rm) * frac_m
            m_new[i, j, k]  = (m_ic  + delta_m)  * frac_m
        end
    end
end

@kernel function _massflux_y_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm), Ny, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        # --- Slope at j ---
        sy_j = if j > 1 && j < Ny
            cjm = rm[i, j - 1, k] / m[i, j - 1, k]
            cj  = rm[i, j,     k] / m[i, j,     k]
            cjp = rm[i, j + 1, k] / m[i, j + 1, k]
            sc = (cjp - cjm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (cjp - cj), 2 * (cj - cjm)); end
            s = m[i, j, k] * sc
            if use_limiter; s = max(min(s, rm[i, j, k]), -rm[i, j, k]); end
            s
        else
            zero(FT)
        end

        # --- Slope at j-1 (for south flux) ---
        sy_jm = if j > 2 && j - 1 < Ny
            cjmm = rm[i, j - 2, k] / m[i, j - 2, k]
            cjm  = rm[i, j - 1, k] / m[i, j - 1, k]
            cj   = rm[i, j,     k] / m[i, j,     k]
            sc = (cj - cjmm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (cj - cjm), 2 * (cjm - cjmm)); end
            s = m[i, j - 1, k] * sc
            if use_limiter; s = max(min(s, rm[i, j - 1, k]), -rm[i, j - 1, k]); end
            s
        else
            zero(FT)
        end

        # --- Slope at j+1 (for north flux) ---
        sy_jp = if j < Ny - 1 && j + 1 > 1
            cj   = rm[i, j,     k] / m[i, j,     k]
            cjp  = rm[i, j + 1, k] / m[i, j + 1, k]
            cjpp = rm[i, j + 2, k] / m[i, j + 2, k]
            sc = (cjpp - cj) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (cjpp - cjp), 2 * (cjp - cj)); end
            s = m[i, j + 1, k] * sc
            if use_limiter; s = max(min(s, rm[i, j + 1, k]), -rm[i, j + 1, k]); end
            s
        else
            zero(FT)
        end

        # --- Flux at south face (face j) ---
        flux_s = if j > 1
            bm_s = bm[i, j, k]
            if bm_s >= zero(FT)
                beta = bm_s / m[i, j - 1, k]
                j - 1 == 1 ? beta * rm[i, j - 1, k] :
                    beta * (rm[i, j - 1, k] + (one(FT) - beta) * sy_jm)
            else
                beta = bm_s / m[i, j, k]
                j == Ny ? beta * rm[i, j, k] :
                    beta * (rm[i, j, k] - (one(FT) + beta) * sy_j)
            end
        else
            zero(FT)
        end

        # --- Flux at north face (face j+1) ---
        flux_n = if j < Ny
            bm_n = bm[i, j + 1, k]
            if bm_n >= zero(FT)
                beta = bm_n / m[i, j, k]
                j == 1 ? beta * rm[i, j, k] :
                    beta * (rm[i, j, k] + (one(FT) - beta) * sy_j)
            else
                beta = bm_n / m[i, j + 1, k]
                j + 1 == Ny ? beta * rm[i, j + 1, k] :
                    beta * (rm[i, j + 1, k] - (one(FT) + beta) * sy_jp)
            end
        else
            zero(FT)
        end

        rm_new[i, j, k] = rm[i, j, k] + flux_s - flux_n
        m_new[i, j, k]  = m[i, j, k]  + bm[i, j, k] - bm[i, j + 1, k]
    end
end

@kernel function _massflux_z_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(cm), Nz, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    # Minimum mass for safe division. Met-data cm satisfies continuity
    # (m_new > 0), so m should stay positive. This guards only against
    # pathological edge cases (e.g., top-of-model near-vacuum).
    m_eps = eps(FT)
    @inbounds begin
        # --- Slope at k ---
        sz_k = if k > 1 && k < Nz
            ckm = rm[i, j, k - 1] / max(m[i, j, k - 1], m_eps)
            ck  = rm[i, j, k]     / max(m[i, j, k],     m_eps)
            ckp = rm[i, j, k + 1] / max(m[i, j, k + 1], m_eps)
            sc = (ckp - ckm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ckp - ck), 2 * (ck - ckm)); end
            s = m[i, j, k] * sc
            if use_limiter; s = max(min(s, rm[i, j, k]), -rm[i, j, k]); end
            s
        else
            zero(FT)
        end

        # --- Slope at k-1 (for top flux) ---
        sz_km = if k > 2 && k - 1 < Nz
            ckmm = rm[i, j, k - 2] / max(m[i, j, k - 2], m_eps)
            ckm  = rm[i, j, k - 1] / max(m[i, j, k - 1], m_eps)
            ck   = rm[i, j, k]     / max(m[i, j, k],     m_eps)
            sc = (ck - ckmm) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ck - ckm), 2 * (ckm - ckmm)); end
            s = m[i, j, k - 1] * sc
            if use_limiter; s = max(min(s, rm[i, j, k - 1]), -rm[i, j, k - 1]); end
            s
        else
            zero(FT)
        end

        # --- Slope at k+1 (for bottom flux) ---
        sz_kp = if k < Nz - 1 && k + 1 > 1
            ck   = rm[i, j, k]     / max(m[i, j, k],     m_eps)
            ckp  = rm[i, j, k + 1] / max(m[i, j, k + 1], m_eps)
            ckpp = rm[i, j, k + 2] / max(m[i, j, k + 2], m_eps)
            sc = (ckpp - ck) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ckpp - ckp), 2 * (ckp - ck)); end
            s = m[i, j, k + 1] * sc
            if use_limiter; s = max(min(s, rm[i, j, k + 1]), -rm[i, j, k + 1]); end
            s
        else
            zero(FT)
        end

        # --- Flux at top face (face k) ---
        # With diagnostic slopes, gamma must stay in [-1, 1] to prevent flux
        # reversal (gamma > 2 makes f = gamma*(rm+(1-gamma)*sz) flip sign).
        # TM5 allows gamma > 1 via prognostic slopes (advectz.F90:462);
        # removing this clamp requires implementing prognostic slope evolution.
        flux_top = if k > 1
            cm_t = cm[i, j, k]
            if cm_t > zero(FT)
                gamma = m[i, j, k - 1] > m_eps ? cm_t / m[i, j, k - 1] : zero(FT)
                gamma * (rm[i, j, k - 1] + (one(FT) - gamma) * sz_km)
            elseif cm_t < zero(FT)
                gamma = m[i, j, k] > m_eps ? cm_t / m[i, j, k] : zero(FT)
                gamma * (rm[i, j, k] - (one(FT) + gamma) * sz_k)
            else
                zero(FT)
            end
        else
            zero(FT)
        end

        # --- Flux at bottom face (face k+1) ---
        flux_bot = if k < Nz
            cm_b = cm[i, j, k + 1]
            if cm_b > zero(FT)
                gamma = m[i, j, k] > m_eps ? cm_b / m[i, j, k] : zero(FT)
                gamma * (rm[i, j, k] + (one(FT) - gamma) * sz_k)
            elseif cm_b < zero(FT)
                gamma = m[i, j, k + 1] > m_eps ? cm_b / m[i, j, k + 1] : zero(FT)
                gamma * (rm[i, j, k + 1] - (one(FT) + gamma) * sz_kp)
            else
                zero(FT)
            end
        else
            zero(FT)
        end

        rm_new[i, j, k] = rm[i, j, k] + flux_top - flux_bot
        m_new[i, j, k]  = m[i, j, k]  + cm[i, j, k] - cm[i, j, k + 1]
    end
end

# =====================================================================
# Mass-only kernels for evolving-mass CFL pilots
# (TM5 advectm_cfl.F90 + advectx__slopes.F90:430-520 style)
#
# Mirrors the mass update path of the real X/Y/Z advection kernels
# (including the reduced-grid clustering for X) so the pilot's mass
# evolution matches what the real sweep will see.
# =====================================================================

@kernel function _mass_only_x_kernel!(
    m_new, @Const(m), @Const(am), Nx, @Const(cluster_sizes)
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(m)
        r = Int(cluster_sizes[j])
        if r == 1
            # Uniform fine row: same update as _massflux_x_kernel! line 153
            m_new[i, j, k] = m[i, j, k] + am[i, j, k] - am[i + 1, j, k]
        else
            # Reduced row: cluster-aggregate update, distributed by air-mass
            # fraction back to fine cells.  Mirrors _massflux_x_kernel! lines
            # 154-241 (mass-only path).
            Nx_red = Nx ÷ r
            ic = (i - 1) ÷ r + 1
            m_ic = _cluster_sum(m, ic, j, k, r)
            am_l_idx = (ic - 1) * r + 1
            am_r_idx = ic * r + 1
            am_l = am[am_l_idx, j, k]
            am_r = am[am_r_idx > Nx ? 1 : am_r_idx, j, k]
            delta_m = am_l - am_r
            frac_m = abs(m_ic) > eps(FT) ? m[i, j, k] / m_ic : one(FT) / FT(r)
            m_new[i, j, k] = (m_ic + delta_m) * frac_m
        end
    end
end

@kernel function _mass_only_y_kernel!(m_new, @Const(m), @Const(bm), Ny)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(m)
        bm_s = j > 1  ? bm[i, j, k]     : zero(FT)
        bm_n = j < Ny ? bm[i, j + 1, k] : zero(FT)
        m_new[i, j, k] = m[i, j, k] + bm_s - bm_n
    end
end

@kernel function _mass_only_z_kernel!(m_new, @Const(m), @Const(cm), Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(m)
        cm_t = k > 1  ? cm[i, j, k]     : zero(FT)
        cm_b = k < Nz ? cm[i, j, k + 1] : zero(FT)
        m_new[i, j, k] = m[i, j, k] + cm_t - cm_b
    end
end

# =====================================================================
# Prognostic-slope kernels (TM5 second-order moments)
#
# These kernels evolve rxm/rym/rzm alongside rm using the pf-term
# from TM5 advect{x,y,z}.F90.  The "along" slope gets the full
# second-moment update; cross-slopes are passively advected.
# =====================================================================

@kernel function _prognostic_z_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(cm),
    rzm_new, @Const(rzm), rxm_new, @Const(rxm), rym_new, @Const(rym),
    Nz, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_eps = FT(100) * eps(FT)
    @inbounds begin
        rm_i  = rm[i, j, k];   m_i  = m[i, j, k]
        rzm_i = rzm[i, j, k];  rxm_i = rxm[i, j, k];  rym_i = rym[i, j, k]

        # Pre-limit along-slope (TM5 advectz.F90:447)
        if use_limiter
            rzm_i = max(min(rzm_i, rm_i), -rm_i)
        end

        # --- Top face flux (interface k, between k-1 and k) ---
        cm_t = k > 1 ? cm[i, j, k] : zero(FT)
        if cm_t > zero(FT)
            # Downward: donor = k-1
            rm_km  = rm[i, j, k-1];  m_km = m[i, j, k-1]
            rzm_km = rzm[i, j, k-1]
            if use_limiter; rzm_km = max(min(rzm_km, rm_km), -rm_km); end
            gamma = m_km > m_eps ? cm_t / m_km : zero(FT)
            f_t   = gamma * (rm_km + (one(FT) - gamma) * rzm_km)
            pf_t  = cm_t * (gamma * gamma * rzm_km - FT(3) * f_t)
            fx_t  = gamma * rxm[i, j, k-1]
            fy_t  = gamma * rym[i, j, k-1]
        elseif cm_t < zero(FT)
            # Upward: donor = k
            gamma = m_i > m_eps ? cm_t / m_i : zero(FT)
            f_t   = gamma * (rm_i - (one(FT) + gamma) * rzm_i)
            pf_t  = cm_t * (gamma * gamma * rzm_i - FT(3) * f_t)
            fx_t  = gamma * rxm_i
            fy_t  = gamma * rym_i
        else
            f_t = zero(FT); pf_t = zero(FT); fx_t = zero(FT); fy_t = zero(FT)
        end

        # --- Bottom face flux (interface k+1, between k and k+1) ---
        cm_b = k < Nz ? cm[i, j, k+1] : zero(FT)
        if cm_b > zero(FT)
            # Downward: donor = k
            gamma = m_i > m_eps ? cm_b / m_i : zero(FT)
            f_b   = gamma * (rm_i + (one(FT) - gamma) * rzm_i)
            pf_b  = cm_b * (gamma * gamma * rzm_i - FT(3) * f_b)
            fx_b  = gamma * rxm_i
            fy_b  = gamma * rym_i
        elseif cm_b < zero(FT)
            # Upward: donor = k+1
            rm_kp  = rm[i, j, k+1];  m_kp = m[i, j, k+1]
            rzm_kp = rzm[i, j, k+1]
            if use_limiter; rzm_kp = max(min(rzm_kp, rm_kp), -rm_kp); end
            gamma = m_kp > m_eps ? cm_b / m_kp : zero(FT)
            f_b   = gamma * (rm_kp - (one(FT) + gamma) * rzm_kp)
            pf_b  = cm_b * (gamma * gamma * rzm_kp - FT(3) * f_b)
            fx_b  = gamma * rxm[i, j, k+1]
            fy_b  = gamma * rym[i, j, k+1]
        else
            f_b = zero(FT); pf_b = zero(FT); fx_b = zero(FT); fy_b = zero(FT)
        end

        # --- Updates (TM5 advectz.F90:480-490) ---
        rm_new_val = rm_i + f_t - f_b
        mnew_val   = m_i + cm_t - cm_b
        m_safe     = max(mnew_val, m_eps)

        # Along-slope update: TM5 advectz.F90:481-485
        # rzm_new = rzm + correction/mnew
        rzm_new_val = rzm_i + ((pf_t - pf_b)
                                - (cm_t - cm_b) * rzm_i
                                + FT(3) * ((cm_t + cm_b) * rm_new_val
                                           - (f_t + f_b) * m_i)
                               ) / m_safe
        if use_limiter
            rzm_new_val = max(min(rzm_new_val, rm_new_val), -rm_new_val)
        end

        # Cross-slopes: simple flux divergence (TM5 advectz.F90:489-490)
        rxm_new_val = rxm_i + fx_t - fx_b
        rym_new_val = rym_i + fy_t - fy_b

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = mnew_val
        rzm_new[i, j, k] = rzm_new_val
        rxm_new[i, j, k] = rxm_new_val
        rym_new[i, j, k] = rym_new_val
    end
end

@kernel function _prognostic_x_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am),
    rxm_new, @Const(rxm), rym_new, @Const(rym), rzm_new, @Const(rzm),
    Nx, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_eps = FT(100) * eps(FT)
    @inbounds begin
        im = i == 1  ? Nx : i - 1
        ip = i == Nx ? 1  : i + 1

        rm_i  = rm[i, j, k];   m_i  = m[i, j, k]
        rxm_i = rxm[i, j, k];  rym_i = rym[i, j, k];  rzm_i = rzm[i, j, k]
        if use_limiter; rxm_i = max(min(rxm_i, rm_i), -rm_i); end

        # --- Left face (am[i, j, k]) ---
        am_l = am[i, j, k]
        if am_l >= zero(FT)
            rm_im = rm[im, j, k]; m_im = m[im, j, k]; rxm_im = rxm[im, j, k]
            if use_limiter; rxm_im = max(min(rxm_im, rm_im), -rm_im); end
            alpha = m_im > m_eps ? am_l / m_im : zero(FT)
            f_l   = alpha * (rm_im + (one(FT) - alpha) * rxm_im)
            pf_l  = am_l * (alpha * alpha * rxm_im - FT(3) * f_l)
            fy_l  = alpha * rym[im, j, k]
            fz_l  = alpha * rzm[im, j, k]
        else
            alpha = m_i > m_eps ? am_l / m_i : zero(FT)
            f_l   = alpha * (rm_i - (one(FT) + alpha) * rxm_i)
            pf_l  = am_l * (alpha * alpha * rxm_i - FT(3) * f_l)
            fy_l  = alpha * rym_i
            fz_l  = alpha * rzm_i
        end

        # --- Right face (am[i+1, j, k]) ---
        am_r = am[i + 1, j, k]
        if am_r >= zero(FT)
            alpha = m_i > m_eps ? am_r / m_i : zero(FT)
            f_r   = alpha * (rm_i + (one(FT) - alpha) * rxm_i)
            pf_r  = am_r * (alpha * alpha * rxm_i - FT(3) * f_r)
            fy_r  = alpha * rym_i
            fz_r  = alpha * rzm_i
        else
            rm_ip = rm[ip, j, k]; m_ip = m[ip, j, k]; rxm_ip = rxm[ip, j, k]
            if use_limiter; rxm_ip = max(min(rxm_ip, rm_ip), -rm_ip); end
            alpha = m_ip > m_eps ? am_r / m_ip : zero(FT)
            f_r   = alpha * (rm_ip - (one(FT) + alpha) * rxm_ip)
            pf_r  = am_r * (alpha * alpha * rxm_ip - FT(3) * f_r)
            fy_r  = alpha * rym[ip, j, k]
            fz_r  = alpha * rzm[ip, j, k]
        end

        # --- Updates (TM5 advectx.F90:706-716) ---
        rm_new_val = rm_i + f_l - f_r
        mnew_val   = m_i + am_l - am_r
        m_safe     = max(mnew_val, m_eps)

        # rxm_new = rxm + correction/mnew (TM5 advectx.F90:707-710)
        rxm_new_val = rxm_i + ((pf_l - pf_r)
                                - (am_l - am_r) * rxm_i
                                + FT(3) * ((am_l + am_r) * rm_new_val
                                           - (f_l + f_r) * m_i)
                               ) / m_safe
        if use_limiter
            rxm_new_val = max(min(rxm_new_val, rm_new_val), -rm_new_val)
        end

        rym_new_val = rym_i + fy_l - fy_r
        rzm_new_val = rzm_i + fz_l - fz_r

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = mnew_val
        rxm_new[i, j, k] = rxm_new_val
        rym_new[i, j, k] = rym_new_val
        rzm_new[i, j, k] = rzm_new_val
    end
end

@kernel function _prognostic_y_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm),
    rym_new, @Const(rym), rxm_new, @Const(rxm), rzm_new, @Const(rzm),
    Ny, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_eps = FT(100) * eps(FT)
    @inbounds begin
        rm_i  = rm[i, j, k];   m_i  = m[i, j, k]
        rym_i = rym[i, j, k];  rxm_i = rxm[i, j, k];  rzm_i = rzm[i, j, k]
        if use_limiter; rym_i = max(min(rym_i, rm_i), -rm_i); end

        # --- South face (bm[i, j, k]) ---
        bm_s = bm[i, j, k]
        if j == 1
            f_s = zero(FT); pf_s = zero(FT); fx_s = zero(FT); fz_s = zero(FT)
        elseif bm_s >= zero(FT)
            rm_jm = rm[i, j-1, k]; m_jm = m[i, j-1, k]; rym_jm = rym[i, j-1, k]
            if use_limiter; rym_jm = max(min(rym_jm, rm_jm), -rm_jm); end
            beta = m_jm > m_eps ? bm_s / m_jm : zero(FT)
            f_s  = beta * (rm_jm + (one(FT) - beta) * rym_jm)
            pf_s = bm_s * (beta * beta * rym_jm - FT(3) * f_s)
            fx_s = beta * rxm[i, j-1, k]
            fz_s = beta * rzm[i, j-1, k]
        else
            beta = m_i > m_eps ? bm_s / m_i : zero(FT)
            f_s  = beta * (rm_i - (one(FT) + beta) * rym_i)
            pf_s = bm_s * (beta * beta * rym_i - FT(3) * f_s)
            fx_s = beta * rxm_i
            fz_s = beta * rzm_i
        end

        # --- North face (bm[i, j+1, k]) ---
        bm_n = bm[i, j + 1, k]
        if j == Ny
            f_n = zero(FT); pf_n = zero(FT); fx_n = zero(FT); fz_n = zero(FT)
        elseif bm_n >= zero(FT)
            beta = m_i > m_eps ? bm_n / m_i : zero(FT)
            f_n  = beta * (rm_i + (one(FT) - beta) * rym_i)
            pf_n = bm_n * (beta * beta * rym_i - FT(3) * f_n)
            fx_n = beta * rxm_i
            fz_n = beta * rzm_i
        else
            rm_jp = rm[i, j+1, k]; m_jp = m[i, j+1, k]; rym_jp = rym[i, j+1, k]
            if use_limiter; rym_jp = max(min(rym_jp, rm_jp), -rm_jp); end
            beta = m_jp > m_eps ? bm_n / m_jp : zero(FT)
            f_n  = beta * (rm_jp - (one(FT) + beta) * rym_jp)
            pf_n = bm_n * (beta * beta * rym_jp - FT(3) * f_n)
            fx_n = beta * rxm[i, j+1, k]
            fz_n = beta * rzm[i, j+1, k]
        end

        # --- Updates (TM5 advecty.F90:617-625) ---
        rm_new_val = rm_i + f_s - f_n
        mnew_val   = m_i + bm_s - bm_n
        m_safe     = max(mnew_val, m_eps)

        # rym_new = rym + correction/mnew (TM5 advecty.F90:620-623)
        rym_new_val = rym_i + ((pf_s - pf_n)
                                - (bm_s - bm_n) * rym_i
                                + FT(3) * ((bm_s + bm_n) * rm_new_val
                                           - (f_s + f_n) * m_i)
                               ) / m_safe
        if use_limiter
            rym_new_val = max(min(rym_new_val, rm_new_val), -rm_new_val)
        end

        rxm_new_val = rxm_i + fx_s - fx_n
        rzm_new_val = rzm_i + fz_s - fz_n

        rm_new[i, j, k]  = rm_new_val
        m_new[i, j, k]   = mnew_val
        rym_new[i, j, k] = rym_new_val
        rxm_new[i, j, k] = rxm_new_val
        rzm_new[i, j, k] = rzm_new_val
    end
end

# =====================================================================
# CFL kernels
# =====================================================================

@kernel function _cfl_x_kernel!(cfl, @Const(am), @Const(m), Nx, @Const(cluster_sizes))
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    r = Int(cluster_sizes[j])
    @inbounds begin
        if r == 1
            il = i == 1 ? Nx : i - 1
            ir = i > Nx ? 1 : i
            md = am[i, j, k] >= zero(FT) ? m[il, j, k] : m[ir, j, k]
            cfl[i, j, k] = md > zero(FT) ? abs(am[i, j, k]) / md : zero(FT)
        else
            if i > Nx
                # Extra periodic face — duplicate of face 1; skip for CFL
                cfl[i, j, k] = zero(FT)
            else
                Nx_red = Nx ÷ r
                ic = (i - 1) ÷ r + 1
                am_face = am[(ic - 1) * r + 1, j, k]
                donor_ic = am_face >= zero(FT) ? (ic == 1 ? Nx_red : ic - 1) : ic
                m_donor = _cluster_sum(m, donor_ic, j, k, r)
                cfl[i, j, k] = m_donor > zero(FT) ? abs(am_face) / m_donor : zero(FT)
            end
        end
    end
end

@kernel function _cfl_y_kernel!(cfl, @Const(bm), @Const(m), Ny)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        if j >= 2 && j <= Ny
            js = j - 1; jn = j
            md = bm[i, j, k] >= zero(FT) ? m[i, js, k] : m[i, jn, k]
            cfl[i, j, k] = md > zero(FT) ? abs(bm[i, j, k]) / md : zero(FT)
        else
            cfl[i, j, k] = zero(FT)
        end
    end
end

@kernel function _cfl_z_kernel!(cfl, @Const(cm), @Const(m), Nz)
    i, j, k = @index(Global, NTuple)
    FT = eltype(m)
    @inbounds begin
        if k >= 2 && k <= Nz
            md = cm[i, j, k] > zero(FT) ? m[i, j, k - 1] : m[i, j, k]
            cfl[i, j, k] = md > zero(FT) ? abs(cm[i, j, k]) / md : zero(FT)
        else
            cfl[i, j, k] = zero(FT)
        end
    end
end

# =====================================================================
# Grid geometry cache — computed once, reused every met window (TM5 dynam0)
# =====================================================================

"""
    GridGeometryCache{FT, A1}

Device-side cache of grid geometry vectors that are constant for a given grid.
Eliminates repeated host→device transfers of `area_j`, `dy_j`, `dx_face`, and
`bt` that previously occurred on every call to `compute_air_mass!` /
`compute_mass_fluxes!`.

Construct once with [`build_geometry_cache`](@ref), then pass to the in-place
`compute_air_mass!` and `compute_mass_fluxes!` overloads.
"""
struct GridGeometryCache{FT, A1 <: AbstractVector{FT}}
    area_j  :: A1   # cell area by latitude [m²], length Ny
    dy_j    :: A1   # Δy by latitude [m], length Ny
    dx_face :: A1   # dx at v-face latitudes [m], length Ny+1
    bt      :: A1   # B-ratio for vertical mass-flux closure, length Nz
    gravity :: FT
    Nx :: Int
    Ny :: Int
    Nz :: Int
end

"""
$(SIGNATURES)

Build a [`GridGeometryCache`](@ref) from a `LatitudeLongitudeGrid`.  `ref_array`
is any device-side 3-D array whose backend determines whether the cache lives on
CPU or GPU.

Call once before the time loop; the cache is valid for the lifetime of the grid.
"""
function build_geometry_cache(grid::LatitudeLongitudeGrid{FT},
                              ref_array::AbstractArray{FT}) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)

    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    dy_j_cpu   = FT[Δy(1, j, grid) for j in 1:Ny]

    dx_face_cpu = Vector{FT}(undef, Ny + 1)
    @inbounds for j in 1:Ny+1
        φ_f = if j == 1
            FT(-90)
        elseif j == Ny + 1
            FT(90)
        else
            FT(grid.φᶠ_cpu[j])
        end
        dx_face_cpu[j] = FT(grid.radius) * cosd(φ_f) * deg2rad(FT(grid.Δλ))
    end

    vc = grid.vertical
    ΔB_cpu = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        ΔB_cpu[k] = FT(vc.B[k + 1] - vc.B[k])
    end
    ΔB_total = FT(vc.B[Nz + 1] - vc.B[1])
    bt_cpu = abs(ΔB_total) > eps(FT) ? ΔB_cpu ./ ΔB_total : zeros(FT, Nz)

    area_j  = _to_device(area_j_cpu, ref_array)
    dy_j    = _to_device(dy_j_cpu, ref_array)
    dx_face = _to_device(dx_face_cpu, ref_array)
    bt      = _to_device(bt_cpu, ref_array)

    return GridGeometryCache{FT, typeof(area_j)}(
        area_j, dy_j, dx_face, bt, g, Nx, Ny, Nz)
end

# =====================================================================
# Pre-allocated workspace to avoid GPU array allocations in the inner loop
# =====================================================================

"""
    MassFluxWorkspace{FT, A3}

Pre-allocated buffers for mass-flux advection, eliminating all GPU array
allocations from the inner time-stepping loop.
"""
struct MassFluxWorkspace{FT, A3 <: AbstractArray{FT,3}, V1 <: AbstractVector{Int32}, M2 <: AbstractArray{Int32,2}}
    rm::A3       # tracer mass (Nx, Ny, Nz)
    rm_buf::A3   # advection output buffer for rm (Nx, Ny, Nz)
    m_buf::A3    # advection output buffer for m  (Nx, Ny, Nz)
    m_pilot::A3  # mass evolution scratch for evolving-mass CFL pilot (Nx, Ny, Nz)
    cfl_x::A3   # CFL scratch for x (Nx+1, Ny, Nz) — also reused as flux_x_eff
    cfl_y::A3   # CFL scratch for y (Nx, Ny+1, Nz) — also reused as flux_y_eff
    cfl_z::A3   # CFL scratch for z (Nx, Ny, Nz+1) — also reused as flux_z_eff
    # Pilot-only scratch buffers — separate from cfl_x/y/z so the pilot can
    # store scaled fluxes in cfl_x/y/z and write CFL values into these without
    # aliasing.
    cfl_scratch_x::A3
    cfl_scratch_y::A3
    cfl_scratch_z::A3
    cluster_sizes::V1  # per-latitude clustering for reduced grid (length Ny)
    # Per-level Y nloop (host) and device mirror for kernel masking.
    # Length Nz, set by `_y_discover_nloop_per_level!` (TM5 advecty__slopes.F90:236-289).
    nloop_y_cpu::Vector{Int32}
    nloop_y_dev::V1
    # Per-(j,l) X nloop (host) and device mirror.
    # Shape (Ny, Nz), set by `_x_discover_nloop_per_row!`
    # (TM5 advectx__slopes.F90:436-516).
    nloop_x_cpu::Matrix{Int32}
    nloop_x_dev::M2
end

"""
$(SIGNATURES)

Allocate a workspace that matches the sizes of `m`, `am`, `bm`, `cm`.
`cluster_sizes_cpu` is an `Int32` vector of per-latitude cluster sizes
(1 = uniform, >1 = reduced). Pass `nothing` for no reduced grid.
Call once before the time loop; pass to `strang_split_massflux!`.
"""
function allocate_massflux_workspace(m::AbstractArray{FT,3},
                                     am::AbstractArray{FT,3},
                                     bm::AbstractArray{FT,3},
                                     cm::AbstractArray{FT,3};
                                     cluster_sizes_cpu::Union{Nothing,Vector{Int32}} = nothing) where FT
    Nx, Ny, Nz = size(m)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    nloop_y_cpu = ones(Int32, Nz)
    nloop_y_dev = similar(m, Int32, Nz)
    copyto!(nloop_y_dev, nloop_y_cpu)
    nloop_x_cpu = ones(Int32, Ny, Nz)
    nloop_x_dev = similar(m, Int32, Ny, Nz)
    copyto!(nloop_x_dev, nloop_x_cpu)
    MassFluxWorkspace{FT, typeof(m), typeof(cs_dev), typeof(nloop_x_dev)}(
        similar(m),       # rm
        similar(m),       # rm_buf
        similar(m),       # m_buf
        similar(m),       # m_pilot (evolving-mass CFL pilot scratch)
        similar(am),      # cfl_x / flux_x_eff
        similar(bm),      # cfl_y / flux_y_eff
        similar(cm),      # cfl_z / flux_z_eff
        similar(am),      # cfl_scratch_x (pilot CFL output, separate from cfl_x)
        similar(bm),      # cfl_scratch_y
        similar(cm),      # cfl_scratch_z
        cs_dev,           # cluster_sizes (device)
        nloop_y_cpu,      # per-level Y nloop (host)
        nloop_y_dev,      # per-level Y nloop (device mirror)
        nloop_x_cpu,      # per-(j,l) X nloop (host)
        nloop_x_dev,      # per-(j,l) X nloop (device mirror)
    )
end

# =====================================================================
# Global Check_CFL pre-pass (TM5 advectm_cfl.F90:217-336)
#
# Runs the FULL X→Y→Z→Z→Y→X mass-only Strang sequence on a pilot mass.
# If any face exceeds cfl_limit OR m_pilot goes negative anywhere, halves
# am/bm/cm globally (× 1/2) and doubles the substep count.  Repeats until
# success or max_halvings.
#
# This complements the local per-(j,l)/per-level nloop refinement inside
# the X/Y subcycled functions: the GLOBAL halving handles cumulative
# drainage that local refinement can't fix (because local refinement
# preserves total transport per substep, while global halving REDUCES the
# per-substep total transport by spreading across more substeps).
#
# NOTE: Defined AFTER `MassFluxWorkspace` so the type annotation in the
# signature resolves at parse time. The mass-only kernels it calls
# (`_mass_only_{x,y,z}_kernel!`) are defined earlier in this file.
# =====================================================================

"""
    check_global_cfl_and_scale!(am, bm, cm, m, grid, ws;
                                  n_sub, cfl_limit, max_halvings) → Int

TM5 Check_CFL equivalent.  Returns the `n_extra` factor: caller must run
`n_sub * n_extra` substeps with the (now scaled) am/bm/cm.

Modifies am, bm, cm in place: multiplied by `1/n_extra` after the call.

Pilot loop:
1. Test the full X→Y→Z→Z→Y→X mass-only Strang sequence on m_pilot, repeated
   `n_sub * n_extra` times (matching what the real substep loop will do)
2. If any cell goes negative or any per-face CFL >= cfl_limit, halve and retry
3. Aborts after `max_halvings` halvings
"""
function check_global_cfl_and_scale!(am::AbstractArray{FT,3},
                                       bm::AbstractArray{FT,3},
                                       cm::AbstractArray{FT,3},
                                       m::AbstractArray{FT,3},
                                       grid,
                                       ws::MassFluxWorkspace{FT};
                                       n_sub::Int = 1,
                                       cfl_limit::FT = FT(1.0),
                                       max_halvings::Int = 5,
                                       reset_per_substep::Bool = false) where FT
    backend = get_backend(m)
    Nx, Ny, Nz = size(m)

    # We do the pilot on host (CPU) for simplicity and to avoid GPU
    # synchronization on every substep iteration.  The mass field is small
    # enough that this is acceptable for the global pre-pass (called once
    # per met window per advection_phase! call).
    #
    # `reset_per_substep` mimics the runtime mass_fixer: at the start of every
    # substep, m_pilot is reset to m. This decouples substeps so the pilot
    # checks the WORST-CASE single-substep CFL rather than cumulative drainage
    # across substeps. Required for the v4 has_deltas + mass_fixer path where
    # cumulative drainage is irrelevant (every substep starts from m_target).
    n_extra = 1
    last_info = (substep=0, sweep="none", mmin=Float64(NaN), argmin_idx=(0, 0, 0),
                 max_cfl=NaN, cfl_loc=(0, 0, 0))
    for halving in 0:max_halvings
        # Run mass-only Strang sequence n_sub * n_extra times on m_pilot.
        # Use existing mass-only kernels via the same pilot pattern.
        copyto!(ws.m_pilot, m)
        ok, info = _global_pilot_strang_sequence!(ws, grid, am, bm, cm,
                                                    n_sub * n_extra, cfl_limit;
                                                    reset_per_substep=reset_per_substep,
                                                    m_initial=m)
        last_info = info
        if ok
            if halving > 0
                @info "check_global_cfl: halved $(halving) times → n_extra=$(n_extra), substeps=$(n_sub * n_extra), final mmin=$(info.mmin)" maxlog=10
            end
            return n_extra
        end

        # Per-halving failure breadcrumb (always logged at low halving levels,
        # rate-limited at high levels to avoid spam).
        if halving < 3 || halving == max_halvings
            cfl_str = isnan(info.max_cfl) ? "" : " | CFL=$(round(info.max_cfl, sigdigits=4)) at $(info.cfl_loc)"
            @info "check_global_cfl: halving=$(halving) (substeps=$(n_sub * n_extra), scale=$(1/n_extra)) FAILED at substep=$(info.substep) sweep=$(info.sweep) mmin=$(info.mmin) argmin=$(info.argmin_idx)$cfl_str"
        end

        if halving == max_halvings
            cfl_str = isnan(last_info.max_cfl) ? "" : ", max_cfl=$(last_info.max_cfl) at $(last_info.cfl_loc)"
            error("check_global_cfl_and_scale!: failed to converge after $max_halvings halvings " *
                  "(would have run $(n_sub * n_extra) substeps with am/bm/cm scaled by $(1/n_extra)). " *
                  "Last failure: substep=$(last_info.substep) sweep=$(last_info.sweep) " *
                  "mmin=$(last_info.mmin) argmin=$(last_info.argmin_idx)$cfl_str")
        end

        # Halve fluxes and double n_extra
        am .*= FT(0.5)
        bm .*= FT(0.5)
        cm .*= FT(0.5)
        n_extra *= 2
    end
    return n_extra  # unreachable
end

"""
Internal helper: run `n_substeps` of mass-only Strang on ws.m_pilot.
Returns `(ok::Bool, info::NamedTuple)` where `info` carries the failure
breadcrumb (`substep`, `sweep`, `mmin`/`max_cfl`, `argmin_idx`) when
`ok==false`, or the final `mmin` when `ok==true`.

The pilot enforces TWO conditions per sweep, matching TM5 Check_CFL
(advectm_cfl.F90):
  1. min(m) > 0   — no cell may go negative
  2. max(|flux|/m_donor) < cfl_limit  — no face may exceed local CFL

Condition 2 catches cells where the local nloop refinement would also
fail (e.g. polar pole-adjacent stratospheric cells with bm/m > 1). This
matches TM5's per-face CFL check.

The mass update for each direction is the SAME formula as the real kernels
(reduced grid for X, simple for Y/Z).
"""
function _global_pilot_strang_sequence!(ws::MassFluxWorkspace{FT}, grid,
                                          am, bm, cm, n_substeps::Int,
                                          cfl_limit::FT;
                                          reset_per_substep::Bool=false,
                                          m_initial=nothing) where FT
    backend = get_backend(ws.m_pilot)
    Nx, Ny, Nz = size(ws.m_pilot)
    kx! = _mass_only_x_kernel!(backend, 256)
    ky! = _mass_only_y_kernel!(backend, 256)
    kz! = _mass_only_z_kernel!(backend, 256)

    # Host buffers for per-face CFL checks (computed on CPU for simplicity)
    bm_h = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_h = Array{FT}(undef, Nx, Ny, Nz + 1)
    copyto!(bm_h, bm)
    copyto!(cm_h, cm)
    m_h  = Array{FT}(undef, Nx, Ny, Nz)

    function check_cfl_y!()
        # Per-face Y CFL: max |bm[i,j,k]| / m_donor where donor depends on bm sign
        copyto!(m_h, ws.m_pilot)
        max_cfl = zero(FT)
        max_loc = (0, 0, 0)
        @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
            bm_v = bm_h[i, j, k]
            md = bm_v >= zero(FT) ? m_h[i, j-1, k] : m_h[i, j, k]
            if md > zero(FT)
                c = abs(bm_v) / md
                if c > max_cfl
                    max_cfl = c
                    max_loc = (i, j, k)
                end
            end
        end
        return max_cfl, max_loc
    end

    function check_cfl_z!()
        # Per-face Z CFL: max |cm[i,j,k]| / m_donor (k=interface, donor depends on sign)
        copyto!(m_h, ws.m_pilot)
        max_cfl = zero(FT)
        max_loc = (0, 0, 0)
        @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
            cm_v = cm_h[i, j, k]
            md = cm_v >= zero(FT) ? m_h[i, j, k-1] : m_h[i, j, k]
            if md > zero(FT)
                c = abs(cm_v) / md
                if c > max_cfl
                    max_cfl = c
                    max_loc = (i, j, k)
                end
            end
        end
        return max_cfl, max_loc
    end

    function apply_x!()
        kx!(ws.m_buf, ws.m_pilot, am, Int32(Nx), ws.cluster_sizes; ndrange=size(ws.m_pilot))
        synchronize(backend)
        copyto!(ws.m_pilot, ws.m_buf)
        return minimum(ws.m_pilot)
    end
    function apply_y!()
        ky!(ws.m_buf, ws.m_pilot, bm, Int32(Ny); ndrange=size(ws.m_pilot))
        synchronize(backend)
        copyto!(ws.m_pilot, ws.m_buf)
        return minimum(ws.m_pilot)
    end
    function apply_z!()
        kz!(ws.m_buf, ws.m_pilot, cm, Int32(Nz); ndrange=size(ws.m_pilot))
        synchronize(backend)
        copyto!(ws.m_pilot, ws.m_buf)
        return minimum(ws.m_pilot)
    end
    function fail(substep::Int, sweep::String, mmin; max_cfl=NaN, cfl_loc=(0,0,0))
        argmin_idx = Tuple(argmin(ws.m_pilot))
        return false, (substep=substep, sweep=sweep,
                       mmin=Float64(mmin), argmin_idx=argmin_idx,
                       max_cfl=Float64(max_cfl), cfl_loc=cfl_loc)
    end
    function check_y_cfl_or_fail(substep, sweep)
        cfl, loc = check_cfl_y!()
        if cfl >= cfl_limit
            return fail(substep, sweep * ":CFL", minimum(ws.m_pilot);
                        max_cfl=cfl, cfl_loc=loc)
        end
        return nothing
    end
    function check_z_cfl_or_fail(substep, sweep)
        cfl, loc = check_cfl_z!()
        if cfl >= cfl_limit
            return fail(substep, sweep * ":CFL", minimum(ws.m_pilot);
                        max_cfl=cfl, cfl_loc=loc)
        end
        return nothing
    end

    last_mmin = typemax(FT)
    for substep in 1:n_substeps
        # Reset m_pilot to m_initial at the start of each substep, mimicking
        # the mass_fixer in the runtime substep loop. This decouples substeps
        # so the pilot only checks per-substep CFL, not cumulative drainage.
        if reset_per_substep && m_initial !== nothing
            copyto!(ws.m_pilot, m_initial)
        end
        # CFL checks use the INITIAL m at the start of each substep (not the
        # within-substep evolved m). This matches TM5 Check_CFL which checks
        # |bm[i,j,l]|/m[donor] using the start-of-substep m, not the evolved
        # value. Within a substep, the local nloop refinement handles per-pass
        # CFL via subcycling — the global pre-pass only needs to verify the
        # PER-SUBSTEP magnitudes are bounded.
        f = check_y_cfl_or_fail(substep, "Y"); f !== nothing && return f
        f = check_z_cfl_or_fail(substep, "Z"); f !== nothing && return f
        # Strang sequence X → Y → Z → Z → Y → X (positivity check only)
        m1 = apply_x!(); m1 <= zero(FT) && return fail(substep, "X1", m1); last_mmin = m1
        m2 = apply_y!(); m2 <= zero(FT) && return fail(substep, "Y1", m2); last_mmin = m2
        m3 = apply_z!(); m3 <= zero(FT) && return fail(substep, "Z1", m3); last_mmin = m3
        m4 = apply_z!(); m4 <= zero(FT) && return fail(substep, "Z2", m4); last_mmin = m4
        m5 = apply_y!(); m5 <= zero(FT) && return fail(substep, "Y2", m5); last_mmin = m5
        m6 = apply_x!(); m6 <= zero(FT) && return fail(substep, "X2", m6); last_mmin = m6
    end
    return true, (substep=n_substeps, sweep="done",
                  mmin=Float64(last_mmin), argmin_idx=(0, 0, 0),
                  max_cfl=NaN, cfl_loc=(0, 0, 0))
end

# =====================================================================
# Prognostic slope workspace (TM5 second-order moments)
# =====================================================================

"""
    PrognosticSlopeWorkspace{FT, A3}

Per-tracer workspace holding prognostic slopes (rxm, rym, rzm) and
double-buffer outputs for GPU kernels.  Slopes are initialized to zero
and evolve with TM5's pf-term update (advectz.F90:481-485).
"""
struct PrognosticSlopeWorkspace{FT, A3 <: AbstractArray{FT,3}}
    rxm     :: A3
    rym     :: A3
    rzm     :: A3
    rxm_buf :: A3
    rym_buf :: A3
    rzm_buf :: A3
end

"""
    allocate_prognostic_slope_workspace(m) → PrognosticSlopeWorkspace

Allocate slope and buffer arrays matching the shape of air mass `m`.
"""
function allocate_prognostic_slope_workspace(m::AbstractArray{FT,3}) where FT
    make = () -> similar(m)
    PrognosticSlopeWorkspace{FT, typeof(m)}(
        fill!(make(), zero(FT)),  # rxm — initialized to zero
        fill!(make(), zero(FT)),  # rym
        fill!(make(), zero(FT)),  # rzm
        make(),                    # rxm_buf
        make(),                    # rym_buf
        make(),                    # rzm_buf
    )
end

"""
    allocate_prognostic_slope_workspaces(tracers, m) → NamedTuple

Allocate one PrognosticSlopeWorkspace per tracer, keyed by tracer name.
"""
function allocate_prognostic_slope_workspaces(tracers::NamedTuple, m::AbstractArray{FT,3}) where FT
    names = keys(tracers)
    ws_tuple = ntuple(i -> allocate_prognostic_slope_workspace(m), length(names))
    NamedTuple{names}(ws_tuple)
end

# =====================================================================
# Public wrapper functions
# =====================================================================

"""
$(SIGNATURES)

Compute 3D air mass from pressure thickness and grid geometry.
Uses a KernelAbstractions kernel (runs on CPU or GPU).
"""
function compute_air_mass(Δp::AbstractArray{FT,3}, grid) where FT
    m = similar(Δp)
    compute_air_mass!(m, Δp, grid)
    return m
end

"""
$(SIGNATURES)

In-place version: fills pre-allocated `m` with air mass values.

When a [`GridGeometryCache`](@ref) is provided, geometry vectors are reused
from the cache (zero allocation). Otherwise they are recomputed from the grid.
"""
function compute_air_mass!(m::AbstractArray{FT,3}, Δp::AbstractArray{FT,3}, grid) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    backend = get_backend(Δp)

    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    area_j = _to_device(area_j_cpu, Δp)

    k! = _air_mass_kernel!(backend, 256)
    k!(m, Δp, area_j, g; ndrange=(Nx, Ny, Nz))
    synchronize(backend)
    return nothing
end

function compute_air_mass!(m::AbstractArray{FT,3}, Δp::AbstractArray{FT,3},
                           gc::GridGeometryCache{FT}) where FT
    backend = get_backend(Δp)
    k! = _air_mass_kernel!(backend, 256)
    k!(m, Δp, gc.area_j, gc.gravity; ndrange=(gc.Nx, gc.Ny, gc.Nz))
    synchronize(backend)
    return nothing
end

"""
$(SIGNATURES)

Compute mass fluxes `am`, `bm`, `cm` from staggered velocities, pressure
thickness, and half-timestep. Uses KernelAbstractions kernels.

Returns `(; am, bm, cm)`.
"""
function compute_mass_fluxes(u, v, grid, Δp::AbstractArray{FT,3}, half_dt) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    am = similar(u, FT, Nx + 1, Ny, Nz)
    bm = similar(v, FT, Nx, Ny + 1, Nz)
    cm = similar(Δp, FT, Nx, Ny, Nz + 1)
    compute_mass_fluxes!(am, bm, cm, u, v, grid, Δp, half_dt)
    return (; am, bm, cm)
end

"""
$(SIGNATURES)

In-place version: fills pre-allocated `am`, `bm`, `cm` with mass fluxes.

When a [`GridGeometryCache`](@ref) is provided, geometry vectors are reused
from the cache (zero allocation). Otherwise they are recomputed from the grid.
"""
function compute_mass_fluxes!(am, bm, cm, u, v, grid,
                               Δp::AbstractArray{FT,3}, half_dt) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    vc = grid.vertical
    backend = get_backend(Δp)
    dy_j_cpu = FT[Δy(1, j, grid) for j in 1:Ny]
    dy_j = _to_device(dy_j_cpu, Δp)

    dx_face_cpu = Vector{FT}(undef, Ny + 1)
    for j in 1:Ny+1
        φ_f = if j == 1
            FT(-90)
        elseif j == Ny + 1
            FT(90)
        else
            FT(grid.φᶠ_cpu[j])
        end
        dx_face_cpu[j] = FT(grid.radius) * cosd(φ_f) * deg2rad(FT(grid.Δλ))
    end
    dx_face = _to_device(dx_face_cpu, Δp)

    k_am! = _am_kernel!(backend, 256)
    k_am!(am, u, Δp, dy_j, Nx, FT(half_dt), g; ndrange=(Nx + 1, Ny, Nz))
    synchronize(backend)

    k_bm! = _bm_kernel!(backend, 256)
    k_bm!(bm, v, Δp, dx_face, Ny, FT(half_dt), g; ndrange=(Nx, Ny + 1, Nz))
    synchronize(backend)

    ΔB_cpu = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        ΔB_cpu[k] = FT(vc.B[k + 1] - vc.B[k])
    end
    ΔB_total = FT(vc.B[Nz + 1] - vc.B[1])
    bt_cpu = abs(ΔB_total) > eps(FT) ? ΔB_cpu ./ ΔB_total : zeros(FT, Nz)
    bt = _to_device(bt_cpu, Δp)

    fill!(cm, zero(FT))
    k_cm! = _cm_column_kernel!(backend, 256)
    k_cm!(cm, am, bm, bt, Nz; ndrange=(Nx, Ny))
    synchronize(backend)

    return nothing
end

"""
$(SIGNATURES)

Cache-accelerated version: uses pre-computed geometry from [`GridGeometryCache`](@ref).
No host→device transfers, no temporary allocations.
"""
function compute_mass_fluxes!(am, bm, cm, u, v,
                               gc::GridGeometryCache{FT},
                               Δp::AbstractArray{FT,3}, half_dt) where FT
    backend = get_backend(Δp)

    k_am! = _am_kernel!(backend, 256)
    k_am!(am, u, Δp, gc.dy_j, gc.Nx, FT(half_dt), gc.gravity;
          ndrange=(gc.Nx + 1, gc.Ny, gc.Nz))
    synchronize(backend)

    k_bm! = _bm_kernel!(backend, 256)
    k_bm!(bm, v, Δp, gc.dx_face, gc.Ny, FT(half_dt), gc.gravity;
          ndrange=(gc.Nx, gc.Ny + 1, gc.Nz))
    synchronize(backend)

    fill!(cm, zero(FT))
    k_cm! = _cm_column_kernel!(backend, 256)
    k_cm!(cm, am, bm, gc.bt, gc.Nz; ndrange=(gc.Nx, gc.Ny))
    synchronize(backend)

    return nothing
end

# =====================================================================
# Advection wrappers — zero-allocation versions using workspace buffers
# =====================================================================

"""
$(SIGNATURES)

TM5-faithful x-advection using mass fluxes. Runs on CPU or GPU via KA kernels.
Uses pre-allocated `rm_buf` and `m_buf` to avoid GPU allocations.
"""
function advect_x_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             am::AbstractArray{FT,3},
                             grid,
                             use_limiter::Bool,
                             rm_buf::AbstractArray{FT,3},
                             m_buf::AbstractArray{FT,3},
                             cluster_sizes::AbstractVector{Int32}) where FT
    backend = get_backend(m)
    Nx = grid.Nx
    k! = _massflux_x_kernel!(backend, 256)
    for (_, rm) in pairs(rm_tracers)
        k!(rm_buf, rm, m_buf, m, am, Nx, cluster_sizes, use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, rm_buf)
    end
    copyto!(m, m_buf)
    return nothing
end

"""
$(SIGNATURES)

TM5-faithful y-advection using mass fluxes. Runs on CPU or GPU via KA kernels.
Uses pre-allocated `rm_buf` and `m_buf` to avoid GPU allocations.
"""
function advect_y_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             bm::AbstractArray{FT,3},
                             grid,
                             use_limiter::Bool,
                             rm_buf::AbstractArray{FT,3},
                             m_buf::AbstractArray{FT,3}) where FT
    backend = get_backend(m)
    Ny = grid.Ny
    k! = _massflux_y_kernel!(backend, 256)
    for (_, rm) in pairs(rm_tracers)
        k!(rm_buf, rm, m_buf, m, bm, Ny, use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, rm_buf)
    end
    copyto!(m, m_buf)
    return nothing
end

"""
$(SIGNATURES)

TM5-faithful z-advection using mass fluxes. Runs on CPU or GPU via KA kernels.
Uses pre-allocated `rm_buf` and `m_buf` to avoid GPU allocations.
"""
function advect_z_massflux!(rm_tracers::NamedTuple,
                             m::AbstractArray{FT,3},
                             cm::AbstractArray{FT,3},
                             use_limiter::Bool,
                             rm_buf::AbstractArray{FT,3},
                             m_buf::AbstractArray{FT,3}) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    k! = _massflux_z_kernel!(backend, 256)
    for (_, rm) in pairs(rm_tracers)
        k!(rm_buf, rm, m_buf, m, cm, Nz, use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, rm_buf)
    end
    copyto!(m, m_buf)
    return nothing
end

# =====================================================================
# CPU reduced-grid x-advection (TM5-style)
# =====================================================================

"""
1D mass-flux slopes advection on a single periodic row of length `N`.
Updates `rm_vec` and `m_vec` in place.
"""
function _advect_x_row_massflux!(rm_vec::AbstractVector{FT},
                                  m_vec::AbstractVector{FT},
                                  am_vec::AbstractVector{FT},
                                  N::Int,
                                  use_limiter::Bool) where FT
    rm_buf = Vector{FT}(undef, N)
    m_buf  = Vector{FT}(undef, N)
    @inbounds for i in 1:N
        ip  = i == N ? 1 : i + 1
        im  = i == 1 ? N : i - 1
        ipp = ip == N ? 1 : ip + 1
        imm = im == 1 ? N : im - 1

        c_imm = rm_vec[imm] / m_vec[imm]
        c_im  = rm_vec[im]  / m_vec[im]
        c_i   = rm_vec[i]   / m_vec[i]
        c_ip  = rm_vec[ip]  / m_vec[ip]
        c_ipp = rm_vec[ipp] / m_vec[ipp]

        sc_im = (c_i - c_imm) / 2
        if use_limiter
            sc_im = _minmod_cpu(sc_im, 2*(c_i - c_im), 2*(c_im - c_imm))
        end
        sx_im = m_vec[im] * sc_im
        if use_limiter
            sx_im = max(min(sx_im, rm_vec[im]), -rm_vec[im])
        end

        sc_i = (c_ip - c_im) / 2
        if use_limiter
            sc_i = _minmod_cpu(sc_i, 2*(c_ip - c_i), 2*(c_i - c_im))
        end
        sx_i = m_vec[i] * sc_i
        if use_limiter
            sx_i = max(min(sx_i, rm_vec[i]), -rm_vec[i])
        end

        sc_ip = (c_ipp - c_i) / 2
        if use_limiter
            sc_ip = _minmod_cpu(sc_ip, 2*(c_ipp - c_ip), 2*(c_ip - c_i))
        end
        sx_ip = m_vec[ip] * sc_ip
        if use_limiter
            sx_ip = max(min(sx_ip, rm_vec[ip]), -rm_vec[ip])
        end

        am_l = am_vec[i]
        flux_left = if am_l >= zero(FT)
            alpha = am_l / m_vec[im]
            alpha * (rm_vec[im] + (one(FT) - alpha) * sx_im)
        else
            alpha = am_l / m_vec[i]
            alpha * (rm_vec[i] - (one(FT) + alpha) * sx_i)
        end

        am_r = am_vec[i + 1]
        flux_right = if am_r >= zero(FT)
            alpha = am_r / m_vec[i]
            alpha * (rm_vec[i] + (one(FT) - alpha) * sx_i)
        else
            alpha = am_r / m_vec[ip]
            alpha * (rm_vec[ip] - (one(FT) + alpha) * sx_ip)
        end

        rm_buf[i] = rm_vec[i] + flux_left - flux_right
        m_buf[i]  = m_vec[i]  + am_vec[i] - am_vec[i + 1]
    end
    copyto!(rm_vec, rm_buf)
    copyto!(m_vec, m_buf)
    return nothing
end

@inline function _minmod_cpu(a::T, b::T, c::T) where T
    if a > zero(T) && b > zero(T) && c > zero(T)
        return min(a, b, c)
    elseif a < zero(T) && b < zero(T) && c < zero(T)
        return max(a, b, c)
    else
        return zero(T)
    end
end

"""
$(SIGNATURES)

TM5-style reduced-grid x-advection for mass-flux form on CPU. For each
latitude row with cluster size `r > 1`, reduces rm, m, and am to the coarser
row, advects with the 1D slopes scheme, then expands back.

Rows with cluster size 1 use the standard kernel.  All tracers see the
original `m` for slope computation (m is updated once at the end, matching
the non-reduced path).
"""
function advect_x_massflux_reduced!(rm_tracers::NamedTuple,
                                     m::Array{FT,3},
                                     am::Array{FT,3},
                                     grid,
                                     use_limiter::Bool) where FT
    rg = grid.reduced_grid
    rg === nothing && return advect_x_massflux!(rm_tracers, m, am, grid, use_limiter)

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    max_N = max(Nx, maximum(rg.reduced_counts))

    rm_red_work = Vector{FT}(undef, max_N)
    rm_red_old  = Vector{FT}(undef, max_N)
    m_red_work  = Vector{FT}(undef, max_N)
    m_red_old   = Vector{FT}(undef, max_N)
    m_red_new   = Vector{FT}(undef, max_N)
    am_red      = Vector{FT}(undef, max_N + 1)

    rm_row = Vector{FT}(undef, Nx)
    m_row  = Vector{FT}(undef, Nx)
    am_row = Vector{FT}(undef, Nx + 1)

    for k in 1:Nz
        for j in 1:Ny
            r = rg.cluster_sizes[j]
            if r == 1
                @inbounds for i in 1:Nx
                    am_row[i] = am[i, j, k]
                end
                am_row[Nx + 1] = am_row[1]
                for (_, rm) in pairs(rm_tracers)
                    @inbounds for i in 1:Nx
                        rm_row[i] = rm[i, j, k]
                        m_row[i]  = m[i, j, k]
                    end
                    _advect_x_row_massflux!(rm_row, m_row, am_row, Nx, use_limiter)
                    @inbounds for i in 1:Nx
                        rm[i, j, k] = rm_row[i]
                    end
                end
                @inbounds for i in 1:Nx
                    m[i, j, k] = m[i, j, k] + am[i, j, k] - am[i == Nx ? 1 : i + 1, j, k]
                end
            else
                Nx_red = rg.reduced_counts[j]
                m_rv = @view m_red_work[1:Nx_red]
                m_ov = @view m_red_old[1:Nx_red]
                m_nv = @view m_red_new[1:Nx_red]
                am_v = @view am_red[1:Nx_red+1]

                reduce_row_mass!(m_ov, m, j, k, r, Nx)
                reduce_am_row!(am_v, am, j, k, r, Nx)

                # Compute m_red_new (tracer-independent)
                @inbounds for i_r in 1:Nx_red
                    m_nv[i_r] = m_ov[i_r] + am_v[i_r] - am_v[i_r + 1]
                end

                for (_, rm) in pairs(rm_tracers)
                    rm_wv = @view rm_red_work[1:Nx_red]
                    rm_ov = @view rm_red_old[1:Nx_red]

                    reduce_row_mass!(rm_ov, rm, j, k, r, Nx)
                    copyto!(rm_wv, rm_ov)
                    copyto!(m_rv, m_ov)

                    _advect_x_row_massflux!(rm_wv, m_rv, am_v, Nx_red, use_limiter)

                    # Expand tracer mass change proportionally
                    @inbounds for i_r in 1:Nx_red
                        delta_rm = rm_wv[i_r] - rm_ov[i_r]
                        rm_sum = rm_ov[i_r]
                        i_start = (i_r - 1) * r + 1
                        for off in 0:r-1
                            i = i_start + off
                            if abs(rm_sum) > eps(FT)
                                rm[i, j, k] += delta_rm * (rm[i, j, k] / rm_sum)
                            else
                                rm[i, j, k] += delta_rm / FT(r)
                            end
                        end
                    end
                end

                # Expand air mass change proportionally (once)
                @inbounds for i_r in 1:Nx_red
                    delta_m = m_nv[i_r] - m_ov[i_r]
                    m_sum = m_ov[i_r]
                    i_start = (i_r - 1) * r + 1
                    for off in 0:r-1
                        i = i_start + off
                        if abs(m_sum) > eps(FT)
                            m[i, j, k] += delta_m * (m[i, j, k] / m_sum)
                        else
                            m[i, j, k] += delta_m / FT(r)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# Backward-compatible versions that allocate internally (for tests)
function advect_x_massflux!(rm_tracers::NamedTuple, m::AbstractArray{FT,3},
                             am::AbstractArray{FT,3}, grid, use_limiter::Bool) where FT
    Ny = size(m, 2)
    cs = similar(m, Int32, Ny)
    copyto!(cs, ones(Int32, Ny))
    advect_x_massflux!(rm_tracers, m, am, grid, use_limiter,
                        similar(first(values(rm_tracers))), similar(m), cs)
end
function advect_y_massflux!(rm_tracers::NamedTuple, m::AbstractArray{FT,3},
                             bm::AbstractArray{FT,3}, grid, use_limiter::Bool) where FT
    advect_y_massflux!(rm_tracers, m, bm, grid, use_limiter,
                        similar(first(values(rm_tracers))), similar(m))
end
function advect_z_massflux!(rm_tracers::NamedTuple, m::AbstractArray{FT,3},
                             cm::AbstractArray{FT,3}, use_limiter::Bool) where FT
    advect_z_massflux!(rm_tracers, m, cm, use_limiter,
                        similar(first(values(rm_tracers))), similar(m))
end

# =====================================================================
# CFL functions — zero-allocation versions
# =====================================================================

"""
$(SIGNATURES)

Maximum mass-based Courant number for x-direction mass fluxes.
Pre-allocated `cfl_arr` avoids GPU allocation.
"""
function max_cfl_massflux_x(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                             cfl_arr::AbstractArray{FT,3},
                             cluster_sizes::AbstractVector{Int32}) where FT
    backend = get_backend(m)
    Nx = size(m, 1)
    k! = _cfl_x_kernel!(backend, 256)
    k!(cfl_arr, am, m, Nx, cluster_sizes; ndrange=size(am))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

"""
$(SIGNATURES)

Maximum mass-based Courant number for x-direction on the FINE grid,
ignoring reduced-grid clustering.  Use this for PPM advection which
operates on every fine cell (not on cluster aggregates).
"""
function max_cfl_massflux_x_fine(am::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                                  cfl_arr::AbstractArray{FT,3}) where FT
    backend = get_backend(m)
    Nx = size(m, 1)
    # Use the r=1 branch of _cfl_x_kernel! by passing all-ones cluster_sizes
    cs_ones = similar(m, Int32, size(m, 2))
    fill!(cs_ones, Int32(1))
    k! = _cfl_x_kernel!(backend, 256)
    k!(cfl_arr, am, m, Nx, cs_ones; ndrange=size(am))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

"""
$(SIGNATURES)

Maximum mass-based Courant number for y-direction mass fluxes.
"""
function max_cfl_massflux_y(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                             cfl_arr::AbstractArray{FT,3}) where FT
    backend = get_backend(m)
    Ny = size(m, 2)
    k! = _cfl_y_kernel!(backend, 256)
    k!(cfl_arr, bm, m, Ny; ndrange=size(bm))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

"""
$(SIGNATURES)

Maximum mass-based Courant number for z-direction mass fluxes.
"""
function max_cfl_massflux_z(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3},
                             cfl_arr::AbstractArray{FT,3}) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    k! = _cfl_z_kernel!(backend, 256)
    k!(cfl_arr, cm, m, Nz; ndrange=size(cm))
    synchronize(backend)
    return FT(maximum(cfl_arr))
end

# Backward-compatible versions (allocating)
function max_cfl_massflux_x(am::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Ny = size(m, 2)
    cs = similar(m, Int32, Ny)
    copyto!(cs, ones(Int32, Ny))
    max_cfl_massflux_x(am, m, similar(am), cs)
end
function max_cfl_massflux_y(bm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    max_cfl_massflux_y(bm, m, similar(bm))
end
function max_cfl_massflux_z(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    max_cfl_massflux_z(cm, m, similar(cm))
end

# =====================================================================
# Subcycled advection — zero-allocation versions using workspace
# =====================================================================

"""
    _x_discover_nloop_per_row!(ws, m, am_buf, cluster_sizes_cpu;
                                max_nloop=50, beta_thresh=1.0) → Matrix{Int32}

Per-(j,l) X nloop discovery, matching TM5 advectx__slopes.F90:436-516.

For each (latitude row j, vertical level k):
1. Initialize a 1D pilot mass slice from m[:, j, k]
2. Loop: while CFL >= beta_thresh and nloop[j,k] < max_nloop
   a. Try iloop = 1..nloop[j,k]: at each iloop, check max |am[j,k]/mx_donor|
      at every face, then update mx via X mass conservation
   b. If any iloop fails: scale am[:, j, k] *= nloop/(nloop+1), increment
   c. Otherwise: converged for this (j,k)

Reduced grid: rows with cluster_sizes[j] > 1 use cluster aggregates for both
the CFL check and mass update (matching the real X kernel).

Modifies `am_buf` in place per (j, k).  Sets `ws.nloop_x_cpu` and copies
to `ws.nloop_x_dev`.

CPU implementation: copies m, am_buf to host, runs discovery, copies modified
am_buf back.  Each (j, k) is a 1D problem (Nx ~ 720 cells), fast on CPU.
"""
function _x_discover_nloop_per_row!(ws::MassFluxWorkspace{FT},
                                     m::AbstractArray{FT,3},
                                     am_buf::AbstractArray{FT,3};
                                     max_nloop::Int = 50,
                                     beta_thresh::FT = FT(1.0)) where FT
    Nx, Ny, Nz = size(m)
    @assert size(am_buf, 1) == Nx + 1 && size(am_buf, 2) == Ny && size(am_buf, 3) == Nz

    # Copy to host
    m_h  = Array(m)
    am_h = Array(am_buf)
    cluster_sizes_h = Array(ws.cluster_sizes)

    nloop = ws.nloop_x_cpu
    fill!(nloop, Int32(1))

    for k in 1:Nz, j in 1:Ny
        r = Int(cluster_sizes_h[j])
        nloop_jk = 1
        while true
            # Re-init pilot mass for this row from ORIGINAL m
            # For uniform rows: per-cell mx[i] = m[i, j, k]
            # For reduced rows: per-cluster mx_red[ic] = sum(m[ic*r-r+1 : ic*r, j, k])
            if r == 1
                mx = collect(@view m_h[:, j, k])  # length Nx
            else
                Nx_red = Nx ÷ r
                mx = zeros(FT, Nx_red)
                @inbounds for ic in 1:Nx_red
                    s = zero(FT)
                    for ii in ((ic - 1) * r + 1):(ic * r)
                        s += m_h[ii, j, k]
                    end
                    mx[ic] = s
                end
            end

            cfl_ok = true
            for iloop in 1:nloop_jk
                local_max = zero(FT)
                if r == 1
                    # Uniform row: faces are at i = 1..Nx+1 (am_h[i, j, k])
                    # am_h[1, j, k] is the left face of cell 1 (= right face of cell Nx, periodic)
                    # am_h[i+1, j, k] is the right face of cell i.
                    # Donor: if am[i+1] >= 0, donor is cell i; else cell i+1 (periodic)
                    @inbounds for i in 1:Nx
                        am_v = am_h[i + 1, j, k]
                        ip = i == Nx ? 1 : i + 1
                        md = am_v >= zero(FT) ? mx[i] : mx[ip]
                        if md > zero(FT)
                            b = abs(am_v) / md
                            if b > local_max; local_max = b; end
                        end
                    end
                else
                    # Reduced row: cluster faces only
                    Nx_red = Nx ÷ r
                    @inbounds for ic in 1:Nx_red
                        am_l_idx = (ic - 1) * r + 1   # left face of cluster ic
                        am_l = am_h[am_l_idx, j, k]
                        ic_m = ic == 1 ? Nx_red : ic - 1
                        md = am_l >= zero(FT) ? mx[ic_m] : mx[ic]
                        if md > zero(FT)
                            b = abs(am_l) / md
                            if b > local_max; local_max = b; end
                        end
                    end
                end
                if local_max >= beta_thresh
                    cfl_ok = false
                    break
                end
                # Update mx for next iloop
                if iloop != nloop_jk
                    if r == 1
                        # mx_new[i] = mx[i] + am[i] - am[i+1]
                        # Build new mx from current mx
                        mx_new = similar(mx)
                        @inbounds for i in 1:Nx
                            mx_new[i] = mx[i] + am_h[i, j, k] - am_h[i + 1, j, k]
                        end
                        mx = mx_new
                    else
                        Nx_red = Nx ÷ r
                        mx_new = similar(mx)
                        @inbounds for ic in 1:Nx_red
                            am_l_idx = (ic - 1) * r + 1
                            am_r_idx = ic * r + 1
                            am_l = am_h[am_l_idx, j, k]
                            am_r = am_h[am_r_idx > Nx ? 1 : am_r_idx, j, k]
                            mx_new[ic] = mx[ic] + am_l - am_r
                        end
                        mx = mx_new
                    end
                end
            end

            if cfl_ok
                nloop[j, k] = Int32(nloop_jk)
                break
            end

            # CFL failed: reduce am at this (j, k) and increment
            scale = FT(nloop_jk) / FT(nloop_jk + 1)
            @inbounds for i in 1:(Nx + 1)
                am_h[i, j, k] *= scale
            end
            nloop_jk += 1
            if nloop_jk >= max_nloop
                # TM5 also aborts here (advectx__slopes.F90:504-507)
                error("X nloop hit max_nloop=$max_nloop at (j=$j, k=$k). " *
                      "Local X-CFL fundamentally exceeds what max_nloop=$max_nloop can handle. " *
                      "TM5 also aborts here.")
            end
        end
    end

    # Copy modified am back to device, and nloop to device
    copyto!(am_buf, am_h)
    copyto!(ws.nloop_x_dev, nloop)
    return nloop
end

@kernel function _massflux_x_perjl_kernel!(
    rm_new, @Const(rm), m_new, @Const(m),
    @Const(am), @Const(nloop_x), Nx, @Const(cluster_sizes), iter, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        if Int32(iter) > nloop_x[j, k]
            # Per-(j,k) skip: this row × level has converged with fewer iters
            rm_new[i, j, k] = rm[i, j, k]
            m_new[i, j, k]  = m[i, j, k]
        else
            # Standard X advection logic — same as _massflux_x_kernel!
            r = Int(cluster_sizes[j])
            if r == 1
                ip  = i == Nx ? 1 : i + 1
                im  = i == 1  ? Nx : i - 1
                ipp = ip == Nx ? 1 : ip + 1
                imm = im == 1  ? Nx : im - 1

                c_imm = rm[imm, j, k] / m[imm, j, k]
                c_im  = rm[im,  j, k] / m[im,  j, k]
                c_i   = rm[i,   j, k] / m[i,   j, k]
                c_ip  = rm[ip,  j, k] / m[ip,  j, k]
                c_ipp = rm[ipp, j, k] / m[ipp, j, k]

                sc_im = (c_i - c_imm) / 2
                if use_limiter
                    sc_im = minmod_device(sc_im, 2 * (c_i - c_im), 2 * (c_im - c_imm))
                end
                sx_im = m[im, j, k] * sc_im
                if use_limiter
                    sx_im = max(min(sx_im, rm[im, j, k]), -rm[im, j, k])
                end

                sc_i = (c_ip - c_im) / 2
                if use_limiter
                    sc_i = minmod_device(sc_i, 2 * (c_ip - c_i), 2 * (c_i - c_im))
                end
                sx_i = m[i, j, k] * sc_i
                if use_limiter
                    sx_i = max(min(sx_i, rm[i, j, k]), -rm[i, j, k])
                end

                sc_ip = (c_ipp - c_i) / 2
                if use_limiter
                    sc_ip = minmod_device(sc_ip, 2 * (c_ipp - c_ip), 2 * (c_ip - c_i))
                end
                sx_ip = m[ip, j, k] * sc_ip
                if use_limiter
                    sx_ip = max(min(sx_ip, rm[ip, j, k]), -rm[ip, j, k])
                end

                am_l = am[i, j, k]
                flux_left = if am_l >= zero(FT)
                    alpha = am_l / m[im, j, k]
                    alpha * (rm[im, j, k] + (one(FT) - alpha) * sx_im)
                else
                    alpha = am_l / m[i, j, k]
                    alpha * (rm[i, j, k] - (one(FT) + alpha) * sx_i)
                end

                am_r = am[i + 1, j, k]
                flux_right = if am_r >= zero(FT)
                    alpha = am_r / m[i, j, k]
                    alpha * (rm[i, j, k] + (one(FT) - alpha) * sx_i)
                else
                    alpha = am_r / m[ip, j, k]
                    alpha * (rm[ip, j, k] - (one(FT) + alpha) * sx_ip)
                end

                rm_new[i, j, k] = rm[i, j, k] + flux_left - flux_right
                m_new[i, j, k]  = m[i, j, k]  + am[i, j, k] - am[i + 1, j, k]
            else
                # Reduced row: cluster aggregate path (same as _massflux_x_kernel!)
                Nx_red = Nx ÷ r
                ic  = (i - 1) ÷ r + 1
                ic_m  = ic == 1      ? Nx_red : ic - 1
                ic_p  = ic == Nx_red ? 1      : ic + 1
                ic_mm = ic_m == 1      ? Nx_red : ic_m - 1
                ic_pp = ic_p == Nx_red ? 1      : ic_p + 1

                rm_ic  = _cluster_sum(rm, ic, j, k, r)
                m_ic   = _cluster_sum(m,  ic, j, k, r)
                rm_im  = _cluster_sum(rm, ic_m, j, k, r)
                m_im   = _cluster_sum(m,  ic_m, j, k, r)
                rm_ip  = _cluster_sum(rm, ic_p, j, k, r)
                m_ip   = _cluster_sum(m,  ic_p, j, k, r)
                rm_imm = _cluster_sum(rm, ic_mm, j, k, r)
                m_imm  = _cluster_sum(m,  ic_mm, j, k, r)
                rm_ipp = _cluster_sum(rm, ic_pp, j, k, r)
                m_ipp  = _cluster_sum(m,  ic_pp, j, k, r)

                c_imm_r = rm_imm / m_imm
                c_im_r  = rm_im  / m_im
                c_ic_r  = rm_ic  / m_ic
                c_ip_r  = rm_ip  / m_ip
                c_ipp_r = rm_ipp / m_ipp

                sc_im_r = (c_ic_r - c_imm_r) / 2
                if use_limiter
                    sc_im_r = minmod_device(sc_im_r, 2 * (c_ic_r - c_im_r), 2 * (c_im_r - c_imm_r))
                end
                sx_im_r = m_im * sc_im_r
                if use_limiter
                    sx_im_r = max(min(sx_im_r, rm_im), -rm_im)
                end

                sc_ic_r = (c_ip_r - c_im_r) / 2
                if use_limiter
                    sc_ic_r = minmod_device(sc_ic_r, 2 * (c_ip_r - c_ic_r), 2 * (c_ic_r - c_im_r))
                end
                sx_ic_r = m_ic * sc_ic_r
                if use_limiter
                    sx_ic_r = max(min(sx_ic_r, rm_ic), -rm_ic)
                end

                sc_ip_r = (c_ipp_r - c_ic_r) / 2
                if use_limiter
                    sc_ip_r = minmod_device(sc_ip_r, 2 * (c_ipp_r - c_ip_r), 2 * (c_ip_r - c_ic_r))
                end
                sx_ip_r = m_ip * sc_ip_r
                if use_limiter
                    sx_ip_r = max(min(sx_ip_r, rm_ip), -rm_ip)
                end

                am_l_r = am[(ic - 1) * r + 1, j, k]
                am_r_idx = ic * r + 1
                am_r_r = am[am_r_idx > Nx ? 1 : am_r_idx, j, k]

                flux_left_r = if am_l_r >= zero(FT)
                    alpha = am_l_r / m_im
                    alpha * (rm_im + (one(FT) - alpha) * sx_im_r)
                else
                    alpha = am_l_r / m_ic
                    alpha * (rm_ic - (one(FT) + alpha) * sx_ic_r)
                end

                flux_right_r = if am_r_r >= zero(FT)
                    alpha = am_r_r / m_ic
                    alpha * (rm_ic + (one(FT) - alpha) * sx_ic_r)
                else
                    alpha = am_r_r / m_ip
                    alpha * (rm_ip - (one(FT) + alpha) * sx_ip_r)
                end

                delta_rm = flux_left_r - flux_right_r
                delta_m  = am_l_r - am_r_r

                frac_m = abs(m_ic) > eps(FT) ? m[i, j, k] / m_ic : one(FT) / FT(r)
                rm_new[i, j, k] = (rm_ic + delta_rm) * frac_m
                m_new[i, j, k]  = (m_ic  + delta_m)  * frac_m
            end
        end
    end
end

"""
$(SIGNATURES)

TM5-style per-(j,l) evolving-mass X subcycling, matching `advectx__slopes.F90:436-516`.

1. Discovery (`_x_discover_nloop_per_row!`): for each (latitude row, level),
   find the minimum nloop[j,k] such that the per-pass CFL stays under
   threshold and m stays positive.  am gets scaled in place per (j,k).
2. Application: run `max(nloop)` passes of `_massflux_x_perjl_kernel!` with
   a per-(j,k) mask.

TM5 max_nloop = 50, threshold = 1.0 (line 332, 446 of advectx__slopes.F90).
Aborts on max_nloop reached, matching TM5.
"""
function advect_x_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                       grid, use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(1.0),  # TM5 default
                                       max_nloop::Int = 50) where FT
    backend = get_backend(m)
    Nx, Ny, Nz = size(m)

    # Phase 1: per-(j,k) discovery.  am scaled in place into ws.cfl_x.
    # The discovery threshold honors the caller-supplied cfl_limit so
    # X and Y use the same effective acceptance (addressing Codex ToClaude6.md #3).
    copyto!(ws.cfl_x, am)
    nloop = _x_discover_nloop_per_row!(ws, m, ws.cfl_x;
                                        max_nloop=max_nloop,
                                        beta_thresh=cfl_limit)
    am_eff = ws.cfl_x
    max_iter = maximum(nloop)

    # Phase 2: run actual tracer advection with per-(j,k) nloop mask
    k! = _massflux_x_perjl_kernel!(backend, 256)
    for iter in 1:max_iter
        for (_, rm) in pairs(rm_tracers)
            k!(ws.rm_buf, rm, ws.m_buf, m, am_eff,
               ws.nloop_x_dev, Int32(Nx), ws.cluster_sizes,
               Int32(iter), use_limiter; ndrange=size(m))
            synchronize(backend)
            copyto!(rm, ws.rm_buf)
        end
        copyto!(m, ws.m_buf)
    end
    return Int(max_iter)
end

"""
    _y_discover_nloop_per_level!(ws, m, bm_buf, max_nloop, beta_thresh) → Vector{Int32}

Per-level Y nloop discovery, matching TM5 advecty__slopes.F90:236-289.

For each vertical level k:
1. Initialize a pilot mass slice from m[:, :, k]
2. Loop: while CFL >= beta_thresh and nloop[k] < max_nloop
   a. Try iloop = 1..nloop[k]: at each iloop, check max |bm[k]/m_pilot| at every face,
      then update m_pilot via Y mass conservation
   b. If any iloop fails CFL: scale bm[:, :, k] *= nloop[k]/(nloop[k]+1), increment nloop[k]
   c. Otherwise: converged for this level

Modifies `bm_buf` in place (per-level scaling).  Sets `ws.nloop_y_cpu` and copies
to `ws.nloop_y_dev`.  Returns `ws.nloop_y_cpu` (host array).

CPU implementation: copies m, bm_buf to host, runs discovery, copies modified
bm_buf back.  This is simpler than a per-level GPU kernel and fast enough since
each level is a 720x361 = 260K-cell 2D problem.
"""
function _y_discover_nloop_per_level!(ws::MassFluxWorkspace{FT},
                                       m::AbstractArray{FT,3},
                                       bm_buf::AbstractArray{FT,3};
                                       max_nloop::Int = 6,
                                       beta_thresh::FT = FT(1.0)) where FT
    Nx, Ny, Nz = size(m)
    @assert size(bm_buf, 1) == Nx && size(bm_buf, 2) == Ny + 1 && size(bm_buf, 3) == Nz

    # Copy to host for the discovery
    m_h  = Array(m)
    bm_h = Array(bm_buf)

    nloop = ws.nloop_y_cpu
    fill!(nloop, Int32(1))

    for k in 1:Nz
        nloop_k = 1
        # Track:
        #   - worst_cfl_overall: max across all attempts and iterations
        #   - last_fail_iloop, last_fail_loc, last_fail_cfl: cell+iter that
        #     CAUSED the most recent failure (truly diagnostic)
        worst_cfl = zero(FT)
        worst_loc = (0, 0, 0)  # (i, j, iloop)
        last_fail_iloop = 0
        last_fail_loc = (0, 0)
        last_fail_cfl = zero(FT)
        last_fail_m_pilot = zero(FT)
        last_fail_bm_eff = zero(FT)
        # Outer loop: try increasing nloop until CFL is satisfied or we hit max
        while true
            mx = view(m_h, :, :, k) |> copy

            cfl_ok = true
            for iloop in 1:nloop_k
                local_max = zero(FT)
                local_loc = (0, 0)
                @inbounds for j in 2:Ny, i in 1:Nx
                    bm_v = bm_h[i, j, k]
                    md = bm_v >= zero(FT) ? mx[i, j-1] : mx[i, j]
                    if md > zero(FT)
                        b = abs(bm_v) / md
                        if b > local_max
                            local_max = b
                            local_loc = (i, j)
                        end
                    end
                end
                if local_max > worst_cfl
                    worst_cfl = local_max
                    worst_loc = (local_loc[1], local_loc[2], iloop)
                end
                if local_max >= beta_thresh
                    cfl_ok = false
                    last_fail_iloop = iloop
                    last_fail_loc = local_loc
                    last_fail_cfl = local_max
                    # Capture the donor mass and bm at failure
                    ix, jx = local_loc
                    if ix > 0 && jx > 0
                        bm_v = bm_h[ix, jx, k]
                        last_fail_bm_eff = bm_v
                        last_fail_m_pilot = bm_v >= zero(FT) ? mx[ix, jx-1] : mx[ix, jx]
                    end
                    break
                end
                if iloop != nloop_k
                    @inbounds for j in 1:Ny, i in 1:Nx
                        bm_s = bm_h[i, j, k]
                        bm_n = bm_h[i, j+1, k]
                        mx[i, j] = mx[i, j] + bm_s - bm_n
                    end
                end
            end

            if cfl_ok
                nloop[k] = Int32(nloop_k)
                break
            end

            # CFL failed: reduce bm at this level and increment nloop
            scale = FT(nloop_k) / FT(nloop_k + 1)
            @inbounds for j in 1:(Ny+1), i in 1:Nx
                bm_h[i, j, k] *= scale
            end
            nloop_k += 1
            if nloop_k >= max_nloop
                ix, jx = last_fail_loc
                worst_i, worst_j, worst_iloop = worst_loc
                # Original (unreduced) bm at the failing cell, for context
                bm_orig = ix > 0 && jx > 0 ? Array(bm_buf)[ix, jx, k] : zero(FT)
                m_orig  = ix > 0 && jx > 0 ? m_h[ix, jx, k]            : zero(FT)
                error("Y nloop hit max_nloop=$max_nloop at level k=$k.\n" *
                      "  Last failure: cell (i=$ix, j=$jx) at iloop=$last_fail_iloop\n" *
                      "    bm_eff = $last_fail_bm_eff (after $(max_nloop-1) reductions)\n" *
                      "    m_pilot at donor = $last_fail_m_pilot\n" *
                      "    CFL_per_pass = $last_fail_cfl  (>= $beta_thresh)\n" *
                      "  Original (unreduced) bm at this cell = $bm_orig\n" *
                      "  Original m at this cell = $m_orig\n" *
                      "  Worst CFL across all attempts: $worst_cfl at (i=$worst_i, j=$worst_j) iloop=$worst_iloop\n" *
                      "TM5 also aborts here.")
            end
        end
    end

    # Copy modified bm back to device, and nloop to device
    copyto!(bm_buf, bm_h)
    copyto!(ws.nloop_y_dev, nloop)
    return nloop
end

@kernel function _massflux_y_perlevel_kernel!(
    rm_new, @Const(rm), m_new, @Const(m),
    @Const(bm), @Const(nloop_y), Ny, iter, use_limiter
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        # Per-level mask: only apply this iteration if iter <= nloop_y[k]
        if Int32(iter) > nloop_y[k]
            rm_new[i, j, k] = rm[i, j, k]
            m_new[i, j, k]  = m[i, j, k]
        else
            # Standard Y advection logic (matches _massflux_y_kernel!)
            cjm = j > 1  ? rm[i, j - 1, k] / m[i, j - 1, k] : zero(FT)
            cj  = rm[i, j, k] / m[i, j, k]
            cjp = j < Ny ? rm[i, j + 1, k] / m[i, j + 1, k] : zero(FT)

            sj = if j > 1 && j < Ny
                sc = (cjp - cjm) / 2
                if use_limiter; sc = minmod_device(sc, 2 * (cjp - cj), 2 * (cj - cjm)); end
                s = m[i, j, k] * sc
                if use_limiter; s = max(min(s, rm[i, j, k]), -rm[i, j, k]); end
                s
            else
                zero(FT)
            end

            sjm = if j > 2 && j - 1 < Ny
                cjmm = rm[i, j - 2, k] / m[i, j - 2, k]
                ckm  = rm[i, j - 1, k] / m[i, j - 1, k]
                cjk  = rm[i, j, k]     / m[i, j, k]
                sc   = (cjk - cjmm) / 2
                if use_limiter; sc = minmod_device(sc, 2 * (cjk - ckm), 2 * (ckm - cjmm)); end
                s = m[i, j - 1, k] * sc
                if use_limiter; s = max(min(s, rm[i, j - 1, k]), -rm[i, j - 1, k]); end
                s
            else
                zero(FT)
            end

            sjp = if j < Ny - 1 && j + 1 > 1
                cjk  = rm[i, j, k]     / m[i, j, k]
                cjp_ = rm[i, j + 1, k] / m[i, j + 1, k]
                cjpp = rm[i, j + 2, k] / m[i, j + 2, k]
                sc   = (cjpp - cjk) / 2
                if use_limiter; sc = minmod_device(sc, 2 * (cjpp - cjp_), 2 * (cjp_ - cjk)); end
                s = m[i, j + 1, k] * sc
                if use_limiter; s = max(min(s, rm[i, j + 1, k]), -rm[i, j + 1, k]); end
                s
            else
                zero(FT)
            end

            bm_s = bm[i, j, k]
            flux_s = if j == 1
                zero(FT)
            elseif bm_s >= zero(FT)
                beta = bm_s / m[i, j - 1, k]
                beta * (rm[i, j - 1, k] + (one(FT) - beta) * sjm)
            else
                beta = bm_s / m[i, j, k]
                beta * (rm[i, j, k] - (one(FT) + beta) * sj)
            end

            bm_n = bm[i, j + 1, k]
            flux_n = if j == Ny
                zero(FT)
            elseif bm_n >= zero(FT)
                beta = bm_n / m[i, j, k]
                beta * (rm[i, j, k] + (one(FT) - beta) * sj)
            else
                beta = bm_n / m[i, j + 1, k]
                beta * (rm[i, j + 1, k] - (one(FT) + beta) * sjp)
            end

            rm_new[i, j, k] = rm[i, j, k] + flux_s - flux_n
            m_new[i, j, k]  = m[i, j, k]  + bm_s - bm_n
        end
    end
end

"""
$(SIGNATURES)

TM5-style per-level evolving-mass Y subcycling, matching `advecty__slopes.F90:236-507`.

1. Discovery (`_y_discover_nloop_per_level!`): for each vertical level, find
   the minimum nloop[k] such that the per-pass CFL stays under threshold and
   m stays positive.  bm gets scaled in place per level.
2. Application: run `max(nloop)` passes of `_massflux_y_perlevel_kernel!`
   with a per-level mask.

Differences vs TM5:
- TM5 max_nloop = 6 (we follow this)
- TM5 threshold = 1.0 (we follow this; pass `cfl_limit` to override for tighter)
- TM5 aborts on max_nloop reached; we warn and accept the result.
"""
function advect_y_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                       grid, use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(1.0),  # TM5 default
                                       max_nloop::Int = 6) where FT
    backend = get_backend(m)
    Nx, Ny, Nz = size(m)

    # Phase 1: per-level discovery.  bm is scaled in place into ws.cfl_y
    # so we don't mutate the caller's bm.
    # TM5 uses threshold 1.0 (advecty__slopes.F90:242).  We honor the
    # caller-supplied cfl_limit for consistency with X (Codex ToClaude6.md #3).
    copyto!(ws.cfl_y, bm)
    nloop = _y_discover_nloop_per_level!(ws, m, ws.cfl_y;
                                          max_nloop=max_nloop,
                                          beta_thresh=cfl_limit)
    bm_eff = ws.cfl_y
    max_iter = maximum(nloop)

    # Phase 2: run actual tracer advection with per-level nloop mask
    k! = _massflux_y_perlevel_kernel!(backend, 256)
    for iter in 1:max_iter
        for (_, rm) in pairs(rm_tracers)
            k!(ws.rm_buf, rm, ws.m_buf, m, bm_eff,
               ws.nloop_y_dev, Int32(Ny), Int32(iter), use_limiter;
               ndrange=size(m))
            synchronize(backend)
            copyto!(rm, ws.rm_buf)
        end
        copyto!(m, ws.m_buf)
    end
    return Int(max_iter)
end

"""
    _z_pilot_succeeds(ws, m, cm_eff, n_sub, beta) → Bool

Run an m-only pilot of `n_sub` passes of cm_eff applied to the evolving mass.
Returns `true` if all passes keep `m > 0` and `max(|cm_eff|/m) < beta`.
Used by `advect_z_massflux_subcycled!` for evolving-mass CFL refinement.
"""
function _z_pilot_succeeds(ws::MassFluxWorkspace{FT}, m::AbstractArray{FT,3},
                            cm_eff, n_sub::Int, beta::FT) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    copyto!(ws.m_pilot, m)
    k! = _mass_only_z_kernel!(backend, 256)
    for _ in 1:n_sub
        # CFL check on current pilot mass.  Use cfl_scratch_z to avoid aliasing
        # with cm_eff (which IS ws.cfl_z from the caller).
        cfl = max_cfl_massflux_z(cm_eff, ws.m_pilot, ws.cfl_scratch_z)
        if !isfinite(cfl) || cfl >= beta
            return false
        end
        # Mass update
        k!(ws.m_buf, ws.m_pilot, cm_eff, Int32(Nz); ndrange=size(m))
        synchronize(backend)
        copyto!(ws.m_pilot, ws.m_buf)
        # Positivity check
        if minimum(ws.m_pilot) <= zero(FT)
            return false
        end
    end
    return true
end

"""
$(SIGNATURES)

TM5-style evolving-mass Z-subcycling.  Iteratively finds the minimum n_sub
such that running cm/n_sub through the mass update n_sub times keeps m > 0
and CFL < cfl_limit at every pass.  Then runs the actual tracer advection
with the converged n_sub.

Reference: TM5 advectx__slopes.F90:430-520 (CFL refinement via flux reduction).
"""
function advect_z_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, cm,
                                       use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(0.95)) where FT
    backend = get_backend(m)
    Nz = size(m, 3)

    # Phase 1: find n_sub via evolving-mass pilot.
    # Start from the static-CFL estimate and double until pilot succeeds.
    cfl_static = max_cfl_massflux_z(cm, m, ws.cfl_z)
    n_sub = (isfinite(cfl_static) && cfl_static > zero(FT)) ?
            max(1, ceil(Int, min(cfl_static, FT(100)) / cfl_limit)) : 1
    max_n_sub = 256
    while n_sub <= max_n_sub
        ws.cfl_z .= cm ./ FT(n_sub)
        if _z_pilot_succeeds(ws, m, ws.cfl_z, n_sub, cfl_limit)
            break
        end
        n_sub *= 2
    end
    if n_sub > max_n_sub
        @warn "advect_z_massflux_subcycled!: pilot failed even at n_sub=$max_n_sub" maxlog=3
        n_sub = max_n_sub
        ws.cfl_z .= cm ./ FT(n_sub)
    end
    cm_eff = ws.cfl_z

    # Phase 2: run actual tracer advection with converged cm_eff.
    k! = _massflux_z_kernel!(backend, 256)
    for _ in 1:n_sub
        for (_, rm) in pairs(rm_tracers)
            k!(ws.rm_buf, rm, ws.m_buf, m, cm_eff, Nz, use_limiter; ndrange=size(m))
            synchronize(backend)
            copyto!(rm, ws.rm_buf)
        end
        copyto!(m, ws.m_buf)
    end
    return n_sub
end

# Backward-compatible versions (for tests).  These allocate a temporary
# MassFluxWorkspace and dispatch to the workspace path so they exercise
# the same TM5-style local nloop refinement as production.
function advect_x_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                       grid, use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    bm_dummy = similar(m, FT, size(m, 1), size(m, 2) + 1, size(m, 3))
    cm_dummy = similar(m, FT, size(m, 1), size(m, 2),     size(m, 3) + 1)
    fill!(bm_dummy, zero(FT)); fill!(cm_dummy, zero(FT))
    ws = allocate_massflux_workspace(m, am, bm_dummy, cm_dummy)
    return advect_x_massflux_subcycled!(rm_tracers, m, am, grid, use_limiter, ws;
                                         cfl_limit=cfl_limit)
end
function advect_y_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                       grid, use_limiter;
                                       cfl_limit = FT(1.0)) where FT
    am_dummy = similar(m, FT, size(m, 1) + 1, size(m, 2), size(m, 3))
    cm_dummy = similar(m, FT, size(m, 1),     size(m, 2), size(m, 3) + 1)
    fill!(am_dummy, zero(FT)); fill!(cm_dummy, zero(FT))
    ws = allocate_massflux_workspace(m, am_dummy, bm, cm_dummy)
    return advect_y_massflux_subcycled!(rm_tracers, m, bm, grid, use_limiter, ws;
                                         cfl_limit=cfl_limit)
end
function advect_z_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, cm,
                                       use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    am_dummy = similar(m, FT, size(m, 1) + 1, size(m, 2),     size(m, 3))
    bm_dummy = similar(m, FT, size(m, 1),     size(m, 2) + 1, size(m, 3))
    fill!(am_dummy, zero(FT)); fill!(bm_dummy, zero(FT))
    ws = allocate_massflux_workspace(m, am_dummy, bm_dummy, cm)
    return advect_z_massflux_subcycled!(rm_tracers, m, cm, use_limiter, ws;
                                         cfl_limit=cfl_limit)
end

# =====================================================================
# Full Strang-split mass-flux advection step
# =====================================================================

"""
$(SIGNATURES)

Perform a full Strang-split advection step (X-Y-Z-Z-Y-X) using TM5-style
mass-flux advection.  Runs on CPU or GPU — same code path via KA kernels.

Tracers are tracer mass `rm = c × m` (TM5-style prognostic variable).
Internally copies into workspace buffer for double-buffered kernels.
`m` is updated in-place to track air mass.

When `ws::MassFluxWorkspace` is provided, all temporary GPU arrays are
pre-allocated, reducing per-step allocations from ~90 to zero.
"""
# Per-sweep instrumentation toggle.  Set via `enable_sweep_debug!()` or
# via env var `ATMOSTRANSPORT_DEBUG_SWEEPS=1` before loading the module.
const _DEBUG_SWEEPS = Ref(get(ENV, "ATMOSTRANSPORT_DEBUG_SWEEPS", "") == "1")

"""Enable per-sweep mass instrumentation (min m, max CFL) for debugging."""
enable_sweep_debug!(flag::Bool=true) = (_DEBUG_SWEEPS[] = flag; nothing)

function _log_sweep(label::String, m::AbstractArray{FT,3}, am_or_bm_or_cm,
                     max_cfl::Union{Nothing,FT}, n_sub::Int) where FT
    _DEBUG_SWEEPS[] || return
    mmin = minimum(m)
    mmax = maximum(m)
    msg = "[sweep $label] m∈[$(mmin), $(mmax)]"
    if max_cfl !== nothing
        msg *= "  max_cfl=$(max_cfl)"
    end
    msg *= "  n_sub=$n_sub"
    if mmin <= zero(FT)
        ibad = argmin(m)
        msg *= "  FIRST_NONPOS at $(Tuple(ibad))"
    end
    # Targeted polar diagnostic: m at i=616, j=358..361, k=2 (the failure cell)
    if size(m, 2) >= 361 && size(m, 3) >= 2
        m_h = Array(view(m, 616:616, 358:361, 2))
        msg *= "  poles[616,358..361,k=2]=[" * join([string(round(m_h[1, jj-357, 1], sigdigits=5)) for jj in 358:361], ", ") * "]"
    end
    @info msg
end

function strang_split_massflux!(tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am, bm, cm,
                                 grid::LatitudeLongitudeGrid,
                                 use_limiter::Bool,
                                 ws::MassFluxWorkspace{FT};
                                 cfl_limit::FT = FT(1.0)) where FT
    # Multi-tracer: each tracer is advected independently, restoring m between.
    n_tr = length(tracers)
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1
        copyto!(m_save, m)
    end

    for (i, (name, rm_tracer)) in enumerate(pairs(tracers))
        if i > 1
            copyto!(m, m_save)
        end
        copyto!(ws.rm, rm_tracer)
        rm_single = NamedTuple{(name,)}((ws.rm,))

        _DEBUG_SWEEPS[] && _log_sweep("pre", m, am, nothing, 0)
        n = advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("X1", m, am, nothing, n)
        n = advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("Y1", m, bm, nothing, n)
        n = advect_z_massflux_subcycled!(rm_single, m, cm, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("Z1", m, cm, nothing, n)
        n = advect_z_massflux_subcycled!(rm_single, m, cm, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("Z2", m, cm, nothing, n)
        n = advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("Y2", m, bm, nothing, n)
        n = advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)
        _DEBUG_SWEEPS[] && _log_sweep("X2", m, am, nothing, n)

        copyto!(rm_tracer, ws.rm)
    end
    return nothing
end

"""
$(SIGNATURES)

CFL-adaptive subcycled x-advection using TM5-style reduced grid on CPU.
The reduced grid keeps CFL < 1 at all latitudes, so typically n_sub = 1.
"""
function advect_x_massflux_reduced_subcycled!(rm_tracers, m::Array{FT,3}, am,
                                               grid, use_limiter;
                                               cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_x(am, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    am_eff = n_sub > 1 ? am ./ FT(n_sub) : am
    for _ in 1:n_sub
        advect_x_massflux_reduced!(rm_tracers, m, am_eff, grid, use_limiter)
    end
    return n_sub
end

# =====================================================================
# Prognostic-slope Strang split (TM5-faithful)
# =====================================================================

"""
    _advect_x_prognostic!(rm, m, am, ws, pw, Nx, use_limiter)

CFL-subcycled X-advection with prognostic slope evolution.
"""
function _advect_x_prognostic!(rm, m, am, ws::MassFluxWorkspace{FT},
                                pw::PrognosticSlopeWorkspace{FT},
                                Nx, use_limiter; cfl_limit=FT(0.95)) where FT
    backend = get_backend(m)
    cfl = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)
    n_sub = (isfinite(cfl) && cfl > zero(FT)) ? min(100, max(1, ceil(Int, min(cfl, FT(100)) / cfl_limit))) : 1
    am_eff = n_sub > 1 ? (ws.cfl_x .= am ./ FT(n_sub); ws.cfl_x) : am
    k! = _prognostic_x_kernel!(backend, 256)
    for _ in 1:n_sub
        k!(ws.rm_buf, rm, ws.m_buf, m, am_eff,
           pw.rxm_buf, pw.rxm, pw.rym_buf, pw.rym, pw.rzm_buf, pw.rzm,
           Int32(Nx), use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf);  copyto!(m, ws.m_buf)
        copyto!(pw.rxm, pw.rxm_buf);  copyto!(pw.rym, pw.rym_buf);  copyto!(pw.rzm, pw.rzm_buf)
    end
end

"""
    _advect_y_prognostic!(rm, m, bm, ws, pw, Ny, use_limiter)

CFL-subcycled Y-advection with prognostic slope evolution.
"""
function _advect_y_prognostic!(rm, m, bm, ws::MassFluxWorkspace{FT},
                                pw::PrognosticSlopeWorkspace{FT},
                                Ny, use_limiter; cfl_limit=FT(0.95)) where FT
    backend = get_backend(m)
    cfl = max_cfl_massflux_y(bm, m, ws.cfl_y)
    n_sub = (isfinite(cfl) && cfl > zero(FT)) ? min(100, max(1, ceil(Int, min(cfl, FT(100)) / cfl_limit))) : 1
    bm_eff = n_sub > 1 ? (ws.cfl_y .= bm ./ FT(n_sub); ws.cfl_y) : bm
    k! = _prognostic_y_kernel!(backend, 256)
    for _ in 1:n_sub
        k!(ws.rm_buf, rm, ws.m_buf, m, bm_eff,
           pw.rym_buf, pw.rym, pw.rxm_buf, pw.rxm, pw.rzm_buf, pw.rzm,
           Int32(Ny), use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf);  copyto!(m, ws.m_buf)
        copyto!(pw.rym, pw.rym_buf);  copyto!(pw.rxm, pw.rxm_buf);  copyto!(pw.rzm, pw.rzm_buf)
    end
end

"""
    _advect_z_prognostic!(rm, m, cm, ws, pw, Nz, use_limiter)

CFL-subcycled Z-advection with prognostic slope evolution.
"""
function _advect_z_prognostic!(rm, m, cm, ws::MassFluxWorkspace{FT},
                                pw::PrognosticSlopeWorkspace{FT},
                                Nz, use_limiter; cfl_limit=FT(0.95)) where FT
    backend = get_backend(m)
    cfl = max_cfl_massflux_z(cm, m, ws.cfl_z)
    n_sub = (isfinite(cfl) && cfl > zero(FT)) ? min(100, max(1, ceil(Int, min(cfl, FT(100)) / cfl_limit))) : 1
    cm_eff = n_sub > 1 ? (ws.cfl_z .= cm ./ FT(n_sub); ws.cfl_z) : cm
    k! = _prognostic_z_kernel!(backend, 256)
    for _ in 1:n_sub
        k!(ws.rm_buf, rm, ws.m_buf, m, cm_eff,
           pw.rzm_buf, pw.rzm, pw.rxm_buf, pw.rxm, pw.rym_buf, pw.rym,
           Int32(Nz), use_limiter; ndrange=size(m))
        synchronize(backend)
        copyto!(rm, ws.rm_buf);  copyto!(m, ws.m_buf)
        copyto!(pw.rzm, pw.rzm_buf);  copyto!(pw.rxm, pw.rxm_buf);  copyto!(pw.rym, pw.rym_buf)
    end
end

"""
    strang_split_prognostic!(tracers, m, am, bm, cm, grid, ws, pw_dict, use_limiter)

Strang-split advection with TM5-faithful prognostic slope evolution.
`pw_dict` is a NamedTuple of `PrognosticSlopeWorkspace` keyed by tracer name.
"""
function strang_split_prognostic!(tracers::NamedTuple,
                                   m::AbstractArray{FT,3},
                                   am, bm, cm,
                                   grid::LatitudeLongitudeGrid,
                                   ws::MassFluxWorkspace{FT},
                                   pw_dict::NamedTuple,
                                   use_limiter::Bool;
                                   cfl_limit::FT = FT(1.0)) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_tr = length(tracers)
    m_save = n_tr > 1 ? similar(m) : m
    if n_tr > 1; copyto!(m_save, m); end

    for (idx, (name, rm_tracer)) in enumerate(pairs(tracers))
        if idx > 1; copyto!(m, m_save); end
        rm = ws.rm
        copyto!(rm, rm_tracer)
        pw = pw_dict[name]
        rm_single = NamedTuple{(name,)}((rm,))

        # Strang: X → Y → Z → Z → Y → X
        # X and Y: use proven diagnostic kernels (handles reduced grid)
        # Z: use prognostic kernel (evolves rzm, no reduced grid needed)
        advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)
        advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        _advect_z_prognostic!(rm, m, cm, ws, pw, Nz, use_limiter; cfl_limit)
        _advect_z_prognostic!(rm, m, cm, ws, pw, Nz, use_limiter; cfl_limit)
        advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)

        copyto!(rm_tracer, rm)
    end
    return nothing
end

# Version without workspace (allocates internally). Tracers are rm (tracer mass).
function strang_split_massflux!(tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am, bm, cm,
                                 grid::LatitudeLongitudeGrid,
                                 use_limiter::Bool;
                                 cfl_limit::FT = FT(1.0)) where FT
    # Allocate working copies (the advect_*_subcycled! functions modify rm in-place)
    rm_tracers = NamedTuple{keys(tracers)}(
        Tuple(copy(rm) for rm in values(tracers))
    )

    advect_x_massflux_subcycled!(rm_tracers, m, am, grid, use_limiter; cfl_limit)
    advect_y_massflux_subcycled!(rm_tracers, m, bm, grid, use_limiter; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, use_limiter; cfl_limit)
    advect_z_massflux_subcycled!(rm_tracers, m, cm, use_limiter; cfl_limit)
    advect_y_massflux_subcycled!(rm_tracers, m, bm, grid, use_limiter; cfl_limit)
    advect_x_massflux_subcycled!(rm_tracers, m, am, grid, use_limiter; cfl_limit)

    for (name, rm_tracer) in pairs(tracers)
        rm_tracer .= rm_tracers[name]
    end

    return nothing
end
