using KernelAbstractions: @kernel, @index, synchronize, get_backend

"""
$(SIGNATURES)

Second-order van Leer slopes advection (TM5 `advectx__slopes`, `advecty__slopes`).
"""
struct RussellLernerAdvection <: AbstractLinearReconstruction
    use_limiter :: Bool
end

RussellLernerAdvection(; use_limiter::Bool = true) = RussellLernerAdvection(use_limiter)

# ---------------------------------------------------------------------------
# Device-safe helpers — all @inline, all branchless via ifelse
# ---------------------------------------------------------------------------

@inline function _minmod(a, b, c)
    all_positive = (a > zero(a)) & (b > zero(b)) & (c > zero(c))
    all_negative = (a < zero(a)) & (b < zero(b)) & (c < zero(c))
    return ifelse(all_positive, min(a, b, c),
           ifelse(all_negative, max(a, b, c), zero(a)))
end

@inline function _limited_slope(sc, c_left, c_center, c_right, use_limiter)
    return ifelse(use_limiter,
                  _minmod(sc, convert(typeof(sc), 2) * (c_right - c_center),
                              convert(typeof(sc), 2) * (c_center - c_left)),
                  sc)
end

@inline function _limited_moment(sx, rm_cell, use_limiter)
    return ifelse(use_limiter, max(min(sx, rm_cell), -rm_cell), sx)
end

@inline function _upwind_flux(flux_face, m_donor_pos, rm_donor_pos, sx_donor_pos,
                              m_donor_neg, rm_donor_neg, sx_donor_neg)
    FT = eltype(flux_face)
    α_pos = flux_face / m_donor_pos
    α_neg = flux_face / m_donor_neg
    f_pos = α_pos * (rm_donor_pos + (one(FT) - α_pos) * sx_donor_pos)
    f_neg = α_neg * (rm_donor_neg - (one(FT) + α_neg) * sx_donor_neg)
    return ifelse(flux_face >= zero(FT), f_pos, f_neg)
end

@inline function _cluster_sum(arr, ic::Int, j::Int, k::Int, r::Int)
    T = eltype(arr)
    s = zero(T)
    i_start = (ic - 1) * r + 1
    for off in 0:r-1
        s += arr[i_start + off, j, k]
    end
    return s
end

# ---------------------------------------------------------------------------
# X-advection kernel — one thread per (i, j, k)
# ---------------------------------------------------------------------------

@kernel function _rl_x_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am),
    Nx, @Const(cluster_sizes), use_limiter, flux_scale
)
    i, j, k = @index(Global, NTuple)
    r = Int(cluster_sizes[j])
    FT = eltype(rm)
    two = convert(FT, 2)
    m_floor = eps(FT)  # prevent NaN if cell mass → 0 (consistent with z-kernel line 226)
    @inbounds begin
        if r == 1
            ip  = ifelse(i == Nx, 1, i + 1)
            im  = ifelse(i == 1,  Nx, i - 1)
            ipp = ifelse(ip == Nx, 1, ip + 1)
            imm = ifelse(im == 1,  Nx, im - 1)

            c_imm = rm[imm, j, k] / max(m[imm, j, k], m_floor)
            c_im  = rm[im,  j, k] / max(m[im,  j, k], m_floor)
            c_i   = rm[i,   j, k] / max(m[i,   j, k], m_floor)
            c_ip  = rm[ip,  j, k] / max(m[ip,  j, k], m_floor)
            c_ipp = rm[ipp, j, k] / max(m[ipp, j, k], m_floor)

            sc_im = _limited_slope((c_i - c_imm) / two, c_imm, c_im, c_i, use_limiter)
            sx_im = _limited_moment(m[im, j, k] * sc_im, rm[im, j, k], use_limiter)

            sc_i = _limited_slope((c_ip - c_im) / two, c_im, c_i, c_ip, use_limiter)
            sx_i = _limited_moment(m[i, j, k] * sc_i, rm[i, j, k], use_limiter)

            sc_ip = _limited_slope((c_ipp - c_i) / two, c_i, c_ip, c_ipp, use_limiter)
            sx_ip = _limited_moment(m[ip, j, k] * sc_ip, rm[ip, j, k], use_limiter)

            am_l = flux_scale * am[i, j, k]
            am_r = flux_scale * am[i + 1, j, k]
            flux_left  = _upwind_flux(am_l,
                                      m[im, j, k], rm[im, j, k], sx_im,
                                      m[i, j, k],  rm[i, j, k],  sx_i)
            flux_right = _upwind_flux(am_r,
                                      m[i, j, k],  rm[i, j, k],  sx_i,
                                      m[ip, j, k], rm[ip, j, k], sx_ip)

            rm_new[i, j, k] = rm[i, j, k] + flux_left - flux_right
            m_new[i, j, k]  = m[i, j, k]  + am_l - am_r
        else
            Nx_red = Nx ÷ r
            ic     = (i - 1) ÷ r + 1
            ic_m   = ifelse(ic == 1,      Nx_red, ic - 1)
            ic_p   = ifelse(ic == Nx_red, 1,      ic + 1)
            ic_mm  = ifelse(ic_m == 1,      Nx_red, ic_m - 1)
            ic_pp  = ifelse(ic_p == Nx_red, 1,      ic_p + 1)

            rm_ic  = _cluster_sum(rm, ic,    j, k, r)
            m_ic   = _cluster_sum(m,  ic,    j, k, r)
            rm_im  = _cluster_sum(rm, ic_m,  j, k, r)
            m_im   = _cluster_sum(m,  ic_m,  j, k, r)
            rm_ip  = _cluster_sum(rm, ic_p,  j, k, r)
            m_ip   = _cluster_sum(m,  ic_p,  j, k, r)
            rm_imm = _cluster_sum(rm, ic_mm, j, k, r)
            m_imm  = _cluster_sum(m,  ic_mm, j, k, r)
            rm_ipp = _cluster_sum(rm, ic_pp, j, k, r)
            m_ipp  = _cluster_sum(m,  ic_pp, j, k, r)

            c_imm_r = rm_imm / m_imm
            c_im_r  = rm_im  / m_im
            c_ic_r  = rm_ic  / m_ic
            c_ip_r  = rm_ip  / m_ip
            c_ipp_r = rm_ipp / m_ipp

            sc_im_r = _limited_slope((c_ic_r - c_imm_r) / two, c_imm_r, c_im_r, c_ic_r, use_limiter)
            sx_im_r = _limited_moment(m_im * sc_im_r, rm_im, use_limiter)

            sc_ic_r = _limited_slope((c_ip_r - c_im_r) / two, c_im_r, c_ic_r, c_ip_r, use_limiter)
            sx_ic_r = _limited_moment(m_ic * sc_ic_r, rm_ic, use_limiter)

            sc_ip_r = _limited_slope((c_ipp_r - c_ic_r) / two, c_ic_r, c_ip_r, c_ipp_r, use_limiter)
            sx_ip_r = _limited_moment(m_ip * sc_ip_r, rm_ip, use_limiter)

            am_l_r = flux_scale * am[(ic - 1) * r + 1, j, k]
            am_r_idx = ic * r + 1
            am_r_r = flux_scale * am[ifelse(am_r_idx > Nx, 1, am_r_idx), j, k]

            flux_left_r  = _upwind_flux(am_l_r, m_im, rm_im, sx_im_r,
                                                 m_ic, rm_ic, sx_ic_r)
            flux_right_r = _upwind_flux(am_r_r, m_ic, rm_ic, sx_ic_r,
                                                 m_ip, rm_ip, sx_ip_r)

            δ_rm = flux_left_r - flux_right_r
            δ_m  = am_l_r - am_r_r
            frac_m = ifelse(abs(m_ic) > eps(FT), m[i, j, k] / m_ic, one(FT) / FT(r))
            rm_new[i, j, k] = (rm_ic + δ_rm) * frac_m
            m_new[i, j, k]  = (m_ic  + δ_m)  * frac_m
        end
    end
end

# ---------------------------------------------------------------------------
# Y-advection kernel — one thread per (i, j, k)
# ---------------------------------------------------------------------------

@kernel function _rl_y_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm), Ny, use_limiter, flux_scale
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    two = convert(FT, 2)
    m_floor = eps(FT)  # prevent NaN if cell mass → 0 (consistent with z-kernel)
    @inbounds begin
        jm  = max(j - 1, 1)
        jp  = min(j + 1, Ny)
        jmm = max(j - 2, 1)
        jpp = min(j + 2, Ny)

        cjm  = rm[i, jm,  k] / max(m[i, jm,  k], m_floor)
        cj   = rm[i, j,   k] / max(m[i, j,   k], m_floor)
        cjp  = rm[i, jp,  k] / max(m[i, jp,  k], m_floor)
        cjmm = rm[i, jmm, k] / max(m[i, jmm, k], m_floor)
        cjpp = rm[i, jpp, k] / max(m[i, jpp, k], m_floor)

        interior = (j > 1) & (j < Ny)
        sc_j = _limited_slope((cjp - cjm) / two, cjm, cj, cjp, use_limiter)
        sy_j = _limited_moment(m[i, j, k] * sc_j, rm[i, j, k], use_limiter)
        sy_j = ifelse(interior, sy_j, zero(FT))

        interior_jm = (j > 2) & (j - 1 < Ny)
        sc_jm = _limited_slope((cj - cjmm) / two, cjmm, cjm, cj, use_limiter)
        sy_jm = _limited_moment(m[i, jm, k] * sc_jm, rm[i, jm, k], use_limiter)
        sy_jm = ifelse(interior_jm, sy_jm, zero(FT))

        interior_jp = (j < Ny - 1) & (j + 1 > 1)
        sc_jp = _limited_slope((cjpp - cj) / two, cj, cjp, cjpp, use_limiter)
        sy_jp = _limited_moment(m[i, jp, k] * sc_jp, rm[i, jp, k], use_limiter)
        sy_jp = ifelse(interior_jp, sy_jp, zero(FT))

        # South face (j)
        bm_s = flux_scale * bm[i, j, k]
        sy_jm_use = ifelse(jm == 1, zero(FT), sy_jm)
        sy_j_use_s = ifelse(j == Ny, zero(FT), sy_j)
        flux_s = ifelse(j > 1,
                        _upwind_flux(bm_s,
                                     m[i, jm, k], rm[i, jm, k], sy_jm_use,
                                     m[i, j,  k], rm[i, j,  k], sy_j_use_s),
                        zero(FT))

        # North face (j+1)
        bm_n_flux = flux_scale * bm[i, jp, k]
        bm_n_mass = flux_scale * bm[i, j + 1, k]
        sy_j_use_n = ifelse(j == 1, zero(FT), sy_j)
        sy_jp_use = ifelse(jp == Ny, zero(FT), sy_jp)
        flux_n = ifelse(j < Ny,
                        _upwind_flux(bm_n_flux,
                                     m[i, j,  k], rm[i, j,  k], sy_j_use_n,
                                     m[i, jp, k], rm[i, jp, k], sy_jp_use),
                        zero(FT))

        rm_new[i, j, k] = rm[i, j, k] + flux_s - flux_n
        m_new[i, j, k]  = m[i, j, k]  + bm_s - bm_n_mass
    end
end

# ---------------------------------------------------------------------------
# Z-advection kernel — one thread per (i, j, k)
# ---------------------------------------------------------------------------

@kernel function _rl_z_kernel!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(cm), Nz, use_limiter, flux_scale
)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    two = convert(FT, 2)
    m_floor = eps(FT)
    @inbounds begin
        km  = max(k - 1, 1)
        kp  = min(k + 1, Nz)
        kmm = max(k - 2, 1)
        kpp = min(k + 2, Nz)

        ckm  = rm[i, j, km]  / max(m[i, j, km],  m_floor)
        ck   = rm[i, j, k]   / max(m[i, j, k],   m_floor)
        ckp  = rm[i, j, kp]  / max(m[i, j, kp],  m_floor)
        ckmm = rm[i, j, kmm] / max(m[i, j, kmm], m_floor)
        ckpp = rm[i, j, kpp] / max(m[i, j, kpp], m_floor)

        interior = (k > 1) & (k < Nz)
        sc_k = _limited_slope((ckp - ckm) / two, ckm, ck, ckp, use_limiter)
        sz_k = _limited_moment(m[i, j, k] * sc_k, rm[i, j, k], use_limiter)
        sz_k = ifelse(interior, sz_k, zero(FT))

        interior_km = (k > 2) & (k - 1 < Nz)
        sc_km = _limited_slope((ck - ckmm) / two, ckmm, ckm, ck, use_limiter)
        sz_km = _limited_moment(m[i, j, km] * sc_km, rm[i, j, km], use_limiter)
        sz_km = ifelse(interior_km, sz_km, zero(FT))

        interior_kp = (k < Nz - 1) & (k + 1 > 1)
        sc_kp = _limited_slope((ckpp - ck) / two, ck, ckp, ckpp, use_limiter)
        sz_kp = _limited_moment(m[i, j, kp] * sc_kp, rm[i, j, kp], use_limiter)
        sz_kp = ifelse(interior_kp, sz_kp, zero(FT))

        # Top face (k)
        cm_t = flux_scale * cm[i, j, k]
        sz_km_use = ifelse(km == 1, zero(FT), sz_km)
        sz_k_use_t = ifelse(k == Nz, zero(FT), sz_k)
        flux_top = ifelse(k > 1,
                          _upwind_flux(cm_t,
                                       max(m[i, j, km], m_floor), rm[i, j, km], sz_km_use,
                                       max(m[i, j, k],  m_floor), rm[i, j, k],  sz_k_use_t),
                          zero(FT))

        # Bottom face (k+1)
        cm_b = flux_scale * cm[i, j, k + 1]
        sz_k_use_b = ifelse(k == 1, zero(FT), sz_k)
        sz_kp_use = ifelse(kp == Nz, zero(FT), sz_kp)
        flux_bot = ifelse(k < Nz,
                          _upwind_flux(cm_b,
                                       max(m[i, j, k],  m_floor), rm[i, j, k],  sz_k_use_b,
                                       max(m[i, j, kp], m_floor), rm[i, j, kp], sz_kp_use),
                          zero(FT))

        rm_new[i, j, k] = rm[i, j, k] + flux_top - flux_bot
        m_new[i, j, k]  = m[i, j, k]  + cm_t - cm_b
    end
end

export RussellLernerAdvection
