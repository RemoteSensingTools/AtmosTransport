using KernelAbstractions: @kernel, @index

"""
    FirstOrderUpwindAdvection <: AbstractAdvection

First-order upwind advection. This is the simplest conservative operator and
serves as the reference implementation for the generic `src_v2` core.
"""
struct FirstOrderUpwindAdvection <: AbstractAdvection end

@inline function _upwind_face_flux(face_flux, c_left, c_right)
    return ifelse(face_flux >= zero(face_flux), face_flux * c_left, face_flux * c_right)
end

@kernel function _upwind_x_kernel!(rm_new, @Const(rm), m_new, @Const(m), @Const(am),
                                   Nx, @Const(cluster_sizes))
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_floor = eps(FT)
    r = Int(cluster_sizes[j])
    @inbounds begin
        if r == 1
            ip = ifelse(i == Nx, 1, i + 1)
            im = ifelse(i == 1, Nx, i - 1)

            c_im = rm[im, j, k] / max(m[im, j, k], m_floor)
            c_i  = rm[i,  j, k] / max(m[i,  j, k], m_floor)
            c_ip = rm[ip, j, k] / max(m[ip, j, k], m_floor)

            flux_left  = _upwind_face_flux(am[i, j, k],     c_im, c_i)
            flux_right = _upwind_face_flux(am[i + 1, j, k], c_i,  c_ip)

            rm_new[i, j, k] = rm[i, j, k] + flux_left - flux_right
            m_new[i, j, k]  = m[i, j, k]  + am[i, j, k] - am[i + 1, j, k]
        else
            Nx_red = Nx ÷ r
            ic     = (i - 1) ÷ r + 1
            ic_m   = ifelse(ic == 1, Nx_red, ic - 1)
            ic_p   = ifelse(ic == Nx_red, 1, ic + 1)

            rm_ic = _cluster_sum(rm, ic, j, k, r)
            m_ic  = _cluster_sum(m,  ic, j, k, r)
            rm_im = _cluster_sum(rm, ic_m, j, k, r)
            m_im  = _cluster_sum(m,  ic_m, j, k, r)
            rm_ip = _cluster_sum(rm, ic_p, j, k, r)
            m_ip  = _cluster_sum(m,  ic_p, j, k, r)

            c_im = rm_im / max(m_im, m_floor)
            c_i  = rm_ic / max(m_ic, m_floor)
            c_ip = rm_ip / max(m_ip, m_floor)

            am_l = am[(ic - 1) * r + 1, j, k]
            am_r_idx = ic * r + 1
            am_r = am[ifelse(am_r_idx > Nx, 1, am_r_idx), j, k]

            flux_left  = _upwind_face_flux(am_l, c_im, c_i)
            flux_right = _upwind_face_flux(am_r, c_i, c_ip)

            δ_rm = flux_left - flux_right
            δ_m  = am_l - am_r
            frac_m = ifelse(abs(m_ic) > eps(FT), m[i, j, k] / m_ic, one(FT) / FT(r))
            rm_new[i, j, k] = (rm_ic + δ_rm) * frac_m
            m_new[i, j, k]  = (m_ic  + δ_m)  * frac_m
        end
    end
end

@kernel function _upwind_y_kernel!(rm_new, @Const(rm), m_new, @Const(m), @Const(bm), Ny)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_floor = eps(FT)
    @inbounds begin
        flux_s = ifelse(j > 1,
                        _upwind_face_flux(bm[i, j, k],
                                          rm[i, j - 1, k] / max(m[i, j - 1, k], m_floor),
                                          rm[i, j,     k] / max(m[i, j,     k], m_floor)),
                        zero(FT))

        flux_n = ifelse(j < Ny,
                        _upwind_face_flux(bm[i, j + 1, k],
                                          rm[i, j,     k] / max(m[i, j,     k], m_floor),
                                          rm[i, j + 1, k] / max(m[i, j + 1, k], m_floor)),
                        zero(FT))

        rm_new[i, j, k] = rm[i, j, k] + flux_s - flux_n
        m_new[i, j, k]  = m[i, j, k]  + bm[i, j, k] - bm[i, j + 1, k]
    end
end

@kernel function _upwind_z_kernel!(rm_new, @Const(rm), m_new, @Const(m), @Const(cm), Nz)
    i, j, k = @index(Global, NTuple)
    FT = eltype(rm)
    m_floor = eps(FT)
    @inbounds begin
        flux_t = ifelse(k > 1,
                        _upwind_face_flux(cm[i, j, k],
                                          rm[i, j, k - 1] / max(m[i, j, k - 1], m_floor),
                                          rm[i, j, k    ] / max(m[i, j, k    ], m_floor)),
                        zero(FT))

        flux_b = ifelse(k < Nz,
                        _upwind_face_flux(cm[i, j, k + 1],
                                          rm[i, j, k    ] / max(m[i, j, k    ], m_floor),
                                          rm[i, j, k + 1] / max(m[i, j, k + 1], m_floor)),
                        zero(FT))

        rm_new[i, j, k] = rm[i, j, k] + flux_t - flux_b
        m_new[i, j, k]  = m[i, j, k]  + cm[i, j, k] - cm[i, j, k + 1]
    end
end

export FirstOrderUpwindAdvection
