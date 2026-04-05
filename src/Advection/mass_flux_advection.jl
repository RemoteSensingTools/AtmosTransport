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
    @inbounds begin
        # --- Slope at k ---
        sz_k = if k > 1 && k < Nz
            ckm = rm[i, j, k - 1] / m[i, j, k - 1]
            ck  = rm[i, j, k]     / m[i, j, k]
            ckp = rm[i, j, k + 1] / m[i, j, k + 1]
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
            ckmm = rm[i, j, k - 2] / m[i, j, k - 2]
            ckm  = rm[i, j, k - 1] / m[i, j, k - 1]
            ck   = rm[i, j, k]     / m[i, j, k]
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
            ck   = rm[i, j, k]     / m[i, j, k]
            ckp  = rm[i, j, k + 1] / m[i, j, k + 1]
            ckpp = rm[i, j, k + 2] / m[i, j, k + 2]
            sc = (ckpp - ck) / 2
            if use_limiter; sc = minmod_device(sc, 2 * (ckpp - ckp), 2 * (ckp - ck)); end
            s = m[i, j, k + 1] * sc
            if use_limiter; s = max(min(s, rm[i, j, k + 1]), -rm[i, j, k + 1]); end
            s
        else
            zero(FT)
        end

        # --- Flux at top face (face k) ---
        flux_top = if k > 1
            cm_t = cm[i, j, k]
            if cm_t > zero(FT)
                gamma = cm_t / m[i, j, k - 1]
                gamma * (rm[i, j, k - 1] + (one(FT) - gamma) * sz_km)
            elseif cm_t < zero(FT)
                gamma = cm_t / m[i, j, k]
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
                gamma = cm_b / m[i, j, k]
                gamma * (rm[i, j, k] + (one(FT) - gamma) * sz_k)
            elseif cm_b < zero(FT)
                gamma = cm_b / m[i, j, k + 1]
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
# Pole flux clamp — prevent cumulative Y/Z flux from draining pole cells
# =====================================================================

"""
    clamp_fluxes_at_poles!(bm, cm, m, Ny, n_sub)

Clamp bm and cm at pole cells so that the combined cumulative flux over
`2×n_sub` Strang applications stays within cell mass. For each pole cell
(i, j=1 or Ny, k), the outgoing CFL from bm (Y-face) and cm (both Z-faces)
is computed, and if the cumulative total exceeds a safety margin, all fluxes
at that cell are scaled down proportionally.

TM5 avoids this via real-time bm/cm recomputation (dynam0, advect_tools.F90:638)
each substep. This static clamp approximates that feedback for constant-flux paths.

Operates in-place on GPU arrays.
"""
function clamp_fluxes_at_poles!(bm::AbstractArray{FT,3}, cm::AbstractArray{FT,3},
                                 m::AbstractArray{FT,3}, Ny::Int, n_sub::Int) where FT
    backend = get_backend(bm)
    Nx, _, Nz = size(m)
    k! = _clamp_pole_fluxes_kernel!(backend, 256)
    k!(bm, cm, m, Int32(Ny), Int32(Nz), Int32(n_sub); ndrange=(Nx, Nz))
    return nothing
end

@kernel function _clamp_pole_fluxes_kernel!(bm, cm, @Const(m), Ny, Nz, n_sub)
    i, k = @index(Global, NTuple)
    FT = eltype(bm)
    n_apps = FT(2 * n_sub)   # number of Strang applications per window
    safety = FT(0.5)         # leave 50% headroom for Strang interaction
    @inbounds begin
        # --- South pole (j=1) ---
        m_cell = m[i, 1, k]
        # Outgoing fluxes (all face fluxes that can drain this cell):
        #   Y: bm[i,2,k] > 0 drains pole northward (bm[i,1,k]=0 by convention)
        #   Z-above: cm[i,1,k] < 0 drains cell upward (mass flows from k to k-1)
        #   Z-below: cm[i,1,k+1] > 0 drains cell downward (mass flows from k to k+1)
        # Per-app outgoing CFL = sum of outgoing |flux|/m
        bm_out = abs(bm[i, 2, k])
        cm_above_out = k > 1 ? max(-cm[i, 1, k], zero(FT)) : zero(FT)
        cm_below_out = k < Nz ? max(cm[i, 1, k + 1], zero(FT)) : zero(FT)
        total_out = bm_out + cm_above_out + cm_below_out
        # Cumulative outgoing over all applications
        max_out = safety * m_cell / n_apps
        if total_out > max_out && total_out > zero(FT)
            scale = max_out / total_out
            bm[i, 2, k] *= scale
            if k > 1; cm[i, 1, k] *= scale; end
            if k < Nz; cm[i, 1, k + 1] *= scale; end
        end

        # --- North pole (j=Ny) ---
        m_cell_np = m[i, Ny, k]
        bm_out_np = abs(bm[i, Ny, k])
        cm_above_out_np = k > 1 ? max(-cm[i, Ny, k], zero(FT)) : zero(FT)
        cm_below_out_np = k < Nz ? max(cm[i, Ny, k + 1], zero(FT)) : zero(FT)
        total_out_np = bm_out_np + cm_above_out_np + cm_below_out_np
        max_out_np = safety * m_cell_np / n_apps
        if total_out_np > max_out_np && total_out_np > zero(FT)
            scale_np = max_out_np / total_out_np
            bm[i, Ny, k] *= scale_np
            if k > 1; cm[i, Ny, k] *= scale_np; end
            if k < Nz; cm[i, Ny, k + 1] *= scale_np; end
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
struct MassFluxWorkspace{FT, A3 <: AbstractArray{FT,3}, V1 <: AbstractVector{Int32}}
    rm::A3       # tracer mass (Nx, Ny, Nz)
    rm_buf::A3   # advection output buffer for rm (Nx, Ny, Nz)
    m_buf::A3    # advection output buffer for m  (Nx, Ny, Nz)
    cfl_x::A3   # CFL scratch for x (Nx+1, Ny, Nz) — reused as flux_x_eff
    cfl_y::A3   # CFL scratch for y (Nx, Ny+1, Nz) — reused as flux_y_eff
    cfl_z::A3   # CFL scratch for z (Nx, Ny, Nz+1) — reused as flux_z_eff
    cluster_sizes::V1  # per-latitude clustering for reduced grid (length Ny)
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
    Ny = size(m, 2)
    cs_cpu = cluster_sizes_cpu !== nothing ? cluster_sizes_cpu : ones(Int32, Ny)
    cs_dev = similar(m, Int32, Ny)
    copyto!(cs_dev, cs_cpu)
    MassFluxWorkspace{FT, typeof(m), typeof(cs_dev)}(
        similar(m),       # rm
        similar(m),       # rm_buf
        similar(m),       # m_buf
        similar(am),      # cfl_x / flux_x_eff
        similar(bm),      # cfl_y / flux_y_eff
        similar(cm),      # cfl_z / flux_z_eff
        cs_dev,           # cluster_sizes (device)
    )
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
$(SIGNATURES)

Uniform-subdivision X-subcycling with workspace buffers.
"""
function advect_x_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                       grid, use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)
    n_sub = isfinite(cfl) ? min(50, max(1, ceil(Int, cfl / cfl_limit))) : 1
    if n_sub > 1
        ws.cfl_x .= am ./ FT(n_sub)
        am_eff = ws.cfl_x
    else
        am_eff = am
    end
    for _ in 1:n_sub
        advect_x_massflux!(rm_tracers, m, am_eff, grid, use_limiter,
                            ws.rm_buf, ws.m_buf, ws.cluster_sizes)
    end
    return n_sub
end

"""
$(SIGNATURES)

Uniform-subdivision Y-subcycling with workspace buffers.
"""
function advect_y_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                       grid, use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_y(bm, m, ws.cfl_y)
    n_sub = isfinite(cfl) ? min(50, max(1, ceil(Int, cfl / cfl_limit))) : 1
    if n_sub > 1
        ws.cfl_y .= bm ./ FT(n_sub)
        bm_eff = ws.cfl_y
    else
        bm_eff = bm
    end
    for _ in 1:n_sub
        advect_y_massflux!(rm_tracers, m, bm_eff, grid, use_limiter,
                            ws.rm_buf, ws.m_buf)
    end
    return n_sub
end

"""
$(SIGNATURES)

Uniform-subdivision Z-subcycling using the proven double-buffered 3D kernel.
Computes max Z-CFL, divides cm by n_sub, runs n_sub passes of `_massflux_z_kernel!`.

Replaces the flux-remaining approach which diverged with structured IC
(global fraction penalized all cells for worst-case; hit 100-iter cap → NaN).
"""
function advect_z_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, cm,
                                       use_limiter,
                                       ws::MassFluxWorkspace{FT};
                                       cfl_limit = FT(0.95)) where FT
    backend = get_backend(m)
    Nz = size(m, 3)
    cfl = max_cfl_massflux_z(cm, m, ws.cfl_z)
    n_sub = isfinite(cfl) ? min(50, max(1, ceil(Int, cfl / cfl_limit))) : 1
    if n_sub > 1
        ws.cfl_z .= cm ./ FT(n_sub)
        cm_eff = ws.cfl_z
    else
        cm_eff = cm
    end
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

# Backward-compatible versions (allocating, for tests)
function advect_x_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, am,
                                       grid, use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_x(am, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    am_eff = n_sub > 1 ? am ./ FT(n_sub) : am
    for _ in 1:n_sub
        advect_x_massflux!(rm_tracers, m, am_eff, grid, use_limiter)
    end
    return n_sub
end
function advect_y_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, bm,
                                       grid, use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_y(bm, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    bm_eff = n_sub > 1 ? bm ./ FT(n_sub) : bm
    for _ in 1:n_sub
        advect_y_massflux!(rm_tracers, m, bm_eff, grid, use_limiter)
    end
    return n_sub
end
function advect_z_massflux_subcycled!(rm_tracers, m::AbstractArray{FT,3}, cm,
                                       use_limiter;
                                       cfl_limit = FT(0.95)) where FT
    cfl = max_cfl_massflux_z(cm, m)
    n_sub = max(1, ceil(Int, cfl / cfl_limit))
    cm_eff = n_sub > 1 ? cm ./ FT(n_sub) : cm
    for _ in 1:n_sub
        advect_z_massflux!(rm_tracers, m, cm_eff, use_limiter)
    end
    return n_sub
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
function strang_split_massflux!(tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am, bm, cm,
                                 grid::LatitudeLongitudeGrid,
                                 use_limiter::Bool,
                                 ws::MassFluxWorkspace{FT};
                                 cfl_limit::FT = FT(0.95)) where FT
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

        advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)
        advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        # Z-advection with standard CFL subcycling. TM5 allows gamma > 1 because it
        # uses prognostic slopes (rzm) that carry subgrid history. Our diagnostic slopes
        # (recomputed fresh each step via minmod) cannot prevent negative rm when gamma > 1
        # with uniform fields: flux = gamma*rm > rm is mathematically unavoidable.
        # Subcycling with cfl_limit=0.95 ensures gamma < 1 (3 iterations for Z-CFL=2.5).
        # Implementing prognostic slopes (Session 3) would allow removing this constraint.
        advect_z_massflux_subcycled!(rm_single, m, cm, use_limiter, ws; cfl_limit)
        advect_z_massflux_subcycled!(rm_single, m, cm, use_limiter, ws; cfl_limit)
        advect_y_massflux_subcycled!(rm_single, m, bm, grid, use_limiter, ws; cfl_limit)
        advect_x_massflux_subcycled!(rm_single, m, am, grid, use_limiter, ws; cfl_limit)

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

# Version without workspace (allocates internally). Tracers are rm (tracer mass).
function strang_split_massflux!(tracers::NamedTuple,
                                 m::AbstractArray{FT,3},
                                 am, bm, cm,
                                 grid::LatitudeLongitudeGrid,
                                 use_limiter::Bool;
                                 cfl_limit::FT = FT(0.95)) where FT
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
