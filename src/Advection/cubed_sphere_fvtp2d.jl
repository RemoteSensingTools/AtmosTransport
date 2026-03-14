# ---------------------------------------------------------------------------
# Lin-Rood / COSMIC Cross-Term Advection for Cubed-Sphere Grids
#
# Implements FV3's fv_tp_2d algorithm (Putman & Lin 2007, Lin & Rood 1996):
# Both X-then-Y and Y-then-X orderings are computed from the ORIGINAL field,
# and fluxes are averaged. This eliminates the directional splitting error
# that causes wave artifacts at CS panel boundaries.
#
# Reference: tp_core.F90 in FV3/fvdycore, fv_tracer2d.F90 in GCHP
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

struct LinRoodWorkspace{FT, A3h <: AbstractArray{FT,3},
                         A3x <: AbstractArray{FT,3},
                         A3y <: AbstractArray{FT,3}}
    q_buf  :: NTuple{6, A3h}   # pre-advected mixing ratio (haloed)
    fx_in  :: NTuple{6, A3x}   # inner X face values (Nc+1 × Nc × Nz)
    fx_out :: NTuple{6, A3x}   # outer X face values
    fy_in  :: NTuple{6, A3y}   # inner Y face values (Nc × Nc+1 × Nz)
    fy_out :: A3y               # outer Y face values (reused per panel)
end

function LinRoodWorkspace(grid::CubedSphereGrid{FT}) where FT
    (; Nc, Hp, Nz) = grid
    N  = Nc + 2Hp
    AT = array_type(architecture(grid))

    q_buf  = ntuple(_ -> AT(zeros(FT, N, N, Nz)), 6)
    fx_in  = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    fx_out = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    fy_in  = ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6)
    fy_out = AT(zeros(FT, Nc, Nc + 1, Nz))

    return LinRoodWorkspace(q_buf, fx_in, fx_out, fy_in, fy_out)
end

# ---------------------------------------------------------------------------
# Shared PPM helpers (inline, GPU-safe)
# ---------------------------------------------------------------------------

"""Apply ORD=7 discontinuous boundary treatment at panel edges (compile-time eliminated for ORD≠7)."""
@inline function _apply_ord7_boundary(q_L_m, q_R_m, q_L_0, q_R_0,
                                       c_m1, c_m2, c_0, c_p1,
                                       face_idx, Nc, ::Val{ORD}) where ORD
    if ORD == 7
        if face_idx == 1 || face_idx == Nc + 1
            q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_m1, c_m2, c_0, c_p1)
            q_R_m = q_bdy
            q_L_0 = q_bdy
        end
    end
    return q_L_m, q_R_m, q_L_0, q_R_0
end

"""Apply Colella-Woodward monotonicity: flatten reconstruction at local extrema."""
@inline function _apply_monotonicity(q_L, q_R, c)
    FT = typeof(c)
    if (q_R - c) * (c - q_L) <= zero(FT)
        return c, c
    end
    return q_L, q_R
end

"""Compute upwind PPM face value given mass flux, donor mass, and PPM edges.

Uses the full parabolic integral (FV3 xppm/yppm formula):
  Positive flow: face = c + (1-α)(br - α·b0)
  Negative flow: face = c + (1+α)(bl + α·b0)
where bl = q_L - c, br = q_R - c, b0 = bl + br (curvature).
"""
@inline function _ppm_face_value(flux, m_lo, m_hi, c_lo, c_hi,
                                  q_L_lo, q_R_lo, q_L_hi, q_R_hi)
    FT = typeof(c_lo)
    if flux >= zero(FT)
        alpha = m_lo > 100 * eps(FT) ? flux / m_lo : zero(FT)
        bl = q_L_lo - c_lo
        br = q_R_lo - c_lo
        b0 = bl + br
        return c_lo + (one(FT) - alpha) * (br - alpha * b0)
    else
        alpha = m_hi > 100 * eps(FT) ? flux / m_hi : zero(FT)
        bl = q_L_hi - c_hi
        br = q_R_hi - c_hi
        b0 = bl + br
        return c_hi + (one(FT) + alpha) * (bl + alpha * b0)
    end
end

# ---------------------------------------------------------------------------
# PPM Face Value Kernels (mixing ratio at cell faces)
#
# Two variants per direction:
#   _ppm_{x,y}_face_kernel!     — reads rm/m (original field)
#   _ppm_{x,y}_face_from_q_kernel! — reads pre-computed mixing ratio q
# ---------------------------------------------------------------------------

@kernel function _ppm_y_face_kernel!(
    fy_face, @Const(rm), @Const(m), @Const(bm), Hp, Nc, ::Val{ORD}
) where ORD
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        c_m3 = _safe_mixing_ratio(rm[ii, jj_b - 2, k], m[ii, jj_b - 2, k])
        c_m2 = _safe_mixing_ratio(rm[ii, jj_b - 1, k], m[ii, jj_b - 1, k])
        c_m1 = _safe_mixing_ratio(rm[ii, jj_b,     k], m[ii, jj_b,     k])
        c_0  = _safe_mixing_ratio(rm[ii, jj_a,     k], m[ii, jj_a,     k])
        c_p1 = _safe_mixing_ratio(rm[ii, jj_a + 1, k], m[ii, jj_a + 1, k])
        c_p2 = _safe_mixing_ratio(rm[ii, jj_a + 2, k], m[ii, jj_a + 2, k])

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, jf, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fy_face[i, jf, k] = _ppm_face_value(
            bm[i, jf, k], m[ii, jj_b, k], m[ii, jj_a, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_x_face_kernel!(
    fx_face, @Const(rm), @Const(m), @Const(am), Hp, Nc, ::Val{ORD}
) where ORD
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        c_m3 = _safe_mixing_ratio(rm[ii_l - 2, jj, k], m[ii_l - 2, jj, k])
        c_m2 = _safe_mixing_ratio(rm[ii_l - 1, jj, k], m[ii_l - 1, jj, k])
        c_m1 = _safe_mixing_ratio(rm[ii_l,     jj, k], m[ii_l,     jj, k])
        c_0  = _safe_mixing_ratio(rm[ii_r,     jj, k], m[ii_r,     jj, k])
        c_p1 = _safe_mixing_ratio(rm[ii_r + 1, jj, k], m[ii_r + 1, jj, k])
        c_p2 = _safe_mixing_ratio(rm[ii_r + 2, jj, k], m[ii_r + 2, jj, k])

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, iif, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fx_face[iif, j, k] = _ppm_face_value(
            am[iif, j, k], m[ii_l, jj, k], m[ii_r, jj, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_x_face_from_q_kernel!(
    fx_face, @Const(q), @Const(am), @Const(m), Hp, Nc, ::Val{ORD}
) where ORD
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        c_m3 = q[ii_l - 2, jj, k]; c_m2 = q[ii_l - 1, jj, k]
        c_m1 = q[ii_l,     jj, k]; c_0  = q[ii_r,     jj, k]
        c_p1 = q[ii_r + 1, jj, k]; c_p2 = q[ii_r + 2, jj, k]

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, iif, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fx_face[iif, j, k] = _ppm_face_value(
            am[iif, j, k], m[ii_l, jj, k], m[ii_r, jj, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_y_face_from_q_kernel!(
    fy_face, @Const(q), @Const(bm), @Const(m), Hp, Nc, ::Val{ORD}
) where ORD
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii   = Hp + i
        jj_b = Hp + jf - 1
        jj_a = Hp + jf

        c_m3 = q[ii, jj_b - 2, k]; c_m2 = q[ii, jj_b - 1, k]
        c_m1 = q[ii, jj_b,     k]; c_0  = q[ii, jj_a,     k]
        c_p1 = q[ii, jj_a + 1, k]; c_p2 = q[ii, jj_a + 2, k]

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, jf, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fy_face[i, jf, k] = _ppm_face_value(
            bm[i, jf, k], m[ii, jj_b, k], m[ii, jj_a, k],
            c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

# ---------------------------------------------------------------------------
# q_buf initialization kernel (mixing ratio from rm/m, full haloed domain)
# ---------------------------------------------------------------------------

@kernel function _init_q_buf_kernel!(q_buf, @Const(rm), @Const(m))
    i, j, k = @index(Global, NTuple)
    @inbounds q_buf[i, j, k] = _safe_mixing_ratio(rm[i, j, k], m[i, j, k])
end

# ---------------------------------------------------------------------------
# Pre-advection kernels (advective-form transport for cross-term)
# ---------------------------------------------------------------------------

@kernel function _pre_advect_y_kernel!(
    q_i, @Const(rm), @Const(m), @Const(bm), @Const(fy_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        bm_s = bm[i, j, k];  bm_n = bm[i, j + 1, k]
        rm_new = rm[ii, jj, k] + bm_s * fy_face[i, j, k] - bm_n * fy_face[i, j + 1, k]
        m_new  = m[ii, jj, k]  + bm_s - bm_n
        q_i[ii, jj, k] = _safe_mixing_ratio(rm_new, m_new)
    end
end

@kernel function _pre_advect_x_kernel!(
    q_j, @Const(rm), @Const(m), @Const(am), @Const(fx_face), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        am_w = am[i, j, k];  am_e = am[i + 1, j, k]
        rm_new = rm[ii, jj, k] + am_w * fx_face[i, j, k] - am_e * fx_face[i + 1, j, k]
        m_new  = m[ii, jj, k]  + am_w - am_e
        q_j[ii, jj, k] = _safe_mixing_ratio(rm_new, m_new)
    end
end

# ---------------------------------------------------------------------------
# Combined update kernel (applies averaged x+y fluxes simultaneously)
# ---------------------------------------------------------------------------

@kernel function _linrood_update_kernel!(
    rm_new, m_new, @Const(rm), @Const(m), @Const(am), @Const(bm),
    @Const(fx_in), @Const(fx_out), @Const(fy_in), @Const(fy_out), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(rm)
        half = FT(0.5)

        avg_fx_w = half * (fx_out[i,   j, k] + fx_in[i,   j, k])
        avg_fx_e = half * (fx_out[i+1, j, k] + fx_in[i+1, j, k])
        avg_fy_s = half * (fy_out[i, j,   k] + fy_in[i, j,   k])
        avg_fy_n = half * (fy_out[i, j+1, k] + fy_in[i, j+1, k])

        am_w = am[i, j, k];  am_e = am[i+1, j, k]
        bm_s = bm[i, j, k];  bm_n = bm[i, j+1, k]

        rm_new[ii, jj, k] = rm[ii, jj, k] +
            (am_w * avg_fx_w - am_e * avg_fx_e) +
            (bm_s * avg_fy_s - bm_n * avg_fy_n)
        m_new[ii, jj, k] = m[ii, jj, k] +
            (am_w - am_e) + (bm_s - bm_n)
    end
end

# ---------------------------------------------------------------------------
# Main Lin-Rood Horizontal Advection
# ---------------------------------------------------------------------------

"""
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  grid, ::Val{ORD}, ws, ws_lr; damp_coeff=0.0)

Lin-Rood horizontal advection for cubed-sphere grids.
Averages X-first and Y-first PPM orderings (FV3 fv_tp_2d algorithm).
"""
function fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                       grid::CubedSphereGrid, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                       damp_coeff=0.0) where ORD
    (; Hp, Nc, Nz) = grid
    N = Nc + 2Hp
    backend = get_backend(rm_panels[1])

    # Optional divergence damping
    damp_coeff > 0 && apply_divergence_damping_cs!(rm_panels, m_panels, grid, ws, damp_coeff)

    # Pre-instantiate all kernels (avoid repeated compilation inside loops)
    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Edge halos + Y-corners → inner Y-PPM + pre-advect q_i
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)
    copy_corners!(rm_panels, grid, 2)
    copy_corners!(m_panels, grid, 2)

    # Initialize q_buf with original mixing ratio (halos persist for outer PPM)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.fy_in)
        y_face_k!(ws_lr.fy_in[p], rm_panels[p], m_panels[p], bm_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], bm_panels[p],
                 ws_lr.fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + pre-advect q_j
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 1)
    copy_corners!(rm_panels, grid, 1)
    copy_corners!(m_panels, grid, 1)

    for p in eachindex(ws_lr.fx_out)
        xq_face_k!(ws_lr.fx_out[p], ws_lr.q_buf[p], am_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        x_face_k!(ws_lr.fx_in[p], rm_panels[p], m_panels[p], am_panels[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    # Re-initialize q_buf (halos retain original q; interior overwritten with q_j)
    for p in eachindex(ws_lr.q_buf)
        init_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p]; ndrange=(N, N, Nz))
    end
    synchronize(backend)

    for p in eachindex(ws_lr.q_buf)
        pre_x_k!(ws_lr.q_buf[p], rm_panels[p], m_panels[p], am_panels[p],
                 ws_lr.fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM → averaged update
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 2)

    for p in eachindex(ws_lr.fx_in)
        yq_face_k!(ws_lr.fy_out, ws_lr.q_buf[p], bm_panels[p], m_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        update_k!(ws.rm_buf, ws.m_buf,
                  rm_panels[p], m_panels[p], am_panels[p], bm_panels[p],
                  ws_lr.fx_in[p], ws_lr.fx_out[p], ws_lr.fy_in[p], ws_lr.fy_out,
                  Hp; ndrange=(Nc, Nc, Nz))
        synchronize(backend)  # required: ws.rm_buf/m_buf reused across panels
        _copy_interior!(rm_panels[p], ws.rm_buf, Hp, Nc, Nz)
        _copy_interior!(m_panels[p], ws.m_buf, Hp, Nc, Nz)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Strang Split: Lin-Rood Horizontal + Vertical Z-sweep
# ---------------------------------------------------------------------------

"""
    strang_split_linrood_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                               grid, ::Val{ORD}, ws, ws_lr; cfl_limit=0.95, damp_coeff=0.0)

Full 3D advection: Horizontal(LR) → Z → Z → Horizontal(LR).
"""
function strang_split_linrood_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                                    grid::CubedSphereGrid, ::Val{ORD}, ws, ws_lr::LinRoodWorkspace;
                                    cfl_limit=0.95, damp_coeff=0.0) where ORD
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  grid, Val(ORD), ws, ws_lr; damp_coeff)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, true, ws)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, true, ws)
    fv_tp_2d_cs!(rm_panels, m_panels, am_panels, bm_panels,
                  grid, Val(ORD), ws, ws_lr; damp_coeff=0.0)
    return nothing
end
