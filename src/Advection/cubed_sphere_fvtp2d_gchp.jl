# ---------------------------------------------------------------------------
# GCHP-Faithful fv_tp_2d Port — Area-based pre-advection + Courant PPM
#
# Line-by-line port of GCHP's fv_tp_2d (tp_core.F90:108-242) and
# tracer_2d_1L (fv_tracer2d.F90:95-333).
#
# KEY DIFFERENCES from our existing fv_tp_2d_cs_q!:
#   1. PPM uses Courant number (cx/cy) instead of mass fraction (am/m)
#   2. Pre-advection uses area fluxes (xfx*fy2) not mass fluxes (bm*fy2)
#   3. Pre-advection denominator is evolved AREA (ra_y) not evolved MASS
#   4. Final flux averaging uses mass flux (am/bm) — same as existing
#
# The area-based pre-advection means q_i doesn't depend on dp, eliminating
# level-dependent artifacts that manifest as SH oscillations at 750 hPa.
#
# Reference:
#   tp_core.F90 in FVdycoreCubed_GridComp/fvdycore/model/
#   fv_tracer2d.F90 in FVdycoreCubed_GridComp/fvdycore/model/
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# Grid geometry for GCHP-faithful transport
# ---------------------------------------------------------------------------

"""
    GCHPGridGeometry{FT, A2, A2x, A2y}

Precomputed grid metric terms on GPU for GCHP-faithful fv_tp_2d.
Computed once from the gnomonic projection and uploaded to device.

Fields:
- `area_dev` — cell area [m²], shape (Nc, Nc) per panel
- `dxa_dev`  — cell X-width [m], shape (Nc, Nc) per panel
- `dya_dev`  — cell Y-width [m], shape (Nc, Nc) per panel
- `dy_dev`   — Y-edge length at X-faces [m], shape (Nc+1, Nc) per panel
- `dx_dev`   — X-edge length at Y-faces [m], shape (Nc, Nc+1) per panel
"""
struct GCHPGridGeometry{FT, A2 <: AbstractArray{FT,2},
                         A2x <: AbstractArray{FT,2},
                         A2y <: AbstractArray{FT,2}}
    area_dev :: NTuple{6, A2}
    dxa_dev  :: NTuple{6, A2}
    dya_dev  :: NTuple{6, A2}
    dy_dev   :: NTuple{6, A2x}
    dx_dev   :: NTuple{6, A2y}
end

"""
    GCHPGridGeometry(grid::CubedSphereGrid)

Compute GCHP grid geometry from the gnomonic projection and upload to device.
Edge lengths (dy at X-faces, dx at Y-faces) are computed from face coordinates
via great-circle distance.

NOTE: sin_sg is approximated as 1.0 (exact at panel center, ~5% error at corners).
This can be refined later by computing exact great-circle angles.
"""
function GCHPGridGeometry(grid::CubedSphereGrid{FT}) where FT
    (; Nc, Hp) = grid
    AT = array_type(architecture(grid))
    R = grid.radius

    # Cell areas and widths — already computed in grid
    area_dev = ntuple(p -> AT(FT.(grid.Aᶜ[p])), 6)
    dxa_dev  = ntuple(p -> AT(FT.(grid.Δxᶜ[p])), 6)
    dya_dev  = ntuple(p -> AT(FT.(grid.Δyᶜ[p])), 6)

    # Edge lengths at faces: compute from face coordinates
    #   dy_face[iif, j] = great-circle distance between vertices (iif,j) and (iif,j+1)
    #   dx_face[i, jf]  = great-circle distance between vertices (i,jf) and (i+1,jf)
    dy_cpu = ntuple(6) do p
        dy = zeros(FT, Nc + 1, Nc)
        for j in 1:Nc, iif in 1:(Nc + 1)
            λ1, φ1 = deg2rad(grid.λᶠ[p][iif, j]),   deg2rad(grid.φᶠ[p][iif, j])
            λ2, φ2 = deg2rad(grid.λᶠ[p][iif, j+1]), deg2rad(grid.φᶠ[p][iif, j+1])
            # Haversine formula for great-circle distance
            dλ = λ2 - λ1
            dφ = φ2 - φ1
            a = sin(dφ/2)^2 + cos(φ1) * cos(φ2) * sin(dλ/2)^2
            dy[iif, j] = R * 2 * atan(sqrt(a), sqrt(1 - a))
        end
        dy
    end

    dx_cpu = ntuple(6) do p
        dx = zeros(FT, Nc, Nc + 1)
        for jf in 1:(Nc + 1), i in 1:Nc
            λ1, φ1 = deg2rad(grid.λᶠ[p][i, jf]),   deg2rad(grid.φᶠ[p][i, jf])
            λ2, φ2 = deg2rad(grid.λᶠ[p][i+1, jf]), deg2rad(grid.φᶠ[p][i+1, jf])
            dλ = λ2 - λ1
            dφ = φ2 - φ1
            a = sin(dφ/2)^2 + cos(φ1) * cos(φ2) * sin(dλ/2)^2
            dx[i, jf] = R * 2 * atan(sqrt(a), sqrt(1 - a))
        end
        dx
    end

    dy_dev = ntuple(p -> AT(dy_cpu[p]), 6)
    dx_dev = ntuple(p -> AT(dx_cpu[p]), 6)

    return GCHPGridGeometry(area_dev, dxa_dev, dya_dev, dy_dev, dx_dev)
end

# ---------------------------------------------------------------------------
# GCHP Workspace — area fluxes (recomputed per window from cx/cy + geometry)
# ---------------------------------------------------------------------------

"""
    GCHPTransportWorkspace{FT, A3x, A3y}

Runtime workspace for GCHP-faithful transport. Holds area fluxes computed
from Courant numbers and grid geometry.

The LinRoodWorkspace (shared buffers: q_buf, fx_in, fx_out, fy_in, fy_out,
q_out, dp_out) is passed separately.
"""
struct GCHPTransportWorkspace{FT, A3x <: AbstractArray{FT,3},
                                A3y <: AbstractArray{FT,3}}
    xfx :: NTuple{6, A3x}   # area flux X (Nc+1, Nc, Nz)
    yfx :: NTuple{6, A3y}   # area flux Y (Nc, Nc+1, Nz)
end

function GCHPTransportWorkspace(grid::CubedSphereGrid{FT}) where FT
    (; Nc, Nz) = grid
    AT = array_type(architecture(grid))
    xfx = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    yfx = ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6)
    return GCHPTransportWorkspace(xfx, yfx)
end

# ---------------------------------------------------------------------------
# PPM face value using Courant number (GCHP xppm/yppm formula)
#
# Same parabolic integral as _ppm_face_value, but uses Courant number
# directly instead of mass fraction (flux/m).
#
# GCHP tp_core.F90:621-627 (xppm, iord≥8):
#   if c > 0: flux = q(i-1) + (1-c)*(br(i-1) - c*(bl(i-1)+br(i-1)))
#   else:     flux = q(i)   + (1+c)*(bl(i)   + c*(bl(i)+br(i)))
# ---------------------------------------------------------------------------

@inline function _ppm_face_value_courant(crx, c_lo, c_hi,
                                          q_L_lo, q_R_lo, q_L_hi, q_R_hi)
    FT = typeof(c_lo)
    if crx > zero(FT)
        bl = q_L_lo - c_lo
        br = q_R_lo - c_lo
        b0 = bl + br
        return c_lo + (one(FT) - crx) * (br - crx * b0)
    else
        bl = q_L_hi - c_hi
        br = q_R_hi - c_hi
        b0 = bl + br
        return c_hi + (one(FT) + crx) * (bl + crx * b0)
    end
end

# ---------------------------------------------------------------------------
# PPM Face Kernels using Courant number
#
# Same edge value computation as _ppm_{x,y}_face_from_q_kernel!, but the
# upwind integral uses the Courant number (cx/cy) instead of mass fraction.
# ---------------------------------------------------------------------------

@kernel function _ppm_x_face_courant_kernel!(
    fx_face, @Const(q), @Const(cx), Hp, Nc, ::Val{ORD}
) where ORD
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        jj   = Hp + j
        ii_l = Hp + iif - 1
        ii_r = Hp + iif

        # 5-point stencil from mixing ratio q
        c_m3 = q[ii_l - 2, jj, k]; c_m2 = q[ii_l - 1, jj, k]
        c_m1 = q[ii_l,     jj, k]; c_0  = q[ii_r,     jj, k]
        c_p1 = q[ii_r + 1, jj, k]; c_p2 = q[ii_r + 2, jj, k]

        q_L_m, q_R_m = _ppm_edge_values(c_m3, c_m2, c_m1, c_0, c_p1, Val(ORD))
        q_L_0, q_R_0 = _ppm_edge_values(c_m2, c_m1, c_0, c_p1, c_p2, Val(ORD))
        q_L_m, q_R_m, q_L_0, q_R_0 = _apply_ord7_boundary(
            q_L_m, q_R_m, q_L_0, q_R_0, c_m1, c_m2, c_0, c_p1, iif, Nc, Val(ORD))
        q_L_m, q_R_m = _apply_monotonicity(q_L_m, q_R_m, c_m1)
        q_L_0, q_R_0 = _apply_monotonicity(q_L_0, q_R_0, c_0)

        fx_face[iif, j, k] = _ppm_face_value_courant(
            cx[iif, j, k], c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

@kernel function _ppm_y_face_courant_kernel!(
    fy_face, @Const(q), @Const(cy), Hp, Nc, ::Val{ORD}
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

        fy_face[i, jf, k] = _ppm_face_value_courant(
            cy[i, jf, k], c_m1, c_0, q_L_m, q_R_m, q_L_0, q_R_0)
    end
end

# ---------------------------------------------------------------------------
# Area flux computation (GCHP fv_tracer2d.F90:162-182)
#
#   xfx(i,j,k) = cx(i,j,k) * dxa(i_up,j) * dy(i,j) * sin_sg(i_up,j,edge)
#   yfx(i,j,k) = cy(i,j,k) * dya(i,j_up) * dx(i,j) * sin_sg(i,j_up,edge)
#
# sin_sg is approximated as 1.0 for now (exact at panel center).
# Upwind cell index is clamped to [1, Nc] at panel boundaries.
# ---------------------------------------------------------------------------

@kernel function _compute_xfx_kernel!(
    xfx, @Const(cx), @Const(dxa), @Const(dy_face), Nc
)
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cx)
        c = cx[iif, j, k]
        if c > zero(FT)
            i_up = max(1, iif - 1)
            xfx[iif, j, k] = c * dxa[i_up, j] * dy_face[iif, j]
        else
            i_up = min(Nc, iif)
            xfx[iif, j, k] = c * dxa[i_up, j] * dy_face[iif, j]
        end
    end
end

@kernel function _compute_yfx_kernel!(
    yfx, @Const(cy), @Const(dya), @Const(dx_face), Nc
)
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cy)
        c = cy[i, jf, k]
        if c > zero(FT)
            j_up = max(1, jf - 1)
            yfx[i, jf, k] = c * dya[i, j_up] * dx_face[i, jf]
        else
            j_up = min(Nc, jf)
            yfx[i, jf, k] = c * dya[i, j_up] * dx_face[i, jf]
        end
    end
end

"""
    compute_area_fluxes!(ws_gchp, cx_panels, cy_panels, geom, grid)

Compute area fluxes (xfx, yfx) from Courant numbers and grid geometry.
Called once per window after CX/CY are uploaded to GPU.

GCHP reference: fv_tracer2d.F90:162-182.
"""
function compute_area_fluxes!(ws_gchp::GCHPTransportWorkspace,
                               cx_panels, cy_panels,
                               geom::GCHPGridGeometry,
                               grid::CubedSphereGrid)
    (; Nc, Nz) = grid
    for_panels_nosync() do p
        be = get_backend(cx_panels[p])
        _compute_xfx_kernel!(be, 256)(ws_gchp.xfx[p], cx_panels[p], geom.dxa_dev[p], geom.dy_dev[p], Nc;
            ndrange=(Nc + 1, Nc, Nz))
        _compute_yfx_kernel!(be, 256)(ws_gchp.yfx[p], cy_panels[p], geom.dya_dev[p], geom.dx_dev[p], Nc;
            ndrange=(Nc, Nc + 1, Nz))
    end
end

# ---------------------------------------------------------------------------
# Area-based pre-advection kernels (GCHP tp_core.F90:172-192)
#
# Pre-advect q using AREA flux divergence instead of MASS flux divergence.
# The denominator is evolved AREA (ra_y or ra_x), not evolved MASS.
# This eliminates the dp-dependent artifacts in our mass-based version.
#
# GCHP:
#   fyy(i,j) = yfx(i,j) * fy2(i,j)
#   ra_y(i,j) = area(i,j) + yfx(i,j) - yfx(i,j+1)
#   q_i(i,j) = (q(i,j)*area(i,j) + fyy(i,j) - fyy(i,j+1)) / ra_y(i,j)
# ---------------------------------------------------------------------------

@kernel function _pre_advect_y_area_kernel!(
    q_i, @Const(q), @Const(fy_face), @Const(yfx), @Const(area), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)

        # Tracer flux = area_flux × face_value
        fyy_s = yfx[i, j,   k] * fy_face[i, j,   k]
        fyy_n = yfx[i, j+1, k] * fy_face[i, j+1, k]

        # Evolved area (GCHP ra_y)
        ra_y = area[i, j] + (yfx[i, j, k] - yfx[i, j+1, k])

        # Area-based pre-advection
        q_area = q[ii, jj, k] * area[i, j]
        q_i[ii, jj, k] = ra_y > FT(100) * eps(FT) ? (q_area + fyy_s - fyy_n) / ra_y : zero(FT)
    end
end

@kernel function _pre_advect_x_area_kernel!(
    q_j, @Const(q), @Const(fx_face), @Const(xfx), @Const(area), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)

        # Tracer flux = area_flux × face_value
        fx1_w = xfx[i,   j, k] * fx_face[i,   j, k]
        fx1_e = xfx[i+1, j, k] * fx_face[i+1, j, k]

        # Evolved area (GCHP ra_x)
        ra_x = area[i, j] + (xfx[i, j, k] - xfx[i+1, j, k])

        # Area-based pre-advection
        q_area = q[ii, jj, k] * area[i, j]
        q_j[ii, jj, k] = ra_x > FT(100) * eps(FT) ? (q_area + fx1_w - fx1_e) / ra_x : zero(FT)
    end
end

# ---------------------------------------------------------------------------
# GCHP update kernel — NO per-cell CFL guard
#
# Unlike _linrood_update_q_kernel! (cubed_sphere_fvtp2d.jl:373), this kernel
# does NOT clamp mass divergence. GCHP handles CFL via per-level subcycling
# (uniform flux reduction per level), not per-cell clamping. The per-cell
# guard creates m_evolved ≠ m_ref + full_mass_div, which makes the remap
# path's prescale (rm *= m_save / m_evolved) inflate tracer by 10× at thin
# TOA cells, causing O(1000 ppm) mass fixer corrections.
# ---------------------------------------------------------------------------

@kernel function _linrood_update_gchp_kernel!(
    q_new, m_new, @Const(q), @Const(m), @Const(am), @Const(bm),
    @Const(fx_in), @Const(fx_out), @Const(fy_in), @Const(fy_out), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        half = FT(0.5)

        avg_fx_w = half * (fx_out[i,   j, k] + fx_in[i,   j, k])
        avg_fx_e = half * (fx_out[i+1, j, k] + fx_in[i+1, j, k])
        avg_fy_s = half * (fy_out[i, j,   k] + fy_in[i, j,   k])
        avg_fy_n = half * (fy_out[i, j+1, k] + fy_in[i, j+1, k])

        am_w = am[i, j, k];  am_e = am[i+1, j, k]
        bm_s = bm[i, j, k];  bm_n = bm[i, j+1, k]

        m1 = m[ii, jj, k]

        # NO CFL guard — CFL handled by ×0.5 scaling + GEOS dynamics stability
        mass_div = (am_w - am_e) + (bm_s - bm_n)
        m2 = m1 + mass_div

        # q_new = (q*m1 + tracer_flux_div) / m2
        rm_old = q[ii, jj, k] * m1
        tracer_div = (am_w * avg_fx_w - am_e * avg_fx_e) +
                     (bm_s * avg_fy_s - bm_n * avg_fy_n)
        rm_new = rm_old + tracer_div

        q_new[ii, jj, k] = m2 > FT(100) * eps(FT) ? rm_new / m2 : zero(FT)
        m_new[ii, jj, k] = m2
    end
end

# ---------------------------------------------------------------------------
# GCHP-Faithful Lin-Rood Horizontal Advection
#
# Same 3-phase structure as fv_tp_2d_cs_q!, but with:
#   1. Courant-number PPM (cx/cy) instead of mass-fraction PPM
#   2. Area-based pre-advection instead of mass-based
#   3. Mass-flux-based final update (unchanged from fv_tp_2d_cs_q!)
#
# The final update kernel (_linrood_update_gchp_kernel!) is a guard-free
# version of _linrood_update_q_kernel! — no per-cell CFL clamping, since
# GCHP handles CFL via per-level subcycling.
# ---------------------------------------------------------------------------

"""
    fv_tp_2d_gchp!(q_panels, m_panels, am_panels, bm_panels,
                     cx_panels, cy_panels, xfx_panels, yfx_panels,
                     area_panels, grid, ::Val{ORD}, ws, ws_lr)

GCHP-faithful Lin-Rood horizontal advection. Key differences from `fv_tp_2d_cs_q!`:
- PPM uses Courant number (cx/cy) for upwind integration
- Pre-advection uses precomputed area fluxes (xfx/yfx) with exact sin_sg
- Final flux averaging uses mass flux (am/bm) — identical to existing

Arguments:
- `q_panels`    — mixing ratio panels (haloed, modified in-place)
- `m_panels`    — air mass panels (haloed, modified in-place)
- `am/bm`       — mass flux panels (staggered, read-only)
- `cx/cy`       — Courant number panels (staggered, read-only)
- `xfx/yfx`     — precomputed area flux panels (staggered, read-only)
- `area_panels`  — cell areas per panel (Nc×Nc, read-only)
- `grid`        — CubedSphereGrid
- `ws`          — shared CubedSphereMassFluxWorkspace (rm_buf, m_buf)
- `ws_lr`       — LinRoodWorkspace (q_buf, fx_in/out, fy_in/out, q_out, dp_out)
"""
function fv_tp_2d_gchp!(q_panels, m_panels, am_panels, bm_panels,
                          cx_panels, cy_panels,
                          xfx_panels, yfx_panels, area_panels,
                          grid::CubedSphereGrid, ::Val{ORD},
                          ws, ws_lr::LinRoodWorkspace) where ORD
    (; Hp, Nc, Nz) = grid

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Edge halos + Y-corners → inner Y-PPM (Courant) + area pre-advect
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(q_panels, grid)
    fill_panel_halos!(m_panels, grid)
    copy_corners!(q_panels, grid, 2)
    copy_corners!(m_panels, grid, 2)

    for_panels_nosync() do p
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _ppm_y_face_courant_kernel!(be, 256)(ws_lr.fy_in[p], q_panels[p], cy_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        _pre_advect_y_area_kernel!(be, 256)(ws_lr.q_buf[p], q_panels[p], ws_lr.fy_in[p],
                   yfx_panels[p], area_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + area pre-advect
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 1)
    copy_corners!(q_panels, grid, 1)
    copy_corners!(m_panels, grid, 1)

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _ppm_x_face_courant_kernel!(be, 256)(ws_lr.fx_out[p], ws_lr.q_buf[p], cx_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        _ppm_x_face_courant_kernel!(be, 256)(ws_lr.fx_in[p], q_panels[p], cx_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end

    for_panels_nosync() do p
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _pre_advect_x_area_kernel!(be, 256)(ws_lr.q_buf[p], q_panels[p], ws_lr.fx_in[p],
                   xfx_panels[p], area_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM → averaged update
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 2)

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _ppm_y_face_courant_kernel!(be, 256)(ws_lr.fy_out[p], ws_lr.q_buf[p], cy_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        _linrood_update_gchp_kernel!(be, 256)(ws_lr.q_out[p], ws_lr.dp_out[p],
                    q_panels[p], m_panels[p], am_panels[p], bm_panels[p],
                    ws_lr.fx_in[p], ws_lr.fx_out[p], ws_lr.fy_in[p], ws_lr.fy_out[p],
                    Hp; ndrange=(Nc, Nc, Nz))
    end

    # Copy results back
    for_panels_nosync() do p
        _copy_interior!(q_panels[p], ws_lr.q_out[p], Hp, Nc, Nz)
        _copy_interior!(m_panels[p], ws_lr.dp_out[p], Hp, Nc, Nz)
    end

    # NOTE: fillz_q! (positivity fixer) is NOT called here.
    # In the remap path, fv_tp_2d_gchp! is called twice per substep.
    # Calling fillz after each would break the mass budget that the
    # rm-rescaling relies on. Positivity is handled by the vertical
    # remap + mass fixer instead.

    return nothing
end

# ---------------------------------------------------------------------------
# Full 3D: GCHP horizontal + vertical (Strang or remap)
# ---------------------------------------------------------------------------

"""
    strang_split_gchp_ppm!(q_panels, m_panels, am_panels, bm_panels,
                             cm_panels, cx_panels, cy_panels,
                             geom, ws_gchp, grid, ::Val{ORD}, ws, ws_lr)

Full 3D advection with GCHP-faithful horizontal:
  Horizontal(GCHP) → Z → Z → Horizontal(GCHP)
"""
function strang_split_gchp_ppm!(q_panels, m_panels, am_panels, bm_panels, cm_panels,
                                  cx_panels, cy_panels,
                                  geom::GCHPGridGeometry,
                                  ws_gchp::GCHPTransportWorkspace,
                                  grid::CubedSphereGrid, ::Val{ORD}, ws,
                                  ws_lr::LinRoodWorkspace) where ORD
    # Horizontal in q-space (GCHP-faithful)
    fv_tp_2d_gchp!(q_panels, m_panels, am_panels, bm_panels,
                     cx_panels, cy_panels, geom, ws_gchp,
                     grid, Val(ORD), ws, ws_lr)

    # Z-sweep operates in rm-space: convert q→rm, sweep, convert rm→q
    q_to_rm_panels!(q_panels, m_panels, grid)
    _sweep_z!(q_panels, m_panels, cm_panels, grid, true, ws)
    _sweep_z!(q_panels, m_panels, cm_panels, grid, true, ws)
    rm_to_q_panels!(q_panels, m_panels, grid)

    # Second horizontal pass in q-space
    fv_tp_2d_gchp!(q_panels, m_panels, am_panels, bm_panels,
                     cx_panels, cy_panels, geom, ws_gchp,
                     grid, Val(ORD), ws, ws_lr)
    return nothing
end

# ---------------------------------------------------------------------------
# Humidity correction for mass fluxes (GCHP CORRECT_MASS_FLUX_FOR_HUMIDITY)
# Port of GCHPctmEnv_GridCompMod.F90:1029-1031
#
# MFX(I,J,L) /= (1 - SPHU(I,J,L,1))
#
# Converts mass fluxes to dry-air basis for use with dry pressure surfaces.
# Uses cell-center QV at the face index (GCHP approximation).
# ---------------------------------------------------------------------------

@kernel function _correct_mfx_humidity_kernel!(mfx, @Const(qv), Hp, Nc)
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        # GCHP (GCHPctmEnv:1029): MFX(I,J,L) /= (1 - SPHU(I,J,L))
        # am[iif] = mfxc[iif-1], so face iif = east face of cell iif-1.
        # Use SOURCE cell (iif-1), matching GCHP's cell-I convention.
        ii = Hp + max(1, iif - 1)
        jj = Hp + j
        FT = eltype(mfx)
        mfx[iif, j, k] /= max(FT(1) - qv[ii, jj, k], eps(FT))
    end
end

@kernel function _correct_mfy_humidity_kernel!(mfy, @Const(qv), Hp, Nc)
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        # bm[jf] = mfyc[jf-1], so face jf = north face of cell jf-1.
        # Use SOURCE cell (jf-1).
        ii = Hp + i
        jj = Hp + max(1, jf - 1)
        FT = eltype(mfy)
        mfy[i, jf, k] /= max(FT(1) - qv[ii, jj, k], eps(FT))
    end
end

@kernel function _reverse_mfx_humidity_kernel!(mfx, @Const(qv), Hp, Nc)
    iif, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + max(1, iif - 1)
        jj = Hp + j
        FT = eltype(mfx)
        mfx[iif, j, k] *= max(FT(1) - qv[ii, jj, k], eps(FT))
    end
end

@kernel function _reverse_mfy_humidity_kernel!(mfy, @Const(qv), Hp, Nc)
    i, jf, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + max(1, jf - 1)
        FT = eltype(mfy)
        mfy[i, jf, k] *= max(FT(1) - qv[ii, jj, k], eps(FT))
    end
end

# ---------------------------------------------------------------------------
# Tracer basis conversion kernels (AdvCore_GridCompMod.F90:1070, 1300)
#
# GCHP converts tracers between dry and wet VMR at the boundaries of
# offline_tracer_advection. q_wet = q_dry × (1-SPHU), q_dry = q_wet / (1-SPHU).
# In our code, the back-conversion (wet→dry) is implicit: rm / m_dry = q_dry.
# ---------------------------------------------------------------------------

@kernel function _multiply_by_1_minus_qv_kernel!(q, @Const(qv), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        q[ii, jj, k] *= (one(eltype(q)) - qv[ii, jj, k])
    end
end

@kernel function _divide_by_1_minus_qv_kernel!(q, @Const(qv), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        q[ii, jj, k] /= max(one(FT) - qv[ii, jj, k], eps(FT))
    end
end

# ---------------------------------------------------------------------------
# Dry dp kernel: dp_dry = DELP × (1 - QV)
#
# Converts moist pressure thickness to dry-air basis. Operates on full
# haloed arrays (Nc+2Hp × Nc+2Hp × Nz) so no Hp offset is needed.
# ---------------------------------------------------------------------------

@kernel function _compute_dry_dp_kernel!(dp_dry, @Const(delp), @Const(qv))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        dp_dry[i, j, k] = delp[i, j, k] * (one(eltype(dp_dry)) - qv[i, j, k])
    end
end

"""Copy interior-only array (Nc×Nc×Nz) into haloed array (Nc+2Hp × Nc+2Hp × Nz).
Used after vertical remap to prepare dp_work for the next horizontal transport substep."""
@kernel function _copy_nohalo_to_halo_kernel!(dst, @Const(src), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[Hp + i, Hp + j, k] = src[i, j, k]
end

"""Interpolate dry dp from two moist DELP snapshots with temporally interpolated QV.
GCHP uses SPHU0 for DryPLE0 and SPHU2 for DryPLE1 (each endpoint uses its own QV).
dp = ((1-frac)×delp0 + frac×delp1) × (1 - ((1-frac)×qv0 + frac×qv1))"""
@kernel function _interpolate_dry_dp_kernel!(dp_out, @Const(delp_0), @Const(delp_1),
                                              @Const(qv0), @Const(qv1), frac)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(dp_out)
        qv_interp = (one(FT) - frac) * qv0[i, j, k] + frac * qv1[i, j, k]
        dp_out[i, j, k] = ((one(FT) - frac) * delp_0[i, j, k] +
                            frac * delp_1[i, j, k]) * (one(FT) - qv_interp)
    end
end

"""Interpolate moist dp from two DELP snapshots: dp = (1-frac)×delp0 + frac×delp1."""
@kernel function _interpolate_dp_kernel!(dp_out, @Const(delp_0), @Const(delp_1), frac)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(dp_out)
        dp_out[i, j, k] = (one(FT) - frac) * delp_0[i, j, k] + frac * delp_1[i, j, k]
    end
end

"""Per-column dp correction (dry basis): scale evolved dp so column sum matches
interpolated dry target PS. Prevents dp drift in n_sub loop while keeping q unchanged.
`frac` = i/n_sub (endpoint fraction for this substep)."""
@kernel function _column_dp_correction_kernel!(dp::AbstractArray{FT, 3},
                                                @Const(delp_0), @Const(delp_1),
                                                @Const(qv), frac, Hp, Nz) where {FT}
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        # Target column dry dp = sum of interpolated DELP × (1-QV)
        ps_tgt = zero(FT)
        comp1 = zero(FT)
        for k in 1:Nz
            val = ((one(FT) - frac) * delp_0[ii, jj, k] +
                    frac * delp_1[ii, jj, k]) * (one(FT) - qv[ii, jj, k])
            ps_tgt, comp1 = _kahan_add(ps_tgt, comp1, val)
        end
        # Evolved column dp
        ps_evol = zero(FT)
        comp2 = zero(FT)
        for k in 1:Nz
            ps_evol, comp2 = _kahan_add(ps_evol, comp2, dp[ii, jj, k])
        end
        # Scale dp to match target PS
        scale = ps_tgt / max(ps_evol, eps(FT))
        for k in 1:Nz
            dp[ii, jj, k] *= scale
        end
    end
end

"""Per-column dp correction (moist basis): scale evolved dp so column sum matches
interpolated moist target PS."""
@kernel function _column_dp_correction_moist_kernel!(dp::AbstractArray{FT, 3},
                                                      @Const(delp_0), @Const(delp_1),
                                                      frac, Hp, Nz) where {FT}
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i; jj = Hp + j
        ps_tgt = zero(FT)
        comp1 = zero(FT)
        for k in 1:Nz
            val = (one(FT) - frac) * delp_0[ii, jj, k] + frac * delp_1[ii, jj, k]
            ps_tgt, comp1 = _kahan_add(ps_tgt, comp1, val)
        end
        ps_evol = zero(FT)
        comp2 = zero(FT)
        for k in 1:Nz
            ps_evol, comp2 = _kahan_add(ps_evol, comp2, dp[ii, jj, k])
        end
        scale = ps_tgt / max(ps_evol, eps(FT))
        for k in 1:Nz
            dp[ii, jj, k] *= scale
        end
    end
end

# ===========================================================================
# 1:1 PORT OF GCHP offline_tracer_advection (fv_tracer2d.F90)
#
# The functions below implement GCHP's tracer_2d algorithm exactly:
# - Called ONCE per met window (not per substep)
# - Works in dp-space (Pa) with rarea = 1/area
# - Per-level subcycling: ksplt(k) = int(1 + cmax(k))
# - fv_tp_2d computes fluxes only; update is external
# ===========================================================================

# ---------------------------------------------------------------------------
# dp continuity kernel: dp2 = dp1 + mfx_divergence × rarea
# Port of fv_tracer2d.F90:515-519
# ---------------------------------------------------------------------------

@kernel function _gchp_dp_evolve_kernel!(
    dp2, @Const(dp1), @Const(mfx), @Const(mfy), @Const(rarea), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(dp2)
        # dp2(i,j) = dp1(i,j,k) + (mfx(i,j,k) - mfx(i+1,j,k)
        #                         + mfy(i,j,k) - mfy(i,j+1,k)) * rarea(i,j)
        mfx_div = mfx[i, j, k] - mfx[i+1, j, k]
        mfy_div = mfy[i, j, k] - mfy[i, j+1, k]
        dp2[ii, jj, k] = dp1[ii, jj, k] + (mfx_div + mfy_div) * rarea[i, j]
    end
end

# ---------------------------------------------------------------------------
# q update from averaged Lin-Rood fluxes (dp-space)
# Port of fv_tracer2d.F90:544-546
#
# q_new = (q*dp1 + (fx_w - fx_e + fy_s - fy_n) * rarea) / dp2
# where fx = 0.5*(fx_in + fx_out) * mfx  (already mass-weighted)
# ---------------------------------------------------------------------------

@kernel function _gchp_q_update_kernel!(
    q_new, @Const(q), @Const(dp1), @Const(dp2),
    @Const(fx_in), @Const(fx_out), @Const(fy_in), @Const(fy_out),
    @Const(mfx), @Const(mfy), @Const(rarea), Hp
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i;  jj = Hp + j
        FT = eltype(q)
        half = FT(0.5)

        # Average Lin-Rood face values (dimensionless mixing ratio)
        avg_fx_w = half * (fx_out[i,   j, k] + fx_in[i,   j, k])
        avg_fx_e = half * (fx_out[i+1, j, k] + fx_in[i+1, j, k])
        avg_fy_s = half * (fy_out[i, j,   k] + fy_in[i, j,   k])
        avg_fy_n = half * (fy_out[i, j+1, k] + fy_in[i, j+1, k])

        # Tracer flux: face_value × mass_flux (Pa·m²)
        # Then divergence × rarea → Pa
        tfx_w = avg_fx_w * mfx[i, j, k]
        tfx_e = avg_fx_e * mfx[i+1, j, k]
        tfy_s = avg_fy_s * mfy[i, j, k]
        tfy_n = avg_fy_n * mfy[i, j+1, k]

        flux_div = ((tfx_w - tfx_e) + (tfy_s - tfy_n)) * rarea[i, j]

        dp1_val = dp1[ii, jj, k]
        dp2_val = dp2[ii, jj, k]

        # GCHP line 546: q = (q*dp1 + flux_div*rarea) / dp2
        q_new[ii, jj, k] = dp2_val > FT(100) * eps(FT) ?
            (q[ii, jj, k] * dp1_val + flux_div) / dp2_val : zero(FT)
    end
end

# ---------------------------------------------------------------------------
# fv_tp_2d_gchp_fluxes! — Lin-Rood face values only (no q/dp update)
#
# Port of tp_core.F90 fv_tp_2d (lines 108-242).
# Computes the 4 face value arrays (fx_in, fx_out, fy_in, fy_out) using
# Courant-number PPM and area-based pre-advection.
# Does NOT update q or dp. The caller handles the update.
#
# Output stored in ws_lr.{fx_in, fx_out, fy_in, fy_out}.
# ---------------------------------------------------------------------------

"""
    fv_tp_2d_gchp_fluxes!(q_panels, cx_panels, cy_panels,
                            xfx_panels, yfx_panels, area_panels,
                            grid, ::Val{ORD}, ws_lr)

Compute Lin-Rood averaged face values using Courant-PPM.
Does NOT update q or dp — only computes fluxes.

Output stored in ws_lr:
- `fx_in[p]`  — inner X face values (from original q)
- `fx_out[p]` — outer X face values (from Y-pre-advected q_i)
- `fy_in[p]`  — inner Y face values (from original q)
- `fy_out`    — outer Y face values (from X-pre-advected q_j), reused per panel
"""
function fv_tp_2d_gchp_fluxes!(q_panels, cx_panels, cy_panels,
                                 xfx_panels, yfx_panels, area_panels,
                                 grid::CubedSphereGrid, ::Val{ORD},
                                 ws_lr::LinRoodWorkspace) where ORD
    (; Hp, Nc, Nz) = grid

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Y-corners → inner Y-PPM (Courant) + area pre-advect → q_i
    # ═══════════════════════════════════════════════════════════════════════
    fill_panel_halos!(q_panels, grid)
    copy_corners!(q_panels, grid, 2)

    for_panels_nosync() do p
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _ppm_y_face_courant_kernel!(be, 256)(ws_lr.fy_in[p], q_panels[p], cy_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        _pre_advect_y_area_kernel!(be, 256)(ws_lr.q_buf[p], q_panels[p], ws_lr.fy_in[p],
                   yfx_panels[p], area_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end

    fill_panel_halos!(ws_lr.q_buf, grid)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: X-corners → outer X-PPM on q_i + inner X-PPM + area pre-advect
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 1)
    copy_corners!(q_panels, grid, 1)

    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _ppm_x_face_courant_kernel!(be, 256)(ws_lr.fx_out[p], ws_lr.q_buf[p], cx_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        _ppm_x_face_courant_kernel!(be, 256)(ws_lr.fx_in[p], q_panels[p], cx_panels[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end

    for_panels_nosync() do p
        copyto!(ws_lr.q_buf[p], q_panels[p])
    end
    for_panels_nosync() do p
        be = get_backend(q_panels[p])
        _pre_advect_x_area_kernel!(be, 256)(ws_lr.q_buf[p], q_panels[p], ws_lr.fx_in[p],
                   xfx_panels[p], area_panels[p], Hp; ndrange=(Nc, Nc, Nz))
    end

    # Exchange q_j halos from neighboring panels so the outer Y-PPM reads
    # pre-advected values at halo positions (not original q).
    fill_panel_halos!(ws_lr.q_buf, grid)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Y-corners on q_j → outer Y-PPM
    # fy_out is a single buffer, computed per-panel in Phase 3 of the
    # caller's update loop. We stop here — Phase 3 is done by the caller.
    # ═══════════════════════════════════════════════════════════════════════
    copy_corners!(ws_lr.q_buf, grid, 2)

    # Output: ws_lr.{fx_in, fx_out, fy_in} are per-panel arrays (6 each).
    # ws_lr.q_buf has q_j (X-pre-advected) with Y-corners filled.
    # Caller must compute outer Y-PPM (fy_out) per panel and consume immediately.
    return nothing
end

# ---------------------------------------------------------------------------
# gchp_tracer_2d! — Port of tracer_2d (fv_tracer2d.F90:336-578)
#
# Called ONCE per met window with full accumulated fluxes.
# Works in dp-space (Pa). Implements per-level subcycling.
#
# Arguments:
# - q_tracers: NamedTuple of tracer panels (haloed, mixing ratio, modified)
# - dp_panels: pressure thickness panels (haloed, Pa, modified → dpA)
# - mfx/mfy: mass fluxes (Pa·m², accumulated over dynamics timestep)
# - cx/cy: Courant numbers (dimensionless, accumulated)
# - xfx/yfx: area fluxes (m², precomputed)
# - area/rarea: cell area and reciprocal area
# - ws_lr: LinRoodWorkspace for face value buffers
# - dp2_work: workspace for evolved dp (haloed)
# ---------------------------------------------------------------------------

"""
    gchp_tracer_2d!(q_tracers, dp_panels, mfx, mfy, cx, cy,
                      xfx, yfx, area, rarea, grid, ::Val{ORD},
                      ws_lr, dp2_work)

Port of GCHP's `tracer_2d` (fv_tracer2d.F90:336-578).
Horizontal advection with per-level subcycling, called ONCE per met window.

All tracers in `q_tracers` are advected simultaneously, sharing the same
dp evolution (as in GCHP). `dp_panels` is modified in-place to contain
the post-advection pressure thickness (dpA).
"""
function gchp_tracer_2d!(q_tracers, dp_panels, mfx, mfy, cx, cy,
                           xfx, yfx, area, rarea,
                           grid::CubedSphereGrid{FT}, ::Val{ORD},
                           ws_lr::LinRoodWorkspace, dp2_work) where {FT, ORD}
    (; Hp, Nc, Nz) = grid

    # ── Step 1: Compute max |CX|, |CY| per level → nsplt ──────────────
    cmax_global = zero(FT)
    for p in 1:6
        cmax_global = max(cmax_global, FT(maximum(abs, cx[p])))
        cmax_global = max(cmax_global, FT(maximum(abs, cy[p])))
    end
    nsplt = max(1, Int(floor(1 + cmax_global)))

    # ── Step 2: Scale all fluxes by 1/nsplt ────────────────────────────
    if nsplt > 1
        frac = FT(1) / FT(nsplt)
        for_panels_nosync() do p
            cx[p]  .*= frac;  cy[p]  .*= frac
            mfx[p] .*= frac;  mfy[p] .*= frac
            xfx[p] .*= frac;  yfx[p] .*= frac
        end
    end

    # ── Step 3: Subcycled advection loop ───────────────────────────────
    for it in 1:nsplt
        # 3a. dp continuity: dp2 = dp1 + mfx_div × rarea
        for_panels_nosync() do p
            be = get_backend(dp_panels[p])
            _gchp_dp_evolve_kernel!(be, 256)(dp2_work[p], dp_panels[p], mfx[p], mfy[p], rarea[p],
                         Hp; ndrange=(Nc, Nc, Nz))
        end

        # 3b. For each tracer: compute face values, then update q
        for (_, q_t) in pairs(q_tracers)
            fv_tp_2d_gchp_fluxes!(q_t, cx, cy, xfx, yfx, area,
                                    grid, Val(ORD), ws_lr)

            # Phase 3 + update: per-panel
            for_panels_nosync() do p
                be = get_backend(q_t[p])
                _ppm_y_face_courant_kernel!(be, 256)(ws_lr.fy_out[p], ws_lr.q_buf[p], cy[p],
                           Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
                _gchp_q_update_kernel!(be, 256)(ws_lr.q_out[p], q_t[p], dp_panels[p], dp2_work[p],
                            ws_lr.fx_in[p], ws_lr.fx_out[p],
                            ws_lr.fy_in[p], ws_lr.fy_out[p],
                            mfx[p], mfy[p], rarea[p], Hp;
                            ndrange=(Nc, Nc, Nz))
            end

            for_panels_nosync() do p
                _copy_interior!(q_t[p], ws_lr.q_out[p], Hp, Nc, Nz)
            end
        end

        # 3c. dp1 = dp2 for next substep
        for_panels_nosync() do p
            copyto!(dp_panels[p], dp2_work[p])
        end
    end

    # ── Step 4: Restore flux scaling ───────────────────────────────────
    if nsplt > 1
        restore = FT(nsplt)
        for_panels_nosync() do p
            cx[p]  .*= restore;  cy[p]  .*= restore
            mfx[p] .*= restore;  mfy[p] .*= restore
            xfx[p] .*= restore;  yfx[p] .*= restore
        end
    end

    # dp_panels now contains dpA (evolved DELP after all horizontal transport)
    return nothing
end
