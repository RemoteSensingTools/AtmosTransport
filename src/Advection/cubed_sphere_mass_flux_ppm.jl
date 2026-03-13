# ---------------------------------------------------------------------------
# Putman & Lin (2007) PPM Mass-Flux Advection Kernels (Cubed-Sphere)
#
# Piecewise Parabolic Method variants for cubed-sphere mass flux advection.
# Integrates PPM subgrid distributions (_ppm_edge_values_*) with the existing
# mass-flux framework (same as Russell-Lerner, but with PPM edge values).
#
# Variants via Val{ORD} dispatcher (compile-time specialization):
#   ORD=4: Optimized PPM (LR96 + minmod)
#   ORD=5: PPM with Huynh's 2nd constraint (quasi-monotonic)
#   ORD=6: Quasi-5th order (non-monotonic)
#   ORD=7: ORD=5 + special CS face edge treatment (RECOMMENDED)
#
# File structure mirrors cubed_sphere_mass_flux.jl:
#   - X-direction kernel + sweep
#   - Y-direction kernel + sweep
#   - Z-direction column kernel (no PPM variant needed, uses positivity guards)
#   - Strang splitting wrapper
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

# ---------------------------------------------------------------------------
# Safe mixing ratio extraction (guards against zero air mass)
# ---------------------------------------------------------------------------

@inline function _safe_mixing_ratio(rm::FT, m::FT) where FT
    """Extract mixing ratio, returning zero if air mass too small."""
    return m > 100 * eps(FT) ? rm / m : zero(FT)
end

# ---------------------------------------------------------------------------
# X-Direction PPM Kernel
# ---------------------------------------------------------------------------

"""
    _massflux_x_cs_kernel_ppm!(rm_new, rm, m_new, m, am, Hp, Nc, ::Val{ORD})

X-direction mass-flux advection using Putman & Lin PPM (all ORD variants).

Uses PPM subgrid distribution (ORD=4,5,6,7) instead of Russell-Lerner slopes.
Fluxes computed identically to Russell-Lerner path (TM5-style mass conserving).
"""
@kernel function _massflux_x_cs_kernel_ppm!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(am),
    Hp, Nc, ::Val{ORD}
) where ORD
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        # Extract wider stencil (7-point: i-3 to i+3) for PPM edge computation.
        # We need edges of cells i-1, i, and i+1 in the flux formula.
        # Each cell's edge values require a 5-point centered stencil.
        c_imm2 = _safe_mixing_ratio(rm[ii - 3, jj, k], m[ii - 3, jj, k])  # cell i-3
        c_imm  = _safe_mixing_ratio(rm[ii - 2, jj, k], m[ii - 2, jj, k])  # cell i-2
        c_im   = _safe_mixing_ratio(rm[ii - 1, jj, k], m[ii - 1, jj, k])  # cell i-1
        c_i    = _safe_mixing_ratio(rm[ii,     jj, k], m[ii,     jj, k])  # cell i
        c_ip   = _safe_mixing_ratio(rm[ii + 1, jj, k], m[ii + 1, jj, k])  # cell i+1
        c_ipp  = _safe_mixing_ratio(rm[ii + 2, jj, k], m[ii + 2, jj, k])  # cell i+2
        c_ipp2 = _safe_mixing_ratio(rm[ii + 3, jj, k], m[ii + 3, jj, k])  # cell i+3

        # Compute PPM edge values using 5-point centered stencils for each cell.
        # Dispatcher selects ORD variant at compile time.
        q_L_im, q_R_im = _ppm_edge_values(c_imm2, c_imm, c_im,  c_i,   c_ip,   Val(ORD))  # edges of cell i-1
        q_L_i,  q_R_i  = _ppm_edge_values(c_imm,  c_im,  c_i,   c_ip,  c_ipp,  Val(ORD))  # edges of cell i
        q_L_ip, q_R_ip = _ppm_edge_values(c_im,   c_i,   c_ip,  c_ipp, c_ipp2, Val(ORD))  # edges of cell i+1

        # ORD=7: discontinuous treatment at CS panel boundaries (Putman & Lin App. C)
        # Average two one-sided extrapolations at the face discontinuity.
        # Compile-time eliminated for ORD != 7.
        if ORD == 7
            if i == 1
                # Left panel boundary: face between halo cell i-1 and interior cell i=1
                q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_im, c_imm, c_i, c_ip)
                q_R_im = q_bdy
                q_L_i = q_bdy
            end
            if i == Nc
                # Right panel boundary: face between interior cell i=Nc and halo cell i+1
                q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_i, c_im, c_ip, c_ipp)
                q_R_i = q_bdy
                q_L_ip = q_bdy
            end
        end

        # Colella & Woodward monotonicity constraint: flatten at local extrema.
        # If both edge departures have opposite sign to expected monotonicity,
        # the reconstruction overshoots — set edges to cell mean (first-order).
        if (q_R_im - c_im) * (c_im - q_L_im) <= zero(FT); q_L_im = c_im; q_R_im = c_im; end
        if (q_R_i  - c_i)  * (c_i  - q_L_i)  <= zero(FT); q_L_i  = c_i;  q_R_i  = c_i;  end
        if (q_R_ip - c_ip) * (c_ip - q_L_ip) <= zero(FT); q_L_ip = c_ip; q_R_ip = c_ip; end

        # PPM correction terms (mass-weighted edge departures), clamped for positivity.
        # Same role as sx in Russell-Lerner: |correction| ≤ rm prevents negative tracer.
        sx_im = clamp(m[ii - 1, jj, k] * (q_R_im - c_im), -rm[ii - 1, jj, k], rm[ii - 1, jj, k])
        sx_i  = clamp(m[ii,     jj, k] * (q_R_i  - c_i),  -rm[ii,     jj, k], rm[ii,     jj, k])
        # Left-edge corrections (c - q_L) for negative-flow branches
        sx_L_i  = clamp(m[ii,     jj, k] * (c_i  - q_L_i),  -rm[ii,     jj, k], rm[ii,     jj, k])
        sx_L_ip = clamp(m[ii + 1, jj, k] * (c_ip - q_L_ip), -rm[ii + 1, jj, k], rm[ii + 1, jj, k])


        # Flux at left face (face index i in am, interior-indexed)
        am_l = am[i, j, k]
        flux_left = if am_l >= zero(FT)
            alpha = am_l / m[ii - 1, jj, k]
            alpha * (rm[ii - 1, jj, k] + (one(FT) - alpha) * sx_im)
        else
            alpha = am_l / m[ii, jj, k]
            alpha * (rm[ii, jj, k] - (one(FT) + alpha) * sx_L_i)
        end

        # Flux at right face (face i+1 in am)
        am_r = am[i + 1, j, k]
        flux_right = if am_r >= zero(FT)
            alpha = am_r / m[ii, jj, k]
            alpha * (rm[ii, jj, k] + (one(FT) - alpha) * sx_i)
        else
            alpha = am_r / m[ii + 1, jj, k]
            alpha * (rm[ii + 1, jj, k] - (one(FT) + alpha) * sx_L_ip)
        end

        # Update tracers and air mass
        rm_new[ii, jj, k] = rm[ii, jj, k] + flux_left - flux_right
        m_new[ii, jj, k]  = m[ii, jj, k]  + am[i, j, k] - am[i + 1, j, k]
    end
end

# ---------------------------------------------------------------------------
# Wrapper function: X-direction sweep with PPM
# ---------------------------------------------------------------------------

function _sweep_x_ppm!(rm_panels, m_panels, am_panels, grid::CubedSphereGrid,
                       ::Val{ORD}, ws; cfl_limit=0.95) where ORD
    """X-direction Strang split sweep using PPM advection (ORD=4,5,6,7)."""
    FT = floattype(grid)
    Hp = grid.Hp
    Nc = grid.Nc
    Nz = grid.Nz

    # 1. Fill halos for tracer and air mass
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    # 2. Compute max CFL and determine subcycling
    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_x_cs(am_panels[p], m_panels[p], ws.cfl_x, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))

    # 3. Subdivide fluxes if needed
    if n_sub > 1
        for p in 1:6
            am_panels[p] .*= inv(FT(n_sub))
        end
    end

    # 4. Perform n_sub subcycles
    for _ in 1:n_sub
        for p in 1:6
            # Launch PPM kernel for this panel (ws.rm_buf is a single shared buffer)
            backend = get_backend(rm_panels[p])
            k! = _massflux_x_cs_kernel_ppm!(backend, 256)
            k!(ws.rm_buf, rm_panels[p], ws.m_buf, m_panels[p], am_panels[p],
               Hp, Nc, Val(ORD); ndrange=(Nc, Nc, Nz))
            synchronize(backend)

            # Copy interior cells from buffer back to panel
            _copy_interior!(rm_panels[p], ws.rm_buf, Hp, Nc, Nz)
            _copy_interior!(m_panels[p], ws.m_buf, Hp, Nc, Nz)
        end
    end

    # Restore original flux magnitudes after subcycling
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; am_panels[p] .*= fwd; end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Y-Direction PPM Kernel (similar structure)
# ---------------------------------------------------------------------------

@kernel function _massflux_y_cs_kernel_ppm!(
    rm_new, @Const(rm), m_new, @Const(m), @Const(bm),
    Hp, Nc, ::Val{ORD}
) where ORD
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        # Extract wider stencil (7-point: j-3 to j+3) for PPM edge computation in j-direction.
        # We need edges of cells j-1, j, and j+1 in the flux formula.
        c_jmm2 = _safe_mixing_ratio(rm[ii, jj - 3, k], m[ii, jj - 3, k])  # cell j-3
        c_jmm  = _safe_mixing_ratio(rm[ii, jj - 2, k], m[ii, jj - 2, k])  # cell j-2
        c_jm   = _safe_mixing_ratio(rm[ii, jj - 1, k], m[ii, jj - 1, k])  # cell j-1
        c_j    = _safe_mixing_ratio(rm[ii, jj,     k], m[ii, jj,     k])  # cell j
        c_jp   = _safe_mixing_ratio(rm[ii, jj + 1, k], m[ii, jj + 1, k])  # cell j+1
        c_jpp  = _safe_mixing_ratio(rm[ii, jj + 2, k], m[ii, jj + 2, k])  # cell j+2
        c_jpp2 = _safe_mixing_ratio(rm[ii, jj + 3, k], m[ii, jj + 3, k])  # cell j+3

        # Compute PPM edge values using 5-point centered stencils for each cell.
        # Dispatcher selects ORD variant at compile time.
        q_L_jm, q_R_jm = _ppm_edge_values(c_jmm2, c_jmm, c_jm,  c_j,   c_jp,   Val(ORD))  # edges of cell j-1
        q_L_j,  q_R_j  = _ppm_edge_values(c_jmm,  c_jm,  c_j,   c_jp,  c_jpp,  Val(ORD))  # edges of cell j
        q_L_jp, q_R_jp = _ppm_edge_values(c_jm,   c_j,   c_jp,  c_jpp, c_jpp2, Val(ORD))  # edges of cell j+1

        # ORD=7: discontinuous treatment at CS panel boundaries (Putman & Lin App. C)
        if ORD == 7
            if j == 1
                # South panel boundary: face between halo cell j-1 and interior cell j=1
                q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_jm, c_jmm, c_j, c_jp)
                q_R_jm = q_bdy
                q_L_j = q_bdy
            end
            if j == Nc
                # North panel boundary: face between interior cell j=Nc and halo cell j+1
                q_bdy = _ppm_face_edge_value_ord7_discontinuous(c_j, c_jm, c_jp, c_jpp)
                q_R_j = q_bdy
                q_L_jp = q_bdy
            end
        end

        # Colella & Woodward monotonicity constraint: flatten at local extrema.
        if (q_R_jm - c_jm) * (c_jm - q_L_jm) <= zero(FT); q_L_jm = c_jm; q_R_jm = c_jm; end
        if (q_R_j  - c_j)  * (c_j  - q_L_j)  <= zero(FT); q_L_j  = c_j;  q_R_j  = c_j;  end
        if (q_R_jp - c_jp) * (c_jp - q_L_jp) <= zero(FT); q_L_jp = c_jp; q_R_jp = c_jp; end

        # PPM correction terms (mass-weighted edge departures), clamped for positivity.
        sy_jm = clamp(m[ii, jj - 1, k] * (q_R_jm - c_jm), -rm[ii, jj - 1, k], rm[ii, jj - 1, k])
        sy_j  = clamp(m[ii, jj,     k] * (q_R_j  - c_j),  -rm[ii, jj,     k], rm[ii, jj,     k])
        # Left-edge corrections (c - q_L) for negative-flow branches
        sy_L_j  = clamp(m[ii, jj,     k] * (c_j  - q_L_j),  -rm[ii, jj,     k], rm[ii, jj,     k])
        sy_L_jp = clamp(m[ii, jj + 1, k] * (c_jp - q_L_jp), -rm[ii, jj + 1, k], rm[ii, jj + 1, k])


        # Flux at south face (j in bm, interior-indexed)
        bm_s = bm[i, j, k]
        flux_south = if bm_s >= zero(FT)
            alpha = bm_s / m[ii, jj - 1, k]
            alpha * (rm[ii, jj - 1, k] + (one(FT) - alpha) * sy_jm)
        else
            alpha = bm_s / m[ii, jj, k]
            alpha * (rm[ii, jj, k] - (one(FT) + alpha) * sy_L_j)
        end

        # Flux at north face (j+1 in bm)
        bm_n = bm[i, j + 1, k]
        flux_north = if bm_n >= zero(FT)
            alpha = bm_n / m[ii, jj, k]
            alpha * (rm[ii, jj, k] + (one(FT) - alpha) * sy_j)
        else
            alpha = bm_n / m[ii, jj + 1, k]
            alpha * (rm[ii, jj + 1, k] - (one(FT) + alpha) * sy_L_jp)
        end

        # Update tracers and air mass
        rm_new[ii, jj, k] = rm[ii, jj, k] + flux_south - flux_north
        m_new[ii, jj, k]  = m[ii, jj, k]  + bm[i, j, k] - bm[i, j + 1, k]
    end
end

function _sweep_y_ppm!(rm_panels, m_panels, bm_panels, grid::CubedSphereGrid,
                       ::Val{ORD}, ws; cfl_limit=0.95) where ORD
    """Y-direction Strang split sweep using PPM advection."""
    FT = floattype(grid)
    Hp = grid.Hp
    Nc = grid.Nc
    Nz = grid.Nz

    # Similar to X sweep (fill halos, compute CFL, subdivide, launch kernel)
    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    max_cfl = zero(FT)
    for p in 1:6
        max_cfl = max(max_cfl, max_cfl_y_cs(bm_panels[p], m_panels[p], ws.cfl_y, Hp))
    end
    n_sub = max(1, ceil(Int, max_cfl / cfl_limit))

    if n_sub > 1
        for p in 1:6
            bm_panels[p] .*= inv(FT(n_sub))
        end
    end

    for _ in 1:n_sub
        for p in 1:6
            # Launch PPM kernel for this panel (ws.rm_buf is a single shared buffer)
            backend = get_backend(rm_panels[p])
            k! = _massflux_y_cs_kernel_ppm!(backend, 256)
            k!(ws.rm_buf, rm_panels[p], ws.m_buf, m_panels[p], bm_panels[p],
               Hp, Nc, Val(ORD); ndrange=(Nc, Nc, Nz))
            synchronize(backend)

            # Copy interior cells from buffer back to panel
            _copy_interior!(rm_panels[p], ws.rm_buf, Hp, Nc, Nz)
            _copy_interior!(m_panels[p], ws.m_buf, Hp, Nc, Nz)
        end
    end

    # Restore original flux magnitudes after subcycling
    if n_sub > 1
        fwd = FT(n_sub)
        for p in 1:6; bm_panels[p] .*= fwd; end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Divergence Damping (del-2 diffusion on mixing ratio)
#
# FV3-style horizontal diffusion to suppress grid imprinting at panel boundaries.
# Conservative flux-form Laplacian: face fluxes telescope exactly → mass conserving.
# Applied once before the first Strang sweep (not per-subcycle).
#
# Reference: tp_core.F90:deln_flux (simplified to del-2 for cubed-sphere)
# ---------------------------------------------------------------------------

@kernel function _divergence_damping_cs_kernel!(rm_new, @Const(rm), @Const(m), damp, Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ii = Hp + i
        jj = Hp + j
        FT = eltype(rm)

        # Mixing ratios at cell center and 4 face neighbors
        m_ij = m[ii, jj, k]
        c_ij = _safe_mixing_ratio(rm[ii, jj, k], m_ij)

        m_xm = m[ii - 1, jj, k]; c_xm = _safe_mixing_ratio(rm[ii - 1, jj, k], m_xm)
        m_xp = m[ii + 1, jj, k]; c_xp = _safe_mixing_ratio(rm[ii + 1, jj, k], m_xp)
        m_ym = m[ii, jj - 1, k]; c_ym = _safe_mixing_ratio(rm[ii, jj - 1, k], m_ym)
        m_yp = m[ii, jj + 1, k]; c_yp = _safe_mixing_ratio(rm[ii, jj + 1, k], m_yp)

        # Face-averaged air mass for conservative flux form
        m_face_xm = FT(0.5) * (m_xm + m_ij)
        m_face_xp = FT(0.5) * (m_xp + m_ij)
        m_face_ym = FT(0.5) * (m_ym + m_ij)
        m_face_yp = FT(0.5) * (m_yp + m_ij)

        # Conservative Laplacian: net diffusive flux into cell from all 4 faces
        diff = m_face_xm * (c_xm - c_ij) + m_face_xp * (c_xp - c_ij) +
               m_face_ym * (c_ym - c_ij) + m_face_yp * (c_yp - c_ij)

        rm_new[ii, jj, k] = rm[ii, jj, k] + FT(damp) * diff
    end
end

"""
    apply_divergence_damping_cs!(rm_panels, m_panels, grid, ws, damp_coeff)

Apply conservative del-2 divergence damping to tracer panels on cubed-sphere grid.
Mass-conserving flux-form Laplacian diffusion on mixing ratio (c = rm/m).
Typical `damp_coeff` values: 0.02–0.05 for mild smoothing of panel-boundary noise.
"""
function apply_divergence_damping_cs!(rm_panels, m_panels, grid::CubedSphereGrid, ws, damp_coeff)
    FT = floattype(grid)
    Hp = grid.Hp
    Nc = grid.Nc
    Nz = grid.Nz

    fill_panel_halos!(rm_panels, grid)
    fill_panel_halos!(m_panels, grid)

    for p in 1:6
        backend = get_backend(rm_panels[p])
        k! = _divergence_damping_cs_kernel!(backend, 256)
        k!(ws.rm_buf, rm_panels[p], m_panels[p], FT(damp_coeff), Hp;
           ndrange=(Nc, Nc, Nz))
        synchronize(backend)
        _copy_interior!(rm_panels[p], ws.rm_buf, Hp, Nc, Nz)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Main Strang Split Wrapper for PPM
# ---------------------------------------------------------------------------

"""
    strang_split_massflux_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                               grid, ::Val{ORD}, ws; cfl_limit=0.95, damp_coeff=0.0)

Strang-split mass-flux advection using Putman & Lin PPM (all ORD variants).

Sequence: [optional damping] → X → Y → Z → Z → Y → X
where Z uses the standard column-sequential kernel (PPM not needed vertically).

# Keyword arguments
- `cfl_limit=0.95`: maximum CFL before subcycling
- `damp_coeff=0.0`: del-2 divergence damping coefficient (0 = off, typical: 0.02–0.05)

Dispatch on Val{ORD} ensures compile-time kernel specialization.
"""
function strang_split_massflux_ppm!(rm_panels, m_panels, am_panels, bm_panels, cm_panels,
                                    grid::CubedSphereGrid, ::Val{ORD}, ws;
                                    cfl_limit=0.95, damp_coeff=0.0) where ORD
    # Optional divergence damping (applied once before Strang sweeps)
    if damp_coeff > 0
        apply_divergence_damping_cs!(rm_panels, m_panels, grid, ws, damp_coeff)
    end

    # X → Y → Z → Z → Y → X (Strang splitting)
    _sweep_x_ppm!(rm_panels, m_panels, am_panels, grid, Val(ORD), ws; cfl_limit)
    _sweep_y_ppm!(rm_panels, m_panels, bm_panels, grid, Val(ORD), ws; cfl_limit)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, true, ws)
    _sweep_z!(rm_panels, m_panels, cm_panels, grid, true, ws)
    _sweep_y_ppm!(rm_panels, m_panels, bm_panels, grid, Val(ORD), ws; cfl_limit)
    _sweep_x_ppm!(rm_panels, m_panels, am_panels, grid, Val(ORD), ws; cfl_limit)

    return nothing
end
