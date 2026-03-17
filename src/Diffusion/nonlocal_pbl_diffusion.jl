# ---------------------------------------------------------------------------
# Non-local PBL diffusion — Holtslag & Boville (1993) counter-gradient scheme
#
# Extends the local K-diffusion (PBLDiffusion / pbl_diffusion.jl) with a
# counter-gradient term γ_c that represents non-local transport by organized
# thermals in convective (unstable) boundary layers.
#
# Physics (Holtslag & Moeng 1991, Holtslag & Boville 1993, GEOS-Chem VDIFF):
#   flux = -Kz × (∂c/∂z - γ_c)
#
# The counter-gradient term (only in unstable BL, L < 0):
#   γ_c = fak × sffrac × (w'c')_sfc / (w_s × h)
#
# where:
#   fak    = 8.5 (GEOS-Chem vdiff_mod.F90 tuning constant)
#   sffrac = 0.1 (surface layer fraction)
#   (w'c')_sfc = surface tracer flux [ppm·m/s]
#   w_s = w* = (g/θ × H_kin × h)^(1/3) — convective velocity scale
#   h = PBL height
#
# The tridiagonal uses pressure-based diffusion coefficients (GeosChem
# vdiff_mod.F90): D = Kz × (g/R)² × (p_int/T)² / (Δp_mid × Δp_k),
# ensuring D(k) × Δp(k) = D(k±1) × Δp(k±1) → exact mass conservation.
# RHS gets an additive source from the counter-gradient flux divergence:
#   (I - dt·D) c_new = c_old + dt·S_nl
#
# where S_nl[k] = (g/R)/Δp × γ_c × [potbar_b × Kz_b - potbar_a × Kz_a].
# A per-column mass correction (Jintai Lin 2018) ensures roundoff-level
# conservation of Σ q × Δp.
#
# In stable conditions (L > 0), γ_c = 0 and the scheme degenerates to
# the local PBLDiffusion.
#
# Surface tracer flux: if sfc_flux[i,j] != 0, use it directly (GEOS-Chem
# approach — emissions passed explicitly). Otherwise diagnose from the
# tracer gradient at the lowest interface.
#
# Reuses _pbl_kz from pbl_diffusion.jl (shared, not duplicated).
# ---------------------------------------------------------------------------

# Uses imports from pbl_diffusion.jl (included before this file):
#   CubedSphereGrid, LatitudeLongitudeGrid, floattype, PlanetParameters
#   @kernel, @index, synchronize, get_backend

# =====================================================================
# KA kernel: non-local PBL diffusion with counter-gradient + Thomas solve
# =====================================================================

"""
Grid-agnostic KA kernel for non-local PBL diffusion (Holtslag-Boville).

Identical to `_pbl_diffuse_kernel!` except:
- Computes counter-gradient γ_c in unstable boundary layers
- Adds non-local source S_nl to the Thomas RHS at each BL level
- Accepts `sfc_flux` (2D) for explicit surface tracer flux

The `sfc_flux` array provides per-column surface tracer flux [ppm·m/s].
If a column's `sfc_flux == 0`, the flux is diagnosed from the tracer gradient.
"""
@kernel function _nonlocal_pbl_diffuse_kernel!(
    arr, @Const(m), @Const(delp), @Const(pblh), @Const(ustar),
    @Const(hflux), @Const(t2m), @Const(sfc_flux), w_scratch,
    i_off, j_off, Nz, dt,
    kappa_vk, grav, cp_dry, rho_ref,
    β_h, Kz_bg, Kz_min, Kz_max, fak, sffrac,
    ::Val{tracer_mode}
) where tracer_mode
    i, j = @index(Global, NTuple)
    ii = i_off + i
    jj = j_off + j

    FT = eltype(arr)
    κ_vk = FT(kappa_vk)
    g = FT(grav)
    cp = FT(cp_dry)
    ρ = FT(rho_ref)
    R_d = cp / FT(3.5)

    # --- Surface fields ---
    h_pbl = max(FT(pblh[ii, jj]), FT(100))
    us    = max(FT(ustar[ii, jj]), FT(0.01))
    H_sfc = FT(hflux[ii, jj])
    T_sfc = max(FT(t2m[ii, jj]), FT(200))

    # --- Obukhov length ---
    H_kin = H_sfc / (ρ * cp)
    H_safe = H_kin + sign(H_kin + FT(1e-20)) * FT(1e-10)
    L_ob = -T_sfc * us^3 / (κ_vk * g * H_safe)

    # --- Compute column height using hydrostatic with pressure-dependent density ---
    R_T_over_g = R_d * T_sfc / g

    z_col = FT(0)
    p_top = FT(0)
    @inbounds for k in 1:Nz
        delp_k = delp[ii, jj, k]
        p_bot_k = p_top + delp_k
        p_mid_k = max((p_top + p_bot_k) / FT(2), FT(1))
        z_col += delp_k * R_T_over_g / p_mid_k
        p_top = p_bot_k
    end

    # --- Convert rm → mixing ratio (CS only) ---
    if tracer_mode === :rm
        @inbounds for k in 1:Nz
            _m = m[ii, jj, k]
            arr[ii, jj, k] = _m > FT(0) ? arr[ii, jj, k] / _m : FT(0)
        end
    end

    # --- Compute counter-gradient γ_c (unstable BL only) ---
    γ_c = FT(0)
    if L_ob < FT(0)
        # Convective velocity scale w* = (g/θ × H_kin × h)^(1/3)
        # H_kin = H_sfc / (ρ × cp) is the kinematic heat flux [K·m/s]
        # Only compute if H_kin > 0 (upward heat flux = convective)
        if H_kin > FT(0)
            w_star = cbrt(g / T_sfc * H_kin * h_pbl)
            w_s = max(w_star, FT(0.01))  # safety minimum

            # Surface tracer flux: use explicit value from emissions.
            # GCHP passes actual emission fluxes to the PBL scheme; the
            # counter-gradient term only activates for tracers with real
            # surface sources. When sfc_flux == 0 (no emissions for this
            # tracer), F_sfc stays zero → γ_c = 0 → no counter-gradient.
            # The previous gradient-diagnosis fallback was removed because
            # it treated advection-induced vertical gradients as real surface
            # fluxes, creating spurious hemispheric biases (0.17 ppm/day
            # in a no-emission CO2 run with asymmetric SH/NH met fields).
            F_sfc = FT(sfc_flux[ii, jj])

            γ_c = FT(fak) * FT(sffrac) * F_sfc / (w_s * h_pbl)
        end
    end

    # --- Forward sweep: build tridiagonal + Thomas elimination ---
    # Pressure-based diffusion coefficients (GeosChem vdiff_mod.F90):
    #   D = Kz × (g/R)² × (p_int/T)² / (Δp_mid × Δp_k)
    # ensures D(k) × Δp(k) = D(k±1) × Δp(k±1) at each interface → mass conservation
    z_above = z_col
    w_prev = FT(0)
    g_prev = FT(0)
    p_top_acc = FT(0)
    p_mid_prev_k = FT(0)

    # Pressure-based coefficient constants
    gor = g / R_d
    gorsq = gor * gor
    T_inv = FT(1) / T_sfc

    # Pre-diffusion mass-weighted sum (for per-column mass correction)
    mass_sum_before = FT(0)
    @inbounds for k in 1:Nz
        mass_sum_before += arr[ii, jj, k] * delp[ii, jj, k]
    end

    @inbounds for k in 1:Nz
        delp_k = delp[ii, jj, k]
        p_bot_k = p_top_acc + delp_k
        p_mid_k = max((p_top_acc + p_bot_k) / FT(2), FT(1))
        dz_k = delp_k * R_T_over_g / p_mid_k
        z_below = z_above - dz_k

        # Interface pressures for this level
        p_int_above = p_top_acc   # pressure at top of level k
        p_int_below = p_bot_k     # pressure at bottom of level k

        # Kz at interface above (between k-1 and k)
        D_above = FT(0)
        Kz_a = FT(0)
        if k > 1
            Kz_a = _pbl_kz(z_above, h_pbl, us, L_ob, κ_vk,
                            β_h, Kz_bg, Kz_min, Kz_max, FT)
            potbar_a = p_int_above * T_inv
            dp_mid_a = max(p_mid_k - p_mid_prev_k, FT(0.01))
            D_above = Kz_a * gorsq * potbar_a * potbar_a / (dp_mid_a * delp_k)
        end

        # Kz at interface below (between k and k+1)
        D_below = FT(0)
        Kz_b = FT(0)
        if k < Nz
            Kz_b = _pbl_kz(z_below, h_pbl, us, L_ob, κ_vk,
                            β_h, Kz_bg, Kz_min, Kz_max, FT)
            potbar_b = p_int_below * T_inv
            delp_next = delp[ii, jj, k + 1]
            p_mid_next = max(p_bot_k + delp_next / FT(2), FT(1))
            dp_mid_b = max(p_mid_next - p_mid_k, FT(0.01))
            D_below = Kz_b * gorsq * potbar_b * potbar_b / (dp_mid_b * delp_k)
        end

        p_mid_prev_k = p_mid_k
        p_top_acc = p_bot_k

        # Tridiagonal coefficients: (I - dt*D) c_new = c_old + dt*S_nl
        a_k = k > 1  ? -dt * D_above : FT(0)
        b_k = FT(1) + dt * (D_above + D_below)
        c_k = k < Nz ? -dt * D_below : FT(0)

        # RHS: tracer value + non-local counter-gradient source (pressure-based)
        c_val = arr[ii, jj, k]

        # Non-local source (pressure coords, GeosChem vdiff_mod.F90):
        # S_nl = dt × (g/R) / Δp × [potbar_b × Kz_b - potbar_a × Kz_a] × γ_c
        # Sign: (below - above) → removes tracer from surface, adds to upper BL
        if γ_c != FT(0) && z_below < h_pbl
            pb_a = p_int_above * T_inv
            pb_b = p_int_below * T_inv
            S_nl = dt * gor * γ_c / delp_k * (pb_b * Kz_b - pb_a * Kz_a)
            c_val += S_nl
        end

        # Thomas forward elimination
        if k == 1
            w_k = c_k / b_k
            g_k = c_val / b_k
        else
            denom = b_k - a_k * w_prev
            w_k = c_k / denom
            g_k = (c_val - a_k * g_prev) / denom
        end

        # Store w in workspace, g in arr
        w_scratch[ii, jj, k] = w_k
        arr[ii, jj, k] = g_k
        w_prev = w_k
        g_prev = g_k

        z_above = z_below
    end

    # --- Back-substitution using stored w-factors ---
    @inbounds for k in (Nz - 1):-1:1
        arr[ii, jj, k] -= w_scratch[ii, jj, k] * arr[ii, jj, k + 1]
    end

    # --- Per-column mass correction (GeosChem vdiff_mod.F90, Jintai Lin fix) ---
    # Ensure Σ q × Δp is conserved exactly despite floating-point roundoff
    mass_sum_after = FT(0)
    @inbounds for k in 1:Nz
        mass_sum_after += arr[ii, jj, k] * delp[ii, jj, k]
    end
    if mass_sum_after > FT(0) && mass_sum_before > FT(0)
        _mass_corr = mass_sum_before / mass_sum_after
        @inbounds for k in 1:Nz
            arr[ii, jj, k] *= _mass_corr
        end
    end

    # --- Convert mixing ratio back to tracer mass (CS only) ---
    if tracer_mode === :rm
        @inbounds for k in 1:Nz
            arr[ii, jj, k] *= m[ii, jj, k]
        end
    end
end

# =====================================================================
# Dispatch: CubedSphereGrid — loop over 6 haloed panels
# =====================================================================

"""
    diffuse_nonlocal_pbl!(rm_panels, m_panels, delp_panels,
                          pblh_panels, ustar_panels, hflux_panels, t2m_panels,
                          sfc_flux_panels, w_scratch_panels,
                          diff, grid, dt, planet)

Apply non-local PBL diffusion (Holtslag-Boville) to cubed-sphere panel arrays.

`sfc_flux_panels` is a 6-tuple of 2D arrays with per-column surface tracer flux
[ppm·m/s]. Pass zeros to use gradient-diagnosed flux.
"""
function diffuse_nonlocal_pbl!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                                delp_panels::NTuple{6},
                                pblh_panels::NTuple{6}, ustar_panels::NTuple{6},
                                hflux_panels::NTuple{6}, t2m_panels::NTuple{6},
                                sfc_flux_panels::NTuple{6},
                                w_scratch_panels::NTuple{6},
                                diff::NonLocalPBLDiffusion, grid::CubedSphereGrid, dt,
                                planet::PlanetParameters)
    FT = eltype(rm_panels[1])
    backend = get_backend(rm_panels[1])
    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz
    kernel! = _nonlocal_pbl_diffuse_kernel!(backend, 256)
    for p in 1:6
        kernel!(rm_panels[p], m_panels[p], delp_panels[p],
                pblh_panels[p], ustar_panels[p], hflux_panels[p], t2m_panels[p],
                sfc_flux_panels[p], w_scratch_panels[p],
                Hp, Hp, Nz, FT(dt),
                FT(planet.kappa_vk), FT(planet.gravity),
                FT(planet.cp_dry), FT(planet.rho_ref),
                FT(diff.β_h), FT(diff.Kz_bg), FT(diff.Kz_min), FT(diff.Kz_max),
                FT(diff.fak), FT(diff.sffrac),
                Val(:rm); ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

# Convenience: pass nothing for sfc_flux → use zeros (gradient diagnosis)
function diffuse_nonlocal_pbl!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                                delp_panels::NTuple{6},
                                pblh_panels::NTuple{6}, ustar_panels::NTuple{6},
                                hflux_panels::NTuple{6}, t2m_panels::NTuple{6},
                                ::Nothing,
                                w_scratch_panels::NTuple{6},
                                diff::NonLocalPBLDiffusion, grid::CubedSphereGrid, dt,
                                planet::PlanetParameters)
    # Create zero sfc_flux panels for gradient-based diagnosis
    Nc, Hp = grid.Nc, grid.Hp
    FT = eltype(rm_panels[1])
    zero_sfc = ntuple(_ -> fill!(similar(rm_panels[1], FT, Nc + 2Hp, Nc + 2Hp), FT(0)), 6)
    diffuse_nonlocal_pbl!(rm_panels, m_panels, delp_panels,
                           pblh_panels, ustar_panels, hflux_panels, t2m_panels,
                           zero_sfc, w_scratch_panels,
                           diff, grid, dt, planet)
end

# Also dispatch via diffuse_pbl! for seamless integration with run loop
function diffuse_pbl!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                       delp_panels::NTuple{6},
                       pblh_panels::NTuple{6}, ustar_panels::NTuple{6},
                       hflux_panels::NTuple{6}, t2m_panels::NTuple{6},
                       w_scratch_panels::NTuple{6},
                       diff::NonLocalPBLDiffusion, grid::CubedSphereGrid, dt,
                       planet::PlanetParameters)
    diffuse_nonlocal_pbl!(rm_panels, m_panels, delp_panels,
                           pblh_panels, ustar_panels, hflux_panels, t2m_panels,
                           nothing, w_scratch_panels,
                           diff, grid, dt, planet)
end

# =====================================================================
# Dispatch: LatitudeLongitudeGrid — operate on mixing-ratio tracers
# =====================================================================

"""
    diffuse_nonlocal_pbl!(tracers, delp, pblh, ustar, hflux, t2m,
                          sfc_flux, w_scratch, diff, grid, dt, planet)

Apply non-local PBL diffusion to lat-lon tracers (mixing ratios).

`sfc_flux` is a 2D array with per-column surface tracer flux [ppm·m/s],
or `nothing` to diagnose from gradient.
"""
function diffuse_nonlocal_pbl!(tracers::NamedTuple, delp,
                                pblh, ustar, hflux, t2m,
                                sfc_flux,
                                w_scratch,
                                diff::NonLocalPBLDiffusion, grid::LatitudeLongitudeGrid, dt,
                                planet::PlanetParameters)
    FT = floattype(grid)
    for tracer in values(tracers)
        arr = tracer_data(tracer)
        backend = get_backend(arr)
        Nx, Ny = size(arr, 1), size(arr, 2)
        Nz = size(arr, 3)

        # Use provided sfc_flux or zeros for gradient diagnosis
        sf = sfc_flux === nothing ? fill!(similar(arr, FT, Nx, Ny), FT(0)) : sfc_flux

        kernel! = _nonlocal_pbl_diffuse_kernel!(backend, 256)
        kernel!(arr, delp, delp, pblh, ustar, hflux, t2m,
                sf, w_scratch, 0, 0, Nz, FT(dt),
                FT(planet.kappa_vk), FT(planet.gravity),
                FT(planet.cp_dry), FT(planet.rho_ref),
                FT(diff.β_h), FT(diff.Kz_bg), FT(diff.Kz_min), FT(diff.Kz_max),
                FT(diff.fak), FT(diff.sffrac),
                Val(:mixing_ratio); ndrange=(Nx, Ny))
        synchronize(backend)
    end
    return nothing
end

# Also dispatch via diffuse_pbl! for seamless integration with run loop
function diffuse_pbl!(tracers::NamedTuple, delp,
                       pblh, ustar, hflux, t2m,
                       w_scratch,
                       diff::NonLocalPBLDiffusion, grid::LatitudeLongitudeGrid, dt,
                       planet::PlanetParameters)
    diffuse_nonlocal_pbl!(tracers, delp, pblh, ustar, hflux, t2m,
                           nothing, w_scratch,
                           diff, grid, dt, planet)
end
