# ---------------------------------------------------------------------------
# Met-data-driven PBL diffusion — variable Kz from PBLH, USTAR, HFLUX, T2M
#
# Follows the revised LTG scheme (Beljaars & Viterbo 1998) as implemented
# in TM5 (diffusion.F90, bldiff routine).
#
# Unlike BoundaryLayerDiffusion (static exponential Kz), this computes Kz
# per column from prescribed surface fields, capturing diurnal and spatial
# variability in boundary-layer turbulence.
#
# TM5 reference formulas (diffusion.F90 lines 1197-1230):
#   Obukhov length: L = -θ_v * u*³ / (κ g H_kin)
#   Unstable BL:
#     Surface layer (z < 0.1h): Kz = u* κ z (1-z/h)² (1 - β_h z/L)^(1/3)
#     Mixed layer (z ≥ 0.1h):   Kz = w_m κ z (1-z/h)²
#       where w_m = u*(1 - 0.1 β_h h/L)^(1/3)
#   Stable BL:
#     Kz = u* κ z (1-z/h)² / (1 + 5 z/L)
#   Above PBL:
#     Kz = Kz_bg
#
# Single KA kernel handles both grid types via `Val{tracer_mode}`:
#   :rm             — cubed-sphere: convert rm ↔ mixing ratio, with halo offsets
#   :mixing_ratio   — lat-lon: operate on mixing ratios directly, no offsets
#
# Height computation uses hydrostatic approximation with pressure-dependent
# density: dz = Δp × R_d × T_sfc / (p_mid × g), where R_d = cp/3.5.
# T_sfc (isothermal) is adequate within the PBL (~lowest 3 km).
#
# Above the PBL, Kz transitions smoothly from the BL value to Kz_bg over
# an entrainment zone from h_pbl to 1.2·h_pbl (linear taper).
#
# PBLH definitions vary by met source:
#   GEOS:  Kh-threshold — 10% of column-max eddy diffusivity
#          (Molod et al. 2015; McGrath-Spangler & Molod, JAMES 2022)
#   ERA5:  Bulk Richardson number Ri > 0.25 (Seidel et al. 2012)
# These give systematically different PBL heights; the physics here is
# agnostic to the PBLH definition — it just uses whatever the met driver provides.
#
# Physical constants (kappa_vk, grav, cp_dry, rho_ref) are passed as kernel
# arguments from PlanetParameters — nothing is hardcoded here.
# ---------------------------------------------------------------------------

using ..Grids: CubedSphereGrid, LatitudeLongitudeGrid, floattype
using ..Parameters: PlanetParameters
using KernelAbstractions: @kernel, @index, synchronize, get_backend

# =====================================================================
# Kz profile function (pure, GPU-safe)
# =====================================================================

"""
Compute Kz [m²/s] at a given height using PBL-similarity (revised LTG).
Pure function, no side effects — suitable for calling inside GPU kernels.

Physics constants are passed explicitly (no module-level globals).

Above the PBL, Kz is linearly tapered from h_pbl to 1.2·h_pbl
(smoothed entrainment zone), avoiding the unrealistic sharp cutoff
that can produce artificial tracer gradients at the inversion.
"""
@inline function _pbl_kz(z, h_pbl, ustar, L_ob, kappa_vk,
                          β_h, Kz_bg, Kz_min, Kz_max, ::Type{FT}) where FT
    κ = FT(kappa_vk)

    # Above entrainment zone: background only
    h_taper = FT(1.2) * h_pbl
    if z >= h_taper
        return FT(Kz_bg)
    end

    # Compute PBL Kz as if still within BL (clamped to h_pbl for formula)
    z_eff = min(z, h_pbl - FT(1))  # avoid z/h = 1 singularity
    zh = z_eff / h_pbl
    zzh2 = (FT(1) - zh)^2  # (1 - z/h)² shape function

    if L_ob < FT(0)
        # Unstable BL
        if z_eff < FT(0.1) * h_pbl
            # Surface layer (TM5 sffrac = 0.1)
            Kz = ustar * κ * z_eff * zzh2 * cbrt(FT(1) - β_h * z_eff / L_ob)
        else
            # Mixed layer
            w_m = ustar * cbrt(FT(1) - FT(0.1) * β_h * h_pbl / L_ob)
            Kz = w_m * κ * z_eff * zzh2
        end
    else
        # Stable BL
        Kz = ustar * κ * z_eff * zzh2 / (FT(1) + FT(5) * z_eff / L_ob)
    end

    Kz = clamp(Kz, FT(Kz_min), FT(Kz_max))

    # Taper zone (h_pbl to 1.2·h_pbl): linear blend to Kz_bg
    if z >= h_pbl
        frac = (h_taper - z) / (FT(0.2) * h_pbl)
        Kz = FT(Kz_bg) + frac * (Kz - FT(Kz_bg))
    end

    return Kz
end

# =====================================================================
# Unified KA kernel: PBL diffusion with inline Kz + Thomas solve
#
# Grid-agnostic via `Val{tracer_mode}`:
#   :rm             — cubed-sphere panels (arr = tracer mass, needs m for conversion)
#   :mixing_ratio   — lat-lon arrays (arr = mixing ratio, m unused)
#
# Index offsets (i_off, j_off) handle halos:
#   CS:  i_off = j_off = Hp  → ii = Hp + i
#   LL:  i_off = j_off = 0   → ii = i
# =====================================================================

"""
Grid-agnostic KA kernel for met-driven PBL diffusion.

For each (i,j) column:
1. (if :rm) Convert tracer mass → mixing ratio
2. Compute interface heights from DELP (hydrostatic approximation)
3. Compute Obukhov length from surface fields
4. Thomas solve: build tridiagonal + forward eliminate + back-substitute
5. (if :rm) Convert mixing ratio back to tracer mass

`w_scratch` stores Thomas w-factors (avoids per-thread allocation on GPU).
"""
@kernel function _pbl_diffuse_kernel!(
    arr, @Const(m), @Const(delp), @Const(pblh), @Const(ustar),
    @Const(hflux), @Const(t2m), w_scratch,
    i_off, j_off, Nz, dt,
    kappa_vk, grav, cp_dry, rho_ref,
    β_h, Kz_bg, Kz_min, Kz_max,
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
    # R_d = cp / 3.5 for ideal diatomic gas (cp = 7/2 R_d)
    R_d = cp / FT(3.5)

    # --- Surface fields ---
    h_pbl = max(FT(pblh[ii, jj]), FT(100))
    us    = max(FT(ustar[ii, jj]), FT(0.01))
    H_sfc = FT(hflux[ii, jj])
    T_sfc = max(FT(t2m[ii, jj]), FT(200))  # clamp cold extremes

    # --- Obukhov length ---
    H_kin = H_sfc / (ρ * cp)
    H_safe = H_kin + sign(H_kin + FT(1e-20)) * FT(1e-10)
    L_ob = -T_sfc * us^3 / (κ_vk * g * H_safe)

    # --- Compute surface pressure and layer heights using hydrostatic ---
    # dz_k = Δp_k * R_d * T_sfc / (p_mid_k * g)
    # where p_mid_k is evaluated at the layer center.
    # We use T_sfc throughout the PBL (isothermal approximation, adequate
    # for the lowest ~3 km where diffusion matters).
    ps = FT(0)
    @inbounds for k in 1:Nz
        ps += delp[ii, jj, k]
    end
    R_T_over_g = R_d * T_sfc / g

    z_col = FT(0)
    p_top = FT(0)  # pressure at top of atmosphere
    @inbounds for k in 1:Nz
        delp_k = delp[ii, jj, k]
        p_top_k = p_top
        p_bot_k = p_top + delp_k
        p_mid_k = (p_top_k + p_bot_k) / FT(2)
        # Avoid division by zero at TOA (p_mid ~ 0)
        p_mid_k = max(p_mid_k, FT(1))
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

    # --- Forward sweep: build tridiagonal + Thomas elimination ---
    # Process top-to-bottom (k=1 = TOA, k=Nz = surface).
    # z_above tracks the height of the top of current layer.
    # Heights use pressure-dependent density: dz = Δp * R_d * T / (p_mid * g)
    z_above = z_col
    w_prev = FT(0)
    g_prev = FT(0)
    p_top = FT(0)

    @inbounds for k in 1:Nz
        delp_k = delp[ii, jj, k]
        p_bot_k = p_top + delp_k
        p_mid_k = max((p_top + p_bot_k) / FT(2), FT(1))
        dz_k = delp_k * R_T_over_g / p_mid_k
        z_below = z_above - dz_k

        # Kz at interface above (between k-1 and k)
        D_above = FT(0)
        if k > 1
            Kz_a = _pbl_kz(z_above, h_pbl, us, L_ob, κ_vk,
                            β_h, Kz_bg, Kz_min, Kz_max, FT)
            delp_prev = delp[ii, jj, k - 1]
            p_top_prev = p_top - delp_prev
            p_mid_prev = max((p_top_prev + p_top) / FT(2), FT(1))
            dz_prev = delp_prev * R_T_over_g / p_mid_prev
            dz_mid = (dz_prev + dz_k) / FT(2)
            D_above = Kz_a / (dz_k * dz_mid)
        end

        # Kz at interface below (between k and k+1)
        D_below = FT(0)
        if k < Nz
            Kz_b = _pbl_kz(z_below, h_pbl, us, L_ob, κ_vk,
                            β_h, Kz_bg, Kz_min, Kz_max, FT)
            delp_next = delp[ii, jj, k + 1]
            p_mid_next = max(p_bot_k + delp_next / FT(2), FT(1))
            dz_next = delp_next * R_T_over_g / p_mid_next
            dz_mid = (dz_k + dz_next) / FT(2)
            D_below = Kz_b / (dz_k * dz_mid)
        end

        p_top = p_bot_k

        # Tridiagonal coefficients: (I - dt*D) c_new = c_old
        a_k = k > 1  ? -dt * D_above : FT(0)
        b_k = FT(1) + dt * (D_above + D_below)
        c_k = k < Nz ? -dt * D_below : FT(0)

        # Thomas forward elimination
        c_val = arr[ii, jj, k]
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
    diffuse_pbl!(rm_panels, m_panels, delp_panels,
                 pblh_panels, ustar_panels, hflux_panels, t2m_panels,
                 w_scratch_panels, diff, grid, dt, planet)

Apply met-driven PBL diffusion to cubed-sphere panel arrays.

Each panel's tracer mass (`rm`) is converted to mixing ratio, diffused
with column-varying Kz, then converted back.
"""
function diffuse_pbl!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                       delp_panels::NTuple{6},
                       pblh_panels::NTuple{6}, ustar_panels::NTuple{6},
                       hflux_panels::NTuple{6}, t2m_panels::NTuple{6},
                       w_scratch_panels::NTuple{6},
                       diff::PBLDiffusion, grid::CubedSphereGrid, dt,
                       planet::PlanetParameters)
    FT = eltype(rm_panels[1])
    backend = get_backend(rm_panels[1])
    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz
    kernel! = _pbl_diffuse_kernel!(backend, 256)
    for p in 1:6
        kernel!(rm_panels[p], m_panels[p], delp_panels[p],
                pblh_panels[p], ustar_panels[p], hflux_panels[p], t2m_panels[p],
                w_scratch_panels[p],
                Hp, Hp, Nz, FT(dt),
                FT(planet.kappa_vk), FT(planet.gravity),
                FT(planet.cp_dry), FT(planet.rho_ref),
                FT(diff.β_h), FT(diff.Kz_bg), FT(diff.Kz_min), FT(diff.Kz_max),
                Val(:rm); ndrange=(Nc, Nc))
    end
    synchronize(backend)
    return nothing
end

# =====================================================================
# Dispatch: LatitudeLongitudeGrid — operate on mixing-ratio tracers
# =====================================================================

"""
    diffuse_pbl!(tracers, delp, pblh, ustar, hflux, t2m,
                 w_scratch, diff, grid, dt, planet)

Apply met-driven PBL diffusion to lat-lon tracers (mixing ratios).

Surface fields (pblh, ustar, hflux, t2m) and pressure thickness (delp)
come from the met driver. The `w_scratch` array must have the same shape
as the tracer arrays.
"""
function diffuse_pbl!(tracers::NamedTuple, delp,
                       pblh, ustar, hflux, t2m,
                       w_scratch,
                       diff::PBLDiffusion, grid::LatitudeLongitudeGrid, dt,
                       planet::PlanetParameters)
    FT = floattype(grid)
    for tracer in values(tracers)
        arr = tracer_data(tracer)
        backend = get_backend(arr)
        Nx, Ny = size(arr, 1), size(arr, 2)
        Nz = size(arr, 3)
        kernel! = _pbl_diffuse_kernel!(backend, 256)
        # m argument unused in :mixing_ratio mode; pass delp as placeholder
        kernel!(arr, delp, delp, pblh, ustar, hflux, t2m,
                w_scratch, 0, 0, Nz, FT(dt),
                FT(planet.kappa_vk), FT(planet.gravity),
                FT(planet.cp_dry), FT(planet.rho_ref),
                FT(diff.β_h), FT(diff.Kz_bg), FT(diff.Kz_min), FT(diff.Kz_max),
                Val(:mixing_ratio); ndrange=(Nx, Ny))
        synchronize(backend)
    end
    return nothing
end
