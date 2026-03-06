# ---------------------------------------------------------------------------
# Relaxed Arakawa-Schubert (RAS) convection — forward
#
# Implements the offline tracer transport portion of the RAS convection
# scheme as used in GEOS-Chem (convection_mod.F90, DO_CONVECTION).
#
# Physics:
#   GEOS met data provides two key fields:
#     CMFMC  — updraft convective mass flux at level interfaces [kg/m²/s]
#     DTRAIN — detraining mass flux at layer centers [kg/m²/s]
#
#   For each column (processed bottom-to-top then top-to-bottom):
#
#   1. Bottom-to-top pass: compute updraft concentration q_cloud(k)
#      At each level, environmental air is entrained and mixed into the
#      updraft. The entrainment is diagnosed from mass balance:
#        ENTRN(k) = max(0, CMFMC(k) + DTRAIN(k) - CMFMC(k+1))
#      where CMFMC(k+1) is the updraft flux entering from below.
#
#      The updraft concentration evolves via mass-weighted mixing:
#        q_cloud(k) = [CMFMC(k+1)·q_cloud(k+1) + ENTRN(k)·q_env(k)]
#                     / [CMFMC(k) + DTRAIN(k)]
#
#   2. Top-to-bottom pass: apply tendency to environment
#      Two physical processes modify the environment at each level:
#        - Compensating subsidence: environment air from level k-1 (above)
#          descends to replace the ascending updraft. This brings environment
#          concentration q_env(k-1) into level k:
#            CMFMC(k) × (q_env(k-1) - q_env(k))
#        - Detrainment: cloud air detrained into the environment:
#            DTRAIN(k) × (q_cloud(k) - q_env(k))
#
#      Layer tendency:
#        Δq(k) = (Δt / BMASS(k)) · [CMFMC(k)·(q_env(k-1) - q(k))
#                                   + DTRAIN(k)·(q_cloud(k) - q(k))]
#      where BMASS(k) = DELP(k) / g  [kg/m²].
#
#   For inert tracers (CO₂, SF₆, ²²²Rn), wet scavenging fraction = 0,
#   so all updraft air is conserved (no rainout removal).
#
# Air mass convention:
#   Uses total (moist) DELP for BMASS, matching GEOS-Chem. Both CMFMC
#   and DELP are total-pressure-based, so the scheme is self-consistent.
#
# Level ordering:
#   k=1 = TOA, k=Nz = surface (consistent with internal convention).
#   Bottom-to-top pass goes k=Nz → 1; top-to-bottom goes k=1 → Nz.
#
# KA kernel:
#   Single @kernel handles both grid types via Val{tracer_mode}:
#     :rm           — cubed-sphere: arr is tracer mass, uses m for conversion
#     :mixing_ratio — lat-lon: arr is mixing ratio directly
#
# References:
#   Moorthi & Suarez (1992), doi:10.1175/1520-0493(1992)120<0978:RASMAS>2.0.CO;2
#   GEOS-Chem: github.com/geoschem/geos-chem/blob/main/GeosCore/convection_mod.F90
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid, CubedSphereGrid
using ..Parameters: PlanetParameters
using KernelAbstractions: @kernel, @index, synchronize, get_backend


# =====================================================================
# KA kernel: RAS convective transport (CMFMC + DTRAIN)
#
# Two-pass algorithm per column:
#   Pass 1 (bottom → top): track updraft concentration q_cloud
#   Pass 2 (top → bottom): apply environmental tendency
#
# The workspace array `q_cloud_ws` stores the updraft concentration at
# each level, enabling the second pass. It must be pre-allocated at the
# same shape as the tracer array (avoids GPU stack overflow).
# =====================================================================

"""
KA kernel for RAS convective transport with explicit entrainment/detrainment.

For each (i,j) column:
- Pass 1 (k = Nz → 1): Compute updraft concentration `q_cloud` at each level
  by entraining environment air and mixing with the updraft from below.
- Pass 2 (k = 1 → Nz): Apply tendency from updraft flux divergence and
  detrainment to the environment mixing ratio.

Arguments:
- `arr`: tracer data (rm for CS, mixing ratio for LL), modified in-place
- `m`: air mass per cell [kg] (used only in :rm mode)
- `cmfmc`: convective mass flux at interfaces [kg/m²/s], size (..., Nz+1)
- `dtrain`: detraining mass flux at layer centers [kg/m²/s], size (..., Nz)
- `delp`: pressure thickness [Pa], size (..., Nz)
- `q_cloud_ws`: workspace for updraft concentration, same shape as arr
- `i_off, j_off`: index offsets for halo (Hp for CS, 0 for LL)
- `Nz`: number of vertical levels
- `dt`: timestep [s]
- `grav`: gravitational acceleration [m/s²]
"""
@kernel function _ras_column_kernel!(
    arr, @Const(m), @Const(cmfmc), @Const(dtrain), @Const(delp),
    q_cloud_ws,
    i_off, j_off, Nz, dt, grav,
    ::Val{tracer_mode}
) where tracer_mode
    i, j = @index(Global, NTuple)
    ii = i_off + i
    jj = j_off + j

    FT = eltype(arr)
    tiny = FT(1e-30)

    # ── Pass 1: bottom-to-top — compute updraft concentration ────────
    # q_cloud_below tracks the updraft concentration entering from the
    # level below. At the surface (k=Nz), there is no updraft from below,
    # so q_cloud starts as the surface environment concentration.
    q_cloud_below = FT(0)

    @inbounds for k in Nz:-1:1
        # Environment mixing ratio at level k
        if tracer_mode === :rm
            _m = m[ii, jj, k]
            q_k = _m > tiny ? arr[ii, jj, k] / _m : FT(0)
        else
            q_k = arr[ii, jj, k]
        end

        # Mass flux from below entering this level (interface k+1)
        # Convention: cmfmc[k+1] is the flux at the bottom of level k
        # At the surface (k=Nz), cmfmc[Nz+1] = 0 (no flux through surface)
        cmfmc_below = k < Nz ? cmfmc[ii, jj, k + 1] : FT(0)

        # Total outflow from cloud at level k:
        #   CMOUT = updraft leaving upward (CMFMC at top of level k)
        #         + air detrained into environment (DTRAIN)
        cmfmc_above = cmfmc[ii, jj, k]  # flux at top of level k (interface k)
        dtrain_k = dtrain[ii, jj, k]
        cmout = cmfmc_above + dtrain_k

        # Entrainment: environment air pulled into updraft to balance
        # the mass budget. ENTRN = outflow - inflow from below.
        entrn = max(FT(0), cmout - cmfmc_below)

        # Updraft concentration: mass-weighted mixture of
        #   - updraft air from below (cmfmc_below × q_cloud_below)
        #   - entrained environment air (entrn × q_env)
        if cmout > tiny
            qc = (cmfmc_below * q_cloud_below + entrn * q_k) / cmout
        else
            qc = q_k
        end

        q_cloud_ws[ii, jj, k] = qc
        q_cloud_below = qc
    end

    # ── Pass 2: top-to-bottom — apply tendency to environment ────────
    # Track the ORIGINAL environment concentration at the level above (k-1)
    # for the compensating subsidence term. Must use pre-update value to
    # match GEOS-Chem's simultaneous-update semantics.
    q_env_prev = FT(0)

    @inbounds for k in 1:Nz
        # Environment mixing ratio
        if tracer_mode === :rm
            _m = m[ii, jj, k]
            q_k = _m > tiny ? arr[ii, jj, k] / _m : FT(0)
        else
            q_k = arr[ii, jj, k]
        end

        # Layer air mass per unit area [kg/m²]
        bmass = delp[ii, jj, k] / grav

        # Tendency: two source terms
        tsum = FT(0)

        # 1. Compensating subsidence at level k
        #    The updraft removes air from the environment; to conserve mass,
        #    environment air from level k-1 (above) descends into level k.
        #    Net effect: CMFMC(k) × (q_env(k-1) - q_env(k))
        #    Uses ENVIRONMENT concentration, NOT cloud concentration.
        #    (Ref: GEOS-Chem convection_mod.F90, DO_CONVECTION)
        if k > 1
            tsum += cmfmc[ii, jj, k] * (q_env_prev - q_k)
        end

        # 2. Detrainment: cloud air at q_cloud(k) replaces environment air at q_k
        tsum += dtrain[ii, jj, k] * (q_cloud_ws[ii, jj, k] - q_k)

        # Save original q_k BEFORE update (for next level's subsidence term)
        q_env_prev = q_k

        # Apply tendency
        q_new = q_k + dt / bmass * tsum

        # Write back
        if tracer_mode === :rm
            arr[ii, jj, k] = q_new * m[ii, jj, k]
        else
            arr[ii, jj, k] = q_new
        end
    end
end


# Cached RAS CFL: computed once per invalidation, reused for all tracers/substeps.
# Call `invalidate_ras_cfl_cache!()` when CMFMC or DTRAIN data changes (once per window).
const _RAS_CFL_CACHE = Ref((valid=false, n_sub=1, dt_conv=Float32(0)))

"""Invalidate the RAS CFL cache. Call once per window when CMFMC/DTRAIN data changes."""
invalidate_ras_cfl_cache!() = (_RAS_CFL_CACHE[] = (valid=false, n_sub=1, dt_conv=Float32(0)); nothing)

"""
    _ras_subcycling(cmfmc_panels, dtrain_panels, delp_panels, dt, grav, Hp, Nc, Nz)

Compute RAS subcycling parameters, caching result until `invalidate_ras_cfl_cache!()`.
CPU-side scalar loop avoids GPU temporary allocations.
"""
function _ras_subcycling(cmfmc_panels::NTuple{6}, dtrain_panels::NTuple{6},
                          delp_panels::NTuple{6}, dt, grav, Hp::Int, Nc::Int, Nz::Int)
    FT = typeof(dt)
    cache = _RAS_CFL_CACHE[]
    if cache.valid
        return cache.n_sub, FT(cache.dt_conv)
    end

    # CPU-side CFL: copy once, scan with scalar loop (column-major: i innermost)
    max_cfl = zero(FT)
    for p in 1:6
        cmf_h = Array(cmfmc_panels[p])
        dtr_h = Array(dtrain_panels[p])
        dp_h  = Array(delp_panels[p])
        @inbounds for k in 2:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
            eff = abs(cmf_h[i, j, k]) + abs(dtr_h[i, j, k])
            cfl_b = eff / dp_h[i, j, k]
            cfl_a = eff / dp_h[i, j, k - 1]
            max_cfl = max(max_cfl, cfl_b, cfl_a)
        end
    end
    max_cfl *= dt * grav
    n_sub = max(1, ceil(Int, max_cfl / FT(0.9)))
    dt_conv = FT(dt) / FT(n_sub)
    _RAS_CFL_CACHE[] = (valid=true, n_sub=n_sub, dt_conv=Float32(dt_conv))
    @info "RAS CFL: max=$(round(max_cfl; digits=2)), n_sub=$n_sub, dt_conv=$(round(dt_conv; digits=1))s" maxlog=5
    return n_sub, dt_conv
end


# =====================================================================
# Dispatch: CubedSphereGrid — loop over 6 haloed panels
# =====================================================================

"""
$(SIGNATURES)

Apply RAS convective transport to cubed-sphere panel arrays.

Uses CMFMC (updraft mass flux at interfaces) and DTRAIN (detraining mass flux
at layer centers) to redistribute tracers vertically via the Relaxed
Arakawa-Schubert scheme.

Each panel's tracer mass (`rm`) is processed in two passes:
1. Bottom-to-top: compute updraft concentration by entraining environment air
2. Top-to-bottom: apply tendency from updraft flux divergence and detrainment

If `dtrain_panels` is `nothing`, falls back to Tiedtke-style CMFMC-only transport
with a one-time warning.

Adaptive subcycling keeps the convective CFL below 0.9 per substep.

# Arguments
- `rm_panels`: NTuple{6} of tracer mass arrays, modified in-place
- `m_panels`: NTuple{6} of air mass arrays [kg]
- `cmfmc_panels`: NTuple{6} of convective mass flux at interfaces [kg/m²/s]
- `delp_panels`: NTuple{6} of pressure thickness [Pa]
- `conv::RASConvection`: convection scheme selector
- `grid::CubedSphereGrid`: grid specification
- `dt`: timestep [s]
- `planet`: planet parameters (gravity)

# Keyword Arguments
- `dtrain_panels`: NTuple{6} of detraining mass flux [kg/m²/s], or `nothing`
- `workspace`: NTuple{6} of pre-allocated workspace arrays for q_cloud
"""
function convect!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                   cmfmc_panels::NTuple{6}, delp_panels::NTuple{6},
                   conv::RASConvection, grid::CubedSphereGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)

    # Fall back to Tiedtke if DTRAIN unavailable
    if dtrain_panels === nothing
        @warn "RAS: DTRAIN not available, falling back to Tiedtke-style transport" maxlog=1
        _tiedtke = TiedtkeConvection()
        convect!(rm_panels, m_panels, cmfmc_panels, delp_panels,
                 _tiedtke, grid, dt, planet)
        return nothing
    end

    FT = eltype(rm_panels[1])
    backend = get_backend(rm_panels[1])
    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz

    # Adaptive subcycling: RAS needs to account for BOTH cmfmc AND dtrain
    # Stability: (cmfmc[k] + dtrain[k]) × dt × g / Δp < 1 at every level
    # Cached per window — call invalidate_ras_cfl_cache!() when met data changes
    n_sub_conv, dt_conv = _ras_subcycling(cmfmc_panels, dtrain_panels, delp_panels,
                                           FT(dt), FT(planet.gravity), Hp, Nc, Nz)

    # Workspace for q_cloud — must be pre-allocated for GPU
    if workspace === nothing
        error("RAS convection requires pre-allocated workspace arrays. " *
              "Pass `workspace` keyword to convect!.")
    end

    kernel! = _ras_column_kernel!(backend, 256)
    for _ in 1:n_sub_conv
        for p in 1:6
            kernel!(rm_panels[p], m_panels[p],
                    cmfmc_panels[p], dtrain_panels[p], delp_panels[p],
                    workspace[p],
                    Hp, Hp, Nz, dt_conv, FT(planet.gravity),
                    Val(:rm); ndrange=(Nc, Nc))
        end
        synchronize(backend)
    end
    return nothing
end


# =====================================================================
# Dispatch: LatitudeLongitudeGrid — operate on mixing-ratio tracers
# =====================================================================

"""
$(SIGNATURES)

Apply RAS convective transport to lat-lon tracers (mixing ratios, GPU).

`cmfmc` is `(Nx, Ny, Nz+1)` updraft convective mass flux [kg/m²/s].
`delp` is `(Nx, Ny, Nz)` pressure thickness per layer [Pa].
`dtrain` keyword: `(Nx, Ny, Nz)` detraining mass flux [kg/m²/s], or `nothing`.

If `dtrain` is `nothing`, falls back to Tiedtke-style CMFMC-only transport.
Adaptive subcycling keeps the convective CFL below 0.9 per substep.
"""
function convect!(tracers::NamedTuple, cmfmc, delp,
                   conv::RASConvection, grid::LatitudeLongitudeGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)

    # Fall back to Tiedtke if DTRAIN unavailable
    if dtrain_panels === nothing
        @warn "RAS: DTRAIN not available, falling back to Tiedtke-style transport" maxlog=1
        _tiedtke = TiedtkeConvection()
        convect!(tracers, cmfmc, delp, _tiedtke, grid, dt, planet)
        return nothing
    end

    FT = floattype(grid)
    Nz = size(delp, 3)

    # Adaptive subcycling — include dtrain for RAS stability
    cmf_int = @view cmfmc[:, :, 2:Nz]
    dtr_int = @view dtrain_panels[:, :, 2:Nz]
    dp_below = @view delp[:, :, 2:Nz]
    dp_above = @view delp[:, :, 1:Nz-1]
    eff_flux = abs.(cmf_int) .+ abs.(dtr_int)
    max_cfl = max(FT(maximum(eff_flux ./ dp_below)),
                  FT(maximum(eff_flux ./ dp_above))) * FT(dt) * FT(planet.gravity)
    n_sub = max(1, ceil(Int, max_cfl / FT(0.9)))
    dt_conv = FT(dt) / FT(n_sub)

    # For LL, workspace can be lazily allocated (CPU typically)
    ws = if workspace !== nothing
        workspace
    else
        similar(delp)
    end

    for tracer in values(tracers)
        arr = conv_tracer_data(tracer)
        backend = get_backend(arr)
        Nx, Ny = size(arr, 1), size(arr, 2)
        kernel! = _ras_column_kernel!(backend, 256)
        for _ in 1:n_sub
            kernel!(arr, delp, cmfmc, dtrain_panels, delp,
                    ws,
                    0, 0, Nz, dt_conv, FT(planet.gravity),
                    Val(:mixing_ratio); ndrange=(Nx, Ny))
            synchronize(backend)
        end
    end
    return nothing
end
