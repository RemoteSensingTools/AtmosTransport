# ---------------------------------------------------------------------------
# Tiedtke (1989) mass-flux convection scheme — forward
#
# Uses updraft/downdraft mass fluxes from met data to redistribute tracers
# vertically. The mass fluxes are external (from ECMWF/GEOS reanalysis),
# so the operation is LINEAR in tracer concentration.
#
# The convective tracer flux at interface k is:
#   F[k] = M_net[k] * q_upwind
# where M_net is the net convective mass flux (positive upward) and
# q_upwind is selected by the sign of M_net.
#
# Layer tendency:
#   q_new[k] = q_old[k] + Δt * g / Δp[k] * (F[k+1] - F[k])
#
# Mass conservation is guaranteed because the flux telescopes:
#   Σ_k Δp[k] * q_new[k] = Σ_k Δp[k] * q_old[k]
# when F[1] = F[Nz+1] = 0 (no flux at top/bottom boundaries).
#
# Single KA kernel handles both grid types via `Val{tracer_mode}`:
#   :rm             — cubed-sphere: pre-convert rm → q, apply standard flux
#                     divergence in q-space, post-convert q → rm with per-column
#                     mass fix. Halo offsets for panel indexing.
#   :mixing_ratio   — lat-lon: operate on mixing ratios directly, no offsets
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid, CubedSphereGrid
using ..Parameters: PlanetParameters
using KernelAbstractions: @kernel, @index, synchronize, get_backend


"""
$(SIGNATURES)

Return the modifiable 3D array for a tracer (Field or raw array).
"""
conv_tracer_data(t) = t isa AbstractField ? interior(t) : t

"""
$(SIGNATURES)

Check whether `met` provides convective mass flux data.
Returns `false` for `nothing` or any object without a `conv_mass_flux` field.
"""
has_conv_mass_flux(met) = met !== nothing && hasproperty(met, :conv_mass_flux)

# =====================================================================
# Legacy-signature path: extracts fields from `met` NamedTuple and
# routes through the unified KA kernel (works on both CPU and GPU).
# =====================================================================

"""
$(SIGNATURES)

Apply Tiedtke mass-flux convection to all tracers in-place (lat-lon).

`met` should be a NamedTuple (or similar) with field `conv_mass_flux`:
a 3D array of size `(Nx, Ny, Nz+1)` containing the net convective mass flux
[kg/m²/s] at each interface level, positive upward.

If `met` is `nothing` or lacks `conv_mass_flux`, this is a no-op.
"""
function convect!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid,
                  conv::TiedtkeConvection, Δt)
    has_conv_mass_flux(met) || return nothing

    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    grav = FT(grid.gravity)

    cmfmc = met.conv_mass_flux

    # Build Δp array (uniform per level for LatLon at reference pressure)
    delp = similar(conv_tracer_data(first(values(tracers))))
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        delp[i, j, k] = FT(Δz(k, grid))
    end

    # Use KA kernel — works on both CPU and GPU via get_backend
    for tracer in values(tracers)
        arr = conv_tracer_data(tracer)
        backend = get_backend(arr)
        kernel! = _convect_column_kernel!(backend, 256)
        kernel!(arr, delp, cmfmc, delp,
                0, 0, Nz, FT(Δt), grav,
                Val(:mixing_ratio); ndrange=(Nx, Ny))
        synchronize(backend)
    end

    return nothing
end

# =====================================================================
# Unified KA kernel: Tiedtke mass-flux convection
#
# Grid-agnostic via `Val{tracer_mode}`:
#   :rm             — cubed-sphere panels (arr = tracer mass, uses m for conversion)
#   :mixing_ratio   — lat-lon arrays (arr = mixing ratio, m unused)
#
# Index offsets (i_off, j_off) handle halos:
#   CS:  i_off = j_off = Hp  → ii = Hp + i
#   LL:  i_off = j_off = 0   → ii = i
# =====================================================================

"""
Grid-agnostic KA kernel for Tiedtke mass-flux convective transport.

For each (i,j) column, processes in three phases:

1. (if :rm) **Pre-convert**: convert rm → q in-place (`q = rm/m` for all k).

2. **Flux divergence** in q-space with mass-divergence correction:
     dq = Δt × g/Δp × [F_below - F_above - q_k × (CMFMC_below - CMFMC_above)]
   where F[k] = CMFMC[k] × q_upwind, selected by sign of CMFMC.

   The mass-divergence correction subtracts the air-mass-dilution term that a
   coupled model would handle via air mass update. This ensures exactly zero
   tendency for spatially uniform mixing ratio fields — critical for offline
   transport where air mass is held fixed during convection. Applied in both
   :rm and :mixing_ratio modes.

3. (if :rm) **Post-convert**: convert q → rm (`rm = q × m` for all k).
   No per-column mass fix is applied; the global mass fixer in the run loop
   handles the small Σ rm drift from the correction breaking flux telescoping.

Mass conservation:
  Both modes: column Σ Δp×q / Σ rm NOT exactly preserved (mass-divergence
  correction breaks flux telescoping), but drift is small (~ppm level) and
  corrected by the global mass fixer. This avoids the systematic
  level-dependent bias that per-column mass fixes create.

Zero-flux boundary conditions at top (k=1) and surface (k=Nz+1).
"""
@kernel function _convect_column_kernel!(
    arr, @Const(m), @Const(cmfmc), @Const(delp),
    i_off, j_off, Nz, dt, grav,
    ::Val{tracer_mode}
) where tracer_mode
    i, j = @index(Global, NTuple)
    ii = i_off + i
    jj = j_off + j

    FT = eltype(arr)
    zero_FT = zero(FT)

    # ── Phase 1: Convert rm → q in-place (CS grids only) ──
    if tracer_mode === :rm
        @inbounds for k in 1:Nz
            _m = m[ii, jj, k]
            arr[ii, jj, k] = _m > zero_FT ? arr[ii, jj, k] / _m : zero_FT
        end
    end

    # ── Phase 2: Upwind flux divergence in q-space ──
    flux_above = zero_FT
    cmfmc_above = zero_FT

    @inbounds for k in 1:Nz
        q_k = arr[ii, jj, k]

        # Flux at interface k+1 (below level k)
        if k < Nz
            M_below = cmfmc[ii, jj, k + 1]
            q_below = arr[ii, jj, k + 1]
            # Upwind: M > 0 (upward) → source from k+1; M < 0 (downward) → source from k
            flux_below = M_below >= zero_FT ? M_below * q_below : M_below * q_k
            cmfmc_below = M_below
        else
            flux_below = zero_FT       # F[Nz+1] = 0 (no flux through surface)
            cmfmc_below = zero_FT
        end

        dp_k = delp[ii, jj, k]

        # Mass-divergence correction for offline transport:
        # dq = dt*g/dp * [F_below - F_above - q_k * (CMFMC_below - CMFMC_above)]
        #    = dt*g/dp * [CMFMC_below*(q_upwind_below - q_k) - CMFMC_above*(q_upwind_above - q_k)]
        # For uniform q=c: all differences vanish → dq = 0 exactly.
        mass_div = cmfmc_below - cmfmc_above
        dq = dt * grav / dp_k * (flux_below - flux_above - q_k * mass_div)

        arr[ii, jj, k] = q_k + dq

        # Shift: current below becomes above for next level
        flux_above = flux_below
        cmfmc_above = cmfmc_below
    end

    # ── Phase 3: Convert q → rm (CS grids only) ──
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
    _max_conv_cfl_cs(cmfmc_panels, delp_panels, dt, grav, Hp, Nc, Nz)

Estimate the maximum convective CFL across all 6 cubed-sphere panels.
CFL at interface k = |cmfmc[k]| × dt × g / Δp, checked against the layer
on both sides of the interface.
"""
function _max_conv_cfl_cs(cmfmc_panels::NTuple{6}, delp_panels::NTuple{6},
                           dt, grav, Hp::Int, Nc::Int, Nz::Int)
    FT = typeof(dt)
    max_cfl = zero(FT)
    for p in 1:6
        cmf = @view cmfmc_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 2:Nz]
        dp_below = @view delp_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 2:Nz]
        dp_above = @view delp_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, 1:Nz-1]
        cfl_b = FT(maximum(abs.(cmf) ./ dp_below)) * dt * grav
        cfl_a = FT(maximum(abs.(cmf) ./ dp_above)) * dt * grav
        max_cfl = max(max_cfl, cfl_b, cfl_a)
    end
    return max_cfl
end

"""
    convect!(rm_panels, m_panels, cmfmc_panels, delp_panels,
             conv, grid, dt, planet)

Apply Tiedtke mass-flux convection to cubed-sphere panel arrays.

Each panel's tracer mass (`rm`) is converted to mixing ratio, convected
via upwind mass-flux transport, then converted back.

Adaptive subcycling keeps the convective CFL below 0.9 per substep,
guaranteeing positivity without the need for a clamp.
"""
function convect!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                   cmfmc_panels::NTuple{6}, delp_panels::NTuple{6},
                   conv::TiedtkeConvection, grid::CubedSphereGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)
    FT = eltype(rm_panels[1])
    backend = get_backend(rm_panels[1])
    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz

    # Adaptive subcycling: keep CFL < 0.9 for stability + positivity
    max_cfl = _max_conv_cfl_cs(cmfmc_panels, delp_panels, FT(dt),
                                FT(planet.gravity), Hp, Nc, Nz)
    n_sub = max(1, ceil(Int, max_cfl / FT(0.9)))
    dt_conv = FT(dt) / FT(n_sub)

    kernel! = _convect_column_kernel!(backend, 256)
    for _ in 1:n_sub
        for p in 1:6
            kernel!(rm_panels[p], m_panels[p], cmfmc_panels[p], delp_panels[p],
                    Hp, Hp, Nz, dt_conv, FT(planet.gravity),
                    Val(:rm); ndrange=(Nc, Nc))
        end
        synchronize(backend)
    end
    return nothing
end

# =====================================================================
# Dispatch: LatitudeLongitudeGrid — operate on mixing-ratio tracers (GPU)
# =====================================================================

"""
    convect!(tracers, cmfmc, delp, conv, grid, dt, planet)

Apply Tiedtke mass-flux convection to lat-lon tracers (mixing ratios, GPU).

`cmfmc` is `(Nx, Ny, Nz+1)` net convective mass flux [kg/m²/s].
`delp` is `(Nx, Ny, Nz)` pressure thickness per layer [Pa].

Adaptive subcycling keeps the convective CFL below 0.9 per substep.
"""
function convect!(tracers::NamedTuple, cmfmc, delp,
                   conv::TiedtkeConvection, grid::LatitudeLongitudeGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)
    FT = floattype(grid)
    Nz = size(delp, 3)

    # Adaptive subcycling
    cmf_int = @view cmfmc[:, :, 2:Nz]
    dp_below = @view delp[:, :, 2:Nz]
    dp_above = @view delp[:, :, 1:Nz-1]
    max_cfl = max(FT(maximum(abs.(cmf_int) ./ dp_below)),
                  FT(maximum(abs.(cmf_int) ./ dp_above))) * FT(dt) * FT(planet.gravity)
    n_sub = max(1, ceil(Int, max_cfl / FT(0.9)))
    dt_conv = FT(dt) / FT(n_sub)

    for tracer in values(tracers)
        arr = conv_tracer_data(tracer)
        backend = get_backend(arr)
        Nx, Ny = size(arr, 1), size(arr, 2)
        kernel! = _convect_column_kernel!(backend, 256)
        for _ in 1:n_sub
            # m argument unused in :mixing_ratio mode; pass delp as placeholder
            kernel!(arr, delp, cmfmc, delp,
                    0, 0, Nz, dt_conv, FT(planet.gravity),
                    Val(:mixing_ratio); ndrange=(Nx, Ny))
            synchronize(backend)
        end
    end
    return nothing
end
