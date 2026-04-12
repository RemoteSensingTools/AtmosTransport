# TM5 Alignment Gaps And Numerical Risks

## Current State

- `live behavior`: the current investigation status is that the daily ERA5 LL binary/preprocessor path is broadly clean, while the main blockers are runtime stepping and basis-contract issues. See the validated Session 1/2 summary in [`SESSION2_STATUS.md`](SESSION2_STATUS.md).
- `live behavior`: Session 2 status explicitly records clean binary/grid diagnostics and clean Dec 1-2 binaries at [`SESSION2_STATUS.md`](SESSION2_STATUS.md).
- `conclusion`: binary/preprocessor payloads are not the leading blocker now; remaining gaps are runtime mass stepping, basis consistency, and selected Float32 donor-mass issues.

## Pole Handling

### TM5 reference

- TM5 reduced-grid handling is an X-direction device, not a general “zero out pole dynamics” rule.
- In the newer Julia preprocessors, only `bm` is zeroed at pole faces:
  - [`preprocess_era5_daily.jl:725`](../scripts/preprocessing/preprocess_era5_daily.jl#L725)
  - [`preprocess_spectral_v4_binary.jl:1060`](../scripts/preprocessing/preprocess_spectral_v4_binary.jl#L1060)

### Current Julia LL runtime

- `live behavior`: the runtime still zeros `am` at pole rows in the LL path:
  - [`physics_phases.jl:341`](../src/Models/physics_phases.jl#L341)
  - [`physics_phases.jl:346`](../src/Models/physics_phases.jl#L346)
- `live behavior`: substep interpolation re-applies that pole zeroing:
  - [`physics_phases.jl:752`](../src/Models/physics_phases.jl#L752)
- `likely mismatch`: this is inconsistent with the stated TM5-style preprocessing convention that `j=1,Ny` are real cells and only `bm` at pole faces should be zero.

## Latitude And Reduced-Grid Handling

### TM5 reference

- `live behavior`: TM5 reduced-grid clustering is X-only and mass-based.
- On reduction, TM5 sums mass across clustered longitudes:
  - [`redgridZoom.F90:1123`](../deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/redgridZoom.F90#L1123)
- On expansion, TM5 redistributes tracer mass by air-mass fraction:
  - [`redgridZoom.F90:1609`](../deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/redgridZoom.F90#L1609)

### Current Julia handling

- GPU reduced-grid cap is now full-size, not the older cap-of-4 behavior:
  - [`run_helpers.jl:134`](../src/Models/run_helpers.jl#L134)
- GPU cluster summation uses compensated summation:
  - [`mass_flux_advection.jl:23`](../src/Advection/mass_flux_advection.jl#L23)
- CPU reduced-grid row reduction still uses plain Float32 accumulation:
  - [`reduced_grid.jl:234`](../src/Grids/reduced_grid.jl#L234)
- `likely mismatch`: CPU fallback/parity checks are therefore numerically weaker than the GPU path and weaker than ideal for high-latitude extensive sums.

## Layer Merging

### TM5 reference

- `live behavior`: TM5 target vertical structures are explicit interface selections (`echlevs`) followed by `FillLevels`-style level combination, not heuristic pressure-threshold grouping.
- The fused spectral v4 path in Julia can follow this explicit TM5 approach:
  - [`preprocess_spectral_v4_binary.jl:108`](../scripts/preprocessing/preprocess_spectral_v4_binary.jl#L108)
  - [`preprocess_spectral_v4_binary.jl:1320`](../scripts/preprocessing/preprocess_spectral_v4_binary.jl#L1320)

### Current daily LL path

- `live behavior`: the daily LL path still uses heuristic thickness-based merging:
  - [`preprocess_era5_daily.jl:55`](../scripts/preprocessing/preprocess_era5_daily.jl#L55)
- `likely mismatch`: `merge_thin_levels(min_thickness_Pa=1000)` is an intentional Julia heuristic, not strict TM5 `echlevs` parity.

## Moist vs Dry Mass Basis

### TM5 reference

- `live behavior`: TM5 forward mass is moist/total-air mass from pressure thickness:
  - [`grid_3d.F90:816`](../deps/tm5-cy3-4dvar/base/src/grid_3d.F90#L816)
  - [`meteo.F90:4846`](../deps/tm5-cy3-4dvar/base/src/meteo.F90#L4846)
- `live behavior`: generic TM5 sampling/output paths use `rm / m`, not `rm / (m*(1-q))`; see the companion memo [TM5_4DVAR_DRY_VMR_AUDIT.md](TM5_4DVAR_DRY_VMR_AUDIT.md).

### Current Julia LL path

- `live behavior`: the LL runtime computes dry mass for rm↔c conversions:
  - [`run_loop.jl:148`](../src/Models/run_loop.jl#L148)
  - [`physics_phases.jl:111`](../src/Models/physics_phases.jl#L111)
- `live behavior`: LL output converts `rm` to dry VMR using `m_dry`:
  - [`run_loop.jl:250`](../src/Models/run_loop.jl#L250)
  - [`physics_phases.jl:1736`](../src/Models/physics_phases.jl#L1736)
- `live behavior`: explicit LL dry-air kernels exist:
  - [`latlon_dry_air.jl:24`](../src/Advection/latlon_dry_air.jl#L24)

### Consequence

- `likely mismatch`: if LL transport uses moist preprocessed `m/am/bm/cm` but the model corrects IC/output on a dry basis, the path is mixed-basis rather than purely moist or purely dry.
- `conclusion`: exact dry transport would require at least:
  - `QV_start` and `QV_end` per window
  - face-consistent dry correction of `am/bm`
  - dry mass from endpoint-consistent `m*(1-q)`
  - dry `cm` recomputed from dry horizontal flux convergence

## Runtime Stepping Differences

### TM5 reference

- `live behavior`: newer TM5 forward transport does evolving-mass local loop refinement in X:
  - [`advectx__slopes.F90:441`](../deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90#L441)
- `live behavior`: it also does evolving-mass local loop refinement in Y:
  - [`advecty__slopes.F90:236`](../deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L236)

### Current Julia handling

- `live behavior`: current Julia LL X/Y/Z subcycling computes one pre-loop CFL and then applies uniform subdivision:
  - [`mass_flux_advection.jl:1137`](../src/Advection/mass_flux_advection.jl#L1137)
  - [`mass_flux_advection.jl:1161`](../src/Advection/mass_flux_advection.jl#L1161)
  - [`mass_flux_advection.jl:1190`](../src/Advection/mass_flux_advection.jl#L1190)
- `likely mismatch`: because donor mass evolves during the inner loop, a fixed pre-loop subdivision can under-estimate later effective CFLs.
- `conclusion`: this is a stronger candidate for the remaining LL instability than the validated binary payload itself.

## Float32 Risk Areas

- `live behavior`: CPU reduced-grid accumulation is plain Float32 summation in [`reduced_grid.jl:234`](../src/Grids/reduced_grid.jl#L234).
- `live behavior`: GPU reduced-grid clustering is already compensated in [`mass_flux_advection.jl:23`](../src/Advection/mass_flux_advection.jl#L23).
- `live behavior`: LL X/Y/Z subcycling uses Float32 CFL estimates and fixed divided fluxes in:
  - [`mass_flux_advection.jl:1141`](../src/Advection/mass_flux_advection.jl#L1141)
  - [`mass_flux_advection.jl:1165`](../src/Advection/mass_flux_advection.jl#L1165)
  - [`mass_flux_advection.jl:1196`](../src/Advection/mass_flux_advection.jl#L1196)
- `live behavior`: runtime `cm` reconstruction in `physics_phases.jl` uses Float64 accumulation on CPU before copying back, which is safer than raw Float32 accumulation:
  - [`physics_phases.jl:562`](../src/Models/physics_phases.jl#L562)
  - [`physics_phases.jl:576`](../src/Models/physics_phases.jl#L576)
- `recommendation`: use compensated summation for any remaining CPU extensive reductions, and avoid making CFL pass/fail decisions from large low-precision global sums when local donor-mass inequalities are available.

## Recommended Next Moves

### TM5-faithful path

- Keep LL transport on moist/total-air mass basis unless there is an explicit decision to depart from TM5.
- Prefer explicit `echlevs`-based layer selection over `merge_thin_levels` for TM5 parity.
- Remove or justify any remaining LL runtime pole-row `am` zeroing if the preprocessor already treats `j=1,Ny` as real cells.
- Bring LL stepping closer to TM5’s evolving-mass local loop refinement rather than relying on one pre-loop CFL estimate.

### Pragmatic Julia-only fixes

- If strict TM5 parity is not the goal, a safer Julia subcycling algorithm can still be worthwhile:
  - recompute safe directional chunks against the remaining flux and current donor mass
  - keep this clearly documented as a Julia stabilization choice, not TM5 behavior
- Upgrade CPU reduced-grid reductions to compensated accumulation for parity with the GPU path.

### Scope boundary

- Exact dry transport is a separate deliberate design, not current TM5 parity.
- Binary/preprocessor payloads should stay de-prioritized unless a new source-vs-binary mismatch is demonstrated at the failing hotspots.

