# Advection with Direct Reanalysis Mass Fluxes: GCHP vs AtmosTransportModel vs TM5

## Scope

This note tracks how three systems handle tracer advection when reanalysis (or model-diagnostic) mass fluxes are used directly:

- GCHP (`/home/cfranken/code/gitHub/GCHP`)
- AtmosTransportModel (this repo)
- TM5 (`deps/tm5`)

Focus areas:

1. Horizontal advection
2. Vertical advection
3. Dry vs wet pressure/mass handling
4. Mixing-ratio basis handling
5. What is needed in AtmosTransportModel to support multiple approaches systematically
6. A unified scheme to compare ERA5 transport and GEOS transport with the exact same tracer-propagation algorithms

## Executive Summary

- AtmosTransportModel and TM5 both run a flux-form `rm,m` transport core; GCHP offline transport uses horizontal flux update followed by vertical pressure remap.
- In the inspected production paths, AtmosTransportModel CS and TM5 diagnose vertical transport from horizontal mass-flux convergence; GCHP does vertical remap in pressure coordinates.
- GCHP has the most explicit dry/total basis controls and explicit tracer basis conversion dry <-> total around advection.
- AtmosTransportModel already has the right abstractions for a unified framework (`AbstractRawMetDriver` vs `AbstractMassFluxMetDriver`) but policy choices are still spread across metadata flags and path-specific code.
- AtmosTransportModel can support strict ERA5-vs-GEOS algorithmic comparison by introducing a single transport-policy layer and canonical flux-state contract, then driving both met sources through that same transport kernel stack.

## Evidence Map (Code Pointers)

### AtmosTransportModel

- GEOS CS direct flux read and conversion:
  - `src/IO/geosfp_cubed_sphere_reader.jl:17-31, 162-176, 192`
  - `src/IO/geosfp_cs_met_driver.jl:162-163, 1256-1258`
- Imported flux scaling to transported mass per half-step:
  - `src/Models/physics_phases.jl:227-233`
- Air-mass basis selection (dry when enabled and QV loaded):
  - `src/Models/physics_phases.jl:252-268`
  - `src/Advection/cubed_sphere_mass_flux.jl:129-151`
- Vertical transport from continuity in CS path:
  - `src/Models/physics_phases.jl:302-313`
  - `src/Advection/cubed_sphere_mass_flux.jl:321-361`
- CS flux-form updates and split order:
  - `src/Advection/cubed_sphere_mass_flux.jl:653-654, 725-726, 810-811, 1160-1193`
- Remap vertical path exists (optional):
  - `src/Models/run_helpers.jl:127-129`
  - `src/Models/physics_phases.jl:421-510`
- Dry-correction kernels exist for CS and LL:
  - `src/Advection/cubed_sphere_mass_flux.jl:153-280`
  - `src/Advection/latlon_dry_air.jl:1-136`

### GCHP

- Direct external mass-flux import controls:
  - `src/GCHP_GridComp/GCHPctmEnv_GridComp/GCHPctmEnv_GridCompMod.F90:44, 125-137, 129, 187-200`
- Dry/total pressure and humidity correction controls:
  - `.../GCHPctmEnv_GridCompMod.F90:560-581`
- Pressure-edge exports (`PLE`, `DryPLE`) and dry pressure construction:
  - `.../GCHPctmEnv_GridCompMod.F90:702-836, 1148-1299`
- Direct `MFXC/MFYC` path and optional humidity correction:
  - `.../GCHPctmEnv_GridCompMod.F90:999-1032`
- AdvCore basis conversions and pressure-mode advection calls:
  - `src/GCHP_GridComp/FVdycoreCubed_GridComp/AdvCore_GridCompMod.F90:769-784, 1065-1071, 1192-1232, 1294-1301`
- Offline FV3 transport operator:
  - Horizontal update: `fv_tracer2d.F90:517` (and `269` in `tracer_2d_1L`)
  - Vertical remap: `fv_tracer2d.F90:993-1028`
  - Post-remap scaling: `fv_tracer2d.F90:1048-1064`

### TM5

- Direct flux read path (`tmpp`, `tm5-nc`):
  - `deps/tm5/base/src/tmm.F90:2046-2056` (`mfu/mfv`)
  - `deps/tm5/base/src/tmm.F90:2366-2371` (`mfw`)
- Flux setup and balancing:
  - `deps/tm5/base/src/meteo.F90:1513-1521, 1608-1611, 1707-1710`
- Transport setup uses time-interpolated `pu/pv`, then computes `cm` in `dynam0`:
  - `deps/tm5/base/src/advectm_cfl.F90:347-363`
  - `deps/tm5/base/src/advect_tools.F90:719-752, 767-789`
- Flux-form tracer updates (`rm,m`) in x/y/z:
  - `deps/tm5/base/src/advectx.F90:630, 662-671, 706-710`
  - `deps/tm5/base/src/advecty.F90:481-491, 503-512, 619-624`
  - `deps/tm5/base/src/advectz.F90:441, 462-470, 480-486, 496`
- Pressure-to-mass basis is pressure-thickness based (no `(1-q)` factor in transport mass):
  - `deps/tm5/base/src/meteo.F90:4845-4852`
  - `deps/tm5/base/src/meteo.F90:4947-4957`
- Humidity is separate (`Q`, kg/kg):
  - `deps/tm5/base/src/tmm.F90:2720-2723`
  - `deps/tm5/base/src/meteo.F90:5086-5091`

## Side-by-Side: Key Similarities and Differences

| Dimension | AtmosTransportModel | GCHP | TM5 | Similarity / Difference |
|---|---|---|---|---|
| Direct horizontal mass-flux ingestion | Yes for GEOS CS (`MFXC/MFYC`) | Yes (`MFXC/MFYC`) | Yes (`mfu/mfv`) | All can ingest horizontal mass flux directly |
| Horizontal transport update form | Flux-form update of `rm` and `m` | Flux-form update of tracer with pressure-thickness update in FV3 transport stage | Flux-form update of `rm` and `m` | All use mass-flux divergence for horizontal transport |
| Primary vertical transport operator (offline path inspected) | `cm` diagnosed from horizontal convergence, then z advection (or optional remap path) | Pressure-coordinate remap after horizontal transport | `cm` diagnosed from horizontal convergence (`dynam0`), then z advection | GCHP differs: remap-based vertical transport; Atmos/TM5 similar in continuity-based `cm` path |
| Use of imported vertical mass flux in traced advection path | No (CS path computes `cm`) | No imported vertical flux in `offline_tracer_advection`; vertical flux export is diagnostic | `mfw` can be read, but transport `cm` comes from `dynam0` in advection setup path | In these paths, vertical transport is not directly driven by imported `w` flux field |
| Dry vs wet pressure basis control | Partial: dry air mass from `DELP*(1-qv)` when enabled | Explicit total vs dry pressure modes (`PLE` vs `DryPLE`) | Pressure/mass transport basis from `dp/g` (moist-air mass by default) | GCHP most explicit and end-to-end configurable |
| Dry correction of horizontal fluxes | Kernels exist for CS + LL | Optional humidity correction of imported `MFX/MFY` | Not explicit in core transport path | Atmos and GCHP have explicit mechanisms; TM5 path here does not apply dry conversion in transport core |
| Tracer basis around advection | Tracer mass `rm`; mixing ratio recovered as `rm/m` | Converts dry tracers to total basis in total-air mode; converts back after advection | Tracer mass `rm`; mixing ratio recovered as `rm/m` | Atmos/TM5 directly mass-based; GCHP explicitly bridges dry/total mixing-ratio conventions |

## Horizontal vs Vertical Treatment (Independent View)

### Horizontal

- AtmosTransportModel:
  - Imports GEOS CS horizontal mass fluxes, converts units, scales by half-step, then applies flux-form transport kernels.
- GCHP:
  - Uses imported or wind-derived `MFX/MFY`, runs FV3 horizontal tracer transport with time splitting and CFL control.
- TM5:
  - Reads or computes `mfu/mfv`, balances to `pu/pv`, then advects horizontally in flux form.

### Vertical

- AtmosTransportModel:
  - CS continuity closure computes `cm` from `am/bm` and `bt`; z-advection in Strang sequence.
  - Alternate optional path: horizontal-only + vertical remap.
- GCHP:
  - Vertical advection is a remap from post-horizontal pressure distribution (`pe1`) to target pressure distribution (`pe2`).
- TM5:
  - `dynam0` diagnoses `cm` from horizontal convergence; `advectz` applies flux-form vertical transport.

## Dry/Wet Pressures and Mixing Ratios

### AtmosTransportModel

- Dry basis can be used for air mass when QV exists (`dry_correction` metadata in run loop).
- Dry corrections for fluxes and convection fields are implemented in kernels for CS and LL, but not yet centralized as a single policy switch in the run loop.

### GCHP

- Explicit dry vs total pressure mode in advection interface.
- Explicit tracer-basis conversion dry <-> total around transport in total-air mode.
- Optional `MFX/MFY` humidity correction (`/(1-SPHU0)`) when importing fluxes.

### TM5

- Transport mass is pressure-derived (`dp/g * area`); humidity is tracked as separate `Q`.
- No explicit transport-core dry conversion using `(1-q)` in the traced path.

## How AtmosTransportModel Can Enable Multiple Approaches Systematically

### Current strengths already in code

- Driver abstraction already separates:
  - raw-wind drivers (`AbstractRawMetDriver`)
  - direct mass-flux drivers (`AbstractMassFluxMetDriver`)
- Both mass-flux kernels and vertical-remap path exist.
- Dry correction kernels exist for both CS and LL.

### Main gap

The transport choice space is currently spread across per-path flags and metadata (`dry_correction`, `mass_fixer`, `remap_pressure_fix`, `use_vertical_remap`, etc.), not one central policy object.

### Proposed policy layer

Introduce one explicit transport policy object consumed by the run loop:

```julia
@kwdef struct TransportMassFluxPolicy
    horizontal_operator::Symbol = :flux_form_slopes      # :flux_form_slopes | :flux_form_ppm
    vertical_operator::Symbol   = :continuity_cm         # :continuity_cm | :pressure_remap
    pressure_basis::Symbol      = :dry                   # :dry | :total
    tracer_basis::Symbol        = :tracer_mass_dry       # :tracer_mass_dry | :tracer_mass_total
    humidity_flux_correction::Symbol = :face_averaged    # :none | :face_averaged
    continuity_closure::Symbol  = :bt_weighted           # :bt_weighted | :mass_weighted
    mass_balance_mode::Symbol   = :none                  # :none | :column | :global
end
```

Then make all transport paths consume a canonical state contract:

- `m_ref` (cell air mass)
- `rm` (tracer mass)
- `am,bm` (horizontal mass fluxes, common units and sign convention)
- `cm` (vertical mass flux if `:continuity_cm`)
- `pe_src, pe_tgt` (if `:pressure_remap`)
- `qv` (if dry/total conversion is active)

## Unified Scheme for ERA5 vs GEOS Comparison (Exact Same Tracer Propagation Algorithms)

To compare ERA5 transport and GEOS transport with exactly the same propagation algorithms, only met ingestion should differ; transport operators must be identical.

### Canonical comparison protocol

1. Select one common grid/level definition and one `TransportMassFluxPolicy`.
2. Normalize both met sources to the same canonical flux state:
   - ERA5 path:
     - either precompute and load mass fluxes (`PreprocessedLatLonMetDriver`)
     - or compute from raw winds then immediately normalize into the same canonical `am,bm,m_ref` representation
   - GEOS path:
     - convert imported `MFXC/MFYC` to same units and sign conventions
3. Apply identical transport stack:
   - same horizontal operator (`slopes` or `PPM`)
   - same vertical operator (`continuity_cm` or `remap`)
   - same split order and CFL policy
   - same dry/total basis and tracer-basis conversion rules
4. Compute the same diagnostics for both runs:
   - global tracer mass
   - column mass closure residuals
   - pressure tendency mismatch
   - layer-by-layer mass budgets
5. Compare ERA5 vs GEOS outputs; differences should now mainly reflect meteorology, not algorithmic path divergence.

### Two useful comparison modes

- **Mode A: Same algorithm, native met characteristics preserved**
  - Keep each source native in time and geometry as much as possible.
  - Use same transport operators and basis rules.
  - Best for "meteorology impact" attribution.
- **Mode B: Strictly harmonized forcing**
  - Regrid/relevel both sources to one identical transport grid and cadence before run.
  - Use same transport operators and basis rules.
  - Best for apples-to-apples numerical benchmarking.

## Concrete Implementation Plan in AtmosTransportModel

### Phase 1: Centralize policy and basis choices

- Add `TransportMassFluxPolicy` type and defaults.
- Replace scattered metadata checks with policy-driven dispatch in run loop and phase functions.
- Suggested touch points:
  - `src/Models/Models.jl`
  - `src/Models/run_loop.jl`
  - `src/Models/physics_phases.jl`

### Phase 2: Canonical flux normalization API

- Add IO-side adapter function that returns canonical `am,bm,m_ref,ps,delp,qv` from any met driver.
- Ensure unit and sign normalization is explicit and tested.
- Suggested touch points:
  - `src/IO/abstract_met_driver.jl`
  - `src/IO/era5_met_driver.jl`
  - `src/IO/preprocessed_latlon_driver.jl`
  - `src/IO/geosfp_cs_met_driver.jl`

### Phase 3: Wire dry/wet conversion consistently

- Promote existing LL/CS dry-air kernels into policy-controlled preprocessing stage.
- Ensure flux and mass basis transformations happen once and in a documented order.
- Suggested touch points:
  - `src/Advection/latlon_dry_air.jl`
  - `src/Advection/cubed_sphere_mass_flux.jl`
  - `src/Models/physics_phases.jl`

### Phase 4: Unify vertical-operator dispatch

- Dispatch on `policy.vertical_operator`:
  - `:continuity_cm` -> `compute_cm_*` + z advection
  - `:pressure_remap` -> horizontal transport + remap
- Ensure both operators consume same upstream basis conventions.
- Suggested touch points:
  - `src/Models/run_helpers.jl`
  - `src/Models/physics_phases.jl`

### Phase 5: Add a reproducible ERA5-vs-GEOS experiment harness

- Build a single config schema that sets source + policy separately.
- Run matrix: `{ERA5, GEOS} x {policy variants}` with shared diagnostics.
- Suggested touch points:
  - `src/IO/configuration.jl`
  - `config/*.toml`
  - `scripts/*comparison*.jl`
  - `test/`

### Phase 6: Validation tests

- Unit tests:
  - dry/total transforms for mass and fluxes
  - sign/units normalization from each driver
- Integration tests:
  - same synthetic fluxes from two driver paths produce same tracer evolution
  - ERA5-vs-GEOS harness reproduces expected mass-closure metrics

## Recommended First Milestone

Implement a minimal "policy lock" mode:

- force both ERA5 and GEOS runs to `:flux_form_slopes + :continuity_cm + :dry`
- use one canonical normalization path
- emit standard budget diagnostics every window

This is the shortest path to eliminating algorithm-path drift and making ERA5-vs-GEOS transport comparisons interpretable.
