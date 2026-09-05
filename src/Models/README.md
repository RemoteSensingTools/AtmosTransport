# Models

Minimal runtime composition layer for `src`.

This folder turns the lower-level state, grid, met-driver, and operator
pieces into runnable model objects. If you want to understand what
actually happens during a step, this is one of the first folders to
read.

## Entry Points

- Module assembly:
  [`Models.jl`](Models.jl)
- Core runtime object:
  [`TransportModel.jl`](TransportModel.jl)
  defines `TransportModel`, `step!`, and the `with_*` operator installers
- Fixed-Δt smoke harness:
  [`Simulation.jl`](Simulation.jl)
  defines `Simulation` and `run!`
- Window-driven production-style harness:
  [`DrivenSimulation.jl`](DrivenSimulation.jl)
  defines `DrivenSimulation`, window progression, forcing refresh, and
  runtime validation

## Runtime Composition Today

- `TransportModel.step!` runs:
  - transport block (advection, with diffusion and surface flux at
    the Strang midpoint)
  - convection block (CMFMC, matrix CMFMC, or TM5 on supported topologies)
  - chemistry block

## File Map

- [`Models.jl`](Models.jl) — submodule assembly
- [`TransportModel.jl`](TransportModel.jl) — main model struct,
  constructors, operator installers, runtime block order
- [`Simulation.jl`](Simulation.jl) — simple fixed-step loop for direct
  model runs
- [`DrivenSimulation.jl`](DrivenSimulation.jl) — met-window-driven loop,
  forcing interpolation, air-mass refresh, and runtime compatibility checks
- [`RuntimeRecipeStyles.jl`](RuntimeRecipeStyles.jl) — runtime-style traits
  (`AbstractRuntimeRecipeStyle` + LatLon/ReducedGaussian/CubedSphere) the
  physics-spec `materialize` methods dispatch on
- [`RuntimePhysicsSpecs.jl`](RuntimePhysicsSpecs.jl) — typed config specs parsed
  once from TOML + `materialize` (Oceananigans-style); convection family
  (`convection_spec`, the `lmax_conv`/`n_merge`-needs-`use_collab_lu` guard) +
  advection family (`advection_spec`, LinRood is cubed-sphere only) +
  chemistry family (`chemistry_spec`, `materialize` dispatches on run `FT`) +
  diffusion family (`diffusion_spec`; `materialize(spec, style, FT, context)`
  threads all three — topology/capability helpers stay in `RuntimePhysicsRecipe.jl`)
- [`RuntimePhysicsRecipe.jl`](RuntimePhysicsRecipe.jl) — topology-dispatched
  runtime recipe construction and capability validation for advection,
  diffusion, convection, chemistry, and surface forcing
- [`InitialConditionIO.jl`](InitialConditionIO.jl) — topology-dispatched
  VMR builder (`build_initial_mixing_ratio` on LL/RG/CS),
  basis-aware VMR → tracer-mass packer (`pack_initial_tracer_mass`),
  surface-flux loader + LL/RG/CS `build_surface_flux_source` builders
  with conservative regrid + cell-area integration,
  `FileInitialConditionSource` / `FileSurfaceFluxField` containers
- [`BinaryPathExpander.jl`](BinaryPathExpander.jl) —
  `expand_binary_paths(input_cfg)` resolves either an explicit
  `binary_paths = [...]` list or a `folder + start_date + end_date
  (+ file_pattern)` shape to a sorted `Vector{String}`; continuity
  check on the closed date range
- [`DrivenRunner.jl`](DrivenRunner.jl) — library-level
  `run_driven_simulation(cfg)` entry point for all driven runs. Owns the
  runtime flow behind `scripts/run_transport.jl`: first-driver
  construction, config and capability validation against TOML physics,
  tracer init via `build_initial_mixing_ratio` + basis-aware
  `pack_initial_tracer_mass`, surface-source wiring, GPU-residency
  assertion (`feedback_verify_gpu_runs_on_gpu`), per-window loop,
  and snapshot NetCDF output
- [`runner/`](runner/) — the runner's progress timer, configuration validation,
  runtime summary, output helpers, and model setup. These files are included
  inside `DrivenRunner`; the top-level file retains the transport loops.
- [`initial_conditions/`](initial_conditions/) — cubed-sphere initialization,
  surface-inventory loading and storage-unit conversion, and conservative
  surface-flux remapping, included inside `InitialConditionIO`.
- [`InputStaging.jl`](InputStaging.jl) — opt-in rolling NVMe input staging
  (`InputStager`, `staged_path_for!`, `cleanup_staging!`) for the per-day
  binary loop: copies upcoming days NAS→local NVMe ahead of the GPU loop and
  evicts processed days, bounding local-disk use for multi-month/year runs.
  Default off ⇒ bit-identical to a non-staged run

## Common Tasks

- Changing operator block order:
  start in [`TransportModel.jl`](TransportModel.jl) and the runtime
  walkthrough in [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Debugging "operator exists but never runs":
  check `TransportModel.step!` before editing operator code
- Debugging driver/model mismatch:
  start in [`DrivenSimulation.jl`](DrivenSimulation.jl), especially grid
  and basis compatibility checks
- Adding a new model-level runtime option:
  decide whether it belongs on `TransportModel`, `DrivenSimulation`, or
  both before threading it through the step loop

## Cross-Dependencies

- [`../State/README.md`](../State/README.md) provides the state and flux
  containers carried by the model
- [`../Operators/README.md`](../Operators/README.md) provides the actual
  physics blocks the model calls
- [`../MetDrivers/README.md`](../MetDrivers/README.md) provides the
  window-driven forcing and timing contracts
- [`../Grids/README.md`](../Grids/README.md) determines topology and
  therefore runtime dispatch

## Related Docs And Tests

- Runtime walkthrough:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Block-order design:
  [`TransportModel.jl`](TransportModel.jl) and
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../test/core/test_driven_simulation.jl`](../../test/core/test_driven_simulation.jl)
  - [`../../test/core/test_no_advection.jl`](../../test/core/test_no_advection.jl)
  - [`../../test/orphan/test_transport_model_emissions.jl`](../../test/orphan/test_transport_model_emissions.jl)
  - [`../../test/orphan/test_current_time.jl`](../../test/orphan/test_current_time.jl)
