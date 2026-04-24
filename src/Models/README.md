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
  - convection block (`CMFMCConvection` live on LatLon, RG, CS via
    plan 22D; `TM5Convection` in progress under plan 23)
  - chemistry block

## File Map

- [`Models.jl`](Models.jl) — submodule assembly
- [`TransportModel.jl`](TransportModel.jl) — main model struct,
  constructors, operator installers, runtime block order
- [`Simulation.jl`](Simulation.jl) — simple fixed-step loop for direct
  model runs
- [`DrivenSimulation.jl`](DrivenSimulation.jl) — met-window-driven loop,
  forcing interpolation, air-mass refresh, and runtime compatibility checks
- [`InitialConditionIO.jl`](InitialConditionIO.jl) — topology-dispatched
  VMR builder (`build_initial_mixing_ratio` on LL/RG/CS),
  basis-aware VMR → tracer-mass packer (`pack_initial_tracer_mass`),
  surface-flux loader + LL/RG/CS `build_surface_flux_source` builders
  with conservative regrid + cell-area integration,
  `FileInitialConditionSource` / `FileSurfaceFluxField` containers
  (plan 40 Commits 1b–1d + 2)
- [`BinaryPathExpander.jl`](BinaryPathExpander.jl) —
  `expand_binary_paths(input_cfg)` resolves either an explicit
  `binary_paths = [...]` list or a `folder + start_date + end_date
  (+ file_pattern)` shape to a sorted `Vector{String}`; continuity
  check on the closed date range (plan 40 Commit 4)
- [`DrivenRunner.jl`](DrivenRunner.jl) — library-level
  `run_driven_simulation(cfg)` entry point for LL/RG driven runs
  (CS dispatch added in plan 40 Commit 6b). Owns the runtime flow
  hoisted from `scripts/run_transport_binary.jl`: first-driver
  construction, capability validation against TOML physics,
  tracer init via `build_initial_mixing_ratio` + basis-aware
  `pack_initial_tracer_mass`, surface-source wiring, GPU-residency
  assertion (`feedback_verify_gpu_runs_on_gpu`), per-window loop,
  snapshot NetCDF output (plan 40 Commit 6a)

## Common Tasks

- Changing operator block order:
  start in [`TransportModel.jl`](TransportModel.jl) and compare against
  [`../../docs/plans/OPERATOR_COMPOSITION.md`](../../docs/plans/OPERATOR_COMPOSITION.md)
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
  [`../../docs/plans/OPERATOR_COMPOSITION.md`](../../docs/plans/OPERATOR_COMPOSITION.md)
- Tests:
  - [`../../test/test_driven_simulation.jl`](../../test/test_driven_simulation.jl)
  - [`../../test/test_transport_model_diffusion.jl`](../../test/test_transport_model_diffusion.jl)
  - [`../../test/test_transport_model_emissions.jl`](../../test/test_transport_model_emissions.jl)
  - [`../../test/test_current_time.jl`](../../test/test_current_time.jl)
