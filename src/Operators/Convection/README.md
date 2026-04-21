# Convection

Convective transport operators and workspaces.

This folder owns the convection operator hierarchy and the current
CMFMC implementation. As of plan 22D, the convection block is live
in `TransportModel.step!` and `CMFMCConvection` supports all three
topologies: structured LatLon, face-indexed reduced Gaussian, and
panel-native cubed sphere. See
[`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md) for the canonical
operator × topology matrix.

## Entry Points

- Type hierarchy:
  [`operators.jl`](operators.jl)
  defines `AbstractConvectionOperator`, `NoConvection`, and
  `apply_convection!`
- Concrete structured operator:
  [`CMFMCConvection.jl`](CMFMCConvection.jl)
  defines `CMFMCConvection` and its `apply!` methods
- Workspace:
  [`convection_workspace.jl`](convection_workspace.jl)
  defines `CMFMCWorkspace` and `invalidate_cmfmc_cache!`
- Kernel implementation:
  [`cmfmc_kernels.jl`](cmfmc_kernels.jl)
- Forcing contract:
  `ConvectionForcing` lives in [`../../MetDrivers/`](../../MetDrivers/)

## Current Status

- `CMFMCConvection` runs on all three topologies via dedicated
  `apply!` methods in [`CMFMCConvection.jl`](CMFMCConvection.jl):
  - LatLon (rank-4 `tracers_raw`)
  - reduced Gaussian (rank-3 face-indexed `tracers_raw`)
  - cubed sphere (`NTuple{6}` panel storage)
- `TransportModel.step!` executes a convection block when the model
  carries a non-`NoConvection` operator; wiring landed as plan 22D
- `NoConvection` is a no-op (compile-time dead branch in `step!`)
- `DrivenSimulation` refreshes `model.convection_forcing` each substep
  from `sim.window.convection`

If you are extending convection behavior, read the existing topology
dispatches in [`CMFMCConvection.jl`](CMFMCConvection.jl) first — they
are genuine fast-path implementations, not generic wrappers.

## File Map

- [`Convection.jl`](Convection.jl) — submodule assembly and status notes
- [`operators.jl`](operators.jl) — type hierarchy, public helper surface,
  no-op paths
- [`convection_workspace.jl`](convection_workspace.jl) — workspace and
  cache invalidation
- [`cmfmc_kernels.jl`](cmfmc_kernels.jl) — CMFMC transport kernels and
  inline helpers
- [`CMFMCConvection.jl`](CMFMCConvection.jl) — concrete CMFMC operator,
  forcing validation, topology restrictions, state-level `apply!`

## Common Tasks

- Tracing the live block wiring:
  start in [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  at the convection block in `step!`, then follow the topology-
  specific `apply!` in [`CMFMCConvection.jl`](CMFMCConvection.jl)
- Debugging forcing compatibility:
  inspect `ConvectionForcing` producers in `MetDrivers/` and the
  validation logic in [`CMFMCConvection.jl`](CMFMCConvection.jl)
- Adding a new convection operator:
  subtype `AbstractConvectionOperator` in [`operators.jl`](operators.jl);
  provide per-topology `apply!` methods alongside the CMFMC dispatches
- Debugging numerical behavior:
  start with [`cmfmc_kernels.jl`](cmfmc_kernels.jl) and
  [`convection_workspace.jl`](convection_workspace.jl)

## Cross-Dependencies

- [`../../MetDrivers/`](../../MetDrivers/) owns `ConvectionForcing` and
  window refresh logic
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  owns the convection block execution point
- [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)
  refreshes model forcing each substep
- [`../../State/`](../../State/) and [`../../Grids/`](../../Grids/)
  define `CellState` (LatLon, RG) and `CubedSphereState` (CS) runtime
  containers
- [`../../../docs/plans/PLAN_HISTORY.md`](../../../docs/plans/PLAN_HISTORY.md)
  carries the plan 22A/B/C/D retrospective

## Related Docs And Tests

- Runtime/block ordering target:
  [`../../../docs/plans/OPERATOR_COMPOSITION.md`](../../../docs/plans/OPERATOR_COMPOSITION.md)
- Tests:
  - [`../../../test/test_convection_types.jl`](../../../test/test_convection_types.jl)
  - [`../../../test/test_convection_forcing.jl`](../../../test/test_convection_forcing.jl)
  - [`../../../test/test_cmfmc_convection.jl`](../../../test/test_cmfmc_convection.jl)
