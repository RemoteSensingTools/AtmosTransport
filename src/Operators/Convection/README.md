# Convection

Convective transport operators and workspaces.

This folder owns the convection operator hierarchy. As of plan 22D,
the convection block is live in `TransportModel.step!` and
`CMFMCConvection` supports all three topologies: structured LatLon,
face-indexed reduced Gaussian, and panel-native cubed sphere.
`TM5Convection` lands via plan 23 — Commit 1 ships the struct +
workspace + dispatch stubs; Commit 4 lands the real kernels on all
three topologies. See
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
- TM5 operator (in progress, plan 23):
  [`TM5Convection.jl`](TM5Convection.jl)
  defines `TM5Convection` — four-field Tiedtke 1989 with in-kernel
  partial-pivot LU solve. Commit 1 ships the struct + dispatch
  stubs; Commits 2 + 4 ship the column solver and per-topology
  kernels.
- Workspace:
  [`convection_workspace.jl`](convection_workspace.jl)
  defines `CMFMCWorkspace`, `TM5Workspace`, and
  `invalidate_cmfmc_cache!`
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
- `TM5Convection` type + `TM5Workspace` + dispatch stubs are live
  as of plan 23 Commit 1. The struct is stateless; forcing comes
  via `ConvectionForcing.tm5_fields`. `apply!` throws a stub
  `ArgumentError` pointing at Commit 4 — the type, workspace
  factory, and validator are exercised without fake numerics.
- `TransportModel.step!` executes a convection block when the model
  carries a non-`NoConvection` operator; wiring landed as plan 22D
- `NoConvection` is a no-op (compile-time dead branch in `step!`)
- `DrivenSimulation` refreshes `model.convection_forcing` each substep
  from `sim.window.convection`; plan 23 Commit 1 refactored the
  per-operator validator (`_validate_convection_window!`) into
  dispatch so adding operators does not re-edit the old if/elseif
  chain

If you are extending convection behavior, read the existing topology
dispatches in [`CMFMCConvection.jl`](CMFMCConvection.jl) first — they
are genuine fast-path implementations, not generic wrappers.

## File Map

- [`Convection.jl`](Convection.jl) — submodule assembly and status notes
- [`operators.jl`](operators.jl) — type hierarchy, public helper surface,
  no-op paths
- [`convection_workspace.jl`](convection_workspace.jl) — `CMFMCWorkspace`
  (CFL cache + scratch) and `TM5Workspace` (`conv1` matrix slab +
  pivots + cloud-dim indices); cache invalidation helper
- [`cmfmc_kernels.jl`](cmfmc_kernels.jl) — CMFMC transport kernels and
  inline helpers
- [`CMFMCConvection.jl`](CMFMCConvection.jl) — concrete CMFMC operator,
  forcing validation, topology restrictions, state-level `apply!`
- [`TM5Convection.jl`](TM5Convection.jl) — `TM5Convection` struct +
  dispatch stubs (plan 23 Commit 1). Real kernels land in plan 23
  Commit 4.

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
