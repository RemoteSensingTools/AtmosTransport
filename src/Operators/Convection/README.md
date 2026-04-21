# Convection

Convective transport operators and workspaces.

This folder owns the convection operator hierarchy and the current
CMFMC implementation, but it is important to separate "code exists" from
"runtime is live": the structured convection block is not yet executed
by `TransportModel.step!`, and reduced-Gaussian / cubed-sphere
convection are still deferred.

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

- Structured CMFMC kernels and workspaces exist
- `NoConvection` is a no-op everywhere
- `TransportModel.step!` does not yet execute a convection block
- Face-indexed reduced-Gaussian convection is deferred
- Cubed-sphere convection is deferred

If you are trying to "make convection work," start by checking whether
the missing piece is runtime block wiring or kernel support. Most recent
topology work has deliberately not invented a parallel convection runtime.

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

- Bringing convection live in the model runtime:
  start in [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl),
  not in the kernels
- Debugging forcing compatibility:
  inspect `ConvectionForcing` producers in `MetDrivers/` and the
  validation logic in [`CMFMCConvection.jl`](CMFMCConvection.jl)
- Extending topology coverage:
  read the existing rejection paths first; reduced Gaussian and cubed
  sphere are intentionally not "almost supported"
- Debugging numerical behavior:
  start with [`cmfmc_kernels.jl`](cmfmc_kernels.jl) and
  [`convection_workspace.jl`](convection_workspace.jl)

## Cross-Dependencies

- [`../../MetDrivers/`](../../MetDrivers/) owns `ConvectionForcing` and
  window refresh logic
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  will eventually own the convection block execution point
- [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)
  refreshes model forcing each substep
- [`../../State/`](../../State/) and [`../../Grids/`](../../Grids/)
  define the currently supported structured runtime containers
- [`../../../docs/plans/22_TOPOLOGY_COMPLETION_PLAN_v2.md`](../../../docs/plans/22_TOPOLOGY_COMPLETION_PLAN_v2.md)
  records why topology convection remains gated

## Related Docs And Tests

- Runtime/block ordering target:
  [`../../../docs/plans/OPERATOR_COMPOSITION.md`](../../../docs/plans/OPERATOR_COMPOSITION.md)
- Tests:
  - [`../../../test/test_convection_types.jl`](../../../test/test_convection_types.jl)
  - [`../../../test/test_convection_forcing.jl`](../../../test/test_convection_forcing.jl)
  - [`../../../test/test_cmfmc_convection.jl`](../../../test/test_cmfmc_convection.jl)
