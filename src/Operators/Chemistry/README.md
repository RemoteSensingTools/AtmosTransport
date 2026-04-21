# Chemistry

Tracer source/sink operators applied after transport.

This folder owns the chemistry operator hierarchy, the current decay
operators, and the `chemistry_block!` composition step that
`TransportModel.step!` runs after the transport block.

## Entry Points

- Type hierarchy and concrete operators:
  [`Chemistry.jl`](Chemistry.jl)
  defines `AbstractChemistryOperator`, `NoChemistry`,
  `ExponentialDecay`, and `CompositeChemistry`
- Kernel implementation:
  [`chemistry_kernels.jl`](chemistry_kernels.jl)
- Step-level block composition:
  [`chemistry_block.jl`](chemistry_block.jl)
  provides `chemistry_block!`
- Model-facing runtime entrypoint:
  [`Chemistry.jl`](Chemistry.jl)
  provides `apply!(state, meteo, grid, op, dt; workspace=nothing)`

## Current Scope

- `CellState` chemistry is live
- Time-varying scalar rate fields are supported through the
  `AbstractTimeVaryingField{FT, 0}` contract
- Chemistry is not yet wired for `CubedSphereState`

## File Map

- [`Chemistry.jl`](Chemistry.jl) — submodule assembly, operator types,
  state-level `apply!`, and composition rules
- [`chemistry_kernels.jl`](chemistry_kernels.jl) — fused multi-tracer
  decay kernel
- [`chemistry_block.jl`](chemistry_block.jl) — post-transport chemistry
  block called from the model step

## Common Tasks

- Adding a new chemistry operator:
  define the type and `apply!` method in [`Chemistry.jl`](Chemistry.jl),
  then decide whether it belongs inside `CompositeChemistry`
- Adding a time-varying scalar rate:
  implement the field in `State/Fields`, then plug it into
  `ExponentialDecay`
- Debugging tracer selection:
  inspect `tracer_index` resolution in [`Chemistry.jl`](Chemistry.jl)
  before changing kernel code
- Tracing runtime behavior:
  start at [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  and then follow `chemistry_block!`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage and the
  time-varying field interface
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time` for
  time-varying rate fields
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  executes `chemistry_block!` after transport
- [`../../../docs/plans/OPERATOR_COMPOSITION.md`](../../../docs/plans/OPERATOR_COMPOSITION.md)
  defines the block-order contract

## Related Docs And Tests

- Tests:
  - [`../../../test/test_chemistry.jl`](../../../test/test_chemistry.jl)
  - [`../../../test/test_current_time.jl`](../../../test/test_current_time.jl)
