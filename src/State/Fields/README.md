# Fields

Time-varying field abstraction for operator inputs.

This folder provides the host-side field objects that physics operators
query through a uniform interface. The core idea is that operators
should not care whether a rate or diffusivity came from a constant, a
profile, a precomputed array, a derived meteorological calculation, or a
panel-native cubed-sphere wrapper.

## Entry Points

- Interface and common exports:
  [`Fields.jl`](Fields.jl)
- Precomputed spatial field wrapper:
  [`PreComputedKzField.jl`](PreComputedKzField.jl)
- Horizontally uniform profile field:
  [`ProfileKzField.jl`](ProfileKzField.jl)
- Panel-native cubed-sphere field wrapper:
  [`CubedSphereField.jl`](CubedSphereField.jl)
- Derived boundary-layer Kz:
  [`DerivedKzField.jl`](DerivedKzField.jl)
- Piecewise-constant time-varying field:
  [`StepwiseField.jl`](StepwiseField.jl)

## Interface Contract

Every concrete field type must satisfy:

- `field_value(f, idx) -> FT`
- `update_field!(f, t) -> f`

`field_value` must be kernel-safe and allocation-free. `update_field!`
runs on the host and may refresh caches before an operator launch.

## File Map

- [`Fields.jl`](Fields.jl) — abstract interface, `ConstantField`,
  exports, include order
- [`ProfileKzField.jl`](ProfileKzField.jl) — vertically varying,
  horizontally uniform Kz profiles
- [`PreComputedKzField.jl`](PreComputedKzField.jl) — wrapper around a
  rank-2 or rank-3 spatial array
- [`CubedSphereField.jl`](CubedSphereField.jl) — explicit panel-native
  wrapper over six structured rank-3 fields
- [`DerivedKzField.jl`](DerivedKzField.jl) — Kz derived from
  meteorological inputs and PBL parameter choices
- [`StepwiseField.jl`](StepwiseField.jl) — piecewise-constant-in-time
  field cache

## Common Tasks

- Adding a new operator-consumable field:
  implement the two required interface methods in a new file and include
  it from [`Fields.jl`](Fields.jl)
- Wiring time into an operator:
  check where `current_time(meteo)` is passed into `update_field!`
  before changing the field type
- Extending cubed-sphere physics cleanly:
  prefer [`CubedSphereField.jl`](CubedSphereField.jl) over inventing fake
  rank-polymorphic panel packing
- Debugging unexpected stale values:
  verify that the caller actually invokes `update_field!` once per
  operator application

## Cross-Dependencies

- [`../README.md`](../README.md) re-exports these field contracts into
  the runtime state layer
- [`../../Operators/Diffusion/README.md`](../../Operators/Diffusion/README.md)
  is the main consumer today
- [`../../Operators/Chemistry/README.md`](../../Operators/Chemistry/README.md)
  consumes scalar time-varying rate fields
- [`../../MetDrivers/README.md`](../../MetDrivers/README.md) and
  [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)
  provide the time source for `update_field!`

## Related Docs And Tests

- Field-model design:
  [`../../../docs/plans/TIME_VARYING_FIELD_MODEL.md`](../../../docs/plans/TIME_VARYING_FIELD_MODEL.md)
- Tests:
  - [`../../../test/test_fields.jl`](../../../test/test_fields.jl)
  - [`../../../test/test_diffusion_operator.jl`](../../../test/test_diffusion_operator.jl)
  - [`../../../test/test_current_time.jl`](../../../test/test_current_time.jl)
