# Diffusion

Implicit vertical diffusion for the transport runtime.

This folder owns the Thomas-solve-based vertical diffusion operator and
the topology-specific runtime adapters that let the same operator family
run on structured, face-indexed, and panel-native cubed-sphere state.

## Entry Points

- Operator types:
  [`operators.jl`](operators.jl)
  defines `AbstractDiffusionOperator`, `NoDiffusion`, and
  `ImplicitVerticalDiffusion`
- Model-facing runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply!(state, meteo, grid, op, dt; workspace)`
- Array-level runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply_vertical_diffusion!`
- Numerical reference pieces:
  [`thomas_solve.jl`](thomas_solve.jl)
  provides `solve_tridiagonal!` and `build_diffusion_coefficients`
- Kernel implementations:
  [`diffusion_kernels.jl`](diffusion_kernels.jl)

## Supported Layouts

- Structured:
  `q_raw :: (Nx, Ny, Nz, Nt)` with rank-3 Kz fields
- Face-indexed reduced Gaussian:
  `q_raw :: (ncells, Nz, Nt)` or `(ncells, Nz)` with rank-2 Kz fields
- Cubed sphere:
  `NTuple{6}` of halo-padded `(Nc + 2Hp, Nc + 2Hp, Nz)` tracer panels,
  with panel-native Kz wrapped in
  [`../../State/Fields/CubedSphereField.jl`](../../State/Fields/CubedSphereField.jl)

## File Map

- [`Diffusion.jl`](Diffusion.jl) — submodule assembly and public exports
- [`thomas_solve.jl`](thomas_solve.jl) — reference Thomas solve and
  coefficient builder
- [`diffusion_kernels.jl`](diffusion_kernels.jl) — KernelAbstractions
  kernels for structured, face-indexed, and cubed-sphere panel solves
- [`operators.jl`](operators.jl) — operator hierarchy, constructor
  validation, state-level `apply!`, array-level `apply_vertical_diffusion!`

## Common Tasks

- Adding a new Kz field type:
  make it satisfy the field contract in `State/Fields`, then validate it
  through [`operators.jl`](operators.jl)
- Debugging a workspace mismatch:
  read the shape checks in [`operators.jl`](operators.jl) before looking
  at kernels
- Debugging cubed-sphere diffusion:
  check the panel-native workspace shape and `CubedSphereField`
  handling before touching arithmetic
- Verifying arithmetic changes:
  keep [`thomas_solve.jl`](thomas_solve.jl) and the kernel formulas in
  [`diffusion_kernels.jl`](diffusion_kernels.jl) aligned
- Tracing runtime integration:
  follow calls from `TransportModel.step!` into advection midpoint hooks
  in `../Advection/StrangSplitting.jl`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage, `eachtracer`,
  and time-varying field contracts
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time`,
  which time-varying Kz fields consume
- [`../Advection/StrangSplitting.jl`](../Advection/StrangSplitting.jl)
  embeds diffusion at the transport midpoint
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  determines whether diffusion is active in a given run

## Related Docs And Tests

- Runtime/block ordering:
  [`../../../docs/plans/OPERATOR_COMPOSITION.md`](../../../docs/plans/OPERATOR_COMPOSITION.md)
- Topology status:
  [`../../../docs/plans/22_TOPOLOGY_COMPLETION_PLAN_v2.md`](../../../docs/plans/22_TOPOLOGY_COMPLETION_PLAN_v2.md)
- Tests:
  - [`../../../test/test_diffusion_kernels.jl`](../../../test/test_diffusion_kernels.jl)
  - [`../../../test/test_diffusion_operator.jl`](../../../test/test_diffusion_operator.jl)
  - [`../../../test/test_transport_model_diffusion.jl`](../../../test/test_transport_model_diffusion.jl)
  - [`../../../test/test_diffusion_palindrome.jl`](../../../test/test_diffusion_palindrome.jl)
  - [`../../../test/test_driven_simulation.jl`](../../../test/test_driven_simulation.jl)
  - [`../../../test/test_cubed_sphere_runtime.jl`](../../../test/test_cubed_sphere_runtime.jl)
