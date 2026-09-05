# Convection

This directory owns the convection operator hierarchy, forcing validation,
workspaces, and topology-specific kernels. Convection is a model operator: the
meteorology driver supplies a `ConvectionForcing`, while multiple dispatch
selects the implementation from the operator, state, and grid types.

See [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md) for the authoritative
operator × topology matrix and the rendered [operator guide](../../../docs/src/concepts/operators.md)
for user-facing configuration.

## Operator families

| Type | Forcing | Numerical path |
|---|---|---|
| `NoConvection` | none | compile-time no-op |
| `CMFMCConvection` | GEOS `cmfmc` with optional `dtrain` | direct interface-mass-flux redistribution |
| `TM5Convection` | `entu`, `detu`, `entd`, `detd` | TM5 four-field backward-Euler column matrix |
| `CMFMCMatrixConvection` | GEOS `cmfmc` / `dtrain` | derives a four-field column matrix, then reuses the TM5 solve |

The concrete operators support latitude–longitude, reduced-Gaussian, and
cubed-sphere storage through dedicated `apply!` methods. The model refreshes
forcing from the active transport window before applying the selected
operator.

## File map

- [`Convection.jl`](Convection.jl) — module assembly and exports.
- [`operators.jl`](operators.jl) — abstract hierarchy, no-op, and public helper
  surface.
- [`convection_workspace.jl`](convection_workspace.jl) — reusable CMFMC and TM5
  scratch storage.
- [`CMFMCConvection.jl`](CMFMCConvection.jl) — direct CMFMC operator,
  validation, and topology dispatch.
- [`cmfmc_kernels.jl`](cmfmc_kernels.jl) — direct CMFMC kernels.
- [`TM5Convection.jl`](TM5Convection.jl) — four-field TM5 operator and dispatch.
- [`tm5_column_solve.jl`](tm5_column_solve.jl) — backend-agnostic column-matrix
  construction and partial-pivot LU solve.
- [`tm5_kernels.jl`](tm5_kernels.jl) — latitude–longitude,
  reduced-Gaussian, and cubed-sphere kernel wrappers.
- [`CMFMCMatrixConvection.jl`](CMFMCMatrixConvection.jl) — matrix-form CMFMC
  operator.
- [`cmfmc_matrix_kernels.jl`](cmfmc_matrix_kernels.jl) — topology-specific
  forcing derivation kernels.

## Follow a runtime call

1. [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) owns the
   convection placement in `step!`.
2. [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)
   refreshes the forcing carrier.
3. The operator's `apply!` method validates the fields and launches the
   topology-specific kernel.

To add an implementation, subtype `AbstractConvection`, define its forcing and
workspace requirements, provide state/grid-dispatched `apply!` methods, and add
mass-conservation plus topology tests. Do not add an operator-name conditional
to the model loop.

### Factorization without downdrafts

CMFMC matrix convection derives updraft rates and has no downdraft pass.
Its transport matrix has zeros below the first subdiagonal (an
upper-Hessenberg matrix). Factorization therefore needs quadratic rather
than cubic work in the active level count. TM5 columns with no diagnosed
downdraft use the same optimization. Any positive diagnosed downdraft,
however small, keeps the general dense factorization.

Both routes retain partial pivoting and the same factor representation.
After Hessenberg LU, the solver checks whether any active row was swapped.
With no swaps, L has only one subdiagonal, so its forward solve (and the
adjoint's transpose-L solve) takes linear work per tracer. The U solve remains
quadratic. If a row was swapped, or a downdraft was diagnosed, the solver keeps
the general triangular solves: swaps can move earlier L entries below the
first subdiagonal. No forcing threshold, timestep change, vertical aggregation,
or additional approximation is introduced. CPU forward/adjoint checks exercise
both routes. Collaborative GPU kernels use the same selection once per column
for all tracer batches, pending A100 validation. See the
[CPU measurements and validation](../../../docs/memos/2026-09-05_matrix_convection_rhs.md).

### Multiple tracers with collaborative LU

For `TM5Convection` and `CMFMCMatrixConvection`, `use_collab_lu=true` uses
workgroup shared memory on Float32 GPUs. Each column is built and factored
once. Tracers are then loaded, solved, and stored in batches of six against
the retained factors. Six is the buffer capacity, not the total tracer limit;
seven, 32, or more tracers use additional batches without changing the matrix
or shared-memory allocation. The final batch may contain fewer than six.

Runtime workspaces defer the legacy solver's global matrix scratch when
collaborative LU is requested, including during backend adaptation. If a CPU,
Float64, or supported adjoint solve needs those buffers, it allocates the
configured tile once and reuses it. Cell metrics and cached CMFMC rates remain
available throughout. This avoids an unused allocation near the default 1 GiB
tile budget on the collaborative path.

The effective matrix depth still must fit 1..85 levels. Tracer batching does
not require vertical truncation or aggregation. CPU and Float64 requests use
the legacy solver with a warning. The collaborative changes have CPU arithmetic
coverage; device compilation, synchronization, and speed still require the
opt-in A100 regression in
[`test_tm5_tracer_batching_gpu.jl`](../../../test/diagnostic/test_tm5_tracer_batching_gpu.jl).

### Finding the implementation

- Tracing the live block wiring:
  start in [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  at the convection block in `step!`, then follow the topology-
  specific `apply!` in [`CMFMCConvection.jl`](CMFMCConvection.jl)
- Debugging forcing compatibility:
  inspect `ConvectionForcing` producers in `MetDrivers/` and the
  validation logic in [`CMFMCConvection.jl`](CMFMCConvection.jl)
- Adding a new convection operator:
  subtype `AbstractConvection` in [`operators.jl`](operators.jl);
  provide per-topology `apply!` methods alongside the CMFMC dispatches
- Debugging numerical behavior:
  start with [`cmfmc_kernels.jl`](cmfmc_kernels.jl) and
  [`convection_workspace.jl`](convection_workspace.jl)

## Tests

- [`../../../test/core/test_transport_model_convection.jl`](../../../test/core/test_transport_model_convection.jl)
- [`../../../test/core/test_tm5_convection.jl`](../../../test/core/test_tm5_convection.jl)
- [`../../../test/core/test_cmfmc_matrix_convection.jl`](../../../test/core/test_cmfmc_matrix_convection.jl)
- [`../../../test/core/test_tm5_vs_cmfmc_parity.jl`](../../../test/core/test_tm5_vs_cmfmc_parity.jl)
