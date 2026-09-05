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
  defines `AbstractConvection`, `NoConvection`, and
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
- `TM5Convection` is live on all three topologies (LatLon,
  ReducedGaussian, CubedSphere) as of plan 23 Commit 4. The struct
  is stateless; forcing comes via `ConvectionForcing.tm5_fields`.
  Implementation: `_tm5_solve_column!` builds and solves the
  `conv1 = I - dt·D` backward-Euler matrix with partial-pivot LU
  per column; `tm5_kernels.jl` ships the three topology KA kernels;
  `TM5Convection.jl` has the `apply!` / `apply_convection!` dispatch
  that launches them. Mass conservation to F64 machine precision
  via the TM5 matrix column-sum-is-1 invariant.
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
  topology dispatch, convection-depth validation, and workspace budget
- [`CMFMCMatrixConvection.jl`](CMFMCMatrixConvection.jl) — conservation-
  by-construction CMFMC variant; derives TM5 (entu/detu/entd/detd)
  from GEOS (cmfmc/dtrain) per column, then reuses the TM5 LU
  forward + adjoint. Selectable via `[convection].kind = "cmfmc_matrix"`.
- [`cmfmc_matrix_kernels.jl`](cmfmc_matrix_kernels.jl) — three
  derivation kernels (LL/RG/CS) for `CMFMCMatrixConvection`, with
  inline column closure so the boundary residual is absorbed at
  allocation time.
- [`tm5_column_solve.jl`](tm5_column_solve.jl) — backend-agnostic
  column solver `_tm5_solve_column!`: builds `conv1 = I - dt·D`,
  selects dense or upper-Hessenberg partial-pivot LU, and solves each tracer
- [`tm5_kernels.jl`](tm5_kernels.jl) — `@kernel` wrappers around
  `_tm5_solve_column!` and collaborative shared-memory LU for all three
  topologies: lat-lon, reduced Gaussian, and cubed-sphere panels

## Common Tasks

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
for all tracer batches. V100 device checks pass; A100 validation remains pending. See the
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
coverage and V100 device validation through 65 tracers. See the
[V100 timings and remaining bottlenecks](../../../docs/memos/2026-09-05_matrix_convection_v100.md).
The subsequent [parallel matrix assembly](../../../docs/memos/2026-09-05_matrix_convection_parallel_assembly.md)
distributes independent matrix columns across the workgroup while preserving
the top-to-bottom dependency within each column. It further improves V100
throughput without additional shared storage.
The opt-in CUDA regression defaults to an A100; set
`ATMOSTR_MATRIX_GPU_NAME=V100` when explicitly testing a V100:
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
## Related Docs And Tests

- Runtime/block ordering target:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../../test/test_convection_types.jl`](../../../test/test_convection_types.jl)
  - [`../../../test/test_convection_forcing.jl`](../../../test/test_convection_forcing.jl)
  - [`../../../test/test_cmfmc_convection.jl`](../../../test/test_cmfmc_convection.jl)
