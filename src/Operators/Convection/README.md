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

## Tests

- [`../../../test/core/test_transport_model_convection.jl`](../../../test/core/test_transport_model_convection.jl)
- [`../../../test/core/test_tm5_convection.jl`](../../../test/core/test_tm5_convection.jl)
- [`../../../test/core/test_cmfmc_matrix_convection.jl`](../../../test/core/test_cmfmc_matrix_convection.jl)
- [`../../../test/core/test_tm5_vs_cmfmc_parity.jl`](../../../test/core/test_tm5_vs_cmfmc_parity.jl)
