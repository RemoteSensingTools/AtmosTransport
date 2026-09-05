# Structured factorization for matrix convection

Implemented 2026-09-04 after [tracer batching](2026-09-04_matrix_convection_batching.md)
and [deferred scratch](2026-09-04_matrix_convection_workspace.md). GPU execution
remains paused. The earlier Hessenberg prototype is now integrated into the
forward and adjoint column solvers and the collaborative kernel row bounds.

## Selection and arithmetic

The existing cloud diagnosis returns `icllfs > Nz` when there is no downdraft
pass. In that case the matrix builder produces an upper-Hessenberg matrix:
updraft transport supplies the upper triangle, subsidence adds the first
subdiagonal, and flux-divergence assembly introduces no entries below it.
CMFMC matrix convection always takes this route; TM5 columns without a
diagnosed downdraft also qualify. Any positive diagnosed downdraft, including
very small values, keeps the general dense factorization. No forcing threshold
or additional physical approximation is introduced.

`_tm5_factorize!` selects between the unchanged general `_tm5_lu!` and the new
`_tm5_hessenberg_lu!`. The latter searches rows k and k+1 for each pivot and
updates only row k+1, reducing factorization from cubic to quadratic work in
the active level count. Full active-row swaps retain earlier L multipliers.
The same factor/pivot representation feeds the existing forward and transpose
triangular solves. Those solves remain general: row swaps can move earlier
L entries below the first subdiagonal.

Selection occurs once outside the serial LU loops. The general dense routine's
body is unchanged from the preceding commit. The regular column solve and both
CS adjoint replay column entry points use the selector. Each of the three
collaborative kernels uses the corresponding workgroup-uniform row bound for
pivot search, multiplier calculation, and trailing updates; full row swaps,
barriers, and tracer batching remain in place.

This optimization reduces arithmetic, not matrix storage. The shared-memory
matrix remains dense, so the 85-level collaborative envelope is unchanged.
Factors are still reused across all tracer batches in a call.

## CPU validation

- 850 new assertions in `test/core/test_tm5_hessenberg.jl` pass on Julia 1.12.6
  and 1.10.12. They cover pivot stress matrices, residuals, 65-tracer physical
  columns, inactive rows, no/finite/tiny downdrafts, and both adjoint directions.
  Physical forward and transpose results match explicitly dense LU bit for bit
  in these tests. Factor values and pivot sequences also agree.
- 2,371 assertions pass across the new tests and existing TM5, CMFMC matrix,
  batching, alias/tile equality, and workspace-lifetime tests on Julia 1.12.6.
  Six CUDA testsets are skipped.
- The benchmark probe now calls the production structured routine, with general
  LU as its independent reference. Its 2,489 assertions pass, including stress
  fixtures with 1,958 row swaps.
- The future A100 regression now compares against an explicitly dense CPU
  reference and mixes updraft-only, finite-downdraft, and tiny-downdraft columns
  within a launch. Its fixture/reference generation passes 96 CPU checks.
- A clean export of staged changes passes Aqua (10 assertions), the JET
  inference gate (179 reports against the existing 181-report baseline), and
  all 850 focused structured-LU assertions. Unrelated working-tree edits are
  excluded from this export.

## Serial CPU timings

Julia 1.12.6, Intel Xeon Platinum 8462Y+, Float32 synthetic CMFMC columns.
Medians of nine samples with 30 repetitions after warmup; buffers are reused.
Complete-column times include matrix construction, factorization, RHS copy,
and triangular solves. Dense and explicitly structured timings use six-wide
RHS batches; the automatic production entry point uses its normal all-tracer
CPU solve. Neither includes forcing derivation, grid-wide kernel overhead,
advection, or I/O.

| Levels | Tracers | Dense column, µs | Structured column, µs | Matched CPU speedup | Automatic production column, µs |
| --- | --- | --- | --- | --- | --- |
| 60 | 6 | 45.5 | 17.3 | 2.63× | 17.1 |
| 85 | 6 | 107.3 | 33.6 | 3.19× | 33.3 |
| 85 | 32 | 216.6 | 135.1 | 1.60× | 134.6 |
| 137 | 6 | 383.6 | 84.7 | 4.53× | 84.3 |

At 85 levels, six tracers, factorization plus matrix copy takes approximately
74.4 µs for dense LU and 2.1 µs for structured LU. As tracer count increases,
triangular solves account for more of the remaining cost. These measurements
are evidence for serial CPU work reduction, not A100 speed predictions.

[Full benchmark results](../../scripts/benchmarks/results/matrix_convection_structured_lu_cpu_20260904.toml)
and [reproduction script](../../scripts/benchmarks/probe_matrix_convection_cpu.jl).

## GPU validation still pending

No GPU kernel was compiled for a device or executed in this follow-up. The
opt-in A100 suite still needs device correctness, race checks, and timings.
Profile the serial matrix build, barriers, and RHS solves after the LU work
reduction; their fractions of total runtime will increase. Real-forcing and
full-window benchmarks remain necessary before reporting production GPU gains.
