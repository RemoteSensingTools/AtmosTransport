# Faster tracer solves after unpivoted Hessenberg LU

Implemented 2026-09-05, following the
[structured factorization](2026-09-04_matrix_convection_structured_lu.md).
GPU execution remains paused. This change reduces the work repeated for each
tracer after the matrix has been built and factored.

## Selection and arithmetic

Without a diagnosed downdraft, the matrix is upper-Hessenberg. If its partial
pivoting makes no row swaps, the stored lower factor L has only its unit
diagonal and first subdiagonal. Forward substitution then becomes
`q[k] -= L[k,k-1] * q[k-1]`, taking linear work per tracer. Back substitution
through U still takes quadratic work. The adjoint similarly keeps its general
U-transpose solve and uses one superdiagonal for the L-transpose solve.

Every caller checks both conditions: no diagnosed downdraft and an actual
identity pivot sequence over the active levels. Hessenberg structure alone
does not suffice, because row swaps can move earlier L entries below the
first subdiagonal. Any swap or diagnosed downdraft retains the general solve.
Partial pivoting, active-row boundaries, factor storage, and physical forcing
are unchanged. The specialization omits arithmetic on structural zeros; its
finite-input tests match the general solver bit for bit. It does not promise
identical propagation of nonfinite tracer values.

The serial forward solver and both adjoint column entry points use this
selection. All three collaborative GPU kernels share the forward RHS helper
and evaluate the same predicate once before the tracer batch loop. The
existing six-slot shared RHS buffer and stored factors are reused for every
batch; this adds no shared storage or synchronization barriers. Total tracer
count remains independent of that buffer capacity, and the collaborative
matrix depth limit remains 85.

## CPU measurements

Julia 1.12.6, Intel Xeon Platinum 8462Y+, Float32 synthetic CMFMC columns,
serial execution with reused buffers. Timings are medians of nine samples
with 30 repetitions after warmup. Both columns below use Hessenberg LU;
the baseline uses general triangular solves and the new timing calls the
automatic production column solver, including its structure/pivot checks.
Times include matrix construction, factorization, RHS copy, and solves.

| Levels | Tracers | General RHS column, µs | New column, µs | Further speedup |
| --- | --- | --- | --- | --- |
| 60 | 6 | 17.2 | 13.7 | 1.25× |
| 85 | 6 | 33.9 | 26.5 | 1.28× |
| 85 | 32 | 134.8 | 94.6 | 1.43× |
| 85 | 65 | 263.4 | 181.7 | 1.45× |
| 137 | 65 | 669.6 | 433.9 | 1.54× |

At 85 levels and 65 tracers, RHS copy plus triangular solves alone drops from
253.5 to 171.3 µs. U substitution remains quadratic and will dominate as tracer
count grows. These are synthetic serial CPU measurements, excluding forcing
derivation, grid-wide kernel overhead, advection, and I/O. They do not establish
production or GPU speedups. The 137-level measurements exercise the CPU path.

[Full results](../../scripts/benchmarks/results/matrix_convection_bidiagonal_rhs_cpu_20260905.toml)
and [reproduction script](../../scripts/benchmarks/probe_matrix_convection_cpu.jl).

## Validation

- 1,228 focused assertions pass on Julia 1.12.6 and 1.10.12, covering Float32
  and Float64, 1–85 levels, inactive prefixes, 1–129 tracers, strided views,
  partial shared-buffer batches, residuals, and both adjoint directions.
  General triangular solves provide the independent reference.
- The existing 850 structured-LU assertions also pass on both Julia versions,
  including automatic forward/adjoint selection and finite/tiny downdrafts.
- The broader Julia 1.12.6 run passes 3,507 assertions across these tests,
  tracer batching, TM5 and CMFMC matrix integration, conservation, alias safety,
  tile equality, and cross-scheme parity. Six CUDA testsets are skipped.
- The CPU benchmark's 2,489 reference/stress assertions pass.
- A clean export of the staged changes passes Aqua (10 assertions), the JET
  inference gate (179 reports against the existing 181-report baseline), and
  all 2,078 focused RHS/structured-LU assertions. Unrelated working-tree edits
  are excluded from this export.

No GPU kernel was compiled for a device or executed. The opt-in
[A100 suite](../../test/diagnostic/test_tm5_tracer_batching_gpu.jl) already mixes
updraft-only and downdraft columns and compares against explicitly dense CPU
LU, so it will exercise both RHS routes when GPU validation resumes. Device
compilation, synchronization correctness, and performance remain unverified.
