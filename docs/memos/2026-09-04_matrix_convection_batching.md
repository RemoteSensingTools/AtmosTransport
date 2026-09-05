# Matrix convection tracer batching implementation

Follow-up to [the performance review](2026-09-04_matrix_convection_performance.md).
Implemented on 2026-09-04 with GPU execution paused by user instruction.

## Behavior

The Float32 collaborative LU kernels for lat-lon, reduced Gaussian, and
cubed-sphere panels now process any positive tracer count in batches of six.
Each column's matrix is constructed and factored once, before the batch loop.
The shared RHS array remains six slots wide; additional tracers do not increase
its allocation. The final partial batch uses only its valid slots.

The same kernel path serves `TM5Convection` and `CMFMCMatrixConvection`.
The effective matrix-depth limit remains 85 levels. Neither vertical truncation
nor aggregation is required to accommodate more tracers. CPU/Float64 fallback
behavior is unchanged. The previous total-tracer rejection is removed.

Each active thread owns one RHS through permutation, forward substitution, and
back substitution. These operations are shared across the three kernels through
an inline helper containing no workgroup operations. Pivot application is no
longer serial across tracers. Two intermediate solve barriers are eliminated;
a store-completion barrier protects the buffer before the next batch loads it.
Load and solve-completion barriers remain. LU arithmetic is unchanged.

## Validation completed without GPU execution

- 979 assertions in `test/core/test_tm5_tracer_batching.jl`, on Julia 1.12.6 and
  1.10.12: positive tracer counts through 129, partial batches, pivot swaps,
  identity/active rows, residuals, immutable factors, and untouched unused slots.
  The production RHS helper matches the existing CPU solve bit for bit for
  Float32 and Float64. Host selection uses a dummy GPU backend without a device.
- 434 existing assertions on Julia 1.12.6: TM5 topology application and column
  solves, CMFMC matrix derivation/conservation/adjoint, tile equality, alias
  safety, and CMFMC adjoint identity. Six CUDA testsets were skipped.
- 96 CPU checks of the new GPU suite's fixtures: positivity, conservation,
  cloud-free columns, and passthrough layers at 8 and 91 total levels.
- All three topology batch bodies checked for agreement after substituting
  topology coordinates. Package loading validates the KA kernel declarations.
- Julia 1.10 checks used `/tmp/atmos-review-julia110`, the previously resolved
  compatible environment. The workspace's Julia 1.12 manifest could not load
  its PrecompileTools version on 1.10; no project dependencies were changed.

## Device validation still required

`test/diagnostic/test_tm5_tracer_batching_gpu.jl` is explicitly opt-in and
requires an A100. It checks the actual three topology kernels with 1, 6, 7, 12,
32, and 65 tracers; cloud-free, shallow, and deep clouds; updraft-only and
downdraft forcing; a full column and a truncated active span; and CS halos.
Besides the CPU reference, it compares each result bitwise with independent
launches containing at most six tracers.

When GPU use is authorized again, run on the selected A100:

```sh
ATMOSTR_RUN_MATRIX_BATCH_GPU_TESTS=1 julia --project=. \
  test/diagnostic/test_tm5_tracer_batching_gpu.jl
```

The test's default path skips without importing CUDA. No A100/Metal kernel
execution, device compilation, race checking, or speed measurement has been
performed for this implementation. CPU arithmetic equivalence cannot establish
device synchronization correctness or GPU throughput. Wider CUDA batches and
the CMFMC Hessenberg specialization remain separate follow-ups.
