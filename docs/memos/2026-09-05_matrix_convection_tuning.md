# V100 tuning experiments after parallel matrix assembly

Two follow-up prototypes were measured against the retained
[parallel final assembly](2026-09-05_matrix_convection_parallel_assembly.md).
Neither prototype is enabled in production.

## Measurements

Tesla V100-PCIE-16GB on tofu GPU 0, CUDA runtime 12.6, Float32,
4,096 synthetic columns. Times are medians of nine warmed device samples,
excluding setup, host transfers, forcing derivation, and I/O.

| 85 levels | Tracers | Retained kernel, ms | Parallel plume columns, ms | 32-slot RHS batch, ms |
| --- | ---: | ---: | ---: | ---: |
| Updraft only | 6 | 6.814 | 6.435 | 9.883 |
| Updraft only | 12 | 8.113 | 7.735 | 10.144 |
| Updraft only | 32 | 13.197 | 12.748 | 11.052 |
| Updraft only | 65 | 19.806 | 19.270 | 15.766 |
| With downdrafts | 6 | 15.500 | 15.176 | 22.608 |
| With downdrafts | 12 | 17.843 | 17.504 | 22.806 |
| With downdrafts | 32 | 27.112 | 26.690 | 23.517 |
| With downdrafts | 65 | 38.907 | 38.352 | 31.059 |

Parallel plume columns keep the scalar plume-mass recurrence serial, then
compute independent matrix columns concurrently. This adds forcing reads and
coefficient reconstruction to each column. The measured additional benefit is
small (roughly 1–6% in the table), so the simpler retained implementation remains
the default pending repeatability and wider validation.

The 32-slot batch trades fewer RHS batches for more shared memory. Compiled
shared allocation at 85 levels rises from 32,420 to 41,252 bytes per workgroup,
reducing the theoretical residency from three to two workgroups per V100 SM.
It improves 65-tracer timings about 20%, but makes six-tracer cases about 45%
slower. It must not replace the six-slot default globally. A future adaptive
choice needs intermediate/boundary tracer counts, different column counts,
and other GPU architectures; the present evidence does not establish a portable
selection threshold.

## Reproducibility and limits

Both prototypes pass the existing 487-assertion V100 topology/depth/tracer suite
and all benchmark CPU-reference, conservation, and positivity checks. They have
not received the independent bitwise comparison and Compute Sanitizer checks
used for the retained assembly change. Neither has been tested on A100.

[Raw measurements and source patches](../../scripts/benchmarks/results/matrix_convection_tuning_20260905/)
are retained for further work. Each source patch applies independently to
`db9bb0cd` (not sequentially). Benchmark revision labels describe the temporary
exports, whose common source is that parallel-assembly implementation. For the
32-slot experiment, the benchmark's logical shared-memory estimate was also
updated from six to 32 RHS slots; compiled resource measurements are obtained
from CUDA itself. Run with the CUDA 12.6 environment and the same
`bench_matrix_convection_gpu.jl ... native` command as the retained benchmark.
