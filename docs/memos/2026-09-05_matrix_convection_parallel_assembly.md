# Parallel final assembly of the convection matrix

The [V100 follow-up](2026-09-05_matrix_convection_v100.md) measured low warp
occupancy and showed that the remaining serial matrix work needed attention.
This change distributes the final flux-divergence assembly across the existing
32-thread workgroup in all three collaborative kernels.

## Dependency and arithmetic

The plume recurrences and subsidence subtraction produce the intermediate
flux matrix `f` in thread 1. Final assembly forms
`A[k,j] = -dt * (f[k+1,j] - f[k,j])`, with the identity added on the diagonal.
Each matrix column is independent. Within a column, rows must be traversed
from top to bottom so that the in-place write does not overwrite the next
row before it is read.

A workgroup barrier now separates the serial recurrences from this assembly.
Each thread owns whole matrix columns and preserves the original ascending
row order, active-block guards, and diagonal arithmetic. The existing barrier
before LU remains. No extra shared storage, forcing approximation, changed
factorization, or different tracer batching is introduced.

## V100 results

Same synthetic 4,096-column benchmark, CUDA 12.6 environment, and nine-sample
median methodology as the preceding V100 report. Comparison is against the
already optimized `9f5008db` kernels, not against pre-batching code.

| Levels | Tracers | Forcing | Before, ms | Parallel assembly, ms | Further speedup |
| --- | --- | --- | --- | --- | --- |
| 60 | 6 | Updraft only | 5.73 | 3.03 | 1.89× |
| 85 | 6 | Updraft only | 15.85 | 6.81 | 2.33× |
| 85 | 65 | Updraft only | 28.59 | 19.81 | 1.44× |
| 85 | 6 | With downdrafts | 24.41 | 15.50 | 1.57× |
| 85 | 65 | With downdrafts | 47.62 | 38.91 | 1.22× |

The six-tracer downdraft time is now below the original pre-batching kernel's
22.99 ms. Shared allocation and the theoretical occupancy ceiling remain
unchanged. These measurements exclude setup, transfers, forcing derivation,
I/O, and the rest of the transport model.

## Validation

- The V100 tracer-batching suite passes all 487 assertions on the new kernels.
- All 72 output arrays from the suite's topology/depth/tracer/forcing cases
  match the previous GPU kernels bit for bit, including halos and inactive rows
  (73 assertions including the case-key check).
- CUDA Compute Sanitizer racecheck reports zero errors and zero warnings for
  six deep 65-tracer fixtures, with 40 numerical assertions passing.
- All benchmark cases pass dense CPU reference, conservation, and positivity
  checks before timing.
- The existing TM5 integration suite also passes on V100, including comparison
  with legacy kernels, depth truncation, vertical aggregation, and repeated
  clipped-cloud conservation checks.

[Raw measurements](../../scripts/benchmarks/results/matrix_convection_parallel_20260905/v100.toml)
use the existing [GPU benchmark](../../scripts/benchmarks/bench_matrix_convection_gpu.jl).
The independent baseline arrays were captured from the preceding clean export;
the optimized arrays were captured from a separate export with only this kernel
change. GPU tests and benchmarks ran on tofu's V100 GPU 0.
