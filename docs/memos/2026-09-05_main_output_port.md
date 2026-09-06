# Output port onto current main

The old revamp's selected capture, incremental NetCDF, and output-task ownership
are ported onto main's format-4 runtime and typed architecture API. Main's
compensated signed-tracer totals and ATMSNAP header fields are preserved.

## Behavior

- `capture_snapshot(...; fields=spec)` retains the union of requested layers
  and required column reductions. Existing calls without `fields` retain full
  native state.
- Each selected tracer retains a compensated Float64 total even if all its
  layer and column diagnostics are disabled. CUDA returns separate compensated
  partial sums and corrections for host reduction. Metal copies bounded slabs
  for CPU Float64 accumulation because it has no device Float64 support.
- Single-file NetCDF output appends and flushes individual records. Daily
  output keeps one owned background write, which the outer run drains on any
  exit. A cleanup error does not replace the original run error.
- Streams record `completed_snapshots`, reject incompatible frames before
  writing, and close after an I/O failure. Resumption is not implemented.
- Cubed-sphere visualization reads column-only files and maps selected layers
  to their original model indices. Missing levels produce an explicit error.
- Batch shape, mass-basis, and total validation runs before opening an existing
  output for replacement.

## Evidence

Focused CPU tests pass 107 existing output-contract checks, 354 full/selected
comparisons, 11 signed omitted-field checks, 1,198 stream/batch comparisons,
17 stream failure checks, 28 asynchronous lifetime checks, and 15 visualization
checks. Aqua passes; JET reports 141 findings against main's unchanged limit of
144. The complete CPU suite passes 81,851 checks with 22 existing skips /
expected-broken checks across 113 files (112 core plus regridding).

On tofu GPU 0 (V100, CUDA runtime 12.6), all 375 output tests pass. Real ERA5
C90 L66 runs with six and 32 tracers produce exactly equal before/after arrays:
196 arrays, 280 comparison/conservation checks. For 32 tracers, cumulative host
allocation falls from 8.855 to 8.052 GB and the median of two warm whole-run
measurements falls from 16.396 to 15.524 s. These are short warm-cache results;
most setup allocation remains. See the
[reproduction scripts, limits, and raw results](../../scripts/benchmarks/results/main_output_v100_20260905/README.md).
