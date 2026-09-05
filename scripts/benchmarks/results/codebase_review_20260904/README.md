# Codebase review measurements — 2026-09-04

These measurements accompany `docs/memos/2026-09-04_review_followup.md`.

- `hotpath_cpu.json`: old strided versus contiguous column diagnostics, plus
  retained full/column snapshot size. Exact diagnostic equality was asserted.
- `gpu_hotpath_results.json`: original versus batched Lin–Rood horizontal update,
  alternating 20 warm samples. Bitwise equality was asserted for air and tracer
  fields. Device: NVIDIA A100-PCIE-40GB on `curry`, `CUDA_VISIBLE_DEVICES=0`.
- `pipeline_cpu.json`, `pipeline_a100.json`: two synthetic meteorological windows,
  binary reader → driven runtime → NetCDF, three warm measured repeats per case.
  All three topologies and one/four tracers are covered. `layers=none` means
  column-only output. These use the OS page cache; they do not measure cold NAS.
- `netcdf_io.json`: benchmark harness `io` cases using real NetCDF capture/write/read.

The two `*_bench.jl` scripts and `linrood_reference.jl` preserve the comparison
implementations used for this review. The reference is an archived benchmark
baseline, not a second production implementation.

Run CPU comparisons from the repository root:

```bash
julia --project=. scripts/benchmarks/results/codebase_review_20260904/hotpath_bench.jl /tmp/hotpath_cpu.json
```

The archived GPU comparison writes `gpu_hotpath_results.json` alongside its
script; copy its script and `linrood_reference.jl` to a scratch directory before
rerunning if you want to retain the recorded measurement. Use the benchmarking
environment and verify that GPU 0 is the A100.

CPU hardware: Intel Xeon Platinum 8462Y+. CPU and GPU microbenchmarks are not
whole-model speedups. Host allocation totals exclude device allocations; retained
snapshot sizes are not peak process RSS or peak GPU memory.
