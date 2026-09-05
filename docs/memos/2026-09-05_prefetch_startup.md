# Read the first prefetch window once

`DrivenSimulation` used to call `_load_window(driver, start_window)` twice
when GPU prefetch was active, solely to obtain two independent device buffers.
A counted-driver V100 regression observed `[1, 1, 2]`: two initial-window loads
followed by the actual next-window prefetch.

The constructor now loads once and adapts the host payload separately into
current and spare device buffers. Drivers that already return device arrays
use an explicit deep copy for the spare buffer, since adapting a device array
to its own backend may return an alias. CPU, disabled-prefetch, and single-window
paths retain their existing single-buffer behavior. This removes one redundant
load/decode per prefetching simulation; other runner setup and input inspection
can still read meteorology independently.

## Validation

- Before the change, the new V100 startup test passed 88 checks and failed the
  read-count check. Afterward, all 140 checks pass, including an added custom
  device-array driver case. Every humidity, flux-delta and TM5 forcing array is
  checked; mutating all spare arrays leaves active forcing unchanged.
- CS multifile GPU physics: 75 checks pass for CMFMC, CMFMC-derived matrix,
  and TM5 convection. Comparison against the earlier saved workspace-reuse
  results passes 37 checks: 23 of 36 arrays are bitwise equal, and the worst
  relative difference is 1.736e-7, within the existing four-epsilon Float32
  bound. Whole-runtime results are not claimed to be bitwise identical.
- CPU startup: 57 checks on Julia 1.10.12 and 1.12.6.
- Clean Julia 1.12.6 export: existing driven-simulation checks, timer/resource
  checks, Aqua (10), and JET (180 reports against the unchanged 181 threshold).

## End-to-end V100 measurement

The existing `benchmarking/run_pipeline_benchmarks.jl` benchmark times
reader setup, transport, snapshot capture and actual NetCDF writes. Public
reader checks run outside the measured interval.
C48, 40 levels, one tracer, three files/six windows, Float32, five warmed samples:

| Output | Median before | Median after | Host allocations before | Host allocations after |
| --- | ---: | ---: | ---: | ---: |
| Full layers | 0.625636 s | 0.566284 s | 751.58 MB | 695.15 MB |
| Column only | 0.219199 s | 0.182605 s | 270.70 MB | 214.26 MB |

Both cases eliminate about 56.4 MB of cumulative host allocations. Column-only
wall time falls 16.7%; full-layer samples are more variable (before range
0.419–0.677 s, after 0.439–0.700 s), so the median difference is not a universal
speedup claim. All output times and tracer column means are checked by the
benchmark. Output bytes are unchanged. These are synthetic, warm-cache results,
not cold-NAS throughput or peak memory measurements. Device buffers still have
two copies; the change removes the redundant host load and its allocations.

Before: `1ff412e5`. After: `e1f61e53` plus the single-load constructor change.
Both use Julia 1.12.6, CUDA.jl 5.11.3, CUDA runtime 12.6, and tofu's V100 GPU 0
(`GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`). No other GPU was used.
Raw samples and startup test output are in
[`prefetch_startup_v100_20260905`](../../scripts/benchmarks/results/prefetch_startup_v100_20260905/).

Reproduce from the repository root with a V100-compatible CUDA environment:

```sh
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 \
ATMOSTR_BENCH_GPU_NAME=V100 ATMOSTR_BENCH_NC=48 ATMOSTR_BENCH_NZ=40 \
ATMOSTR_BENCH_REPEATS=5 ATMOSTR_BENCH_FILES=3 \
ATMOSTR_BENCH_TOPOLOGIES=cs ATMOSTR_BENCH_TRACERS=1 \
julia --project=. benchmarking/run_pipeline_benchmarks.jl cuda prefetch.json
```

The runtime walkthrough now reflects current function signatures, window
ownership, optional prefetch, actual convection forcing fields, and the
separate transport/window physics cadence. Stale line-number links and a
repeated topology link were removed; every local link resolves.
