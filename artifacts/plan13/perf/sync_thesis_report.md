# Sync-Removal Thesis: Measurement Report

**Plan 13 §1.2** proposed removing redundant `synchronize(backend)` calls
from the Strang sweep orchestrator for a **20–40% GPU speedup** (§2.1,
§4.4 Commit 3). The thesis was: "~3 ms per step on L40S, of which only
~0.18 ms was copyto. The remaining ~2.8 ms is dominated by sync
overhead and kernel arithmetic."

Plan 13 Commit 0 introduced CUDA-event-based device timing in
`scripts/benchmarks/bench_strang_sweep.jl --events` and captured post-
plan-11 baselines on wurst (NVIDIA L40S, F32). The measurement
**falsifies the thesis**. This report documents the methodology,
numbers, and implications.

## Methodology

`CUDA.@elapsed` wraps an expression in `CUDA.CuEvent` records. Between
the start and end events, the returned value is **device-side elapsed
time**: the GPU timeline from just before the first enclosed operation
to just after the last one, measured by the device's own clock. If the
GPU goes idle (because the host is blocked on a `synchronize` and
hasn't yet submitted the next kernel), that idle time is included.

We compare two measurements per `strang_split!` call:

- **host(ms):** Julia `time_ns()` deltas bracketing the call, each
  bracketed by `CUDA.synchronize()` (so we wait for all prior device
  work first and for the call to finish). This captures the full
  round-trip wall time: device work + host-side launch overhead +
  Julia dispatch + any blocking on sync primitives.
- **cuda(ms):** `CUDA.@elapsed` of the same call. This captures only
  device-side elapsed time.

Their difference `Δ host − cuda` is everything happening on the host
that does *not* hold up the device: dispatch, argument marshalling,
kernel launch latency, and `synchronize(backend)` return-trip
latency. This is the upper bound on what sync removal can save.

## Results (wurst L40S F32, post-plan-11 HEAD = 6179059)

### Medium (288×144×32, 5 tracers, 40 steps, 3 warmup)

| Scheme | host (ms) | cuda (ms) | Δ host − cuda |
|--------|----------:|----------:|--------------:|
| Upwind |     3.091 |     3.081 |         0.010 |
| Slopes |     3.145 |     3.134 |         0.011 |
| PPM    |     3.303 |     3.294 |         0.009 |

### Large (576×288×72, 10 tracers, 60 steps, 3 warmup)

| Scheme | host (ms) | cuda (ms) | Δ host − cuda |
|--------|----------:|----------:|--------------:|
| Upwind |    46.181 |    46.170 |         0.011 |
| Slopes |    47.080 |    47.068 |         0.012 |
| PPM    |    49.277 |    49.265 |         0.012 |

Both measurements have MAD < 1% of median (tight signal, not noise).

## Finding

Δ host − cuda is **~10–12 μs regardless of problem size**. That's
0.02–0.4% of total per-step wall time. Sync + dispatch together consume
~60–70 μs across six sweep calls per step (10–12 μs × 6), which is
consistent: each `synchronize` on the default stream waits for the
host to observe that the GPU is idle, round-trip ~2 μs.

The ~3 ms (medium) / ~47 ms (large) *is* kernel arithmetic + in-stream
d2d copyto. Removing `synchronize(backend)` cannot reduce that.

## Why the plan inferred a different cost

Plan 13 §1.2 estimated sync cost **by subtraction from a guessed
total**: "~3 ms/step, ~0.18 ms copyto, therefore ~2.8 ms must be sync +
arithmetic." The unknown split between "sync" and "arithmetic" was
assumed to favor sync based on KA documentation and stream-blocking
reasoning. Direct CUDA-event measurement shows the split is
overwhelmingly arithmetic.

The failure mode was: asking "what is sync cost?" via subtraction
instead of direct measurement. The number could have been anywhere
from 0 to 2.8 ms and the plan would still have read plausibly.

## Decision

Plan 13's Commit 3 (sync removal) is retired. `synchronize(backend)`
calls in structured sweeps stay in place. The maximum per-step
savings (~12 μs) is well below Commit 3's 10% ship gate.

The other three plan-13 refactors have independent merit and proceed:

- **Commit 2 (CFL unification)** — correctness/maintainability win
  (shipped: `9f894c1`).
- **Commit 4 (rename + shim drop)** — cleanup win.
- **Commit 5 (docs)** — propagate this finding into `CLAUDE.md` so
  future plans that assume "KA sync is expensive" start from
  measurement-based reality.

## Practical recommendation for future perf plans

Before budgeting speedup from removing an operation, **time the
operation directly** rather than inferring its cost from subtraction.
For GPU: use `CUDA.@elapsed` or nsys. For CPU: use `@elapsed` or
BenchmarkTools. A 30-minute measurement up front would have saved
plan 13 from the premise error.

The extension `scripts/benchmarks/bench_strang_sweep.jl --events`
added in Commit 0 is the right shape for this: report host vs
device time, report their delta, and gate performance plans on that
delta being non-trivial.

## Artifacts

- Baseline measurement: [`before_gpu_medium.log`](before_gpu_medium.log),
  [`before_gpu_large.log`](before_gpu_large.log).
- Bench script: [`../../../scripts/benchmarks/bench_strang_sweep.jl`](../../../scripts/benchmarks/bench_strang_sweep.jl)
  (see `--events` flag and the `gpu_event_time` helper).
- Related commits: Commit 0 (`3bc06ea`).
