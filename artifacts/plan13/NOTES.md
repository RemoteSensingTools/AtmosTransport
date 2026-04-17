# Plan 13 Execution Notes — Sync Removal + CFL Unification + Rename

Plan: [docs/plans/13_SYNC_AND_CLEANUP_PLAN.md](../../docs/plans/13_SYNC_AND_CLEANUP_PLAN.md)
Effective overrides agreed in conversation (pre-start):
 1. Sync rule: remove `synchronize(backend)` only if immediately followed
    by another KA kernel launch on the same backend. Keep everywhere else.
 2. LinRood is hands-off except for the Commit 4 textual rename
    (`rm_buf → rm_A`, `m_buf → m_A`). Any sync observation there is
    logged as deferred.
 3. Benchmark tightening: default GPU to `large` (576×288×72 × 10
    tracers), bump `n_steps` (40 medium / 60 large), median ± MAD
    statistics, optional CUDA event-based per-sweep timing.

Started: 2026-04-16
Baseline commit: 6179059 (post-plan-11 `restructure/dry-flux-interface`)
Host: wurst (NVIDIA L40S, F32-only on GPU)

## Decisions made beyond the plan

- **Plan rescoped at Commit 0** (2026-04-16): the GPU baseline capture used
  CUDA-event-based timing (`CUDA.@elapsed` wrapped so the macro only
  materializes on GPU) and revealed that `synchronize(backend)` is
  ~10–12 μs per step on L40S — 0.02–0.4% of total time, not "the dominant
  GPU cost" as plan 13 §1.2 assumed. The sync-removal thesis is falsified.
  Commit 3 is replaced with Commit 3′ (investigation report, no code
  changes). CFL unification (Commit 2), rename cleanup (Commit 4), and
  docs (Commit 5) are still worth doing for maintainability.
- **CPU bench scope:** medium only. Large CPU would take >4 hours
  (60 steps × 6 configs × ~40 s/step) without adding information
  beyond what medium already gives. GPU bench runs both medium and
  large — large shows per-step at ~47 ms, comfortably above noise.

## Deferred observations

- `LinRood.jl:73/709` sync may be redundant post-ping-pong —
  `ws.rm_buf` is an alias of `ws.rm_A`, but buffer is still reused
  across panels. Investigation scoped out of plan 13 per user guidance
  ("LinRood is hands-off"); belongs to a future LinRood-specific
  cleanup plan.
- `AtmosTransportCUDAExt` emits a loading warning on GPU bench startup
  ("Package … is required but does not seem to be installed"). Bench
  still runs correctly because CUDA.jl is loaded directly. Harmless;
  not plan-13 scope.

## Surprises vs. the plan

- **Sync is not the GPU bottleneck.** Direct CUDA-event measurement
  vs. host-side `time_ns()` brackets show Δ host−cuda = 10–12 μs across
  medium and large problem sizes, constant regardless of scheme.
  Plan 13 §1.2 inferred sync cost by subtraction from a guessed
  bandwidth ceiling, which over-counted. Direct event timing is
  authoritative.
- **Kernel arithmetic + d2d copyto dominate GPU time.** At medium F32,
  each step is ~3.1–3.3 ms, essentially all device work. At large F32,
  ~47–49 ms — scales linearly with problem size × tracer count,
  consistent with arithmetic-bound behavior.

## Open questions
(Resolved at Commit 0)

## Test anomalies
(None — 77 pre-existing failures match plan 11's baseline exactly)

## Benchmark results

Methodology: 40/60 timed steps (medium/large), 3 warmup, median ± MAD.
MAD is robust to outliers; median-relative MAD < 1% everywhere → clean signal.

### Baseline (commit 6179059, plan-13 start)

**CPU medium** (288×144×32, 5 tracers, single-thread, 40 steps):

| FT      | Scheme | median(ms) | MAD(ms) |
|---------|--------|-----------:|--------:|
| Float64 | Upwind |    423.204 |   0.706 |
| Float64 | Slopes |   2516.103 |   1.163 |
| Float64 | PPM    |   2273.601 |   1.296 |
| Float32 | Upwind |    379.522 |   0.508 |
| Float32 | Slopes |   2405.955 |   0.672 |
| Float32 | PPM    |   2176.212 |   0.760 |

**GPU medium** (L40S F32, 40 steps, with --events):

| Scheme | host(ms) | cuda(ms) | Δ host−cuda |
|--------|---------:|---------:|------------:|
| Upwind |    3.091 |    3.081 |       0.010 |
| Slopes |    3.145 |    3.134 |       0.011 |
| PPM    |    3.303 |    3.294 |       0.009 |

**GPU large** (L40S F32, 576×288×72, 10 tracers, 60 steps, with --events):

| Scheme | host(ms) | cuda(ms) | Δ host−cuda |
|--------|---------:|---------:|------------:|
| Upwind |   46.181 |   46.170 |       0.011 |
| Slopes |   47.080 |   47.068 |       0.012 |
| PPM    |   49.277 |   49.265 |       0.012 |

### After Commit 2 (CFL unification)
_To be filled_

### After Commit 3′ (sync thesis report)
_No bench re-run — no code changes_

### After Commit 4 (rename)
_To be filled_

### Final (commit 5)
_To be filled_

## Commit log
- Commit 0: NOTES.md + baseline (77 pre-existing test failures captured,
  same as plan-11 baseline) + bench extension (--events, MAD, tighter
  n_steps) + CPU & GPU baselines captured. Premise check of
  sync-removal thesis: FAILED (sync = ~12 μs, not 2-3 ms). Plan
  rescoped in approved plan file.
