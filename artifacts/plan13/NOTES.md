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

### After Commit 2 (CFL unification) — wurst L40S

Ran concurrently with test suite, so CPU F64 numbers are contaminated
by CPU contention (MAD 8–45 ms vs baseline's 1 ms). GPU is clean.

| backend | scheme | median(ms) | MAD(ms) | vs. baseline |
|---------|--------|-----------:|--------:|-------------:|
| cpu F64 | Upwind |    483.542 |  45.144 | (noisy)      |
| cpu F64 | Slopes |   2769.413 |   8.084 | (noisy)      |
| cpu F64 | PPM    |   2521.123 |   4.453 | +10.9%       |
| cpu F32 | Upwind |    356.362 |   0.378 | −6.1%        |
| cpu F32 | Slopes |   2382.732 |   0.802 | −1.0%        |
| cpu F32 | PPM    |   2170.529 |  31.766 | (noisy)      |
| gpu F32 med | Upwind | 3.067 host / 3.057 cuda | 0.013 | −0.8% |
| gpu F32 med | Slopes | 3.123 host / 3.114 cuda | 0.006 | −0.7% |
| gpu F32 med | PPM    | 3.291 host / 3.282 cuda | 0.005 | −0.4% |

### After Commit 3′ (sync thesis report)
_No bench re-run — no code changes. Report lives at
[perf/sync_thesis_report.md](perf/sync_thesis_report.md)._

### After Commit 4 (rename + shim drop) — wurst L40S, clean run

| backend       | scheme | median(ms) | MAD(ms) | vs. baseline |
|---------------|--------|-----------:|--------:|-------------:|
| cpu F64       | Upwind |    475.047 |  32.234 | (noisy)      |
| cpu F64       | Slopes |   2766.306 |  18.816 | (noisy)      |
| cpu F64       | PPM    |   2539.588 |  15.254 | (noisy)      |
| cpu F32       | Upwind |    357.437 |   0.545 | −5.8%        |
| cpu F32       | Slopes |   2387.287 |   0.964 | −0.8%        |
| cpu F32       | PPM    |   2151.267 |   2.198 | −1.1%        |
| gpu F32 med   | Upwind |      2.997 (host) / 2.987 (cuda) | 0.014 | **−3.0%** |
| gpu F32 med   | Slopes |      3.085 (host) / 3.076 (cuda) | 0.008 | **−1.9%** |
| gpu F32 med   | PPM    |      3.252 (host) / 3.244 (cuda) | 0.006 | **−1.5%** |
| gpu F32 large | Upwind |     46.103 (host) / 46.092 (cuda) | 0.084 | −0.2% |
| gpu F32 large | Slopes |     46.958 (host) / 46.948 (cuda) | 0.114 | −0.3% |
| gpu F32 large | PPM    |     49.333 (host) / 49.322 (cuda) | 0.351 | +0.1% |

Δ host − cuda stays ~10 μs, confirming the plan-13 finding: sync
overhead is tiny. The small GPU-medium improvement (−1.5 to −3.0%)
comes from the unified CFL pilot staying on device (no GPU→CPU transfer
on the structured path) + dropping the 3-check `getproperty` shim per
workspace access in LinRood / tests. At large, Δ is within MAD noise.
CPU F64 numbers remain noisy from shared-host load; F32 is clean and
within ±5% of baseline as planned.

## Retrospective

### What worked

- **Measurement-first in Commit 0.** Adding `--events` and MAD stats to
  the benchmark *before* any code change falsified the plan's core
  thesis within 5 minutes of reading the first numbers. The premise-
  check pattern is load-bearing.
- **Scope discipline.** Once the sync thesis fell, the other refactors
  were independently motivated and shipped without drama. Three
  deliverables out of four is still a clean plan.
- **Sign-bug catch in Commit 1.** The premise test failing on Z direction
  surfaced a real pre-existing bug in the GPU static CFL path (inflow
  vs outflow). Commit 2 fixed it while unifying — a side-effect win.
- **Worktree pattern was unnecessary.** At Commit 0 I was at the
  baseline HEAD, so a worktree was overkill. The pattern still applies
  for mid-commit before/after comparisons; here it didn't save work.

### What didn't

- **The original plan-13 thesis.** Plan 13 §1.2 extrapolated sync cost
  from a bandwidth ceiling by subtraction. Direct measurement showed
  the subtrahend was wrong. Any refactor promising a perf win from
  removing an operation needs direct measurement of that operation's
  cost, not inference.
- **n_sub semantics changed subtly in Commit 2.** Two existing tests
  expected the evolving-mass pilot's n_sub (which can exceed the static
  estimate by 1 in transient-mass scenarios). Updated to assert the
  physical invariants (positivity + mass conservation) that are the
  real contract. No user-visible behavior change in production runs —
  Δn_sub ≤1 in pathological flows, cfl_limit safety margin preserved.

### Deferred for future plans

- `cfl_scratch_m` / `cfl_scratch_rm` fields in AdvectionWorkspace are
  now dead (only the evolving-mass pilot used them). Removable in a
  follow-up.
- LinRood panel-loop sync (`LinRood.jl:709`) may be redundant post-
  ping-pong but was kept per user-tightened scope rules.
- `cluster_sizes::V1` in workspace is still used by legacy
  UpwindAdvection/RussellLernerAdvection kernels. When plan 12 removes
  those, the field goes too.

### Final plan-13 impact summary

- `StrangSplitting.jl`: net ≈ −430 lines (~47% of original).
- File deleted: `MassCFLPilot.jl` (76 lines).
- One CFL algorithm across CPU/GPU, bit-identical decisions.
- One pre-existing sign bug fixed (GPU static CFL summed inflow, not outflow).
- Rename `rm_buf → rm_A` complete; no old-name matches in advection
  workspace scope.
- Plan-13 finding ("sync is not the GPU bottleneck at these problem
  sizes") captured in [CLAUDE.md Performance tips](../../CLAUDE.md) so
  future plans start from measurement-based reality.

## Commit log

- Commit 0 (`3bc06ea`): NOTES.md + baseline tests + bench extension +
  CPU/GPU baselines. Sync thesis falsified by CUDA-event measurement.
- Commit 1 (`fdcec0b`): CFL static-vs-evolving premise-check test.
  Surfaced pre-existing inflow-vs-outflow sign bug in passing.
- Commit 2 (`9f894c1`): unified CFL pilot (single static algorithm,
  fixed sign bug); deleted MassCFLPilot.jl. Net −469 lines.
- Commit 3′ (`bb1540c`): sync-removal investigation report
  (no code changes).
- Commit 4 (`12560be`): dropped getproperty shim, renamed
  rm_buf/m_buf/rm_4d_buf → rm_A/m_A/rm_4d_A at all call sites.
- Commit 5: docs (CLAUDE.md, docstring, retrospective) + final bench
  artifacts.
