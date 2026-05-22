# TM5 convection: two-round agent-loop synthesis

Date: 2026-05-22. Three agents (atmospheric expert / evaluator /
programmer), 2 rounds, real benchmarks on the production C180 binary.

This memo is the synthesis. Detailed proposals and per-round reports
live at `/tmp/tm5_round{1,2}_{proposals,ranking,programmer_report}.md`.

## TL;DR (what to do)

- **Production answer remains: persistent LU cache (P6) on top of the
  existing KA collaborative LU.** Measured hit-path = 13.5 ms per panel,
  51× over baseline, 4.7× over the standalone collab-LU. Bit-identical
  hit-vs-miss output. APPROX_PHYSICS — needs an `m_col`-drift sentinel
  for production deployment, but no other physics surface added.
- Two-round optimization attempts on top of P6 (cache only the bottom
  Schur block; layout transposition) **both failed**, for instructive
  reasons (below).
- **The next concrete experiment is "P19" — parallelize the matrix
  build phase across the workgroup threads** — not further cache
  compression. Round-2 data shows the build, not the LU, is the
  remaining bottleneck once the LU is cached.

## What we ran

The atmospheric expert generated 11 proposals (Round 1) and 7 new
proposals + revisions (Round 2). The evaluator filtered both rounds
down to 3 candidates each. The programmer implemented and benchmarked
all 6 picks on the same C180 panel (window=1, panel=1, deep
convection) with Float32, Nt=2, dt=1800, 5 reps on L40S.

### Round 1 (P1..P11 proposed; P4, P6, P8 picked + benched)

| Variant | gpu_min | vs baseline (692 ms) | vs collab-LU (63 ms) | risk |
|---|---:|---:|---:|---|
| baseline (per-thread serial LU) | 691.8 ms | 1.00× | 0.09× | — |
| collab-LU (current reference) | 63.3 ms | 10.94× | 1.00× | BIT_EXACT |
| **P6 cache-HIT** | **13.5 ms** | **51.13×** | **4.69×** | APPROX_PHYSICS |
| P6 cache-miss | 68.9 ms | 10.02× | 0.92× | APPROX_PHYSICS |
| P4 bucketed `Val{Nz}` | 32.0 ms | 21.62× | 1.98× | BIT_EXACT |
| P8 Schur-complement | 59.2 ms | 11.69× | 1.07× | BIT_EXACT |

Winner: **P6 cache-HIT at 13.5 ms**. P8 disappointed because the bottom
Schur block dominates on deep panels; P4 delivered an honest 2×.

### Round 2 (P12..P18 proposed; P13, P17, P8-on-shallow picked + benched)

| Variant | gpu_min | vs P6 hit (13.5 ms) | verdict |
|---|---:|---:|---|
| **P13** (cache bottom Schur block only) | **55.4 ms** | **0.24× — 4× SLOWER** | FAILED |
| P17 (transposed cache layout) | 15.6 ms | 0.86× — 16% slower | FAILED |
| P8 on w12/p3 ("shallow" panel) | 51.2 ms | n/a | FAILED — no shallow panels exist |

All three picks failed. P6 hit at 13.5 ms remains the round-1 winner.

## Why the round-2 ideas failed (the structural lesson)

### P13 — cache only the bottom block

The proposer assumed the bottom-block LU dominated the hit path. **It
doesn't.** The serial matrix-build inside `if t == 1` dominates. P6
skips the build entirely (because the FULL factor is cached); P13 must
rebuild to get `A_TT`, the Schur multiplier `A_BT`, and the populated
top-right block needed for the forward solve. The result: P13 has all
the build cost AND part of the factor cost, while P6 has only the
factor-load cost.

**General lesson**: once you cache a factorization, ANY remaining
work that touches the build is the new bottleneck. Caching a smaller
subset of the factor is a regression unless you can also skip the
build.

### P17 — transposed cache layout

The proposer argued that transposing `(Nz, Nz, Nc, Nc) → (Nc, Nc, Nz, Nz)`
would let adjacent workgroups share cache lines. **But coalescing is a
WITHIN-warp property, not a cross-workgroup property.** In the original
layout, the 32 threads in a warp read locations 4 bytes apart
(coalesced). In the transposed layout, the 32 threads in a warp read
locations 130 KB apart (completely uncoalesced).

**General lesson**: for one-workgroup-per-column kernels, optimize for
within-warp coalescing first.

### P8 on a "shallow" panel

The proposer hypothesised that P8's structural insight (top block ~Hessenberg,
bottom block dense+small) would shine when convection is shallow. **The
production binary has no shallow panels.** Probed four candidate panels
(w1/p1, w12/p3, w18/p5, w6/p1): zero columns with active depth ≤ 25
layers. Subsidence anchors convection to the surface, making `n_B`
essentially constant (38-45 layers per panel) regardless of the
geographic intuition the panel selection was based on.

**General lesson**: histogram statistics aren't a substitute for
per-panel statistics. The whole-binary depth distribution doesn't
imply that any single panel has a fat shallow tail.

## What this tells us about the next round

Round-2 data points unambiguously at the next bottleneck:

1. **P6 hit (13.5 ms)** is dominated by matrix LOAD from global memory
   (`cache_A` is 938 MB/panel; 32 400 reads of 85² floats per substep).
2. **P6 miss (68.9 ms)** is dominated by the SERIAL BUILD inside
   `if t == 1`. The build's `amu[k]` and `amd[k]` recurrences make
   trivial parallelisation hard, but the *inner* `kk` loops on each
   `k` ARE parallelisable.
3. **P13 hit (55.4 ms)** sits between miss and hit because it has
   build + reduced LU. This is the empirical evidence that the build
   is on the critical path whenever ANY rebuild is needed.

Two concrete Round-3 candidates fall out:

- **P19 — parallelize the matrix build phase.** Today thread 1 does
  the whole build serially. The `amu/amd` recurrences serialize the
  outer `k` loop, but the `kk` writes inside each `k` are independent
  (each thread takes a stride). Predicted impact: build phase drops
  ~10× → miss path drops from 69 ms toward 25 ms; hit path is
  unaffected. Composes with P6: smaller P6 miss-cost → better
  amortization.
- **P12 — active-window compression of the P6 cache.** Cache only the
  `(Nz − k_lo + 1)²` corner per column. Storage 938 MB → ~360 MB
  (2.9× less memory traffic on hit). Predicted P6 hit: 13.5 → 10-12 ms.
  Modest, BIT_EXACT-on-hit improvement.

These are independent and can be combined: a properly amortized P6+P12+P19
ladder would put convection at roughly `(25 + 47 × 10) / 48 ≈ 10.3 ms
per substep`, which is **5× under advection's 52 ms per step**. That's
the realistic ceiling implied by the cost model.

## What we measured that's portable

All measured variants use only `KernelAbstractions` primitives
(`@kernel`, `@localmem`, `@synchronize`, `@index`, `@Const`). None use
cuBLAS or any CUDA-specific call. **Everything in this synthesis runs
on Metal in principle.** P6's cache-array allocation is a plain
`CuArray` / `MtlArray` of `Float32` — no backend-specific layout. The
~30 KB shared-memory footprint per workgroup is below Metal's M2 32 KB
threadgroup limit.

## Production readiness checklist for P6

If we ship P6 (the round-1 winner):

1. **`m_col`-drift sentinel**: cache invalidates when relative L∞
   change in `m_col` exceeds a threshold ε. Need an empirical study
   of `m_col` evolution within a 30-min met window to set ε. Without
   this, the APPROX_PHYSICS approximation is not bounded.
2. **Cache-miss path determinism**: the first substep in any window
   always misses; that path must produce bit-identical output to a
   from-scratch build. The current bench confirms this for the
   prototype but the production kernel needs to inherit the same
   property.
3. **48-substep ladder timing**: measure on a real 24-window run
   (not just one substep) to confirm amortized cost ≈ 14 ms. The
   prototype only measured single-substep miss / hit, not the
   ladder.
4. **Mass-conservation validation**: 1-week C180 run, P6 vs collab-LU
   reference. Compare global tracer mass and zonal-mean CO2 profiles.
5. **Adjoint replay**: the cached factor + pivots must replay
   identically in `trans='T'` for plan 19. Same factorisation, same
   pivots, just the back-sub uses the transposed pattern.

## Files on disk

Bench scripts (under `scripts/benchmarks/`):
- `bench_tm5_collab_lu.jl` — current reference (10× over baseline)
- `bench_tm5_p4_bucketed.jl` — Val{Nz} buckets (Round 1)
- `bench_tm5_p6_lu_cache.jl` — **winner** (cache hit-path 13.5 ms)
- `bench_tm5_p8_schur.jl` — Schur split (modest win)
- `bench_tm5_p13_cache_bottom.jl` — bottom-only cache (Round 2, FAILED)
- `bench_tm5_p17_layout.jl` — transposed layout (Round 2, FAILED)
- `_probe_panel_depths.jl` — per-panel depth statistics

Memos and per-round reports:
- `docs/memos/TM5_CONVECTION_PERFORMANCE_MEMO.md` — broader perf memo
- `/tmp/tm5_convection_agents_context.md` — shared context the agents read
- `/tmp/tm5_round{1,2}_proposals.md` — proposer outputs
- `/tmp/tm5_round{1,2}_ranking.md` — evaluator outputs
- `/tmp/tm5_round{1,2}_programmer_report.md` — programmer measurements

## Bottom line

After two iterations of propose-rank-test with real measurements:

- **Winner: P6 LU cache at 13.5 ms hit-path** (4.7× over collab-LU,
  51× over baseline). APPROX_PHYSICS, needs `m_col`-drift sentinel
  for production.
- **Two attempts to improve P6 failed** with instructive root causes:
  build cost is the unbreakable floor for any rebuild path; within-warp
  coalescing matters more than cross-workgroup adjacency.
- **The agreed next experiment is "P19" — parallelize the build phase**,
  which Round 2 data identifies as the dominant cost in any non-P6
  hit path.
