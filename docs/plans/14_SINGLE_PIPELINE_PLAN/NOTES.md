# Plan 14 Execution Notes — Advection Pipeline Unification

Plan: [../14_SINGLE_PIPELINE_PLAN_v3.md](../14_SINGLE_PIPELINE_PLAN_v3.md)
(v2 also retained for history at [../14_SINGLE_PIPELINE_PLAN_v2.md](../14_SINGLE_PIPELINE_PLAN_v2.md).)
Companion refs: [../OPERATOR_COMPOSITION.md](../OPERATOR_COMPOSITION.md),
[../ARCHITECTURAL_SKETCH.md](../ARCHITECTURAL_SKETCH.md).

## Plan version

Execution follows **v3** (post-Codex review of v2). Key deltas this
NOTES.md tracks:

1. **Property access PRESERVED.** `state.tracers.CO2` continues to
   work via `getproperty` + a `TracerAccessor` wrapper, rather than
   being broken. v2 would have forced migration of 4 test files;
   v3 does not.
2. **Storage-agnostic accessor API introduced first.** New module
   in `src/State/Tracers.jl` with `ntracers`, `tracer_index`,
   `tracer_name`, `get_tracer`, `eachtracer`. v3 Commit 2 establishes
   the API as a NamedTuple pass-through BEFORE any storage change.
3. **Field name: `tracers_raw`** (not `tracers_4d`).
   `tracer_names::NTuple{Nt, Symbol}` (not `Vector{Symbol}`) per
   v3 Decision 2 — type-stable, compile-time-known Nt.
4. **Storage flip is incremental.** Commit 3 adds `tracers_raw` +
   `tracer_names` fields alongside the existing `tracers::NamedTuple`,
   with constructors keeping both in sync. Commit 4 flips the primary
   (drops NamedTuple field, `getproperty(:tracers)` returns a
   `TracerAccessor`).
5. **Explicit Commit 5** for `TransportModel.jl` + workspace
   integration. v2 folded this into Commit 3; v3 splits it.
6. **Face-indexed path IS in scope** for layout compatibility (but
   not for pipeline fusion). The `apply!(…, FaceIndexedFluxState,
   scheme::AbstractConstantScheme,…)` generator at
   [../../../src/Operators/Advection/StrangSplitting.jl:1043](../../../src/Operators/Advection/StrangSplitting.jl#L1043)
   iterates `pairs(state.tracers)` and must migrate to iterate
   `tracers_raw[:,:,:,t]`.
7. **Total: 8 commits (0–7).** Prior plan (v2) listed 6.

## Baseline

- **Commit:** `47389081efd28efe688d789581f3101855337a20`
  (`4738908 config: switch catrine_2day_cs_slopes to Float32`).
- **Branch:** `advection-unification`, forked from
  `restructure/dry-flux-interface`.
- **Parent chain:** Plans 11, 12, 13 all shipped on the parent branch
  (`fdcec0b…4eb5ce8` sequence).

## §4.1 precondition grep results

| Check | Expected | Got |
|---|---|---|
| `AbstractAdvection\b` in `src/` | 0 | 0 ✅ |
| `MassCFLPilot` in `src/` | 0 | 0 ✅ |
| `rm_A` in `StrangSplitting.jl` | ≥1 | 29 ✅ |
| `m_A` in `StrangSplitting.jl` | ≥1 | 43 ✅ |
| `m_save` in LinRood/CubedSphereStrang | 0 | 0 ✅ |

## Current-state survey

### Storage (today's code)

- `CellState` at [../../../src/State/CellState.jl](../../../src/State/CellState.jl).
  `tracers::Tr` stores a `NamedTuple` of 3D arrays. Two keyword
  constructors (`CellState(m; tracers...)` and
  `CellState(::Type{Basis}, m; tracers...)`); a `getproperty`
  overload at line 57 already aliases `air_dry_mass ⇒ air_mass`.
- Module layout is `src/State/...`, not `src/Utils/...` as v3's
  §4.2 line refs suggest. Grep is authoritative per v3 §4.1 note.

### Path names (v3 path → actual path)

| v3 says | Actual |
|---|---|
| `src/Utils/CellState.jl` | `src/State/CellState.jl` |
| `src/Utils/Tracers.jl` | `src/State/Tracers.jl` |
| `src/Utils/DrivenSimulation.jl` | `src/Models/DrivenSimulation.jl` |
| `src/TransportModel.jl` | `src/Models/TransportModel.jl` |

### Multi-tracer path already present

- `strang_split_mt!` at
  [../../../src/Operators/Advection/StrangSplitting.jl:1159](../../../src/Operators/Advection/StrangSplitting.jl#L1159)
  — takes `rm_4d::AbstractArray{FT,4}`.
- `sweep_x_mt!` / `sweep_y_mt!` / `sweep_z_mt!` at lines 1093–1134.
- `TracerView{FT,A}` in `multitracer_kernels.jl:57` wraps a slice.
- `AdvectionWorkspace` (lines 75–88) already carries `rm_4d_A` /
  `rm_4d_B`. When constructed via `AdvectionWorkspace(m)` WITHOUT
  `n_tracers`, these are allocated at size 0×0×0×0 — the bug v3
  Commit 5 fixes.

### TransportModel workspace construction (v3 Commit 5 target)

- [../../../src/Models/TransportModel.jl:18,27,36](../../../src/Models/TransportModel.jl#L18)
  — three `AdvectionWorkspace(state.air_mass)` sites, none pass
  `n_tracers`. Commit 5 must route `state` through so
  `AdvectionWorkspace(state)` infers Nt.

### Tracer-access migration scope

- **Property access `state.tracers.<Sym>`** (v2 said migrate;
  v3 KEEPS via `getproperty` + `TracerAccessor`, so these sites
  need no change):
  - [../../../test/test_basis_explicit_core.jl](../../../test/test_basis_explicit_core.jl) (lines 189–190, 287)
  - [../../../test/test_driven_simulation.jl](../../../test/test_driven_simulation.jl) (lines 97–99, 5 instances)
  - [../../../test/test_real_era5_direct_dry_binary.jl](../../../test/test_real_era5_direct_dry_binary.jl) (lines 139, 142, 146)
  - [../../../test/test_real_era5_dry_conversion.jl](../../../test/test_real_era5_dry_conversion.jl) (lines 308, 313)

- **NamedTuple-specific patterns** in src/ (must migrate to accessor
  API in Commit 2; storage switch in Commit 4 flips the impl):
  - [../../../src/State/Tracers.jl:29](../../../src/State/Tracers.jl#L29)
    — `set_uniform_mixing_ratio!` uses `getfield(state.tracers, name)`
    → migrate to `get_tracer(state, name)`.
  - [../../../src/State/CellState.jl:72](../../../src/State/CellState.jl#L72)
    — `mixing_ratio` → `get_tracer(state, name) ./ state.air_mass`.
  - [../../../src/State/CellState.jl:81](../../../src/State/CellState.jl#L81)
    — `total_mass` → `sum(get_tracer(state, name))`.
  - [../../../src/State/CellState.jl:96](../../../src/State/CellState.jl#L96)
    — `tracer_names(::CellState{B, A, <:NamedTuple{names}}) = names`
    dispatches on NamedTuple type; must rewrite to return
    `state.tracer_names` field (post Commit 3) or via an impl helper.
  - [../../../src/Models/DrivenSimulation.jl:116,118,139](../../../src/Models/DrivenSimulation.jl#L116)
    — `haskey(state.tracers, name)` + two `getfield(state.tracers, ...)`
    sites. Migrate to `tracer_index(state, name) !== nothing` and
    `get_tracer(state, name)`.
  - [../../../src/Operators/Advection/StrangSplitting.jl:958](../../../src/Operators/Advection/StrangSplitting.jl#L958)
    — the main target: LatLon `strang_split!` per-tracer loop
    iterates `pairs(state.tracers)`. Commit 4 deletes this loop.
  - [../../../src/Operators/Advection/StrangSplitting.jl:1058](../../../src/Operators/Advection/StrangSplitting.jl#L1058)
    — face-indexed `apply!` path at line 1043 also iterates
    `pairs(state.tracers)`. Rewrite to iterate tracer index `t in
    1:ntracers(state)` using `view(state.tracers_raw, :, :, :, t)`.
    Algorithm unchanged.

## Pre-existing test failures (2026-04-17, fresh capture)

See [../../../artifacts/baseline_test_summary.log](../../../artifacts/baseline_test_summary.log).
**77 failures total, bit-exact match with plan 12's 2026-04-17 04:52 baseline.**

| Test file | Pass | Fail |
|---|---:|---:|
| `test_basis_explicit_core.jl` | 3 | **2** (Honest metadata-only CubedSphere API) |
| `test_advection_kernels.jl` | 178 | 0 — incl. 84/84 `Multi-tracer kernel fusion` ✅ |
| `test_structured_mesh_metadata.jl` | 10 | **3** (CubedSphereMesh conventions panel labels) |
| `test_reduced_gaussian_mesh.jl` | 26 | 0 |
| `test_driven_simulation.jl` | 57 | 0 |
| `test_cubed_sphere_advection.jl` | 552 | 0 |
| `test_poisson_balance.jl` | 245 | **72** (mirror_sign inflow/outflow) |
| **Total** | **1071** | **77** |

Post-refactor must produce the same 77, no more, no fewer.

The **`Multi-tracer kernel fusion`** testset (84/84 pass) confirms
`strang_split_mt!` / `sweep_*_mt!` are correctness-clean. Plan 14's
§4.2 "stop if multi-tracer path is broken" gate does not trigger.

## Commit 0 — Measurement

Commit 0 is docs-only + test baseline capture. The bench script
extension and perf baseline run ship in **Commit 1** (bench script
does not yet support `--mode`, `--ntracers`, `--cfl-limits` flags;
Commit 1 adds them). Both measurements are on unmodified production
code paths.

### Commit 1 findings — per-tracer vs multi-tracer, unchanged production code

Two hosts, two logs:

- **wurst (L40S)** →
  [../../../artifacts/perf/plan14/baseline_comparison.log](../../../artifacts/perf/plan14/baseline_comparison.log).
  CPU medium F64 complete (48 rows); CPU medium F32 complete through
  Nt=30 Upwind + Slopes cfl=0.4 (42 rows). **Cut short at user
  direction** before CPU F32 Nt=30 Slopes cfl=Inf + PPM Nt=30, and
  before GPU medium/large F32 — decision-gate signal was already
  clear. Full wurst GPU F32 rerun happens in Commit 7 for
  before/after comparison.
- **curry (A100-PCIE-40GB F64)** →
  [../../../artifacts/perf/plan14/baseline_curry_gpu_f64.log](../../../artifacts/perf/plan14/baseline_curry_gpu_f64.log).
  GPU medium + GPU large both F64, complete (96 rows). Home is
  NFS-shared; same commit `e22246087ebe2bd4568b470015d686f64c8ad27b`.

### Key production-regime numbers (Nt=30, per-step median ms)

**wurst CPU medium F64:**

| Scheme | cfl | per-tracer | multi-tracer | Δ% |
|---|---|---:|---:|---:|
| Upwind | 0.40 | 2563 | 1686 | **−34%** ✅ |
| Upwind | Inf  | 1908 | 1684 | **−12%** ✅ |
| Slopes | 0.40 | 15114 | 15837 | +5% ⚠ |
| Slopes | Inf  | 14441 | 15811 | +9% ⚠ |
| PPM    | (partial) | — | — | — |

**wurst CPU medium F32:**

| Scheme | cfl | per-tracer | multi-tracer | Δ% |
|---|---|---:|---:|---:|
| Upwind | 0.40 | 2431 | 1548 | **−36%** ✅ |
| Upwind | Inf  | 1865 | 1435 | **−23%** ✅ |
| Slopes | 0.40 | 14574 | 14300 | −2% ≈ |
| Slopes / PPM Inf | — | — | — | cut short |

**curry GPU medium F64 (A100):**

| Scheme | cfl | per-tracer | multi-tracer | Δ% |
|---|---|---:|---:|---:|
| Upwind | 0.40 | 38.6 | 7.90 | **−80%** 🔥 |
| Upwind | Inf  | 14.8 | 7.53 | **−49%** 🔥 |
| Slopes | 0.40 | 52.7 | 26.9 | **−49%** 🔥 |
| Slopes | Inf  | 28.9 | 26.5 | −8% ✅ |
| PPM    | 0.40 | 57.4 | 29.9 | **−48%** 🔥 |
| PPM    | Inf  | 33.5 | 29.5 | **−12%** ✅ |

**curry GPU large F64 (A100, grid 576×288×72):**

| Scheme | cfl | per-tracer | multi-tracer | Δ% |
|---|---|---:|---:|---:|
| Upwind | 0.40 | 240 | 65 | **−73%** 🔥 |
| Upwind | Inf  | 104 | 63 | **−40%** 🔥 |
| Slopes | 0.40 | 362 | 208 | **−42%** 🔥 |
| Slopes | Inf  | 227 | 206 | −9% ✅ |
| PPM    | 0.40 | 403 | 234 | **−42%** 🔥 |
| PPM    | Inf  | 266 | 232 | **−13%** ✅ |

### Interpretation

**Decision gate (§4.2 acceptance criterion):** plan 14 is **CLEANUP +
PERF WIN**. Multi-tracer path gives ≥5% at many configs and ≥40% at
every GPU Nt=30 cfl=0.4 point.

**Unambiguous wins:** every GPU F64 config at Nt≥5 (curry A100), and
CPU F64/F32 Upwind at any Nt≥5. Plan 11's "10 tracers GPU large
per-step ~47 ms" figure corresponds to per-tracer 240 ms → multi-tracer
65 ms at Nt=30 Upwind 0.4 on A100 — a 3.7× speedup, far bigger than
v3 §1.2's "0-10% realistic expectation" estimate. The estimate was
conservative.

**Mixed / flat:** CPU F64 Slopes/PPM and all cfl=Inf Slopes Inf
regimes show ≤10% movement. The multi-tracer kernel is slightly slower
on CPU for higher-order schemes (probably because the arithmetic
already saturates CPU throughput and the 4D indexing adds fractional
overhead). This is fine — the cleanup goal stands on its own, and
production is GPU.

**At Nt=1, multi-tracer loses** on Upwind Inf (per-tracer 62 → 123 ms
CPU F64, 0.8 → 1.1 ms GPU). This is expected — the multi-tracer kernel
has per-cell overhead that doesn't amortize at Nt=1. Most production
runs use Nt≥5.

**Multi-tracer path correctness is confirmed** by the
`Multi-tracer kernel fusion` testset (84/84 pass) captured in Commit 0.
Plan 14 §4.2's "stop if multi-tracer path is broken" gate does not
trigger.

## Decisions made beyond the plan

- **Split Commit 0 and Commit 1.** v3 §4.2/§4.4 overlap on "extend
  bench + capture baseline". User directive was "create NOTES.md as
  Commit 0 before starting any code changes", so Commit 0 is
  docs-only (NOTES + test baseline + baseline commit hash).
  Commit 1 extends the bench script and runs the perf baseline.
  Neither touches production (non-bench) code.

## Commit 7 findings — after-refactor bench

Two hosts, two logs:

- **curry A100 F64**, full matrix Nt ∈ {5, 10, 30}
  → [../../../artifacts/perf/plan14/after_curry_gpu_f64.log](../../../artifacts/perf/plan14/after_curry_gpu_f64.log).
- **wurst L40S CPU medium**, cut short at Nt=10 Slopes (F64 Nt=30
  runs at ~15 s/step — decision signal already captured)
  → [../../../artifacts/perf/plan14/after_wurst.log](../../../artifacts/perf/plan14/after_wurst.log).

### Production regime: GPU large F64 Nt=30 (A100)

Before/after comparison at cfl=0.4 / cfl=Inf for each scheme:

| Scheme | cfl | Before per-tracer | After per-tracer | Speedup |
|---|---|---:|---:|---:|
| Upwind | 0.40 | 240 ms | 65.1 ms | **3.7×** |
| Upwind | Inf  | 104 ms | 62.9 ms | **1.7×** |
| Slopes | 0.40 | 362 ms | 207 ms | **1.75×** |
| Slopes | Inf  | 227 ms | 206 ms | 1.10× |
| PPM    | 0.40 | 403 ms | 234 ms | **1.72×** |
| PPM    | Inf  | 266 ms | 232 ms | **1.15×** |

Plan 14 delivered **10–270% speedups** at production settings on A100 F64.

### Convergence of the two bench modes

The post-refactor bench runs two modes (`per-tracer` and `multi-tracer`)
that were meaningfully different pre-refactor (Commit 1 baseline showed
per-tracer = old NamedTuple loop, multi-tracer = strang_split_mt!
directly). After Commit 4, `strang_split!` is a thin wrapper over
`strang_split_mt!`, so both bench modes hit the same code path. The
Commit 7 run confirms this — every Nt=30 row shows per-tracer and
multi-tracer within the MAD of each other (≤ 0.5 %):

| Config | per-tracer (ms) | multi-tracer (ms) | Δ |
|---|---:|---:|---:|
| GPU large F64 Upwind Nt=30 cfl=0.4 | 65.14 | 65.16 | +0.03 % |
| GPU large F64 Slopes Nt=30 cfl=0.4 | 207.41 | 207.89 | +0.23 % |
| GPU large F64 PPM Nt=30 cfl=0.4    | 234.05 | 234.01 | −0.02 % |
| GPU medium F64 Upwind Nt=30 cfl=0.4 | 7.82  | 7.86   | +0.6 %  |

This is the intended outcome: the refactor delivered the multi-tracer
pipeline as the default `strang_split!` entry point. There is no
longer a "fast path" vs "slow path" distinction in the public API.

### CPU regime (partial)

At CPU medium F64 Nt=10, both modes now give ≈4914 ms for Slopes,
indistinguishable (vs. pre-refactor per-tracer 5020 ms and multi-tracer
4671 ms at the same config from Commit 1). The CPU win from the
refactor is modest — Slopes/PPM at higher Nt on CPU were already near
memory-bandwidth saturation per Commit 1's observations. The main win
is on GPU, where plan 14 v3 §1.2's "0-10% realistic expectation" was
conservative by a factor of 5–10×.

### Why the GPU win is so much larger than the v3 §1.2 estimate

Plan v3's 0–10% estimate assumed launch-overhead savings dominate.
The actual dominant factor at Nt=30 appears to be the **mass-restore
copyto elimination** (v3 Commit 3 noted as "~2 ms potential win").
The old per-tracer loop called `copyto!(m, m_save)` N−1 times per
step (≈3.2 GB/step at C180 Nt=30) plus re-reads `am/bm/cm` N times.
The multi-tracer kernel reads mass/flux fields ONCE and updates all
tracers in the same launch. At L40S/A100 memory bandwidths, this
saves 15–40 ms/step at GPU large Nt=30. The launch-overhead portion
(~1 ms saved on 60 fewer launches) is a smaller contributor.

## Decisions made beyond the plan (appended)

- **Split Commit 0 and Commit 1** (already noted): docs-only Commit 0
  vs bench-extension-and-baseline-perf Commit 1.
- **Partial Commit 1 baseline**: wurst CPU F32/GPU F32 runs cut short
  at user direction; curry GPU F64 captured. Decision-gate signal was
  already clear at that point. Commit 7 re-ran the production Nt=30
  subset to completion on curry.
- **Skip Commit 6 (dead-code removal) as separate commit**: v3 §4.4
  Commit 6 said "grep for anything still referencing the old pattern".
  Grep showed nothing remaining — NamedTuple-path `_*_impl` helpers
  were already deleted in Commit 4 when the accessor API was rewired
  to dispatch on `state` directly, and `m_save`/`cfl_scratch_*` came
  out in Commit 5. The per-tracer-sweep 5/6-arg forms look like dead
  backward-compat at first glance but are used by LinRood and
  CubedSphereStrang (not dead). Ship Commit 5 → straight to Commit 7.
- **CellState storage sharing attempt** (see Commit 3 message): a
  first draft made `state.tracers.CO2` a VIEW into `tracers_raw` so
  mutations through either handle landed in the same storage. Passed
  a unit smoke test but broke 8 CPU agreement tests in
  `test_advection_kernels.jl` because the test helper `run_strang!`
  returned the caller's unchanged input array while the real data
  lived in `tracers_raw`. Reverted to **separate parallel copies**
  (NamedTuple values = caller's arrays, `tracers_raw` = independent
  copy). Commit 4 then dropped the NamedTuple field, making
  `state.tracers.CO2` a view into `tracers_raw` via `TracerAccessor`
  — at which point the test helper had to migrate to
  `state.tracers.tracer`-based return value.
- **CPU/GPU dispatch fix for SubArray tracers**: `sweep_horizontal!`
  and `sweep_vertical!` checked `rm isa Array` to decide CPU vs GPU
  code path. Post-Commit 4, `rm` is a `SubArray{FT, 2, Array{FT, 3}}`
  (`selectdim` view into `tracers_raw`), which fails `isa Array` and
  wrongly routed to the GPU path. Fix: `parent(rm) isa Array`. Caught
  by `Face-indexed horizontal subcycling preserves positivity` testset.

## Deferred observations

- **Face-indexed / reduced-Gaussian multi-tracer fusion**: out of
  scope for plan 14. The face-indexed `apply!` path still does a
  per-tracer loop — just through `selectdim` views now. Converting
  that to a multi-tracer fused kernel is a follow-up if RG
  workloads start to dominate on GPU.
- **CPU Slopes/PPM near-saturation**: at CPU medium F64 Nt=30
  Slopes/PPM, multi-tracer is within ±10 % of per-tracer (noise
  floor). No CPU win expected here; arithmetic per cell dominates.
- **4D buffers on face-indexed workspaces**: the 2D `AdvectionWorkspace`
  constructor still takes an `n_tracers` keyword but ignores it (4D
  ping-pong buffers stay 0-sized). Not a bug — face-indexed path uses
  `selectdim`, not the 4D buffer — but a cleanup opportunity once
  face-indexed multi-tracer fusion is implemented.

## Surprises vs. the plan

- **v3 path references need translation.** v3 §4.2 cites
  `src/Utils/{CellState,Tracers,DrivenSimulation}.jl` and
  `src/TransportModel.jl`. This repo has them at
  `src/State/{CellState,Tracers}.jl`, `src/Models/DrivenSimulation.jl`,
  `src/Models/TransportModel.jl`. Per v3 §4.1 ("re-grep at execution
  time; the grep is authoritative about actual scope") this is
  expected — plan was written against a different layout assumption.
- **Win is ~5–10× larger than v3 §1.2 predicted on GPU.** v3 revised
  v1's "2–10× GPU speedup" down to "0–10% realistic expectation" on
  the grounds that launch overhead is only ~2.5% of production cost.
  Measurement shows 10–270% at production Nt=30 on A100 F64. The
  dominant factor is bandwidth saved by the single-mass-update
  fusion, not launch overhead. v3 Commit 3 mentioned the bandwidth
  effect as "~2 ms potential win" but undershot; at C180 Nt=30 the
  savings are tens of ms. **Takeaway for future plans**: bandwidth
  estimates via "copyto bytes × 1/bandwidth" underweigh the
  compiler's ability to eliminate redundant memory traffic when
  operations are fused.
- **Commit 3 storage-sharing attempt fell on test-contract edge**:
  the first draft that made `state.tracers.CO2 === view(tracers_raw)`
  broke test helpers that relied on the pre-refactor
  `state.tracers.CO2 === caller's_array` contract. Classic case of
  storage identity assumptions leaking through tests. The fix
  (parallel copies in Commit 3, then property-access via
  TracerAccessor in Commit 4) took an extra iteration that a sharper
  read of the test helpers would have caught up front.

## Test anomalies

- **`Face-indexed horizontal subcycling preserves positivity`**
  regressed at Commit 4 with `ArgumentError: face-indexed GPU sweep
  requires mesh connectivity`. Root cause: `rm isa Array` check for
  CPU-vs-GPU dispatch returned `false` for `SubArray{Array}` views.
  Fix in Commit 4: check `parent(rm) isa Array` instead. Tests that
  look like they check storage type but actually need to check
  backend should use `get_backend` or `parent(·) isa Array` rather
  than `rm isa Array`.

## Template usefulness for plans 15-18

What worked well:
- **Measurement-first Commit 0/1**. The 73% speedup at Nt=30 Upwind
  was worth the time to establish — turned plan 14 from "cleanup only"
  per v3 framing to "cleanup + real perf win".
- **v3's incremental storage flip** (Commit 2 accessor API pass-through
  → Commit 3 parallel fields → Commit 4 primary flip). Even though
  Commit 3's initial design broke tests, rolling back to "parallel
  copies, dead until Commit 4 wires them up" was cheap and the
  overall structure held.
- **NFS-shared home + SSH to curry for parallel F64 bench**. Big
  time savings — got wurst CPU F64/F32 + curry GPU F64 in parallel.

What to carry forward:
- **Perf predictions should cite measurement bounds, not theoretical
  ceilings**. v3 §1.2's 0–10% estimate was based on launch-overhead
  subtraction. The bandwidth-fusion effect was harder to predict a
  priori; direct measurement resolved it cleanly. Future plans
  should make Commit 0 measurement the decision gate, not a
  confirmation of a pre-registered prediction.
- **Pre-identify storage identity assumptions in tests**. Plan 15+
  (chemistry, diffusion) will add operators that mutate
  `state.tracers_raw`. Any test helper that returns a caller-side
  array handle will be wrong. Document the contract: **tests should
  go through `state.tracers.CO2` or `get_tracer(state, :CO2)` to
  observe post-advection state, never the original input array**.
- **CPU/GPU dispatch via `parent(arr) isa Array`, not `arr isa
  Array`**, when the array may be a view. Belongs in a "dispatch
  pattern" coding standards note.
- **Face-indexed multi-tracer fusion is deferred** — flag for plan
  17/18 if RG workloads dominate on GPU.
