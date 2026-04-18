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

## Deferred observations

(Updated as execution proceeds.)

## Surprises vs. the plan

- **v3 path references need translation.** v3 §4.2 cites
  `src/Utils/{CellState,Tracers,DrivenSimulation}.jl` and
  `src/TransportModel.jl`. This repo has them at
  `src/State/{CellState,Tracers}.jl`, `src/Models/DrivenSimulation.jl`,
  `src/Models/TransportModel.jl`. Per v3 §4.1 ("re-grep at execution
  time; the grep is authoritative about actual scope") this is
  expected — plan was written against a different layout assumption.

## Test anomalies

(Updated as execution proceeds.)

## Template usefulness for plans 15-18

(Filled at Commit 7 retrospective.)
