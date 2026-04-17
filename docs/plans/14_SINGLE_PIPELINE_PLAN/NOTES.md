# Plan 14 Execution Notes — Advection Pipeline Unification

Plan: [../14_SINGLE_PIPELINE_PLAN_v2.md](../14_SINGLE_PIPELINE_PLAN_v2.md)
Companion refs: [../OPERATOR_COMPOSITION.md](../OPERATOR_COMPOSITION.md),
[../ARCHITECTURAL_SKETCH.md](../ARCHITECTURAL_SKETCH.md).

## Baseline

- **Commit:** `47389081efd28efe688d789581f3101855337a20`
  (`4738908 config: switch catrine_2day_cs_slopes to Float32`).
- **Branch:** `advection-unification`, forked from
  `restructure/dry-flux-interface`.
- **Parent chain:** Plans 11, 12, 13 all shipped on the parent branch
  (`fdcec0b…4eb5ce8` sequence).
- **Pre-existing test failures (fresh capture, 2026-04-17):**
  see [../../../artifacts/baseline_test_summary.log](../../../artifacts/baseline_test_summary.log).
  **77 failures total, bit-exact match with plan 12's 2026-04-17 04:52 baseline:**

  | Test file | Pass | Fail | Notes |
  |---|---:|---:|---|
  | `test_basis_explicit_core.jl` | 3 | **2** | `Honest metadata-only CubedSphere API` (lines 212-215) |
  | `test_advection_kernels.jl` | 178 | 0 | incl. `Multi-tracer kernel fusion` (84/84 pass) ✅ |
  | `test_structured_mesh_metadata.jl` | 10 | **3** | `CubedSphereMesh conventions` panel labels |
  | `test_reduced_gaussian_mesh.jl` | 26 | 0 | |
  | `test_driven_simulation.jl` | 57 | 0 | |
  | `test_cubed_sphere_advection.jl` | 552 | 0 | |
  | `test_poisson_balance.jl` | 245 | **72** | `mirror_sign: inflow +1 outflow -1` (line 207) |
  | **Total** | **1071** | **77** | |

  All 77 are pre-existing and **NOT** caused by plan 14 work. Post-refactor
  must produce the same 77, no more, no fewer.

  Critical signal from Commit 0 baseline: **`Multi-tracer kernel fusion`**
  testset (84/84 pass) confirms `strang_split_mt!` / `sweep_*_mt!` are
  correctness-clean. Plan 14's "stop if multi-tracer path is broken" gate
  (§4.2 acceptance criterion 4) does not trigger.

## §4.1 precondition grep results (freshly verified)

| Check | Expected | Got |
|---|---|---|
| `AbstractAdvection\b` in `src/` | 0 | 0 ✅ |
| `MassCFLPilot` in `src/` | 0 | 0 ✅ |
| `rm_A` in `StrangSplitting.jl` | ≥1 | 29 ✅ |
| `m_A` in `StrangSplitting.jl` | ≥1 | 43 ✅ |
| `m_save` in LinRood/CubedSphereStrang | 0 | 0 ✅ |

## Current-state survey (from precondition exploration)

- `CellState` is at [../../../src/State/CellState.jl](../../../src/State/CellState.jl).
  `tracers::Tr` stores a `NamedTuple` of 3D arrays today; two keyword
  constructors (`CellState(m; tracers...)` and
  `CellState(::Type{Basis}, m; tracers...)`).
- `strang_split!` (LatLon) lives in
  [../../../src/Operators/Advection/StrangSplitting.jl](../../../src/Operators/Advection/StrangSplitting.jl).
  Per-tracer loop: lines 958–999 with `copyto!(m, m_save)` restore.
- Multi-tracer path already exists alongside:
  - `strang_split_mt!` (lines 1159–1206) — takes `rm_4d::AbstractArray{FT,4}`.
  - `sweep_x_mt!` / `sweep_y_mt!` / `sweep_z_mt!` (lines 1093–1134).
  - `TracerView{FT,A}` in `multitracer_kernels.jl:57` wraps a slice of
    a 4D buffer as a 3D-index target.
- `AdvectionWorkspace` (lines 75–88) already carries `rm_4d_A` / `rm_4d_B`
  ping-pong buffers. Dead pre-plan-13 fields present:
  `cfl_scratch_m`, `cfl_scratch_rm`. `m_save` is referenced only in
  `strang_split!` (lines 949–951, 960). Commit 3 drops these.
- `get_tracer(state, name)` does not exist yet — Commit 2 introduces it.
- Property-access migration scope (`state.tracers.<Sym>`, Commit 3):
  - [../../../test/test_basis_explicit_core.jl](../../../test/test_basis_explicit_core.jl) (lines 189–190, 287)
  - [../../../test/test_driven_simulation.jl](../../../test/test_driven_simulation.jl) (lines 97–99, 5 instances)
  - [../../../test/test_real_era5_direct_dry_binary.jl](../../../test/test_real_era5_direct_dry_binary.jl) (lines 139, 142, 146)
  - [../../../test/test_real_era5_dry_conversion.jl](../../../test/test_real_era5_dry_conversion.jl) (lines 308, 313)
- Additional `src/` migration sites discovered in Commit 0 (must be
  handled in Commit 3):
  - [../../../src/State/Tracers.jl](../../../src/State/Tracers.jl)`:29`
    — `set_uniform_mixing_ratio!` uses `getfield(state.tracers, name)`.
  - [../../../src/State/CellState.jl](../../../src/State/CellState.jl)`:72,81,96`
    — `mixing_ratio`, `total_mass`, `tracer_names` (the last dispatches
    on `<:NamedTuple{names}` — must rewrite).
  - [../../../src/Models/DrivenSimulation.jl](../../../src/Models/DrivenSimulation.jl)`:116,118,139`
    — `haskey(state.tracers, ...)` and two `getfield(state.tracers, ...)`.
  - [../../../src/Operators/Advection/StrangSplitting.jl](../../../src/Operators/Advection/StrangSplitting.jl)`:1058`
    — the **face-indexed** `apply!` path also iterates `pairs(state.tracers)`.
    Plan §3.1 lists face-indexed paths as out-of-scope for pipeline
    fusion, but the data-layout flip is mandatory. Fix: rewrite loop to
    iterate `t in 1:size(state.tracers, 4)` and use
    `view(state.tracers, :, :, :, t)` instead of `pairs(...)`. No
    performance fusion; just layout compatibility.
  - `strang_split!` LatLon loop at line 958 (the main target).
- Keyword-form callers (preserved unchanged, internal representation
  flips) in 9 test files, all constructing via
  `CellState(m; CO2=..., CH4=...)`.
- LinRood and CubedSphereStrang do NOT reference `m_save` — plan 14
  does not touch them.

## Commit 0 — Measurement

**Scheduled for Commit 1** (this branch): the current bench script at
[../../../scripts/benchmarks/bench_strang_sweep.jl](../../../scripts/benchmarks/bench_strang_sweep.jl)
only accepts positional `size`/`backend` + `--events`; it does not yet
support `--mode=per-tracer,multi-tracer`, `--ntracers=…`,
`--cfl-limits=…`, `--schemes=…`, or `--dtype=…`. Commit 0 (this commit)
captures the test baseline and writes the NOTES.md skeleton without
code changes. Commit 1 extends the bench script and captures the
per-tracer vs multi-tracer perf comparison into
`artifacts/perf/plan14/baseline_comparison.log`.

This keeps the "no code before baseline perf capture" discipline intact
— the perf baseline is captured from the EXTENDED bench against the
UNCHANGED production path (`strang_split!` + per-tracer loop) vs the
UNCHANGED multi-tracer path (`strang_split_mt!`). Both paths are
current production code at the Commit-1 SHA; the refactor (Commit 3)
is still downstream.

Findings (to be populated after Commit 1 runs):

- CPU medium F64 {Upwind, Slopes, PPM} × {per-tracer, multi-tracer} × Nt
- CPU medium F32 — same grid
- GPU medium F32 — same grid
- GPU large F32 — same grid
- For each: cfl_limit ∈ {0.4, Inf}

Acceptance gates:
- ≥5% multi-tracer win anywhere → `CLEANUP + PERF`. Proceed.
- <5% everywhere → `CLEANUP ONLY`. Proceed, reframe.
- Multi-tracer slower anywhere → STOP; fix multi-tracer correctness.

## Decisions made beyond the plan

(Updated as execution proceeds.)

- **Split Commit 0 and Commit 1.** Plan §4.2 and §4.4 overlap on
  "extend bench + capture baseline". User directive was "create NOTES.md
  as Commit 0 before starting any code changes", so Commit 0 is
  docs-only (NOTES + artifacts/baseline_test_summary.log + baseline
  commit hash). Commit 1 is the bench extension + baseline perf run.
  Neither touches production code.

## Deferred observations

(Updated as execution proceeds.)

## Surprises vs. the plan

(Updated as execution proceeds.)

## Test anomalies

(Updated as execution proceeds.)

## Template usefulness for plans 15-18

(Filled at Commit 5 retrospective.)
