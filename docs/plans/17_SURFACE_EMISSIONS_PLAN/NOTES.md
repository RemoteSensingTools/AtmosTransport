# Plan 17 Execution Notes — Surface Emissions Operator

**Plan:** [17_SURFACE_EMISSIONS_PLAN.md](../17_SURFACE_EMISSIONS_PLAN.md) (v1)
— builds on plan 16b's palindrome-centered `V` integration to add
`S` at palindrome center (`X Y Z V(dt/2) S V(dt/2) Z Y X`). Ships
`StepwiseField{FT, N}` for piecewise-constant monthly/annual inventories.
Resolves plan-15's `DrivenSimulation` chemistry-ordering workaround.

## Baseline

- **Parent commit:** `6bbf512` (plan 16b Commit 7, tip of `vertical-diffusion`).
- **Branch:** `surface-emissions` created from `vertical-diffusion`
  HEAD per CLAUDE_additions.md §"Branch hygiene" stack-branches rule.
- **Pre-existing test failures:** 77, stable since plan 12
  (`test_basis_explicit_core: 2`, `test_structured_mesh_metadata: 3`,
  `test_poisson_balance: 72`). Plan 17 preserves this count.
- **Fast-path baseline** captured in
  [../../../artifacts/plan17/baseline_test_summary.log](../../../artifacts/plan17/baseline_test_summary.log).

## Pre-Commit-0 survey findings

Contrary to plan doc §4.1 assumption of "minimal existing emissions
code", an active `SurfaceFluxSource` infrastructure exists. Full survey
in [../../../artifacts/plan17/existing_emissions_survey.txt](../../../artifacts/plan17/existing_emissions_survey.txt).
Summary:

1. **Primary path** — [SurfaceFluxSource](../../../src/Models/DrivenSimulation.jl):
   `cell_mass_rate::RateT` in **kg/s per cell**, applied at `rm[:,:,Nz]`
   (k=Nz surface). Called inline from `DrivenSimulation.step!` between
   transport and chemistry.
2. **Secondary path (dead)** — [`_inject_source_kernel!`](../../../src/Kernels/CellKernels.jl#L43):
   exported but uncalled in `src/`. Per-area `kg/m²/s × areas[j] × dt`
   convention. Only `scripts_legacy/` uses a similarly-named function.
   Plan 17 does NOT consume or remove it; a future cleanup plan can.
3. **`StepwiseField` is genuinely new** — only hint is a comment at
   [PreComputedKzField.jl:13](../../../src/State/Fields/PreComputedKzField.jl#L13)
   anticipating this type.

[StepwiseField survey](../../../artifacts/plan17/existing_stepwise_survey.txt).

## Deviations from plan doc §4.4

Three scope revisions recorded up front, resolved with user on
2026-04-19 (plan execution file [~/.claude/plans/you-can-compact-context-mutable-token.md](file:///home/cfranken/.claude/plans/you-can-compact-context-mutable-token.md)):

1. **Commit 3 MIGRATES existing `SurfaceFluxSource`** into
   `src/Operators/SurfaceFlux/` rather than building greenfield.
   Re-export from original location for backward compat for one
   release. `PerTracerFluxMap` is `NTuple{N, SurfaceFluxSource}`-backed,
   matching existing `sim.surface_sources::Tuple` storage — plan doc
   recommended Dict, but the existing NTuple is simpler and type-stable.

2. **`k = Nz` = surface convention kept** (plan doc Decision 4 had
   `k = 1` which would silently emit into the top model layer). Matches
   existing [DrivenSimulation.jl:126](../../../src/Models/DrivenSimulation.jl#L126)
   and all LatLon callers. No `AbstractLayerOrdering{TopDown, BottomUp}`
   abstraction in plan 17 — driver-level flip at read time is the
   current contract (CLAUDE.md Invariant 2). A dedicated layer-ordering
   plan is a follow-up candidate (sibling to plan 18).

3. **Flux kept as `kg/s` per cell** (plan doc Decision 4 had
   `kg/m²/s × cell_area × dt`). Prognostic tracer is mass
   ([CellState.jl:23](../../../src/State/CellState.jl#L23)), so
   `rm[i, j, Nz, t] += rate[i, j] * dt` is dimensionally consistent.
   Acceptance: Commit 3 test asserts that summed `rate × dt × step_count`
   equals global Earth-total emissions over the run window. Per-area
   flux variant is a follow-up candidate (some CATRINE inventories
   arrive in kg/m²/s).

4. **Commit 6 simpler than plan doc** — since existing `DrivenSimulation`
   already owns `surface_sources`, Commit 6 is "move the call site into
   the palindrome via `TransportModel.emissions`, install via
   `with_emissions` at sim construction, delete the 2 sim-level
   application lines + 1 workaround line" rather than building
   `SurfaceFluxOperator` plumbing from zero.

## Follow-up plan candidates (not shipped in 17)

Recorded here at Commit 0 so the retrospective at Commit 9 has a
pre-registered list:

- **`AbstractLayerOrdering{TopDown, BottomUp}`** — cross-cutting type
  for operators to dispatch on vertical-layer convention. Motivated
  by GEOS-FP (top-down) vs GEOS-IT (bottom-up). Touches every
  vertically-indexed kernel.
- **Per-area flux variant** — `kg/m²/s` × `cell_area` sibling to
  `SurfaceFluxSource`, for inventories that arrive per-area.
- **Stack emissions** (3D source fields) — non-surface emission layers
  (tall stacks, aviation, volcanic plumes).
- **`DepositionOperator`** — dry deposition as a first-class operator.
- **Remove `_inject_source_kernel!`** — dead code in
  [src/Kernels/CellKernels.jl](../../../src/Kernels/CellKernels.jl).

## Commit sequence (initial draft; will evolve)

- **Commit 0** — NOTES.md + baseline + surveys + memory compaction.
- **Commit 1** — `StepwiseField{FT, N, A, T}` + tests.
- **Commit 2** — `PerTracerFluxMap{FT, Sources}` + tests.
- **Commit 3** — Migrate `SurfaceFluxSource` + `AbstractSurfaceFluxOperator`,
  `NoSurfaceFlux`, `SurfaceFluxOperator`, kernel, `apply!`,
  `apply_surface_flux!` array-level.
- **Commit 4** — `current_time(meteo)` threading through chemistry
  and diffusion `apply!`.
- **Commit 5** — Palindrome integration in `strang_split_mt!`.
- **Commit 6** — `TransportModel.emissions` + `with_emissions` +
  `DrivenSimulation` cleanup.
- **Commit 7** — Ordering study.
- **Commit 8** — Benchmarks.
- **Commit 9** — Retrospective + `ARCHITECTURAL_SKETCH_v3.md`.

## Commit-by-commit notes

### Commit 0 — NOTES + baseline

This file. Surveys saved to `artifacts/plan17/`. Baseline test summary
log captured. Plan docs `17_SURFACE_EMISSIONS_PLAN.md`,
`ARCHITECTURAL_SKETCH_v2.md`, `CLAUDE_additions.md` (pre-existing
untracked, authored by user) included in Commit 0.

---

Subsequent commit sections will be filled in as execution proceeds
(per CLAUDE_additions.md §"Retrospective sections are cumulative").
