# Plan 23 — TM5 Convection — Execution Notes

## Baseline (Commit 0, 2026-04-21)

**Parent commit:** `80ae14656750ace2dc145f10f17d9fe90631e935`
(`fix(docs): resolve two review findings from cleanup commit b5fe56b`).

**Branch:** `convection` (tracking `origin/convection`).

**Scope:** ship `TM5Convection` alongside `CMFMCConvection`; resolve
CMFMC-only assumptions in `DrivenSimulation`, driver load paths, and
workspace factory; add ERA5 `(entu, detu, entd, detd)` preprocessing
path. Plan doc lives outside the repo at
`/home/cfranken/.claude/plans/bring-last-session-into-lively-scroll.md`.

## Commit 0 artifacts

- [`artifacts/plan23/baseline_commit.txt`](../../../artifacts/plan23/baseline_commit.txt)
- [`artifacts/plan23/baseline_core.log`](../../../artifacts/plan23/baseline_core.log)
  — `julia test/runtests.jl` stopped at `test_advection_kernels.jl`
  with an `UndefVarError: LatLonMesh` caused by include-twice /
  export-ambiguity between test files. Same behaviour on parent
  `80ae146` — not caused by this plan, inherited from post-plan-21
  Phase 6 test harness shape. Runtests needs per-file isolation for
  a clean baseline.
- [`artifacts/plan23/baseline_core_per_file.log`](../../../artifacts/plan23/baseline_core_per_file.log)
  — each core test file run in isolation. This is the effective
  baseline plan 23 preserves.
- [`artifacts/plan23/matrix_structure.md`](../../../artifacts/plan23/matrix_structure.md)
  — TM5 `conv1` sparsity survey: dense lower + upper triangular
  within cloud window, identity rows above. Solver locked to
  partial-pivot GE on `lmc × lmc` active sub-block (see principle 2).
- [`artifacts/plan23/basis_decision.md`](../../../artifacts/plan23/basis_decision.md)
  — `TM5Convection` is basis-polymorphic, identical to
  `CMFMCConvection`. Preprocessor writes `mass_basis = :moist` by
  default; dry-basis variant is out of plan 23 scope.
- [`artifacts/plan23/cmfmc_dispatch_survey.txt`](../../../artifacts/plan23/cmfmc_dispatch_survey.txt)
  — every source site that dispatches on `CMFMCConvection` or
  hard-codes CMFMC in a non-Convection module. Input to Commits 1
  and 3.

## Pre-existing state that this plan must preserve

- `test_advection_kernels.jl` include-order bug when invoked via
  `runtests.jl` sequence.
- `test_structured_mesh_metadata.jl` — 3 pre-existing failures under
  "CubedSphereMesh conventions" testset (CLAUDE.md baseline).
- `test_poisson_balance.jl` — 72 pre-existing failures (CLAUDE.md
  baseline; plan 12-era, not in scope).

## Deviations from plan doc §4.4 (execution log)

*(Filled cumulatively as commits land; not a final dump at Commit 7.)*

### Commit 0

- None so far. Baseline captured, survey complete, auto-memory
  refreshed, stale repo doc claims fixed.

### Commit 1

- Compressed against plan doc: the plan's "Commit 1" step 5
  ("add tm5_fields slot") was dropped because `ConvectionForcing`
  already exposed the slot (verified at
  [`ConvectionForcing.jl:42, :79, :82`](../../../src/MetDrivers/ConvectionForcing.jl#L42);
  `copy_convection_forcing!` at `:167–182` already copies
  `tm5_fields`). Commit 1 consumed existing infrastructure.
- `_validate_convection_runtime` was refactored to the
  `_validate_convection_window!` dispatch pattern (plan doc Commit 1
  step 4) without changing the outer signature, so no downstream
  callers needed touching. The three internal methods are:
  `NoConvection` (no-op), `CMFMCConvection` (existing check),
  `TM5Convection` (new: rejects `nothing` `tm5_fields`), and an
  `AbstractConvectionOperator` fallback that points the developer
  at this file (principle 10 "error message names the fix").
- `TM5Workspace` parametric on `{FT, M, P, C}` with NTuple{6}
  variant for CS — identical pattern to `CMFMCWorkspace`.
  `Adapt.adapt_structure` implemented.
- `TM5Convection.jl` stub `apply!` / `apply_convection!` throw
  `ArgumentError` via a shared `_tm5_stub_throw` helper that
  validates the workspace type and names Commit 4 as the next
  landing point.
- Test file `test/test_tm5_convection.jl` registered in
  `test/runtests.jl` `core_tests`. 36 passing tests covering:
  type + workspace construction, LL/RG/CS allocation shapes,
  `_convection_workspace_for` dispatch, stub error messages,
  `with_convection(model, TM5Convection())` end-to-end plumbing.
- Module README `src/Operators/Convection/README.md` updated to
  document `TM5Convection.jl` + `TM5Workspace` (required by the
  `test_readme_current.jl` freshness gate).
- 0 regressions against test_basis_explicit_core,
  test_driven_simulation, test_transport_model_convection,
  test_cubed_sphere_runtime, test_cs_chemistry,
  test_convection_forcing, test_readme_current.

## Retrospective sections (filled during execution)

### Decisions beyond the plan

*(Filled as they happen.)*

### Surprises

*(Filled as they happen.)*

### Interface validation findings

*(Filled as they happen.)*

### Measurement vs. prediction

*(Filled at Commit 7.)*

### Template usefulness for plans N+1

*(Filled at Commit 7.)*
