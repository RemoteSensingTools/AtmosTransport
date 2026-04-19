# Plan 16b Execution Notes — Vertical Diffusion Operator

**Plan:** [16_VERTICAL_DIFFUSION_PLAN.md](16_VERTICAL_DIFFUSION_PLAN.md) (v1)
— the second half of the original plan 16, after 16a ([docs/plans/16A_TIME_VARYING_FIELDS_PLAN/NOTES.md](../16A_TIME_VARYING_FIELDS_PLAN/NOTES.md)) shipped the
`AbstractTimeVaryingField{FT, N}` + `ConstantField{FT, N}`
abstraction and retrofitted `ExponentialDecay.decay_rates`.

## Baseline

- **Parent commit:** `3cd9010` (plan 16a Commit 3, tip of
  `time-varying-fields`).
- **Branch:** `vertical-diffusion` created from `time-varying-fields`
  HEAD. The resume memo stated `time-varying-fields` had been
  merged into `advection-unification`; verification on 2026-04-19
  showed the merge had NOT happened (`git log advection-unification..time-varying-fields`
  shows the four 16a commits). Branching directly from
  `time-varying-fields` keeps history linear and defers the
  `advection-unification` merge as a separate concern.
- **Pre-existing test failures:** 77, inherited from plan 16a
  (`test_basis_explicit_core: 2`, `test_structured_mesh_metadata: 3`,
  `test_poisson_balance: 72`). Plan 16b must preserve this count.
- **Fast-path baseline** (captured in
  [artifacts/plan16/baseline_test_summary.log](../../../artifacts/plan16/baseline_test_summary.log)):
  - `test_fields.jl`: 21 pass (N=0 ConstantField only)
  - `test_chemistry.jl`: 37 pass

## Deviations from plan doc §4.4

Two scope decisions recorded up front:

1. **Commit 2a (chemistry rates retrofit) is already done.** Plan 16a
   landed it as its Commit 2 (`e749b2a`); `ExponentialDecay.decay_rates`
   already stores `NTuple{N, ConstantField{FT, 0}}`. Skip.

2. **`ConstantKzField` is redundant.** Plan doc §4.3 Decision 3 lists
   `ConstantKzField{FT} <: AbstractTimeVaryingField{FT, 3}` as one of
   three Kz field types; that predates `ConstantField{FT, N}`, which
   already provides the scalar spatially-uniform case at any rank. The
   existing `test_fields.jl` includes a `@testset "volume (N=3) — anticipating
   Kz in plan 16b"` (lines 55–60) that already demonstrates
   `ConstantField{Float64, 3}` returning a scalar from a 3-tuple index.
   The first new concrete type will be **`ProfileKzField`** (per-level
   profile, rank-3, vertically-varying) — the simplest field type that
   actually exercises the rank-3 path beyond the spatially-uniform
   case.

## Commit sequence (initial draft; will evolve)

- **Commit 0** — NOTES.md + baseline, plan doc moved into subfolder
  (convention-matching with 14/15/16a).
- **Commit 1a** — `ProfileKzField` rank-3 profile field + tests.
- **Commit 1b** — `PreComputedKzField` (deferred; needs `StepwiseField`
  design).
- **Commit 1c** — `DerivedKzField` Beljaars-Viterbo port
  (~300 lines, highest-risk).
- **Commit 2** — Thomas solve + diffusion kernel.
- **Commit 3** — `ImplicitVerticalDiffusion` operator + `apply!`.
- **Commit 4** — Palindrome integration (`X Y Z V Z Y X`) in
  `strang_split_mt!`.
- **Commit 5** — `TransportModel` + `DrivenSimulation` wiring.
- **Commit 6** — Benchmarks.
- **Commit 7** — Documentation + retrospective.

## Commit-by-commit notes

### Commit 0 — NOTES + baseline

Moved `docs/plans/16_VERTICAL_DIFFUSION_PLAN.md` into the subfolder
to match the plan-folder layout used by 14/15/16a. Wrote this file
as the execution log.

## Decisions beyond the plan

(To be filled in as they arise.)

## Surprises

(To be filled in as they arise.)

## Interface validation findings

(To be filled in — per plan doc §5.3, three specific items:
`apply!` signature works unchanged for diffusion; tridiagonal
structure is transposable without rewrite; `TimeVaryingField`
works cleanly for 3D Kz.)
