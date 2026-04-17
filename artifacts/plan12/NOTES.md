# Plan 12 — Legacy Scheme Type Consolidation: Execution Notes

Plan source: [docs/plans/12_SCHEME_CONSOLIDATION_PLAN_v2.md](../../docs/plans/12_SCHEME_CONSOLIDATION_PLAN_v2.md)
Approved execution plan: `/home/cfranken/.claude/plans/do-this-in-plan-majestic-meadow.md` (captures plan-12-v2 + post-plan-13 adjustments).

Started: 2026-04-17
Baseline commit: `4eb5ce8` (post-plan-13)
Baseline failures: 77 (same set as post-plan-11 / post-plan-13, see [baseline_failures.log](baseline_failures.log))

## Adjustments made at Commit 0 (from plan-12-v2)

1. **Equivalence test already exists** at [test/test_advection_kernels.jl:286-333](../../test/test_advection_kernels.jl#L286). Commit 1 becomes a no-op premise check (run + confirm).
2. **PPMAdvection / PPM.jl added to deletion scope** — legacy stub tied to the `AbstractQuadraticReconstruction` hierarchy.
3. **Extra module-plumbing files**: [src/Operators/Operators.jl](../../src/Operators/Operators.jl) + [src/AtmosTransport.jl](../../src/AtmosTransport.jl) carry legacy exports.
4. **cfl_scratch_* cleanup deferred** (plan 12 v2 §3.1 bans workspace-field edits).
5. **Abstract type subtype checks** at `test_basis_explicit_core.jl:31-33` migrate alongside concrete types.
6. **Duplicate `@test_throws` deleted** rather than renamed (`:227-229` and `:310-311` in basis_explicit_core).

## Decisions made beyond the plan
_(populate as execution proceeds)_

## Deferred observations
- Workspace `cfl_scratch_m` / `cfl_scratch_rm` fields remain dead post-plan-13 / post-plan-12. Candidate for deletion in a future workspace-focused cleanup.

## Surprises vs. the plan
_(populate as execution proceeds)_

## Test anomalies
_(populate as execution proceeds)_

## Commit log

- **Commit 0**: plan 12 NOTES.md + baseline capture.
- **Commit 1** (empty): premise check. Ran
  `test/test_advection_kernels.jl`. All equivalence testsets pass
  with `rm_ulp=0.0 m_ulp=0.0` across {F32,F64}×{MonotoneLimiter,
  NoLimiter} for RussellLernerAdvection vs SlopesScheme, and the
  UpwindAdvection-vs-UpwindScheme `==` bit-identical assertions
  pass. Premise for plan 12 holds; proceeding to migration.
