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

## Benchmark — post-plan-12 vs post-plan-13 baseline (wurst L40S)

### CPU medium (F64 noisy from shared-host load; F32 clean)

| FT  | Scheme | median (ms) | MAD (ms) | vs. baseline |
|-----|--------|------------:|---------:|-------------:|
| F64 | Upwind |     480.146 |   36.209 | +1.1% (noisy) |
| F64 | Slopes |    2822.837 |   14.727 | +2.0% (noisy) |
| F64 | PPM    |    2517.511 |    3.513 | −0.9% |
| F32 | Upwind |     355.806 |    0.571 | −0.5% |
| F32 | Slopes |    2379.604 |    0.835 | −0.3% |
| F32 | PPM    |    2154.191 |    0.528 | +0.1% |

### GPU F32 medium (with --events)

| Scheme | host(ms) | cuda(ms) | Δ | vs. baseline |
|--------|---------:|---------:|---:|-------------:|
| Upwind |    2.990 |    2.980 | 0.010 | −0.2% |
| Slopes |    3.095 |    3.087 | 0.008 | +0.3% |
| PPM    |    3.254 |    3.246 | 0.009 | +0.1% |

All within ±5% as planned. Plan 12 is a naming/plumbing cleanup, not
a performance change; the effective noise-floor result confirms no
regression.

## Retrospective

### What worked

- **Precondition audit caught scope expansions early.** Plan 12 v2
  didn't list `PPMAdvection`, [src/Operators/Operators.jl](../../src/Operators/Operators.jl),
  or [src/AtmosTransport.jl](../../src/AtmosTransport.jl) as touchpoints,
  but a single grep at Commit 0 surfaced all three. Folding them into
  Commit 4 avoided a post-facto follow-up.
- **"Commit 1 is a no-op" recognition.** The equivalence test already
  existed at [test_advection_kernels.jl:286-333](../../test/test_advection_kernels.jl)
  (from earlier work). Plan-12-v2's Commit 1 didn't need to write it —
  just needed to run it as a premise check. That's a quick win if
  future plans inherit tests from earlier plans.
- **Two-step struct deletion (Commits 3 + 4).** Deleting kernels +
  dispatch first, then structs, gave two small clean commits instead
  of one 600-line monolith. Each is independently revertable.

### What didn't

- **Plan 12 v2 split of "delete kernels" vs "keep wrappers" was
  impossible** because there were no separate "wrappers through modern
  kernels" — legacy types dispatched only through legacy kernels.
  Commit 3 collapsed the distinction and deleted both together. Logged
  as an adjustment in the approved execution plan.
- **CPU F64 bench numbers were noisy on wurst** (MAD 14–36 ms) due to
  shared-host load during the measurement run. F32 + GPU numbers are
  clean; F64 bench should be re-run on an idle host for a tight
  comparison (not blocking — bench is sanity check not perf gate).

### Deferred for future plans

- [`cfl_scratch_m` / `cfl_scratch_rm`](../../src/Operators/Advection/StrangSplitting.jl#L81-L82)
  fields in AdvectionWorkspace are now dead (only the evolving-mass
  pilot used them). Plan 12 v2 §3.1 explicitly excluded ping-pong-era
  workspace edits, so removal deferred to a future workspace-focused
  cleanup. Small 5-line struct edit.
- [test_dry_flux_interface.jl:35](../../test/test_dry_flux_interface.jl#L35)
  now checks `isdefined(AtmosTransport, :SlopesScheme)` as the API
  coverage anchor. If UpwindScheme is ever deleted from the public
  API, a similar migration will be needed.

### Final plan-12 impact summary

- **Files deleted:** [Upwind.jl](../../src/Operators/Advection/),
  [RussellLerner.jl](../../src/Operators/Advection/),
  [PPM.jl](../../src/Operators/Advection/) (425 lines).
- **Net lines removed:** ~760 across 10 files. [StrangSplitting.jl](../../src/Operators/Advection/StrangSplitting.jl)
  is ~165 lines shorter than post-plan-13 (exceeds plan's ≥100 target).
- **One scheme hierarchy:** `AbstractAdvectionScheme` and descendants;
  no more `AbstractAdvection` / `AbstractConstantReconstruction` /
  `AbstractLinearReconstruction` / `AbstractQuadraticReconstruction`.
- **One rule for schemes:** `UpwindScheme()`, `SlopesScheme(L)`,
  `PPMScheme(L)`.
- Zero `Union{AbstractAdvection, AbstractAdvectionScheme}` signatures
  in any file — simplified down to plain `AbstractAdvectionScheme`.
- 77 pre-existing failures unchanged at every commit; ULP, mass
  conservation unchanged; CPU/GPU perf within ±5%.

## Commit log

- **Commit 0** (`77967ff`): plan 12 NOTES.md + baseline capture.
- **Commit 1** (`4e3d8b1`): premise check — equivalence tests pass
  with `rm_ulp=0.0 m_ulp=0.0` across all {F32,F64}×{limiter} cases.
- **Commit 2** (`a077d97`): migrated ~24 test call sites; deleted
  duplicate `@test_throws` tests in
  [test_basis_explicit_core.jl](../../test/test_basis_explicit_core.jl);
  redirected [test_dry_flux_interface.jl:35](../../test/test_dry_flux_interface.jl#L35)
  `isdefined` check to `SlopesScheme`.
- **Commit 3** (`631a288`): deleted legacy @kernel functions (in
  Upwind.jl + RussellLerner.jl), `@eval` dispatch loops, face-indexed
  UpwindAdvection rows, legacy no-scheme overloads, and equivalence
  tests. Net −606 / +8.
- **Commit 4** (`d441391`): deleted struct UpwindAdvection,
  FirstOrderUpwindAdvection, RussellLernerAdvection, PPMAdvection
  and their files; deleted abstract AbstractAdvection hierarchy;
  pruned exports in Operators.jl + AtmosTransport.jl; collapsed
  every `Union{AbstractAdvection, AbstractAdvectionScheme}` to plain
  `AbstractAdvectionScheme`. Net −146 / +26.
- **Commit 5**: CLAUDE.md update (modern scheme names) + NOTES.md
  retrospective + final bench artifacts.
