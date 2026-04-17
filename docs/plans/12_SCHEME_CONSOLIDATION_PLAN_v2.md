# Scheme Type Consolidation — Implementation Plan for Plan Agent (v2)

**Status:** Ready for execution.
**Target branch:** new branch from whichever branch currently
  holds plan 11's shipped work — typically
  `restructure/dry-flux-interface` if plan 11 hasn't been merged
  to `main` yet, or `main` if it has. Verify in §4.1.
**Estimated effort:** 1-2 weeks, single engineer / agent.
**Primary goal:** Migrate all callers from legacy advection scheme
  types (`UpwindAdvection`, `RussellLernerAdvection`) to the modern
  scheme hierarchy (`UpwindScheme`, `SlopesScheme{L}`), then delete
  the legacy types and their associated kernel machinery.

Self-contained document. ~25 minutes to read.

**Dependencies verified:**
- Plan 11 (ping-pong) has shipped — `AdvectionWorkspace` has
  `rm_A`/`rm_B`/`m_A`/`m_B` fields; sweep functions no longer
  `copyto!` between sweeps.
- Plan 11 added a union-typed ping-pong signature so BOTH legacy
  scheme types AND modern scheme types get the ping-pong win. This
  means the legacy types are no longer "parasitic duplicate code" —
  they share the ping-pong infrastructure. The migration is still
  valid, but the motivation shifts slightly: we're eliminating
  duplicate NAMES, not duplicate MACHINERY.

**Companion document:** `ARCHITECTURAL_SKETCH.md` describes the
target end state of the advection subsystem after all refactors.
Skim before starting; your diff should converge toward it.

---

# Revisions from v1 (based on plan 11 execution)

What changed between v1 and this v2:

1. **Motivation updated.** The legacy types now share ping-pong
   infrastructure (thanks to plan 11 decision 1). This is a
   cleaner starting point than we predicted — the machinery is
   unified, only the names are duplicated. Migration is still the
   right call, but it's now a pure naming cleanup rather than a
   deduplication.

2. **NOTES.md is now a required Commit 0**, not an optional
   practice. Plan 11's execution had to be reminded to create the
   file. Making it a hard dependency before any code change forces
   the discipline.

3. **Baseline failure capture added to preconditions.** Plan 11
   revealed ~77 pre-existing test failures (unrelated to advection)
   that blocked clean test-suite runs. Plan 12's agent MUST
   capture the full baseline failure set, so regressions can be
   distinguished from pre-existing noise.

4. **Worktree pattern noted.** Plan 11's agent discovered
   `git worktree add` as a clean way to capture before/after
   perf baselines. For this cleanup refactor, the ±5%
   performance tolerance is wide enough that a worktree baseline
   isn't needed (the standard bench script against current HEAD
   is sufficient). The worktree pattern is noted here so plans
   13 and 14 — which target real performance wins — can adopt it
   as their standard.

5. **Tighter design decision rules.** Plan 11 uncovered one
   contradictory rule ("likely DELETE... but err on the side of
   KEEPING") which forced the agent to make a judgment call that
   was mine to make. v2 resolves every such case to a single clear
   answer.

6. **Performance expectations removed.** v1 had vague language
   about "no performance regression expected." This is a cleanup
   refactor, not a performance refactor. Plan 11 taught us to
   measure but not over-promise. v2 says: measure with the
   benchmark, expect ±5%, investigate if out of that range.

---

# Part 1 — Orientation

## 1.1 The problem in one paragraph

The advection subsystem has TWO parallel scheme type hierarchies:

```
Legacy (being removed):                 Modern (keeping):
  AbstractAdvection                       AbstractAdvectionScheme
  ├── AbstractConstantReconstruction     ├── AbstractConstantScheme
  │   └── UpwindAdvection                │   └── UpwindScheme
  └── AbstractLinearReconstruction       └── AbstractLinearScheme
      └── RussellLernerAdvection             └── SlopesScheme{L}
                                          └── AbstractQuadraticScheme
                                              └── PPMScheme{L}
```

After plan 11, both sets of scheme types dispatch to the same
ping-pong sweep functions (via a union-typed signature). So they're
not two independent code paths — they're two sets of NAMES for the
same machinery.

Functional equivalences (verified bit-identically in existing tests):
- `UpwindAdvection()` ≡ `UpwindScheme()`
- `RussellLernerAdvection(use_limiter=true)` ≡ `SlopesScheme(MonotoneLimiter())`
- `RussellLernerAdvection(use_limiter=false)` ≡ `SlopesScheme(NoLimiter())`

The costs of keeping duplicate names:
- Two sets of `@eval` blocks generating sweep variants in
  `StrangSplitting.jl`
- Two sets of `@kernel` definitions in `Upwind.jl` and
  `RussellLerner.jl` (these are now dead code post-plan-11 but still
  take up space)
- Two abstract hierarchies (`AbstractAdvection` vs
  `AbstractAdvectionScheme`)
- Confusion for new contributors ("which scheme type do I use?")

Deletion target: ~700 lines across `StrangSplitting.jl`,
`Upwind.jl`, `RussellLerner.jl`, and the abstract type definitions
in `AbstractOperators.jl`.

## 1.2 What "legacy" means precisely

`src/Operators/Advection/Upwind.jl` (120 lines) — contains:
- `struct UpwindAdvection <: AbstractConstantReconstruction`
- `const FirstOrderUpwindAdvection = UpwindAdvection`
- Legacy monolithic `@kernel` functions `_upwind_x_kernel!`,
  `_upwind_y_kernel!`, `_upwind_z_kernel!`
- CPU fallback `_upwind_face_flux` helper

`src/Operators/Advection/RussellLerner.jl` (299 lines) — contains:
- `struct RussellLernerAdvection <: AbstractLinearReconstruction`
- Legacy `@kernel` functions `_rl_x_kernel!`, `_rl_y_kernel!`,
  `_rl_z_kernel!`

These are the "to be deleted" files.

**Note on plan-11 aftermath:** the legacy @kernel functions in
these files are already dead code after plan 11 because the legacy
scheme types dispatch through the modern ping-pong sweep wrappers.
Verify by grepping:
```bash
grep -rn "_upwind_x_kernel!\|_rl_x_kernel!" src/ test/ --include="*.jl"
```
Expected: matches only in the file where they're defined, plus
possibly in `@eval` loops in `StrangSplitting.jl` that may now be
unreachable. If anything OUTSIDE those files calls them, the plan
needs adjustment — tell me before proceeding.

## 1.3 Where the legacy types are USED

Every use site must be migrated or deleted before the types can
be removed. Critical audit:

**In source:**
- `src/Operators/Advection/StrangSplitting.jl` — `@eval` loops
  generating `sweep_x!`/`sweep_y!`/`sweep_z!` variants that
  dispatch on legacy types. These pass through to ping-pong
  wrappers post-plan-11.
- `src/Operators/Advection/StrangSplitting.jl` — face-indexed
  `apply!` dispatch that mentions `FirstOrderUpwindAdvection`
- `src/Operators/Advection/StrangSplitting.jl` — error stub for
  `AbstractAdvection` (line varies post-plan-11; re-grep)
- `Union{AbstractAdvection, AbstractAdvectionScheme}` type
  signatures throughout `StrangSplitting.jl` — plan 11 kept these
  on purpose so both hierarchies benefit from ping-pong. This
  plan simplifies them to `AbstractAdvectionScheme` after
  migration is complete.

**In tests (these MUST be migrated, not deleted):**
- `test/test_real_era5_dry_conversion.jl` — one call site
- `test/test_real_era5_v1_vs_v2.jl` — one call site
- `test/test_real_era5_direct_dry_binary.jl` — one call site
- `test/test_basis_explicit_core.jl` — multiple sites including
  `@test_throws` tests
- `test/test_dry_flux_interface.jl` — extensive use (7+ sites)
- `test/test_era5_latlon_e2e.jl` — one call site

**CRITICAL: re-grep before starting.** Plan-11 execution changed
line numbers. The precondition check (§4.1) re-runs grep and
updates the agent's internal map of where changes need to happen.

## 1.4 The migration strategy

Two phases:

**Phase 1 — Verify equivalence and add a safety net.**
Add a bit-identical test that compares legacy and modern scheme
outputs. If it passes, the migration is safe (they produce the
same results). If it fails, STOP — the whole plan premise breaks.

**Phase 2 — Migrate all callers to use the new names explicitly.**
Each call site moves from legacy to modern names via mechanical
search-and-replace. Then the legacy types and their associated
abstract type hierarchy are deleted.

**Why two phases:** by verifying equivalence first, we prove
correctness before changing anything. Then Phase 2 is pure textual
migration with no semantic risk.

## 1.5 Test suite discipline

Read §1.7 of `11_PINGPONG_REFACTOR_PLAN.md` (test suite section) —
the tolerance numbers and test commands apply unchanged here.

**Baseline failures discovery (from plan 11):** there are ~77
pre-existing test failures in the baseline across:
- `test_basis_explicit_core.jl` (2 failures at lines 211-212,
  CubedSphereMesh API)
- `test_structured_mesh_metadata.jl` (3 failures)
- `test_poisson_balance.jl` (72 failures)

These are NOT caused by plan 11 and should NOT be caused by plan
12. The precondition check (§4.1) captures the exact set so
regressions can be distinguished from pre-existing noise.

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before this refactor:
- Two scheme hierarchies (legacy + modern)
- Two sets of `@kernel` functions in `Upwind.jl` and
  `RussellLerner.jl` (dead code after plan 11 but still
  allocated)
- `StrangSplitting.jl` has ~150 lines of @eval loops dispatching
  legacy types
- Every contributor asks "why are there two scheme systems?"

After this refactor:
- One scheme hierarchy (`AbstractAdvectionScheme`)
- `Upwind.jl` and `RussellLerner.jl` deleted
- `StrangSplitting.jl` ~150 lines shorter
- The answer to "why two systems" is: "there aren't two, there's
  one."

## 2.2 What doesn't change

- Production runs using `RussellLernerAdvection(use_limiter=true)`
  produce the same results (bit-identically, via the migration)
- Kernel physics (`reconstruction.jl`, `limiters.jl`) untouched
- Face-indexed (reduced Gaussian) support untouched
- Cubed-sphere Lin-Rood untouched

## 2.3 What this enables downstream

- Plan 13 (`13_SYNC_AND_CLEANUP_PLAN.md`) and plan 14
  (`14_SINGLE_PIPELINE_PLAN.md`) can assume a single scheme
  hierarchy. This plan (in Commit 4) also simplifies
  `Union{AbstractAdvection, AbstractAdvectionScheme}` signatures
  to just `AbstractAdvectionScheme`, so downstream plans don't
  inherit that complexity.
- Future plans that touch scheme dispatch don't need to worry
  about the legacy path.

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- **Kernel arithmetic.** The modern kernels (`_xsweep_kernel!`)
  and the legacy kernels (`_rl_x_kernel!`, `_upwind_x_kernel!`)
  implement the same math. Don't try to "improve" either.
- **`reconstruction.jl`, `limiters.jl`, `schemes.jl`,
  `ppm_subgrid_distributions.jl`** — out of scope entirely.
- **`LinRood.jl`** — explicitly out of scope for plan 12.
  Do not investigate, do not modify. Any perceived issues there
  are logged to NOTES.md as deferred.
- **`CubedSphereStrang.jl`** — uses its own separate workspace
  (`CSAdvectionWorkspace`) that doesn't reference the legacy
  scheme types. Leave alone.
- **Ping-pong buffer work.** Plan 11 is done. Don't re-visit.
- **Sync removal / CFL unification.** That's plan 13.

## 3.2 Do NOT add

- **Deprecation warnings for legacy types.** We're deleting, not
  going through a multi-release deprecation cycle. If someone
  outside the repo uses these types, they'll find out via a
  `MethodError` and update.
- **New scheme types** (e.g., face-indexed PPM). Different plan.
- **Performance optimizations.** This is a cleanup refactor.

## 3.3 Potential confusion — clarified

You will see `FirstOrderUpwindAdvection`. It's an alias for
`UpwindAdvection` (line 11 of `Upwind.jl`). Both die in this
refactor.

You will see `Union{AbstractAdvection, AbstractAdvectionScheme}`
in many signatures throughout `StrangSplitting.jl`. Plan 11 added
these unions (or preserved them) so legacy types get the ping-pong
benefit. After Commit 3 of THIS plan, they simplify to just
`AbstractAdvectionScheme`.

You will see abstract types `AbstractConstantReconstruction`,
`AbstractLinearReconstruction`, `AbstractQuadraticReconstruction`
— the LEGACY abstract types. Deleted along with their concrete
leaf types in Commit 4.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine the parent branch.
# Plan 11 shipped onto restructure/dry-flux-interface (or similar).
# Branch plan 12 from wherever plan 11's commits are currently
# accessible — NOT from main if plan 11 hasn't been merged yet.

# Check current branch state:
git branch -a | head -20
git log --oneline --all | grep -i "ping-pong\|plan 11" | head -5
# Find the branch that contains plan 11's commits.

# Branch from there:
git checkout <parent-branch>   # e.g., restructure/dry-flux-interface
git pull
git log --oneline | head -10
# Expected: ping-pong commits visible at the top.

git checkout -b scheme-consolidation

# 2. Clean working tree
git status
# Expected: "nothing to commit, working tree clean"

# 3. Verify plan 11 state
grep -c "rm_A\|rm_B" src/Operators/Advection/StrangSplitting.jl
# Expected: multiple matches (ping-pong buffers exist)

grep -c "Union{AbstractAdvection, AbstractAdvectionScheme}" src/Operators/Advection/StrangSplitting.jl
# Expected: multiple matches (plan 11 kept these on purpose)

# 4. Re-grep call sites. Line numbers have shifted since the plan
# was written; capture current state.
grep -rn "UpwindAdvection\|RussellLernerAdvection\|FirstOrderUpwindAdvection" \
    src/ test/ --include="*.jl" | tee artifacts/legacy_scheme_callers.txt
# Review the list. If it's significantly different from §1.3,
# adjust §4.2 accordingly BEFORE proceeding. Specifically:
# - If there are new src/ call sites beyond what §1.3 lists,
#   ask a human before proceeding.
# - If there are new test/ call sites, add them to the Commit 3
#   migration list.

# 5. Capture baseline failure set (distinguish regressions from
# pre-existing noise)
for testfile in test/test_basis_explicit_core.jl \
                test/test_advection_kernels.jl \
                test/test_structured_mesh_metadata.jl \
                test/test_reduced_gaussian_mesh.jl \
                test/test_driven_simulation.jl \
                test/test_cubed_sphere_advection.jl \
                test/test_poisson_balance.jl; do
    echo "=== $testfile ==="
    julia --project=. $testfile 2>&1 | tail -20
done | tee artifacts/baseline_test_summary.log

# Extract pass/fail count per file. Plan 11 showed ~77 pre-existing
# failures across test_basis_explicit_core (2), structured_mesh (3),
# and poisson_balance (72). Capture whatever your baseline shows —
# this becomes the "no new regressions" reference.

# 6. Record commit hash
git rev-parse HEAD > artifacts/baseline_commit.txt
```

**If any precondition fails, STOP.** Plan 11 is incomplete, or this
isn't the right branch, or the snapshot has drifted too far.

## 4.2 Change scope — the exact file list

**Files to DELETE (at Commit 4):**
- `src/Operators/Advection/Upwind.jl`
- `src/Operators/Advection/RussellLerner.jl`

**Files to MODIFY:**

Source:
- `src/Operators/Advection/Advection.jl`:
  - Remove `include("Upwind.jl")` and `include("RussellLerner.jl")`
  - Remove exports of legacy types and scheme aliases
  - Update module docstring if it references "legacy types"
- `src/Operators/AbstractOperators.jl`:
  - Delete `AbstractAdvection`, `AbstractConstantReconstruction`,
    `AbstractLinearReconstruction`,
    `AbstractQuadraticReconstruction` abstract types
- `src/Operators/Advection/StrangSplitting.jl`:
  - Delete `@eval` loops generating legacy sweep variants
    (at whatever line numbers they now occupy post-plan-11 —
    re-grep)
  - Delete face-indexed sweep generator entry that references
    `UpwindAdvection`
  - Delete error stub for `AbstractAdvection`
  - Simplify `Union{AbstractAdvection, AbstractAdvectionScheme}`
    to `AbstractAdvectionScheme` throughout

Tests (migration required):
- `test/test_real_era5_dry_conversion.jl`
- `test/test_real_era5_v1_vs_v2.jl`
- `test/test_real_era5_direct_dry_binary.jl`
- `test/test_basis_explicit_core.jl`
- `test/test_dry_flux_interface.jl`
- `test/test_era5_latlon_e2e.jl`

## 4.3 Design decisions (pre-answered)

Every decision here is final. Do NOT waffle, do NOT "err on the
side of X." If a rule is ambiguous, that's a bug in this
document — STOP and ask.

**Decision 1: Two-phase migration (verify then delete), not
one-phase "big bang" rewrite.**

Phase 1 adds a safety net. Phase 2 is mechanical migration. This
is the only correct structure; no alternatives considered.

**Decision 2: Verify bit-identical equivalence BEFORE any
migration.**

At Commit 1, add a test that runs both schemes on identical inputs
and asserts `max_diff == 0`. See Commit 1 below for exact code.
If this test FAILS, STOP. The legacy/modern schemes are NOT
bit-identical, and the whole migration premise is broken. Log to
NOTES.md and escalate to a human. Do not try to diagnose or fix
on your own — this is a fundamental premise check.

**Decision 3: Delete the abstract types too.**

`AbstractAdvection`, `AbstractConstantReconstruction`,
`AbstractLinearReconstruction`,
`AbstractQuadraticReconstruction` — all deleted.

Temptation: keep `AbstractAdvection` as an alias for
`AbstractAdvectionScheme` "in case someone needs it." REJECT this
temptation. The whole point of the refactor is to collapse two
hierarchies into one.

Audit external users:
```bash
grep -rn "AbstractAdvection[^S]" src/ --include="*.jl" | \
    grep -v "Operators/Advection/" | grep -v "Operators/AbstractOperators.jl"
```
Expected: zero matches. If there are matches, those external
users need updating — add them to the migration list in Commit 3.

**Decision 4: `FirstOrderUpwindAdvection` deletion is part of this
plan.**

It's an alias for `UpwindAdvection`. Both die together.

**Decision 5: Migration uses explicit limiter form.**

When migrating, always specify the limiter explicitly:
- `UpwindAdvection()` → `UpwindScheme()`
- `FirstOrderUpwindAdvection()` → `UpwindScheme()`
- `RussellLernerAdvection(use_limiter=true)` → `SlopesScheme(MonotoneLimiter())`
- `RussellLernerAdvection(use_limiter=false)` → `SlopesScheme(NoLimiter())`
- `RussellLernerAdvection()` (no kwarg — verify default is
  `use_limiter=true` per RussellLerner.jl line 12)
  → `SlopesScheme(MonotoneLimiter())`

Do NOT migrate to `SlopesScheme()` (default constructor). Always
specify the limiter explicitly — this makes the diff auditable
and intent-revealing.

**Decision 6: @test_throws tests migrate to modern type names.**

`test_basis_explicit_core.jl` has tests like:
```julia
@test_throws ArgumentError TransportModel(state, fluxes, grid, UpwindAdvection())
```
After migration:
```julia
@test_throws ArgumentError TransportModel(state, fluxes, grid, UpwindScheme())
```
The error-raising behavior should be unchanged (basis check is
independent of scheme type). If the migrated `@test_throws` FAILS
(the error doesn't get raised), it means the basis check had
scheme-type-specific logic. Log to NOTES.md and escalate — don't
try to fix silently.

**Decision 7: When running tests, use `--compiled-modules=existing`
or similar if import recompilation is slow.**

After deleting `Upwind.jl` and `RussellLerner.jl`, Julia may
recompile the module. On first run after deletion, allow time for
this. Subsequent runs should be fast.

## 4.4 Atomic commit sequence

### Commit 0: NOTES.md (REQUIRED, not optional)

Before any code change, create `NOTES.md` in
`docs/plans/12_SCHEME_CONSOLIDATION_PLAN/NOTES.md` (matching the
convention plan 11 established — each plan has its own NOTES.md
in its own subdirectory under `docs/plans/`):

```markdown
# Scheme Type Consolidation — Execution Notes

## Plan
`docs/plans/12_SCHEME_CONSOLIDATION_PLAN.md`

## Baseline
Commit: (fill in from artifacts/baseline_commit.txt)
Pre-existing test failures: (fill in from artifacts/baseline_test_summary.log)

## Decisions made beyond the plan
(fill in as you go)

## Deferred observations
(fill in as you go)

## Surprises vs. the plan
(fill in as you go)

## Test anomalies
(fill in as you go)
```

Commit this file. It's now tracked — future commits MUST update
it as decisions/surprises arise. This is non-negotiable: the file
is your hand-off to the next plan.

```bash
git add docs/plans/12_SCHEME_CONSOLIDATION_PLAN/NOTES.md
git commit -m "Commit 0: Add NOTES.md for scheme consolidation refactor"
```

### Commit 1: Add legacy/modern equivalence test

**File:** `test/test_advection_kernels.jl`

Add this test (at the end of the file, or wherever it fits):

```julia
@testset "Legacy/modern scheme equivalence" begin
    for FT in (Float64, Float32)
        grid, m, _, rm_grad, am, bm, cm = build_test_problem(FT)

        # Upwind legacy vs modern
        m_leg = copy(m); rm_leg = copy(rm_grad)
        state_leg = CellState(m_leg; tracer=rm_leg)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_leg = AdvectionWorkspace(m_leg)
        strang_split!(state_leg, fluxes, grid,
                      UpwindAdvection();
                      workspace=ws_leg)

        m_mod = copy(m); rm_mod = copy(rm_grad)
        state_mod = CellState(m_mod; tracer=rm_mod)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_mod = AdvectionWorkspace(m_mod)
        strang_split!(state_mod, fluxes, grid,
                      UpwindScheme();
                      workspace=ws_mod)

        @test maximum(abs.(m_leg .- m_mod)) == zero(FT)
        @test maximum(abs.(rm_leg .- rm_mod)) == zero(FT)

        # Slopes legacy vs modern (with limiter)
        m_leg = copy(m); rm_leg = copy(rm_grad)
        state_leg = CellState(m_leg; tracer=rm_leg)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_leg = AdvectionWorkspace(m_leg)
        strang_split!(state_leg, fluxes, grid,
                      RussellLernerAdvection(use_limiter=true);
                      workspace=ws_leg)

        m_mod = copy(m); rm_mod = copy(rm_grad)
        state_mod = CellState(m_mod; tracer=rm_mod)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_mod = AdvectionWorkspace(m_mod)
        strang_split!(state_mod, fluxes, grid,
                      SlopesScheme(MonotoneLimiter());
                      workspace=ws_mod)

        @test maximum(abs.(m_leg .- m_mod)) == zero(FT)
        @test maximum(abs.(rm_leg .- rm_mod)) == zero(FT)

        # Slopes no-limiter
        m_leg = copy(m); rm_leg = copy(rm_grad)
        state_leg = CellState(m_leg; tracer=rm_leg)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_leg = AdvectionWorkspace(m_leg)
        strang_split!(state_leg, fluxes, grid,
                      RussellLernerAdvection(use_limiter=false);
                      workspace=ws_leg)

        m_mod = copy(m); rm_mod = copy(rm_grad)
        state_mod = CellState(m_mod; tracer=rm_mod)
        fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
        ws_mod = AdvectionWorkspace(m_mod)
        strang_split!(state_mod, fluxes, grid,
                      SlopesScheme(NoLimiter());
                      workspace=ws_mod)

        @test maximum(abs.(m_leg .- m_mod)) == zero(FT)
        @test maximum(abs.(rm_leg .- rm_mod)) == zero(FT)
    end
end
```

Run it:
```bash
julia --project=. test/test_advection_kernels.jl
```

**Expected:** test passes (legacy and modern are bit-identical).

**If it FAILS:** STOP. Update NOTES.md:
```
## Surprises vs. the plan

Commit 1 equivalence test FAILED. Legacy and modern schemes are
NOT bit-identical. Halting migration pending investigation.
Failure details: (paste test output)
```
Escalate to a human. Do not proceed.

If it passes, commit:
```bash
git add test/test_advection_kernels.jl
git commit -m "Commit 1: Add legacy/modern scheme equivalence test"
```

### Commit 2: Migrate test callers to modern names

Six test files need migration. Do them one at a time, committing
each separately. That way if something breaks, the bisect is
surgical.

Migration rules (from Decision 5):
- `UpwindAdvection()` → `UpwindScheme()`
- `FirstOrderUpwindAdvection()` → `UpwindScheme()`
- `RussellLernerAdvection(use_limiter=true)` → `SlopesScheme(MonotoneLimiter())`
- `RussellLernerAdvection(use_limiter=false)` → `SlopesScheme(NoLimiter())`
- `RussellLernerAdvection()` → `SlopesScheme(MonotoneLimiter())`

For each file:

1. Edit the file
2. Run the specific test:
   ```bash
   julia --project=. test/<filename>
   ```
3. Compare pass/fail count to baseline (artifacts/baseline_test_summary.log)
4. If pass count matches baseline → commit
5. If pass count is LOWER than baseline → STOP, revert, investigate

```bash
git commit -m "Commit 2a: Migrate test_real_era5_dry_conversion.jl to modern schemes"
git commit -m "Commit 2b: Migrate test_real_era5_v1_vs_v2.jl to modern schemes"
# ...etc
```

Sub-commits within Commit 2 are fine — group by file.

After ALL six test files are migrated:
```bash
grep -rn "UpwindAdvection\|RussellLernerAdvection\|FirstOrderUpwindAdvection" test/ --include="*.jl"
```
Expected: **zero matches** (or only matches inside
`test_advection_kernels.jl`, which still runs the legacy/modern
equivalence test).

### Commit 3: Delete legacy kernel definitions

After Commit 2, nothing outside `StrangSplitting.jl`'s @eval loops
dispatches through legacy kernels. The `_upwind_x_kernel!`,
`_rl_x_kernel!` etc. definitions are dead code.

In `src/Operators/Advection/Upwind.jl`:
- Delete `_upwind_x_kernel!`, `_upwind_y_kernel!`, `_upwind_z_kernel!`
- Delete `_upwind_face_flux` helper if it's only used by those
  kernels (grep to verify)
- Keep `struct UpwindAdvection` and `const FirstOrderUpwindAdvection`
  for now (deleted in Commit 4)

In `src/Operators/Advection/RussellLerner.jl`:
- Delete `_rl_x_kernel!`, `_rl_y_kernel!`, `_rl_z_kernel!`
- Delete any helpers used only by those kernels (grep first)
- Keep `struct RussellLernerAdvection` for now

In `src/Operators/Advection/StrangSplitting.jl`:
- Delete the `@eval` loops that generate legacy sweep variants
  using `_upwind_x_kernel!` / `_rl_x_kernel!` etc.
- Plan-11 decision 1 kept the ping-pong wrappers working for
  legacy types via a separate `@eval` block — that block stays
  (it dispatches legacy types through modern kernels). After
  Commit 2, those wrappers are still called because the
  equivalence test in `test_advection_kernels.jl` uses them.

**Test after Commit 3:**
```bash
julia --project=. test/test_advection_kernels.jl
```
Expected: passes. The equivalence test still runs legacy vs
modern, and both now go through modern kernels.

Full suite:
```bash
julia --project=. test/runtests.jl
```
Expected: same pass/fail count as baseline.

```bash
git add -A
git commit -m "Commit 3: Delete dead legacy @kernel functions"
```

### Commit 4: Delete legacy types and files

Now the legacy types have no callers except the
`test_advection_kernels.jl` equivalence test. Delete the
equivalence test first (it's served its purpose), then delete
the types.

In `test/test_advection_kernels.jl`:
- Delete the "Legacy/modern scheme equivalence" testset added in
  Commit 1

In `src/Operators/Advection/Advection.jl`:
- Remove `include("Upwind.jl")` and `include("RussellLerner.jl")`
- Remove exports of `UpwindAdvection`, `FirstOrderUpwindAdvection`,
  `RussellLernerAdvection`

In `src/Operators/AbstractOperators.jl`:
- Delete `abstract type AbstractAdvection <: AbstractOperator end`
- Delete
  `abstract type AbstractConstantReconstruction <: AbstractAdvection end`
- Delete
  `abstract type AbstractLinearReconstruction <: AbstractAdvection end`
- Delete
  `abstract type AbstractQuadraticReconstruction <: AbstractAdvection end`

In `src/Operators/Advection/StrangSplitting.jl`:
- Delete the `@eval` blocks that dispatch legacy types through
  modern kernels (plan-11 Decision 1 wrappers) — these are now
  dead code
- Simplify every `Union{AbstractAdvection, AbstractAdvectionScheme}`
  to `AbstractAdvectionScheme`
- Delete the face-indexed sweep generator entry that references
  `FirstOrderUpwindAdvection`
- Delete the `AbstractAdvection` error stub

Delete the files:
```bash
git rm src/Operators/Advection/Upwind.jl
git rm src/Operators/Advection/RussellLerner.jl
```

**Test after Commit 4:**
```bash
grep -rn "UpwindAdvection\|RussellLernerAdvection\|FirstOrderUpwindAdvection\|AbstractAdvection[^S]" src/ test/ --include="*.jl"
```
Expected: **zero matches**.

```bash
julia --project=. test/runtests.jl
```
Expected: same pass/fail count as baseline (test count may be
slightly smaller because the equivalence test is gone).

Line count check:
```bash
wc -l src/Operators/Advection/*.jl
```
Expected:
- `Upwind.jl` and `RussellLerner.jl` absent
- `StrangSplitting.jl` is ≥100 lines shorter than post-plan-11
  baseline (because of the @eval block deletions)

```bash
git add -A
git commit -m "Commit 4: Delete legacy scheme types and abstract hierarchy"
```

### Commit 5: Documentation

- Update `src/Operators/Advection/Advection.jl` module docstring
  if it mentions legacy types
- Update CLAUDE.md if it references `UpwindAdvection`,
  `RussellLernerAdvection`, or the legacy abstract hierarchy
- Update any in-source comments that say "legacy" or
  "`AbstractAdvection`"
- Update NOTES.md with final summary

Run full test suite one more time:
```bash
julia --project=. test/runtests.jl
```

Run benchmark (sanity check — should be ±5% of baseline):
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl > artifacts/perf/after_12.log
```

Commit:
```bash
git commit -m "Commit 5: Documentation and final verification"
```

## 4.5 Test plan per commit

After EACH commit:

```bash
# 1. Compile check
julia --project=. -e 'using AtmosTransport'

# 2. Core tests
julia --project=. test/runtests.jl

# 3. Real-data tests if available (most affected by this change)
julia --project=. test/runtests.jl --all
```

Compare each run's pass/fail count to baseline. NEW failures →
STOP, revert, investigate. Pre-existing failures (the ~77 from
baseline) → ignore, they're noted.

Stop conditions (standard):
- New test regression: STOP, revert, investigate
- Compile error: STOP, revert
- Line count EXPANSION (refactor should DELETE lines): STOP,
  review the diff — something's off

## 4.6 Acceptance criteria

**Correctness:**
- All tests that passed in baseline pass post-refactor
- No NEW test failures beyond the pre-existing baseline set
- The `test_dry_flux_interface.jl` tests (which exercised
  `RussellLernerAdvection` most heavily) pass unchanged
- Real-data tests (`--all`) pass if run

**Code cleanliness (hard requirements):**
- `src/Operators/Advection/Upwind.jl` no longer exists
- `src/Operators/Advection/RussellLerner.jl` no longer exists
- `grep -rn "UpwindAdvection\|RussellLernerAdvection\|FirstOrderUpwindAdvection"
   src/ test/ --include="*.jl"` returns ZERO matches
- `grep -rn "AbstractAdvection[^S]" src/ --include="*.jl"`
  returns zero matches
- `grep -rn "Union{AbstractAdvection" src/ --include="*.jl"`
  returns zero matches
- `StrangSplitting.jl` is ≥100 lines shorter than post-plan-11
  baseline

**Performance:**
- Per-step wall time within ±5% of post-plan-11 baseline
  (this is a cleanup refactor, not a perf refactor; but
  shouldn't regress)
- If out of ±5% range, investigate before declaring done.

**Documentation:**
- NOTES.md exists and is complete with decisions, surprises,
  deferred observations
- CLAUDE.md updated if it referenced legacy types
- No in-source comments still mention legacy types

## 4.7 Rollback plan

Standard principles:
- Do not "fix forward"
- Revert to last-known-good commit
- Write the failure in NOTES.md
- Stop and ask if stuck >30 minutes

Specific rollback points:
- **Commit 1 fails (equivalence test).** Revert. The premise is
  broken. Do NOT attempt to diagnose alone — this is a premise-
  level correctness issue that needs human review.
- **Commit 2 (test migration) fails on a specific file.** Revert
  just that sub-commit. Look at the diff for that file — there
  may have been a subtle edit error (e.g., a stray comma,
  wrong limiter). Fix in a new attempt.
- **Commit 3 (delete kernel definitions) breaks something.**
  Revert. Something is still calling the legacy kernels. Run
  `grep -rn "_upwind_x_kernel!\|_rl_x_kernel!" src/ test/` to
  find the caller.
- **Commit 4 (delete types) breaks something.** Revert. Either
  the grep audit missed a caller, or there's a dynamic dispatch
  (e.g., string-based construction) that doesn't show up in
  grep. Investigate.

## 4.8 Known pitfalls

1. **"I'll just delete the types now and fix callers as errors
   appear."** NO. The two-phase migration exists to prevent
   exactly this. Phase 2 callers must be migrated BEFORE types
   are deleted, so the test suite stays green throughout.

2. **"The RussellLernerAdvection default use_limiter is probably
   true — I'll assume."** NO. VERIFY at
   `RussellLerner.jl` line 12 (or wherever the constructor is
   defined) before doing migrations that omit the kwarg. If
   the default is `false`, migrations translate differently.

3. **"I'll rename use_limiter=true to use_limiter=true in
   SlopesScheme."** NO. `SlopesScheme` takes a limiter TYPE
   (like `MonotoneLimiter()`), not a boolean kwarg.

4. **"Tests reference `AbstractConstantReconstruction` and
   `AbstractLinearReconstruction` directly — I need these
   types around."** Check the tests:
   ```bash
   grep -rn "AbstractConstantReconstruction\|AbstractLinearReconstruction" test/
   ```
   Expected: only in type-check tests like
   `@test UpwindAdvection <: AbstractConstantReconstruction`.
   Migrate those tests to check the modern hierarchy:
   `@test UpwindScheme <: AbstractConstantScheme`.

5. **"Union signatures look tricky — I'll leave them as
   `Union{AbstractAdvection, AbstractAdvectionScheme}` just in
   case."** NO. After Commit 4, `AbstractAdvection` no longer
   exists. The signature MUST simplify to
   `AbstractAdvectionScheme`. Leaving the Union in would cause
   a `UndefVarError`.

6. **"LinRood.jl has a similar kernel structure — I'll clean it
   up too."** NO. Out of scope. Log to NOTES.md as deferred.

7. **"CubedSphereStrang.jl has its own workspace — should I
   update it?"** NO. `CSAdvectionWorkspace` is a separate type
   from `AdvectionWorkspace`. It uses modern schemes already
   (or does its own thing). Out of scope for this plan.

8. **"The equivalence test passed on CPU. Do I need to run it
   on GPU?"** Yes, if GPU hardware is available. The GPU test
   should be bit-identical too. If only CPU is available,
   note in NOTES.md that GPU equivalence is unverified; the
   next pause point can re-run with GPU access.

9. **"I noticed an unrelated cleanup I could do."** Log to
   NOTES.md as deferred. Do not include in this PR.

---

# Part 5 — How to Work

## 5.1 Session cadence

- **Session 1:** Precondition, Commit 0 (NOTES.md), Commit 1
  (equivalence test — the highest-risk check)
- **Session 2-3:** Commit 2 (test migration, one file at a time)
- **Session 4:** Commit 3 (delete legacy kernels)
- **Session 5:** Commit 4 (delete legacy types and files)
- **Session 6:** Commit 5 (documentation)

Each session: `git status` at start, `NOTES.md` update at end,
decision log for anything not covered by the plan.

## 5.2 When to stop and ask

- Equivalence test (Commit 1) fails — STOP
- Any test that passed in baseline now fails — STOP
- A file you need to edit isn't in §4.2 — STOP, update NOTES.md,
  ask
- The scope feels like it's expanding — STOP
- You spent >30 minutes on a single issue — STOP, write it up,
  ask

## 5.3 NOTES.md discipline

Update NOTES.md at these moments (not just at the end):
- After Commit 0: fill in baseline info
- After Commit 1: note whether equivalence test passed cleanly
  or had surprises
- After each test migration sub-commit in Commit 2: note if
  anything was non-mechanical (e.g., a test's expected error
  message changed)
- After Commit 3-4: note any callers that were missed in the
  initial grep audit (→ these are learnings for plan 13, 14)
- At the end: a "Template usefulness for plans 13, 14" section
  similar to what plan 11's NOTES.md produced

The NOTES.md is your handoff to the human (me) at the pause
point. Treat it as part of the deliverable, not paperwork.

---

# End of Plan

After this refactor ships:
- Two files deleted (`Upwind.jl`, `RussellLerner.jl`)
- ~700 lines removed total
- ONE scheme hierarchy visible to users
- Convergent toward `ARCHITECTURAL_SKETCH.md`

The next plan in the sequence, `13_SYNC_AND_CLEANUP_PLAN.md`,
will build on this to remove unnecessary synchronizations and
unify the CFL pilot. The v2 of plan 13 will incorporate learnings
from plans 11 and 12's execution.
