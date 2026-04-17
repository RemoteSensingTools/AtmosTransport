# Sync Removal + CFL Pilot Unification + Rename — Implementation Plan for Plan Agent

**Version:** 2 (revised after plan 11 execution)
**Status:** Ready for execution. Run immediately after plan 11.
**Target branch:** new branch from whichever branch holds plan 11.
**Estimated effort:** 4-7 days, single engineer / agent.
**Primary goal:** Three related cleanups in one refactor:
  1. Remove `synchronize(backend)` calls that aren't needed for
     correctness (expected to be the biggest GPU win of any plan)
  2. Collapse the dual CPU/GPU CFL pilot into a single algorithm
  3. Rename buffer fields from `rm_buf`/`m_buf` to `rm_A`/`m_A`
     (drop the plan-11 backward-compat shim)

This document is self-contained. Read it top-to-bottom. ~25 minutes.

**Dependencies verified:**
- Plan 11 (ping-pong) has merged; commits `8b8d283..527eef0` on
  `restructure/dry-flux-interface`
- `AdvectionWorkspace` has `rm_A`, `rm_B`, `m_A`, `m_B`, `rm_4d_A`,
  `rm_4d_B` fields plus the `getproperty` compat shim
- Both scheme hierarchies (legacy and modern) use ping-pong

**Companion document:** `ARCHITECTURAL_SKETCH.md`.

---

# Revisions from v1

This v2 plan incorporates lessons from plan 11's execution. Changes
from v1:

1. **Performance expectations updated.** Plan 11 revealed that
   `synchronize(backend)` is the dominant GPU cost at typical
   problem sizes — ~3 ms per step on L40S, of which only ~0.18 ms
   was copyto. The remaining ~2.8 ms is dominated by sync overhead
   and kernel arithmetic. This means plan 13's sync removal is
   likely the biggest single GPU win of any plan in the sequence —
   20-40% per-step speedup expected, not 50-150 μs.

2. **Baseline failure capture added as explicit precondition.**
   Plan 11 hit 77 pre-existing test failures that had to be
   distinguished from regressions mid-execution. Plan 13 now
   captures the full baseline failure set upfront so the agent
   can reason about regressions cleanly.

3. **Worktree pattern codified for benchmarks.** Plan 11's agent
   used `git worktree add` to capture baseline performance without
   disturbing the working tree. This is now the standard pattern.

4. **NOTES.md creation promoted to Commit 0.** Plan 11's agent
   treated NOTES.md as optional and had to be reminded. Making
   it Commit 0 forces the file to exist before any code change.

5. **Commit 1 (CFL diagnostic test) is now mandatory as a premise
   check.** If the static CFL estimate diverges substantially from
   the evolving-mass estimate, the whole plan is wrong and must
   halt. Plan 11 showed that the plan-anticipated premises held;
   plan 13 has a different premise that needs empirical check.

6. **Clarified sync-removal taxonomy** — a few specific synchronizes
   that looked removable from reading have been explicitly
   categorized as "KEEP" based on closer reading of the Lin-Rood
   panel handling (line 709's comment was accurate as-of-plan-11).

---

# Part 1 — Orientation

## 1.1 The three-in-one structure

Three related but distinct cleanups live in this plan because they
interact:

- **Sync removal** makes kernel launches overlap, for major GPU
  perf
- **CFL pilot unification** replaces two algorithms with one, for
  maintainability and testability
- **Rename cleanup** drops the compat shim so names reflect reality

Doing them together avoids three separate test cycles and keeps the
diff focused on one theme: "make the orchestrator clean and fast."

## 1.2 Sync removal — what and why

Current state (post-plan-11): every sweep function ends with
`synchronize(backend)`. The ping-pong refactor removed `copyto!`
after the sync but kept the sync itself.

Why each sync was there:
- Pre-ping-pong: the `copyto!` after the kernel needed the kernel
  to be finished. `synchronize` ensured that.
- Post-ping-pong: the NEXT kernel is about to launch, reading
  from the previous kernel's output. KernelAbstractions orders
  operations on the same stream — the next kernel won't start
  until the previous one finishes. **The synchronize is
  redundant.**

**Cost of redundant syncs (measured via plan 11):**

Per-step time on L40S F32 medium (288×144×32, 5 tracers): ~3 ms.
Ping-pong eliminated ~0.18 ms of copyto but left sync overhead
intact. Each `synchronize(backend)` blocks the host until the GPU
is idle, preventing the host from scheduling the next kernel
concurrently with the previous kernel's tail.

Rough budget at 6 sweeps × ~0.4-0.5 ms sync-induced overhead per
sweep = 2.4-3.0 ms per step "wasted" on sync. Removing this gets
us to the arithmetic-bound regime where the kernel itself
dominates.

**Expected speedup on GPU: 20-40% per-step.** This is the biggest
single win in any plan of this sequence.

Expected speedup on CPU: <5%. CPU "sync" is essentially free (Julia
threading primitives) so there's little to gain.

**When syncs ARE needed:**
- Before host reads device memory (CFL pilot transfers, any
  `Array(rm)` call, conservation diagnostics)
- Between operations on different streams (not applicable here —
  KA uses one stream by default)
- At the end of `apply!` if the caller immediately reads the
  result

## 1.3 CFL pilot — what and why

Today, `_x_subcycling_pass_count` (similar for y, z, face-indexed)
has TWO code paths:

**GPU path (lines 964-971 approx, re-grep post-plan-11):**
```julia
if !(m isa Array)  # on-device
    out = max.(@view(am[1:Nx, :, :]), zero(FT)) .+
          max.(.- @view(am[2:Nx+1, :, :]), zero(FT))
    static_cfl = maximum(out ./ max.(m, eps(FT)))
    return _subcycling_pass_count(static_cfl, cfl_limit)
end
```
Broadcast + reduction, stays on device, uses static mass
estimate (worst-case outgoing flux / current mass).

**CPU path (lines 972-1004 approx):**
```julia
# Evolving-mass iterative algorithm
n_sub = _subcycling_pass_count(max_cfl_x(am, m, ws.cluster_sizes), cfl_limit)
while true
    # Simulate applying the flux in n_sub steps
    # Check CFL at each step against evolving mass
    # If any step violates, increase n_sub and retry
end
```
Evolves mass pass-by-pass, checks CFL against evolving mass, retries
with higher n_sub until satisfied.

**The problem:** the two algorithms give different `n_sub` values
for the same input. This breaks CPU/GPU test parity and creates a
class of "it works on CPU but GPU gets a different answer" bugs
that aren't real bugs — just artifacts of subcycling differences.

**The fix:** use the static estimate on both backends. It's always
≥ the evolving estimate (by construction — static assumes
worst-case mass), so it may over-subcycle slightly (at most by
factor 1.2-1.5 in pathological cases), but it's deterministic,
cheap, and gives identical results on CPU and GPU.

**Cost of switching:** very pathological flows might subcycle 1-2
extra passes. Typical runs: no change. Benefit: one algorithm,
not two; CPU/GPU bit-identical CFL decisions; simpler code.

## 1.4 Rename cleanup — what and why

Plan 11 added `rm_A`, `rm_B`, `m_A`, `m_B`, `rm_4d_A`, `rm_4d_B`
to `AdvectionWorkspace`, with a `getproperty` shim so
`ws.rm_buf → ws.rm_A` still works. This kept plan 11's diff narrow.

Now: drop the shim, make all callers use the new names. The old
names (`rm_buf`, `m_buf`, `rm_4d_buf`) go away. Any file that still
references them is updated.

Current users of old names (per plan 11's NOTES):
- `src/Operators/Advection/LinRood.jl` (lines 71, 74, 705-711)
- `src/Operators/Advection/CubedSphereStrang.jl` (lines 460, 553)
- `test/test_basis_explicit_core.jl` (lines 326, 333)

Per plan 11's observation #3, the `getproperty` shim currently
does three sequential name checks per access. Negligible in
current call sites (only touched once per panel per sweep) but
worth cleaning up.

## 1.5 What this plan does NOT cover

- **Scheme consolidation** — plan 12 (next, after this one)
- **Single advection pipeline** — plan 14 (multi-tracer default)
- **Face-indexed scheme extension** — still only UpwindScheme on
  face-indexed grids after this refactor
- **Cubed-sphere unification** — LinRood stays separate

---

# Part 2 — Why This Specific Change

## 2.1 Performance impact (updated from v1)

**Sync removal:** 20-40% per-step speedup on GPU expected, based
on plan 11's measurement that sync is the dominant GPU cost.
Combined with plan 11's ~6% ping-pong gain, total speedup over
pre-plan-11 baseline reaches ~25-45%.

**CFL unification:** zero-to-negative performance impact (maybe a
couple of extra subcycles in edge cases). This is a correctness /
maintainability change, not a performance change.

**Rename:** zero performance impact. Removes the 3-check
`getproperty` dispatch, which may save a few ns per access (not
measurable).

## 2.2 Code quality impact

Code deleted:
- ~60 lines of CPU-path CFL pilot in `_x_subcycling_pass_count`,
  `_y_subcycling_pass_count`, `_z_subcycling_pass_count`,
  `_horizontal_face_subcycling_pass_count`,
  `_vertical_face_subcycling_pass_count` (each has a ~25-30 line
  CPU-specific block)
- `getproperty` shim (~10 lines in `AdvectionWorkspace`)
- Redundant `synchronize(backend)` calls (~6 in
  `StrangSplitting.jl`, each 1 line)
- Possibly most of `MassCFLPilot.jl` if `max_cfl_x/y/z` have no
  external callers (check)

Net: ~150-200 lines deleted from `StrangSplitting.jl`, possibly
one file entirely (`MassCFLPilot.jl`).

---

# Part 3 — Out of Scope

**Do NOT touch:**
- Kernel code (`structured_kernels.jl`, `multitracer_kernels.jl`,
  `reconstruction.jl`, `limiters.jl`) — untouched
- Scheme types (plan 12's job)
- Buffer structure in `AdvectionWorkspace` (plan 11 finalized it;
  this plan only renames)
- `apply!` dispatch (plan 14's concern)
- LinRood's internal synchronize calls — those are between
  independent panel operations and are needed. Specifically,
  `LinRood.jl:709` has comment "required: ws.rm_buf/m_buf reused
  across panels" — this WAS accurate pre-plan-11 and remains
  accurate post-plan-11 because LinRood has its own panel loop
  that reuses buffers across panels (not a sweep-to-sweep ping-pong).

**Do NOT add:**
- Deprecation warnings for old names (plan 11 was the grace period;
  now the names are just gone)
- New synchronize calls anywhere (this plan only removes them)

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Branch from current main (which should have plan 11)
git checkout restructure/dry-flux-interface
git pull
git checkout -b sync-and-cleanup

# 2. Verify plan 11 state
grep -c "rm_A\|rm_B" src/Operators/Advection/StrangSplitting.jl
# Expected: multiple matches (buffers exist)

# Verify getproperty shim is still there
grep -A5 "getproperty.*AdvectionWorkspace" src/Operators/Advection/StrangSplitting.jl
# Expected: at least one getproperty method with :rm_buf, :m_buf, :rm_4d_buf

# Verify LinRood and CubedSphereStrang still use old names
grep -n "ws\.rm_buf\|ws\.m_buf\|workspace\.rm_buf\|workspace\.m_buf" \
    src/Operators/Advection/LinRood.jl \
    src/Operators/Advection/CubedSphereStrang.jl
# Expected: several matches — these are the sites to update in Commit 4

# 3. Capture full baseline failure list
# This is important: plan 11 hit 77 pre-existing failures. Knowing
# which are pre-existing vs. caused by this plan's changes is
# essential for debugging.
mkdir -p artifacts/plan13
julia --project=. test/runtests.jl 2>&1 | tee artifacts/plan13/baseline_tests.log
# Extract the failing test names for quick reference
grep -E "Got exception|Test Failed|Error During" artifacts/plan13/baseline_tests.log \
    > artifacts/plan13/baseline_failures.log
# Review artifacts/plan13/baseline_failures.log
# Per plan 11 NOTES, expect ~77 pre-existing failures across:
#   - test_basis_explicit_core.jl lines 211-212 (CubedSphereMesh API)
#   - test_structured_mesh_metadata.jl (3 failures)
#   - test_poisson_balance.jl (72 failures)
# These are NOT caused by plan 13 and should NOT block it.
# If the number is different (especially larger), STOP and
# investigate — something else broke since plan 11.

# 4. Capture baseline benchmark using worktree pattern
# (Codified from plan 11's successful approach.)
mkdir -p artifacts/plan13/perf

# Capture pre-change baseline on current branch
julia --project=. scripts/benchmarks/bench_strang_sweep.jl small cpu \
    > artifacts/plan13/perf/before_cpu_small.log
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium cpu \
    > artifacts/plan13/perf/before_cpu_medium.log

# If you have GPU access: also capture GPU baseline
# julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium gpu \
#     > artifacts/plan13/perf/before_gpu_medium.log

# 5. Record baseline commit
git rev-parse HEAD > artifacts/plan13/baseline_commit.txt
```

If any precondition fails, STOP. Plan 11 is incomplete or this
isn't the right branch.

## 4.2 Change scope — the exact file list

Re-grep line numbers at start of work (the ones below are from
plan 11's end-state snapshot):

**Files to MODIFY:**

Source:
- `src/Operators/Advection/StrangSplitting.jl`:
  - Remove `synchronize(backend)` calls inside sweep functions
  - Remove CPU-path CFL pilot code from
    `_x_subcycling_pass_count`, `_y_subcycling_pass_count`,
    `_z_subcycling_pass_count`,
    `_horizontal_face_subcycling_pass_count`,
    `_vertical_face_subcycling_pass_count`
  - Keep only the static (GPU-style) path; have it work on CPU
    via Julia's broadcast (it already does — the broadcast
    operators are backend-agnostic)
  - Remove the `getproperty` compat shim
  - Rename all internal references from `rm_buf`, `m_buf`,
    `rm_4d_buf` to `rm_A`, `m_A`, `rm_4d_A`
- `src/Operators/Advection/LinRood.jl`:
  - Update `ws.rm_buf` → `ws.rm_A`, `ws.m_buf` → `ws.m_A` at
    lines 71, 74, 705, 710, 711 (re-verify line numbers)
- `src/Operators/Advection/CubedSphereStrang.jl`:
  - Update `workspace.rm_buf` → `workspace.rm_A`,
    `workspace.m_buf` → `workspace.m_A` at lines 460, 553
- `src/Operators/Advection/MassCFLPilot.jl`:
  - Check if `max_cfl_x`, `max_cfl_y`, `max_cfl_z` have callers
    outside the deleted CPU pilot. If not: delete the file.
    If yes: keep the file but delete the now-unused helpers.

Tests:
- `test/test_basis_explicit_core.jl` (lines 326, 333):
  - Update `model.workspace.rm_buf` → `model.workspace.rm_A`

## 4.3 Design decisions (pre-answered)

**Decision 1: Static CFL estimate, always.**

No fallback. No "use evolving when available." Just one algorithm.
This is the core simplification.

Implementation: take the existing GPU-path code (broadcast +
reduction) and make it the only path. Works identically on CPU —
Julia's broadcast and reduction operators are backend-agnostic.

Concretely, in `_x_subcycling_pass_count`:

```julia
# AFTER (this plan):
function _x_subcycling_pass_count(am, m, ws, cfl_limit; max_n_sub=4096)
    isinf(cfl_limit) && return 1
    Nx = size(m, 1)
    # Broadcast + reduction — works on CPU and GPU identically
    out = max.(@view(am[1:Nx, :, :]), zero(eltype(am))) .+
          max.(.- @view(am[2:Nx+1, :, :]), zero(eltype(am)))
    static_cfl = maximum(out ./ max.(m, eps(eltype(m))))
    n_sub = _subcycling_pass_count(static_cfl, cfl_limit)
    n_sub <= max_n_sub ||
        throw(ArgumentError("x-direction subcycling exceeded max_n_sub=$(max_n_sub)"))
    return n_sub
end
```

8 lines, one algorithm. No `isa Array` check.

**Decision 2: Verify CFL estimate similarity BEFORE unifying.**

Critical: the static estimate is different from the evolving one.
Before removing the evolving path, add a diagnostic test that
compares them on realistic flows. If the ratio is wildly off
(>2×), the static estimate is not a drop-in replacement.

See Commit 1 below. This test runs before any CFL code is
changed, so the refactor halts cheaply if the premise is wrong.

**Decision 3: When to keep a `synchronize(backend)`.**

A synchronize call STAYS if:
1. The next operation is host-side array access (reading device
   memory, conservation diagnostic, etc.)
2. It's at the boundary of `apply!` / `strang_split!` (so the
   caller sees the result immediately)
3. It's inside LinRood or CubedSphereStrang across panel
   boundaries (those paths reuse buffers across panels, not a
   sweep-to-sweep ping-pong)
4. It's in `HaloExchange.jl`, `MassCFLPilot.jl`, or
   `VerticalRemap.jl` — those are separate operators, not in
   this plan's scope

A synchronize call GOES if:
1. It's between two sweeps in the same Strang palindrome
2. It's between kernel launches on the same stream where the
   next kernel reads what the previous kernel wrote

**Applied to `StrangSplitting.jl` specifically:**
- The `synchronize(backend)` at the end of each `sweep_x!` /
  `sweep_y!` / `sweep_z!` generated function → **DELETE**
- The `synchronize(backend)` in `_sweep_horizontal_face_gpu!`
  and `_sweep_vertical_face_gpu!` at lines ~304 and ~319 →
  **DELETE** (same reasoning as sweep_x/y/z)
- Any synchronize at the END of `strang_split!` (if it exists
  explicitly) → **KEEP**

**Applied to `LinRood.jl`:**
- `LinRood.jl:73` (after divergence damping kernel) — likely
  **DELETE** if the next kernel launches on the same backend;
  but the comment about panel-loop buffer reuse at line 709
  suggests this path has multi-panel ordering. Err on the side
  of KEEPING and log for future investigation.
- `LinRood.jl:709` (`# required: ws.rm_buf/m_buf reused across
  panels`) — **KEEP**, comment is accurate
- Other LinRood syncs — **KEEP** unless there's a very obvious
  case and no panel-loop adjacency. Plan 13 is conservative
  about LinRood.

When in doubt: KEEP the sync, log in NOTES.md as "deferred —
revisit in a future perf pass."

**Decision 4: Rename is a pure textual change, no API change.**

The field goes from `rm_buf` to `rm_A`. Users who accessed
`ws.rm_buf` directly will get a `type DoesNotHaveField error`.
This is intentional — plan 11 added the compat shim with the
understanding it would be removed here.

No deprecation warnings. No compat alias. Clean break.

**Decision 5: Delete `MassCFLPilot.jl` if unused.**

After Commit 2 deletes the evolving-mass pilot, the `max_cfl_x`,
`max_cfl_y`, `max_cfl_z` helpers may have no callers. Grep:

```bash
grep -rn "max_cfl_x\|max_cfl_y\|max_cfl_z" src/ test/ --include="*.jl"
```

If only callers are in the code being deleted: delete
`MassCFLPilot.jl`, remove its `include` from `Advection.jl`,
remove exports.

If there are other callers (e.g., diagnostics, tests): keep
the file, but inside it simplify — its current implementation
does evolving-mass logic; replace with the static broadcast version.

**Decision 6: NOTES.md is Commit 0, mandatory.**

Learning from plan 11: the agent treated NOTES.md as optional
and had to be reminded. This time, NOTES.md creation IS Commit 0.
No code change precedes it. The file must exist before any
refactor work begins, and it gets updated at least once per
subsequent commit (even just "Commit N done, nothing surprising").

## 4.4 Atomic commit sequence

### Commit 0: Create NOTES.md

**File:** `artifacts/plan13/NOTES.md` (NEW)

Create with this exact template (do not skip this):

```markdown
# Plan 13 Execution Notes — Sync Removal + CFL Unification + Rename

Plan: `docs/plans/13_SYNC_AND_CLEANUP_PLAN.md`
Started: [date]
Baseline commit: [from artifacts/plan13/baseline_commit.txt]

## Decisions made beyond the plan
(Add entries as you make choices not covered by §4.3)

## Deferred observations
(Note anything broken/suboptimal that you spot but do NOT fix)

## Surprises vs. the plan
(Places where reality differs from what the plan described)

## Open questions
(Anything unclear — review with a human)

## Test anomalies
(Any test that behaved oddly, even if it eventually passed)

## Benchmark results
(Copy from each benchmark run)

## Commit log
- Commit 0: NOTES.md created
```

Commit with message "plan 13: add execution notes skeleton".

Update NOTES.md at least once per subsequent commit. Even a one-line
"Commit N done, nothing surprising" is fine.

### Commit 1: Add CFL estimate diagnostic test

Before changing any CFL code, add a test that captures current
behavior. This is the premise check — if it fails, the whole
plan halts.

**File:** `test/test_advection_kernels.jl`

Add this test (adapt names to current module structure):

```julia
@testset "CFL pilot estimate bounds (plan 13 premise check)" begin
    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"
        grid, m, _, _, am, bm, cm = build_test_problem(FT)
        ws = AdvectionWorkspace(m)

        # Static estimate (the target behavior — what plan 13 will
        # become the only algorithm)
        Nx = size(m, 1)
        out_x = max.(@view(am[1:Nx, :, :]), zero(FT)) .+
                max.(.- @view(am[2:Nx+1, :, :]), zero(FT))
        static_cfl_x = maximum(out_x ./ max.(m, eps(FT)))

        # Evolving estimate (current CPU behavior)
        evolving_cfl_x = Operators.Advection.max_cfl_x(am, m, ws.cluster_sizes)

        # Static should be ≥ evolving (by construction)
        @test static_cfl_x >= evolving_cfl_x - 10 * eps(FT)
        # And not wildly different — log ratio for inspection
        ratio = static_cfl_x / max(evolving_cfl_x, eps(FT))
        println("    $precision_tag x-CFL static/evolving = $(ratio)")
        @test ratio <= 2.0  # if this fails, the static estimate
                            # is not a drop-in replacement
    end
end
```

Run:
```bash
julia --project=. test/test_advection_kernels.jl
```

Expected: test passes. Ratios printed should be close to 1 for
realistic synthetic flows.

**If the ratio exceeds 2 for any scheme/precision:** STOP. The
static estimate is pathologically different from evolving in this
codebase. This isn't a plan-13 bug — it's the premise being wrong.
Write the observation in NOTES.md, commit the test (it's still
valuable), and escalate to a human. Do not proceed to Commit 2.

**If the test passes:** commit with message "plan 13: add CFL
estimate diagnostic test". Update NOTES.md: "Commit 1: CFL ratio
diagnostic passes; static/evolving ratio ~1.0 across schemes."

### Commit 2: Unify CFL pilot to static algorithm

For each of the 5 pilot functions, replace the implementation
with the static broadcast (per Decision 1). Line numbers given
are post-plan-11; re-grep to confirm:

- `_x_subcycling_pass_count` (line ~957)
- `_y_subcycling_pass_count` (line ~1007)
- `_z_subcycling_pass_count` (line ~1055)
- `_horizontal_face_subcycling_pass_count` (line ~797)
- `_vertical_face_subcycling_pass_count` (line ~869)

Each becomes ~8 lines (vs. ~40 today). Total deletion: ~150 lines.

Also delete the CPU-only internal helpers
`_horizontal_face_subcycling_pass_count_host` and
`_vertical_face_subcycling_pass_count_host` (lines ~703-795) —
they were the CPU path's support functions.

Delete references to `max_cfl_x`, `max_cfl_y`, `max_cfl_z` in
`StrangSplitting.jl`. Then grep to see if those helpers are used
elsewhere:

```bash
grep -rn "max_cfl_x\|max_cfl_y\|max_cfl_z" src/ test/ --include="*.jl"
```

Act on Decision 5: delete `MassCFLPilot.jl` entirely if unused,
or leave it and simplify if still referenced.

**Test after Commit 2:**
```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass (modulo the 77 pre-existing failures
captured in Commit 0's baseline). The diagnostic test from
Commit 1 should still show ratios close to 1.

Run benchmark:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium cpu \
    > artifacts/plan13/perf/after_commit2_cpu_medium.log
```

Expected: no significant change (±5%). CFL unification is not
performance-focused. If there's a regression, the static estimate
might be over-subcycling dramatically — check via NOTES.md.

Commit with message "plan 13: unify CFL pilot to static algorithm".

### Commit 3: Remove synchronize calls from sweeps

In `StrangSplitting.jl`, find every `synchronize(backend)` call:

```bash
grep -n "synchronize(backend)" src/Operators/Advection/StrangSplitting.jl
```

For each occurrence, apply Decision 3's rules. In the
`StrangSplitting.jl` scope specifically:

- Inside sweep function body (`sweep_x!`, `sweep_y!`, `sweep_z!`
  generated variants and their flux-scale overloads) just before
  `return nothing` → **DELETE**
- Inside `_sweep_horizontal_face_gpu!` (line ~304), just before
  the return → **DELETE**
- Inside `_sweep_vertical_face_gpu!` (line ~319), just before
  the return → **DELETE**

Total expected deletions in `StrangSplitting.jl`: ~6-8 lines.

Do NOT touch syncs in:
- `LinRood.jl` (different plan; some are load-bearing)
- `CubedSphereStrang.jl` (different plan)
- `HaloExchange.jl` (different operator)
- `MassCFLPilot.jl` (if still exists)
- `VerticalRemap.jl` (different operator)

**Test after Commit 3:**
```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass.

Critically, ULP-tolerance CPU-GPU tests should still pass — sync
removal is a host-side optimization; kernel arithmetic is identical.
The tests at `test_advection_kernels.jl` lines 215-255 (1-step and
4-step CPU-GPU agreement) are the best verification.

Run benchmark:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium cpu \
    > artifacts/plan13/perf/after_commit3_cpu_medium.log
```

Expected: 0-5% improvement on CPU.

If GPU access is available, measure GPU:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium gpu \
    > artifacts/plan13/perf/after_commit3_gpu_medium.log
```

Expected: **20-40% per-step improvement**. This is the big win.

If GPU improvement is <10%, STOP and investigate before proceeding:
- Verify the sync calls were actually removed (`grep -n synchronize`)
- Check if another synchronization is hidden somewhere the plan
  didn't anticipate

Commit with message "plan 13: remove redundant synchronize calls
from structured sweeps".

### Commit 4: Rename buffer fields, remove compat shim

In `StrangSplitting.jl`:
- Remove the `Base.getproperty` method on `AdvectionWorkspace`
  that aliases `rm_buf → rm_A`, `m_buf → m_A`, `rm_4d_buf → rm_4d_A`
- Any internal references to `ws.rm_buf`, `ws.m_buf`,
  `ws.rm_4d_buf` in `StrangSplitting.jl` (should be rare
  post-plan-11) get updated

In `LinRood.jl`:
- Line 71: `ws.rm_buf` → `ws.rm_A`
- Line 74: `ws.rm_buf` → `ws.rm_A`
- Line 705: `ws.rm_buf, ws.m_buf` → `ws.rm_A, ws.m_A`
- Line 710: `ws.rm_buf` → `ws.rm_A`
- Line 711: `ws.m_buf` → `ws.m_A`
- Re-verify line numbers — the file may have shifted slightly

In `CubedSphereStrang.jl`:
- Line 460: `workspace.rm_buf, workspace.m_buf` → `workspace.rm_A, workspace.m_A`
- Line 553: `ws.rm_buf, ws.m_buf` → `ws.rm_A, ws.m_A`

In `test/test_basis_explicit_core.jl`:
- Line 326: `model_host.workspace.rm_buf` → `model_host.workspace.rm_A`
- Line 333: `model_gpu.workspace.rm_buf` → `model_gpu.workspace.rm_A`

**Test after Commit 4:**

Verify the old names are gone:
```bash
grep -rn "rm_buf\|m_buf\|rm_4d_buf" src/ test/ --include="*.jl"
```

Expected: zero matches.

Run tests:
```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass.

Commit with message "plan 13: rename workspace buffers to rm_A/m_A,
remove compat shim".

### Commit 5: Documentation and final check

- Update `StrangSplitting.jl` docstring on `AdvectionWorkspace` to
  reflect final field names (no `rm_buf`, just `rm_A`/`rm_B`)
- Update any comments that reference the old dual CFL algorithm
  or old buffer names
- Update `CLAUDE.md` invariant #4 if needed (plan 11 should have
  handled ping-pong; this plan should verify the text is current)

Final verification:
```bash
# All tests pass (modulo baseline pre-existing failures)
julia --project=. test/runtests.jl 2>&1 | tee artifacts/plan13/after_tests.log
# Should have same or fewer failures than baseline

# Diff baseline vs current
diff artifacts/plan13/baseline_failures.log \
     <(grep -E "Got exception|Test Failed|Error During" artifacts/plan13/after_tests.log)
# Expected: diff is empty or shows only plan-13-deliberate additions
# (e.g., the diagnostic test added in Commit 1)
```

Final benchmark:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl small cpu \
    > artifacts/plan13/perf/final_cpu_small.log
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium cpu \
    > artifacts/plan13/perf/final_cpu_medium.log
# If GPU available:
# julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium gpu \
#     > artifacts/plan13/perf/final_gpu_medium.log
```

Write a retrospective section in NOTES.md with:
- Actual speedup achieved on CPU and GPU per scheme
- Any deviations from the plan
- Any deferred observations for plans 12/14
- Lessons for future plans

Line count check:
```bash
wc -l src/Operators/Advection/StrangSplitting.jl
```
Expected: 150-200 lines shorter than post-plan-11 baseline.

Commit with message "plan 13: update docs and retrospective notes".

## 4.5 Test plan per commit

After each commit:

```bash
# 1. Compile
julia --project=. -e 'using AtmosTransport'

# 2. Core suite
julia --project=. test/runtests.jl

# 3. ULP tests specifically (sync removal shouldn't affect ULP,
# but verify — this is the highest-sensitivity check)
julia --project=. test/test_advection_kernels.jl

# 4. Cubed-sphere (touched by rename in Commit 4)
julia --project=. test/test_cubed_sphere_advection.jl

# 5. GPU tests if available
HAS_GPU=true julia --project=. test/test_advection_kernels.jl
```

Compare test output against baseline:
```bash
diff artifacts/plan13/baseline_failures.log \
     <(grep -E "Got exception|Test Failed|Error During" CURRENT_TEST_OUTPUT)
```

The diff should show:
- Same pre-existing failures (unchanged through refactor)
- NO new failures (this is the hard requirement)

Stop conditions:
- NEW test failure (not in baseline): STOP, revert, investigate
- ULP tolerance exceeds limits: STOP, revert — shouldn't happen
  for sync removal or rename; would indicate CFL unification has
  a subtle bug
- Benchmark REGRESSES on GPU: STOP, investigate — sync removal
  should speed up, not slow down
- GPU speedup is <10% after Commit 3: investigate before proceeding

## 4.6 Acceptance criteria

**Correctness:**
- All baseline-passing tests still pass
- ULP tolerances unchanged (≤4 for 1-step, ≤16 for 4-step both
  FT)
- Mass conservation unchanged (1e-12 Float64, 5e-5 Float32)
- No new warnings

**Code cleanliness (hard requirements):**
- `grep -rn "rm_buf\|m_buf\|rm_4d_buf" src/ test/ --include="*.jl"`
  returns zero matches
- `grep -rn "getproperty.*AdvectionWorkspace" src/` returns zero
  matches (shim deleted)
- Exactly ONE CFL pilot algorithm per direction (the static one)
- `StrangSplitting.jl` is ≥150 lines shorter than post-plan-11
  baseline

**Performance (updated from v1 with plan 11 learnings):**
- GPU per-step: **≥20% improvement over post-plan-11 baseline**
  (big expected win from sync removal). If <10%, STOP.
- CPU per-step: within ±5% of post-plan-11 baseline (sync
  removal is GPU-focused; CPU may be flat or slightly better)

**Convergence to architectural sketch:**
- `MassCFLPilot.jl` deleted OR significantly reduced
- Clean buffer names throughout (no compat shim)
- Ready for plan 12 (scheme consolidation) to delete legacy
  sweep variants cleanly

## 4.7 Rollback plan

Standard: revert, don't fix forward, write in NOTES.md, ask if
stuck >30 min.

Specific rollback concerns:

- **Commit 1 (diagnostic test) fails ratio check:** this means
  the static estimate is NOT a drop-in replacement. Do NOT revert
  the test — it's a valuable finding. But halt the plan at this
  point. The CFL unification requires a different approach
  (possibly keeping evolving-mass but making it backend-agnostic).
  Write up the finding in NOTES.md, commit the test, and escalate.

- **Commit 2 (CFL unification) causes test regression:** the
  static algorithm has a bug somewhere. Revert Commit 2. The
  diagnostic test from Commit 1 passed, so we know static ≈
  evolving; the issue must be in the implementation. Review
  carefully.

- **Commit 3 (sync removal) causes test flakiness:** some sync
  was load-bearing (masking a race condition). Revert. It's
  probably a sync that looks cosmetic but is actually required
  for ordering between independent streams. Add back carefully.

- **Commit 4 (rename) breaks a test you didn't expect:**
  something outside the known call-site list was using the old
  names. `grep -rn` more broadly, including `docs/`, `scripts/`,
  `artifacts/`. Fix and re-test.

## 4.8 Known pitfalls

1. **"I'll just remove ALL synchronizes while I'm here."**
   NO. Some sync calls are load-bearing (across panel operations
   in LinRood, at apply! boundaries for caller visibility, inside
   halo exchange). Follow Decision 3 strictly. When in doubt,
   KEEP the sync and log in NOTES.md.

2. **"The CFL ratio test is annoying; I'll delete it."**
   After Commit 2 ships, the test is less useful (only one
   algorithm now). Either delete it in Commit 2 OR keep as a
   regression guard. Either is fine; don't silently drop it.

3. **"Renaming breaks a test but I can just suppress the error."**
   NO. If a test references `rm_buf`, update it. Don't try to
   keep the old name alive via a secondary shim.

4. **"The CFL unification caused n_sub=5 where it used to be
   n_sub=3."** Check if this is only at extreme CFL (near the
   limit). Slight over-subcycling is expected; bit-identical
   output between old and new CFL is NOT required (it's a
   semantic change). If the physical answer is preserved
   (conservation OK, error norms OK), accept.

5. **"Some `synchronize` call has a comment saying 'required'."**
   Read the comment carefully. In `LinRood.jl:709` the comment
   `# required: ws.rm_buf/m_buf reused across panels` describes
   a REAL panel-loop requirement that remains valid post-plan-11.
   Do NOT delete it. The comment was accurate.

6. **"MassCFLPilot has only internal callers; delete it."**
   Check tests carefully. Some test files might use
   `AtmosTransport.max_cfl_x` for diagnostics. Grep includes
   test files.

7. **"GPU speedup is only 10-15%, not 20-40%."**
   Don't ship — investigate first. Possibilities:
   - Hidden sync somewhere the plan didn't catch (grep for
     `CUDA.synchronize`, `CuArray` explicit synchronizations)
   - L40S has different characteristics than the A100 the plan
     was calibrated for; different GPUs have different sync
     overheads
   - Kernel arithmetic dominates more than expected at the test
     problem size
   Log the finding in NOTES.md. If the speedup is real and
   positive (≥10%), probably OK to ship with a note explaining
   the shortfall.

8. **"Tempting to ALSO delete legacy scheme types while I'm here."**
   NO. That's plan 12 (next). Scope discipline. Plan 13 affects
   scheme-independent plumbing (sync, CFL, buffer names). Legacy
   types are plan 12's job.

---

# Part 5 — How to Work

Same as plan 11. Session cadence:

- **Session 1:** Precondition, Commit 0 (NOTES.md), Commit 1
  (diagnostic test)
- **Session 2:** Commit 2 (CFL unification) — biggest single
  change, save for fresh eyes
- **Session 3:** Commit 3 (sync removal)
- **Session 4:** Commit 4 (rename)
- **Session 5:** Commit 5 (documentation + retrospective)

## 5.1 Worktree benchmark pattern (from plan 11)

For clean before/after benchmarks:

```bash
# Capture baseline from specific commit without disturbing working tree
git worktree add /tmp/atmos_baseline BASELINE_SHA
cp Manifest.toml /tmp/atmos_baseline/  # if needed
cd /tmp/atmos_baseline
julia --project=. scripts/benchmarks/bench_strang_sweep.jl medium gpu \
    > $MAIN_REPO/artifacts/plan13/perf/baseline_gpu_medium.log
cd $MAIN_REPO
git worktree remove /tmp/atmos_baseline --force
```

Use this for any before/after where you need to switch between
the pre-change and post-change state cleanly.

## 5.2 NOTES.md discipline

Update NOTES.md **at least once per commit**. Even a one-liner
"Commit N completed, all tests pass, no surprises" is useful.

The structured sections (Decisions / Deferred / Surprises / Open
Questions / Test anomalies / Benchmarks / Commit log) make it
easy for the reviewer (human or another Claude session) to
extract signal at pause points.

---

# End of Plan

After this refactor:
- `StrangSplitting.jl` is substantially shorter and linear
- Buffer names (`rm_A`, `m_A`, etc.) reflect their actual role
- ONE CFL algorithm, ONE synchronization discipline
- 20-40% per-step GPU speedup over post-plan-11 baseline
- Cumulative 25-45% speedup over pre-plan-11 baseline

The next plan (`12_SCHEME_CONSOLIDATION_PLAN.md`) deletes the
legacy scheme types (`UpwindAdvection`, `RussellLernerAdvection`)
and the `@eval` sweep generators that route to them. That work
builds on plan 13's cleanup: with sync/CFL/rename settled,
deleting the legacy types is pure subtraction.
