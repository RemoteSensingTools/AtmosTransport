# Ping-Pong Refactor — Execution Notes

Retrospective write-up after completing all 7 commits. Plan:
[11_PINGPONG_REFACTOR_PLAN.md](../../11_PINGPONG_REFACTOR_PLAN.md).
Commits on `restructure/dry-flux-interface`: `8b8d283..527eef0`.

## Decisions made beyond the plan

1. **Kept the union-typed `strang_split!` signature rather than splitting
   into two methods.** Plan §4.3 Decision 1 allowed either. I kept the
   existing `scheme::Union{AbstractAdvection, AbstractAdvectionScheme}`
   method and made the new ping-pong entry points work for both scheme
   hierarchies — so legacy UpwindAdvection / RussellLernerAdvection
   paths also get the ping-pong win, at zero cost. Tradeoff: one more
   `@eval` block for each legacy variant; benefit: no dead code
   duplication and both scheme families benefit from the refactor.

2. **Merged Commit 5 (LinRood/CS audit) into Commit 6 (docs).** Audit
   found both callers already work unchanged via the `getproperty`
   shim and `CubedSphereStrang.jl` uses its own separate
   `CSAdvectionWorkspace` struct (never touched). Since there were no
   code changes for Commit 5, it became a paragraph in the Commit 6
   message rather than a no-op empty commit. Net: 6 code commits + 1
   GPU follow-up = 7 total, same as the plan.

3. **Added GPU backend support to the benchmark as a follow-up.** Plan
   had the benchmark CPU-only by default. User asked for GPU numbers
   on wurst (L40S), so I extended the benchmark with an optional
   `gpu` argument and captured before/after on the L40S. Worktree
   trick (`git worktree add`) was used for the baseline measurement
   to avoid tree-disturbing `git checkout`.

4. **For `strang_split_mt!` ping-pong, used an inner `@inline` helper
   `_pass!` that returns the swapped tuple** rather than writing six
   copies of `if isodd(n); swap; end`. Cleaner, same codegen. Not
   covered by the plan but equivalent to the pattern plan §4.4
   Commit 3 uses for `strang_split!`.

## Deferred observations

These are real but outside scope for plan 11; they should inform
plans 12-14.

1. **`synchronize(backend)` after every kernel is the dominant GPU
   cost.** On L40S F32 medium, per-step time is ~3 ms. The eliminated
   copyto savings total ~0.18 ms, about 6%, even though the
   theoretical bandwidth ceiling is ~0.9 ms. The gap is the
   synchronize overhead — this is exactly what plan 13 (sync
   removal) targets.

2. **Workspace memory allocates both pairs even when `n_tracers=0`.**
   The 4D buffers correctly drop to 0×0×0×0 when no tracers are
   declared (per existing logic), but the 3D A/B pairs always
   allocate. At small problem sizes this is fine; at full scale
   (C720 × 30 tracers) it adds ~224 MB. Optional `rm_B` (via
   `Union{Nothing, A}` or a `NoOp` sentinel) was explicitly
   rejected by plan §4.9 pitfall #5 as "type instability +
   special-case code paths." If memory ever becomes a constraint,
   batching (advect N tracers at a time, reuse 4D buffers) is the
   plan-documented tier-3 fix.

3. **`Base.getproperty` on `AdvectionWorkspace` uses three sequential
   `name === :` checks.** At ~nanoseconds per check this is
   negligible, but if any future hot path accesses `ws.rm_buf` in
   a tight inner loop, replacing the shim with direct `rm_A`
   references would save a few ns per call. No current caller
   does this — LinRood and CS touch `ws.rm_buf` only once per
   panel per sweep.

4. **Legacy scheme sweeps (UpwindAdvection, RussellLernerAdvection)
   now have a new 7-arg / 8-arg form each**, but nothing calls them
   — `strang_split!` uses the 5-arg / 6-arg backward-compat
   wrappers for the legacy path since those were the entry points
   the per-tracer pre-refactor code used. After plan 13 or 14 the
   legacy types themselves can be removed entirely, collapsing
   four `@eval` blocks into two. Not in scope for plan 11.

## Surprises vs. the plan

1. **CPU speedup is smaller than the plan expected** (2-8% vs ≥10%
   target). Root cause: at this problem size, the kernel arithmetic
   dominates total time for Slopes / PPM, so the copyto fraction is
   smaller than the plan's bandwidth analysis assumed. Upwind
   (bandwidth-bound kernel) is the only scheme that hits the
   plan's target. The correct framing is: ping-pong eliminates a
   fixed bandwidth cost; the PERCENTAGE impact depends on what
   fraction of per-step time that cost represents. The measurement
   should drive the number, not the plan's a-priori estimate.

2. **Test file `test_basis_explicit_core.jl` has 2 pre-existing
   failures** (lines 211-212, CubedSphereMesh API) that block
   `runtests.jl` from running downstream files via its `include`
   pattern. Worked around by running each core test file
   individually. Unrelated to this refactor; flagged for future
   cleanup (could be a separate PR).

3. **Workspace field renaming concern was moot.** Plan §4.3 Decision
   4 warned that renaming `rm_buf` → `rm_A` could cause test
   failures. In practice the `getproperty` shim made the rename
   invisible to callers and the two type-check asserts in
   `test_basis_explicit_core.jl:326,333` continue to pass
   unchanged. The shim was the right call.

4. **Worktree approach for baseline GPU measurement worked
   cleanly** — `git worktree add /tmp/atmos_baseline 8b8d283`,
   copy Manifest.toml and the bench script over, run, remove
   with `--force`. No stash juggling, no partial checkouts of
   the primary worktree. Recommend this as the standard pattern
   for before/after perf comparisons in plans 12-14.

## Template usefulness for plans 12-14

This plan's structure worked well:

  * **Pre-committed design decisions (§4.3)** — answered the
    judgment calls up front, so execution stayed mechanical.
  * **Atomic commit sequence (§4.4)** — each commit independently
    compilable + testable + revertable. Made rollback a clean
    `git reset --hard HEAD~1`, never needed it.
  * **Explicit non-goals (§3, §4.9)** — kept scope narrow. Every
    "tempting" adjacent cleanup deferred to a later plan.
  * **Benchmark-first (Commit 0)** — baseline captured before
    any code changed, so every later measurement was comparable.

Recommend reusing this template for plans 12 (scheme consolidation),
13 (sync removal), and 14 (single pipeline). Carry-over
prerequisites:

  * Plan 13 depends on the ping-pong separation that this plan
    provides (can't remove `synchronize(backend)` if dst == src).
  * Plan 13's GPU win is expected to be the big one — 20%+ in
    the plan — building on plan 11's groundwork.

## Test anomalies

None from this refactor. The 2 pre-existing `test_basis_explicit_core`
and 3 pre-existing `test_structured_mesh_metadata` and 72 pre-existing
`test_poisson_balance` failures observed in baseline remain
unchanged through all 7 commits. They're not caused by and not
affected by ping-pong.

## Benchmark results

### CPU medium (single thread, 288×144×32, 5 tracers)

                 before (ms)   after (ms)    Δ
  F64 Upwind      458.686      423.002      -7.8%
  F64 Slopes     2564.061     2510.517      -2.1%
  F64 PPM        2327.489     2272.198      -2.4%
  F32 Upwind      395.660      379.025      -4.2%
  F32 Slopes     2421.618     2400.191      -0.9%
  F32 PPM        2175.612     2159.508      -0.7%

### GPU medium (L40S F32, 288×144×32, 5 tracers)

                 before (ms)   after (ms)    Δ
  Upwind            3.223        3.030      -6.0%
  Slopes            3.317        3.130      -5.6%
  PPM               3.483        3.305      -5.1%

GPU measurement note: L40S has no F64 units; F32 only. Before
captured via `git worktree add 8b8d283` to keep measurement clean.
