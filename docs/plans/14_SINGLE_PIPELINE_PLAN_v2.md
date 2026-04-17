# Advection Pipeline Unification — Implementation Plan for Plan Agent (v2)

**Status:** Ready for execution after plans 11, 12, 13 have shipped.
**Target branch:** new branch from wherever plans 11/12/13's shipped
  work lives (typically `restructure/dry-flux-interface` or its
  successor). Verify in §4.1.
**Estimated effort:** 2-3 weeks, single engineer / agent.
**Primary goal:** Restructure `strang_split!` to eliminate the
  per-tracer loop, moving to a single 4D tracer array and unified
  multi-tracer kernels. Ship a cleaner pipeline that matches the
  operator interface specified in `OPERATOR_COMPOSITION.md`.

Self-contained document. ~30 minutes to read.

**Dependencies verified:**
- Plan 11 (ping-pong) shipped
- Plan 12 (scheme consolidation) shipped — single scheme hierarchy
- Plan 13 (sync + CFL + rename) shipped — unified CFL pilot, clean
  workspace field names

**Companion documents:**
- `OPERATOR_COMPOSITION.md` — cross-operator architecture reference.
  Plan 14 MUST produce a data layout and operator interface
  consistent with this doc. Read §4 (data layout) and §6 (operator
  interface) before executing this plan.
- `ARCHITECTURAL_SKETCH.md` — advection subsystem shape. Describes
  the internal structure plan 14 should converge toward.
- `CLAUDE.md` § "Performance tips" — the "measure, don't subtract"
  lesson from plan 13 applies here.

---

# Revisions from v1 (based on plan 11, 12, 13 execution)

What changed between v1 and this v2:

1. **MEASUREMENT-FIRST COMMIT 0 is now mandatory.** Plan 13 taught
   us that performance predictions by subtraction from theoretical
   ceilings are unreliable. The ping-pong plan (11) predicted
   20-35% GPU speedup and delivered 5-6%. The sync removal plan
   (13) predicted 20-40% and delivered <1% (sync wasn't the
   bottleneck at all). Plan 14 v1 predicted 2-10× for multi-tracer
   fusion. That number is almost certainly wrong for the same
   reason: launch overhead may not be the dominant cost at
   production problem sizes. Commit 0 of this plan measures
   directly BEFORE any code change.

2. **Framing reversed: cleanup-motivated, not performance-motivated.**
   The real value of plan 14 is code cleanliness — one pipeline,
   not two; one data structure, not a NamedTuple of arrays; one
   interface, consistent with future operators. Performance
   improvements are a POSSIBLE SIDE EFFECT, not the goal. If
   Commit 0's measurement shows the multi-tracer fusion gives
   <5% win at production settings (with `cfl_limit=Inf`, large
   problem, production tracer count), the refactor still ships
   as pure cleanup and we accept that.

3. **Alignment with OPERATOR_COMPOSITION.md required.** Plan 14
   produces the data layout and operator interface that the
   composition doc specifies. Specifically:
   - Tracers: 4D array `Q[i, j, k, t]` with companion
     `tracer_names::Vector{Symbol}`
   - Interface: `apply!(state, meteo, grid, operator, dt;
     workspace)` signature
   - Workspace: per-operator struct, groupable in a larger
     `TransportWorkspace` container

4. **Mass-restore copy elimination flagged as potentially bigger
   than launch reduction.** Plan 11 measurement revealed that
   bandwidth-bound savings scale with the fraction of per-step
   time spent on that bandwidth. For multi-tracer, the per-tracer
   loop does N-1 `copyto!(m, m_save)` calls per step. For 30
   tracers at C180, that's ~3.2 GB of bandwidth per step. On
   L40S (1.5 TB/s bandwidth) that's ~2 ms, possibly the biggest
   fusion win on large problems. Commit 0 should measure this
   separately.

5. **Benchmark methodology tightened.** Following plan 13's
   pattern: median ± MAD statistics (not mean ± std), optional
   CUDA event timing for per-sweep cost, larger default problem
   for GPU measurement. Worktree-based baseline capture.

6. **Acceptance criteria reframed.** v1 required "per-step
   wall time 2-10× faster." v2 requires "per-step wall time
   not regressed (within ±5% of baseline at worst)." If there's
   a genuine perf win, great. If not, the cleanup is still the
   deliverable.

7. **NOTES.md as Commit 0** (matches pattern from plans 12, 13).

---

# Part 1 — Orientation

## 1.1 The problem in one paragraph

`CellState.tracers` is currently a `NamedTuple` of 3D arrays, one
per tracer. `strang_split!` iterates over tracers, running the
full 6-sweep advection pipeline for each. This means:

- N × 6 kernel launches per step (one per tracer per direction)
- N-1 mass restorations per step (each tracer needs the same
  initial mass field, so m gets restored between tracers)
- Two parallel code paths: per-tracer (for NamedTuple tracers)
  and multi-tracer (for 4D tracer arrays used in some tests)

Plan 14 unifies these into a single multi-tracer pipeline:

- CellState.tracers becomes `Array{FT, 4}` with shape `(Nx, Ny, Nz, Nt)`
- `tracer_names::Vector{Symbol}` tracks the names
- `strang_split!` calls 6 multi-tracer sweeps (one per direction)
  that process ALL tracers in a single kernel launch
- The per-tracer loop is gone

## 1.2 Why this is cleanup-motivated, not perf-motivated

Plan 14 v1 predicted 2-10× speedup from kernel launch reduction.
Let's check that math against plan 13's measurements:

- GPU large (576×288×72, 10 tracers) per-step time: ~47 ms
- Kernel launches per step with per-tracer loop: 10 × 6 = 60
- Launch overhead per kernel: ~20 μs on L40S
- Total launch overhead: 60 × 20 μs = 1.2 ms
- **Launch overhead as fraction of total: 2.5%**

So the naive "reduce launches by 10×" gives at most 2.5% savings,
not 2-10×. The 2-10× prediction was based on small-problem regimes
where arithmetic is tiny and launches dominate — which isn't
where production runs.

Other potential wins that MIGHT be bigger at production scale:
- **Mass-restore copyto elimination.** 29 restores × 112 MB per
  restore (C180, 30 tracers) = 3.2 GB bandwidth. On L40S, ~2 ms.
  Roughly 4% of 47 ms per-step. Small but measurable.
- **Better GPU occupancy.** Multi-tracer kernels have more work
  per launch, potentially better warp utilization. Unmeasured.
- **Smaller code path.** Fewer conditionals, fewer dispatches,
  more straight-line code. Probably 0-3% depending on compiler.

**Realistic expectation: 0-10% improvement at production settings.**
Could be smaller if bandwidth/arithmetic already saturates.

Therefore plan 14 is primarily about CODE CLEANLINESS. Two
pipelines becomes one. NamedTuple of arrays becomes one 4D array.
The operator interface aligns with the composition doc.
Performance is a possible bonus, not the goal.

## 1.3 What "clean" means, concretely

### Data layout

Before:
```julia
struct CellState{...}
    m::Array{FT, 3}            # (Nx, Ny, Nz)
    tracers::NamedTuple        # {CO2::Array{FT,3}, CH4::Array{FT,3}, ...}
end
```

After:
```julia
struct CellState{...}
    m::Array{FT, 3}               # (Nx, Ny, Nz)
    tracers::Array{FT, 4}         # (Nx, Ny, Nz, Nt)
    tracer_names::Vector{Symbol}  # length Nt
end
```

This matches `OPERATOR_COMPOSITION.md` §4 exactly.

### Pipeline

Before:
```julia
function strang_split!(state, fluxes, grid, scheme, dt; workspace)
    for (tracer_name, tracer) in pairs(state.tracers)
        m = copy(state.m)
        # 6 sweeps (X Y Z Z Y X) for this one tracer
        sweep_x!(tracer, m, fluxes, ...)
        # ...
        # restore m for next tracer
    end
end
```

After:
```julia
function strang_split!(state, fluxes, grid, scheme, dt; workspace)
    # ONE call per direction, processes all tracers
    sweep_x!(state.tracers, state.m, fluxes, ...)
    sweep_y!(...)
    sweep_z!(...)
    sweep_z!(...)
    sweep_y!(...)
    sweep_x!(...)
end
```

The multi-tracer kernels already exist (used in some tests).
Plan 14 wires them as the primary path.

### Interface

Before: `strang_split!(state, fluxes, grid, scheme, dt; workspace)`
After: same signature, but now the one-and-only path.

Eventually (future plans) this generalizes to:
`apply!(state, meteo, grid, operator::Advection, dt; workspace)`
matching the operator interface in OPERATOR_COMPOSITION.md §6.
Plan 14 does NOT rename `strang_split!` to `apply!` — that's a
later plan when multiple operators exist and the dispatch needs
to unify. Plan 14 just establishes the SHAPE that future plans
will rename.

## 1.4 What's going away

- `CellState.tracers::NamedTuple` — replaced by 4D array
- The per-tracer loop in `strang_split!` — replaced by 6 calls
- The `m_save` workspace field — no longer needed (no restore
  between tracers; see §4.4 Commit 3 for details)
- Per-tracer kernel variants (if any duplicate the multi-tracer
  kernels) — collapsed into one

## 1.5 What stays the same

- Kernel arithmetic (`reconstruction.jl`, `limiters.jl`,
  `schemes.jl`) — untouched
- CFL pilot (unified in plan 13) — untouched
- Face-indexed / reduced Gaussian paths — out of scope
- LinRood / CubedSphereStrang — out of scope (they have their
  own workspace and tracer structure)
- Scheme hierarchy (`UpwindScheme`, `SlopesScheme`, `PPMScheme`)
  — unchanged

## 1.6 Test suite discipline

Same approach as plans 11-13:
- Baseline failures captured at Commit 0
- Per-commit test runs compare pass/fail to baseline
- ULP tolerance for advection correctness tests
- Mass conservation must hold exactly

77 pre-existing failures observed through plans 11-13:
- `test_basis_explicit_core.jl` (2 failures)
- `test_structured_mesh_metadata.jl` (3 failures)
- `test_poisson_balance.jl` (72 failures)

Capture the current baseline in §4.1 — these may have shifted.

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before this refactor:
- TWO tracer data layouts (NamedTuple of 3D, Array 4D) — tests use
  one, production uses the other
- TWO pipeline paths (per-tracer loop, multi-tracer direct)
- `m_save` workspace field for restore-between-tracers
- Future operator addition would need to handle both layouts

After this refactor:
- ONE tracer data layout (4D array + names vector)
- ONE pipeline (6 multi-tracer sweeps)
- No `m_save`
- Future operators (diffusion, convection, chemistry) extend the
  same pipeline — they get a 4D tracer array and act on it

## 2.2 What enables downstream work

Plans that benefit from plan 14 shipping:
- **Plan 15 (slow chemistry):** operates on 4D tracer array with
  per-tracer decay constants. Natural fit.
- **Plan 16 (surface flux BCs):** modifies advection Z-sweep
  bottom face. Unified sweep makes this a single-site change.
- **Plan 17 (vertical diffusion):** column solver on 4D tracer
  array. Same layout advection uses.
- **Plan 18 (convection):** mass-flux on columns of 4D array.

All downstream plans assume the composition-doc data layout.
Plan 14 ships that layout.

## 2.3 What this does NOT enable

- Faster production runs (probably — measure first, don't assume)
- New scientific capabilities
- Different meteorology support

This is an internal refactor. User-visible behavior unchanged
except the `CellState.tracers` API (which has no external users
per repo status).

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- **Kernel arithmetic.** Multi-tracer kernels exist; plan 14
  wires them up. Don't change their math.
- **CFL pilot.** Unified in plan 13. Don't touch.
- **`LinRood.jl`, `CubedSphereStrang.jl`.** Out of scope. They
  have separate workspaces and tracer handling. Log any perceived
  issues to NOTES.md.
- **Reconstruction, limiters, schemes.** Out of scope.
- **Face-indexed / reduced Gaussian paths.** Out of scope.

## 3.2 Do NOT add

- **New operator types.** This is advection unification only.
  Diffusion, convection, chemistry are separate future plans.
- **New schemes.** Plan 12 settled the scheme hierarchy.
- **`apply!` dispatch.** The rename from `strang_split!` to
  `apply!` happens when multiple operators exist (plan 15+).
  Plan 14 just establishes the SHAPE.
- **Performance optimizations beyond the refactor.** If you
  notice something during execution, log to NOTES.md as deferred.

## 3.3 Potential confusion — clarified

You will see tests that create tracers as NamedTuples:
```julia
CellState(m; CO2=rm_CO2, CH4=rm_CH4)
```
These constructors continue to work — they internally assemble
the 4D array. User-facing API (keyword arguments for tracer
names) is preserved. Internal representation is 4D.

You will see tests that access `state.tracers.CO2`. This API
BREAKS in plan 14 — `tracers` is now an Array, not a
NamedTuple. Tests that use this pattern must migrate to
`state.tracers[:, :, :, idx]` where
`idx = findfirst(==(:CO2), state.tracer_names)`. Or provide
a `get_tracer(state, :CO2)` accessor function.

**Decision:** provide a `get_tracer(state, name::Symbol)`
accessor for readable test code. See §4.3 Decision 6.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch. Plan 13 shipped onto the refactor
# branch (not main). Branch plan 14 from wherever 13's commits
# are accessible.
git branch -a | head -20
git log --oneline --all | grep -i "plan 13\|sync.*cfl\|cleanup" | head -5
# Find the branch that contains plan 13's commits.
git checkout <parent-branch>
git pull
git log --oneline | head -20
# Expected: plan 11, 12, 13 commits visible.

git checkout -b advection-unification

# 2. Clean working tree
git status
# Expected: "nothing to commit, working tree clean"

# 3. Verify dependency state
grep -c "AbstractAdvection\b" src/ --include="*.jl" -r
# Expected: zero (plan 12 removed it)

grep -c "MassCFLPilot" src/ --include="*.jl" -r
# Expected: zero (plan 13 deleted it)

grep -c "rm_A\|m_A" src/Operators/Advection/StrangSplitting.jl
# Expected: multiple (plans 11, 13 established these)

# 4. Capture baseline failure set
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

# Count pass/fail per file. Should roughly match the post-plan-13
# baseline (77 pre-existing failures). Record exact numbers.

# 5. Record commit hash
git rev-parse HEAD > artifacts/baseline_commit.txt

# 6. Set up benchmark directory
mkdir -p artifacts/perf/plan14
```

If any precondition fails, STOP and investigate.

## 4.2 Commit 0 — THE MEASUREMENT-FIRST STEP

**This commit is the premise check.** Before changing any code,
measure:

1. **Per-tracer loop cost vs. multi-tracer cost on current code.**
   Both code paths exist — per-tracer is production, multi-tracer
   is used in some tests. Benchmark both at multiple (size, Nt,
   cfl_limit) points.

2. **Specifically measure:**
   - CPU medium, F64 and F32, Upwind / Slopes / PPM
   - GPU medium, F32, all three schemes
   - GPU large, F32, all three schemes
   - Each at `Nt = 1, 5, 10, 30` tracers
   - Each with `cfl_limit = 0.4` (default) AND `cfl_limit = Inf`
     (production bypass — CLAUDE.md "Performance tips" shows
     this is what production uses)

3. **Expected finding:** multi-tracer path is faster than per-
   tracer path. By HOW MUCH determines whether plan 14 is a perf
   refactor or a cleanup refactor.

4. **Acceptance criterion for proceeding:**
   - If multi-tracer gives ≥5% win at ANY (size, Nt, cfl_limit)
     point: plan 14 is a CODE CLEANUP + MEASURED PERF WIN. Proceed.
   - If multi-tracer gives <5% win at all points: plan 14 is a
     PURE CLEANUP. Proceed, but reframe NOTES.md / commit messages
     accordingly.
   - If multi-tracer is SLOWER than per-tracer (shouldn't happen
     but possible at some config): STOP. Something is broken in
     the multi-tracer path; fix it first before proceeding.

Execution:
```bash
# Create NOTES.md first (per plan-12/13 pattern)
mkdir -p docs/plans/14_SINGLE_PIPELINE_PLAN
cat > docs/plans/14_SINGLE_PIPELINE_PLAN/NOTES.md << 'EOF'
# Plan 14 Execution Notes — Advection Pipeline Unification

Plan: `docs/plans/14_SINGLE_PIPELINE_PLAN.md` (v2)

## Baseline
Commit: (fill in)
Pre-existing test failures: (fill in from artifacts/baseline_test_summary.log)

## Commit 0 — Measurement
(fill in after running baseline benchmark)

## Decisions made beyond the plan
(fill in as execution proceeds)

## Deferred observations
(fill in as execution proceeds)

## Surprises vs. the plan
(fill in as execution proceeds)

## Test anomalies
(fill in as execution proceeds)
EOF

# Extend bench script to support per-tracer vs multi-tracer variants
# ... (plan details in §4.4 Commit 0 subsection)

# Capture baseline
julia --project=. scripts/benchmarks/bench_strang_sweep.jl \
    --mode=per-tracer,multi-tracer \
    --sizes=medium,large \
    --ntracers=1,5,10,30 \
    --cfl-limits=0.4,Inf \
    > artifacts/perf/plan14/baseline_comparison.log

# Write the findings to NOTES.md Commit 0 section

git add docs/plans/14_SINGLE_PIPELINE_PLAN/NOTES.md artifacts/
git commit -m "Commit 0: NOTES.md + baseline measurement of per-tracer vs multi-tracer paths"
```

**If the measurement reveals that multi-tracer path has bugs** (e.g.,
gives wrong results on some config), STOP. Fix multi-tracer path
correctness before proceeding with the refactor. This becomes a
separate commit before any Commit 1.

## 4.3 Design decisions (pre-answered)

Every decision final. If a rule is ambiguous, that's a bug in this
document — STOP and ask.

**Decision 1: 4D tracer array is `Array{FT, 4}` with shape
`(Nx, Ny, Nz, Nt)`.**

Matches OPERATOR_COMPOSITION.md §4. Do NOT use `(Nt, Nx, Ny, Nz)`
or other permutations. Do NOT use a different container type
(SArray, StructArray, etc.).

Rationale: spatial indices `(i, j, k)` are the high-frequency loop
variables; tracer index `t` is slower-varying. Putting tracer last
keeps spatial stencils contiguous in memory for the fastest axis.

**Decision 2: `tracer_names::Vector{Symbol}` is a companion field,
not embedded in the array type.**

Matches OPERATOR_COMPOSITION.md §4.

Rationale: keeps the tracer array a plain `Array{FT, 4}` without
type-level name encoding. Dispatch on names (e.g., for per-tracer
decay constants in future chemistry) uses the `tracer_names`
vector for lookups.

Do NOT use `NamedDimsArray` or similar packages. Plain array +
vector is enough.

**Decision 3: `strang_split!` signature stays the same.**

Do NOT rename to `apply!`. That rename happens in a future plan
when multiple operators exist. Plan 14 just establishes the shape.

The signature is still:
```julia
strang_split!(state::CellState, fluxes::StructuredFaceFluxState,
              grid::AbstractGrid, scheme::AbstractAdvectionScheme,
              dt; workspace::AdvectionWorkspace)
```

Internal implementation changes (no per-tracer loop) but callers
see no change.

**Decision 4: `m_save` field is deleted from workspace.**

After plan 14, there's no per-tracer loop, so no need to restore
m between tracers. The `m_save` workspace field becomes unused
and is deleted in Commit 3.

Watch for: current LinRood and CubedSphereStrang paths may
reference `m_save`. They use their own workspaces
(`CSAdvectionWorkspace`), so likely unaffected. Verify with grep
before deletion.

**Decision 5: `CellState` constructors accept either keyword args
(tracer names) or direct 4D array.**

Old API:
```julia
CellState(m; CO2=rm_CO2, CH4=rm_CH4)  # NamedTuple internally
```

New API:
```julia
# Keyword form (preserved)
CellState(m; CO2=rm_CO2, CH4=rm_CH4)
# internally assembles 4D array + names = [:CO2, :CH4]

# Direct form (new)
CellState(m, tracers_4d, tracer_names)
```

The keyword form preserves test code; the direct form is useful
for code that already has a 4D array.

**Decision 6: Provide `get_tracer(state, name::Symbol)` accessor.**

For readable test code:
```julia
get_tracer(state, :CO2)  # returns view into state.tracers
```

Implementation:
```julia
function get_tracer(state::CellState, name::Symbol)
    idx = findfirst(==(name), state.tracer_names)
    idx === nothing && throw(KeyError(name))
    return view(state.tracers, :, :, :, idx)
end
```

Use `view`, not copy. Test code that mutates the returned array
must see the mutation reflected in `state.tracers`.

**Decision 7: Workspace structure stays as `AdvectionWorkspace`.**

Plan 14 does NOT introduce a `TransportWorkspace` container yet.
That container gets created when the second operator (chemistry,
diffusion, etc.) arrives. Plan 14 just ensures `AdvectionWorkspace`
has a clean design that fits into such a container later.

Specifically: `AdvectionWorkspace` should have no cross-cutting
concerns (no "this is used by both advection and diffusion" fields).
Plan 14 is the right moment to clean any such fields if they
exist.

**Decision 8: NO external users to worry about.**

Per repo status, there are no external users. Break `state.tracers.CO2`
API freely. Do not add deprecation warnings. Do not preserve backward
compatibility beyond what simplifies testing.

**Decision 9: Plan 14 does NOT touch the surface flux BC mechanism.**

Surface flux BCs (emissions as bottom-face flux in advection Z-sweep)
are plan 16's work. Plan 14's Z-sweep continues to use zero bottom
flux (wall BC).

## 4.4 Atomic commit sequence

### Commit 0: NOTES.md + baseline measurement

Already described in §4.2. The key artifact is
`artifacts/perf/plan14/baseline_comparison.log` with per-tracer vs
multi-tracer measurements at multiple configurations.

### Commit 1: Extend benchmark script to measure both paths

If the current bench script only measures one path, extend it.
The script should be able to run:
- `--mode=per-tracer` (current production, NamedTuple-based)
- `--mode=multi-tracer` (if accessible via test code paths)

If the multi-tracer path isn't easily accessible from the bench
script (e.g., it's only in test code), create a minimal harness in
`scripts/benchmarks/bench_multi_tracer.jl`.

Test: bench script runs without errors, produces sensible numbers.

```bash
git add scripts/benchmarks/
git commit -m "Commit 1: Extend benchmark to compare per-tracer vs multi-tracer paths"
```

### Commit 2: Introduce 4D tracer array plumbing (non-breaking)

BEFORE changing the primary code path, add the 4D infrastructure
alongside the existing NamedTuple path:

1. Add `tracers_4d::Array{FT, 4}` and `tracer_names::Vector{Symbol}`
   fields to `CellState` (or a subtype).
2. Add constructor that accepts 4D input directly.
3. Add `get_tracer` accessor.
4. Do NOT change `strang_split!` yet. Both paths coexist.

Test: all existing tests pass (NamedTuple path unchanged); new
tests for 4D accessors pass.

```bash
git commit -m "Commit 2: Add 4D tracer array plumbing alongside NamedTuple path"
```

### Commit 3: Switch `strang_split!` to 4D multi-tracer pipeline

The big commit. Rewrite `strang_split!` internals to:
1. Accept 4D tracer array
2. Call 6 multi-tracer sweeps directly
3. Remove the per-tracer loop
4. Remove `m_save` references

Update `CellState` default constructor: keyword-form now produces
4D array internally (instead of NamedTuple).

Migrate ALL test call sites that accessed `state.tracers.CO2` to
use `get_tracer(state, :CO2)` or direct indexing.

This is where tests break if any access pattern was missed. The
precondition grep should have caught them; if not, the test suite
does.

Remove `m_save` field from `AdvectionWorkspace`.

Test: full test suite. Every test that passed in baseline must
pass now. The pre-existing 77 failures remain.

```bash
git commit -m "Commit 3: Switch strang_split! to 4D multi-tracer pipeline"
```

**If Commit 3 has test regressions beyond the baseline failures,
STOP and revert.** Find the missed access pattern, add it to
Commit 3's migration list, retry.

### Commit 4: Remove dead code paths

After Commit 3:
- Per-tracer sweep variants (if they were separate from multi-
  tracer) are now dead code
- NamedTuple-path helpers in `strang_split!` are dead
- Any `m_save` users that weren't caught in Commit 3

Grep for anything still referencing the old pattern. Delete.

Test: full suite.

```bash
git commit -m "Commit 4: Delete dead per-tracer code paths"
```

### Commit 5: Benchmark post-refactor + Documentation

Run benchmark on the same configurations as Commit 0:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl \
    --mode=production \
    --sizes=medium,large \
    --ntracers=1,5,10,30 \
    --cfl-limits=0.4,Inf \
    > artifacts/perf/plan14/after.log
```

Compare to Commit 0 baseline. Document the result in NOTES.md:
- If ≥5% win at production settings: "PERF WIN"
- If <5%: "CLEANUP ONLY, perf within ±5%"
- If regression: "UNEXPECTED REGRESSION, investigate"

Update CLAUDE.md if the measurement reveals a new insight worth
capturing. Update ARCHITECTURAL_SKETCH.md if line counts / shape
differ meaningfully from what it describes.

```bash
git commit -m "Commit 5: Documentation and final benchmarks"
```

## 4.5 Test plan per commit

After EACH commit:
```bash
julia --project=. -e 'using AtmosTransport'   # compile check
julia --project=. test/runtests.jl            # core tests
```

Compare pass/fail to baseline (§4.1 captured 77 pre-existing
failures). New failures → STOP, revert, investigate.

For GPU:
```bash
JULIA_CUDA_USE_BINARYBUILDER=false julia --project=. test/runtests.jl
```

## 4.6 Acceptance criteria

**Correctness (hard):**
- All tests that passed in baseline pass post-refactor
- No NEW test failures beyond the 77 pre-existing
- Mass conservation exactly preserved (ULP tolerance)
- `state.tracers` is now `Array{FT, 4}`, not NamedTuple

**Code cleanliness (hard):**
- `CellState.tracers` field is a 4D array
- `tracer_names::Vector{Symbol}` companion field exists
- `strang_split!` has no per-tracer loop
- `m_save` field is gone from `AdvectionWorkspace`
- `get_tracer(state, name)` accessor exists and is used where
  appropriate in tests
- No references to `state.tracers.CO2`-style access anywhere

**Performance (soft):**
- Per-step wall time within ±5% of baseline at production
  settings (cfl_limit=Inf, GPU large, Nt=10-30)
- If Commit 0 found a measurable win at some config, that win
  is preserved in final measurement
- CPU performance unchanged (multi-tracer kernels optimized
  for GPU; CPU may not see gains)

**Alignment with composition doc:**
- Data layout matches OPERATOR_COMPOSITION.md §4 exactly
- No deviations from the specified `(Nx, Ny, Nz, Nt)` order

**Documentation:**
- NOTES.md complete with Commit 0 measurement findings,
  decisions, deferred observations, surprises
- ARCHITECTURAL_SKETCH.md updated if needed
- CLAUDE.md updated if a new measurement insight emerged

## 4.7 Rollback plan

Standard principles:
- Do not "fix forward"
- Revert to last-known-good commit
- Write the failure in NOTES.md
- Stop and ask if stuck >30 minutes

Specific rollback points:
- **Commit 0 reveals multi-tracer path has bugs.** Fix
  multi-tracer path in a separate commit before proceeding.
  If the bug is large, stop and escalate.
- **Commit 2 breaks something.** Likely a constructor
  incompatibility. Revert, check constructor signature
  completeness.
- **Commit 3 breaks tests.** Migration missed a call site.
  Revert, grep harder, retry.
- **Commit 5 shows performance regression.** Unexpected.
  Reveals some cost that isn't the per-tracer loop. Investigate
  before shipping.

## 4.8 Known pitfalls

1. **"I'll make tracers a generic container and let the compiler
   figure it out."** NO. Decision 1 is explicit: `Array{FT, 4}`.
   Do not generalize.

2. **"I'll preserve `state.tracers.CO2` access via `getproperty`
   shim."** NO. Per Decision 8, there are no external users.
   Break the API cleanly; migrate tests to `get_tracer`.

3. **"The multi-tracer kernels might be slower than per-tracer
   on small problems."** Possible, but Commit 0 measures this.
   If small-problem perf is critical, NOTES.md flags it; we
   don't branch on problem size in the production code.

4. **"I notice the workspace struct could be cleaner."**
   Log to NOTES.md as deferred. Do not clean here unless directly
   related to `m_save` removal.

5. **"LinRood uses `m_save`."** Check first. LinRood has its own
   workspace (`CSAdvectionWorkspace`). If it references `m_save`
   on the structured `AdvectionWorkspace`, that's a cross-
   contamination bug worth a separate fix. But most likely:
   LinRood is untouched by plan 14.

6. **"Tests break because they construct tracers as NamedTuples."**
   The constructor signature is preserved (keyword args). The
   INTERNAL representation changes. Tests that do
   `state.tracers.CO2` break and need migration. Tests that do
   `state = CellState(m; CO2=...)` continue to work.

7. **"The benchmark shows no speedup. Did I break something?"**
   Probably not — per §1.2, realistic expectation is 0-10% at
   production settings. No speedup is acceptable. Regression is
   not.

---

# Part 5 — How to Work

## 5.1 Session cadence

- **Session 1:** Precondition + Commit 0 (NOTES.md, measurement)
  — THE CRITICAL SESSION. Measurement determines framing for
  everything else.
- **Session 2:** Commit 1 (benchmark extension)
- **Session 3:** Commit 2 (4D plumbing alongside NamedTuple)
- **Session 4-5:** Commit 3 (the big switch — expect this to
  take 2 sessions)
- **Session 6:** Commit 4 (dead code cleanup)
- **Session 7:** Commit 5 (benchmarks, docs, retrospective)

Each session: `git status` at start, `NOTES.md` update at end,
decisions logged.

## 5.2 When to stop and ask

- Commit 0 measurement shows multi-tracer path is broken
- Any test that passed in baseline now fails
- A file you need to edit isn't in §4.2
- The scope feels like it's expanding
- >30 minutes on a single issue
- You're tempted to add a "small" optimization not in the plan

## 5.3 NOTES.md discipline

Update NOTES.md at these moments:
- After Commit 0: baseline measurement findings (required)
- After each other commit: any surprises, decisions, deferred
  observations
- At the end: "Template usefulness for plans 15-18" section
  capturing what worked and what didn't about this plan's
  structure

---

# End of Plan

After this refactor ships:
- `CellState.tracers` is a 4D array
- `strang_split!` has one pipeline, not two
- Data layout matches OPERATOR_COMPOSITION.md §4
- Foundation laid for plans 15-18 (chemistry, surface BCs,
  diffusion, convection) to slot in without re-architecting

The next plans in the sequence:
- Plan 15: slow chemistry operator (small, validates interface)
- Plan 16: surface flux BCs in advection (small extension)
- Plan 17: vertical diffusion (new machinery)
- Plan 18: convection (new machinery, complex)

Each will reference OPERATOR_COMPOSITION.md as the architectural
reference and this plan's NOTES.md for lessons learned.
