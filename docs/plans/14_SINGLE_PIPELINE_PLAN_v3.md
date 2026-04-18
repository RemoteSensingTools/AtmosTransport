# Advection Pipeline Unification — Implementation Plan for Plan Agent (v3)

**Status:** Ready for execution after plans 11, 12, 13 have shipped.
**Target branch:** new branch from wherever plans 11/12/13's shipped
  work lives (typically `restructure/dry-flux-interface` or its
  successor). Verify in §4.1.
**Estimated effort:** 2-3 weeks, single engineer / agent.
**Primary goal:** Restructure `CellState.tracers` from a NamedTuple
  of 3D arrays to a 4D array with a storage-agnostic accessor API,
  eliminating the per-tracer loop in `strang_split!` and aligning
  with the operator interface in `OPERATOR_COMPOSITION.md`.

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
- `ARCHITECTURAL_SKETCH.md` — advection subsystem shape. Specifies
  the storage-agnostic accessor API this plan must provide.
- `CLAUDE.md` § "Performance tips" — the "measure, don't subtract"
  lesson from plan 13 applies here.

---

# Revisions from v2 (based on reviewer feedback)

What changed between v2 and this v3:

1. **API preservation, not API break.** v2 said to break
   `state.tracers.CO2`-style access and force all callers to
   `get_tracer(state, :CO2)`. v3 preserves property access via
   `getproperty`, because `ARCHITECTURAL_SKETCH.md` line 131
   specifies stable accessor API and because the storage-agnostic
   approach is cleaner architecture regardless of external-user
   concerns.

2. **Storage-agnostic accessor layer added as Commit 2.** Before
   flipping the storage to 4D, establish a generic accessor API
   that works for BOTH the old NamedTuple layout AND the future
   4D layout. Kernel code dispatches on raw storage; everything
   else uses the API.

3. **Expanded scope.** v2 understated the file reach. v3 explicitly
   includes:
   - `CellState.jl` — the struct itself
   - `Tracers.jl` — existing tracer accessor functions
   - `DrivenSimulation.jl` — simulation construction
   - `TransportModel.jl` — default workspace construction
   - `StrangSplitting.jl` face-indexed path (line 1043 range)

4. **TransportModel workspace integration made explicit.** v2
   didn't mention `TransportModel.jl:14` constructs
   `AdvectionWorkspace(state.air_mass)` without n_tracers. Once
   the 4D path is mandatory, this path must allocate 4D buffers
   correctly. v3 adds a specific commit for this.

5. **Julia memory-order correction.** v2 and the old composition
   doc had the layout justification backwards — claimed "k
   fastest" which is wrong for Julia column-major. The updated
   OPERATOR_COMPOSITION.md §4 explains the correct reasoning
   (i fastest). v3 references the corrected doc and does not
   repeat the wrong justification.

6. **Raw-storage vs. API-access distinction formalized.** v3
   adds Decision 10: kernels ONLY dispatch on `tracers_raw` (the
   4D Array); non-kernel code MUST use the accessor API
   (`get_tracer`, `ntracers`, etc.).

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

- `CellState.tracers_raw` becomes `Array{FT, 4}` with shape
  `(Nx, Ny, Nz, Nt)` — the 4D storage
- `CellState.tracer_names` tracks the names (type TBD, see
  Decision 2)
- A storage-agnostic accessor API (`ntracers`, `get_tracer`,
  `eachtracer`) is the primary interface
- Property access `state.tracers.CO2` is preserved via a
  `getproperty` overload that returns `get_tracer(state, :CO2)`
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
pipelines becomes one. NamedTuple of arrays becomes one 4D array
with a storage-agnostic accessor API. The operator interface
aligns with the composition doc. Performance is a possible bonus,
not the goal.

## 1.3 What "clean" means, concretely

### Data layout

Before:
```julia
struct CellState{...}
    air_mass::Array{FT, 3}     # (Nx, Ny, Nz)
    tracers::NamedTuple        # {CO2::Array{FT,3}, CH4::Array{FT,3}, ...}
end
```

After:
```julia
struct CellState{...}
    air_mass::Array{FT, 3}         # (Nx, Ny, Nz)
    tracers_raw::Array{FT, 4}      # (Nx, Ny, Nz, Nt)
    tracer_names                   # companion (see Decision 2)
end
```

With accessor API:
```julia
ntracers(state)                 # Int
tracer_index(state, :CO2)       # Int (or nothing)
tracer_name(state, idx::Int)    # Symbol
get_tracer(state, :CO2)         # view into tracers_raw
get_tracer(state, idx::Int)     # view
eachtracer(state)               # iterator over (name, view) pairs

# Property access preserved via getproperty:
state.tracers.CO2               # == get_tracer(state, :CO2)
```

This matches `OPERATOR_COMPOSITION.md` §4 with i fastest-varying
(Julia column-major) and the storage-agnostic accessor API.

### Pipeline

Before:
```julia
function strang_split!(state, fluxes, grid, scheme, dt; workspace)
    for (tracer_name, tracer) in pairs(state.tracers)
        m = copy(state.air_mass)
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
    # ONE call per direction, processes all tracers in tracers_raw
    sweep_x!(state.tracers_raw, state.air_mass, fluxes, ...)
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

`strang_split!` signature is unchanged. The internal
implementation changes (no per-tracer loop) but callers see no
change IF they use the accessor API instead of NamedTuple
introspection.

## 1.4 What's going away

- `CellState.tracers::NamedTuple` — replaced by 4D
  `tracers_raw` + accessor API
- The per-tracer loop in `strang_split!` — replaced by 6 calls
- The `m_save` workspace field — no longer needed (no restore
  between tracers; see §4.4 Commit 5)
- Per-tracer kernel variants (if any duplicate the multi-tracer
  kernels) — collapsed into one

## 1.5 What stays the same

- Kernel arithmetic (`reconstruction.jl`, `limiters.jl`,
  `schemes.jl`) — untouched
- CFL pilot (unified in plan 13) — untouched
- Face-indexed / reduced Gaussian PATH — touched (generator in
  `StrangSplitting.jl` line 1043 range must migrate to 4D), but
  algorithm unchanged
- LinRood / CubedSphereStrang — untouched (separate workspace
  types; no cross-cutting)
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
- Tracer access pattern is STORAGE-SPECIFIC: code that accesses
  `state.tracers.CO2` assumes NamedTuple; code that does
  `state.tracers[:, :, :, t]` assumes 4D Array

After this refactor:
- ONE tracer STORAGE layout (4D array)
- ONE accessor API (`get_tracer`, `ntracers`, `eachtracer`)
- ONE pipeline (6 multi-tracer sweeps)
- No `m_save`
- Future operators (diffusion, convection, chemistry) extend the
  same pipeline — they get a 4D tracer array and act on it
- Future storage changes (GPU array types, StructArray, etc.) are
  insulated from callers by the accessor API

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
Plan 14 ships that layout AND the accessor API that keeps
downstream plans from depending on storage details.

## 2.3 What this does NOT enable

- Faster production runs (probably — measure first, don't assume)
- New scientific capabilities
- Different meteorology support

This is an internal refactor. User-visible behavior unchanged.
Per-property tracer access preserved via `getproperty`.

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
- **Scheme hierarchy.** Plan 12 settled it.

## 3.2 Do NOT add

- **New operator types.** Advection unification only.
- **New schemes.** Plan 12 settled the hierarchy.
- **`apply!` dispatch.** The rename from `strang_split!` to
  `apply!` happens when multiple operators exist (plan 15+).
  Plan 14 just establishes the SHAPE.
- **Performance optimizations beyond the refactor.** Log to
  NOTES.md as deferred.

## 3.3 Potential confusion — clarified

You will see tests that create tracers as NamedTuples:
```julia
CellState(air_mass; CO2=rm_CO2, CH4=rm_CH4)
```
These constructors continue to work — they internally assemble
the 4D array. The keyword form is the PRIMARY construction API.

You will see tests that access `state.tracers.CO2`. This API
IS PRESERVED via `getproperty` overload. Tests that do this
continue to work. The implementation is:

```julia
function Base.getproperty(state::CellState, name::Symbol)
    # Fast path for concrete fields:
    name === :air_mass && return getfield(state, :air_mass)
    name === :tracers_raw && return getfield(state, :tracers_raw)
    name === :tracer_names && return getfield(state, :tracer_names)
    # Property-style tracer access through :tracers namespace:
    name === :tracers && return TracerAccessor(state)
    # Fallback:
    return getfield(state, name)
end

# TracerAccessor provides the state.tracers.CO2 pattern:
struct TracerAccessor{S}
    state::S
end
function Base.getproperty(acc::TracerAccessor, name::Symbol)
    return get_tracer(getfield(acc, :state), name)
end
```

The `TracerAccessor` is a lazy view — no allocation — so
`state.tracers.CO2` returns the same view that
`get_tracer(state, :CO2)` would.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch.
git branch -a | head -20
git log --oneline --all | grep -i "plan 13\|sync.*cfl\|cleanup" | head -5
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

# 4. MAP THE ACTUAL SCOPE via grep. Line numbers in this plan
# document are approximate; real scope is whatever grep finds.
grep -rn "state.tracers\b\|state\.tracers\.\|\.tracers\[" \
    src/ test/ --include="*.jl" | tee artifacts/tracer_access_sites.txt
# Review the list. These are all sites that touch tracers.
# Every one needs to be understood (not necessarily modified —
# some are fine through the new getproperty pathway).

grep -rn "AdvectionWorkspace\s*(" src/ test/ --include="*.jl" | \
    tee artifacts/workspace_construction_sites.txt
# These are all sites that construct AdvectionWorkspace. TransportModel.jl
# and others need n_tracers-aware construction post-refactor.

grep -rn "NamedTuple\|tracers::NT\|Tracers(" src/ test/ --include="*.jl" | \
    tee artifacts/tracer_type_sites.txt
# Find all sites that touch the NamedTuple tracer TYPE.

# 5. Capture baseline failure set
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

# 6. Record commit hash
git rev-parse HEAD > artifacts/baseline_commit.txt

# 7. Set up benchmark directory
mkdir -p artifacts/perf/plan14
```

If any precondition fails, STOP and investigate.

**IMPORTANT:** If the grep in step 4 finds call sites NOT listed
in §4.2 below, ADD THEM to your working scope before proceeding.
Do not skip them. The plan lists the expected scope; the grep
is authoritative about actual scope.

## 4.2 Change scope — the exact file list

Lines are approximate; re-grep at execution time.

**Files to MODIFY:**

Core storage:
- `src/Utils/CellState.jl` — the struct definition, constructors,
  `getproperty` overload, `TracerAccessor`, line ~57
- `src/Utils/Tracers.jl` — accessor API lives here
  (`ntracers`, `get_tracer`, etc.), line ~28
- `src/Utils/DrivenSimulation.jl` — simulation construction,
  tracer iteration patterns, line ~115

Advection pipeline:
- `src/Operators/Advection/StrangSplitting.jl` — main
  `strang_split!` body + face-indexed path around line ~1043
- `src/TransportModel.jl` — default workspace construction
  (line ~14), must pass n_tracers

Tests (update as accessor API changes):
- `test/test_advection_kernels.jl`
- `test/test_basis_explicit_core.jl`
- `test/test_dry_flux_interface.jl`
- `test/test_driven_simulation.jl`
- `test/test_structured_mesh_metadata.jl`
- `test/test_reduced_gaussian_mesh.jl`
- `test/test_real_era5_*` (any file that constructs CellState)

Any test site that uses `state.tracers.CO2` style access works
unchanged (through `getproperty`). Any test that iterates
`pairs(state.tracers)` or similar NamedTuple-specific patterns
needs migration to `eachtracer(state)`.

## 4.3 Design decisions (pre-answered)

Every decision final. If a rule is ambiguous, STOP and ask.

**Decision 1: 4D storage is `Array{FT, 4}` with shape
`(Nx, Ny, Nz, Nt)`, i fastest-varying.**

Matches OPERATOR_COMPOSITION.md §4. Julia column-major means
i is fastest. Do NOT use `(Nt, Nx, Ny, Nz)` or other permutations.

**Decision 2: `tracer_names` type is `NTuple{Nt, Symbol}`.**

This is a fixed-size tuple known at CellState construction.
Rationale:
- Type-stable: compiler knows the count at compile time
- Faster dispatch for per-tracer kernels
- Matches `ARCHITECTURAL_SKETCH.md` line 121

Alternative rejected: `Vector{Symbol}`. Simpler but type-unstable
(Julia doesn't know length at compile time), slightly worse
inference in downstream code. The simplicity argument isn't
strong enough to override the perf/inference argument.

Construction: `tracer_names = (:CO2, :CH4, ...)` — always a
tuple, never a vector.

**Decision 3: Accessor API is defined in `src/Utils/Tracers.jl`.**

Functions:
```julia
ntracers(state::CellState)::Int
tracer_index(state::CellState, name::Symbol)::Union{Int, Nothing}
tracer_name(state::CellState, idx::Int)::Symbol
get_tracer(state::CellState, name::Symbol)  # returns view
get_tracer(state::CellState, idx::Int)       # returns view
eachtracer(state::CellState)                 # iterator
```

All return views, not copies. Mutation through the view is
reflected in `state.tracers_raw`.

**Decision 4: Property access preserved via `getproperty` +
`TracerAccessor`.**

`state.tracers.CO2` returns the same view as
`get_tracer(state, :CO2)`. `TracerAccessor` is a lazy wrapper
(`struct TracerAccessor{S}; state::S; end`) with its own
`getproperty` overload. No allocation; just a layer of dispatch.

**Decision 5: `strang_split!` signature unchanged.**

```julia
strang_split!(state::CellState, fluxes::StructuredFaceFluxState,
              grid::AbstractGrid, scheme::AbstractAdvectionScheme,
              dt; workspace::AdvectionWorkspace)
```

Internal implementation changes (uses `state.tracers_raw`, no
per-tracer loop) but callers see no change.

**Decision 6: Kernels dispatch ONLY on `tracers_raw`.**

```julia
# YES (kernel code):
sweep_x!(state.tracers_raw, state.air_mass, ...)

# NO (kernel code):
for t in 1:ntracers(state)
    tr = get_tracer(state, t)  # per-tracer loop in KERNEL code is wrong
    sweep_x!(tr, state.air_mass, ...)
end
```

Non-kernel code (tests, setup, simulation) uses the accessor API.
Keep the distinction clean: kernels see raw storage for
multi-tracer ops, everything else sees the API.

**Decision 7: `m_save` field deleted from workspace.**

After plan 14, no per-tracer loop means no need to restore m
between tracers. The `m_save` workspace field is unused and
deleted in Commit 5.

Watch for: LinRood and CubedSphereStrang use separate workspaces
(`CSAdvectionWorkspace`). They should be unaffected. Grep to
verify before deletion.

**Decision 8: `CellState` constructors support both forms.**

Old (keyword form, PRESERVED):
```julia
CellState(air_mass; CO2=rm_CO2, CH4=rm_CH4)
# Internally: tracers_raw is 4D with layers [rm_CO2, rm_CH4]
# tracer_names = (:CO2, :CH4)
```

New (direct form, added):
```julia
CellState(air_mass, tracers_raw::Array{FT, 4},
          tracer_names::NTuple{Nt, Symbol})
```

Keyword form is PRIMARY (preserves test code). Direct form is
useful for code that already has a 4D array.

**Decision 9: No external users to worry about.**

Per repo status, there are no external users. Break internal test
patterns freely IF a test uses NamedTuple-specific iteration
(e.g., `pairs(state.tracers)`) that can't be expressed through
the accessor API. Migrate those to `eachtracer(state)`.

Do NOT break `state.tracers.CO2` — the `getproperty` pathway
preserves it.

**Decision 10: Workspace construction is n_tracers-aware.**

Current `AdvectionWorkspace(air_mass)` constructor allocates 4D
buffers at `0×0×0×0`. Post-plan-14, 4D buffers must be allocated
with correct Nt. Update constructor:

```julia
AdvectionWorkspace(air_mass, n_tracers::Int)
```

OR infer from a CellState:
```julia
AdvectionWorkspace(state::CellState)  # infers n_tracers from state
```

Pick one and update all call sites. Recommended: BOTH forms, with
`AdvectionWorkspace(state)` as the primary path (simpler for
callers).

Sites to update (from grep in §4.1):
- `src/TransportModel.jl:14`
- `src/Operators/Advection/StrangSplitting.jl` (default workspace)
- Tests that construct AdvectionWorkspace directly

**Decision 11: Plan 14 does NOT touch surface flux BC mechanism.**

Surface flux BCs (emissions as bottom-face flux in advection
Z-sweep) are plan 16's work. Plan 14's Z-sweep continues to use
zero bottom flux (wall BC).

## 4.4 Atomic commit sequence

### Commit 0: NOTES.md + baseline measurement

Matches pattern from plans 12, 13.

```bash
mkdir -p docs/plans/14_SINGLE_PIPELINE_PLAN
cat > docs/plans/14_SINGLE_PIPELINE_PLAN/NOTES.md << 'EOF'
# Plan 14 Execution Notes — Advection Pipeline Unification

Plan: `docs/plans/14_SINGLE_PIPELINE_PLAN_v3.md`

## Baseline
Commit: (fill in)
Pre-existing test failures: (fill in)

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
```

**The measurement step:** Before changing any code, measure:

1. **Per-tracer loop cost vs. multi-tracer cost on current code.**
2. Configurations: CPU medium F32/F64, GPU medium/large F32, all
   schemes, `Nt = 1, 5, 10, 30`, `cfl_limit = 0.4` and `Inf`.
3. **Acceptance criterion:**
   - If multi-tracer gives ≥5% win at ANY config: perf win +
     cleanup. Proceed.
   - If multi-tracer gives <5% win at all configs: pure cleanup.
     Proceed, reframe accordingly.
   - If multi-tracer is SLOWER than per-tracer: STOP.
     Multi-tracer path is broken; fix it first.

```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl \
    --mode=per-tracer,multi-tracer \
    --sizes=medium,large \
    --ntracers=1,5,10,30 \
    --cfl-limits=0.4,Inf \
    > artifacts/perf/plan14/baseline_comparison.log

git add docs/plans/14_SINGLE_PIPELINE_PLAN/NOTES.md artifacts/
git commit -m "Commit 0: NOTES.md + baseline measurement of per-tracer vs multi-tracer paths"
```

### Commit 1: Extend benchmark script

If the current bench script doesn't support mode=per-tracer vs
mode=multi-tracer comparison, extend it. Create helper scripts
if needed.

Test: bench script runs without errors, produces sensible numbers.

```bash
git add scripts/benchmarks/
git commit -m "Commit 1: Extend benchmark to compare per-tracer vs multi-tracer paths"
```

### Commit 2: Storage-agnostic accessor API (STORAGE UNCHANGED)

**This commit changes NO storage.** Tracers remain a NamedTuple.
But the accessor API is established now, so all callers can be
migrated to it BEFORE the storage flip in Commit 4.

Add to `src/Utils/Tracers.jl`:

```julia
# Generic accessors that work for both NamedTuple (current) and
# 4D Array (future) storage.

ntracers(state::CellState) = _ntracers_impl(state.tracers)
_ntracers_impl(t::NamedTuple) = length(t)  # current
# After Commit 4, add: _ntracers_impl(t::Array{FT,4} where FT) = size(t, 4)

tracer_index(state::CellState, name::Symbol) = _tracer_index_impl(state.tracers, name)
_tracer_index_impl(t::NamedTuple, name::Symbol) = findfirst(==(name), keys(t))
# After Commit 4: add version for Array{FT,4}

tracer_name(state::CellState, idx::Int) = _tracer_name_impl(state.tracers, idx)
_tracer_name_impl(t::NamedTuple, idx::Int) = keys(t)[idx]

get_tracer(state::CellState, name::Symbol) = _get_tracer_impl(state.tracers, name)
_get_tracer_impl(t::NamedTuple, name::Symbol) = t[name]
# After Commit 4: add version for Array{FT,4} returning view

get_tracer(state::CellState, idx::Int) = _get_tracer_impl(state.tracers, idx)
_get_tracer_impl(t::NamedTuple, idx::Int) = t[idx]

eachtracer(state::CellState) = _eachtracer_impl(state.tracers)
_eachtracer_impl(t::NamedTuple) = pairs(t)  # already a (name, value) iterator
```

The pattern: public API dispatches on `state.tracers`'s type via
a `_*_impl` helper. Currently `state.tracers` is `NamedTuple`, so
the impl functions handle `NamedTuple`. After Commit 4 flips
storage to 4D Array, additional impl functions handle
`Array{FT, 4}`. The public API doesn't change.

Migrate call sites (both source and tests) to the accessor API
where it's a clean replacement:
- `pairs(state.tracers)` → `eachtracer(state)`
- `keys(state.tracers)` → access via `tracer_name(state, i) for i in 1:ntracers(state)` or via `eachtracer(state)`
- `length(state.tracers)` → `ntracers(state)`
- `state.tracers[:CO2]` → `get_tracer(state, :CO2)`
- `state.tracers.CO2` — LEAVE AS IS. The getproperty path works
  for both storage forms.

Test: full suite passes. NO storage change has happened yet; the
accessor API is just a pass-through to NamedTuple.

```bash
git add src/ test/
git commit -m "Commit 2: Storage-agnostic tracer accessor API (NamedTuple pass-through)"
```

### Commit 3: Add 4D tracer array plumbing alongside NamedTuple

BEFORE flipping the primary storage, add 4D infrastructure as a
PARALLEL field:

```julia
struct CellState{FT, ...}
    air_mass::Array{FT, 3}
    tracers::NamedTuple           # CURRENT primary storage
    tracers_raw::Array{FT, 4}     # NEW parallel 4D storage
    tracer_names::NTuple          # NEW companion
end
```

Both `state.tracers` and `state.tracers_raw` hold the same data,
kept consistent by constructors. Kernels that know about 4D can
use `state.tracers_raw` (testing the multi-tracer path). Per-
tracer code uses `state.tracers`.

This is temporary — Commit 4 removes the duplicate.

Add 4D versions of the `_*_impl` functions from Commit 2:
```julia
_ntracers_impl(t::Array{FT, 4}) where FT = size(t, 4)
_tracer_index_impl(t::NamedTuple, name) = ...  # existing
_tracer_index_impl_4d(names::NTuple, name::Symbol) = findfirst(==(name), names)
# etc.
```

But the accessor API still dispatches on `state.tracers` (the
NamedTuple) in this commit — so nothing CHANGES from the user's
perspective.

Test: full suite passes. `state.tracers_raw` is present but
unused except by explicit tests.

```bash
git commit -m "Commit 3: Add parallel 4D tracer storage alongside NamedTuple"
```

### Commit 4: Switch `strang_split!` to 4D multi-tracer pipeline

The big commit. Changes:

1. `strang_split!` now uses `state.tracers_raw` internally. The
   6-sweep pipeline calls multi-tracer kernels.
2. The per-tracer loop is removed.
3. Accessor API `_*_impl` functions are SWITCHED to dispatch on
   `tracers_raw` instead of `tracers`:
   ```julia
   ntracers(state::CellState) = size(state.tracers_raw, 4)
   get_tracer(state::CellState, name::Symbol) =
       view(state.tracers_raw, :, :, :, tracer_index(state, name))
   # etc.
   ```
4. `getproperty(state, :tracers)` returns `TracerAccessor(state)`
   (not the NamedTuple). `state.tracers.CO2` goes through
   `TracerAccessor`'s `getproperty` to `get_tracer`.
5. The `tracers::NamedTuple` field is REMOVED from CellState.
   Only `tracers_raw` + `tracer_names` remain.
6. Face-indexed `StrangSplitting.jl` path (~line 1043) migrates
   to 4D. Same pattern as structured path.

**Watch points:**
- Tests that pass if using the accessor API — should still pass.
- Tests that access `state.tracers` expecting a NamedTuple — fail
  if they use NamedTuple-specific operations. Migrate.
- `getproperty(state, :tracers)` now returns a TracerAccessor, so
  code that does `state.tracers isa NamedTuple` fails. Remove
  such checks.

Test: full suite. Every test that passed in baseline must pass
now. The pre-existing 77 failures remain.

```bash
git commit -m "Commit 4: Switch to 4D tracer storage; strang_split! uses multi-tracer kernels"
```

**If Commit 4 has test regressions beyond baseline, STOP and
revert.** The grep in §4.1 should have caught all sites; if not,
investigate.

### Commit 5: TransportModel + workspace integration

Update workspace construction throughout:
- Add `AdvectionWorkspace(state::CellState)` constructor that
  infers n_tracers from state
- Update `AdvectionWorkspace(air_mass, n_tracers)` to always
  allocate non-zero 4D buffers
- Migrate `src/TransportModel.jl:14` and similar sites to use the
  state-aware constructor
- Grep for `AdvectionWorkspace(` and fix all call sites

Also in this commit:
- Remove `m_save` field from AdvectionWorkspace (now dead)
- Any other dead fields found

Test: full suite.

```bash
git commit -m "Commit 5: TransportModel and workspace integration for 4D storage"
```

### Commit 6: Remove dead code paths

After Commit 4 and 5:
- Per-tracer sweep variants (if separate from multi-tracer) are
  dead code
- NamedTuple-path helpers in Tracers.jl are dead (only 4D impl
  remains)
- Any remaining `m_save` users

Grep for anything still referencing the old pattern. Delete.

Test: full suite.

```bash
git commit -m "Commit 6: Delete dead per-tracer code paths"
```

### Commit 7: Benchmark post-refactor + Documentation

Run benchmark on the same configurations as Commit 0:
```bash
julia --project=. scripts/benchmarks/bench_strang_sweep.jl \
    --mode=production \
    --sizes=medium,large \
    --ntracers=1,5,10,30 \
    --cfl-limits=0.4,Inf \
    > artifacts/perf/plan14/after.log
```

Compare to baseline. Document in NOTES.md:
- If ≥5% win at production: "PERF WIN"
- If <5%: "CLEANUP ONLY, perf within ±5%"
- If regression: "UNEXPECTED REGRESSION, investigate"

Update CLAUDE.md if a new insight emerged. Update
ARCHITECTURAL_SKETCH.md with actual post-14 numbers/shape.

```bash
git commit -m "Commit 7: Documentation and final benchmarks"
```

## 4.5 Test plan per commit

After EACH commit:
```bash
julia --project=. -e 'using AtmosTransport'   # compile check
julia --project=. test/runtests.jl            # core tests
```

Compare pass/fail to baseline. New failures → STOP, revert,
investigate.

For GPU:
```bash
JULIA_CUDA_USE_BINARYBUILDER=false julia --project=. test/runtests.jl
```

## 4.6 Acceptance criteria

**Correctness (hard):**
- All tests that passed in baseline pass post-refactor
- No NEW test failures beyond the 77 pre-existing
- Mass conservation exactly preserved (ULP tolerance)
- `state.tracers_raw` is `Array{FT, 4}`
- `state.tracers.CO2` continues to work via `getproperty` pathway

**Code cleanliness (hard):**
- `CellState` has `tracers_raw` and `tracer_names` fields (not
  `tracers::NamedTuple`)
- Accessor API (`ntracers`, `get_tracer`, `eachtracer`, etc.)
  exists in `src/Utils/Tracers.jl`
- `getproperty(state, :tracers)` returns `TracerAccessor`, not
  the NamedTuple
- `strang_split!` has no per-tracer loop
- `m_save` field gone from AdvectionWorkspace
- No references to `tracers::NamedTuple` pattern anywhere
- Workspace construction is n_tracers-aware throughout
  (no `0×0×0×0` 4D buffers in production paths)

**Performance (soft):**
- Per-step wall time within ±5% of baseline at production
  settings
- If Commit 0 found a measurable win at some config, it's
  preserved
- No regression at any measured config

**Alignment with composition doc:**
- Storage matches OPERATOR_COMPOSITION.md §4 exactly:
  `(Nx, Ny, Nz, Nt)` with i fastest-varying
- Accessor API matches OPERATOR_COMPOSITION.md §4 specification

**Documentation:**
- NOTES.md complete with Commit 0 findings, decisions, deferred
  observations, surprises
- ARCHITECTURAL_SKETCH.md updated if needed
- CLAUDE.md updated if a new measurement insight emerged

## 4.7 Rollback plan

Standard:
- Do not "fix forward"
- Revert to last-known-good commit
- Write the failure in NOTES.md
- Stop and ask if stuck >30 minutes

Specific rollback points:
- **Commit 0 reveals multi-tracer path has bugs.** Fix in a
  separate commit before proceeding. If large, escalate.
- **Commit 2 breaks accessor API tests.** Likely a signature
  mismatch in one of the `_*_impl` functions. Revert, fix, retry.
- **Commit 3 breaks anything.** Shouldn't happen — it only adds
  parallel storage. Revert and investigate.
- **Commit 4 breaks tests.** This is the risky commit. Most
  likely: a NamedTuple-specific pattern was missed in Commit 2
  migration. Revert, grep harder, retry.
- **Commit 5 breaks workspace construction.** `TransportModel.jl`
  or similar site missed. Fix the site, don't revert.
- **Commit 7 shows performance regression.** Unexpected. Reveals
  some cost that isn't the per-tracer loop. Investigate before
  shipping.

## 4.8 Known pitfalls

1. **"I'll make the accessor API a thin wrapper around the
   NamedTuple and skip the _*_impl indirection."** NO. The
   indirection is deliberate. In Commit 2, `_*_impl` dispatches
   on NamedTuple. In Commit 4, it gains Array{FT,4} dispatch. The
   public API is stable across both; only the impl changes.

2. **"state.tracers.CO2 is ugly; let me delete the getproperty
   layer and migrate all callers to get_tracer."** NO. Per
   Decision 4 and reviewer feedback, property access is preserved.
   The `getproperty` layer is small and preserves API continuity.

3. **"Let me use Vector{Symbol} instead of NTuple{Nt, Symbol}
   for tracer_names."** Debatable but NO per Decision 2. NTuple
   gives compile-time known Nt which helps downstream inference.

4. **"Kernels should go through the accessor API for consistency."**
   NO per Decision 6. Kernels dispatch on raw 4D Array; accessor
   API is for non-kernel code.

5. **"The 4D buffers in AdvectionWorkspace are at (0,0,0,0) in
   tests. Is that a bug?"** YES — it's what Commit 5 fixes.
   Tests that rely on n_tracers=0 should continue to work, but
   production paths through TransportModel.jl need n_tracers >= 1.

6. **"I noticed LinRood / CubedSphereStrang uses different
   patterns."** Out of scope per §3.1. Log to NOTES.md.

7. **"The face-indexed path is complex; can I skip updating
   it?"** NO — per §4.2 it's in scope. Face-indexed (RG) path
   in `StrangSplitting.jl` around line 1043 must migrate to 4D
   storage. If you find it's too complex to tackle in plan 14,
   STOP and ask for scope reduction rather than deferring
   silently.

8. **"Tests that construct AdvectionWorkspace without n_tracers
   fail."** YES — fix by updating them to pass state or
   n_tracers. Add to Commit 5 scope.

---

# Part 5 — How to Work

## 5.1 Session cadence

- **Session 1:** Precondition + Commit 0 (NOTES.md, measurement)
  — THE CRITICAL SESSION.
- **Session 2:** Commit 1 (benchmark extension)
- **Session 3:** Commit 2 (accessor API, NamedTuple pass-through)
- **Session 4:** Commit 3 (parallel 4D storage)
- **Session 5-6:** Commit 4 (the big switch — expect 2 sessions)
- **Session 7:** Commit 5 (TransportModel + workspace)
- **Session 8:** Commits 6-7 (cleanup, bench, docs)

Each session: `git status` at start, NOTES.md update at end,
decisions logged.

## 5.2 When to stop and ask

- Commit 0 shows multi-tracer path is broken
- Any test that passed in baseline now fails
- A file you need to edit isn't in §4.2 grep results
- Scope feels like it's expanding
- >30 minutes on a single issue
- You want to add a "small" optimization not in the plan
- Face-indexed path migration looks harder than expected

## 5.3 NOTES.md discipline

Update NOTES.md at these moments:
- After Commit 0: baseline measurement findings (REQUIRED)
- After each other commit: surprises, decisions, deferred
  observations
- At the end: "Template usefulness for plans 15-18" section
  capturing lessons

---

# End of Plan

After this refactor ships:
- `CellState` has 4D `tracers_raw` + `tracer_names` fields
- Storage-agnostic accessor API in `Tracers.jl`
- Property access `state.tracers.CO2` preserved via `getproperty`
- `strang_split!` has one multi-tracer pipeline
- TransportModel constructs workspaces with correct n_tracers
- Face-indexed path migrated to 4D
- Foundation laid for plans 15-18

The next plans:
- Plan 15: slow chemistry operator
- Plan 16: surface flux BCs in advection
- Plan 17: vertical diffusion
- Plan 18: convection

Each references OPERATOR_COMPOSITION.md and uses plan 14's
accessor API for tracer access.
