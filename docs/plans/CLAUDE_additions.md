# CLAUDE.md Additions — Accumulated Lessons (Plans 13-16b)

**Format:** These are ADDITIONS to an existing CLAUDE.md. Each
section below is self-contained; merge into the existing doc at
whatever section matches (Performance tips / Testing discipline /
Refactor patterns, etc.) or create new sections as needed.

**Source:** retrospective findings from NOTES13 through NOTES16b.
Each lesson is grounded in a specific surprise or failure mode
encountered during execution.

---

## § Planning discipline

### Measurement is the decision gate, not a confirmation of prediction

Every refactor plan since 13 has been wrong about performance
predictions in one direction or another:

- **Plan 13** predicted sync-removal would give a measurable win.
  Measurement showed sync was 0.02-0.4% of step time — the
  prediction was an order of magnitude too optimistic.
- **Plan 14** v3 predicted 0-10% GPU speedup from multi-tracer
  fusion (based on launch-overhead math). Measurement showed
  10-270% at production Nt=30 — the prediction was 5-10× too
  conservative.
- **Plan 16b** predicted ≤30% diffusion overhead at large grids
  (based on per-column Thomas arithmetic). Measurement showed
  65-76% at C180 Nt=10 — exceeded the soft target.

**Rule:** Commit 0 of every plan captures baseline measurements.
Those measurements are the decision gate for whether the plan's
motivation is real, not a confirmation of a pre-registered
prediction. If measurement surprises, re-frame the plan's value
proposition based on what's actually true.

**Specific estimation failure modes to avoid:**

- Launch overhead is <1% of production GPU time. Fusing to
  reduce launch count is the wrong optimization target.
- Bandwidth is ~95% of production GPU time. Fusing to reduce
  memory traffic IS the target.
- "Bandwidth math from byte counts" underweighs compiler fusion.
  A 2-3× speedup in bandwidth-dominated kernels is plausible
  when sequential operators read/write the same data.

### Survey before greenfield

Plans 15 and 16b both assumed significant greenfield work and
found substantial existing infrastructure in the repo:

- Plan 15 expected "minimal ad-hoc decay" and found a 78-line
  `src/Operators/Chemistry/Chemistry.jl` with `AbstractChemistry`,
  `RadioactiveDecay`, `CompositeChemistry`, wired into
  `DrivenSimulation`.
- Plan 16b inherited `src_legacy/Diffusion/` (four implementations,
  ~800 lines) as "starting point, not reference."

**Rule:** §4.1 of every plan includes a grep survey for existing
infrastructure in the plan's problem domain. Log results to
`artifacts/*/existing_*_survey.txt`. Revise scope before Commit 1
if the survey reveals the greenfield assumption is wrong.

**Typical patterns for survey queries:**

```bash
grep -rn "decay\|half_life\|exponential" src/ --include="*.jl"
grep -rn -i "diffus\|kz\|thomas\|tridiag" src/ --include="*.jl"
grep -rn -i "emission\|flux\|source\|BC" src/ --include="*.jl"
grep -rn -i "convection\|tiedtke\|mass_flux\|entrainment" src/ --include="*.jl"
```

### Pre-Commit-0 memory compaction

Before starting a new plan, update the AI agent's working memory
docs so downstream agents see current state:

- Update `MEMORY.md` "Current State" with the latest plan completion
- Add `planNN_complete.md` summarizing what shipped, key perf
  numbers, and latent bugs if any
- Mark stale-path warnings in old memos (e.g., `src/Advection/*`
  now in `src_legacy/`)

Plan 15 D3 introduced this. Plan 16a/16b followed. Continue.

### Scope compression is a feature, not a failure

Plan 15 shipped 6 commits against a planned 8 (skipped dead-code
removal because the original refactor cleaned it up as a side
effect). Plan 16b added Commit 1a/1b/1c split that wasn't in the
original plan.

**Rule:** NOTES.md "Deviations from plan doc §4.4" section is
mandatory. Record every commit merge, split, or skip with reason.
This is how the plan doc and reality stay in sync without
plan-doc churn.

---

## § Testing discipline

### Test contract: observe through accessor API

**Plan 14 broke test helpers** that cached `caller_array` and
expected it to reflect post-operator state. The 4D refactor meant
`state.tracers_raw` was separate storage; `caller_array` went
stale.

**Rule:** Tests observe post-operator state via
`get_tracer(state, name)` or `state.tracers.name`, never through
original input arrays. Plan 14 retroactively fixed helpers;
plans 15, 16a, 16b followed the contract.

Sample good pattern:

```julia
# GOOD
state = CellState(air_mass; CO2 = zeros(FT, Nx, Ny, Nz))
apply!(state, meteo, grid, op, dt; workspace)
@test get_tracer(state, :CO2) ≈ expected     # accessor API

# BAD
rm_CO2 = zeros(FT, Nx, Ny, Nz)
state = CellState(air_mass; CO2 = rm_CO2)
apply!(state, meteo, grid, op, dt; workspace)
@test rm_CO2 ≈ expected     # rm_CO2 is STALE after plan 14
```

### CPU/GPU dispatch: `parent(arr) isa Array`, not `arr isa Array`

Plan 14 Commit 4 broke `Face-indexed horizontal subcycling
preserves positivity` because CPU-vs-GPU dispatch used
`rm isa Array`. Post-plan-14, `rm` is a `SubArray{FT, 2, Array{FT, 3}}`
(a `selectdim` view), which fails `isa Array` and misroutes to
GPU path.

**Rule:** Dispatch code that branches on backend uses
`parent(arr) isa Array` or `get_backend(arr)`, not `arr isa Array`.

### Default kwargs must be bit-exact to explicit-default paths

Plan 16b Commit 4 introduced
`diffusion_op::AbstractDiffusionOperator = NoDiffusion()` as a
new kwarg. Critical test: default-kwarg path must be `==`
(byte-for-byte) identical to `diffusion_op = NoDiffusion()`
explicit-path. Not `≈` (ULP-close) — `==`.

Reason: if the default branch does any floating-point work, some
caller without the kwarg sees subtly different numerics.
Regression tests must catch this.

**Rule:** When adding kwargs to hot-path functions with
`NoSomething()` defaults, ship an explicit `==` regression test
comparing default path to explicit-NoSomething-path. Plan 16b
Commit 4's first testset is the pattern.

### Baseline failure count is invariant

77 pre-existing test failures across 7 test files, stable since
plan 12:

- `test_basis_explicit_core.jl`: 2
- `test_structured_mesh_metadata.jl`: 3
- `test_poisson_balance.jl`: 72

**Rule:** Every plan preserves this count. New failures →
STOP, revert, investigate. Plans that would fix any of these need
a separate scope commitment.

Baseline captured in `artifacts/<plan>/baseline_test_summary.log`
at Commit 0. Compared against after every commit.

---

## § Julia/language gotchas

### Default inner constructor handles `Real` → `FT` coercion

Adding an explicit outer constructor for `Real`-to-`FT` coercion
is a common mistake. Julia's synthesized inner constructor already
handles this:

```julia
# UNNECESSARY and causes MethodError ambiguity:
struct Foo{FT}
    x::FT
end
Foo{FT}(value::Real) where FT = Foo{FT}(FT(value))

# Works out of the box:
struct Foo{FT}
    x::FT
end
Foo{Float64}(1)    # automatic: new{Float64}(convert(Float64, 1))
```

Plan 16a hit this on `ConstantField{FT, N}`. Adding the explicit
outer constructor produced `MethodError` ambiguity with the
default inner.

**Rule:** Don't write `FT`-coercing outer constructors for
parametric structs unless the coercion is non-trivial.

### Type-parameterized defaults in kwargs evaluate at module scope

```julia
function DerivedKzField(; …,
                        params = PBLPhysicsParameters{FT}()  # WRONG
                        ) where FT
    ...
end
```

`PBLPhysicsParameters{FT}()` evaluates at module scope, not
method-body scope, so `FT` is undefined there. Produces
`UndefVarError: FT not defined`.

**Workaround:**

```julia
function DerivedKzField(; …,
                        params = nothing
                        ) where FT
    params = something(params, PBLPhysicsParameters{FT}())
    ...
end
```

Plan 16b Commit 1c hit this.

### Parametric type bounds for Adapt compatibility

For GPU dispatch via `Adapt.jl`, concrete field types storing
arrays must be parametric on the array type:

```julia
# Good: Adapt can swap the array type at kernel-launch time
struct ProfileKzField{FT, V <: AbstractVector{FT}} <: AbstractTimeVaryingField{FT, 3}
    profile::V
end

# Bad: stuck with Vector{FT}; can't convert to CuArray inside kernel
struct ProfileKzField{FT} <: AbstractTimeVaryingField{FT, 3}
    profile::Vector{FT}
end
```

Plus a short `Adapt.adapt_structure` method:

```julia
Adapt.adapt_structure(to, f::ProfileKzField) =
    ProfileKzField(Adapt.adapt(to, f.profile))
```

Plan 16b Commit 6 validated this pattern. Zero runtime cost
difference from `ConstantField`.

---

## § Operator design patterns

### `apply!` signature is stable across operator families

Plan 14 established the pattern; plan 15 validated for chemistry;
plan 16b validated for diffusion. All subsequent operators should
conform:

```julia
apply!(state::CellState,
       meteo,
       grid::AbstractGrid,
       op,
       dt::Real;
       workspace) -> state
```

**Nuances by operator:**

- Chemistry: `meteo = nothing` acceptable (pure decay)
- Advection: takes flux state through a separate kwarg path
- Diffusion: requires real workspace; `meteo = nothing` OK for
  non-meteorology-dependent Kz fields, not for `DerivedKzField`

### Coefficient structures preserved for adjoint-friendliness

Plan 16b Decision 12: tridiagonal `(a, b, c, d)` coefficients
stored as named locals per k, not pre-factored into
`w[k]`/`inv_denom[k]` form. Documents the transposition rule:
`a_T[k] = c[k-1]`, `b_T[k] = b[k]`, `c_T[k] = a[k+1]`.

**Rule:** When implementing operators whose adjoint is plausibly
useful later (sensitivity analysis, inversion), preserve the
mathematical structure even if performance optimization tempts
you to fuse it. Adjoint port becomes mechanical instead of a
rewrite.

### Dead-branch dispatch for "No<Operator>" defaults

`NoDiffusion`, `NoChemistry` are literal dead branches — `apply!`
methods that return `state` unchanged (or `nothing` for array-
level entries). Julia's multiple dispatch turns these into zero
floating-point work.

Pattern:

```julia
# Null operator
struct NoDiffusion <: AbstractDiffusionOperator end
apply_vertical_diffusion!(_, ::NoDiffusion, _, _) = nothing

# Default in TransportModel
struct TransportModel{..., Diff = NoDiffusion, ...}
    diffusion::Diff
    ...
end
```

This is what makes backward-compatibility bit-exact — when no one
opts in, the compiler sees dead code and optimizes it out.

### Array-level entry point for palindrome hooks

Plan 16b Commit 4 introduced `apply_vertical_diffusion!(q_raw, op, ws, dt)`
as a lower-level entry point called from inside the palindrome.
The state-level `apply!` delegates. Reason: the palindrome
operates on whichever ping-pong buffer currently holds the tracer
state — not always `state.tracers_raw`.

**Rule:** When an operator needs to be called both at step level
(on `state`) AND at sub-step level (on a raw array buffer),
provide both entry points. State-level delegates to array-level.

---

## § Plan execution rhythm

### Commit 0: NOTES + baseline, no source changes

Every plan's Commit 0 is:

1. Create `docs/plans/<PLAN>/NOTES.md` with baseline section
2. Run precondition verification (§4.1 grep, test suite)
3. Capture baseline test summary to `artifacts/`
4. Record baseline git hash
5. Run existing-infrastructure survey for scope verification
6. Memory compaction (update MEMORY.md, etc.)

No source changes. This is the "last clean commit" for rollback.

### Commit sequence is draft, not contract

Plan docs list expected commit sequences. Reality compresses, splits,
and reorders. Plan 15: 8→6. Plan 16b: 8→9 (Commit 1 split into 1a/1b/1c).

NOTES.md's "Deviations from plan doc §4.4" section captures all
changes. Plan doc stays as original design intent; NOTES is source
of truth for what shipped.

### Rollback point discipline

Commits should be individually revertable. If Commit 4 breaks,
`git revert <commit-4>` should restore Commit-3 state cleanly,
with tests passing.

**Anti-pattern:** A Commit 4 that depends on a fix in Commit 5.
Reverting Commit 4 alone leaves the repo broken. Ship the fix in
Commit 4 itself or restructure commits.

### Retrospective sections are cumulative, not start-at-end

NOTES.md's "Decisions beyond the plan", "Surprises", "Interface
validation findings", "Template usefulness for plans N+" sections
should be filled in AS execution proceeds, not all at once at
Commit N. Future plans reading the NOTES want the narrative of
discovery, not a polished summary.

Exception: minor language editing at plan completion is fine.

---

## § Branch hygiene (emerging from plans 11-16b)

### Stack plan branches, don't parallel them

Plans 14 → 15 → 16a → 16b each forked from the previous plan's
tip. This creates a linear chain:

```
main ← advection-unification ← slow-chemistry ← time-varying-fields ← vertical-diffusion
```

Each plan branch is self-contained and can be reviewed as a PR
against the prior plan's tip. Merge to `main` in order when
reviewing catches up.

**Rule:** Don't branch from `main` for a plan N when plans before
N haven't merged. Chain off the latest plan's tip instead. Keep
history linear.

### Merge staging branches to `main` when the abstraction stabilizes

Don't merge every plan to `main` immediately — intermediate states
can be internally inconsistent. But don't wait for the entire
refactor suite either — `main` falls behind and misses real
production value.

Guideline: merge when a natural grouping of plans reaches a
stable API boundary. Plans 11-14 (advection refactor) = one
logical unit. Plans 15-16b (operator abstractions) = another.
Plan 17 (emissions) + palindrome updates = another.

---

## § What NOT to do

From accumulated plan surprises:

- **Do NOT** estimate GPU perf from launch-overhead math. Bandwidth dominates.
- **Do NOT** add `Real → FT` coercing outer constructors for parametric structs.
- **Do NOT** write `params = SomeParametric{FT}()` as a kwarg default in a `where FT` method.
- **Do NOT** cache `caller_array` in test helpers expecting `===` identity after operator.
- **Do NOT** dispatch backend on `arr isa Array` when views are possible.
- **Do NOT** pre-factor tridiagonal coefficients if adjoint port is future work.
- **Do NOT** assume greenfield — survey the repo first.
- **Do NOT** merge plan branches to `main` before the logical grouping is stable.
- **Do NOT** skip Commit 0 measurement because the plan seems low-risk.
- **Do NOT** write retrospective sections only at plan completion — fill in during.

---

## § Version note

This document accumulates lessons from plans 13 (sync removal),
14 (advection unification), 15 (slow chemistry), 16a (TimeVaryingField),
16b (vertical diffusion). Update after each major plan completion
with specific retrospective findings, not generic advice.

**End of additions.**
