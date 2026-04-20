# CLAUDE.md Additions — Accumulated Lessons (Plans 13-17)

**Format:** These are ADDITIONS to an existing CLAUDE.md. Each
section below is self-contained; merge into the existing doc at
whatever section matches (Performance tips / Testing discipline /
Refactor patterns, etc.) or create new sections as needed.

**Source:** retrospective findings from NOTES13 through NOTES17.
Each lesson is grounded in a specific surprise or failure mode
encountered during execution.

**v2 update (post-plan-17):** Added validation discipline section
(§ "Validation discipline for physics ports") after recognizing
that plan 16b's "validate against legacy" approach has a blind
spot. Also added three plan-17 Julia gotchas.

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
- **Plan 17** predicted ≤5% emissions overhead. Measurement
  showed |Δ%| < 1% — two orders of magnitude below target, because
  the kernel is one read + one multiply-add + one write per
  emitting cell at the surface slab.

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
- Small-kernel operators (like emissions) have per-cell cost
  that's dwarfed by advection noise at production Nt. Don't
  over-budget for overhead on pointwise operators.

### Survey before greenfield

Plans 15, 16b, and 17 all assumed significant greenfield work
and found substantial existing infrastructure in the repo:

- Plan 15 expected "minimal ad-hoc decay" and found a 78-line
  `src/Operators/Chemistry/Chemistry.jl` with `AbstractChemistry`,
  `RadioactiveDecay`, `CompositeChemistry`, wired into
  `DrivenSimulation`.
- Plan 16b inherited `src_legacy/Diffusion/` (four implementations,
  ~800 lines) as "starting point, not reference."
- Plan 17 expected "minimal existing emissions code" and found
  active `SurfaceFluxSource` infrastructure in
  `src/Models/DrivenSimulation.jl` with `cell_mass_rate::RateT`
  path plus a second (dead) path at `_inject_source_kernel!`.

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

Also survey `src_legacy/` for substantive prior art that may
become the port basis, not just the modern `src/`.

### Pre-Commit-0 memory compaction

Before starting a new plan, update the AI agent's working memory
docs so downstream agents see current state:

- Update `MEMORY.md` "Current State" with the latest plan completion
- Add `planNN_complete.md` summarizing what shipped, key perf
  numbers, and latent bugs if any
- Mark stale-path warnings in old memos (e.g., `src/Advection/*`
  now in `src_legacy/`)

Plan 15 D3 introduced this. Plans 16a, 16b, 17 followed. Continue.

### Scope compression is a feature, not a failure

Plan 15 shipped 6 commits against a planned 8 (skipped dead-code
removal because the original refactor cleaned it up as a side
effect). Plan 16b added Commit 1a/1b/1c split that wasn't in the
original plan. Plan 17 compressed Commits 2-3 because existing
`SurfaceFluxSource` had to move before any Operator could
reference it.

**Rule:** NOTES.md "Deviations from plan doc §4.4" section is
mandatory. Record every commit merge, split, or skip with reason.
This is how the plan doc and reality stay in sync without
plan-doc churn.

### Follow-up candidates registered at Commit 0

Plan 17 introduced a practice worth keeping: at Commit 0, the
NOTES include a "Follow-up plan candidates" section listing work
explicitly out of scope. This serves three purposes:

1. Forces explicit scoping decisions rather than silent scope creep
2. Gives retrospective (Commit N) a pre-registered list to close
3. Becomes seed material for future plans

Plan 17's list: `AbstractLayerOrdering`, per-area flux variant,
stack emissions, `DepositionOperator`, `_inject_source_kernel!`
cleanup. Each is a plausible follow-up plan. Carry forward.

---

## § Validation discipline for physics ports

**(New in v2, post-plan-17.)** Plan 16b's `DerivedKzField` port
validated against legacy Julia: "The physics port line-matches
the legacy reference; deviations are limited to scope." This is
a TIGHTER circle than validating against actual atmospheric
physics. If legacy has a bug — say, a sign error in Obukhov
length, a wrong Prandtl-inverse formula — plan 16b's tests would
pass while shipping the same bug.

For CATRINE intercomparison use, this is not an academic concern.
The whole point is to compare transport implementations. An
inherited legacy bug would pass our tests but fail the
intercomparison — or worse, silently skew results.

### Three-tier validation hierarchy for physics ports

Every physics port ships tests at three levels:

**Tier A — Analytic expectations.** Mass conservation, positivity,
convergence rates, limiting cases. These don't depend on any
reference implementation. If they fail, the port is wrong.
Examples: zero forcing → identity; uniform initial state →
remains uniform under symmetric forcing; conserved quantities
preserved to machine precision under time integration.

**Tier B — Literature / paper formulas.** Hand-expand the paper's
equations for a specific test configuration, compute expected
results, compare to the port. Verifies the port implements what
the paper describes — not what a prior port thought the paper
described. Sources: Beljaars & Viterbo 1998 for PBL diffusion,
Moorthi & Suarez 1992 for RAS, Tiedtke 1989 for mass-flux
convection, etc.

**Tier C — Cross-implementation comparison.** For a standard test
case, run the port alongside (a) legacy Julia, (b) upstream
Fortran reference (TM5 `tm5_conv.F90`, GEOS-Chem
`convection_mod.F90`, ECMWF IFS documentation), (c) independent
implementation if available. Differences should be small (~5-10%
on column-integrated quantities, similar qualitative shapes). If
one implementation stands apart from the others, IT IS SUSPECT —
investigate before shipping.

Plan 16b shipped Tier A and a version of Tier B, but Tier B was
against LEGACY'S hand expansion of formulas, not the paper's
direct formulas. And Tier C was skipped.

### Source-of-truth hierarchy

When porting from one codebase to another, there are three
possible references:

1. **The paper.** Authoritative for physics, abstract, sometimes
   ambiguous on edge cases and numerical choices.
2. **Upstream code.** Concrete, testable. TM5 `tm5_conv.F90`,
   GEOS-Chem `convection_mod.F90`, ECMWF IFS source. May have
   bugs but has been exercised at production scale.
3. **The immediate predecessor.** What we're porting from. Most
   proximate, most likely to have accumulated bugs through
   successive translations.

**The right hierarchy is 1 > 2 > 3.** Paper defines physics;
upstream code clarifies edge cases; immediate predecessor is a
starting point, not ground truth.

**Rule:** When porting legacy physics:
- Start from paper equations
- Cross-check against upstream code for edge cases (TM5 Fortran,
  GEOS-Chem Fortran)
- Treat immediate predecessor (`src_legacy/`) as proximate
  reference — faithful reproduction is fine but don't mistake it
  for validation
- Document any divergence from legacy as "investigated; fixed per
  [paper/Fortran reference]" or "preserved; matches legacy and
  paper"

### What this means in practice

For a physics port plan:

**Commit 0 survey adds:** reading the upstream Fortran source, not
just the legacy Julia. If there's a paper, linking it in NOTES.
Example: plan 18 Commit 0 should include reading
`tm5_conv.F90:32-186` (TM5 matrix builder) and
`convection_mod.F90:DO_CONVECTION` (GEOS-Chem driver) before any
Julia is written.

**Test suite includes all three tiers:**

- Tier A: 3-5 tests on analytic properties
- Tier B: 5-10 tests expanding paper formulas for idealized
  columns and comparing port output
- Tier C: 1-3 tests comparing port against upstream implementation
  for a standard reference case. If upstream isn't runnable,
  compare against a PUBLISHED reference calculation (paper figure,
  benchmark dataset) instead.

**Retrospective documents validation depth:** NOTES.md captures
which tiers shipped, any divergences from legacy that were
investigated, any bugs found in legacy and fixed in the port.

### When legacy is NOT a faithful reference

Explicit permission, codified from plan 18 scoping conversation:
"We won't do any wet scavenging at all (yet), should make the
kernels cleaner now though than legacy, which might also still
have bugs."

Physics port plans should state up front whether the intent is:
- **Faithful port:** preserve legacy behavior bit-exact. Tests
  require legacy-match. Use when legacy is battle-tested and
  deployed at scale.
- **Reference port:** legacy is starting point. Port follows
  paper/upstream. Legacy disagreements are investigated and
  resolved toward paper. Use when legacy is unvetted or suspect.

Default to reference port unless there's a reason to require
faithful. CATRINE-use-case code probably wants reference.

### Case study: plan 18 convection port findings

Plan 18 is the first plan to execute the full three-tier
validation discipline during its design phase (pre-Commit 0).
Reading TM5's `tm5_conv.F90` and GCHP's `convection_mod.F90`
alongside the corresponding legacy Julia ports surfaced three
concrete findings that illustrate the discipline's value:

**Finding 1 (TM5 side): Legacy matrix builder IS faithful.**
Spot-check of `tm5_matrix_convection.jl:60-122` against
`tm5_conv.F90:37-191`, accounting for Julia's +1 index shift
from Fortran's 0-based arrays, confirmed the math is correctly
ported. Including the subtle "Sander Houweling correction" to
the subsidence term assembly (Fortran line 145-147) — a
historical bug fix in TM5 that an incautious port could have
reverted.

→ Validation outcome: plan 18 can do LIGHT cleanup of
TM5Convection. No new bugs to catch.

**Finding 2 (GCHP side): Legacy tendency simplification IS
correct, by non-obvious algebraic identity.**
First read of `ras_convection.jl:188-193` looked like a
simplification error — legacy Julia uses a 2-term tendency
while GCHP's `convection_mod.F90:988-1007` uses a 4-term
tendency with two different CMFMC values. Careful algebraic
derivation (see 18_CONVECTION_UPSTREAM_GCHP_NOTES.md §5.3) showed
the 2-term form is mathematically equivalent to the 4-term form
for inert tracers, via substituting the updraft balance relation
`CMFMC_BELOW·old_QC = CMOUT·new_QC - ENTRN·Q(K)`.

→ Validation outcome: plan 18 adopts the 2-term form (correct
for inert). Scavenging hook TODOs placed at BOTH sites where
the 4-term form would need restoration when wet deposition is
added.

**Finding 3 (GCHP side): Legacy MISSING well-mixed sub-cloud
layer.** GCHP `convection_mod.F90:742-782` implements a
pressure-weighted well-mixed treatment below cloud base that
legacy `ras_convection.jl` lacks entirely. This is likely a
real bug for CATRINE surface-source tracers (Rn-222 especially).

→ Validation outcome: plan 18 ADDS the sub-cloud layer
treatment as deliberate improvement over legacy. Tier C test
in Commit 3 verifies the port with sub-cloud fix agrees better
with GCHP output than the port without.

**What these findings show:**

1. **Three tiers catch different bug classes.**
   - Tier A (analytic) catches basic correctness (mass conservation).
   - Tier B (paper formulas) catches algebraic simplification errors.
   - Tier C (cross-implementation) catches missing features and
     sign convention errors.
   All three are needed.

2. **"Suspicious pattern" is not "confirmed bug."** The 2-term
   tendency LOOKED wrong for 20 minutes of comparison, then
   turned out to be equivalent by algebra. This is why Tier B
   = hand-expanding paper formulas matters — it forces the
   comparison to be mathematical, not textual.

3. **Genuine bugs look innocuous.** The missing sub-cloud layer
   was in legacy for no obvious reason — just an omission. No
   warning flag, no FIXME comment, no obvious symptom in unit
   tests that don't test surface-source tracers. Only
   systematic comparison against upstream surfaced it.

**Recommendation:** every physics port plan from plan 18 forward
includes a section in its NOTES titled "Legacy findings from
upstream cross-reference" with discovered items classified as
"verified correct," "simplified correctly," or "bug fixed in
port." Future plans inherit this pattern.

### Basis conventions vary by model — document them explicitly

Plan 18 surfaced a subtle class of issue beyond "is the math
correct" — **different upstream models use different physical
bases for the same-named fields.** GCHP uses dry-air basis
throughout tracer transport; TM5 uses moist-air basis. Same
variable name (e.g., "mass flux") means different physical
quantities in the two systems.

This matters because:

1. **Cross-scheme comparison requires matched bases.** If you
   compare GCHP's dry-basis CMFMC convection against TM5's
   moist-basis matrix convection on the same raw data, you're
   mixing two error sources: the discretization difference
   (which you want to measure) and the basis difference
   (which you don't).

2. **Comments in port code can lie about basis.** Plan 18 found
   `ras_convection.jl:41-46` claiming "No dry conversion is
   applied before convection" — this contradicted the legacy's
   OWN dry-correction kernels (`latlon_dry_air.jl`). Either the
   comment was wrong from the start, or code evolved and the
   comment wasn't updated. Either way: stale comment = bug
   attractor.

3. **Scheme defaults are silent.** GCHP's `DELP_DRY` vs `DELP`
   distinction looks like a variable name choice but has
   physics content. Reading `BMASS = DELP_DRY * G0_100` without
   also reading the comment block in `calc_met_mod.F90:185-200`
   would miss that DELP_DRY is SPECIFICALLY chosen for transport
   consistency with the pressure fixer.

**Rule:** physics port plans explicitly document which basis
each operator expects. In docstrings, not just in separate
design docs. Example from plan 18 Decision 18:

```julia
"""
    CMFMCConvection(cmfmc, dtrain)

GEOS-Chem RAS/Grell-Freitas convective transport on DRY-AIR BASIS.

# Field expectations

- `cmfmc`: dry-corrected CMFMC at interfaces. kg dry-air / m² / s.
- `dtrain`: dry-corrected DTRAIN at centers. kg dry-air / m² / s.

If fields are from raw GEOS met (moist basis), apply dry-air
correction upstream via `apply_dry_cmfmc!` / `apply_dry_dtrain!`
before wrapping.

Tracer state assumed dry-basis (kg tracer per kg dry air).
"""
```

**Rule (continued):** when porting operators whose upstream
reference exists in multiple models with different bases, the
plan's Commit 0 survey MUST check basis convention for each
reference. Add to NOTES: "Basis convention for `X` in [model]:
[dry/moist], verified at [file:lines]."

**Applied to existing operators:** plan 16b's
`ImplicitVerticalDiffusion` port did not document basis
expectation for Kz fields or tracer state. Plan 16c follow-up
should add this audit:

- Does `ProfileKzField` / `PreComputedKzField` / `DerivedKzField`
  expect dry-basis input?
- Does `ImplicitVerticalDiffusion`'s column Thomas operate on
  dry-basis or moist-basis tracer?
- Are the tests (even Tier A mass conservation) sensitive to
  basis mismatch?

Any operator that interacts with other operators through a
shared tracer state IS sensitive to basis consistency across
the suite.

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
plans 15, 16a, 16b, 17 followed the contract.

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
new kwarg. Plan 17 Commit 5 added
`emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux()`
similarly. Critical test: default-kwarg path must be `==`
(byte-for-byte) identical to `op = NoXxx()` explicit-path.
Not `≈` (ULP-close) — `==`.

Reason: if the default branch does any floating-point work, some
caller without the kwarg sees subtly different numerics.
Regression tests must catch this.

**Rule:** When adding kwargs to hot-path functions with
`NoSomething()` defaults, ship an explicit `==` regression test
comparing default path to explicit-NoSomething-path. Plan 16b
Commit 4's and Plan 17 Commit 5's first testsets are the pattern.

### Runtime `if op isa NoOperator` branches compile away

Plan 17 Commit 5 surprise: the palindrome's
`if emissions_op isa NoSurfaceFlux ... else ...` is a RUNTIME
type check, but Julia's compiler constant-folds it away when
`strang_split_mt!` is invoked from `apply!` with a concretely-
typed operator. Performance is identical to a type-level
specialization.

**Rule:** For Option A dispatch patterns (branch on
`op isa NoOperator` to preserve pre-feature bit-exactness), don't
worry about runtime branch cost. Concretely-typed operator at
call site → compiler eliminates the branch.

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

## § Julia / language gotchas

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

### `Ref{Int}` is not kernel-safe

**(New in v2, from plan 17 Commit 1.)** For mutable scalar state
inside a kernel-facing struct, don't use `Base.RefValue{Int}` —
Adapt carries a host-side Ref into kernel closures, and
subsequent `current_window[]` reads on GPU pull from host memory
or error.

**Fix:** use a 1-element `AbstractVector{Int}`:

```julia
# BAD — not kernel-safe on GPU
struct StepwiseField{FT, N, A, T}
    samples::A
    boundaries::T
    current_window::Base.RefValue{Int}    # Ref, host-only
end

# GOOD — adapts naturally to CuArray{Int, 1}
struct StepwiseField{FT, N, A, T, W}
    samples::A
    boundaries::T
    current_window::W   # ::AbstractVector{Int}, 1-element
end
```

`Adapt.adapt_structure` converts the 1-element vector to the
device backend transparently. Plan 17 hit this on
`StepwiseField.current_window`. Plan 18's convection operator
will hit this if its scheme needs time-varying closure state.

### `Adapt.adapt_structure` cannot use validating inner constructors

**(New in v2, from plan 17 Commit 1.)** Inner constructors that
call host-side operations on the stored fields (e.g.,
`issorted(boundaries)`) break when `Adapt.adapt_structure`
reconstructs the struct with device-memory fields — the
validation tries to iterate device memory on the host and errors.

**Fix:** Add a second inner constructor gated by a `Val` sentinel
that bypasses validation. `Adapt.adapt_structure` calls the
unchecked variant; all other construction paths go through the
validating variant.

```julia
struct StepwiseField{FT, N, A, T, W} <: AbstractTimeVaryingField{FT, N}
    samples::A
    boundaries::T
    current_window::W

    # Validating constructor for normal use
    function StepwiseField{FT, N, A, T, W}(samples, boundaries, current_window) where {FT, N, A, T, W}
        issorted(boundaries) || throw(ArgumentError("boundaries must be sorted"))
        length(boundaries) == size(samples, N+1) + 1 ||
            throw(DimensionMismatch("..."))
        new{FT, N, A, T, W}(samples, boundaries, current_window)
    end

    # Unchecked constructor for Adapt.adapt_structure
    function StepwiseField{FT, N, A, T, W}(
        ::Val{:unchecked}, samples, boundaries, current_window
    ) where {FT, N, A, T, W}
        new{FT, N, A, T, W}(samples, boundaries, current_window)
    end
end

Adapt.adapt_structure(to, f::StepwiseField{FT, N}) where {FT, N} = begin
    samples_new = Adapt.adapt(to, f.samples)
    boundaries_new = Adapt.adapt(to, f.boundaries)
    current_window_new = Adapt.adapt(to, f.current_window)
    StepwiseField{FT, N, typeof(samples_new), typeof(boundaries_new),
                  typeof(current_window_new)}(Val(:unchecked),
                                              samples_new, boundaries_new,
                                              current_window_new)
end
```

**Rule:** Any field type that validates its inputs AND is
Adapt-compatible needs both constructors. Plan 18 convection
operators with derived-met-field inputs will hit this.

### Three-arg outer constructor for Adapt round-trip

**(New in v2, from plan 17 Commit 1.)** `Adapt.adapt_structure`'s
generic reconstruction calls the outer constructor positionally
with N arguments matching N fields. If the outer constructor is
defined with fewer-than-N arguments (e.g., synthesizing a missing
field like `current_window = [1]`), the reconstruction fails.

**Fix:** Provide outer constructors at ALL necessary arities —
including the fully-positional arity matching the struct's
declared fields.

```julia
# 2-arg outer (convenience for users)
StepwiseField(samples::AbstractArray{FT, M}, boundaries) where {FT, M} = begin
    StepwiseField{FT, M-1, ...}(samples, boundaries, [1])
end

# 3-arg outer (needed for Adapt round-trip)
StepwiseField(samples::AbstractArray{FT, M}, boundaries, current_window) where {FT, M} = begin
    StepwiseField{FT, M-1, ...}(samples, boundaries, current_window)
end
```

Plan 17 hit this when `Adapt.adapt_structure` tried to reconstruct
a `StepwiseField` and the generic path called the 3-arg form,
which didn't exist.

**Rule:** For structs with N fields that round-trip through
Adapt, provide an N-arg outer constructor (positional, matches
field declaration order).

---

## § Operator design patterns

### `apply!` signature is stable across operator families

Plan 14 established the pattern; plan 15 validated for chemistry;
plan 16b validated for diffusion; plan 17 validated for surface
flux. All subsequent operators should conform:

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
- Surface flux: requires real `meteo` (for `current_time`);
  `tracer_names` passed as kwarg to array-level entry point

### `current_time(meteo)` threading pattern

Plan 16b Commit 5 shipped the `current_time(::AbstractMetDriver)`
stub returning 0.0. Plan 17 Commit 4 threaded it through operator
`apply!` methods uniformly:

```julia
t = meteo === nothing ? zero(FT) : FT(current_time(meteo))
```

This accepts `nothing` for test-helper compatibility (unit tests
that don't care about time) and falls back to real meteorology
time when available.

**Rule:** Any operator `apply!` that calls `update_field!` on a
time-varying field uses this idiom. Plan 18 convection operators
follow it.

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

`NoDiffusion`, `NoChemistry`, `NoSurfaceFlux` are literal dead
branches — `apply!` methods that return `state` unchanged (or
`nothing` for array-level entries). Julia's multiple dispatch
turns these into zero floating-point work.

Pattern:

```julia
# Null operator
struct NoDiffusion <: AbstractDiffusionOperator end
apply_vertical_diffusion!(_, ::NoDiffusion, _, _, _) = nothing

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
Plan 17 Commit 3 followed the pattern with
`apply_surface_flux!(q_raw, op, ws, dt, meteo, grid; tracer_names)`.

Reason: the palindrome operates on whichever ping-pong buffer
currently holds the tracer state — not always
`state.tracers_raw`. Additional kwargs (like `tracer_names`)
let the entry point work without a CellState reference.

**Rule:** When an operator needs to be called both at step level
(on `state`) AND at sub-step level (on a raw array buffer),
provide both entry points. State-level delegates to array-level.

### Option A dispatch: preserve bit-exact pre-feature behavior

Plan 17 Commit 5 pattern: when adding a new operator position
into an existing palindrome that previously had a different
structure, dispatch on the new operator's type to choose:

- `op isa NoOperator` → execute the original structure bit-exact
- otherwise → execute the new structure

Example from plan 17:

```julia
if emissions_op isa NoSurfaceFlux
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt)   # plan 16b behavior
else
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2) # plan 17 behavior
    apply_surface_flux!(rm_cur, emissions_op, ws, dt, meteo, grid; tracer_names)
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2)
end
```

This is how plan 17 preserved plan 16b's `test_diffusion_palindrome.jl`
bit-exact (27 tests unchanged) while adding new behavior.

**Rule:** When a plan modifies an existing structure (palindrome
insertion, kernel replacement, pipeline extension) in a way that
changes numerics, use Option A dispatch to preserve pre-plan
behavior when the new feature is disabled. First testset of the
modifying commit is an `==` (not `≈`) regression against the
pre-plan baseline.

---

## § Plan execution rhythm

### Commit 0: NOTES + baseline, no source changes

Every plan's Commit 0 is:

1. Create `docs/plans/<PLAN>/NOTES.md` with baseline section
2. Run precondition verification (§4.1 grep, test suite)
3. Capture baseline test summary to `artifacts/`
4. Record baseline git hash
5. Run existing-infrastructure survey for scope verification
6. For physics ports: read upstream Fortran / paper references
7. Memory compaction (update MEMORY.md, etc.)
8. Register follow-up plan candidates (plan 17 pattern)

No source changes. This is the "last clean commit" for rollback.

### Commit sequence is draft, not contract

Plan docs list expected commit sequences. Reality compresses, splits,
and reorders. Plan 15: 8→6. Plan 16b: 8→9 (Commit 1 split into 1a/1b/1c).
Plan 17: 9→9 with reshuffled content between Commits 2 and 3.

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

## § Branch hygiene (emerging from plans 11-17)

### Stack plan branches, don't parallel them

Plans 14 → 15 → 16a → 16b → 17 each forked from the previous plan's
tip. This creates a linear chain:

```
main ← advection-unification ← slow-chemistry ← time-varying-fields ← vertical-diffusion ← surface-emissions
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
Plan 17 (emissions + chemistry workaround cleanup) = another.

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
- **(New in v2)** Do NOT use `Base.RefValue{Int}` for kernel-facing mutable state — use 1-element `AbstractVector{Int}`.
- **(New in v2)** Do NOT validate against legacy as the sole source of truth for physics ports — consult paper, upstream Fortran, and cross-reference.
- **(New in v2)** Do NOT skip the three-tier test structure (Analytic / Literature / Cross-reference) for physics port plans.
- **(New in v2)** Do NOT omit the fully-positional outer constructor from Adapt-compatible structs — it breaks round-trip.

---

## § Version note

This document accumulates lessons from plans 13 (sync removal),
14 (advection unification), 15 (slow chemistry), 16a (TimeVaryingField),
16b (vertical diffusion), 17 (surface emissions), 18 design (convection).

**v1** (post-plan-16b): initial accumulated lessons.

**v2** (post-plan-17): added validation discipline section,
three plan-17 Julia gotchas (Ref{Int}, Val-gated validation
bypass, 3-arg outer constructor), `current_time(meteo)` threading
pattern, Option A dispatch pattern.

**v2 extensions** (post-plan-18 design, pre-execution): added
"Case study: plan 18 convection port findings" showing three
concrete findings from the three-tier validation discipline in
action. Added "Basis conventions vary by model — document them
explicitly" covering the dry-vs-moist issue that surfaced during
plan 18's GCHP/TM5 comparison.

These v2 extensions are pre-execution findings from the design
phase of plan 18. If plan 18 execution surfaces more lessons,
v3 will incorporate them.

Update after each major plan completion with specific
retrospective findings, not generic advice.

**End of additions.**
