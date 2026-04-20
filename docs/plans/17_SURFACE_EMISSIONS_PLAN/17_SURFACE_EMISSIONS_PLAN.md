# Surface Emissions + Ordering Study — Implementation Plan for Plan Agent (v1)

**Status:** Ready for execution after plan 16b has shipped.
**Target branch:** new branch from `vertical-diffusion` (or
wherever 16b's final tip lives). Verify in §4.1.
**Estimated effort:** 2 weeks, single engineer / agent.
**Primary goal:** Implement surface emissions as a palindrome-
centered operator conforming to `OPERATOR_COMPOSITION.md` §6. Ship
`StepwiseField{FT, N}` for CATRINE-style monthly/annual inventories.
Integrate into palindrome per composition doc. Run the ordering
study that quantifies layer-1 pileup under different V-S orderings.
Resolve the plan-15 chemistry-at-sim-level workaround by folding
emissions into the transport block.

Self-contained document. ~30 minutes to read.

**Dependencies verified:**
- Plan 11-16b shipped
- `AbstractTimeVaryingField` + `ConstantField` + three rank-3 Kz
  types in place (plans 16a/16b)
- `ImplicitVerticalDiffusion` with palindrome `V(dt)` at
  `X Y Z V Z Y X` center (plan 16b Commit 4)
- `apply_vertical_diffusion!` array-level entry point established
  (plan 16b Commit 4)
- `TransportModel.diffusion` field + `with_diffusion` helper
  (plan 16b Commit 5)
- `current_time(meteo)` accessor stub (plan 16b Commit 5, stub
  returns 0.0; plan 17 ships the threading)
- Chemistry at DrivenSimulation level via
  `with_chemistry(model, NoChemistry())` workaround (plan 15);
  plan 17 resolves this

**Companion documents:**
- `OPERATOR_COMPOSITION.md` §3.1 (step-level), §3.2 (transport
  block), §8 (surface flux application) — authoritative
- `TIME_VARYING_FIELD_MODEL.md` — for `StepwiseField` shape
- `ARCHITECTURAL_SKETCH.md` v2 — current architecture
- `CLAUDE_additions.md` — accumulated planning discipline
- Plan 15 NOTES — chemistry workaround this plan resolves
- Plan 16b NOTES — palindrome integration pattern this plan extends

---

# Revisions note

v1 (this version). Written after plan 16b's shipping. Incorporates
all lessons from plans 13-16b retrospectives (see CLAUDE_additions.md).

Key design context from the retrospectives:

1. **`StepwiseField` is new to this plan.** 16a shipped only
   `ConstantField`; 16b shipped `ProfileKzField`,
   `PreComputedKzField`, `DerivedKzField`. CATRINE monthly/annual
   inventories need `StepwiseField`.
2. **Palindrome extension pattern established.** 16b added
   `diffusion_op` kwarg to `strang_split_mt!` with `NoDiffusion()`
   default. Plan 17 adds `emissions_op` the same way.
3. **`current_time(meteo)` stub exists but not threaded.** Plan
   17's emissions operator is the first user that NEEDS non-zero
   time (monthly windows). Plan 17 ships the threading.
4. **Plan 15 chemistry workaround to resolve.** Currently
   `DrivenSimulation` strips chemistry from the wrapped model and
   applies it at sim level. Plan 17 folds emissions into the
   transport block, after which `TransportModel.step!`'s
   `transport → chemistry` is the correct TM5 order without
   workarounds.

---

# Part 1 — Orientation

## 1.1 The problem in two paragraphs

AtmosTransport needs surface emissions for CATRINE runs (fossil
CO2 monthly, SF6 annual × NOAA growth rates, Rn-222 monthly, total
CO2 biosphere monthly or finer). The plan-15 retrospective
identified the ordering tension: TM5 applies
`advection → emissions → chemistry`, but post-plan-15 the model
applies `transport (adv+diff) → chemistry` inside
`TransportModel.step!`. The workaround strips chemistry from the
model and applies it at sim level after emissions.

Plan 17 resolves this by implementing emissions as an S operator
inside the transport block's palindrome:
`X Y Z V S V Z Y X`. Emissions enter at palindrome center,
wrapped by two V half-steps so fresh emissions mix vertically
before being horizontally advected on the reverse. This matches
TM5 physics while cleaning up the chemistry-at-sim-level
workaround.

## 1.2 What plan 17 is really about

Three distinct pieces of work:

1. **Infrastructure:** ship `StepwiseField{FT, N}` as a
   `TimeVaryingField` concrete type. Thread `current_time(meteo)`
   through `apply!` pathways.
2. **Operator:** implement `SurfaceFluxOperator` conforming to the
   apply! interface. Compose with a per-tracer
   `PerTracerFluxMap{FT, 2}` of 2D flux fields.
3. **Ordering study:** quantify layer-1 pileup under three palindrome
   arrangements:
   - `X Y Z V S V Z Y X` (emissions between two V half-steps,
     recommended by OPERATOR_COMPOSITION.md §3.2)
   - `X Y Z V S Z Y X` (emissions after V; single V call)
   - `X Y Z S V Z Y X` (emissions before V; single V call)
   - `X Y Z S Z Y X` (emissions without V — pathological baseline)

   Generate plots and recommend an operational default.

## 1.3 Scope keywords

- **Operator types:** `AbstractSurfaceFluxOperator`, `NoSurfaceFlux`,
  `SurfaceFluxOperator`
- **Flux map:** `PerTracerFluxMap{FT}` (dictionary-like mapping
  from tracer name to `AbstractTimeVaryingField{FT, 2}`)
- **Field types shipped:** `StepwiseField{FT, N}` (piecewise-
  constant window averages)
- **Palindrome position:** `X Y Z V S V Z Y X` (recommended)
- **Study scope:** quantitative ordering comparison at medium/large
  resolution

## 1.4 What plan 17 defers

- `LinearInterpolatedField{FT, N}` — linear between instantaneous
  snapshots (future plan, when needed)
- `IntegralPreservingField{FT, N}` — smooth + preserves integrals
  (future plan, when needed; CATRINE doesn't need it)
- `ScaledField` / `MaskedField` — wrappers (future plan)
- Stack emissions (3D sources) — future plan
- Dry deposition as a separate operator — future plan
- Non-local diffusion (Holtslag-Boville) — future plan
- Convection C — plan 18

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before:
- Emissions handled ad hoc somewhere (or not at all; see survey
  in §4.1)
- Chemistry-at-sim-level workaround from plan 15 in place
- No `StepwiseField` concrete type (CATRINE monthly emissions
  have no clean abstraction)
- `current_time(meteo)` stub exists but no operator uses real time

After:
- `SurfaceFluxOperator` is a first-class operator in the transport
  block
- `StepwiseField{FT, 2}` carries monthly/annual emission inventories
- `current_time(meteo)` threaded through `apply!` uniformly
- Chemistry workaround removed; `TransportModel.step!` runs
  `transport (adv+diff+emissions) → chemistry` natively
- Ordering study recommends an operational palindrome arrangement

## 2.2 What this enables

- **CATRINE runs.** First time the model can run the full CATRINE
  protocol (surface fluxes from GridFEDv2024.1, EDGARv8.0, Zhang
  Rn-222 inventories).
- **Plan 18 (convection).** Builds on a palindrome with S in place.
  The convection C slot fits naturally alongside or wrapped by V
  (physics-dependent; ordering study preview may hint).
- **Plan 18 meteorology integration.** `DerivedKzField` end-to-end
  validation (currently deferred per plan 16b) becomes possible
  with real time + operational `dz_scratch` filling.

## 2.3 The physics of the ordering question

Surface emissions enter the bottom model layer. If advection runs
BEFORE vertical mixing, horizontal transport moves emitted tracer
along the surface before it has a chance to mix vertically. Real
atmosphere does both simultaneously: boundary-layer turbulence
timescale ~30 minutes, horizontal wind transport across a 100 km
cell at surface wind ~30 minutes. Artificial sequencing introduces
error.

**Layer-1 pileup** is the specific diagnostic: emit a uniform
flux over a large area, run for 24 hours, measure the ratio of
mass in layer 1 to total column mass. Physical answer (good
mixing): ~5-10% of column mass in layer 1 for well-mixed
conditions. Pathological (no diffusion): 100% stuck in layer 1.

Plan 17's ordering study quantifies this for different palindrome
arrangements and recommends the operational default.

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- Advection machinery (plan 14)
- Chemistry operator (plan 15)
- Diffusion operator or Kz field types (plan 16b)
- `AbstractTimeVaryingField` or existing concrete types
  (plans 16a/16b)
- Palindrome structure beyond inserting S — V stays at current
  palindrome-center position
- LinRood, CubedSphereStrang — separate workspaces, not touched

## 3.2 Do NOT add

- Stack emissions (3D sources) — deferred; need separate scope
- Dry deposition as a separate operator — deferred; can be
  modeled as negative flux in `SurfaceFluxOperator` for now
- Unit regridding (NetCDF to model grid) — I/O layer concern,
  out of scope. Plan 17 consumes already-regridded 2D fluxes.
- Scaling / masking wrappers — deferred
- Time-varying diffusivity coupling — V still uses
  plan-16b Kz fields as-is

## 3.3 Potential confusion — clarified

**Surface flux is a 2D field (per tracer).** Mass per unit area
per unit time. Shape `(Nx, Ny)`. NOT 3D — it's applied only at
the bottom face. The kernel converts flux × dt × area into mass
change in layer 1 only.

**CATRINE inventories are window-averaged, not instantaneous.**
The stored monthly value is `average flux over the month`, with
units kg/m²/s. `StepwiseField` stores window averages; querying
at any time within a window returns the average. Sub-stepping
preserves the window integral exactly (integral-over-dt times
average = integral-over-window × fraction-of-window).

**Flux map is per-tracer.** Most tracers don't emit (air, most
advected species). Only the CATRINE tracers (CO2, fossil CO2, SF6,
Rn-222) have emissions. `PerTracerFluxMap` is a dictionary-like
structure; tracers absent from the map have zero flux.

**Dry deposition via negative flux.** For plan 17, dry deposition
is represented as a negative `SurfaceFluxOperator` flux (e.g., a
scaled version of the tracer concentration at layer 1 × deposition
velocity). A dedicated `DepositionOperator` is future work.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch
git branch -a | head -20
git log --oneline --all | grep -i "plan 16b\|vertical-diffusion" | head -10
git checkout <parent-branch>
git pull
git log --oneline | head -20
# Expected: plans 11-16b commits visible

git checkout -b surface-emissions

# 2. Clean working tree
git status

# 3. Verify dependency state
grep -c "AbstractDiffusionOperator\|ImplicitVerticalDiffusion" src/ \
    --include="*.jl" -r
# Expected: non-zero (plan 16b)
grep -c "apply_vertical_diffusion!" src/ --include="*.jl" -r
# Expected: non-zero (plan 16b Commit 4 array-level entry)
grep -c "current_time" src/MetDrivers/ --include="*.jl" -r
# Expected: non-zero (plan 16b Commit 5 stub)
grep -c "with_chemistry\|with_diffusion" src/Models/ --include="*.jl" -r
# Expected: non-zero (both plan 15/16b)

# 4. Survey existing emissions code
grep -rn -i "emission\|flux\|surface_flux\|deposition" src/ \
    --include="*.jl" | tee artifacts/plan17/existing_emissions_survey.txt
# Review. If substantial existing infrastructure, flag for
# revised scope. Plan 17 assumes minimal existing code.

# 5. Survey existing StepwiseField hints
grep -rn -i "stepwise\|window\|monthly\|piecewise" src/ \
    --include="*.jl" | tee artifacts/plan17/existing_stepwise_survey.txt
# Expected: minimal. StepwiseField is new to this plan.

# 6. Capture baseline
for testfile in test/test_basis_explicit_core.jl \
                test/test_advection_kernels.jl \
                test/test_structured_mesh_metadata.jl \
                test/test_reduced_gaussian_mesh.jl \
                test/test_driven_simulation.jl \
                test/test_cubed_sphere_advection.jl \
                test/test_poisson_balance.jl \
                test/test_chemistry.jl \
                test/test_fields.jl \
                test/test_diffusion_kernels.jl \
                test/test_diffusion_operator.jl \
                test/test_diffusion_palindrome.jl \
                test/test_transport_model_diffusion.jl; do
    [ -f "$testfile" ] || continue
    echo "=== $testfile ==="
    julia --project=. $testfile 2>&1 | tail -20
done | tee artifacts/plan17/baseline_test_summary.log

# 7. Record baseline
git rev-parse HEAD > artifacts/plan17/baseline_commit.txt
mkdir -p artifacts/plan17/perf

# 8. Memory compaction per plan 15 D3
# Update MEMORY.md with plan 16b shipped note
# Update plan 16b completion summary
```

If preconditions fail, STOP.

## 4.2 Change scope — expected file list

**Files to ADD (new):**

Field type:
- `src/State/Fields/StepwiseField.jl` — `StepwiseField{FT, N, A, T}`
  wrapping N+1-dimensional sample array + window boundaries

Surface flux operator:
- `src/Operators/SurfaceFlux/` — new directory
- `src/Operators/SurfaceFlux/SurfaceFlux.jl` — module file
- `src/Operators/SurfaceFlux/AbstractSurfaceFluxOperator.jl` — type
  hierarchy
- `src/Operators/SurfaceFlux/SurfaceFluxOperator.jl` — concrete
  operator + `apply!` + `apply_surface_flux!` (array-level)
- `src/Operators/SurfaceFlux/PerTracerFluxMap.jl` — dict-like
  mapping from tracer name to 2D flux field
- `src/Operators/SurfaceFlux/surface_flux_kernels.jl` — KA kernel
  that applies flux × dt × area at bottom layer

Tests:
- `test/test_stepwise_field.jl` — StepwiseField unit tests
- `test/test_surface_flux_kernels.jl` — kernel-level tests
- `test/test_surface_flux_operator.jl` — operator-level tests
- `test/test_per_tracer_flux_map.jl` — flux map tests
- `test/test_emissions_palindrome.jl` — palindrome integration
- `test/test_ordering_study.jl` — the quantitative study

Benchmarks:
- `scripts/benchmarks/bench_emissions_overhead.jl` — parallels
  chemistry/diffusion benches

Docs:
- `docs/plans/17_SURFACE_EMISSIONS_PLAN/NOTES.md`
- `docs/plans/17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md`
  — the study writeup

**Files to MODIFY:**

- `src/AtmosTransport.jl` — include SurfaceFlux module, export types
- `src/Operators/Operators.jl` — include order:
  `Diffusion → SurfaceFlux → Advection → Chemistry`
  (Advection's `strang_split_mt!` now imports SurfaceFlux too)
- `src/Operators/Advection/StrangSplitting.jl` — add
  `emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux()`
  kwarg; insert `V S V` pattern at palindrome center (replacing
  single V)
- `src/Models/TransportModel.jl` — add `emissions::EmT` field,
  `with_emissions` helper, thread through `step!`
- `src/Models/DrivenSimulation.jl` — REMOVE the plan-15 chemistry
  workaround (`with_chemistry(model, NoChemistry())`); emissions
  now live inside transport block
- `src/MetDrivers/AbstractMetDriver.jl` — extend `current_time`
  (plan 16b stub) to be a real accessor when drivers override;
  thread through `apply!` pathways
- `src/Operators/Chemistry/operators.jl` — `apply!` now uses
  `current_time(meteo)` instead of `zero(FT)` placeholder
- `src/Operators/Diffusion/operators.jl` — same as chemistry

## 4.3 Design decisions (pre-answered)

Every decision final. If ambiguous, STOP and ask.

**Decision 1: `StepwiseField{FT, N, A, T}` structure.**

```julia
struct StepwiseField{FT, N, A, T} <: AbstractTimeVaryingField{FT, N}
    samples::A                  # ::AbstractArray{FT, N+1}
                                # last dim iterates over N_windows
    boundaries::T               # ::AbstractVector{<:Real}
                                # length N_windows + 1, sorted
    current_window::Ref{Int}    # cache, updated by update_field!
end
```

- `samples[..., n]` is the value for window n: `[boundaries[n], boundaries[n+1])`
- `update_field!(f, t)` binary-searches `boundaries` for current
  window, caches index
- `field_value(f, idx)` reads `samples[idx..., current_window[]]`
- `integral_between(f, t1, t2)` sums window contributions with
  overlap weighting

Inner constructor validates:
- `ndims(samples) == N + 1`
- `length(boundaries) == size(samples, N+1) + 1`
- `issorted(boundaries)`

Outer convenience constructor:
- `StepwiseField(samples, boundaries)` — infers FT from samples,
  N from samples rank - 1

**Decision 2: `PerTracerFluxMap{FT}` structure.**

```julia
struct PerTracerFluxMap{FT, D}
    fluxes::Dict{Symbol, <:AbstractTimeVaryingField{FT, 2}}
end
```

- Tracers not in `fluxes` have zero flux
- Callers construct:
  ```julia
  emissions = PerTracerFluxMap{FT}(Dict(
      :CO2_fossil => StepwiseField(co2_monthly, month_boundaries),
      :SF6 => StepwiseField(sf6_yearly, year_boundaries),
      :Rn222 => StepwiseField(rn222_monthly, month_boundaries),
  ))
  ```

Debate: dict vs NTuple for performance. Plan 15 chose dict for
chemistry rates; plan 16a used NTuple. For emissions, dict is
natural (most tracers don't emit). Profile if needed.

**Decision 3: Surface flux operator hierarchy.**

```julia
abstract type AbstractSurfaceFluxOperator end

struct NoSurfaceFlux <: AbstractSurfaceFluxOperator end

struct SurfaceFluxOperator{FT, M} <: AbstractSurfaceFluxOperator
    flux_map::M    # ::PerTracerFluxMap{FT, 2}
end
```

No intermediate layer for combining with deposition — dep lives
as negative flux in the same map. Future `DepositionOperator` can
subtype this hierarchy if it grows.

**Decision 4: Kernel applies flux to bottom layer only.**

```julia
@kernel function _surface_flux_kernel!(
    tracers_raw,    # 4D: (Nx, Ny, Nz, Nt)
    flux_values,    # 3D: (Nx, Ny, Nt) — per-tracer 2D flux
    cell_area,      # 2D: (Nx, Ny)
    dt,
    tracer_indices  # ::NTuple of tracer indices to update
)
    i, j, t_idx = @index(Global, NTuple)
    @inbounds begin
        t = tracer_indices[t_idx]
        # Flux is positive = emission (mass added). Negative = deposition.
        tracers_raw[i, j, 1, t] += flux_values[i, j, t_idx] *
                                   cell_area[i, j] * dt
    end
end
```

Only layer 1 is touched. Other layers untouched regardless of
tracer index.

**Decision 5: `apply!` resolves flux fields and calls kernel.**

```julia
function apply!(state::CellState{FT},
                meteo,
                grid::AbstractGrid,
                op::SurfaceFluxOperator,
                dt::Real;
                workspace) where FT
    t_now = current_time(meteo)

    # Materialize per-tracer flux arrays from the map
    # For tracers IN the map: flux_values[i,j,t_idx] = field_value(map[name], (i,j))
    # For tracers NOT in the map: skip (not in tracer_indices)

    # Resolve which tracers have fluxes and their indices
    tracer_indices, flux_values = _resolve_flux_map(
        op.flux_map, state.tracer_names, t_now
    )

    # Launch kernel only for emitting tracers
    backend = get_backend(state.tracers_raw)
    Nx, Ny, Nz, Nt = size(state.tracers_raw)
    kernel = _surface_flux_kernel!(backend, (16, 16, 1))
    kernel(state.tracers_raw, flux_values,
           cell_area(grid), FT(dt), tracer_indices;
           ndrange = (Nx, Ny, length(tracer_indices)))
    synchronize(backend)

    return state
end
```

**Decision 6: Array-level entry point mirrors diffusion.**

```julia
apply_surface_flux!(q_raw, op::NoSurfaceFlux, ws, dt, meteo, grid) = nothing
apply_surface_flux!(q_raw, op::SurfaceFluxOperator, ws, dt, meteo, grid)
    # ... resolve fluxes, launch kernel
end
```

Called from inside the palindrome like
`apply_vertical_diffusion!` is in plan 16b. Dispatched on operator
type for `NoSurfaceFlux` dead branch.

**Decision 7: Palindrome position — `X Y Z V S V Z Y X`.**

Plan 16b shipped `X Y Z V Z Y X`. Plan 17 replaces the single
`V(dt)` call with `V(dt/2) S(dt) V(dt/2)`:

```
X(dt/2) Y(dt/2) Z(dt/2)
V(dt/2)
S(dt)
V(dt/2)
Z(dt/2) Y(dt/2) X(dt/2)
```

This is OPERATOR_COMPOSITION.md §3.2's recommendation: emissions
between two V half-steps gives boundary-layer mixing of fresh
emissions before the next horizontal transport step.

**Implication for plan 16b's V(dt) decision.** Plan 16b shipped
single V(dt) (correct for linear V when Kz is time-constant
within the step). Plan 17 now splits that into two V(dt/2) calls.
Per plan 16b's Decision 8 refinement, this is a "leading order"
equivalent for Backward Euler — the split version has slightly
different truncation error but converges at the same rate. The
physics win (fresh emissions get mixed) dominates the minor
accuracy difference.

**Decision 8: Ordering study compares four palindrome arrangements.**

| Label | Palindrome | Rationale |
|---|---|---|
| A (recommended) | `X Y Z V(dt/2) S V(dt/2) Z Y X` | Composition doc §3.2 |
| B (single-V post-S) | `X Y Z S V Z Y X` | Emissions before mixing |
| C (single-V pre-S) | `X Y Z V S Z Y X` | Emissions after mixing |
| D (no V) | `X Y Z S Z Y X` | Pathological baseline — layer-1 pileup |

Configuration: run each arrangement for 24 hours with identical
initial state (uniform tracer), uniform surface emissions over a
large region, no chemistry, no other forcings. Measure layer-1
mass ratio vs column mass after 24 hours, plot as a function of
PBL height. Write up in
`docs/plans/17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md`.

**Decision 9: Thread `current_time(meteo)` through `apply!`.**

Plan 16b Commit 5 shipped the stub. Plan 17 makes real drivers
override it. Operators use `current_time(meteo)` instead of
`zero(FT)` placeholder:

```julia
# BEFORE (plan 15, 16b):
t = zero(FT)

# AFTER (plan 17):
t = FT(current_time(meteo))
```

Applies to chemistry and diffusion `apply!` methods too.
Meteorology-dependent fields (`DerivedKzField` especially) now
receive real time.

If meteo is `nothing` (some tests pass nothing), default to
`zero(FT)`:

```julia
t = meteo === nothing ? zero(FT) : FT(current_time(meteo))
```

**Decision 10: Remove plan 15 chemistry workaround.**

Before plan 17:
```julia
# DrivenSimulation constructor strips chemistry from model:
model = with_chemistry(incoming_model, NoChemistry())
# Sim applies chemistry separately:
function step!(sim::DrivenSimulation, dt)
    step!(sim.model, dt)  # transport only
    apply_chemistry!(sim.chemistry, sim.model.state, dt)
end
```

After plan 17:
```julia
# DrivenSimulation just wraps the model:
# (no with_chemistry call)
# Sim delegates entirely:
function step!(sim::DrivenSimulation, dt)
    step!(sim.model, dt)  # transport (adv+diff+emissions) + chemistry
end
```

This works because the palindrome now has emissions at center,
so the TM5 order `advection → emissions → chemistry` is:
```
transport_block: [X Y Z V(dt/2) S V(dt/2) Z Y X]  (contains emissions at center)
chemistry_block: [chemistry]                      (runs after transport)
```

Matches TM5 exactly without workarounds.

**Decision 11: Tests use accessor API (plan 14 contract).**

**Decision 12: Regression test for plan 16b's palindrome must pass.**

Plan 16b shipped `X Y Z V Z Y X` with specific bit-exact
regressions. Plan 17's change to `X Y Z V S V Z Y X` with
`NoSurfaceFlux` default must still produce BIT-EXACT equivalence
to plan 16b's behavior:

- `S = NoSurfaceFlux()` → dead branch, zero fp work
- `V(dt/2) S V(dt/2)` with `NoSurfaceFlux` = `V(dt/2) V(dt/2)`
  which is NOT bit-exact equal to `V(dt)` (per plan 16b Decision
  8 refinement — they agree to leading order but not exactly)

**This is a design tension.** Options:

- **Option A:** Keep single `V(dt)` when `NoSurfaceFlux`, switch
  to `V(dt/2) S V(dt/2)` only when a real flux operator is
  present. Preserves 16b bit-exactness.
- **Option B:** Always use `V(dt/2) V(dt/2)` pattern. Breaks 16b
  bit-exactness but is uniform.

**Recommendation: Option A.** Preserves pre-plan-17 behavior
exactly for users who don't opt in. The split into two V
half-steps is only paid when emissions are active.

Implementation:

```julia
if emissions_op isa NoSurfaceFlux
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt)  # single V
else
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2)  # first V half
    apply_surface_flux!(rm_cur, emissions_op, ws, dt, meteo, grid)
    apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2)  # second V half
end
```

This dispatch cost is ~1 branch per step — negligible.

## 4.4 Atomic commit sequence

### Commit 0: NOTES + baseline

Standard pattern. Survey for existing emissions/surface-flux code.
Memory compaction per plan 15 D3.

```bash
mkdir -p docs/plans/17_SURFACE_EMISSIONS_PLAN
# ... (standard NOTES.md scaffold)
git commit -m "Commit 0: NOTES + baseline + survey for plan 17"
```

### Commit 1: `StepwiseField{FT, N, A, T}`

- Create `src/State/Fields/StepwiseField.jl`
- Inner constructor validates rank and boundary monotonicity
- `update_field!` does binary search to cache current window index
- `field_value(f, idx)` reads from cached window
- `integral_between(f, t1, t2)` sums window contributions with
  overlap weighting
- Adapt support: `Adapt.adapt_structure(to, f)` converts backing
  array
- Outer convenience constructor inferring FT, N

Wire through `Fields → State → AtmosTransport`. Tests in
`test/test_stepwise_field.jl`:

1. Construction: rank validation, boundary sorting
2. `field_value` returns correct window value
3. `update_field!` advances window correctly across boundaries
4. `integral_between` exact within a window
5. `integral_between` summing across multiple windows
6. Sub-stepping additivity (plan TIME_VARYING_FIELD_MODEL.md §8.3)
7. Type stability
8. Adapt/GPU path

```bash
git commit -m "Commit 1: StepwiseField — piecewise-constant window averages"
```

### Commit 2: `PerTracerFluxMap{FT}`

- Create `src/Operators/SurfaceFlux/PerTracerFluxMap.jl`
- Dict-backed; lookup by tracer name
- `flux_for(map, tracer_name, t)` returns flux field or nothing
- `update_fields!(map, t)` calls `update_field!` on all entries

Tests in `test/test_per_tracer_flux_map.jl`: construction,
lookup, iteration, update propagation.

```bash
git commit -m "Commit 2: PerTracerFluxMap — per-tracer 2D flux fields"
```

### Commit 3: `SurfaceFluxOperator` + kernel + `apply!`

- Create `src/Operators/SurfaceFlux/` module structure
- `AbstractSurfaceFluxOperator`, `NoSurfaceFlux`, `SurfaceFluxOperator`
- `_surface_flux_kernel!` applies flux × dt × area to layer 1
- State-level `apply!(state, meteo, grid, op, dt; workspace)`
- Array-level `apply_surface_flux!(q_raw, op, ws, dt, meteo, grid)`
- `NoSurfaceFlux` dispatch as dead branch (both entries)

Tests in `test/test_surface_flux_kernels.jl` and
`test/test_surface_flux_operator.jl`:

1. Kernel applies flux to layer 1 only (other layers untouched)
2. Multi-tracer kernel handles per-tracer flux maps correctly
3. `NoSurfaceFlux` is identity
4. Mass balance: added mass matches `∫ flux × area × dt`
5. CPU/GPU consistency
6. Tests use accessor API

```bash
git commit -m "Commit 3: SurfaceFluxOperator + kernel + apply!"
```

### Commit 4: `current_time(meteo)` threading

- Extend `current_time` in `AbstractMetDriver` to read real time
  from concrete drivers (not just the stub returning 0.0)
- Update chemistry `apply!` to use `current_time(meteo)` instead
  of `zero(FT)`
- Update diffusion `apply!` similarly
- Add `meteo === nothing ? zero(FT) : FT(current_time(meteo))`
  dispatch for test-helper compatibility

Tests: ensure existing tests that pass `meteo = nothing` still
work; add new tests with real drivers that verify time is
threaded correctly.

```bash
git commit -m "Commit 4: Thread current_time through operator apply! methods"
```

### Commit 5: Palindrome integration (Decision 12 Option A)

- Modify `strang_split_mt!` to accept
  `emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux()`
- Insert branch logic per Decision 12:
  ```julia
  if emissions_op isa NoSurfaceFlux
      apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt)
  else
      apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2)
      apply_surface_flux!(rm_cur, emissions_op, ws, dt, meteo, grid)
      apply_vertical_diffusion!(rm_cur, diffusion_op, ws, dt/2)
  end
  ```
- Thread through `strang_split!` and structured-mesh `apply!` wrapper

Tests in `test/test_emissions_palindrome.jl`:

1. **Bit-exact regression: `NoSurfaceFlux` default == plan 16b's
   palindrome**. Critical. `X Y Z V Z Y X` behavior preserved
   byte-for-byte when no emissions.
2. With `SurfaceFluxOperator`, a zero-flux map produces no tracer
   change (bit-exact to NoSurfaceFlux path)
3. With `SurfaceFluxOperator`, a non-zero uniform flux increases
   layer-1 mass correctly
4. Palindrome symmetry: forward-reverse pattern is preserved
5. Strang accuracy: halving dt reduces error by ~4× (second-order)

```bash
git commit -m "Commit 5: Palindrome integration — V(dt/2) S V(dt/2) when emissions active"
```

### Commit 6: TransportModel + DrivenSimulation wiring

- Add `emissions::EmT` field to `TransportModel`
- `with_emissions(model, ops)` helper
- Thread `emissions_op = model.emissions` through advection call
- **Remove plan 15 chemistry workaround from `DrivenSimulation`**:
  no more `with_chemistry(model, NoChemistry())` on construction

Tests in `test/test_transport_model_emissions.jl`: default
`NoSurfaceFlux`; `with_emissions` returns new model; `step!` runs
transport+emissions+chemistry in correct order;
`DrivenSimulation.step!` without workaround produces equivalent
results to the old workaround path.

```bash
git commit -m "Commit 6: TransportModel emissions field + DrivenSimulation cleanup"
```

### Commit 7: Ordering study

- Create `test/test_ordering_study.jl` with four palindrome configs
- Run each for 24 hours at medium resolution
- Measure layer-1 mass ratio vs column mass
- Write up in
  `docs/plans/17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md`
  with plots and recommendation

Configurations to compare:

```julia
# A: X Y Z V(dt/2) S V(dt/2) Z Y X (recommended)
# B: X Y Z S V(dt) Z Y X
# C: X Y Z V(dt) S Z Y X
# D: X Y Z S Z Y X (no V, pathological)
```

Test predicate:
- Config A: layer-1 mass ratio should match physical expectation
  (~5-15% of column mass for typical PBL heights)
- Config D: layer-1 mass ratio should be >50% (pileup)
- Configs B, C: intermediate

```bash
git commit -m "Commit 7: Ordering study — V-S palindrome comparison"
```

### Commit 8: Benchmarks

- `scripts/benchmarks/bench_emissions_overhead.jl`
- Parallels chemistry/diffusion benches
- Compare: `NoSurfaceFlux` (baseline) vs real operator
- Configurations: CPU medium, GPU medium, GPU large

Expected: emissions overhead <5% on GPU (single kernel launch
per step, small cells touched). CPU similar.

```bash
git commit -m "Commit 8: Emissions overhead benchmarks"
```

### Commit 9: Documentation + retrospective

Update `ARCHITECTURAL_SKETCH.md` to v3: emissions in palindrome,
chemistry workaround resolved, time threading.

Fill in NOTES.md retrospective sections:
- Decisions beyond the plan
- Surprises
- Interface validation findings
- Template usefulness for plan 18

```bash
git commit -m "Commit 9: Retrospective + documentation polish"
```

## 4.5 Test plan per commit

After EACH commit:
```bash
julia --project=. -e 'using AtmosTransport'
julia --project=. test/runtests.jl
```

Baseline 77-failure count must be preserved. Plan 16b's test suite
(including `test_diffusion_palindrome.jl`) must continue to pass
unchanged — Decision 12 Option A ensures this.

## 4.6 Acceptance criteria

**Correctness (hard):**
- All pre-existing tests pass (77 baseline failures unchanged)
- Plan 16b's `test_diffusion_palindrome.jl` passes bit-exact
  (`NoSurfaceFlux` default preserves pre-17 behavior)
- Mass added via surface flux matches analytic integration
- Palindrome preserves symmetry for reversible configurations

**Code cleanliness (hard):**
- `src/Operators/SurfaceFlux/` directory exists per §4.2
- `StepwiseField` in `src/State/Fields/`
- Palindrome `V(dt/2) S V(dt/2)` when emissions active
- `DrivenSimulation` no longer strips chemistry from wrapped model
- `current_time(meteo)` threaded through all operator `apply!`s

**Performance (soft):**
- Emissions overhead <10% on GPU at production settings
- No regression in non-emissions paths (bit-exact regression
  check passes)

**Interface validation (hard):**
- `apply!` signature unchanged (plan 15 + 16b validated; plan 17
  confirms for a third operator family)
- `StepwiseField` works through the palindrome at kernel scale
- `current_time(meteo)` threading is uniform across chemistry /
  diffusion / surface flux

**Scientific (hard for ordering study):**
- Configuration A (recommended) gives physically reasonable
  layer-1 ratios under varying PBL heights
- Configuration D shows clear pileup — validates the study's
  diagnostic

**Documentation:**
- `ARCHITECTURAL_SKETCH.md` v3 committed
- `ordering_study_results.md` with plots and recommendation
- NOTES.md complete

## 4.7 Rollback plan

Standard. Specific rollback points:

- **Commit 4 (current_time threading) breaks chemistry tests.**
  Plan 15's tests pass `meteo = nothing`. Ensure
  `meteo === nothing` fallback to `zero(FT)`. Revert if deeper.
- **Commit 5 (palindrome) breaks plan 16b's regression.**
  Decision 12 Option A should prevent this. If it fails,
  investigate whether `NoSurfaceFlux` dispatch is actually a
  dead branch (should be compiled away). Revert and debug.
- **Commit 6 (DrivenSimulation cleanup) breaks sim-level tests.**
  The plan-15 chemistry workaround removal must be validated end-
  to-end. If new failures, check whether chemistry is running
  correctly in the new transport-block order.

## 4.8 Known pitfalls

1. **"Let me make the palindrome always `V(dt/2) S V(dt/2)`."**
   NO per Decision 12. Breaks plan 16b's bit-exact regression.
   Use Option A dispatch branch.

2. **"Dry deposition deserves its own operator."** Maybe, but not
   in plan 17. Ship as negative flux in `SurfaceFluxOperator` for
   now. Extract to `DepositionOperator` when a real use case
   demands separation.

3. **"PerTracerFluxMap should be an NTuple for type stability."**
   Maybe. Plan 15 used dict for chemistry, plan 16a used NTuple
   for chemistry rates (via `ExponentialDecay.decay_rates` which
   is `NTuple{N, ConstantField{FT, 0}}`). Dict is natural for
   emissions (most tracers don't emit). Profile if needed and
   switch to NTuple if measurably faster.

4. **"I can compute integrals in the kernel."** NO. Per
   `TIME_VARYING_FIELD_MODEL.md`, `integral_between` is a
   CPU-only interface. Kernels consume pre-computed window values
   via `field_value`.

5. **"The ordering study is just a test; skip the writeup."** NO
   per Commit 7 acceptance criteria. The writeup is the deliverable
   — future plans (plan 18 convection) will want this as prior
   art.

6. **"Stack emissions (3D sources) are easy to add."** They might
   be. Defer anyway. Scope discipline.

7. **"I can skip the bit-exact regression — this is a behavior
   change."** NO. `NoSurfaceFlux` default MUST preserve pre-17
   behavior byte-for-byte. Plan 16b did this for `NoDiffusion`;
   plan 17 must do the same for `NoSurfaceFlux`.

8. **"Let me also port non-local PBL diffusion since I'm in the
   neighborhood."** NO. Stay in scope.

9. **"current_time threading should pass an explicit `t` kwarg
   to apply!, not read from meteo."** Maybe eventually, but not
   this plan. Read from meteo via accessor; uniform across all
   operators.

10. **"The `V(dt/2) S V(dt/2)` split changes perf — halves V's
    work."** Check. V(dt/2) costs roughly the same as V(dt) per
    call (Thomas solve is dominated by column depth, not
    coefficient values); running two half-step V solves is
    approximately 2× the diffusion work of a single V(dt) solve.
    This is the cost of getting the physics right. Expected
    overhead: ~10-20% on top of plan 16b baseline when emissions
    are active.

11. **"Ordering study should use DerivedKzField for realistic
    physics."** Tempting but requires end-to-end meteorology
    integration. For plan 17, use `ProfileKzField` with a
    canonical exponential profile — simpler, reproducible, tests
    the ordering question cleanly without meteorology confounding.
    DerivedKzField integration can be a follow-up.

---

# Part 5 — How to Work

## 5.1 Session cadence

- Session 1: Commit 0 + Commit 1 (StepwiseField)
- Session 2: Commits 2-3 (FluxMap + Operator)
- Session 3: Commit 4 (current_time threading)
- Session 4: Commit 5 (palindrome integration) — the risky one
- Session 5: Commit 6 (TransportModel + DrivenSimulation cleanup)
- Session 6: Commit 7 (ordering study)
- Session 7: Commits 8-9 (benchmarks + docs)

## 5.2 When to stop and ask

- Commit 5 bit-exact regression fails (plan 16b tests break)
- Commit 6 DrivenSimulation cleanup breaks sim tests (chemistry
  ordering broken)
- Commit 7 ordering study shows unexpected results (e.g.,
  Configuration A doesn't beat D significantly — might indicate
  diffusion is too weak in the test setup)
- Scope creep toward stack emissions or deposition operator
- Plan 16b palindrome regression fails at any commit

## 5.3 NOTES.md discipline

Specific items to capture:

- Did `current_time` threading reveal any hidden dependencies
  on `zero(FT)` time? (Likely no, but worth checking.)
- Did the chemistry workaround removal break any tests that
  depended on the specific order? (Unlikely; TM5 order is
  preserved by moving emissions into transport block.)
- Did `StepwiseField` need any API changes from the
  `TIME_VARYING_FIELD_MODEL.md` spec?
- What was the actual ordering-study result? Surprises?

---

# End of Plan

After this refactor ships:
- `SurfaceFluxOperator` in `src/Operators/SurfaceFlux/`
- `StepwiseField` in `src/State/Fields/`
- Palindrome `X Y Z V(dt/2) S V(dt/2) Z Y X` when emissions active
- `current_time(meteo)` threaded through operators
- Plan 15 chemistry workaround removed
- CATRINE runs possible end-to-end
- Ordering study documents the physics tradeoffs

The next plan:
- Plan 18: Convection C. Adds C to palindrome, positions likely
  `X Y Z V C S C V Z Y X` or similar (to be determined).
  References plan 17's ordering study for precedent.
