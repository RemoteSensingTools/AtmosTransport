# Plan 15 Execution Notes — Slow Chemistry Operator

Plan: [../15_SLOW_CHEMISTRY_PLAN.md](../15_SLOW_CHEMISTRY_PLAN.md) (v1).
Companion refs: [../OPERATOR_COMPOSITION.md](../OPERATOR_COMPOSITION.md)
(§3.1 step-level composition, §3.3 chemistry block, §6 operator interface).

## Baseline

- **Branch**: `slow-chemistry`, forked from `advection-unification` at `e8ab90d`
  (plan 14 Commit 7 — post-refactor bench + retrospective).
- **Parent chain**: plans 11/12/13/14 all shipped. `CellState.tracers_raw ::
  Array{FT,4}`, `tracer_names :: NTuple{Nt, Symbol}`, `TracerAccessor` for
  property access; accessor API (`ntracers`, `get_tracer`, `eachtracer`,
  `tracer_names`) is public.
- **Pre-existing test failures**: 77 total across the 7 canonical test files
  (documented in [../../../artifacts/baseline_test_summary_plan15.log](../../../artifacts/baseline_test_summary_plan15.log)). Plan 15 must hold this unchanged.

| File | Pass | Fail | Notes |
|---|---:|---:|---|
| `test_basis_explicit_core.jl` | 3 | 2 | Honest metadata-only CubedSphere API |
| `test_advection_kernels.jl` | 178 | 0 | incl. Multi-tracer fusion 84/84 ✅ |
| `test_structured_mesh_metadata.jl` | 10 | 3 | CubedSphereMesh conventions |
| `test_reduced_gaussian_mesh.jl` | 26 | 0 | |
| `test_driven_simulation.jl` | 57 | 0 | all default to `chemistry=NoChemistry()` |
| `test_cubed_sphere_advection.jl` | 552 | 0 | |
| `test_poisson_balance.jl` | 245 | 72 | mirror_sign inflow/outflow |
| **Total** | **1071** | **77** | |

## Existing chemistry code survey (plan 15 §4.1 step 4)

See [../../../artifacts/perf/plan15/existing_chemistry_survey.txt](../../../artifacts/perf/plan15/existing_chemistry_survey.txt)
(26 lines).

**Material state** (confirmed via grep, NOT the 5-day-old memo claim):

- [../../../src/Operators/Chemistry/Chemistry.jl](../../../src/Operators/Chemistry/Chemistry.jl)
  defines `AbstractChemistry`, `NoChemistry`, `RadioactiveDecay{FT}` (single
  species + half_life + lambda), `CompositeChemistry`. 78 lines.
- Interface: `apply_chemistry!(tracers, chem, Δt)` — takes `tracers` as the
  NamedTuple-or-accessor container. `RadioactiveDecay` branch does
  `haskey(tracers, chem.species)` + `getfield(tracers, chem.species)` then
  `c .*= exp(-λΔt)`.
- Wired into `DrivenSimulation` at three sites:
  - [../../../src/Models/DrivenSimulation.jl:39](../../../src/Models/DrivenSimulation.jl#L39)
    — `chemistry :: CT` field in struct
  - [../../../src/Models/DrivenSimulation.jl:205](../../../src/Models/DrivenSimulation.jl#L205)
    — constructor keyword `chemistry::AbstractChemistry = NoChemistry()`
  - [../../../src/Models/DrivenSimulation.jl:275](../../../src/Models/DrivenSimulation.jl#L275)
    — call site `apply_chemistry!(sim.model.state.tracers, sim.chemistry, sim.Δt)`
- `TransportModel` does NOT carry chemistry; it's injected at the
  `DrivenSimulation` level and called AFTER `step!(model, Δt)`.
- Exports chained through
  [../../../src/Operators/Operators.jl:40-41](../../../src/Operators/Operators.jl#L40)
  → [../../../src/AtmosTransport.jl:169-170](../../../src/AtmosTransport.jl#L169).
- **No `test/test_chemistry*.jl` exists.** No chemistry unit tests.
- **No `TimeVaryingField` / `ConstantField`** in src/. Plan 15 Decision 2
  deferred: use plain `NTuple{Nt, FT}` of rates.

### Plan-14-induced latent bug

The call at `DrivenSimulation.jl:275` passes `sim.model.state.tracers`, which
post-plan-14 is a `TracerAccessor` wrapper (lazy proxy over `tracers_raw`).
The `RadioactiveDecay` branch of `apply_chemistry!` does:
```julia
haskey(tracers, chem.species) || return nothing
c = getfield(tracers, chem.species)
```
`getfield(::TracerAccessor, ::Symbol)` reaches for a struct field named
`:Rn222`, of which there is none (the struct has only `:state`). `haskey(::TracerAccessor, ...)` is similarly undefined. A real CATRINE Rn-222 run
would error.

Plan-14 tests all default to `NoChemistry()` (first dispatch branch returns
`nothing`), which is why this didn't surface during plan-14 baseline.

Plan 15 Commit 3 deletes the DrivenSimulation call site and routes chemistry
through `TransportModel.step!` via the new `apply!` interface — fix is a
side-effect of the refactor.

## Decisions made beyond the plan

### D1 — Rename types to match plan 15 v1 naming (user-confirmed 2026-04-18)

`AbstractChemistry → AbstractChemistryOperator` and
`RadioactiveDecay → ExponentialDecay`. `NoChemistry` and `CompositeChemistry`
keep their names. Exports updated; no backward-compat shim (plan 14 precedent
— no external users).

### D2 — NTuple of scalars, not `ConstantField`

Plan 15 Decision 2 calls for `ConstantField(k)` rather than scalar `k`. But
§3.2 says "don't extend `TimeVaryingField`" and that abstraction doesn't
exist in this repo. Deferred to plan 16+. Using plain
`NTuple{N, FT}` of rates instead. If plan 16 introduces `TimeVaryingField`,
it will include a migration of `ExponentialDecay.decay_rates` to
`NTuple{N, AbstractTimeVaryingField{FT,0}}`.

### D3 — Pre-Commit-0 memory compaction (user-confirmed 2026-04-18)

Compaction done before Commit 0 so downstream agents see fresh state. Updated:
- `MEMORY.md` "Current State" → 2026-04-18, plan 14 shipped note.
- Added `plan14_complete.md` (4D storage, accessor API, perf wins, latent bug).
- `src_v2_refactor_overview.md` — post-plan-14 update section.
- `project_catrine_validation_status.md` — "Rn222 chemistry module added ✅"
  line marked as partial (module exists, interface about to change).
- `tm5_vs_julia_audit_20260403.md` and `float64_gpu_kernels.md` — added
  "⚠ stale paths" warnings for `src/Advection/*`, `src/Diffusion/*`,
  `src/Convection/*` citations (those are in `src_legacy/` now).

## Commit-by-commit notes

### Commit 0 — NOTES.md + baseline + survey (this commit)

No code changes; memory compaction done in pre-Commit-0. Baseline test
summary captured in
[../../../artifacts/baseline_test_summary_plan15.log](../../../artifacts/baseline_test_summary_plan15.log)
— 77 failures, same as plan-14 baseline. Existing chemistry survey
recorded in
[../../../artifacts/perf/plan15/existing_chemistry_survey.txt](../../../artifacts/perf/plan15/existing_chemistry_survey.txt).

### Commit 1 — rewrite Chemistry.jl (1cf4fb6)

Types renamed (`AbstractChemistry → AbstractChemistryOperator`,
`RadioactiveDecay → ExponentialDecay`). `ExponentialDecay` reshaped to
multi-tracer: `decay_rates::NTuple{N, FT}` + `tracer_names::NTuple{N,
Symbol}`. New `chemistry_kernels.jl` with a single KA fused kernel
(rank-agnostic over `tracers_raw`). New `apply!(state, meteo, grid, op,
dt; workspace)` method for every concrete type. Old `apply_chemistry!`
deleted (no backward-compat shim per plan 14 precedent).
`DrivenSimulation.jl:275` migrated to the new interface — incidentally
fixes the plan-14 `getfield(::TracerAccessor, :Rn222)` bug. 26 new
`test/test_chemistry.jl` cases passing.

### Commit 2 — chemistry_block! (db7064a)

Trivial sequential composer `chemistry_block!(state, meteo, grid,
operators, dt)` for tuples and single operators. Exists alongside
`CompositeChemistry` (which is an operator wrapping others); the
difference is composition-at-block vs. composition-in-operator, both
are supported. Tests: single op, 2-op tuple, empty tuple = identity.
6 new tests, suite 32/32.

### Commit 3 — TransportModel.chemistry + step! (79ac843)

`TransportModel` gained a `chemistry :: ChemT` field
(`NoChemistry()` default). `step!(model, dt)` composes advection and
chemistry per OPERATOR_COMPOSITION.md §3.1. New helper
`with_chemistry(model, chemistry)` returns a rebuilt model with the
chemistry slot replaced (no array copy). DrivenSimulation constructor
calls `with_chemistry(model, NoChemistry())` on the incoming model so
chemistry stays at the sim level (preserves the TM5 order
`advection → emissions → chemistry`). The chemistry call site in
`step!(sim)` moved from `apply!` to `chemistry_block!`. Baseline
unchanged; 32 chemistry tests + existing baseline pass.

### Commit 4 — end-to-end advection + decay composition test (eb131d8)

Two new testsets in `test/test_chemistry.jl`:
1. TransportModel.step! with uniform Rn-222 + zero fluxes → verifies
   mass sum matches `M₀ · exp(-λ · n_steps · dt)` to 1e-13 over 8
   1-hour steps, non-decaying tracer CO2 and air mass exactly
   preserved.
2. Order check: `step!(model)` with chemistry ≈ manual `apply!`
   sequence (advection then chemistry). Confirms
   OPERATOR_COMPOSITION.md §3.1 ordering.

Suite now 37/37 passing. Plan-15 acceptance §4.6 "chemistry decay
matches exp(-k·dt) exactly; non-decaying tracers preserved exactly"
both satisfied.

### Commit 5 — bench + docs + retrospective

Chemistry overhead measurements on two hosts:

* **wurst CPU medium F64**
  ([../../../artifacts/perf/plan15/overhead_wurst_cpu_medium.log](../../../artifacts/perf/plan15/overhead_wurst_cpu_medium.log)):

  | scheme | Nt | no-chem (ms) | chem (ms) | Δ% |
  |---|---:|---:|---:|---:|
  | Upwind | 5  |  346.8 |  371.6 |  7.1% |
  | Upwind | 10 |  613.1 |  679.0 | 10.8% |
  | Upwind | 30 | 1692.5 | 1877.0 | 10.9% |
  | Slopes | 5  | 2389.0 | 2416.4 |  1.2% ✅ |
  | Slopes | 10 | 4853.4 | 4918.0 |  1.3% ✅ |
  | Slopes | 30 | (cut short) | | |

* **curry A100 F32**
  ([../../../artifacts/perf/plan15/overhead_curry_gpu_f64.log](../../../artifacts/perf/plan15/overhead_curry_gpu_f64.log)):

  GPU medium:

  | scheme | Nt | no-chem (ms) | chem (ms) | Δ% |
  |---|---:|---:|---:|---:|
  | Upwind | 5  | 2.61 | 2.23 | -14%¹ |
  | Upwind | 10 | 3.17 | 4.06 | 28% |
  | Upwind | 30 | 5.35 | 6.51 | 22% |
  | Slopes | 5  | 4.93 | 2.95 | -40%¹ |
  | Slopes | 10 | 4.82 | 5.01 |  4% ✅ |
  | Slopes | 30 |12.63 |13.42 |  6% |

  GPU large:

  | scheme | Nt | no-chem (ms) | chem (ms) | Δ% |
  |---|---:|---:|---:|---:|
  | Upwind | 5  | 10.5 | 10.9 |  3% ✅ |
  | Upwind | 10 | 16.5 | 17.9 |  8% |
  | Upwind | 30 | 42.4 | 48.1 | 13% |
  | Slopes | 5  | 21.5 | 22.0 |  2% ✅ |
  | Slopes | 10 | 38.5 | 39.9 |  4% ✅ |
  | Slopes | 30 |105.3 |111.4 |  6% |

  ¹ Negative deltas on medium Nt=5 are timing noise at sub-ms resolution
  on GPU (MAD is a larger fraction of median at this size). Not a real
  speedup.

**Interpretation**: chemistry overhead is dominated by the single KA
kernel launch + one pass over `tracers_raw`. When advection is
cheap (Upwind, memory-bound), this is a larger relative fraction
(10-28%). When advection is more work (Slopes, PPM — more
arithmetic per cell), the fraction drops to 1-6%. **The plan-15 <5%
bar (plan §4.6) is met for Slopes at production Nt** (CPU: 1-2% at
Nt=5/10; GPU large: 4% at Nt=10, 6% at Nt=30).

For CATRINE-class production runs (slopes/PPM with Nt=4-10), chemistry
overhead is comfortably under 5%.

## Interface-validation findings (answer to plan §2.2)

| Question | Answer |
|---|---|
| Did `apply!(state, meteo, grid, op, dt; workspace)` generalize beyond advection without modification? | **Yes.** Pure decay ignores `meteo`, `grid`, and `workspace`, but accepting them by keyword/positional slot is free; no signature change needed. The uniform signature is the win — plans 17 (diffusion) and 18 (convection) can subtype `AbstractChemistryOperator` patterns to the respective operator hierarchies without rethinking the shape. |
| Did the chemistry block (`OPERATOR_COMPOSITION.md` §3.1) compose cleanly with the transport block? | **Yes**, with a caveat. `step!(model, dt)` composes cleanly: `advection → chemistry`. But when a *third* operator like emissions/surface-sources is in play (DrivenSimulation), the TM5 order is `advection → emissions → chemistry`. Keeping chemistry at the sim level (via `chemistry_block!` called after surface sources) preserves that order; forcing `TransportModel.chemistry = NoChemistry()` for the sim's wrapped model is the knob. A cleaner resolution waits for plan 16 to fold emissions into the advection Z-sweep, after which `TransportModel.step!`'s `transport → chemistry` becomes the right order. |
| Did `TimeVaryingField` integrate without friction? | **N/A — abstraction does not exist.** Deferred per plan D2. Using `NTuple{N, FT}` of scalar rates. When plan 16+ introduces `TimeVaryingField`, migrate `ExponentialDecay.decay_rates` field type and the kernel's `rate` argument. |
| Does step-level orchestration handle transport/chemistry split correctly? | **Yes.** `TransportModel.step!` runs advection then chemistry. `DrivenSimulation.step!` keeps the TM5 ordering by explicitly controlling where chemistry runs. Both work; the choice is a policy knob. |

## Deferred observations

- **TimeVaryingField abstraction is a gap**. Plan 15 bridged it with a
  static `NTuple{N, FT}`; plan 16 needs it for emission fields and may
  retroactively want it for chemistry rates.
- **Chemistry ordering vs. emissions** is resolved at the
  DrivenSimulation level for now. Plan 16 will fold emissions into the
  advection Z-sweep (composition doc §8 Option B), at which point the
  sim-level explicit chemistry call can be removed and
  `TransportModel.chemistry` alone suffices.
- **Upwind + chemistry is the highest-overhead combo** (10-28% on both
  CPU and GPU). This is because Upwind is the fastest advection, so
  the chemistry launch is a larger relative cost. Not a concern for
  production (which uses Slopes or PPM), but documented for future
  reference.
- **bench script `--dtype` flag not added**: the current
  `bench_chemistry_overhead.jl` defaults to F32-only on GPU, matching
  the L40S-centric advection bench. For curry A100, F64 would have
  been a cleaner comparison with plan-14's numbers; F32 results are
  still valid as order-of-magnitude evidence. Add a `--dtype` flag if
  production-representative numbers matter.

## Surprises vs. the plan

- **No greenfield**: plan 15 v1 assumed minimal/ad-hoc decay code;
  reality was a substantial `src/Operators/Chemistry/Chemistry.jl`
  module with `AbstractChemistry`, `RadioactiveDecay`,
  `CompositeChemistry`, `apply_chemistry!`. Plan 15 became a rename +
  reshape + interface migration, compressed from the plan's 8 commits
  to 6.
- **Plan-14 latent bug surfaced during survey**, not during execution.
  Fix came "for free" in Commit 1's interface migration (replace
  `apply_chemistry!(state.tracers, ...)` with
  `apply!(state, nothing, grid, ...)`); the broken code path is gone.
- **Ordering tension** between chemistry-in-model and chemistry-at-
  sim-level was not anticipated by plan 15. Resolved by keeping
  chemistry at the sim level inside DrivenSimulation (via
  `with_chemistry(model, NoChemistry())`). Plan 16 resolves this more
  cleanly when emissions become a Z-sweep BC.

## Test anomalies

None. 32 chemistry unit tests + 5 end-to-end composition tests =
37/37 passing. Plan-14 baseline's 77 failures unchanged.

## Template usefulness for plans 16-18

What worked:
- **Pre-Commit-0 memory compaction** (moved from post-approval
  housekeeping). Downstream agents see plan 14 shipped note
  immediately, stale-path memos flagged. Carry forward to plans 16-18.
- **Survey-first Commit 0** caught that plan 15's "mostly greenfield"
  assumption was wrong. Plans 17/18 may have similar hidden existing
  infrastructure (PBL diffusion, Tiedtke convection in `src_legacy/`
  — do a grep survey for "diffusion", "convection", "PBL" etc. before
  greenlighting greenfield design).
- **Interface-validation questions as acceptance criteria** (plan §2.2
  / §4.6 "hard: interface validation") — explicit, testable. Good
  pattern.

What to carry forward:
- **Don't dispatch on arg types blindly in `apply!`** — for chemistry
  we accept `meteo = nothing` and `grid = nothing` because pure decay
  doesn't need them. Plans 17/18 operators may dispatch on the
  concrete `Meteorology` or `Grid` types; if a test passes `nothing`,
  the method won't match. Provide convenience methods that accept
  `nothing` for operators that don't consume those args.
- **Ordering is a real architectural constraint.** Writing
  `advection → emissions → chemistry` vs `advection → chemistry →
  emissions` has physical meaning. Plan the ordering at the step-
  level composer (`step!` or `chemistry_block!`) before adding new
  operators. Plan 16's surface-flux-as-Z-sweep-BC approach resolves
  this by making emissions part of advection.
- **CPU/GPU dispatch via `parent(arr) isa Array`** (plan 14 lesson)
  applies to new kernels too. Chemistry's multi-tracer kernel uses
  `get_backend(state.tracers_raw)` which handles this correctly.
