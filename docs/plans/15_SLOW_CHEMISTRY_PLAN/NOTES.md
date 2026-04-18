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

### Commit 1 — rewrite Chemistry.jl

(fill in)

### Commit 2 — chemistry_block!

(fill in)

### Commit 3 — step! + TransportModel + DrivenSimulation call-site fix

(fill in; this commit fixes the plan-14 latent bug)

### Commit 4 — end-to-end Rn-222 advection+decay composition test

(fill in)

### Commit 5 — bench + docs + retrospective

(fill in)

## Deferred observations

(Updated as execution proceeds.)

## Surprises vs. the plan

(Updated as execution proceeds.)

## Test anomalies

(Updated as execution proceeds.)

## Interface-validation findings

(Filled at Commit 5 retrospective — answer to plan §2.2:
"Did `apply!(state, meteo, grid, op, dt; workspace)` generalize beyond
advection without modification, or did advection's needs leak into the
signature?")
