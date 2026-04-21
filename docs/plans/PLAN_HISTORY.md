# Plan History

Canonical manifest of plan status for the basis-explicit transport
core. Update as plans ship or change status. When a plan merges to
its target branch, add the merge-commit SHA.

This file is the source of truth for plan-level status. Individual
executable plan docs (11–18, 22) were deleted in plan 21 (Phase 4b);
their durable execution history lives in the `<N>_*_PLAN/NOTES.md`
subdirectories and in the per-plan summaries below. For original
intent, use git archaeology:
`git show ec2d2c0:docs/plans/<N>_*_PLAN*.md`.

## Shipped plans

| Plan | Title | Status | Key artifact | NOTES.md |
|------|-------|--------|--------------|----------|
| 11 | Ping-pong buffer refactor | shipped | workspace ping-pong pairs in advection state | — |
| 12 | Scheme type consolidation | shipped | `src/Operators/Advection/schemes.jl` | — |
| 13 | Sync removal + CFL pilot unification | shipped | unified CFL pilot; sync thesis disproven | — |
| 14 | Single pipeline (accessor API) | shipped | `src/State/Tracers.jl`, 4D `tracers_raw` | [14_SINGLE_PIPELINE_PLAN/NOTES.md](14_SINGLE_PIPELINE_PLAN/NOTES.md) |
| 15 | Slow chemistry | shipped | `src/Operators/Chemistry/` | [15_SLOW_CHEMISTRY_PLAN/NOTES.md](15_SLOW_CHEMISTRY_PLAN/NOTES.md) |
| 16 | Vertical diffusion | shipped | `src/Operators/Diffusion/` (rank-4 LatLon) | [16_VERTICAL_DIFFUSION_PLAN/NOTES.md](16_VERTICAL_DIFFUSION_PLAN/NOTES.md) |
| 16A | Time-varying fields | shipped | `src/State/Fields/` | [16A_TIME_VARYING_FIELDS_PLAN/NOTES.md](16A_TIME_VARYING_FIELDS_PLAN/NOTES.md) |
| 17 | Surface emissions | shipped | `src/Operators/SurfaceFlux/`, palindrome centering | [17_SURFACE_EMISSIONS_PLAN/NOTES.md](17_SURFACE_EMISSIONS_PLAN/NOTES.md) |
| 18 | Convection | **paused at Commit 3** | `src/Operators/Convection/` (structured kernels) | — |
| 22 | Topology completion (A/B/C/D) | shipped | RG + CS runtime, three-topology convection | see per-plan retrospective below |
| 21 | Post–plan-22 stabilization | **in progress** (this branch) | this file, `TOPOLOGY_SUPPORT.md`, legacy deletion | — |

## Pending plans

| Plan | Title | Status | Blocked by |
|------|-------|--------|------------|
| 19 | Adjoint operator suite | pending | 21 |
| 20 | Documentation overhaul (Documenter + Literate) | pending | 21, 19 |

## Per-plan retrospective summary

Each entry below captures what shipped, any key surprise, and pointers
to durable artifacts. Plans with a NOTES.md defer technical detail
there; plans without one carry a fuller summary here.

### Plan 11 — Ping-pong buffer refactor

Replaced single-buffer advection with paired `(rm_A, m_A)` / `(rm_B, m_B)`
workspaces across all directional sweeps. Strang palindrome parity
tracking keeps the final result in the caller's arrays with zero
intermediate `copyto!` in the common `n_sub == 1` case. See
Invariant 4 in `CLAUDE.md`.

### Plan 12 — Scheme type consolidation

Unified the advection scheme hierarchy to
`UpwindScheme | SlopesScheme{L} | PPMScheme{L, ORD}` with compile-
time dispatch. Removed legacy `@eval` dispatch tables. Equivalence
tests against reference implementations shipped in the same commit
sequence.

### Plan 13 — Sync removal + CFL pilot unification

Tested the hypothesis that removing `synchronize(backend)` from
Strang sweeps would give a measurable GPU win. Direct CUDA-event
measurement showed sync is ~10–12 μs per `strang_split!` on L40S
F32 — negligible vs. ~3–47 ms kernel time. The plan shipped the
unified CFL static-algorithm pilot anyway; the sync-removal
micro-optimization was rejected with a documented decision and
`bench_strang_sweep.jl --events` measurement harness. Reference:
[`artifacts/plan13/perf/sync_thesis_report.md`](../../artifacts/plan13/perf/sync_thesis_report.md).

### Plan 14 — Single pipeline (accessor API)

Switched `CellState.tracers` from a `NamedTuple` of 3D arrays to a
single 4D `Array{FT, 4}` (`tracers_raw`) with a `TracerAccessor`
wrapper preserving `state.tracers.CO2` property access. Multi-tracer
fusion gave 10–270% GPU speedup at production Nt=30. Detail:
[`14_SINGLE_PIPELINE_PLAN/NOTES.md`](14_SINGLE_PIPELINE_PLAN/NOTES.md).

### Plan 15 — Slow chemistry

Introduced the `apply!(state, meteo, grid, op, dt; workspace)`
contract. `ExponentialDecay`, `CompositeChemistry`, and the
chemistry block in `TransportModel.step!`. Detail:
[`15_SLOW_CHEMISTRY_PLAN/NOTES.md`](15_SLOW_CHEMISTRY_PLAN/NOTES.md).

### Plan 16 — Vertical diffusion

Implicit vertical diffusion (`ImplicitVerticalDiffusion`) with
`ProfileKzField` / `PreComputedKzField` / `DerivedKzField` Kz
sources via `Adapt.adapt_structure`. Column-serial Thomas solve;
~5–20% overhead at typical Nz=4–32 but up to 75% at Nz=72.
Detail: [`16_VERTICAL_DIFFUSION_PLAN/NOTES.md`](16_VERTICAL_DIFFUSION_PLAN/NOTES.md).

### Plan 16A — Time-varying fields

Introduced `AbstractTimeVaryingField{FT, N}` and the first concrete
subtype `ConstantField{FT, N}`. Retrofit chemistry rate fields to
the new abstraction. Detail:
[`16A_TIME_VARYING_FIELDS_PLAN/NOTES.md`](16A_TIME_VARYING_FIELDS_PLAN/NOTES.md).

### Plan 17 — Surface emissions

`StepwiseField` + `SurfaceFluxOperator` + `PerTracerFluxMap`.
Palindrome center becomes `V(dt/2) S V(dt/2)` when emissions
active; bit-exact plan 16b path when not. Ordering study: operator
A (recommended) = 12.1% surface fraction, D (pathological) =
100%. Detail: [`17_SURFACE_EMISSIONS_PLAN/NOTES.md`](17_SURFACE_EMISSIONS_PLAN/NOTES.md);
study results at
[`17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md`](17_SURFACE_EMISSIONS_PLAN/ordering_study_results.md).

### Plan 18 — Convection (PAUSED at Commit 3)

Shipped Commits 0–3 (prerequisites A1/A2/A3 + `AbstractConvectionOperator`
+ `ConvectionForcing` scaffolding + `CMFMCConvection` kernel with
CFL sub-cycling + well-mixed sub-cloud). Commits 4–11
(TM5Convection, cross-scheme, `step!` wiring, driver integration,
sim allocation, position study, bench, retrospective) NOT STARTED.

The runtime block wiring that Commit 6 would have done was
completed instead under **plan 22D** (a natural next step after
22A/B/C). Plan 18's remaining TM5 scheme, cross-scheme, and perf
commits remain deferred.

The plan-18 reference docs in `docs/plans/18_ConvectionPlan/`
(`18_CONVECTION_PLAN_v5.md`, upstream GCHP/Fortran notes,
`PRE_PLAN_18_FIXES.md`) are preserved because `src/` docstrings
cite `plan 18 v5.1 §2.X Decision Y` throughout
[`src/Models/TransportModel.jl`](../../src/Models/TransportModel.jl),
[`src/MetDrivers/ConvectionForcing.jl`](../../src/MetDrivers/ConvectionForcing.jl),
and the Convection operator submodule.

### Plan 22 — Topology completion (A/B/C/D) — inline retrospective

Replaces a missing `22_TOPOLOGY_COMPLETION_PLAN_v2/NOTES.md`.

**22A — RG column-local operators.** Face-indexed diffusion and
surface flux on reduced-Gaussian meshes. Merged into the unified
`apply!` contract; no new flux-state type needed.

**22B — CS runtime enablement.** `CubedSphereState` with
`NTuple{6, Array{FT, 4}}` panel-native storage. `TransportModel`
and `DrivenSimulation` now accept CS grids through the same
entry point as LatLon and RG. `CubedSphereBinaryReader` integrated
as a driver.

**22C — CS column-local operators.** Diffusion and surface flux
on cubed-sphere panel storage.

**22D — Topology convection runtime.** `TransportModel.step!` now
executes a convection block between transport and chemistry when
`!(model.convection isa NoConvection)`. `CMFMCConvection` ships
dedicated `apply!` methods for all three topologies (LatLon,
RG, CS). Completed the runtime wiring that plan 18 Commit 6 had
deferred. Tests: `test/test_transport_model_convection.jl` and
CS runtime tests in `test/test_cubed_sphere_runtime.jl`.

**Known debts at close:**

- `artifacts/plan22/perf/` SUMMARY_RG.md not produced (planned
  in v2 §7.2).
- ~~CS chemistry gap~~ — closed 2026-04-21 as a plan 21
  follow-up. `apply!(::CubedSphereState, ...)` shipped for
  `NoChemistry`, `ExponentialDecay`, and `CompositeChemistry`
  with matching `test/test_cs_chemistry.jl` (152 tests, F32 + F64).
- The plan 22 v2 doc (which was deleted in plan 21 Phase 4b)
  contained stale "convection deferred" lines at merge time;
  those were fixed in plan 21 Phase 1 before deletion.

### Plan 21 — Post–plan-22 stabilization (in progress)

This plan. Branch `convection` (tip TBD on completion). Phases 1–5
executed in this session; Phase 6 (Aqua/JET/README-freshness CI
gates) deferred to a follow-up session.

## Reference / strategic documents (kept, not deleted)

These are not executable plans but authoritative references:

- [`OPERATOR_COMPOSITION.md`](OPERATOR_COMPOSITION.md) — operator
  block-ordering contract
- [`TIME_VARYING_FIELD_MODEL.md`](TIME_VARYING_FIELD_MODEL.md) —
  field abstraction (ARCHITECTURE of `src/State/Fields/`)
- [`ARCHITECTURAL_SKETCH_v3.md`](ARCHITECTURAL_SKETCH_v3.md) —
  high-level architecture snapshot

`src/` docstrings cite these files directly and will continue to
resolve.

## Retired / historical

- `CLAUDE_additions.md` (deleted in plan 21 Phase 4b) — content
  already merged into top-level `CLAUDE.md` during plan 17 Session 2.
  Section headings in CLAUDE.md: "Plan execution rhythm", "Branch
  hygiene", "Julia / language gotchas", "Testing discipline",
  "Workflow: Adding a new physics operator", "What NOT to do".

## Note on NOTES.md files and legacy paths

Plan execution retrospectives live in `<N>_*_PLAN/NOTES.md`. These
are historical records from the time of execution. After plan 21's
legacy deletion (`src_legacy/`, `scripts_legacy/`, `test_legacy/`
removed), some NOTES.md files reference paths in those trees that
no longer exist on disk. This is by design: NOTES.md captures what
was consulted at the time.

To read the referenced legacy code: git archaeology at commit
`ec2d2c0` (parent of `d2e813d`, the last commit where legacy trees
existed):

```bash
git show ec2d2c0:src_legacy/<path>
git show ec2d2c0:scripts_legacy/<path>
git show ec2d2c0:test_legacy/<path>
```

Adjoint template files are additionally preserved as first-class
archival artifacts at
[`../resources/developer_notes/legacy_adjoint_templates/`](../resources/developer_notes/legacy_adjoint_templates/)
for plan 19.
