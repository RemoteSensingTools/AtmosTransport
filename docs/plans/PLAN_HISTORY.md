# Plan History

Canonical manifest of plan status for the basis-explicit transport
core. Update as plans ship or change status.

This file is the source of truth for plan-level status. The original
executable plan docs (11–18, 22) and their per-plan execution
retrospectives (`<N>_*_PLAN/NOTES.md`) were deleted during and after
plan 21. For the original intent and retrospectives, use git
archaeology — both live in commit history.

## Shipped plans

| Plan | Title | Status | Key artifact |
|------|-------|--------|--------------|
| 11 | Ping-pong buffer refactor | shipped | workspace ping-pong pairs in advection state |
| 12 | Scheme type consolidation | shipped | `src/Operators/Advection/schemes.jl` |
| 13 | Sync removal + CFL pilot unification | shipped | unified CFL pilot; sync thesis disproven |
| 14 | Single pipeline (accessor API) | shipped | `src/State/Tracers.jl`, 4D `tracers_raw` |
| 15 | Slow chemistry | shipped | `src/Operators/Chemistry/` |
| 16 | Vertical diffusion | shipped | `src/Operators/Diffusion/` (rank-4 LatLon) |
| 16A | Time-varying fields | shipped | `src/State/Fields/` |
| 17 | Surface emissions | shipped | `src/Operators/SurfaceFlux/`, palindrome centering |
| 18 | Convection | **paused at Commit 3** | `src/Operators/Convection/` (structured kernels) |
| 22 | Topology completion (A/B/C/D) | shipped | RG + CS runtime, three-topology convection |
| 21 | Post–plan-22 stabilization | shipped | this file, `TOPOLOGY_SUPPORT.md`, legacy deletion, CI gates |
| 23 | TM5 convection | shipped | `src/Operators/Convection/TM5Convection.jl`, three-topology kernels, ec2tm! math |

## Pending plans

| Plan | Title | Status | Blocked by |
|------|-------|--------|------------|
| 24 | TM5 convection preprocessor integration | **in progress — Commit 0 baseline** | 23 |
| 19 | Adjoint operator suite | pending | 21, 23 |
| 20 | Documentation overhaul (Documenter + Literate) | pending | 21, 19 |

## Per-plan retrospective summary

Each entry below captures what shipped and any key surprise. For full
execution detail use git archaeology at the merge commit.

### Plan 11 — Ping-pong buffer refactor

Replaced single-buffer advection with paired `(rm_A, m_A)` / `(rm_B, m_B)`
workspaces across all directional sweeps. Strang palindrome parity
tracking keeps the final result in the caller's arrays with zero
intermediate `copyto!` in the common `n_sub == 1` case. See Invariant
4 in `CLAUDE.md`.

### Plan 12 — Scheme type consolidation

Unified the advection scheme hierarchy to
`UpwindScheme | SlopesScheme{L} | PPMScheme{L, ORD}` with compile-
time dispatch. Removed legacy `@eval` dispatch tables.

### Plan 13 — Sync removal + CFL pilot unification

Tested the hypothesis that removing `synchronize(backend)` from Strang
sweeps would give a measurable GPU win. Direct CUDA-event measurement
showed sync is ~10–12 μs per `strang_split!` on L40S F32 — negligible
vs. ~3–47 ms kernel time. The sync-removal micro-optimization was
rejected with a documented decision and
`bench_strang_sweep.jl --events` measurement harness. Reference:
[`../../artifacts/plan13/perf/sync_thesis_report.md`](../../artifacts/plan13/perf/sync_thesis_report.md).

### Plan 14 — Single pipeline (accessor API)

Switched `CellState.tracers` from a `NamedTuple` of 3D arrays to a
single 4D `Array{FT, 4}` (`tracers_raw`) with a `TracerAccessor`
wrapper preserving `state.tracers.CO2` property access. Multi-tracer
fusion gave 10–270% GPU speedup at production Nt=30.

### Plan 15 — Slow chemistry

Introduced the `apply!(state, meteo, grid, op, dt; workspace)`
contract. `ExponentialDecay`, `CompositeChemistry`, and the chemistry
block in `TransportModel.step!`.

### Plan 16 — Vertical diffusion

Implicit vertical diffusion (`ImplicitVerticalDiffusion`) with
`ProfileKzField` / `PreComputedKzField` / `DerivedKzField` Kz sources
via `Adapt.adapt_structure`. Column-serial Thomas solve; ~5–20%
overhead at typical Nz=4–32 but up to 75% at Nz=72.

### Plan 16A — Time-varying fields

Introduced `AbstractTimeVaryingField{FT, N}` and the first concrete
subtype `ConstantField{FT, N}`. Retrofit chemistry rate fields to the
new abstraction.

### Plan 17 — Surface emissions

`StepwiseField` + `SurfaceFluxOperator` + `PerTracerFluxMap`.
Palindrome center becomes `V(dt/2) S V(dt/2)` when emissions active;
bit-exact plan 16b path when not. Ordering study: operator A
(recommended) = 12.1% surface fraction, D (pathological) = 100%.

### Plan 18 — Convection (PAUSED at Commit 3)

Shipped Commits 0–3 (prerequisites A1/A2/A3 + `AbstractConvectionOperator`
+ `ConvectionForcing` scaffolding + `CMFMCConvection` kernel with CFL
sub-cycling + well-mixed sub-cloud). Commits 4–11 (TM5Convection,
cross-scheme, `step!` wiring, driver integration, sim allocation,
position study, bench, retrospective) NOT STARTED.

The runtime block wiring that Commit 6 would have done was completed
instead under **plan 22D** (a natural next step after 22A/B/C). Plan
18's remaining TM5 scheme, cross-scheme, and perf commits remain
deferred.

Legacy plan-18 reference docs (`18_CONVECTION_PLAN_v5.md`, upstream
GCHP/Fortran notes, `PRE_PLAN_18_FIXES.md`) were deleted in the
post-plan-21 aggressive cleanup. `src/` docstrings still cite `plan 18
v5.1 §2.X Decision Y` as prose references — those remain valid as
historical references to the design rationale (GCHP upstream remains
the authoritative source for the convection algorithm).

### Plan 22 — Topology completion (A/B/C/D) — inline retrospective

**22A — RG column-local operators.** Face-indexed diffusion and
surface flux on reduced-Gaussian meshes. Merged into the unified
`apply!` contract; no new flux-state type needed.

**22B — CS runtime enablement.** `CubedSphereState` with
`NTuple{6, Array{FT, 4}}` panel-native storage. `TransportModel` and
`DrivenSimulation` now accept CS grids through the same entry point as
LatLon and RG. `CubedSphereBinaryReader` integrated as a driver.

**22C — CS column-local operators.** Diffusion and surface flux on
cubed-sphere panel storage.

**22D — Topology convection runtime.** `TransportModel.step!` now
executes a convection block between transport and chemistry when
`!(model.convection isa NoConvection)`. `CMFMCConvection` ships
dedicated `apply!` methods for all three topologies (LatLon, RG, CS).
Completed the runtime wiring that plan 18 Commit 6 had deferred.
Tests: `test/test_transport_model_convection.jl` and CS runtime tests
in `test/test_cubed_sphere_runtime.jl`.

**Debts closed post-plan-22:**

- CS chemistry gap — closed 2026-04-21 as a plan 21 follow-up.
  `apply!(::CubedSphereState, ...)` shipped for `NoChemistry`,
  `ExponentialDecay`, and `CompositeChemistry` with matching
  `test/test_cs_chemistry.jl` (152 tests, F32 + F64).

### Plan 24 — TM5 convection preprocessor integration (IN PROGRESS)

Closing plan 23's explicitly-deferred preprocessor gap. Adds the
ERA5 → TM5 conversion pipeline so real ERA5 data can produce
transport binaries with populated `tm5_fields`:

- Port TM5's `ECconv_to_TMconv` (phys_convec_ec2tm.F90:87–237) as
  `ec2tm_from_rates!` alongside plan 23's existing `ec2tm!` in
  `src/Preprocessing/tm5_convection_conversion.jl`.
- NC→BIN conversion layer (`era5_physics_nc_to_bin.jl` +
  `era5_physics_binary_reader.jl`): archival NCs under
  `met/era5/<grid>/physics/`; mmap-friendly per-day binary
  intermediate under `met/era5/<grid>/physics_bin/YYYY/`.
- `process_day` hook reads the physics BIN, applies ec2tm with
  hydrostatic virtual-temperature dz, merges to transport Nz via
  the existing `merge_cell_field!`, and attaches `tm5_fields` to
  the window before `write_transport_binary` (plan 23 Commit 3
  writer emits the TM5 sections automatically).
- Monthly CDS chunked download via restored/updated
  `download_era5_physics.py`.
- End-to-end CATRINE-style 1-day test replaces plan 23 Commit 6's
  synthetic stub.

Commit sequence: 0 (baseline + survey) → 1 (`ec2tm_from_rates!` +
dz) → 2 (NC→BIN converter + reader) → 3 (vertical remap wiring)
→ 4 (LL `process_day` integration; RG/CS follow as 4b/4c) → 5
(download script + docs) → 6 (real-data end-to-end) → 7
(production audit + retrospective).

Decisions: single code path (drop main's mass-flux-only
fallback); bad-data handling mirrors TM5 F90 exactly (small-value
clip + top-search + negative redistribution, with diagnostic
counter); dz uses virtual-temperature `T_v = T(1 + 0.608·Q)`
from the same physics download (one step closer to TM5's
geopotential-height approach); binary intermediate keeps the
preprocessor NC-free (consistent with its GRIB-direct pattern).

### Plan 23 — TM5 convection (SHIPPED)

Shipped `TM5Convection` as a sibling of `CMFMCConvection` so ERA5
runs have a first-class four-field Tiedtke 1989 mass-flux scheme.
8 commits (0–7) landed on branch `convection` between 2026-04-21
and 2026-04-22. 170+ new tests; 0 regressions. Plan 18's original
Commits 4–5 folded into plan 23; plan 18 itself is marked "paused
at Commit 3" permanently.

What shipped:

- **Column solver** `_tm5_solve_column!`
  ([`src/Operators/Convection/tm5_column_solve.jl`](../../src/Operators/Convection/tm5_column_solve.jl)):
  backend-agnostic Julia port of TM5-4DVAR `tm5_conv.F90` into
  AtmosTransport orientation (k=1=TOA, k=Nz=surface; principle 1 —
  zero runtime orientation flips). Full partial-pivot LU on the
  `[1, Nz]` range; pivots stored per principle 3 for plan 19
  adjoint replay.
- **Per-topology kernels**
  ([`tm5_kernels.jl`](../../src/Operators/Convection/tm5_kernels.jl)):
  `@kernel` wrappers for LL 4D, RG 3D, and CS per-panel launches.
  Allocation-free inside the kernel via pre-allocated
  `TM5Workspace` scratch slabs (`f`, `fu`, `amu`, `amd` per column).
- **`TM5Convection` operator + workspace**
  ([`TM5Convection.jl`](../../src/Operators/Convection/TM5Convection.jl)):
  stateless struct dispatched on grid mesh type; three `apply!` /
  `apply_convection!` methods per topology. GPU backend via
  `KernelAbstractions.get_backend(q_raw)` (principle 4).
- **Runtime plumbing refactor**
  ([`_validate_convection_window!` in DrivenSimulation.jl](../../src/Models/DrivenSimulation.jl),
  `_convection_workspace_for(::TM5Convection, …)` in TransportModel.jl):
  per-operator dispatch replaces the Commit-0-era
  `if op isa CMFMCConvection / else throw` pattern.
- **ec2tm! math** ([`src/Preprocessing/tm5_convection_conversion.jl`](../../src/Preprocessing/tm5_convection_conversion.jl)):
  pure Julia port of TM5's `phys_convec_ec2tm.F90`. Derives
  `(entu, detu, entd, detd)` at layer centers from ECMWF
  `(mflu_ec, mfld_ec, detu_ec, detd_ec)`. Ships ready-to-call;
  preprocessor integration (wiring into `process_day` + adding
  ECMWF convective download) deferred as a follow-on plan.
- **TM5 binary sections across LL/RG/CS**:
  `TransportBinary.jl` (section-element tables, writer, reader,
  `has_tm5_convection`, `load_tm5_convection_window!`),
  `CubedSphereBinaryReader.jl` (`_cs_section_elements`,
  `load_cs_window` returns `raw.tm5_fields`),
  `CubedSphereTransportDriver.jl` (dropped hardcoded `nothing`).
- **Ten production-readiness principles** ratified during drafting
  and graded at Commit 7 retrospective
  ([`artifacts/plan23/production_audit.md`](../../artifacts/plan23/production_audit.md)):
  all 10 pass.
- **Benchmark on wurst L40S F32**
  ([`artifacts/plan23/bench.md`](../../artifacts/plan23/bench.md)):
  TM5 GPU speedups over CPU range from 3× (small Nt=1) to 43×
  (large Nt=30). Memory overhead ~63 KB/column from the
  Nz+1 × Nz × … scratch slabs needed for allocation-free kernel
  launch.

Tests: 170+ new testsets across
`test/test_tm5_convection.jl` (81),
`test/test_tm5_preprocessing.jl` (43),
`test/test_tm5_vs_cmfmc_parity.jl` (9),
`test/test_tm5_driven_simulation.jl` (15),
all registered in core_tests. README freshness gate tracks all
new files.

Deferred as follow-on:

- Preprocessor integration of `ec2tm!` into `process_day` +
  ECMWF convective-variable downloads from CDS.
- Real-data CATRINE-style 1-day ERA5 sim + plan-17-parallel
  operator-ordering study (A/B/C/D positions).
- `lmc`-limited factorization optimization (reduce work on
  columns with shallow or no convection; current full-range LU
  is cheap enough at Nz ≤ 72).
- Plan 19 adjoint kernel (pivots + factorization structure
  preserved for mechanical port).

### Plan 21 — Post–plan-22 stabilization

Six phases shipped:

1. Stale "deferred convection" doc text fixed.
2. `src_legacy/`, `scripts_legacy/`, `test_legacy/` deleted; adjoint
   templates archived at
   `docs/resources/developer_notes/legacy_adjoint_templates/` for
   plan 19.
3. `docs/00_SCOPE_AND_STATUS.md`, `AGENT_ONBOARDING.md`, `README.md`,
   `20_RUNTIME_FLOW.md` rewritten against current state.
4. This `PLAN_HISTORY.md` added as canonical manifest; shipped plan
   docs deleted; in a follow-up, per-plan NOTES.md subdirectories
   were also deleted aggressively (content captured here).
5. `src/Operators/TOPOLOGY_SUPPORT.md` canonical operator × topology
   matrix.
6. CI hard gates: `test/test_aqua.jl` (Aqua package health),
   `test/test_jet.jl` (JET hot-path inference snapshot),
   `test/test_readme_current.jl` (module-README freshness).

## Reference / strategic documents (kept)

These are not executable plans but authoritative references that
`src/` docstrings cite:

- [`OPERATOR_COMPOSITION.md`](OPERATOR_COMPOSITION.md) — operator
  block-ordering contract
- [`TIME_VARYING_FIELD_MODEL.md`](TIME_VARYING_FIELD_MODEL.md) —
  field abstraction (architecture of `src/State/Fields/`)
- [`ARCHITECTURAL_SKETCH_v3.md`](ARCHITECTURAL_SKETCH_v3.md) —
  high-level architecture snapshot

## Retired / historical

Git history preserves the full content of every deleted doc. To
consult:

- Per-plan execution retrospectives: `<N>_*_PLAN/NOTES.md` existed
  during plans 14–17; deleted in the post-plan-21 aggressive cleanup.
  Use `git log -- docs/plans/<N>_*_PLAN/` to find the commits.
- Plan-18 reference docs (`18_CONVECTION_PLAN_v5.md`, upstream notes,
  `PRE_PLAN_18_FIXES.md`): deleted in the post-plan-21 cleanup. GCHP
  upstream remains authoritative for convection algorithm; `src/`
  docstrings still prose-cite "plan 18 v5.1 §2.X Decision Y".
- `CLAUDE_additions.md`: content merged into top-level `CLAUDE.md`
  during plan 17 Session 2.
- `src_legacy/` / `scripts_legacy/` / `test_legacy/` code trees:
  removed in plan 21 Phase 2. Last commit containing them is `ec2d2c0`;
  use `git show ec2d2c0:<path>` for archaeology. Adjoint templates
  preserved at
  [`../resources/developer_notes/legacy_adjoint_templates/`](../resources/developer_notes/legacy_adjoint_templates/)
  for plan 19.
