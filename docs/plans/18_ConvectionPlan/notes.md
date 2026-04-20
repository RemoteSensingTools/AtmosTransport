# Plan 18 — Execution Notes

Retrospective + deviations log for plan 18 execution. Fill in as
execution proceeds (CLAUDE.md: "retrospective sections are cumulative,
not start-at-end").

## Baseline (Commit 0)

- **Parent tip:** `5c3cf28` (plan 17 complete, branch `surface-emissions`).
- **Plan 18 branch:** `convection`, forked from surface-emissions.
- **Prerequisites shipped:** A1, A2, A3 from `PRE_PLAN_18_FIXES.md`.
  - **A2** (`fc5e255`): removed `test/test_real_era5_v1_vs_v2.jl`;
    dedup'd `test_dry_flux_interface.jl` include; deleted obsolete
    Test 8.
  - **A3** (`6506044`): added `current_time(sim::DrivenSimulation) = sim.time`,
    added `current_time(::Nothing) = 0.0` fallback, changed
    `DrivenSimulation.step!` to `meteo = sim`. Shipped with
    `test/test_current_time.jl` (10 tests).
  - **A1** (`126dbdc`): face-indexed `apply!` now accepts
    `diffusion_op`, `emissions_op`, `meteo` kwargs; defaults fall
    through bit-exact; non-trivial operators raise ArgumentError
    pointing at Plan 18b. Also bumped `cm_fill` in the "oversized cm"
    test from `1e-3` to `0.05` to match the updated
    `max_rel_cm = 0.01` threshold (pre-existing silently broken
    test, newly exposed).
- **Baseline commit:** 126dbdc.
- **77 pre-existing test failures** preserved across
  `test_basis_explicit_core.jl` (2), `test_structured_mesh_metadata.jl`
  (3), `test_poisson_balance.jl` (72). Invariant throughout plan 18.

## Pre-existing issue discovered during A2 work

`test_dry_flux_interface.jl` segfaults under Julia 1.12 codegen in
the "Mass conservation (zonal flow)" testset (line 425 after the
A2 edits). Verified present on pristine `surface-emissions` before
any A2 changes. PRE_PLAN_18_FIXES §A2 hypothesized the duplicate
`include` was the cause; evidence rules that out. The segfault is
a Julia 1.12 codegen bug in an unrelated testset. Plan 18 does not
try to fix it. Not counted against baseline (the test is excluded
from the 77-failure list).

## Follow-up plan candidates (registered at Commit 0)

Per plan 18 v5.1 Part 6:

- **Plan 18b: Face-indexed physics suite.** Face-indexed analogs of
  `ImplicitVerticalDiffusion`, `SurfaceFluxOperator`, and the new
  `CMFMCConvection` / `TM5Convection`. Plan 18 ships structured-only
  (Decision 25); face-indexed error stubs point at 18b. Scope:
  kernels for `(ncells, Nz)` layout; extension of A1's face-indexed
  `apply!` palindrome body to include V and S.
- **Plan 19: Adjoint operator suite.** Backward-mode kernel ports.
  Plan 18 preserves forward structure for mechanical adjoint (no
  positivity clamp, explicit matrix storage for TM5, named-local
  coefficients for CMFMC); 19 ships the backward kernels.
- **DerivedConvMassFluxField** — online CMFMC / DTRAIN from parent
  T, q, p.
- **Wet deposition operator** — reintroduces QC_PRES/QC_SCAV split,
  four-term tendency form for soluble tracers, precip flux fields,
  Henry's law.
- **Multi-tracer fusion in TM5 matrix solve** — build coefficients
  once per column, back-substitute Nt times.
- **GPU-batched BLAS paths** — cuBLAS/rocBLAS getrfBatched.
- **Shared-memory LU factorization** — for Nz≥72 columns.
- **AbstractLayerOrdering{TopDown, BottomUp}** — deferred from plan 17.
- **Palindrome-internal convection** — if Commit 9 study suggests.
- **TiedtkeConvection as standalone type** — if CMFMC-only fallback
  gets heavy production use.
- **Plan 16c retroactive Tier B/C for diffusion** — with basis audit.
- **Sub-window-varying forcing infrastructure** — if the pattern
  generalizes beyond convection.

## Commits executed

### Commit 0 (this commit): docs + baseline + upstream survey

- Docs under `docs/plans/18_ConvectionPlan/` (prior commit `e20d57f`).
- `artifacts/plan18/` initialized: baseline_commit.txt,
  baseline_test_summary.log, existing_convection_survey.txt,
  upstream_fortran_notes.md, upstream_gchp_notes.md, validation/,
  perf/, position_study/ subdirectories.
- Interface claims re-verified against tree at 126dbdc — all match
  plan 18 v5.1 §1.3 table.
- `AbstractConvection` stub at
  `src/Operators/AbstractOperators.jl:41` is orphaned (no subtypes,
  no usage); plan 18 introduces `AbstractConvectionOperator`
  alongside, following the `*Operator`-suffix convention set by
  plan 16b (`AbstractDiffusionOperator`) and plan 17
  (`AbstractSurfaceFluxOperator`). The existing stub is left alone.
- ERA5 binary reader already has `has_cmfmc`, `has_tm5conv`,
  `load_cmfmc_window!`, `load_tm5conv_window!`. Commit 7 will
  mirror these onto `TransportBinaryReader` (generic) and thread
  through the driver → window → model path.

## Deviations from plan doc §4.4 (updated per commit)

- **Commit 2 will absorb window-struct + model-field extensions**
  per v5.1 revision (§2.17 Decision 23 + §2.4 Decision 22).
- TBD — update as execution continues.

## Decisions beyond the plan

TBD — fill in during execution.

## Surprises

- **A2 segfault hypothesis wrong.** Duplicate `include` in
  `test_dry_flux_interface.jl` was real but not the segfault cause;
  segfault persists after dedup. A Julia 1.12 codegen bug in a
  different testset. PRE_PLAN_18_FIXES spec's "duplicate include is
  the likely cause but not proven" language saved us from a wrong
  certainty — kept A2 scope at the real deliverable (stale harness
  cleanup).
- **A1 exposed a pre-existing test bug.** `cm_fill=1e-3` in
  `test_transport_binary_reader.jl` was calibrated against an old
  `max_rel_cm=1e-8` threshold that commit `d4cc36d` loosened to
  `0.01`. The test had been silently broken since then, masked by
  the face-indexed MethodError that aborted the suite earlier.
  Fix folded into A1 (bump cm_fill to 0.05).

## Interface validation findings

TBD — populated in Commit 11 after Commits 1-10 execute.

## Template usefulness for plans 19+

TBD — populated in Commit 11.
