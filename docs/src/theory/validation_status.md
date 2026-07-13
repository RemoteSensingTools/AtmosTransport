# Validation status

This page is the honest current-state report of what AtmosTransport has
been validated against, what hasn't been validated yet, and the
floating-point tolerances that hold in each case. The goal is to let an
atmospheric-transport practitioner decide quickly whether the level of
validation here meets their needs — and where the gaps are.

## Verification vs validation

The terms in their canonical sense:

- **Verification** ("are we solving the equations correctly?") — the
  test suite covers this for the three schemes
  (`UpwindScheme`, `SlopesScheme`, `PPMScheme`) that live in the
  shared kernel test matrix in `test/core/test_advection_kernels.jl`:
  uniform-invariance, mass-budget, and CPU/GPU-agreement tests all
  run for those three. `LinRoodPPMScheme` has a CPU CS runtime smoke test in
  the opt-in `test/orphan/test_cubed_sphere_runtime.jl`, plus core kernel and
  adjoint tests, but is not covered by the same per-step CPU/GPU matrix. The replay gate enforces the
  discrete-conservation contract on every preprocessor write and
  every opt-in runtime load, regardless of scheme. See
  [Conservation budgets](@ref) for the per-test breakdown.
- **Validation** ("are we solving the right equations?") — this page.
  Validation is comparison against external reference data (TM5, GCHP,
  observations, …) and is fundamentally less complete than
  verification.

If you only need the first — reproduction of a published algorithm
under your own forcing within tolerances documented in the test
suite — verification is solid (for the schemes that are in the
test matrix) and you can proceed. If you need cross-model
comparison or observational match, read the gaps below.

## What HAS been validated

### Synthetic-fixture suite (verification, comprehensive)

More than 100 files under `test/core/` run on every push and PR via the `CI`
workflow, with no external-data dependency. `test/runtests.jl` discovers that
tier dynamically; `test/regridding/` is also part of the default CI baseline.
Anchor tables:

| Property | Test files | Status |
|---|---|---|
| Uniform tracer invariance under a synthetic flow (relative error `< 1e-6`) | `test/core/test_advection_kernels.jl` covers CPU for `Upwind` / `Slopes` / `PPM`; GPU uniform coverage is `Upwind`. `LinRoodPPMScheme` is not in this matrix. | green where exercised |
| Global mass conservation (gradient IC, 4 steps) | `test/core/test_advection_kernels.jl` (CPU+GPU), `test/core/test_cubed_sphere_advection.jl` | green |
| Cross-window replay closure | `test/core/test_replay_consistency.jl` | green |
| Cross-day continuity (synthetic GEOS C8 fixture) | `test/core/test_geos_cs_passthrough.jl` | green |
| GEOS native CS preprocessor end-to-end (synthetic fixture) | `test/core/test_geos_reader.jl`, `test/core/test_geos_cs_passthrough.jl`, `test/core/test_geos_convection.jl` | green |
| Conservative regrid mass closure | `test/regridding/test_conservation.jl`, `test/core/test_ll_to_cs_regrid_script.jl` (script-level tolerance `1e-6`) | green |
| CPU / GPU agreement (4 ULP for Upwind 1-step; 16 ULP for Slopes / PPM 4-step, F32 and F64) | CUDA-gated test sets in `test/core/test_advection_kernels.jl`; LinRood is outside this comparison matrix | green where exercised |
| Operator dispatch (palindrome ordering and no-op branches) | `test/core/test_transport_model_convection.jl`, `test/core/test_tm5_convection.jl`, `test/core/test_diffusion_palindrome_contract.jl` | green |

Total core-suite cases: thousands; CI breaks down pass/fail per file.

### Real-data preprocessor smoke tests (verification with real input)

| Path | What was verified | Status |
|---|---|---|
| ERA5 spectral → LL 72×37 F32, Dec 2021 | preprocessor closes write-time replay gate; runtime steps cleanly; conservation tested via uniform IC | green (proven on disk) |
| ERA5 spectral → LL 144×73 F32, Dec 2021 | same | green |
| ERA5 spectral → CS C24 F32, Dec 2021 | same; F32-CS path requires the `f3b3abf` fix to spectral_synthesis.jl | green (post-`f3b3abf`) |
| ERA5 spectral → CS C90 F32, Dec 2021 | same | green (post-`f3b3abf`) |
| GEOS-IT C180 → CS C180 F64 native | preprocessor closes write-time replay gate; binary loads cleanly via `inspect_transport_binary.jl`; per-window snapshot output verified. The 2026-04-25 unified-chain validation flagged a runtime GPU step-1 blocker on the unmerged-vertical C180 binary; the production response targets the `merge_above_pressure = 0.25 hPa` 64-level product with adaptive substeps. Status against that product is tracked in current Catrine config notes. | preprocessor green; current C180 runtime path uses merged + adaptive-substep binaries |

### Model parity (TM5)

| What | Tests | Status |
|---|---|---|
| TM5 four-field convection (`entu/detu/entd/detd`) parity with the TM5 F90 reference | `test/core/test_tm5_preprocessing.jl`, `test/core/test_tm5_preprocessing_rates.jl`, `test/core/test_tm5_vs_cmfmc_parity.jl`, `test/core/test_tm5_driven_simulation.jl`, `test/core/test_tm5_process_day.jl`, `test/core/test_tm5_vertical_remap.jl` | green |
| Russell-Lerner slopes vs TM5's `advectx__slopes` / `advecty__slopes` | line-for-line port; derivation lives beside `_slopes_face_flux` in `src/Operators/Advection/reconstruction.jl` | green by construction (port verified via uniform-invariance + mass-budget tests) |

The TM5 parity work is the most thoroughly validated cross-model
comparison the runtime currently has.

## What HAS NOT been validated end-to-end

The following work is on the roadmap but **not yet done**:

| Gap | Why it matters | Status |
|---|---|---|
| **GCHP parity for full-physics CS runs** | The CMFMC convection and ImplicitVerticalDiffusion operators are independently unit-tested but a full multi-day GCHP-vs-AtmosTransport intercomparison on identical met forcing has not been published. | run scripts exist (`scripts/diagnostics/compare_*` family) but no committed parity report |
| **CATRINE D7.1 intercomparison** | The European CATRINE protocol is the natural validation target (4 tracers: CO2, fossil CO2, SF6, 222Rn; full-physics; multi-month). The configs (`config/runs/catrine_*.toml`) exist and the runtime can produce the output, but no maintained end-to-end CATRINE smoke test, full multi-month regression, or published comparison memo exists. | protocol configs only; end-to-end regression not wired |
| **Observational closure** | Comparison of model output (column CO2, surface SF6 etc.) against an observational network (NOAA in-situ + TCCON / OCO satellite) | not started |
| **Multi-month GPU production runs** | Multi-day experiment configs exist, but CI has no multi-week, real-data GPU regression. | not regression-tested |
| **Adjoint kernels** | See [Adjoint status](@ref). Tape + checkpoint + revolve (bisection variant), the supported advection/convection and halo reverse paths, and the 4D-Var driver are on CI. Gaps: optimized/clamped convection variants, optimal binomial Revolve, and TM5-4DVAR cross-validation. | partial (shipped) |

## Floating-point tolerance practice

Tolerances vary by operation; the canonical sources:

| Operation | F64 tolerance | F32 tolerance | Reference |
|---|---|---|---|
| Per-window replay gate | `1e-10` | `1e-4` | `src/MetDrivers/ReplayContinuity.jl::replay_tolerance(FT)` |
| Window-continuity verification (test variant) | `1e-12` | `1e-6` | `test/core/test_replay_consistency.jl` |
| Per-step uniform-tracer invariance (relative) | `1e-6` | `1e-6` | `test/core/test_advection_kernels.jl` |
| 4-step total mass conservation (gradient IC, structured grid) | `1e-12` | `5e-5` | `test/core/test_advection_kernels.jl` |
| CPU/GPU advection agreement | `4 * eps(FT)` (Upwind 1-step) / `16 * eps(FT)` (Upwind / Slopes / PPM 4-step) | same as F64 column | CUDA-gated test sets in `test/core/test_advection_kernels.jl`; LinRood is not in this matrix. |
| Conservative regrid mass closure (script-level acceptance) | `≤ 1e-6` rel | same | `test/core/test_ll_to_cs_regrid_script.jl` |
| Cross-day GEOS chain continuity | machine epsilon (`5.94e-16` F64 measured) | `~3.5e-7` F32 measured | preprocessor stdout from `process_day` |

The F64 tolerances reflect double-precision noise floors at production
resolutions; F32 tolerances reflect single-precision accumulation.
Production runs on the L40S GPU use F32 by default — the F32 noise
floor is the operational tolerance.

## What this means for users

If you are doing:

- **Advection algorithm research** → verification is solid, F32 / F64
  noise-floor agreement is well-tested. Proceed.
- **CO2 intercomparison studies that need GCHP-equivalent fidelity** →
  the underlying operators are TM5-faithful or GCHP-style; the
  end-to-end intercomparison report has not been written. Run a
  side-by-side and compare yourself; the run scripts in
  `scripts/diagnostics/compare_*` are the starting point.
- **Inverse modelling that needs an adjoint** → the adjoint and
  4D-Var stack ship on CS. See [Adjoint status](@ref) for the supported
  scheme matrix and the remaining gaps (optimized/clamped convection
  variants and TM5-4DVAR cross-validation).
- **Validation against observations** → not in scope today; the
  forward model has the fidelity, but the observation-comparison
  diagnostics are external.

## Where to read next

- [Adjoint status](@ref) — what the README claims vs what actually
  ships.
- [Conservation budgets](@ref) — the explicit `@test` assertions
  that anchor the verification claims above.
- [TOML schema](@ref) — configuration reference for the runs that drive
  the validation work above.
