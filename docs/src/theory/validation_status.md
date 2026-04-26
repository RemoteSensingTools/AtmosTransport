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
  shared kernel test matrix in `test/test_advection_kernels.jl`:
  uniform-invariance, mass-budget, and CPU/GPU-agreement tests all
  run for those three. `LinRoodPPMScheme` has a CPU CS runtime smoke
  test in `test_cubed_sphere_runtime.jl:322` but is not covered by
  the same per-step kernel matrix. The replay gate enforces the
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

The **39** core test files in `test/runtests.jl` lines 26–66 run on
every push and PR via the `CI` workflow, with no external data
dependency:

| Property | Test files | Status |
|---|---|---|
| Uniform tracer invariance under a synthetic flow (relative err < 1e-6) | `test_advection_kernels.jl` covers CPU for `Upwind` / `Slopes` / `PPM` (line 153); GPU coverage is `Upwind` only (line 201). `LinRoodPPMScheme` is not in this matrix. | green where exercised |
| Global mass conservation (gradient IC, 4 steps) | `test_advection_kernels.jl` (CPU+GPU; line 166-199), `test_cubed_sphere_advection.jl` | green |
| Cross-window replay closure | `test_replay_consistency.jl` (Plan 39 H gate) | green |
| Cross-day continuity (synthetic GEOS C8 fixture) | `test_geos_cs_passthrough.jl` (3467 cases) | green |
| GEOS native CS preprocessor end-to-end (synthetic fixture) | `test_geos_reader.jl` (48), `test_geos_cs_passthrough.jl`, `test_geos_convection.jl` (26) | green |
| Conservative regrid mass closure | `test/regridding/test_conservation.jl`, `test_ll_to_cs_regrid_script.jl` (script-level tol `1e-6`) | green |
| CPU / GPU agreement (4 ULP for Upwind 1-step; 16 ULP for Slopes / PPM 4-step, F32 and F64; LinRood not in matrix) | `test_advection_kernels.jl::"CPU-GPU agreement"`; CUDA-gated | green where exercised |
| Operator dispatch (Strang palindrome ordering, NoOp dead branches) | `test_transport_model_convection.jl`, `test_tm5_convection.jl`, `test_cmfmc_convection.jl` | green |

Total core-suite cases: thousands; CI breaks down pass/fail per file.

### Real-data preprocessor smoke tests (verification with real input)

| Path | What was verified | Status |
|---|---|---|
| ERA5 spectral → LL 72×37 F32, Dec 2021 | preprocessor closes write-time replay gate; runtime steps cleanly; conservation tested via uniform IC | green (proven on disk; matches the quickstart bundle) |
| ERA5 spectral → LL 144×73 F32, Dec 2021 | same | green |
| ERA5 spectral → CS C24 F32, Dec 2021 | same; F32-CS path requires the `f3b3abf` fix to spectral_synthesis.jl | green (post-`f3b3abf`) |
| ERA5 spectral → CS C90 F32, Dec 2021 | same | green (post-`f3b3abf`) |
| GEOS-IT C180 → CS C180 F64 native | preprocessor closes write-time replay gate; binary loads cleanly via `inspect_transport_binary.jl`; per-window snapshot output verified. Runtime GPU step on the C180 binary failed at step 1 in the unified-chain validation memo of 2026-04-25 (`docs/validation/geosit_c180_unified_chain_2026_04_25.md:70`); the runtime stepping itself is therefore **not yet proven** on real C180 GPU. | preprocessor green; runtime GPU step pending fix |

### Model parity (TM5)

| What | Tests | Status |
|---|---|---|
| TM5 four-field convection (`entu/detu/entd/detd`) parity with the TM5 F90 reference | `test_tm5_preprocessing.jl`, `test_tm5_preprocessing_rates.jl`, `test_tm5_vs_cmfmc_parity.jl`, `test_tm5_driven_simulation.jl`, `test_tm5_process_day.jl`, `test_tm5_vertical_remap.jl` | green |
| Russell-Lerner slopes vs TM5's `advectx__slopes` / `advecty__slopes` | line-for-line port; documented in `reconstruction.jl:266` and `:213-265` derivation | green by construction (port verified via uniform-invariance + mass-budget tests) |

The TM5 parity work is the most thoroughly validated cross-model
comparison the runtime currently has.

## What HAS NOT been validated end-to-end

The following work is on the roadmap but **not yet done**:

| Gap | Why it matters | Status |
|---|---|---|
| **GCHP parity for full-physics CS runs** | The CMFMC convection and ImplicitVerticalDiffusion operators are independently unit-tested but a full multi-day GCHP-vs-AtmosTransport intercomparison on identical met forcing has not been published. | run scripts exist (`scripts/diagnostics/compare_*` family) but no committed parity report |
| **CATRINE D7.1 intercomparison** | The European CATRINE protocol is the natural validation target (4 tracers: CO2, fossil CO2, SF6, 222Rn; full-physics; multi-month). The configs (`config/runs/catrine_*.toml`) exist; the runtime can produce the output. The **gated 1-day smoke test** `test/test_tm5_catrine_1day.jl` (in the `--all` suite) exercises the Catrine TM5-physics setup over a single day, but **no full multi-month CATRINE-protocol regression test** is committed and no protocol-vs-reference comparison memo has been published. | gated 1-day smoke test in place; output runs successfully (see `docs/validation/geosit_c180_unified_chain_2026_04_25.md` — internal memo); full protocol regression not yet wired |
| **Observational closure** | Comparison of model output (column CO2, surface SF6 etc.) against an observational network (NOAA in-situ + TCCON / OCO satellite) | not started |
| **Multi-month GPU production runs** | The longest GPU validation run committed is 7 days. Multi-week stability has been spot-checked but not regression-tested. | committed test ceiling: 7-day; production target: ~30-day |
| **Adjoint kernels** | See [Adjoint status](@ref) — the README's "TM5-4DVar-style adjoint with Revolve checkpointing" is a roadmap item, not shipped code. | NOT shipped |

## Floating-point tolerance practice

Tolerances vary by operation; the canonical sources:

| Operation | F64 tolerance | F32 tolerance | Reference |
|---|---|---|---|
| Per-window replay gate | `1e-10` | `1e-4` | `src/MetDrivers/ReplayContinuity.jl::replay_tolerance(FT)` |
| Window-continuity verification (test variant) | `1e-12` | `1e-6` | `test/test_replay_consistency.jl:84` |
| Per-step uniform-tracer invariance (relative) | `1e-6` | `1e-6` | `test_advection_kernels.jl:153–157` |
| 4-step total mass conservation (gradient IC, structured grid) | `1e-12` | `5e-5` | `test_advection_kernels.jl:166–171` |
| CPU/GPU advection agreement | `4 * eps(FT)` (Upwind 1-step) / `16 * eps(FT)` (Upwind 4-step, Slopes 4-step, PPM 4-step) | same as F64 column | `test_advection_kernels.jl:216, 237, 345, 451` (LinRoodPPMScheme not in this matrix; only a CPU CS smoke at `test_cubed_sphere_runtime.jl:322`) |
| Conservative regrid mass closure (script-level acceptance) | `≤ 1e-6` rel | same | `test/test_ll_to_cs_regrid_script.jl:175–178` |
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
- **Inverse modelling that needs an adjoint** → the adjoint is **not
  shipped**. See [Adjoint status](@ref) for what's on the roadmap.
- **Validation against observations** → not in scope today; the
  forward model has the fidelity, but the observation-comparison
  diagnostics are external.

## Where to read next

- [Adjoint status](@ref) — what the README claims vs what actually
  ships.
- [Conservation budgets](@ref) — the explicit `@test` assertions
  that anchor the verification claims above.
- *Phase 7: Configuration & Runtime* — TOML schema for the run
  configs that drive the validation work above.
