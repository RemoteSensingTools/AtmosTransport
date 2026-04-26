# Conservation budgets

The conservation contract from [Mass conservation](@ref) is enforced
by **explicit `@test` assertions** in the test suite. This page lists
the tests that anchor each conservation property, with file:line
citations and the exact tolerance bounds so a reader can verify by
inspection.

All tolerance values quoted here come straight from the test source;
none of them are bit-exact. Bit-exact conservation isn't achievable
in floating-point — what the tests assert is closure within
machine-precision noise floors that scale with `FT`.

## Uniform tracer preservation

The simplest non-trivial conservation property: a tracer initialised
to a uniform constant value `χ_0` everywhere should stay close to
`χ_0` after multi-step advection on a non-trivial flow.

| Test (`test/test_advection_kernels.jl`) | Asserts | Tolerance |
|---|---|---|
| `"CPU $precision_tag: uniform invariance"` (line 153–157) | `maximum(abs(χ_out − 4.0e-4)) / 4.0e-4 < 1e-6` after one Strang step on a sinusoidal flow. CPU coverage spans `Upwind` / `Slopes` / `PPM`. | `< 1e-6` (both F32 and F64) |
| `"GPU $precision_tag: uniform invariance"` (line 201–214) | same assertion on `CuArray`-backed state — but only in the **`Upwind`** variant of the test loop. The Slopes / PPM uniform-invariance tests at lines 303 and 407 are CPU-only. | `< 1e-6` |

The test fixture builds a synthetic LL `36 × 18 × 4` setup
(see lines 38–50 for the construction) with a sinusoidal zonal `am`
tapered with latitude and a `cm` diagnosed from continuity to keep
the synthetic forcing self-consistent. CFL ≈ 0.15 at the equator.
Schemes covered: `UpwindScheme`, `SlopesScheme`, `PPMScheme` — each
in its own `@testset` block. CPU coverage spans all three; **GPU
uniform-invariance coverage is `Upwind` only** (line 201). The
Slopes / PPM uniform tests at lines 303 and 407 run on CPU only.

## Mass-budget conservation

Same kernel, but on a non-uniform tracer (a meridional gradient
`rm_grad_cpu`):

| Test (`test/test_advection_kernels.jl`) | Asserts | Tolerance |
|---|---|---|
| `"CPU $precision_tag: mass conservation (uniform)"` (line 159–164) | `abs(Σ m_out − Σ m_cpu) / Σ m_cpu < tol` and the same for tracer mass | F64: `< 1e-13`. F32: `< 1e-5` |
| `"CPU $precision_tag: mass conservation (gradient, 4 steps)"` (line 166–171) | same assertion after 4 Strang steps with a gradient IC | F64: `< 1e-12`. F32: `< 5e-5` |
| `"GPU $precision_tag: mass conservation (gradient, 4 steps)"` (line 182–199) | same on a `CuArray`-backed state | F64: `< 1e-12`. F32: `< 5e-5` |
| `"CPU $precision_tag: non-trivial transport"` (line 173–176) | `maximum(abs(rm_out − rm_grad_cpu)) > 0` — sanity that the run actually moved tracer mass (not just held it constant) | strict `>` |

`test/test_cubed_sphere_advection.jl` (lines 374–455 cover the PPM-
on-CS variants of the same suite, exercising the panel-edge halo
sync end-to-end).

## CPU / GPU agreement

Bit-exact CPU/GPU agreement is **not** asserted: floating-point
associativity differs between sequential CPU sums and parallel GPU
reductions, and on F64 the GPU's FMA instruction can produce a
small ULP-scale difference per multiply-add. The test suite uses
ULP-bounded tolerances that vary by scheme and step count:

| Test (`test/test_advection_kernels.jl`) | Tolerance |
|---|---|
| Upwind, 1 step (line 216-235) | `4 * eps(FT)` per cell |
| Upwind, 4 steps (line 237-257) | `16 * eps(FT)` per cell |
| Slopes, 4 steps (line 345-363) | `16 * eps(FT)` per cell |
| PPM, 4 steps (line 451-469) | `16 * eps(FT)` per cell |

`LinRoodPPMScheme` is not in this CPU/GPU agreement matrix; it has a
CPU CS runtime smoke test in `test/test_cubed_sphere_runtime.jl:322`
but no per-step GPU comparison.

The CPU/GPU agreement check runs ONLY when CUDA.jl is loaded; it's
gated by `HAS_GPU` at the top of the test file. CI runs the CPU
side; GPU coverage is exercised on machines with hardware.

## Cross-window mass closure (replay gate)

Per-window replay is the contract that lets the runtime stream window
N+1 starting from window N's evolved endpoint without drift. Tested
in `test/test_replay_consistency.jl` (Plan 39 Commit H regressions):

| Subtest (line) | Asserts |
|---|---|
| `verify_window_continuity_ll` with continuity-consistent data (line 80–89) | `tol_rel ≤ 1e-12` (F64) / `1e-6` (F32). Both pass on the synthetic fixture. |
| Deliberately broken `cm` storage (line 145–161) | The storage-level gate fires `@test_throws ErrorException` when the binary's stored `cm` violates the explicit-`dm` closure by more than `replay_tolerance(FT)`. The earlier subtest (line 91-110) checks the diagnostic max-residual numbers against the same broken `cm`; the throw lives in the storage subtest. |
| Final-window inconsistent `cm` (line 145–161) | The LL storage replay-gate test deliberately injects an inconsistent final-window `cm` and asserts `@test_throws ErrorException` — i.e. the gate **does** detect inconsistency at the day boundary, not silently pass it as zero-tendency. |

The same gate runs at preprocessing write time (always) and at
runtime load time (opt-in via `[met_data] validate_replay = true` or
`ATMOSTR_REPLAY_CHECK = 1`). The test exercises the function
directly so the contract is validated independently of the
preprocessor / runtime drivers.

## Cross-topology (regridding) conservation

When mass moves between topologies (LL → CS in the spectral
preprocessor; LL → RG in the LL→RG variant), the conservative
regridder preserves total mass exactly but may shift the per-level
distribution by `O(10⁻⁶)`.

| Test | Asserts |
|---|---|
| `test/regridding/test_conservation.jl` | `sum(m_dest) ≈ sum(m_source)` under LL↔CS conservative regrid. |
| `test/test_ll_to_cs_regrid_script.jl:175–178` | End-to-end LL → CS regrid pipeline produces a binary whose stored `m`, summed globally, matches the LL source's stored `m` to `1e-6` relative — the script-level acceptance gate, deliberately looser than the kernel-level `1e-13` to absorb the per-level mass-consistency redistribution. |

The per-level mass-consistency correction in
`cs_transport_helpers.jl::_enforce_perlevel_mass_consistency!` is
what closes the per-level distribution; tested implicitly via the
Poisson balance convergence requirement (would not converge to the
plan-39 dry-basis tolerance without the correction).

## Initial-condition mass conservation

When a uniform-VMR initial condition is constructed, the conversion
`χ × air_mass` should yield a tracer-mass field whose ratio to
`air_mass` is `χ` everywhere. Trivially true for a uniform IC;
matters when the IC interpolates from a different mesh.

| Test | Asserts |
|---|---|
| `test/test_initial_condition_io.jl` | The `from_netcdf` IC kind round-trips a known field through the IC pipeline and asserts the recovered mixing ratio matches the source within tolerance. |
| `test/test_basis_explicit_core.jl` | Dry-basis IC interpretation: `[init.uniform_value = 4.0e-4]` produces a tracer field whose `mixing_ratio(state, :CO2)` is `4.0e-4` exactly when `air_mass` is on dry basis. |

## Test-pass status

The `core_tests` set in `test/runtests.jl` (lines 26–66) ships
**39 test files** that all run without external met data. The CI
workflow runs the core suite on every push and PR; recent runs are
clean. Real-data tests (lines 75–92, gated by `--all`) require
preprocessed binaries in `~/data/AtmosTransport/`.

Indicative case counts on the conservation-relevant tests (from the
Phase 3 / 5 verification runs already in this branch):

| Test file | Pass count |
|---|---|
| `test_advection_kernels.jl` | large (~hundreds across CPU+GPU × 4 schemes × multiple precisions) |
| `test_geos_cs_passthrough.jl` | 3467 |
| `test_geos_convection.jl` | 26 |
| `test_geos_reader.jl` | 48 |
| `test_cs_panel_geographic_roundtrip.jl` | 84 |
| `test_replay_consistency.jl` | smaller (~30) — but the regressions it covers are critical-path |

Total core-suite case count: thousands; CI reports per-test pass /
fail breakdown on every run.

## What's next

- [Validation status](@ref) — what we've validated end-to-end against
  external reference data, vs what's still synthetic-fixture-only.
- [Adjoint status](@ref) — what's actually shipped vs what's
  roadmap.
