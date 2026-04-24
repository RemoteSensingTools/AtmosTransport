# Plan 40 — Unified runtime I/O + Catrine CS48/CS180 validation — NOTES

## Baseline

- **Parent commit:** `2470812` — "Unify runtime physics recipes across runtime runners"
- **Branch:** `convection`
- **Captured at:** 2026-04-24 (UTC)
- **Full plan doc:** `/home/cfranken/.claude/plans/maybe-you-need-to-fancy-pie.md`
- **Follow-up plan (docs overhaul):** `/home/cfranken/.claude/plans/plan-41-docs-overhaul.md`
- **Baseline artifacts:**
  - `artifacts/plan40/baseline_commit.txt`
  - `artifacts/plan40/baseline_test_summary.log`
  - `artifacts/plan40/existing_infra_survey.txt`
- **Pre-existing test failure count at `2470812`:** **5 failures
  across 4 test files**, captured in `baseline_test_summary.log`.
  CLAUDE.md's "77 pre-existing failures" number is **stale** — plans
  21–39 cleaned up the majority. The current invariant to preserve
  is **5 failures**:

  | Test file | Failures | Note |
  |---|---|---|
  | `test_aqua.jl` | 2 | Method ambiguity (ambiguities.jl:78) + stale dep (stale_deps.jl:31) |
  | `test_jet.jl` | 1 | JET count is 123, baseline is 119 — 4 new reports, likely from Codex's recent CS-driven-runner commits |
  | `test_readme_current.jl` | 2 | `src/Models/README.md` + `src/MetDrivers/README.md` missing new files |
  | `runtests.jl` core loop (env artifact) | 0 real | Exit 1 is from `test_aqua.jl` load error (Aqua not in root env); all 93 internal testsets pass |

  CLAUDE.md's "77" should be corrected to "5" in plan 41's trim.
  This plan holds the 5-failure invariant; new commits **must not
  increase the count**, and any commit that closes one of these
  five (by updating the README, bumping the JET baseline with
  justification, or fixing an Aqua issue) must document it as a
  scope deviation in this NOTES file.

### Baseline test-env reality

The test harness uses **two environments** that don't overlap:

- Root project `Project.toml` — has `Adapt`, `NCDatasets`, etc., but
  lacks `Aqua` / `JET` (they're in `[extras]`, `[targets.test]`).
  Entry: `julia --project=. test/runtests.jl`.
- `test/Project.toml` — has `AtmosTransport`, `Aqua`, `JET`, `Test`
  only; lacks `Adapt` and other direct runtime deps.
  Entry: `julia --project=test test/test_aqua.jl` (per-file).

`julia --project=. -e 'using Pkg; Pkg.test()'` fails on Julia 1.12 with
`ArgumentError: Package Adapt not found` — `Pkg.test()`'s sandbox
isolation on 1.12 does not surface `Adapt` even though it's in
`[deps]`. Hitting this cost the first baseline attempt.

The canonical per-commit test gate in this plan is **four
invocations** captured together in the baseline log:

```bash
julia --project=. test/runtests.jl       # runs 20 test files
julia --project=test test/test_aqua.jl
julia --project=test test/test_jet.jl
julia --project=test test/test_readme_current.jl
```

Each exit code is recorded; all four must match the baseline.

## §4.1 Existing-infrastructure survey — key findings

Full log: `artifacts/plan40/existing_infra_survey.txt` (140 lines).
Highlights that shape the commit sequence:

- **IC builders** — the split flagged in the plan is confirmed:
  `scripts/run_transport_binary.jl:{570, 593, 622, 653}` for LL/RG
  (returns VMR, file-based dispatch) vs
  `src/Models/CSPhysicsRecipe.jl:314` `build_cs_tracer_panels` for CS
  (returns halo-padded tracer mass, flat 411 ppm hardcoded at
  `CSPhysicsRecipe.jl:34` const `_CATRINE_BACKGROUND = 4.11e-4`).
- **File-based IC helpers** to hoist into
  `src/Models/InitialConditionIO.jl` (Commit 1):
  - `_sample_bilinear_profile!` at `run_transport_binary.jl:170`
  - `_sample_bilinear_scalar` at `:186`
  - `_load_file_initial_condition_source` at `:353`
  - `_interpolate_log_pressure_profile!` at `:466`
- **Surface flux builders** (LL/RG, to be joined by a new CS method in
  Commit 1):
  - `_load_file_surface_flux_field` at `run_transport_binary.jl:237`
  - `build_surface_flux_source(::AtmosGrid{<:LatLonMesh}, …)` at `:758`
  - `build_surface_flux_source(::AtmosGrid{<:ReducedGaussianMesh}, …)`
    at `:791`
  - `build_surface_flux_sources(…)` at `:822`
- **SurfaceFluxSource contract** confirmed at
  `src/Operators/SurfaceFlux/sources.jl:12`:
  > "The **units are kg/s per cell** — already area-integrated. The
  > surface … A per-area (kg/m²/s) variant that multiplies …"

  The GPT reviewer's Finding 3 was correct — Plan 40 Commit 1 must
  multiply the regridded flux by CS panel cell area after
  `regrid_2d_to_cs_panels!` (which only interpolates).
- **CS helpers to reuse** (from `src/Preprocessing/cs_transport_helpers.jl`):
  - `regrid_3d_to_cs_panels!` at `:150`
  - `regrid_2d_to_cs_panels!` at `:166`
  - `unpack_flat_to_panels_3d!` at `:88`
  - `reconstruct_cs_fluxes!` at `:249`
  - `rotate_winds_to_panel_local!` at `:360`
- **Existing inspector stack** (Commit 5 extends, does not duplicate):
  - `scripts/diagnostics/inspect_transport_binary.jl` (exists)
  - `src/MetDrivers/TransportBinary.jl:133` `Base.show` on header
  - `src/MetDrivers/TransportBinary.jl:154` `Base.show` on reader
  - `src/MetDrivers/TransportBinaryDriver.jl:411` `Base.show` on driver
    (plan 40 said `:404`; actual line is `:411` — off by 7; **no
    design change**, only pointer updated in this NOTES for accuracy).
  - `src/MetDrivers/CubedSphereTransportDriver.jl:48` `Base.show` on
    CS driver (plan 40 said `:45`; actual line is `:48` — off by 3;
    **no design change**).
- **Existing folder/date parsing** is effectively greenfield —
  `src/MetDrivers/AbstractMetDriver.jl:58` defines only a stub
  `start_date(::AbstractMetDriver) = Date(2000, 1, 1)`. No `readdir` /
  `walkdir` scanning exists in runtime code. Plan 40 Commit 4 lands
  on a clean slate.
- **Runtime recipe dispatch is already grid-parametric** —
  `src/Models/CSPhysicsRecipe.jl:44-47` dispatches
  `_runtime_recipe_style` on `AtmosGrid{<:LatLonMesh}`,
  `AtmosGrid{<:ReducedGaussianMesh}`, `AtmosGrid{<:CubedSphereMesh}`,
  and on the driver itself at `:47`. Commit 6's `run_driven_simulation`
  extends this exact pattern; no new dispatch machinery needed.
- **Conservative regridder entry** for Commits 1, 3, 7:
  - `src/Regridding/weights_io.jl:78` `build_regridder(src, dst; …)`
  - `src/Regridding/weights_io.jl:367, 371` `apply_regridder!`
- **Catrine IC file exists** at
  `~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc`
  (82 MB; also `startSF6_202112010000.nc`). CS Commit 1 file-based
  IC can be validated against this file.
- **Dead-reference sweep targets** for Commit 3 — `regrid_latlon_to_cs_binary_v2.jl`
  appears in 9 places:
  - `config/runs/completed_experiments/era5_cs_c90_v2_catrine_2day_f64.toml:8`
  - `config/runs/catrine_c48_10d/advonly.toml:10`
  - `config/runs/catrine_c48_10d/advdiffconv.toml:9`
  - `config/runs/catrine_c48_10d/README.md:27, 42`
  - `docs/reference/CONSERVATIVE_REGRIDDING.md:186, 206`
  - `docs/reference/PREPROCESSING_GUIDE.md:14, 68`

  The two `docs/reference/` references are archival-toned; per plan 40
  Commit 3 they get a pointer-update (not full rewrite). The broader
  PREPROCESSING_GUIDE rewrite is plan 41.

## Deviations from plan doc §4.4

None yet (Commit 0 only). Tracker will be populated per commit as
reality compresses / splits / reorders the plan's draft commit
sequence.

## Correctness rules pinned (read before Commit 1)

### GPU runs must be verified, not declared

`[architecture].use_gpu = true` is a request; silent CPU fallback is
the failure mode. Plan 40 Commit 6's `run_driven_simulation` asserts
device residency before the loop via
`parent(state.air_mass) isa CuArray` (or the per-panel variant for
CS), prints a `[gpu verified] backend=… device=…` line, and aborts
with a precise error otherwise. Commit 7's benchmark harness greps
each run's stdout for that line and writes a `gpu_verified` column
into the output CSV. Any GPU-declared run whose log lacks the line
is rejected from the validation set. Captured as feedback memory
`feedback_verify_gpu_runs_on_gpu` on 2026-04-24.

### Dry-VMR input, basis-aware packing

The Catrine IC NetCDFs (and every IC file format this plan supports)
store **dry VMR**. Converting dry VMR to tracer mass must respect the
binary's `mass_basis`:

- **DryBasis** (the default per invariant 14, and every config in
  `catrine_c48_10d/`): `air_mass == m_dry` in the binary. Packing is
  simply `rm = vmr_dry .* air_mass`.
- **MoistBasis** (legacy GCHP-style `gchp=true` runs, and some C180
  GEOS-IT configs in `config/runs/`): `air_mass == m_moist` in the
  binary. `m_dry = m_moist × (1 − qv)` per invariant 9. Packing must
  be `rm = vmr_dry .* air_mass .* (1 .- qv)`; omitting the `(1-qv)`
  factor on moist binaries would over-estimate `rm` by the humidity
  factor (~1–4% in the lower troposphere).

Plan 40 Commit 1's `pack_initial_tracer_mass` signature therefore
takes `mass_basis::AbstractMassBasis` + optional `qv`, dispatches on
basis, and errors on MoistBasis when `qv` is missing. The moist path
is exercised rarely (the dry-basis contract is the default and
invariant 14 pushes all new work to it) but the correctness rule is
load-bearing — flagged by the user 2026-04-24 to ensure it doesn't
silently regress when someone reuses the packer on a moist binary
months later.

## Decisions beyond the plan

*(Populated during execution.)*

## Surprises

*(Populated during execution.)*

## Interface validation findings

*(Populated during execution.)*

## Template usefulness for plans N+1

*(Populated during execution.)*
