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

### Commit 1 split into 1a / 1b / 1c (time budget)

Plan 40 drafted Commit 1 as a single unit (LL/RG hoist + CS file IC
+ CS surface flux + tests). Working against a 20-minute wall-clock
budget on 2026-04-24, the dependency web of the LL/RG helpers
(`_horizontal_interp_weights`, `_ic_find_coord`,
`FileInitialConditionSource` struct, kind/config resolvers,
`SECONDS_PER_MONTH`, `nomissing`) makes a clean bit-exact hoist
larger than can land in that window. Split:

- **1a (shipped, `1450204`)** — pure-add module scaffold.
- **1b (shipped this commit)** — LL/RG IC hoist: 14 private helpers
  (`_horizontal_interp_weights`, `_bilinear_bracket`,
  `_periodic_bilinear_bracket`, `_sample_bilinear_profile!`,
  `_sample_bilinear_scalar`, `_ic_find_coord`, `_resolve_file_init`,
  `_load_file_initial_condition_source`,
  `_interpolate_log_pressure_profile!`, `_copy_profile!`,
  `wrapped_longitude_distance`, `wrapped_longitude_360`,
  `_init_kind`, `_is_file_init_kind`), the `FileInitialConditionSource`
  struct, and all 4 `build_initial_mixing_ratio` methods moved
  verbatim from `scripts/run_transport_binary.jl:{29-196,198-210,353-420,466-529,570-684}`.
  New `pack_initial_tracer_mass(grid, air_mass, vmr_dry;
  mass_basis::AbstractMassBasis, qv=nothing)` — 4 methods
  (LL/RG × DryBasis/MoistBasis); MoistBasis errors loudly without
  `qv` per feedback memory. Script now imports the helpers via
  `using .AtmosTransport.Models.InitialConditionIO: …`; public
  names re-exported through `Models` → `AtmosTransport`. New test
  file `test/test_initial_condition_io.jl` (17 tests, all pass).
- **1c (shipped this commit)** — CS `build_initial_mixing_ratio` for
  `uniform | file | catrine_co2 | netcdf | file_field` (CS file
  path: `_load_file_initial_condition_source` → LL source mesh via
  `_build_source_latlon_mesh` → conservative regrid via
  `build_regridder` / `apply_regridder!` / `unpack_flat_to_panels_3d!`
  → per-column `_interpolate_log_pressure_profile!` with a regridded
  `ps_src` panel). CS `pack_initial_tracer_mass` (DryBasis +
  MoistBasis, halo-padded output with halo zeroed). Also: reordered
  module loads in `src/AtmosTransport.jl` so Regridding +
  Preprocessing load **before** Models (required by
  InitialConditionIO's `using ..Regridding`/`using ..Preprocessing`;
  verified no back-references via grep).
- **1d (shipped this commit)** — Surface-flux builders now own
  themselves. Hoisted to `src/Models/InitialConditionIO.jl`:
  `FileSurfaceFluxField`, `SECONDS_PER_MONTH`, `_surface_flux_kind`,
  `_resolve_surface_flux_file`, `_normalize_units_string`,
  `_load_file_surface_flux_field`,
  `_renormalize_surface_flux_rate!`, `_REGRID_CACHE_DIR`,
  `_conservative_surface_flux_rate`, `_regridding_method`,
  `build_surface_flux_source` (LL + RG), and
  `build_surface_flux_sources`. `_build_emission_source_mesh` is
  dropped in favour of the shared `_build_source_latlon_mesh`. New:
  `build_surface_flux_source(grid::AtmosGrid{<:CubedSphereMesh}, …)`
  — conservative LL→CS regrid (which already integrates by
  `dst_areas`, yielding kg/s per cell) + panel unpack to
  `NTuple{6, Matrix{FT}}`. Matches the kg/s-per-cell contract at
  `src/Operators/SurfaceFlux/sources.jl:12` and the CS NTuple{6}
  expectation at `sources.jl:88-101`. CS path enforces
  `regridding = "conservative"` (warns if caller asks for
  bilinear). Tests: global mass conservation
  (rel_err ~ 1.2e-16 under CR.jl), shape contract
  `NTuple{6, Matrix{FT}}` with `(Nc, Nc)` per panel, `kind=none`
  shortcut across LL/RG/CS, empty-specs dispatcher returns `()`.
  68 tests in `test_initial_condition_io.jl` (from 54).

All three remain individually revertable. The plan doc still
captures the full Commit 1 design; NOTES is the source of truth for
what actually shipped per session.

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
