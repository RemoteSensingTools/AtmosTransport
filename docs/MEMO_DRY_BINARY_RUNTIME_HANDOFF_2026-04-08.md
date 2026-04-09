# Memo: Dry-Binary Runtime Handoff

Date: 2026-04-08

This memo summarizes the current state of the dry-mass refactor, the new
preprocessor path, and the compatibility work that was added so the existing
`src/` runtime can exercise the new dry-basis binaries without mixing runtime
layers from `src_v2`.

## Scope Boundary

There are now two distinct layers in play:

1. `src_v2` and `scripts/preprocessing/preprocess_spectral_v4_binary*`
   This is the refactor/preprocessing side. It now produces self-describing
   v4/v5-style binaries with explicit mass-basis metadata, embedded QV, and
   range-based structured-grid coordinate headers.

2. `src/` plus `scripts/run.jl`
   This is still the production runtime/advection path. The latest runtime
   validation was done here deliberately, to avoid mixing `src` and `src_v2`
   transport logic inside one experiment.

The practical rule is:

- `src_v2` is currently the producer of the new binaries and design direction.
- `src/` is still the consumer used for real sub-stepped transport runs.

## Main Changes Landed

### 1. Oceananigans-style grid cleanup in `src_v2`

The `src_v2` grid layer was pushed toward a more Julian/Oceananigans-like
style:

- validated constructors
- clearer public mesh APIs
- `summary` / `show` methods
- reduced-Gaussian geometry as a real type instead of a stub
- cleaned structured lat-lon and cubed-sphere metadata

Important files:

- `src_v2/Grids/LatLonMesh.jl`
- `src_v2/Grids/ReducedGaussianMesh.jl`
- `src_v2/Grids/CubedSphereMesh.jl`
- `src_v2/MetDrivers/ERA5/NativeGRIBGeometry.jl`
- `docs/GRID_CONVENTIONS.md`

### 2. Preprocessor split and refactor

The old single large script was split into a thin entrypoint plus ordered
support files under:

- `scripts/preprocessing/preprocess_spectral_v4_binary/`

Key changes:

- small routines instead of one large orchestration function
- more multiple dispatch
- documented workspaces and pipeline stages
- humidity-aware dry-basis generation
- self-describing binary headers

### 3. Dry-basis binary generation with real QV

The preprocessor now supports a true dry-basis output mode using real hourly
ERA5 QV instead of the older global constant-QV approximation for the dry
surface-pressure mass pinning.

Important semantic choices:

- `window i` uses same-day thermo/QV valid time `i` (`00..23 UTC`)
- next-day delta uses next-day `00 UTC`
- if real QV is present and `mass_basis = :dry`, the dry-mass pinning uses the
  native hourly QV field rather than the climatological scalar

This is an improvement over the TM5-style fixed-global-QV dry-pressure fix, but
it is still a globally uniform `ps` offset, not yet a spatially varying dry
pressure correction.

### 4. Binary header cleanup for structured lat-lon grids

Uniform lat-lon binaries now store:

- `lon_center_start_deg`
- `lon_center_step_deg`
- `lat_center_start_deg`
- `lat_center_step_deg`

instead of writing full coordinate arrays into the JSON header.

This fixed the header overflow issue and is also the more Julian structured-grid
representation.

### 5. Legacy runtime compatibility for dry-basis binaries

The existing `src/` runtime was patched so it can consume the new dry-basis
binary cleanly.

Important runtime compatibility changes:

- `src/IO/binary_readers.jl`
  - now reconstructs structured lat-lon axes from range metadata
  - now reads `mass_basis` from the binary header
- `src/IO/abstract_met_driver.jl`
  - adds a generic `mass_basis(driver)` interface
- `src/IO/preprocessed_latlon_driver.jl`
  - propagates `mass_basis` from the binary to the met driver
- `src/Models/physics_phases.jl`
  - LL dry-mass helpers now distinguish moist-basis from dry-basis binaries
  - dry-basis binaries are no longer humidity-corrected a second time
- `src/Models/run_loop.jl`
  - passes the driver into the LL dry-mass/output helpers

This was the critical bridge needed to run the new binary through the real
sub-stepped legacy runtime.

### 6. Visualization run path

New run config:

- `config/runs/era5_f64_startCO2_viz_dry_qv.toml`

This mirrors the older 24-hour CO2 visualization run, but points to the new
dry-basis + embedded-QV binary and keeps the experiment on the existing
`scripts/run.jl` / `src/` stack.

Plot script update:

- `scripts/visualization/plot_era5_24h_co2.py`

It now accepts an optional third argument for selected snapshot hours, for
example:

```bash
python3 scripts/visualization/plot_era5_24h_co2.py \
  /tmp/era5_f64_startCO2_viz_dry_qv.nc \
  /tmp/era5_co2_viz_dry_qv \
  0,5,9,13,17,21,24
```

## Validation Done

### Preprocessor-side

- moist-basis v4 smoke binary produced and compared against the older v4 file
- dry-basis + QV binary produced for `2021-12-01`
- embedded QV payload verified against the source thermo file
- direct `src_v2` dry-binary advection smoke test passed
- `test_v2/test_real_era5_direct_dry_binary.jl` added for the direct-v2 path

### Legacy runtime-side

A real one-window GPU run was executed through:

- `scripts/run.jl`
- `config/runs/era5_f64_startCO2_viz_dry_qv.toml`

Key result:

- the run used proper runtime sub-stepping
- the global CFL pilot triggered and refined from `4` to `8` substeps
- the one-window run completed successfully
- CO2 mass change was small and finite (`~6.5e-05%`)

This is the main proof that the new dry-basis binary can now be exercised
through the real legacy advection system instead of only through an isolated
single-step smoke test.

## Background Run State

At the time of writing, a full 24-hour visualization run was launched in the
background:

- config:
  `config/runs/era5_f64_startCO2_viz_dry_qv.toml`
- log:
  `/tmp/era5_f64_startCO2_viz_dry_qv.log`
- output:
  `/tmp/era5_f64_startCO2_viz_dry_qv.nc`
- plots:
  `/tmp/era5_co2_viz_dry_qv`

The plotting step is chained after the NetCDF run and is configured to emit
snapshots at hours:

- `0, 5, 9, 13, 17, 21, 24`

## What Still Needs Attention

### 1. Avoid long-term split-brain between `src` and `src_v2`

Right now the branch is intentionally split:

- `src_v2` owns the new preprocessing and design direction
- `src` still owns the trusted runtime loop

That was the safest choice for validation, but it should not become permanent.
The next design step is to make the handoff boundary explicit and eventually
replace the old runtime path with the refactored one, rather than continuing to
patch both in parallel.

### 2. Reduced-Gaussian native path is not done yet

The grid objects and ERA5 native geometry groundwork are in place, but the full
native reduced-Gaussian preprocessor path is not finished yet. The important
unfinished part is the real face-flux construction on variable-ring topology.

### 3. Dry-pressure correction is still global, not spatial

The current fix uses actual hourly QV to compute the correct global dry-mass
target, but it still applies one globally uniform `ps` offset. If later work
shows a structured residual, the next likely experiment should be a constrained
low-order spatial correction, not a noisy pointwise correction.

### 4. Some validation scripts hit Julia teardown segfaults

A few real-data test scripts printed correct results and then segfaulted during
process shutdown. The failures looked like teardown/library cleanup issues, not
assertion failures, but they should still be tracked.

## Recommended Next Steps For A Planning Agent

1. Treat `src_v2` as the design/reference layer and `src` as the current
   execution layer.
2. Decide what the long-term runtime boundary should be so we stop duplicating
   logic across both stacks.
3. Finish the native ERA5 reduced-Gaussian path:
   - native GRIB geometry
   - native QV ingest
   - spectral-to-native transform
   - conservative face connectivity and flux construction
4. Add a clean runtime notion of transport basis everywhere instead of patching
   it piecemeal.
5. Turn the current dry-binary legacy runtime validation into a repeatable
   regression test or small reproducible run target.

## Operational Reminder

The local Cursor MCP config likely contains a visible auth token. Rotate it and
replace the hardcoded token in `.cursor/mcp.json` with an environment variable
such as `KAIMON_TOKEN`.
