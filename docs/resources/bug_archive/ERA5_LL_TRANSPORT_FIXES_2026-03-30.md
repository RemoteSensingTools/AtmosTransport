# ERA5 Lat-Lon Transport: Bug Fixes & Investigation Report

**Date**: 2026-03-30 to 2026-04-01
**Goal**: Fix grid-scale noise and filament artifacts in ERA5 spectral lat-lon transport, compare with GeosChem C180

---

## Problem Statement

Running the ERA5 spectral 0.5° lat-lon transport (720×361, 137 levels merged to 68) produced:
- Grid-scale "salt and pepper" noise that grew rapidly (R vs GeosChem: 0.96 at 1h → 0.10 at 48h)
- ±45° diagonal filaments over SH ocean
- XCO2 too low at high latitudes
- Zonal variability 20× larger than GeosChem at 50°S after just 6 hours

---

## Bugs Found & Fixed (6 total)

### 1. CRITICAL: cos(φ) missing in mass flux computation

**File**: `scripts/preprocessing/preprocess_spectral_massflux.jl` lines 510-533

`vod2uv!` (spectral VO/D → U/V) returns `U = u·cos(φ)`, `V = v·cos(φ)` (ECMWF convention).
Our `compute_mass_fluxes!` used these directly without correcting for cos(φ):

- **am (zonal flux)**: was `U × dp/g × R × Δφ × dt` → should be `U/cos(φ) × dp/g × R × Δφ × dt`
- **bm (meridional flux)**: was `V × dp/g × R × Δλ × cos(φ) × dt` → should be `V × dp/g × R × Δλ × dt` (V already has cos)

**Impact**: Zonal transport cos(φ)× too strong, meridional cos²(φ)× too strong. At 50°: 64% correct zonal, 41% correct meridional. Creates latitude-dependent shear → ±45° filaments.

**TM5 reference**: `tmm.F90` line 2137-2154 — `IntLat('(da+exp*db)/cos', ...)` explicitly divides by cos(φ).

**Fix applied**: Added `/cos_lat` to am formula, removed `cos_face` from bm formula.

### 2. CRITICAL: flux/n_sub + m-evolve (Strang split consistency)

**File**: `src/Models/physics_phases.jl` — `advection_phase!` for LatitudeLongitudeGrid

The model applied the **full** half-timestep mass flux 24 times (for 6-hourly data), resetting `m` to `m_ref` between sub-steps. This meant at sub-step 10, slopes were computed from `c = rm_evolved / m_original` — an increasingly wrong concentration.

**Diagnosis**: MCP test showed identical noise with vs without Z-subcycling. But dividing fluxes by n_sub and letting m evolve (no reset) eliminated ALL noise growth: std stayed exactly 9.14 ppm after 24 splits vs 18.17 with the old approach.

**Fix applied**: Divide `gpu.am/bm/cm` by `n_sub` before the loop, move `copyto!(gpu.m_dev, gpu.m_ref)` outside the loop, restore fluxes after. m evolves naturally through sub-steps.

### 3. Reduced grid distribute-back (air mass vs tracer fraction)

**File**: `src/Advection/mass_flux_advection.jl` lines 235-239

After reduced-grid X-advection, changes were distributed back to fine cells proportional to `rm[i]/rm_cluster` (tracer mass fraction). TM5 uses `m_uni[i]/mass` (air mass fraction).

**Impact**: Tracer noise at high latitudes corrupted the distribution; uniform concentration didn't stay uniform through the reduced grid.

**TM5 reference**: `redgridZoom.F90` line 1002 — `rm(is:ie) = m_uni(is:ie)/mass * rmm`

**Fix applied**: Changed to `rm_new[i] = (rm_ic + delta_rm) * (m[i]/m_ic)`. Also fixed `expand_row_mass!` in `reduced_grid.jl`.

### 4. TM5-style reduced grid band definitions

**File**: `src/Grids/reduced_grid.jl` — new function `compute_reduced_grid_tm5`

Auto-computed cluster sizes were arbitrary divisors of Nx without hierarchical structure. TM5 uses discrete bands from config with strict constraint: adjacent bands must have compatible sizes.

**Fix applied**: Added `compute_reduced_grid_tm5()` with TM5's band definitions:
- 720×361 (0.5°): `720 720 360 360 180 180 60 60 30 10` per hemisphere (10 bands)
- Wired into `latitude_longitude_grid.jl` constructor

### 5. Double cm residual correction

**Files**: `convert_merged_massflux_to_binary.jl` line 372 + `preprocessed_latlon_driver.jl` line 258

The vertical mass flux residual correction (`_correct_cm_residual!`) was applied twice:
1. During binary conversion (preprocessor)
2. Again at runtime load

**Impact**: Breaks per-level continuity. Second correction on already-corrected data amplifies numerical noise in cm.

**Fix applied**: Commented out runtime correction (line 258) — preprocessing correction is sufficient.

### 6. Z-subcycling investigation (NOT the cause)

Investigated whether Z-subcycling (dividing cm by n_sub, iterating) amplified noise vs TM5's no-subcycling approach. MCP testing showed **identical results** with and without Z-subcycling. The 1D slope evolution test also showed no difference. **This was NOT a bug.**

---

## Infrastructure Built

### v2 Binary Format for Pre-Merged Data

**File**: `scripts/preprocessing/convert_merged_massflux_to_binary.jl`

Reads native 137-level NetCDF, merges to 68 levels (min dp=1000 Pa), writes self-describing binary:
- 16384-byte JSON header with A/B coefficients, merge_map, feature flags
- Supports optional QV, CMFMC, surface fields (pblh, t2m, ustar, hflux)
- Auto-detected by model: no manual `size` needed in TOML config

### Extended Binary Reader

**File**: `src/IO/binary_readers.jl`

`MassFluxBinaryReader` extended to auto-detect v1/v2, read optional fields via mmap.

### Auto-Detection in Configuration

**File**: `src/IO/configuration.jl`

After building met driver, checks for v2 binary A/B coefficients and rebuilds the grid with correct Nx, Ny, Nz automatically.

---

## Known Remaining Issues

### A. Lat grid mismatch between preprocessor and runtime

- Preprocessor: `dlat = 180/(Nlat-1)`, centers at ±90° exactly
- Runtime: `dlat = 180/Ny`, face-defined centers at ±89.75°
- ~0.25° offset at poles, zero at equator
- **Fix needed**: Align preprocessor to runtime convention

### B. vod2uv top-mode truncation

- We include (T,T) spectral mode; TM5 zeros it
- Minor impact (single highest wavenumber)

### C. Point-wind staggering vs TM5 spectral edge integrals

- Our approach: transform to gridpoint, average to faces, multiply by dp
- TM5: integrate U×dp product along cell edges in spectral space (IntLat/IntLon)
- TM5 approach achieves ~90× better mass conservation
- **Would require significant refactor to implement**

### D. Single-file binary architecture

The merged binary converter writes all windows into one file. For hourly data (31 days = 744 windows), this creates a 212 GB file that causes NFS mmap performance issues.

**Solution needed**: Daily binary files (like the GEOS CS path), ~1.4 GB each for hourly merged data. Requires changes to:
- `convert_merged_massflux_to_binary.jl` — write per-day files
- `find_massflux_shards()` in `file_discovery.jl` — discover daily files
- `PreprocessedLatLonMetDriver` — handle multiple files

### E. Missing physics fields in hourly preprocessed data

The hourly spectral preprocessor only writes m, am, bm, cm, ps. For full physics runs, also need:
- QV (specific humidity) — for dry-mass output conversion
- CMFMC (convective mass flux) — for Tiedtke convection
- Surface fields (PBLH, T2M, USTAR, HFLUX) — for PBL diffusion

These come from separate ERA5 downloads and need to be merged into the daily binaries.

---

## Data Locations

| Dataset | Path | Status |
|---------|------|--------|
| ERA5 hourly spectral GRIB | `~/data/AtmosTransport/met/era5/spectral_hourly/` | Dec 1-10+ downloaded (5.1 GB/day) |
| ERA5 hourly preprocessed NetCDF | `~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine_hourly/` | Full Dec 2021 (343 GB, 744 windows) WITH cos(φ) fix |
| ERA5 hourly merged binary | `.../preprocessed_spectral_catrine_hourly/5day/` | 5-day subset (34 GB), header patched to Nt=120 |
| ERA5 6-hourly spectral GRIB | `~/data/AtmosTransport/met/era5/spectral/` | 6-hourly, Dec 2021 (31 days) |
| ERA5 6-hourly merged binary | `.../preprocessed_spectral_catrine/massflux_..._merged1000Pa_float32.bin` | OLD (pre-cos(φ) fix, needs reprocessing) |
| GeosChem reference | `~/data/AtmosTransport/catrine-geoschem-runs/` | Full Dec 2021, 3-hourly C180 |

---

## Results Before/After Fixes

### Mass Conservation (5-day run, F32)

| Config | CO2 Δ% |
|--------|--------|
| Before fixes (6-hourly, m-reset) | 5.5e-5% |
| After flux/n_sub fix (6-hourly) | 2.1e-5% |
| Hourly data | **1.3e-8%** (machine precision) |

### Correlation vs GeosChem at t=1h

| Level | Before all fixes | After fixes (6-hourly) | Hourly (cos(φ) not yet applied) |
|-------|-----------------|----------------------|------|
| 950 hPa | R=0.35 | R=0.80 | R=0.90 |
| 750 hPa | R=0.19 | R=0.86 | R=0.96 |

### Remaining Noise

With all fixes EXCEPT cos(φ) (hourly data), zonal std at 50°S was still 1.2 ppm after 1h vs GeosChem's 0.26 ppm. The cos(φ) fix should dramatically reduce this since the flux magnitudes and directions will now be correct.

**The cos(φ) fix has been applied to the preprocessor but the hourly data has NOT been rerun through the model yet** (was blocked by the single-file NFS performance issue). The hourly preprocessed NetCDF (343 GB) already has the cos(φ) fix applied. It needs to be converted to daily binary files and run.

---

## Uncommitted Changes in Working Tree

All fixes are in the working tree (branch: main, dirty). Key modified files:

```
M scripts/preprocessing/preprocess_spectral_massflux.jl  ← cos(φ) fix + threading
M src/Advection/mass_flux_advection.jl                   ← reduced grid distribute-back
M src/Grids/reduced_grid.jl                              ← TM5 bands + expand_row_mass!
M src/Grids/latitude_longitude_grid.jl                   ← use compute_reduced_grid_tm5
M src/IO/binary_readers.jl                               ← v2 format support
M src/IO/preprocessed_latlon_driver.jl                   ← v2 load + remove double cm correction
M src/IO/configuration.jl                                ← auto-detect merged grid
M src/Models/physics_phases.jl                           ← flux/n_sub + m-evolve
?? scripts/preprocessing/convert_merged_massflux_to_binary.jl  ← NEW
?? scripts/downloads/download_era5_spectral_hourly.py          ← NEW
?? config/preprocessing/catrine_spectral_dec2021_hourly.toml   ← NEW
?? config/runs/catrine_era5_hourly_merged_dec2021.toml         ← NEW
?? config/runs/catrine_era5_merged_dec2021_pbl.toml            ← NEW
```
