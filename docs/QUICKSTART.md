# Quick Start Guide

End-to-end instructions for running AtmosTransport: downloading
meteorological data, preprocessing, configuring, and running simulations.

For a deep dive into met data formats, vertical coordinates, and TM5
comparison, see [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md).

## 5-Minute Quickstart

Run a 12-hour CO₂ transport simulation on GEOS-IT C180 cubed-sphere (~0.5°)
with EDGAR v8.0 anthropogenic emissions — no GPU or preprocessing required.

```bash
# 1. Install Julia dependencies (first time only)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Download quickstart data (~1.4 GB) from GitHub Releases:
#    https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-v1
#    Then extract:
mkdir -p ~/data/AtmosTransport
tar -xzf quickstart_met_data.tar.gz -C ~/data/AtmosTransport/

# 3. Run the quickstart
julia --project=. scripts/quickstart.jl
```

The tarball contains preprocessed GEOS-IT C180 cubed-sphere met data (12 hours)
and EDGAR v8.0 CO₂ emissions pre-regridded to the C180 grid. The quickstart
runs on CPU and produces a lat-lon regridded visualization.
Output: `quickstart_output.nc` + snapshot PNG.

To run with full control over parameters:
```bash
julia --project=. scripts/run.jl config/runs/quickstart.toml
```

## Prerequisites

- **Julia 1.10+** (tested with 1.12+ via [juliaup](https://github.com/JuliaLang/juliaup))
- **Project dependencies**: install once with
  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```
- **GPU** (optional): NVIDIA GPU with CUDA 12+ drivers, or Apple Silicon
  with Metal.jl. The model loads the appropriate GPU backend automatically
  when the TOML config sets `use_gpu = true`.

## Running a Simulation

All simulations use a single universal runner with a TOML configuration file:

```bash
julia --project=. scripts/run.jl <config.toml>
```

For double-buffered I/O overlap (disk reads in parallel with GPU compute),
start Julia with multiple threads:

```bash
julia --threads=2 --project=. scripts/run.jl <config.toml>
```

### Available Configurations

| Config | Grid | Met Source | GPU | Description |
|--------|------|-----------|-----|-------------|
| **`quickstart.toml`** | **C180 cubed-sphere** | **GEOS-IT C180** | **No** | **12-hour demo, EDGAR CO₂** |
| `geosfp_c720_june2024_fixed.toml` | C720 cubed-sphere | GEOS-FP NetCDF | Yes | 30-day June 2024, mass_flux_dt=450, level merging |
| `geosit_c180_june2023.toml` | C180 cubed-sphere | GEOS-IT NetCDF | Yes | 30-day June 2023, slopes advection |
| `geosit_c180_june2023_ppm.toml` | C180 cubed-sphere | GEOS-IT NetCDF | Yes | 30-day June 2023, PPM-7 advection |
| `era5_spectral_june2023.toml` | 720×361 lat-lon | ERA5 spectral | Yes | 30-day June 2023, PPM-7, Tiedtke, PBL |
| `geosfp_cs_edgar.toml` | C720 cubed-sphere | GEOS-FP preprocessed | Yes | Preprocessed binary, 5 days |
| `catrine_era5.toml` | 720×361 lat-lon | ERA5 spectral | Yes | CATRINE D7.1: CO2, fCO2, SF6, 222Rn |
| `catrine_geosit_c180.toml` | C180 cubed-sphere | GEOS-IT C180 | Yes | CATRINE D7.1 on CS: 4 tracers, RAS conv, nonlocal PBL |
| `catrine_geosit_c180_linrood.toml` | C180 cubed-sphere | GEOS-IT C180 | Yes | CATRINE D7.1 + Lin-Rood cross-term advection |

All configs are in `config/runs/`.

### Examples

```bash
# GEOS-FP C720 cubed-sphere on GPU (30-day, NetCDF mode)
julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_c720_june2024_fixed.toml

# GEOS-IT C180 with PPM advection
julia --threads=2 --project=. scripts/run.jl config/runs/geosit_c180_june2023_ppm.toml

# ERA5 spectral on GPU
julia --threads=2 --project=. scripts/run.jl config/runs/era5_spectral_june2023.toml
```

---

## TOML Configuration Reference

The TOML file is the single source of truth for a simulation. All parameters
that were previously set via environment variables are now in the config file.

### `[architecture]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_gpu` | bool | `false` | Enable GPU via CUDA.jl |
| `float_type` | string | `"Float64"` | `"Float32"` for GPU, `"Float64"` for CPU |

### `[grid]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"latlon"` | `"latlon"` or `"cubed_sphere"` |
| `Nc` | int | `720` | Panel edge size for cubed-sphere (C720 = ~12.5 km) |
| `met_source` | string | `"era5"` | Selects vertical coefficients: `"era5"` (L137) or `"geosfp"` (L72) |
| `size` | [int,int,int] | `[360,181,72]` | Grid dimensions for lat-lon `[Nx, Ny, Nz]` |
| `longitude` | [float,float] | `[0.0, 360.0]` | Longitude range for lat-lon |
| `latitude` | [float,float] | `[-90.0, 90.0]` | Latitude range for lat-lon |
| `merge_levels_above_Pa` | float | (none) | Merge thin upper levels above this pressure [Pa]. Reduces CFL in stratosphere. Only supported in NetCDF mode |

### `[met_data]`

Common keys for all drivers:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `driver` | string | (required) | `"era5"`, `"preprocessed_latlon"`, or `"geosfp_cs"` |
| `dt` | float | `900` | Advection sub-step [seconds] |
| `met_interval` | float | `3600` | Met update interval [seconds] |
| `mass_flux_dt` | float | `450` | **Critical for GEOS-FP/IT.** Dynamics timestep over which mass fluxes are accumulated [seconds]. GEOS products use ~450s, not the 1-hour output interval. Wrong value makes transport 8x too slow. See [CAVEATS.md](CAVEATS.md) |

**ERA5 driver** (`driver = "era5"`):

| Key | Type | Description |
|-----|------|-------------|
| `datadirs` | [string, ...] | List of directories containing ERA5 NetCDF files |
| `level_top` | int | Topmost model level (e.g., 50) |
| `level_bot` | int | Bottommost model level (e.g., 137 = surface) |

**Preprocessed lat-lon driver** (`driver = "preprocessed_latlon"`):

| Key | Type | Description |
|-----|------|-------------|
| `directory` | string | Directory containing monthly `.bin` mass flux shards |
| `file` | string | Single binary file (alternative to `directory`) |

**GEOS-FP/IT cubed-sphere driver** (`driver = "geosfp_cs"`):

| Key | Type | Description |
|-----|------|-------------|
| `netcdf_dir` | string | Directory with raw GEOS-FP/IT `.nc4` files (recommended) |
| `preprocessed_dir` | string | Directory with per-day `.bin` files (faster I/O, requires preprocessing) |
| `surface_data_dir` | string | Directory with raw 0.25° lat-lon surface NetCDF (A1/A3mstE) |
| `surface_data_bin_dir` | string | Directory with binary surface fields (PBLH, USTAR, HFLUX, T2M) — preferred for performance |
| `surface_data_ll_dir` | string | Directory with lat-lon surface data for on-the-fly CS regridding |
| `coord_file` | string | Path to CS gridspec NetCDF with cell corners and areas (for emission regridding) |
| `product` | string | `"geosit_c180"` or `"geosfp_c720"` — selects vertical level ordering and file naming |
| `start_date` | string | Start date `"YYYY-MM-DD"` |
| `end_date` | string | End date `"YYYY-MM-DD"` |
| `Hp` | int | Halo width (default: 3) |

### `[advection]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scheme` | string | `"slopes"` | `"slopes"` (Russell-Lerner, 2nd order) or `"ppm"` (Putman & Lin) |
| `ppm_order` | int | `7` | PPM polynomial order: 4, 5, 6, or 7 (only used when `scheme = "ppm"`) |
| `damp_coeff` | float | `0.0` | Divergence damping coefficient (PPM only) |
| `linrood` | bool | `false` | Lin-Rood cross-term splitting (CS grids, PPM only). Averages X-first and Y-first PPM orderings to eliminate directional bias at panel boundaries |
| `mass_fixer` | bool | `true` | Per-cell mass fixer after each advection step |

### `[convection]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | (none) | `"tiedtke"` (Tiedtke mass-flux) or `"ras"` (Relaxed Arakawa-Schubert; requires CMFMC+DTRAIN from A3dyn surface data) |

### `[diffusion]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | (none) | `"boundary_layer"`, `"pbl"`, or `"nonlocal_pbl"` |

Additional keys for `type = "boundary_layer"`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Kz_max` | float | `100.0` | Maximum eddy diffusivity [Pa²/s] |
| `H_scale` | float | `8.0` | E-folding depth in levels from surface |

Additional keys for `type = "pbl"` or `"nonlocal_pbl"`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `beta_h` | float | `15.0` | Businger-Dyer heat parameter |
| `Kz_bg` | float | `0.1` | Background diffusivity above PBL [m²/s] |
| `Kz_min` | float | `0.01` | Minimum diffusivity in PBL [m²/s] |
| `Kz_max` | float | `500.0` | Maximum diffusivity [m²/s] |

Additional keys for `type = "nonlocal_pbl"` only:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fak` | float | `8.5` | Counter-gradient tuning constant (Holtslag-Boville) |
| `sffrac` | float | `0.1` | Surface layer fraction of PBL |

### `[tracers.<name>]`

Define one section per tracer (e.g., `[tracers.co2]`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `emission` | string | `"none"` | `"edgar"`, `"carbontracker"`, `"gfas"`, `"jena"` |
| `year` | int | `2022` | Emission inventory year |
| `edgar_version` | string | `"v8.0"` | EDGAR version (if `emission = "edgar"`) |
| `edgar_file` | string | (auto) | Path to pre-regridded CS binary (optional) |

### `[output]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `filename` | string | | Output NetCDF path (supports `~`) |
| `interval` | float | `3600` | Output interval [seconds] |
| `output_grid` | string | | `"latlon"` to regrid cubed-sphere output |
| `Nlon` | int | `720` | Output grid longitude points |
| `Nlat` | int | `361` | Output grid latitude points |
| `format` | string | `"netcdf"` | `"netcdf"` or `"binary"` (binary auto-converts on finalize) |
| `deflate_level` | int | `0` | NetCDF compression (0 = off, 1–9 = increasing) |

**Output timing:** Output is written at the **end** of each met window, after all
physics (advection, diffusion, convection, sources). The timestamp is the elapsed
simulation time at that point. For evenly-spaced output, `interval` should be a
multiple of `met_interval`.

**Time variable:** The NetCDF time variable uses CF-convention units
`"seconds since {start_date} 00:00:00"`, where `start_date` is auto-detected from
the met data files. Values are seconds elapsed since simulation start, consistent
with the units string.

**dt / met_interval alignment:** `dt` must evenly divide `met_interval` so that an
integer number of advection sub-steps exactly covers each met window. The model will
error if this constraint is violated (e.g. `dt=1000` with `met_interval=3600` fails;
use `dt=900` instead).

### `[output.fields]`

Key-value pairs mapping field names to diagnostic types:

```toml
[output.fields]
co2 = "column_mean"      # column-integrated mass-weighted mean (XCO2)
# co2 = "surface_slice"  # lowest model level
# co2 = "regrid"         # full 3D regrid to lat-lon
```

### `[buffering]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | string | `"single"` | `"single"` or `"double"` |

`"double"` overlaps disk I/O with GPU compute using `Threads.@spawn`.
Requires `julia --threads=2` or more.

---

## Downloading Meteorological Data

### GEOS-FP C720 Cubed-Sphere Mass Fluxes

Native cubed-sphere mass fluxes from the Washington University GEOS-Chem
archive. No authentication required.

**Source**: `http://geoschemdata.wustl.edu/ExtData/GEOS_C720/GEOS_FP_Native/`
**Reference**: Martin et al. (2022), GMD 15, 8731–8748

```bash
# Configure date range and output directory
export GEOSFP_START_DATE=2024-06-01
export GEOSFP_END_DATE=2024-06-05
export GEOSFP_DATA_DIR=~/data/geosfp_cs

# Download hourly files (~65 GB/day, ~2.7 GB per hourly file)
julia --project=. scripts/download_geosfp_cs_massflux.jl
```

The script downloads hourly `tavg_1hr_ctm_c0720_v72` files containing
MFXC, MFYC, DELP (mass fluxes and pressure thickness) on the native C720
gnomonic cubed-sphere grid. Files are organized as:

```
~/data/geosfp_cs/
  20240601/
    GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_0030.V01.nc4
    GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_0130.V01.nc4
    ...  (24 files per day)
  20240602/
    ...
```

**Storage requirements**:

| Period | Files | Size |
|--------|-------|------|
| 1 day | 24 | ~65 GB |
| 1 week | 168 | ~450 GB |
| 1 month | 720 | ~1.9 TB |
| 1 year | 8,760 | ~23 TB |

**Known gaps**: June 16–18, 2024 are missing from the archive.

Features: automatic retry on failure, skip-if-exists, progress tracking.

### ERA5 Model Levels

Model-level ERA5 data (u, v, omega, lnsp) via the ECMWF Climate Data Store
API. **Requires CDS API credentials.**

**CRITICAL**: Download **model-level** data (`levtype: ml`), NOT pressure-level
data. Pressure-level data destroys terrain-following coordinates and mass
conservation. See [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md#critical-model-levels-only)
for details.

#### CDS API Setup

1. Register at https://cds.climate.copernicus.eu/
2. Accept the ERA5 licence terms
3. Install the Python client: `pip install cdsapi`
4. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-uid>:<your-api-key>
   ```

#### Download

```bash
export RESOLUTION=1.0         # degrees (0.25, 0.5, 1.0)
export START_DATE=2024-06-01
export END_DATE=2024-06-30
export LEVEL_TOP=50           # topmost model level
export LEVEL_BOT=137          # surface level
export MAX_PARALLEL=4         # concurrent CDS requests
export PYTHON=/usr/bin/python3

julia --project=. scripts/download_era5_model_levels.jl
```

The script downloads daily files, merges them into monthly NetCDF, and
cleans up dailies. Output:

```
~/data/metDrivers/era5/era5_ml_10deg_20240601_20240630/
  era5_ml_202406.nc    # merged monthly file
```

**Storage requirements** (1-degree, L88):

| Period | Size |
|--------|------|
| 1 day | ~150 MB |
| 1 month | ~4.5 GB |
| 1 year | ~55 GB |

Features: parallel downloads, automatic retry with backoff, monthly merging,
resume on restart.

---

## Preprocessing Data to Binary Format

Preprocessing converts raw NetCDF files to flat binary format optimized for
fast mmap-based loading. This is a one-time step per date range.

### ERA5: Spectral to Mass Fluxes (Recommended)

Converts ERA5 spectral harmonic fields (vorticity, divergence, log surface
pressure) to mass-conserving mass fluxes, following TM5's approach. This is
the recommended ERA5 pipeline — it achieves better mass conservation than
the gridpoint approach.

**Input**: ERA5 spectral GRIB files (VO, D, LNSP)
**Output**: NetCDF mass fluxes (am, bm, cm, m, ps)

```bash
julia --project=. scripts/preprocess_spectral_massflux.jl \
    config/preprocessing/spectral_june2023.toml
```

See `config/preprocessing/spectral_june2023.toml` for configuration options.

### GEOS-FP/IT CS: NetCDF to Binary

Converts native GEOS-FP/IT cubed-sphere NetCDF mass flux files to flat binary
files with haloed DELP and staggered mass flux panels (am, bm). This is
optional — the model can read raw NetCDF directly via `netcdf_dir`, but
preprocessed binaries are ~15x faster for I/O.

**Input**: Raw `.nc4` files from the download step
**Output**: Per-day `.bin` files for mmap-based GPU ingestion

```bash
julia --project=. scripts/preprocess_geosfp_cs.jl \
    config/preprocessing/geosfp_c720_june2024.toml
```

See `config/preprocessing/geosfp_c720_june2024.toml` for configuration
(date range, paths, `mass_flux_dt = 450`).

**Output**:

```
~/data/geosfp_cs/preprocessed/
  geosfp_cs_20240601_float32.bin    # ~61 GB per day
  geosfp_cs_20240602_float32.bin
  ...
```

Each binary file contains:
- **Header** (8192 bytes): JSON metadata (grid size, float type, array sizes)
- **24 hourly windows**, each containing:
  - 6 panels of haloed DELP: `(Nc+2Hp) × (Nc+2Hp) × Nz` per panel
  - 6 panels of staggered am: `(Nc+1) × Nc × Nz` per panel
  - 6 panels of staggered bm: `Nc × (Nc+1) × Nz` per panel

Processing per window: read C-grid NetCDF → convert units (Pa·m² → kg/s) →
stagger to face-centered (am, bm) → pad DELP with halos → write binary.

Features: skip-if-exists (safe to re-run), progress tracking per window.

**Storage**: ~61 GB/day at C720 Float32. Five days requires ~305 GB.

### Emission Regridding: Lat-Lon to Cubed-Sphere

Conservatively regrids any lat-lon emission inventory to cubed-sphere
panels using the TOML-driven preprocessor. Supports fine-to-coarse
(0.1° EDGAR on C180) and coarse-to-fine (1° LMDZ on C180).

```bash
# Regrid a single source
julia --project=. scripts/preprocessing/regrid_emissions.jl \
    config/emissions/edgar_sf6.toml

# Available configs: edgar_sf6, zhang_rn222, lmdz_co2, gridfed_fossil_co2
```

Output: compact binary with JSON header + 6 panels of `Nc × Nc` Float32
values per time step. The model auto-discovers preprocessed binaries by
species name and grid size.

See [EMISSION_REGRIDDING.md](EMISSION_REGRIDDING.md) for the full
tutorial on adding new sources, the conservative regridding algorithm,
and validation results.

### ERA5: Gridpoint Winds to Binary Mass Fluxes (Stopgap)

Pre-computes mass fluxes from ERA5 gridpoint u/v winds and surface pressure.
This is a stopgap — the spectral pipeline above is preferred for better mass
conservation (~0.9% drift/month with gridpoint vs near-zero with spectral).

```bash
export ERA5_DIRS=~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607
export OUTDIR=~/data/metDrivers/era5/preprocessed
export FT_PRECISION=Float32
export LEVEL_TOP=50
export LEVEL_BOT=137

julia --project=. scripts/preprocess_mass_fluxes.jl
```

---

## End-to-End Workflow

### GEOS-FP C720 Cubed-Sphere (GPU)

```bash
# 1. Download raw GEOS-FP mass flux files (one-time, ~65 GB/day)
julia --project=. scripts/download_geosfp_cs_massflux.jl

# 2. (Optional) Preprocess to flat binary for faster I/O (~61 GB/day)
julia --project=. scripts/preprocess_geosfp_cs.jl \
    config/preprocessing/geosfp_c720_june2024.toml

# 3. Run! (NetCDF mode — reads raw .nc4 directly, no preprocessing needed)
julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_c720_june2024_fixed.toml
```

**Important**: The config must set `mass_flux_dt = 450` (GEOS dynamics timestep).
Without this, transport is 8x too slow. See [CAVEATS.md](CAVEATS.md) for details.

### GEOS-IT C180 Cubed-Sphere (GPU)

```bash
# 1. Download GEOS-IT C180 mass flux files (CTM_A1: MFXC, MFYC, DELP)
julia --project=. scripts/downloads/download_geosit_physics.jl

# 2. Download surface fields for PBL diffusion (A1 + A3mstE: PBLH, USTAR, HFLUX, T2M)
#    Same script with surface=true, or use the general GEOS download framework.

# 3. (Optional) Download A3dyn for RAS convection (CMFMC, DTRAIN)
#    Required only if [convection] type = "ras"

# 4. Preprocess met data to binary (one-time, ~15× faster I/O)
julia --project=. scripts/preprocessing/preprocess_geosfp_cs.jl \
    config/preprocessing/geosit_c180_catrine.toml

# 5. (Optional) Regrid emissions to CS grid
julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/edgar_sf6.toml

# 6. Run (slopes advection)
julia --threads=2 --project=. scripts/run.jl config/runs/geosit_c180_june2023_ppm.toml
```

The `coord_file` in the TOML config must point to a CS gridspec NetCDF file
containing cell corners and areas for emission regridding. Generate one with
`scripts/preprocessing/extract_cs_gridspec.jl` or use the bundled
`data/grids/cs_c180_gridspec.nc`.

### CATRINE D7.1 on GEOS-IT C180 (GPU, 4 Tracers)

Multi-tracer intercomparison run: CO2, fossil CO2, SF6, 222Rn on C180
cubed-sphere with RAS convection and non-local PBL diffusion.

```bash
# 1. Download GEOS-IT C180 met data (mass fluxes + surface fields)
julia --project=. scripts/downloads/download_geosit_physics.jl

# 2. Preprocess met data to binary
julia --project=. scripts/preprocessing/preprocess_geosfp_cs.jl \
    config/preprocessing/catrine_geosit_c180.toml

# 3. Preprocess emissions (one-time per source)
julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/edgar_sf6.toml
julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/zhang_rn222.toml
julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/lmdz_co2.toml
julia --project=. scripts/preprocessing/regrid_emissions.jl config/emissions/gridfed_fossil_co2.toml

# 4. Run (7 days in ~3 min on L40S GPU)
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180.toml
```

Output: 3-hourly native CS binary files (~1.8 GB/day) with 4 tracer
3D fields, column masses, emission fluxes, surface pressure, and PBL height.
See [EMISSION_REGRIDDING.md](EMISSION_REGRIDDING.md) for emission
preprocessing details.

### ERA5 Spectral (GPU, Recommended)

```bash
# 1. Download ERA5 spectral GRIB (VO, D, LNSP)
python scripts/download_era5_grib_tm5.py

# 2. Preprocess spectral → mass fluxes (one-time)
julia --project=. scripts/preprocess_spectral_massflux.jl \
    config/preprocessing/spectral_june2023.toml

# 3. Run
julia --threads=2 --project=. scripts/run.jl config/runs/era5_spectral_june2023.toml
```

---

## Performance Tips

- **Use Float32 for GPU**: `float_type = "Float32"` halves memory and
  doubles throughput on NVIDIA GPUs.
- **Use DoubleBuffer on GPU**: `strategy = "double"` with `--threads=2`
  overlaps disk reads with GPU compute.
- **Preprocess to binary**: Preprocessed flat binaries are ~15x faster
  than on-the-fly NetCDF decompression. The bottleneck is CPU-bound NetCDF
  decoding, not disk bandwidth.
- **Level merging**: `merge_levels_above_Pa = 3000` collapses thin
  stratospheric levels, allowing larger timesteps without CFL violations.

**Performance baselines** (L40S GPU):

  | Configuration | I/O | GPU | Output | Total |
  |--------------|-----|-----|--------|-------|
  | ERA5 LL GPU (spectral, merged 110L) | — | 4.78 s/win | — | 573s / 30d |
  | GEOS-FP C720 GPU (NetCDF, double-buf) | 13.75 | 1.28 | 3.01 | ~18 s/win |
  | GEOS-FP C720 GPU + BL diffusion | 13.4 | 0.80 | 0.13 | ~14 s/win |
  | GEOS-IT C180 GPU (NetCDF, single-buf) | 0.93 | 0.11 | 0.05 | 792s / 30d |
  | GEOS-IT C180 CATRINE (4 tracers, RAS+PBL) | 0.12 | 0.76 | 0.18 | 177s / 7d |
  | GEOS-IT C180 CATRINE + Lin-Rood (4 tracers) | 0.12 | 1.3 | 0.18 | ~2.5 s/win |

---

## Further Reading

- [EMISSION_REGRIDDING.md](EMISSION_REGRIDDING.md) — conservative regridding
  tutorial: algorithm, TOML configs, validation, adding new sources
- [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md) — deep dive on met data
  formats, hybrid vertical coordinates, TM5 mass-flux comparison
- [docs/src/literated/design_principles.md](src/literated/design_principles.md) —
  architectural philosophy and multi-dispatch design
- [docs/src/literated/advection_theory.md](src/literated/advection_theory.md) —
  mathematical framework for mass-flux advection
- [docs/src/grids.md](src/grids.md) — grid types and vertical coordinate system
- [docs/src/gpu_double_buffering.md](src/gpu_double_buffering.md) — GPU
  double-buffering strategy for I/O overlap
