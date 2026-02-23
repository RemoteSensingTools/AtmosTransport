# Quick Start Guide

End-to-end instructions for running AtmosTransportModel: downloading
meteorological data, preprocessing, configuring, and running simulations.

For a deep dive into met data formats, vertical coordinates, and TM5
comparison, see [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md).

## Prerequisites

- **Julia 1.10+** (tested with 1.12.4 via [juliaup](https://github.com/JuliaLang/juliaup))
- **Project dependencies**: install once with
  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```
- **GPU** (optional): NVIDIA GPU with CUDA 12+ drivers. The model loads
  CUDA.jl automatically when the TOML config sets `use_gpu = true`.

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

| Config | Grid | Met Source | GPU | Buffering | Description |
|--------|------|-----------|-----|-----------|-------------|
| `config/runs/geosfp_cs_edgar.toml` | C720 cubed-sphere | GEOS-FP preprocessed | Yes | Single | Native CS mass fluxes, 5 days |
| `config/runs/geosfp_cs_edgar_scratch.toml` | C720 cubed-sphere | GEOS-FP on `/scratch` | Yes | Single | Fast-disk timing test, 2 days |
| `config/runs/era5_edgar.toml` | 1° lat-lon | ERA5 model levels | No | Double | Raw ERA5 winds, June 2024 |
| `config/runs/era5_preprocessed.toml` | 1° lat-lon | Preprocessed binary | Yes | Single | Pre-computed mass fluxes |

### Examples

```bash
# GEOS-FP cubed-sphere on GPU (default: 5-day June 2024)
julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml

# ERA5 lat-lon on CPU
julia --project=. scripts/run.jl config/runs/era5_edgar.toml

# ERA5 preprocessed on GPU
julia --project=. scripts/run.jl config/runs/era5_preprocessed.toml
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

### `[met_data]`

Common keys for all drivers:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `driver` | string | (required) | `"era5"`, `"preprocessed_latlon"`, or `"geosfp_cs"` |
| `dt` | float | `900` | Advection sub-step [seconds] |
| `met_interval` | float | `3600` | Met update interval [seconds] |

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

**GEOS-FP cubed-sphere driver** (`driver = "geosfp_cs"`):

| Key | Type | Description |
|-----|------|-------------|
| `preprocessed_dir` | string | Directory with per-day `.bin` files |
| `netcdf_dir` | string | Fallback: directory with raw GEOS-FP `.nc4` files |
| `start_date` | string | Start date `"YYYY-MM-DD"` |
| `end_date` | string | End date `"YYYY-MM-DD"` |
| `Hp` | int | Halo width (default: 3) |

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

### GEOS-FP CS: NetCDF to Binary

Converts native GEOS-FP C720 NetCDF mass flux files to flat binary files
with haloed DELP and staggered mass flux panels (am, bm).

**Input**: Raw `.nc4` files from the download step
**Output**: Per-day `.bin` files for mmap-based GPU ingestion

```bash
export GEOSFP_DATA_DIR=~/data/geosfp_cs         # NetCDF input
export OUTDIR=~/data/geosfp_cs/preprocessed      # binary output
export GEOSFP_START=2024-06-01
export GEOSFP_END=2024-06-05
export FT_PRECISION=Float32                       # Float32 or Float64

julia --project=. scripts/preprocess_geosfp_cs.jl
```

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

### EDGAR Emissions: Lat-Lon to Cubed-Sphere Binary

Regrids EDGAR v8.0 CO2 emissions from 0.1° lat-lon to cubed-sphere panels
via nearest-neighbor interpolation with area-weighted unit conversion.

**Input**: EDGAR v8.0 NetCDF (`v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc`, ~26 MB)
**Output**: Compact binary with 6 panels of kg/m²/s flux (~12 MB)

```bash
export EDGAR_FILE=~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc
export OUTFILE=~/data/emissions/edgar_v8/edgar_cs_c720_float32.bin
export NC_GRID=720

julia --project=. scripts/preprocess_edgar_cs.jl
```

The binary format has a 4096-byte JSON header followed by 6 panels of
`Nc × Nc` Float32 values. The model auto-loads this file if it exists at the
default path; you can also specify it explicitly in the TOML:

```toml
[tracers.co2]
emission   = "edgar"
edgar_file = "~/data/emissions/edgar_v8/edgar_cs_c720_float32.bin"
```

### ERA5: Winds to Binary Mass Fluxes (Optional)

Pre-computes mass fluxes from ERA5 u/v winds and surface pressure. This
avoids recomputing staggering at each run, and is required for the
`preprocessed_latlon` driver.

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
GEOSFP_START_DATE=2024-06-01 GEOSFP_END_DATE=2024-06-05 \
  julia --project=. scripts/download_geosfp_cs_massflux.jl

# 2. Preprocess to flat binary (one-time, ~61 GB/day)
GEOSFP_DATA_DIR=~/data/geosfp_cs OUTDIR=~/data/geosfp_cs/preprocessed \
  GEOSFP_START=2024-06-01 GEOSFP_END=2024-06-05 \
  julia --project=. scripts/preprocess_geosfp_cs.jl

# 3. Preprocess EDGAR emissions (one-time, ~12 MB)
julia --project=. scripts/preprocess_edgar_cs.jl

# 4. Edit config if needed
#    vim config/runs/geosfp_cs_edgar.toml

# 5. Run!
julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml
```

### ERA5 Lat-Lon (CPU)

```bash
# 1. Download ERA5 model-level data (one-time, ~150 MB/day)
START_DATE=2024-06-01 END_DATE=2024-06-07 \
  julia --project=. scripts/download_era5_model_levels.jl

# 2. Run (no preprocessing needed — ERA5 driver reads NetCDF directly)
julia --project=. scripts/run.jl config/runs/era5_edgar.toml
```

---

## Performance Tips

- **Use Float32 for GPU**: `float_type = "Float32"` halves memory and
  doubles throughput on NVIDIA GPUs.
- **Use DoubleBuffer on GPU**: `strategy = "double"` with `--threads=2`
  overlaps disk reads with GPU compute (~40% faster on scratch disk).
- **Fast local disk**: Copy preprocessed binaries to `/scratch` or NVMe
  for fastest I/O. NAS storage can be 3–4× slower.
- **GEOS-FP C720 performance** (measured on A100 GPU, scratch disk):

  | Mode | I/O | GPU | Output | Total |
  |------|-----|-----|--------|-------|
  | SingleBuffer, NAS | ~3.5 s/win | ~1.5 s/win | ~1.5 s/win | ~6.4 s/win |
  | SingleBuffer, scratch | 0.97 s/win | 0.61 s/win | 0.17 s/win | 1.75 s/win |
  | DoubleBuffer, scratch | 0.27 s/win | 0.61 s/win | 0.16 s/win | ~1.0 s/win |

---

## Further Reading

- [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md) — deep dive on met data
  formats, hybrid vertical coordinates, TM5 mass-flux comparison
- [docs/src/literated/design_principles.md](src/literated/design_principles.md) —
  architectural philosophy and multi-dispatch design
- [docs/src/literated/advection_theory.md](src/literated/advection_theory.md) —
  mathematical framework for mass-flux advection
- [docs/src/grids.md](src/grids.md) — grid types and vertical coordinate system
- [docs/src/gpu_double_buffering.md](src/gpu_double_buffering.md) — GPU
  double-buffering strategy for I/O overlap
