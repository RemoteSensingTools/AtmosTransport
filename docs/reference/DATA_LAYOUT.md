# AtmosTransport Data Layout Guide

Standard folder structure for met data, emissions, initial conditions, and output.
All paths are relative to a **data root** specified in the run config.

## Canonical naming (new default)

Use canonical grid names from `docs/GRID_CONVENTIONS.md` directly in folder paths.
Organize first by met source family, then by canonical grid name, cadence, and payload type.

Recommended hierarchy:

```
<data_root>/met/<met_source>/<grid_name>/<cadence>/<payload_type>/<dataset_variant>/...
```

Where:

- `met_source`: `era5`, `geosfp`, `geosit`, `merra2`, ...
- `grid_name`: canonical names (`0.5x0.5`, `1.0x1.0`, `N320`, `O1280`, `C180`, `C720`, ...)
- `cadence`: `hourly`, `3hourly`, `6hourly`, `daily`, ...
- `payload_type`: `spectral`, `gridpoint`, `surface`, `massflux`, `physics`, ...
- `dataset_variant`: optional source-specific tag (for example
  `v4_tropo34_dry`, `ctm_i1`, `catrine_hourly`)

Examples:

- `met/era5/N320/hourly/spectral/era5_spectral_20211201_vo_d.gb`
- `met/era5/N320/hourly/gridpoint/...`
- `met/era5/N320/hourly/surface/...`
- `met/era5/0.5x0.5/hourly/massflux/v4_tropo34_dry/era5_v4_20211201_merged1000Pa_float32.bin`
- `met/geosfp/C720/hourly/raw/...`
- `met/geosit/C180/hourly/massflux/...`

### ERA5 spectral note (current state)

As of Apr 2026, the old ERA5 6-hourly spectral archive was removed to avoid
ambiguity. Keep only the hourly spectral set for active preprocessing.

### Manifest metadata (required per dataset root)

Each dataset root should include a small `manifest.json` with at least:

- `met_source`
- `grid_name` (canonical, e.g. `C180`, `N320`, `0.5x0.5`)
- `grid_type` (`latlon`, `reduced_gaussian`, `cubed_sphere`)
- `horizontal_topology` (`StructuredDirectional` or `FaceIndexed`)
- `panel_convention` (for cubed-sphere datasets: `GEOSNative` or `Gnomonic`)
- `mass_basis` (`moist` or `dry`)
- `vertical_coordinate_type`
- `format_version`

This keeps runtime path discovery simple and avoids encoding too much logic in
folder-name parsing.

### Legacy aliases

Legacy folder names are still common (`geosit_c180`, `geosfp_c720`, etc.), but
new data should use canonical split paths:

- `geosit_c180` -> `geosit/C180`
- `geosfp_c720` -> `geosfp/C720`
- `era5` remains `era5`, but add explicit grid folder such as `0.5x0.5` or `N320`

## Overview

```
<data_root>/                           # e.g., ~/data/AtmosTransport
├── met/                               # Meteorological driver data
│   ├── geosit/C180/                   # GEOS-IT C180 (canonical)
│   │   ├── raw/                       # Original NetCDF from WashU
│   │   │   ├── 20211201/
│   │   │   │   ├── GEOSIT.20211201.CTM_A1.C180.nc   (4.2 GB, hourly mass flux)
│   │   │   │   ├── GEOSIT.20211201.CTM_I1.C180.nc   (851 MB, hourly QV+PS)
│   │   │   │   ├── GEOSIT.20211201.A1.C180.nc        (200 MB, hourly surface)
│   │   │   │   ├── GEOSIT.20211201.A3mstE.C180.nc    (1.4 GB, 3-hr convection)
│   │   │   │   ├── GEOSIT.20211201.A3dyn.C180.nc     (2.5 GB, 3-hr dynamics)
│   │   │   │   └── GEOSIT.20211201.I3.C180.nc        (330 MB, 3-hr thermo)
│   │   │   ├── 20211202/
│   │   │   └── ...
│   │   └── preprocessed/              # Flat binary (fast mmap, on NVMe)
│   │       ├── massflux/              # CTM_A1 → binary (MFXC, MFYC, DELP)
│   │       │   ├── cs180_20211201.bin
│   │       │   └── ...
│   │       └── physics/               # Surface + 3D fields → binary
│   │           ├── GEOSFP_CS180.20211201.A1.bin       (A1 surface: PBLH etc.)
│   │           ├── GEOSFP_CS180.20211201.A3mstE.bin   (CMFMC)
│   │           ├── GEOSFP_CS180.20211201.A3dyn.bin    (DTRAIN)
│   │           ├── GEOSFP_CS180.20211201.CTM_I1.bin   (hourly QV, GCHP-aligned)
│   │           └── GEOSFP_CS180.20211201.I3.bin       (3-hr QV, fallback)
│   │
│   ├── geosfp/C720/                   # GEOS-FP C720 (canonical)
│   │   ├── raw/
│   │   └── preprocessed/
│   │
│   └── era5/                          # ERA5 (canonical split by grid + cadence + payload)
│       ├── N320/
│       │   └── hourly/
│       │       ├── spectral/          # GRIB spectral VO/D/LNSP (hourly archive)
│       │       ├── gridpoint/         # (optional) N320 gridpoint products
│       │       └── surface/           # (optional) N320 surface products
│       └── 0.5x0.5/
│           └── hourly/
│               ├── massflux/          # transport-ready binaries (v4/v5, moist/dry)
│               ├── cmfmc/             # GRIB/NC convective mass flux staging
│               ├── detrainment/       # GRIB/NC detrainment staging
│               └── surface/           # NetCDF single-level fields
│
├── emissions/                         # Emission inventories
│   ├── edgar_v8/                      # EDGAR v8.0 gridded
│   │   ├── edgar_co2_cs_c180_float32.bin
│   │   ├── edgar_sf6_cs_c180_float32.bin
│   │   └── ...
│   ├── gridfed/                       # GriFED fossil CO2
│   ├── zhang_rn222/                   # Zhang et al. Rn-222
│   └── lmdz_co2/                      # LMDZ posterior CO2 fluxes
│
├── initial_conditions/                # Tracer ICs
│   ├── startCO2_202112010000.nc
│   ├── startSF6_202112010000.nc
│   └── ...
│
├── grids/                             # Grid specification files
│   ├── cs_c180_gridspec.nc            # C180 cell centers, corners, areas
│   └── cs_c720_gridspec.nc            # C720 equivalent
│
└── output/                            # Simulation output
    ├── <run_name>/                    # One directory per run
    │   ├── <run_name>_20211201.bin
    │   ├── <run_name>_20211201.nc     # Auto-converted
    │   └── ...
    └── ...
```

## Migration map (current -> canonical)

The table below gives concrete mapping examples from current paths to the
canonical convention above.

| Current path (example) | Canonical path (target) | Status |
|---|---|---|
| `met/geosit_c180/raw/...` | `met/geosit/C180/raw/...` | Planned — run configs still use legacy |
| `met/geosit_c180/preprocessed/...` | `met/geosit/C180/preprocessed/...` | Planned — run configs still use legacy |
| `met/geosfp_c720/raw/...` | `met/geosfp/C720/raw/...` | Planned — run configs still use legacy |
| `met/geosfp_c720/preprocessed/...` | `met/geosfp/C720/preprocessed/...` | Planned — run configs still use legacy |
| `met/era5/spectral_hourly/...` | `met/era5/N320/hourly/spectral/...` | Planned — preprocessing configs use legacy |
| `met/era5/spectral/...` (6-hourly) | `met/era5/N320/6hourly/spectral/...` | Removed from active archive (Apr 2026 cleanup) |
| `met/era5/spectral_v4_tropo34_dec2021/...` | `met/era5/0.5x0.5/preprocessed/massflux_v4_moist_tropo34_dec2021/...` | Planned — include basis (`moist`/`dry`) in variant |
| `met/era5/spectral_v4_tropo34_dec2021_dry/...` | `met/era5/0.5x0.5/preprocessed/massflux_v4_dry_tropo34_dec2021/...` | Planned — dry-basis explicit in path + manifest |
| `met/era5/model_level_1deg/...` | `met/era5/1.0x1.0/raw/model_level/...` | Planned — keep resolution explicit |

Practical rollout rule: do not break existing runs; support both legacy and
canonical roots during transition, but write new outputs only to canonical
paths.

## Storage Tiers

| Tier | Location | Speed | Capacity | Purpose |
|------|----------|-------|----------|---------|
| **Archive** | `~/data/AtmosTransport/met/<source>/<grid>/<cadence>/<payload>/` | HDD ~200 MB/s | 20+ TB | Long-term raw + staging storage |
| **Staging** | `/temp1/` (NVMe) | NVMe ~3 GB/s | 7 TB (2.3 free) | Preprocessed binary for active runs |
| **GPU** | Device memory | ~900 GB/s | 46 GB per L40S | Current + next window buffers |

### NVMe Staging Strategy

Not all data fits on NVMe simultaneously. For long runs:

1. **Preprocess the full period** to binary on NVMe (one-time cost)
2. If NVMe is too small, use **rolling window**:
   - Preprocess 7 days ahead of the simulation
   - Delete preprocessed files >2 days behind the current window
   - The run loop's double-buffer only needs current + next window
3. **Priority order for NVMe space**:
   - massflux/ (largest, most accessed: ~120 MB/day for C180)
   - physics/ (CTM_I1 at 1.3 GB/day is the largest physics binary)
   - A3mstE, A3dyn at ~200 MB/day each

### Size Budget per Day (GEOS-IT C180)

| Collection | Raw NetCDF | Binary | Windows/day | Note |
|------------|-----------|--------|-------------|------|
| CTM_A1 | 4.2 GB | 120 MB | 24 | Mass fluxes (already preprocessed) |
| CTM_I1 | 851 MB | 1.3 GB | 24 | Hourly QV (GCHP-aligned) |
| A1 | 200 MB | 15 MB | 24 | Surface fields |
| A3mstE | 1.4 GB | 200 MB | 8 | CMFMC (convection) |
| A3dyn | 2.5 GB | 200 MB | 8 | DTRAIN, winds |
| I3 | 330 MB | 200 MB | 8 | 3-hr QV (fallback) |
| **Total** | **~9.5 GB** | **~2.0 GB** | | |

For 761 days (CATRINE): Raw ~7.2 TB, Binary ~1.5 TB.

## Config Mapping

The TOML config maps to this structure:

```toml
[met_data]
# Raw NetCDF (for direct reading or reference)
netcdf_dir           = "~/data/AtmosTransport/met/geosit/C180/raw"

# Preprocessed binary (fast mmap path)
preprocessed_dir     = "/temp1/met/geosit/C180/preprocessed/massflux"
surface_data_bin_dir = "/temp1/met/geosit/C180/preprocessed/physics"

# Grid specification
coord_file           = "~/data/AtmosTransport/grids/cs_c180_gridspec.nc"
```

The runtime automatically searches for data in this order:
1. Binary in `surface_data_bin_dir` (fastest, mmap)
2. NetCDF in `netcdf_dir` (slower but always works)

For CTM_I1 (hourly QV), the runtime checks:
1. `surface_data_bin_dir/GEOSFP_CS180.YYYYMMDD.CTM_I1.bin`
2. `netcdf_dir/YYYYMMDD/GEOSIT.YYYYMMDD.CTM_I1.C180.nc`
3. Falls back to I3 (3-hourly) if CTM_I1 unavailable

Legacy aliases (`geosit_c180`, `geosfp_c720`, etc.) may still exist as
symlinks during transition, but new configs should use canonical paths.

## Size Budget per Day (ERA5 Spectral, hourly archive)

| Collection | Size/day | Windows/day | Note |
|------------|----------|-------------|------|
| Spectral VO/D | ~5.4 GB | 24 | TL639 spectral GRIB, all 137 levels |
| Spectral LNSP | ~20 MB | 24 | Level 1 spectral GRIB |
| CMFMC (model-level) | ~540 MB | 4 | 137-level UDMF+DDMF |
| Detrainment rates | ~540 MB | 4 | 137-level UDRF+DDRF (for TM5 convection) |
| Surface fields | ~50 MB | 24 | BLH, SSHF, T2M, U10, V10 |
| **Raw total** | **~6.5 GB** | | |
| Preprocessed mass flux | ~2 GB | 4 | NetCDF am/bm/cm/m + entu/detu/entd/detd |

For 31 days (Dec 2021): Raw ~45 GB, Preprocessed ~62 GB.
For 761 days (CATRINE): Raw ~1.1 TB, Preprocessed ~1.5 TB.

## Download System

### Unified entry point

All downloads are driven by TOML configs in `config/downloads/` and executed via
a single CLI:

```bash
julia --project=. scripts/downloads/download_data.jl config/downloads/<recipe>.toml \
    [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]
```

Download recipes reference source definitions in `config/met_sources/` and produce
output in canonical paths automatically.

### Download recipes → canonical output paths

| Recipe TOML | Optimal chunk | Canonical output path |
|-------------|---------------|----------------------|
| `era5_native_monthly.toml` | Monthly | `met/era5/N320/hourly/raw/{ml_an_native_core,ml_fc_convection,sfc_an_native}/` |
| `geosfp_c720.toml` | Per-file | `met/geosfp/C720/hourly/raw/YYYYMMDD/` |
| `geosit_c180.toml` | Per-file | `met/geosit/C180/daily/raw/YYYYMMDD/` |
| `merra2.toml` | Per-day | `met/merra2/0.5x0.625/3hourly/raw/` |

### Legacy scripts (in `scripts/downloads/`)

Individual download scripts are retained for reference but deprecated in favor
of the unified system above.

#### ERA5

| Script | Collection | Source | API |
|--------|-----------|--------|-----|
| `download_era5_spectral.py` | VO, D, LNSP (spectral GRIB) | ECMWF MARS / CDS | MARS preferred |
| `download_era5_native_monthly.py` | All core+convection+surface (monthly GRIB) | CDS | CDS |
| `download_era5_physics.py` | Convection + thermodynamics | CDS | CDS |
| `download_era5_surface_fields.py` | BLH, SSHF, T2M, U10, V10 | CDS | CDS |
| `download_era5_model_levels.jl` | u, v, w, lnsp (gridpoint NC) | CDS | CDS |

#### GEOS-IT / GEOS-FP

| Script | Collection | Source |
|--------|-----------|--------|
| `download_geosfp_cs_massflux.jl` | C720/C180 CTM mass flux | WashU archive |
| `download_geosit_c180_s3.sh` | C180 all collections | AWS S3 (public) |
| `download_geosfp_surface_fields.jl` | 0.25° surface+physics | AWS S3 (public) |

## Preprocessing Scripts

### ERA5

| Script | Input | Output |
|--------|-------|--------|
| `scripts/preprocessing/preprocess_spectral_massflux.jl` | Spectral GRIB (VO/D/LNSP) | NetCDF mass fluxes (am, bm, cm, m, ps) |
| `scripts/preprocessing/merge_era5_cmfmc_to_massflux.jl` | CMFMC NetCDF (cmfmc/) | Appends `conv_mass_flux` to massflux NC |
| `scripts/preprocessing/preprocess_era5_tm5_convection.jl` | CMFMC + detrainment NetCDF | Appends entu/detu/entd/detd to massflux NC |

### GEOS-IT / GEOS-FP

| Script | Input | Output |
|--------|-------|--------|
| `scripts/preprocessing/preprocess_geosfp_cs.jl` | CTM_A1 NetCDF | massflux/ binary |
| `scripts/preprocessing/convert_surface_cs_to_binary.jl` | A1, A3mstE, A3dyn NetCDF | physics/ binary |
| `scripts/preprocessing/convert_ctm_i1_to_binary.jl` | CTM_I1 NetCDF | physics/ CTM_I1 binary |
