# AtmosTransport Data Layout Guide

Standard folder structure for met data, emissions, initial conditions, and output.
All paths are relative to a **data root** specified in the run config.

## Overview

```
<data_root>/                           # e.g., ~/data/AtmosTransport
├── met/                               # Meteorological driver data
│   ├── geosit_c180/                   # GEOS-IT C180 (long-term archive)
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
│   ├── geosfp_c720/                   # GEOS-FP C720 (same structure)
│   │   ├── raw/
│   │   └── preprocessed/
│   │
│   └── era5/                          # ERA5 (spectral or gridpoint)
│       ├── spectral/                  # GRIB spectral VO/D/LNSP (T639), 4×/day
│       │   ├── era5_spectral_YYYYMMDD_vo_d.gb    (~850 MB/day)
│       │   └── era5_spectral_YYYYMMDD_lnsp.gb    (~6 MB/day)
│       ├── spectral_hourly/           # Same layout; hourly LNSP+VO/D if retrieved with hourly times
│       │   └── era5_spectral_YYYYMMDD_*.gb       (24 LNSP messages/day; vo_d much larger)
│       ├── cmfmc/                     # GRIB model-level convective mass flux
│       │   └── era5_cmfmc_YYYYMMDD.nc            (~540 MB/day, UDMF+DDMF)
│       ├── detrainment/              # GRIB model-level detrainment rates
│       │   └── era5_detr_YYYYMMDD.nc             (~540 MB/day, UDRF+DDRF)
│       ├── surface/                   # NetCDF single-level fields
│       │   ├── era5_surface_YYYYMMDD.nc           (~50 MB/day)
│       │   └── era5_convective_YYYYMMDD.nc        (~180 MB/day, pressure-level omega)
│       ├── model_level_1deg/          # NetCDF gridpoint u/v/w (1°, monthly files)
│       │   └── era5_ml_YYYYMM.nc                 (~8 GB/month)
│       └── preprocessed/              # NetCDF mass fluxes (from spectral)
│           └── era5_massflux_YYYYMM.nc            (on NVMe: /temp1/)
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

## Storage Tiers

| Tier | Location | Speed | Capacity | Purpose |
|------|----------|-------|----------|---------|
| **Archive** | `~/data/AtmosTransport/met/<product>/raw/` | HDD ~200 MB/s | 20+ TB | Long-term NetCDF storage |
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
netcdf_dir           = "~/data/AtmosTransport/met/geosit_c180/raw"

# Preprocessed binary (fast mmap path)
preprocessed_dir     = "/temp1/met/geosit_c180/preprocessed/massflux"
surface_data_bin_dir = "/temp1/met/geosit_c180/preprocessed/physics"

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

## Size Budget per Day (ERA5 Spectral, 0.5°)

| Collection | Size/day | Windows/day | Note |
|------------|----------|-------------|------|
| Spectral VO/D | ~850 MB | 4 | T639 GRIB, all 137 levels |
| Spectral LNSP | ~6 MB | 4 | Level 1 only |
| CMFMC (model-level) | ~540 MB | 4 | 137-level UDMF+DDMF |
| Detrainment rates | ~540 MB | 4 | 137-level UDRF+DDRF (for TM5 convection) |
| Surface fields | ~50 MB | 24 | BLH, SSHF, T2M, U10, V10 |
| **Raw total** | **~2.0 GB** | | |
| Preprocessed mass flux | ~2 GB | 4 | NetCDF am/bm/cm/m + entu/detu/entd/detd |

For 31 days (Dec 2021): Raw ~45 GB, Preprocessed ~62 GB.
For 761 days (CATRINE): Raw ~1.1 TB, Preprocessed ~1.5 TB.

## Download Scripts

### ERA5

| Script | Collection | Source | API |
|--------|-----------|--------|-----|
| `scripts/downloads/download_era5_spectral.py` | VO, D, LNSP (spectral GRIB) | ECMWF MARS / CDS | MARS preferred |
| `scripts/downloads/download_era5_cmfmc.py` | CMFMC (model-level GRIB) | CDS (reanalysis-era5-complete) | CDS w/ MARS access |
| `scripts/downloads/download_era5_detrainment.py` | UDRF+DDRF (model-level GRIB) | CDS (reanalysis-era5-complete) | CDS w/ MARS access |
| `scripts/downloads/download_era5_surface_fields.py` | BLH, SSHF, T2M, U10, V10 + omega | CDS | CDS |
| `scripts/downloads/download_era5_model_levels.jl` | u, v, w, lnsp (gridpoint NC) | CDS | CDS |

### GEOS-IT / GEOS-FP

| Script | Collection | Source |
|--------|-----------|--------|
| `scripts/download_geosit_ctm_i1.jl` | CTM_I1 | WashU GEOS-IT archive |
| (built into preprocessor) | CTM_A1, A1, A3mstE, A3dyn, I3 | WashU |

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
