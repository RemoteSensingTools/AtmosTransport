# Reproducible reference run (ECMWF/ERA5)

This document defines the canonical forward run used for validation and TM5 comparison. Same config ensures reproducible results and allows side-by-side comparison with TM5 when run with the same met.

## Command

```bash
# 1. Ensure ERA5 data exists (see Data and paths below)
julia --project=. scripts/download_era5_week.jl   # if needed

# 2. Run reference forward
julia --project=. scripts/run_reference_ecmwf.jl
```

Output: `data/era5/output/reference_era5_output.nc`

## Reference config (pinned)

| Parameter | Value |
|-----------|--------|
| Met source | ERA5 (pressure-level + single-level NetCDF) |
| Horizontal | 2.5° (from ERA5 file; typically 144×73) |
| Vertical | 37 pressure levels (1–1000 hPa) |
| Time step Δt | 600 s (10 min) |
| Simulation length | 48 h (2 days) |
| Physics | Slopes advection (limiter on), no convection (no conv_mass_flux), no diffusion in current ERA5 script |
| Initial condition | 420 ppm + Gaussian blobs (Europe, East Asia, Eastern US) |

## Data and paths

- **Pressure-level file:** `data/era5/era5_pressure_levels_20250201_20250207.nc`
- **Single-level file:** `data/era5/era5_single_levels_20250201_20250207.nc`
- **Output:** `data/era5/output/reference_era5_output.nc`

Paths are relative to project root. Download script: `scripts/download_era5_week.jl` (requires CDS API key for ERA5).

## Use for validation

- **Regression:** Compare `reference_era5_output.nc` (or a checksum of key variables) after code changes.
- **TM5 comparison:** Run TM5 with the same period and resolution (and same IC if possible); use `scripts/compare_tm5_output.jl` to compute metrics.
