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
| Met source | ERA5 model levels (spectral/hybrid sigma-pressure, preprocessed to mass fluxes) |
| Horizontal | 1° × 1° (360 × 180) |
| Vertical | 137 hybrid sigma-pressure levels (ERA5 L137, A/B coefficients from `config/era5_L137_coefficients.toml`) |
| Time step Δt | 1800 s (30 min) |
| Simulation length | 1 month (June 2024) |
| Physics | TM5-faithful mass-flux advection (Russell-Lerner slopes, Strang splitting), boundary-layer diffusion (Thomas solver), EDGAR v8.0 surface emissions |
| GPU | Full simulation loop on GPU (Float32, KernelAbstractions.jl) |
| Initial condition | Uniform 0 ppm (anthropogenic CO₂ enhancement only) |

## Data and paths

- **Preprocessed mass fluxes:** `~/data/output/era5_edgar_preprocessed_f32/massflux_era5_202406_float32.nc` (or flat binary equivalent)
- **EDGAR emissions:** `~/data/edgar/v8.0/IEA_EDGAR_CO2_2022_1.nc`
- **Output:** `~/data/output/era5_edgar_preprocessed_f32/output_era5_edgar.nc`

Download scripts: `scripts/download_era5_model_levels.jl` (CDS API), `scripts/preprocess_mass_fluxes.jl` (wind → mass flux conversion).

## Use for validation

- **Regression:** Compare `reference_era5_output.nc` (or a checksum of key variables) after code changes.
- **TM5 comparison:** Run TM5 with the same period and resolution (and same IC if possible); use `scripts/compare_tm5_output.jl` to compute metrics.
