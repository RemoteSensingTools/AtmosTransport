# Reproducible reference run (ECMWF/ERA5)

This document defines the canonical forward run used for validation and TM5 comparison. Same config ensures reproducible results and allows side-by-side comparison with TM5 when run with the same met.

## Command

```bash
# Using the TOML-driven universal runner:
julia --threads=2 --project=. scripts/run.jl config/runs/era5_spectral_june2023.toml
```

## Reference config (pinned)

| Parameter | Value |
|-----------|--------|
| Config | `config/runs/era5_spectral_june2023.toml` |
| Met source | ERA5 spectral (VO, D, LNSP → mass-conserving mass fluxes via `preprocess_spectral_massflux.jl`) |
| Horizontal | 2° × ~2° (720 × 361) |
| Vertical | 137 hybrid sigma-pressure levels (ERA5 L137, A/B coefficients from `config/era5_L137_coefficients.toml`) |
| Time step Δt | 1800 s (30 min) |
| Simulation length | 1 month (June 2023) |
| Advection | PPM-7 (Putman & Lin 2007) |
| Physics | Mass-flux advection (Strang splitting), Tiedtke convection, boundary-layer diffusion (Thomas solver), EDGAR v8.0 surface emissions |
| GPU | Full simulation loop on GPU (Float32, KernelAbstractions.jl) |
| Initial condition | Uniform 0 ppm (anthropogenic CO₂ enhancement only) |

## Data and paths

- **ERA5 spectral GRIB:** `~/data/metDrivers/era5/spectral_june2023/` (VO, D, LNSP)
- **Preprocessed mass fluxes:** `/temp1/atmos_transport/era5_spectral/` (from `preprocess_spectral_massflux.jl`)
- **EDGAR emissions:** auto-downloaded by runner

Download scripts: `scripts/download_era5_grib_tm5.py` (spectral GRIB), `scripts/preprocess_spectral_massflux.jl` (spectral → mass fluxes).

**Note:** The spectral pipeline is recommended over the gridpoint pipeline
(`preprocess_mass_fluxes.jl`) for better mass conservation. See [CAVEATS.md](CAVEATS.md).

## Use for validation

- **Regression:** Compare `reference_era5_output.nc` (or a checksum of key variables) after code changes.
- **TM5 comparison:** Run TM5 with the same period and resolution (and same IC if possible); use `scripts/compare_tm5_output.jl` to compute metrics.
