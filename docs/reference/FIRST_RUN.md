# First Run Guide

End-to-end tutorial: from zero to your first transport simulation.

## Prerequisites

1. **Julia 1.10+** — install from [julialang.org](https://julialang.org/downloads/)
2. **Clone the repo**:
   ```bash
   git clone https://github.com/<org>/AtmosTransportModel.git
   cd AtmosTransportModel
   ```
3. **Install dependencies**:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

## Step 1: Get meteorological data

You need preprocessed transport binaries. See [DOWNLOAD_SETUP.md](DOWNLOAD_SETUP.md)
for credentials and [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) for which
preprocessor to run.

For a quick start, the smallest test is the **ERA5 LatLon 96x48** grid:

```bash
# Download ERA5 spectral data for one day
julia --project=. scripts/downloads/download_era5_spectral.jl \
    config/downloads/era5_spectral_dec2021.toml --day 2021-12-01

# Preprocess to transport binary
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/era5_latlon_transport_binary_v2.toml --day 2021-12-01
```

## Step 2: Create a run configuration

TOML configs live in `config/runs/`. A minimal config:

```toml
[input]
binary_paths = ["path/to/era5_transport_v2_20211201_float64.bin"]

[numerics]
float_type = "Float64"

[run]
scheme = "upwind"        # "upwind", "slopes", "ppm"
start_window = 1
stop_window = 12         # 12 windows = 24 hours (2h per window)

[tracers.co2.init]
kind = "uniform"
background = 4.11e-4     # 411 ppm as mixing ratio
```

See `config/runs/` for complete examples with emissions, snapshots, and multi-tracer setups.

## Step 3: Run the simulation

```bash
# LatLon and Reduced Gaussian grids:
julia --project=. scripts/run_transport_binary.jl config/runs/your_config.toml

# Cubed-sphere grids:
julia --project=. scripts/run_cs_transport.jl config/runs/your_cs_config.toml
```

Expected output:
```
[ Info: Backend: CPU()
[ Info: Running era5_transport_v2_20211201_float64.bin with upwind on LatLonMesh(720×361) (12 windows)
[ Info: Finished era5_transport_v2_20211201_float64.bin in 19.60 s
[ Info: Final air-mass change vs initial state:  5.880e-09
[ Info: Final tracer-mass drift for co2:         5.880e-09
```

Mass drift should be < 10^-4 for a properly balanced binary.

## Step 4: Visualize results

If your config has `[output] snapshot_hours`, a NetCDF snapshot file is produced.
Visualize with:

```bash
python3 scripts/visualization/plot_catrine_8way_snapshots.py \
    --datadir ~/data/AtmosTransport/output/ \
    --outdir ~/data/AtmosTransport/output/png/
```

Or use Julia with CairoMakie (see `scripts/visualization/` for examples).

## Next steps

- **Change the advection scheme**: set `scheme = "slopes"` or `"ppm"` in your TOML
- **Try a different grid**: see [GRID_TYPES.md](GRID_TYPES.md) for LL vs RG vs CS
- **Add emissions**: see the `[tracers.fossil_co2.surface_flux]` sections in Catrine configs
- **Run on GPU**: set `use_gpu = true` in `[architecture]` (requires CUDA.jl)
- **Multi-day runs**: add multiple binary_paths (one per day)
