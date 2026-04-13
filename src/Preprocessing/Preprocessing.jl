"""
    Preprocessing

Transport binary generation from meteorological source data.

Provides the complete pipeline from raw met data (spectral GRIB, gridded NetCDF)
to Poisson-balanced transport binaries ready for the runtime `TransportBinaryDriver`.

## Architecture

Three dispatch axes control the pipeline:

    AbstractMetSource         — where the data comes from
    ├── SpectralSource        — ERA5 spectral GRIB (VO, D, LNSP)
    ├── GriddedFluxSource     — GEOS-FP/IT NetCDF (MFXC, MFYC, DELP)
    └── GriddedWindSource     — MERRA-2 NetCDF (U, V, DELP, PS)

    AbstractTargetGrid        — what grid the binary uses
    ├── LatLonTarget          — regular lat-lon (any Nx × Ny)
    └── ReducedGaussianTarget — octahedral or regular RG (O160, O320, N320)

    AbstractLevelSelection    — how to merge vertical levels
    ├── EchlevsSelection      — TM5-style explicit interface indices
    ├── AutoMergeSelection    — merge by minimum pressure thickness
    └── TargetCountSelection  — merge to ~N levels preserving BL

## Pipeline

    preprocess_day!(source, target, levels, date, settings)

1. `load_source_data!(source, date)` — read raw met fields
2. `compute_native_fields!(raw, target, settings)` — spectral synthesis or wind→flux
3. `merge_levels!(native, levels, settings)` — vertical level merging
4. `balance_fluxes!(merged, target, settings)` — Poisson balance (FFT or CG)
5. `validate!(balanced, target, settings)` — CFL + cm sanity checks
6. `write_binary!(balanced, target, date, settings)` — transport binary output

## Usage

The library is called from a thin CLI script:

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl config.toml --day 2021-12-01
```

Advanced users can call the pipeline functions directly from Julia.
"""
module Preprocessing

using Dates
using Logging
using FFTW
using Printf
using JSON3
using TOML
using LinearAlgebra: mul!, dot
using NCDatasets
using GRIB
using FastGaussQuadrature: gausslegendre

# Re-export parent module types we need
using ..Architectures: CPU
using ..Grids: LatLonMesh, ReducedGaussianMesh, HybridSigmaPressure,
               AtmosGrid, ncells, nfaces, nrings, nlevels, face_cells, cell_area,
               ring_longitudes, ring_cell_count, cell_areas_by_latitude,
               n_levels, pressure_at_interface, level_thickness, floattype
using ..MetDrivers: TransportBinaryReader, TransportBinaryHeader, write_transport_binary

# Physical constants
include("constants.jl")

# Logging utilities
include("logging.jl")

# Vertical coordinate handling and level merging
include("vertical_coordinates.jl")

# Target grid geometry (LL and RG)
include("target_geometry.jl")

# GRIB spectral IO
include("spectral_io.jl")

# Spectral synthesis (Legendre + FFT → gridpoint)
include("spectral_synthesis.jl")

# Poisson balance (LL FFT + RG conjugate gradient)
include("mass_support.jl")

# Reduced Gaussian helpers (RG synthesis, RG balance, RG cm diagnosis)
include("reduced_transport_helpers.jl")

# Configuration parsing
include("configuration.jl")

# Binary pipeline (window storage, header, write)
include("binary_pipeline.jl")

# Exports for the CLI script and advanced users
export build_target_geometry, target_summary
export process_day, preprocess_day!

end # module Preprocessing
