# AtmosTransportModel.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://RemoteSensingTools.github.io/AtmosTransportModel/dev/)

A Julia-based atmospheric transport model for GPU and CPU, inspired by TM5 and designed with
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)-style multiple dispatch patterns.

## Features

- **Multi-grid:** Latitude-longitude and cubed-sphere grids with hybrid sigma-pressure vertical coordinates
- **Multi-backend:** Single codebase for CPU and GPU via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- **Multi-met-data:** Readers for ECMWF ERA5, NASA MERRA-2, and GEOS-FP with automatic regridding
- **Hand-coded discrete adjoint:** TM5-4DVar-style adjoint with Revolve checkpointing for bounded memory
- **Extensible:** Every physics operator is behind an abstract type; new schemes, grids, and data sources plug in via multiple dispatch without modifying core code
- **Operator splitting:** Symmetric Strang splitting (advection, convection, diffusion, sources) with paired forward/adjoint operators

## Quick start

```julia
using AtmosTransportModel

grid = LatitudeLongitudeGrid(CPU();
    size = (360, 180, 60),
    longitude = (-180, 180),
    latitude = (-90, 90),
    vertical = HybridSigmaPressure(levels=60))

model = TransportModel(;
    grid = grid,
    tracers = (:CO2, :CH4),
    advection = SlopesAdvection(),
    diffusion = BoundaryLayerDiffusion(),
    convection = TiedtkeConvection())
```

## Design principles

- **Julian:** Multiple dispatch, parametric types, no OOP inheritance chains
- **TM5-aligned:** Slopes advection, Tiedtke convection, operator splitting, discrete adjoint
- **Grid-agnostic:** Physics code dispatches on grid type; never assumes lat-lon layout
- **Adjoint-paired:** Every forward operator has a hand-coded adjoint counterpart
- **Extension-friendly:** Abstract types + interface contracts; adding a new scheme never requires editing core

## Validation

- **Tests:** 209 unit and integration tests (including 18 mass-flux advection tests); gradient tests for the adjoint. See `docs/VALIDATION.md`.
- **Mass-flux advection:** TM5-faithful co-advection of tracer mass and air mass with machine-precision conservation. See `docs/MASS_FLUX_EVOLUTION.md`.
- **Reproducible run:** `julia --project=. scripts/run_reference_ecmwf.jl` (ECMWF/ERA5 reference case; see `docs/REFERENCE_RUN.md`).
- **TM5 comparison:** Run TM5 locally (see `docs/TM5_LOCAL_SETUP.md`), then `scripts/compare_tm5_output.jl our_output.nc tm5_output.nc`.
- **GPU:** X-advection runs on CUDA via KernelAbstractions when `grid = LatitudeLongitudeGrid(GPU(); ...)` and `using CUDA`; y/z and convection/diffusion remain on CPU until ported.

## Documentation

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://RemoteSensingTools.github.io/AtmosTransportModel/dev/)

Full documentation is available at [RemoteSensingTools.github.io/AtmosTransportModel](https://RemoteSensingTools.github.io/AtmosTransportModel/dev/), including:

- **Theory:** Mathematical framework for mass-flux advection and TM5 comparison
- **Tutorials:** Step-by-step guides for running forward simulations with ERA5 and GEOS-FP
- **Developer Guide:** Validation results, TM5 code alignment, design history
- **API Reference:** Auto-generated docstrings for all exported types and functions

Source files for the documentation are in `docs/literate/` (Literate.jl scripts) and `docs/` (markdown reference docs).

## References

- Krol et al. (2005): TM5 two-way nested zoom algorithm
- Huijnen et al. (2010): TM5 tropospheric chemistry v3.0
- Russell & Lerner (1981): Slopes advection scheme
- Putman & Lin (2007): Finite-volume on cubed-sphere grids

## License

MIT
