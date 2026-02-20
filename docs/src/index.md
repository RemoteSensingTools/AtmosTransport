# AtmosTransportModel.jl

_Fast and flexible atmospheric transport on CPUs and GPUs._

AtmosTransportModel.jl is a Julia-based atmospheric transport model inspired by
[TM5](https://doi.org/10.5194/acp-5-445-2005) and designed with
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)-style multiple dispatch.
It supports latitude-longitude and cubed-sphere grids, multiple meteorological data
sources (ERA5, MERRA-2, GEOS-FP), a hand-coded discrete adjoint with Revolve
checkpointing, and extensible physics operators -- all from a single codebase that
runs identically on CPU and GPU via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Example: Column-Mean CO₂ Transport

![Column-mean CO₂ animation](assets/column_mean_animation_small.gif)

*One-month forward simulation (June 2024) of anthropogenic CO₂ transport on a
1° × 1° × 137-level grid, driven by ERA5 model-level spectral winds and
EDGAR v8.0 surface emissions. Column-averaged mixing ratio enhancement (ppm,
delta-pressure weighted) in Robinson projection.*

Mass fluxes are pre-computed from ERA5 hybrid-level u/v/omega fields following
TM5's continuity-consistent approach: horizontal mass fluxes are derived from the
winds, and vertical fluxes are diagnosed from horizontal convergence to guarantee
column mass conservation. Transport uses TM5-faithful mass-flux advection
(Russell--Lerner slopes scheme with Strang splitting) and boundary-layer diffusion
(implicit Thomas solver). The simulation loop runs entirely on a single NVIDIA L40S
GPU in Float32 arithmetic, with a parallelized tridiagonal solver for diffusion,
device-to-device air-mass resets, GPU-side diagnostics, and memory-mapped
flat-binary I/O for mass-flux ingestion (~15× faster than NetCDF).

## Quick install

AtmosTransportModel requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add(url="https://github.com/RemoteSensingTools/AtmosTransportModel.git")
```

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

## Documentation overview

| Section | Contents |
|:--------|:---------|
| Theory | Mathematical framework: mass-flux advection, continuity equation, TM5 comparison |
| Tutorials | Step-by-step guides for running forward simulations |
| Developer Guide | Validation details, TM5 alignment checklists, design history |
| [API Reference](@ref) | Auto-generated docstrings for all exported types and functions |

## References

- Krol et al. (2005): TM5 two-way nested zoom algorithm
- Huijnen et al. (2010): TM5 tropospheric chemistry v3.0
- Russell & Lerner (1981): Slopes advection scheme
- Putman & Lin (2007): Finite-volume on cubed-sphere grids
