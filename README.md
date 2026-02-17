# AtmosTransportModel.jl

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

## References

- Krol et al. (2005): TM5 two-way nested zoom algorithm
- Huijnen et al. (2010): TM5 tropospheric chemistry v3.0
- Russell & Lerner (1981): Slopes advection scheme
- Putman & Lin (2007): Finite-volume on cubed-sphere grids

## License

MIT
