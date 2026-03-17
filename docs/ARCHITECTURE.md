# AtmosTransport.jl — Architecture Guide

## Design Philosophy

AtmosTransport.jl follows the [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
design philosophy: **physics operators are configuration objects, not stateful processors**.
All behavior is determined by concrete types via Julia's multiple dispatch — no if/else
branching on grid type or met source in hot paths.

## Module Dependency Chain

```
Architectures  →  Parameters  →  Communications
       ↓
Grids  →  Fields
       ↓
Advection  →  Convection  →  Diffusion  →  Chemistry
       ↓
TimeSteppers  →  Adjoint  →  Callbacks
       ↓
Regridding  →  Diagnostics  →  Sources
       ↓
IO  →  Visualization  →  Models
```

Each module may import from earlier modules but never from later ones. `Models`
comes last because it depends on everything.

## Type Hierarchy

### Core Infrastructure

```
AbstractArchitecture
├── CPU                    # array_type → Array, device → KA.CPU()
└── GPU                    # array_type → CuArray/MTLArray, device → CUDA/Metal backend

AbstractGrid{FT, Arch}
├── AbstractStructuredGrid{FT, Arch}
│   ├── LatitudeLongitudeGrid    # ERA5, regular lat-lon
│   └── CubedSphereGrid          # GEOS-FP C720, GEOS-IT C180

AbstractBufferingStrategy
├── SingleBuffer           # sequential IO → compute → output
└── DoubleBuffer           # overlapped IO + compute via async tasks
```

### Physics Operators

Each operator type has a `No*` variant for pass-through and one or more implementations:

```
AbstractAdvectionScheme
├── SlopesAdvection         # van Leer slopes, 2nd-order (lat-lon)
└── PPMAdvection{ORD}       # Putman & Lin PPM, ORD ∈ {4,5,6,7} (cubed-sphere)

AbstractConvection
├── NoConvection
├── TiedtkeConvection       # Tiedtke (1989) mass-flux scheme
└── RASConvection           # Relaxed Arakawa-Schubert

AbstractDiffusion
├── NoDiffusion
├── BoundaryLayerDiffusion  # static exponential Kz profile
├── PBLDiffusion            # met-driven Kz (Beljaars & Viterbo 1998)
└── NonLocalPBLDiffusion    # local + counter-gradient (Holtslag & Boville 1993)

AbstractChemistry
├── NoChemistry
├── RadioactiveDecay        # exponential decay (e.g., ²²²Rn)
└── CompositeChemistry      # multiple schemes combined
```

### Met Data Drivers

```
AbstractMetDriver{FT}
├── ERA5MetDriver                  # reads raw winds, computes mass fluxes on-the-fly
├── PreprocessedLatLonMetDriver    # reads preprocessed binary (lat-lon)
└── GEOSFPCubedSphereMetDriver    # reads GEOS-FP/IT CS mass fluxes (binary or NetCDF)
```

### Emission Sources

```
AbstractSurfaceFlux
├── SurfaceFlux{Layout}              # constant-in-time
├── TimeVaryingSurfaceFlux{Layout}   # time-interpolated, optional cyclic wrapping
├── EdgarSource                      # EDGAR v8.0 anthropogenic inventories
├── CarbonTrackerSource              # CT-NRT biosphere/fire/ocean
└── ...more inventory sources
```

## Data Flow

```
TOML Config
    │
    ▼
build_model_from_config()
    │
    ▼
TransportModel{Arch, Grid, ...}
    │
    ▼
run!(model)
    │
    ▼
_run_loop!(model, grid, buffering)
    │
    ├── IOScheduler: load met data (CPU → GPU transfer)
    ├── compute_air_mass_phase!: DELP × (1-QV) × area / g
    ├── apply_emissions_phase!: surface flux injection
    ├── advection_phase!: Strang split or Lin-Rood + vertical remap
    ├── post_advection_physics!: convection, diffusion, chemistry
    ├── apply_mass_correction!: global mass fixer
    └── write_output!: binary or NetCDF output
```

## Extension Points

| Want to add... | Subtype this | Implement these methods | Register in |
|----------------|-------------|------------------------|-------------|
| New grid type | `AbstractGrid{FT,Arch}` | `xnode`, `ynode`, `cell_area`, `Δx`, `Δy`, etc. | Phase functions in `physics_phases.jl` |
| New advection | `AbstractAdvectionScheme` | `advect!`, `adjoint_advect!` + directional variants | `configuration.jl` |
| New convection | `AbstractConvection` | `convect!`, `adjoint_convect!` | `configuration.jl` |
| New diffusion | `AbstractDiffusion` | `diffuse!`, `adjoint_diffuse!` | `configuration.jl` |
| New chemistry | `AbstractChemistry` | `apply_chemistry!`, `adjoint_apply_chemistry!` | `configuration.jl` |
| New met source | `AbstractMetDriver{FT}` | `total_windows`, `load_met_window!`, etc. | `configuration.jl` + TOML mapping |
| New output | `AbstractOutputWriter` | `write_output!`, `finalize_output!` | `configuration.jl` |
| New emissions | `AbstractSurfaceFlux` | `apply_surface_flux!` | `configuration.jl` |

## GPU Architecture

All GPU kernels use KernelAbstractions.jl (`@kernel`, `@index`):

```julia
@kernel function my_kernel!(output, input, param)
    i, j = @index(Global, NTuple)
    @inbounds output[i, j] = input[i, j] * param
end

# Launch on any backend (CPU, CUDA, Metal)
backend = get_backend(output)
kernel! = my_kernel!(backend, 256)
kernel!(output, input, param; ndrange=(Nx, Ny))
```

GPU extensions are loaded via weak dependencies (`ext/AtmosTransportCUDAExt.jl`,
`ext/AtmosTransportMetalExt.jl`). The core package never imports CUDA or Metal directly.

## TransportPolicy

Central configuration object (see `src/Models/transport_policy.jl`):

```julia
TransportPolicy(
    vertical_operator = :continuity_cm,  # or :pressure_remap
    pressure_basis    = :dry,            # or :moist
    mass_balance_mode = :global_fixer    # or :none, :column
)
```

Resolved from TOML config via `resolve_transport_policy(metadata)`, with backward
compatibility for legacy boolean flags (`vertical_remap`, `dry_correction`, `mass_fixer`).
