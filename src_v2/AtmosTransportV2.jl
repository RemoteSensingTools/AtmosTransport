"""
    AtmosTransportV2

Dry-mass face-flux transport architecture for AtmosTransport.jl.

This is a parallel development module (`src_v2/`) that implements the
restructured architecture from the design memo. It coexists with the
production `src/` code — all existing functionality continues to work.

## Architecture overview

```
Grids (geometry)  →  State (CellState, AbstractFaceFluxState)
                          ↓
MetDrivers  → build_dry_fluxes! → AbstractFaceFluxState
                          ↓
Operators   → apply!(CellState, AbstractFaceFluxState, AtmosGrid, scheme, dt)
                          ↓
Kernels     → CellKernels / FaceKernels / ColumnKernels
```

## Design principles

1. **Dry face mass fluxes** are the formal interface between meteorology and transport.
2. **Transport operators** receive only `CellState + AbstractFaceFluxState + AtmosGrid` — never
   raw winds, humidity, or met-specific variables.
3. **Geometry is face/cell oriented**, not index-direction oriented, enabling support
   for reduced Gaussian grids alongside structured lat-lon and cubed-sphere.
4. **Multiple dispatch** on four orthogonal axes: grid geometry, backend, met driver,
   numerical operator.
5. **KernelAbstractions** for CPU/GPU portability (same code, no branching).

## Module loading order (strict, no circular deps)

1. Grids
2. State (depends on Grids for mesh types)
3. Operators (depends on State and Grids)
4. MetDrivers (depends on State and Grids)
5. Kernels (standalone utility patterns)
"""
module AtmosTransportV2

using KernelAbstractions

# ---- Geometry ----
include("Grids/Grids.jl")
using .Grids

# ---- State containers ----
include("State/State.jl")
using .State

# ---- Physics operators ----
include("Operators/Operators.jl")
using .Operators

# ---- Met-data adapters ----
include("MetDrivers/MetDrivers.jl")
using .MetDrivers

# ---- Kernel patterns ----
include("Kernels/Kernels.jl")
using .Kernels

# ---- Re-exports for convenience ----

# Grids
export AtmosGrid, LatLonMesh, CubedSphereMesh, ReducedGaussianMesh
export HybridSigmaPressure
export AbstractFluxTopology, StructuredFluxTopology, FaceIndexedFluxTopology, flux_topology
export ncells, nfaces, nlevels, cell_area, nx, ny
export n_levels, pressure_at_interface, pressure_at_level, level_thickness

# State
export CellState
export AbstractFaceFluxState, AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export MetState, allocate_face_fluxes, allocate_tracers
export mixing_ratio, total_mass, total_air_mass, tracer_names

# Operators
export AbstractAdvection, RussellLernerAdvection, PPMAdvection
export AdvectionWorkspace, strang_split!, apply!
export diagnose_cm!

# MetDrivers
export AbstractMetDriver, PreprocessedERA5Driver
export build_dry_fluxes!, build_air_mass!
export DiagnoseVerticalFromHorizontal, PressureTendencyClosure
export supports_diffusion, supports_convection

end # module AtmosTransportV2
