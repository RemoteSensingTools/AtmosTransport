"""
    AtmosTransportV2

Basis-explicit face-flux transport architecture for AtmosTransport.jl.

This is a parallel development module (`src_v2/`) for a basis-explicit,
mesh-generic transport core. It coexists with the production `src/` code
while the future runtime architecture is developed and validated.

## Architecture overview

```
Grids (geometry)  ->  State (CellState, AbstractFaceFluxState)
                          ↓
Drivers / adapters -> basis-aware CellState + FluxState
                          ↓
Operators   -> apply!(CellState, AbstractFaceFluxState, AtmosGrid, scheme, dt)
                          ↓
Kernels     -> CellKernels / FaceKernels / ColumnKernels
```

## Design principles

1. **Explicit mass-basis tags** are part of the state/flux contract.
2. **Transport operators** receive only `CellState + AbstractFaceFluxState + AtmosGrid` -- never
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

# ---- Architecture and planetary constants ----
include("Architectures.jl")
using .Architectures
const AbstractArchitecture = Architectures.AbstractArchitecture
const CPU = Architectures.CPU
const GPU = Architectures.GPU

include("Parameters/Parameters.jl")
using .Parameters

# ---- Geometry ----
include("Grids/Grids.jl")
using .Grids

# ---- State containers ----
include("State/State.jl")
using .State

# ---- Met-data adapters ----
include("MetDrivers/MetDrivers.jl")
using .MetDrivers: AbstractDriver, AbstractClosure, AbstractMetDriver,
                   PreprocessedERA5Driver,
                   ERA5BinaryReader, ERA5BinaryHeader,
                   TransportBinaryReader, TransportBinaryHeader, write_transport_binary,
                   load_window!, load_qv_window!, load_flux_delta_window!,
                   load_qv_pair_window!, load_grid,
                   load_cmfmc_window!, load_surface_window!, load_tm5conv_window!,
                   load_temperature_window!,
                   window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc,
                   has_surface, has_tm5conv, has_temperature,
                   grid_type, horizontal_topology,
                   A_ifc, B_ifc,
                   diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!,
                   diagnose_cm_from_continuity_ka!,
                   ERA5ReducedGaussianGeometry,
                   read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh,
                   build_dry_fluxes!, build_air_mass!,
                   supports_diffusion, supports_convection,
                   DiagnoseVerticalFromHorizontal, PressureTendencyClosure

# ---- Physics operators ----
include("Operators/Operators.jl")
using .Operators

# ---- Kernel patterns ----
include("Kernels/Kernels.jl")
using .Kernels

# ---- Minimal runtime/model layer ----
include("Models/Models.jl")
using .Models

# ---- Re-exports for convenience ----

# Architectures and parameters
export AbstractArchitecture, CPU, GPU
export array_type, device, architecture
export PlanetParameters, earth_parameters

# Grids
export AtmosGrid, LatLonMesh, CubedSphereMesh, ReducedGaussianMesh
export AbstractCubedSpherePanelConvention
export GnomonicPanelConvention, GEOSNativePanelConvention
export HybridSigmaPressure
export AbstractFluxTopology, StructuredFluxTopology, FaceIndexedFluxTopology, flux_topology
export StructuredTopology, FaceConnectedTopology
export ncells, nfaces, nlevels, cell_area, cell_faces, nx, ny, nrings, nboundaries, cell_areas_by_latitude
export face_length, face_normal, face_cells, floattype
export planet_parameters, radius, gravity, reference_pressure
export ring_cell_count, ring_longitudes, cell_index
export boundary_face_count, boundary_face_offset, boundary_face_range
export panel_count, panel_convention, panel_labels
export n_levels, pressure_at_interface, pressure_at_level, level_thickness

# State -- types and basis tags
export AbstractMassBasis, MoistBasis, DryBasis, mass_basis
export AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis, flux_basis
export DryCellState, MoistCellState
export DryStructuredFluxState, MoistStructuredFluxState
export CellState
export AbstractFaceFluxState, AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState, FluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export MetState, allocate_face_fluxes, allocate_tracers
export mixing_ratio, total_mass, total_air_mass, tracer_names

# Operators -- public transport API
export AbstractAdvection, FirstOrderUpwindAdvection, RussellLernerAdvection, PPMAdvection
export AdvectionWorkspace, strang_split!, apply!
export diagnose_cm!
# NOTE: sweep_x!, sweep_y!, sweep_z!, max_cfl_*, minmod, van_leer_slope
# are intentionally NOT exported. They accept raw arrays without basis
# checking and are implementation details of strang_split!.
# Access via AtmosTransportV2.Operators.Advection.sweep_x! if needed.

# MetDrivers
export AbstractDriver, AbstractClosure
export AbstractMetDriver, PreprocessedERA5Driver
export ERA5BinaryReader, ERA5BinaryHeader
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export load_window!, load_qv_window!, load_flux_delta_window!
export load_qv_pair_window!, load_grid
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc
export has_surface, has_tm5conv, has_temperature
export grid_type, horizontal_topology
export A_ifc, B_ifc
export build_dry_fluxes!, build_air_mass!
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh
export DiagnoseVerticalFromHorizontal, PressureTendencyClosure
export supports_diffusion, supports_convection

# Models
export TransportModel, Simulation, step!, run!

end # module AtmosTransportV2
