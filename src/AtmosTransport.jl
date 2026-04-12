"""
    AtmosTransport

Basis-explicit face-flux transport architecture.

Mesh-generic transport core with explicit mass-basis tags, topology-aware
grids, and multiple-dispatch operator selection.

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
module AtmosTransport

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
                   TransportBinaryDriver, AbstractTransportWindow,
                   StructuredFluxDeltas, FaceIndexedFluxDeltas,
                   StructuredTransportWindow, FaceIndexedTransportWindow,
                   load_window!, load_qv_window!, load_flux_delta_window!,
                   load_qv_pair_window!, load_grid, load_transport_window,
                   driver_grid, air_mass_basis, has_humidity_endpoints,
                   interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!,
                   load_cmfmc_window!, load_surface_window!, load_tm5conv_window!,
                   load_temperature_window!,
                   window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc,
                   has_surface, has_tm5conv, has_temperature,
                   grid_type, horizontal_topology,
                   source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics,
                   A_ifc, B_ifc,
                   diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!,
                   diagnose_cm_from_continuity_ka!,
                   ERA5ReducedGaussianGeometry,
                   read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh,
                   build_dry_fluxes!, build_air_mass!,
                   total_windows, window_dt, steps_per_window, supports_diffusion, supports_convection,
                   DiagnoseVerticalFromHorizontal, PressureTendencyClosure,
                   CubedSphereBinaryReader, CubedSphereBinaryHeader,
                   load_cs_window, cs_window_count

# ---- Physics operators ----
include("Operators/Operators.jl")
using .Operators

# ---- Kernel patterns ----
include("Kernels/Kernels.jl")
using .Kernels

# ---- Minimal runtime/model layer ----
include("Models/Models.jl")
using .Models

# ---- Offline regridding glue (preprocessing only; hard deps on CR.jl + JLD2) ----
include("Regridding/Regridding.jl")
using .Regridding

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
export PanelEdge, PanelConnectivity, default_panel_connectivity, reciprocal_edge
export EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
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

# Operators -- public transport API (legacy hierarchy)
export AbstractAdvection, AbstractConstantReconstruction, AbstractLinearReconstruction, AbstractQuadraticReconstruction
export UpwindAdvection, FirstOrderUpwindAdvection, RussellLernerAdvection, PPMAdvection
export AdvectionWorkspace, strang_split!, strang_split_mt!, apply!
export TracerView
export diagnose_cm!

# Operators -- new scheme hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter
export UpwindScheme, SlopesScheme, PPMScheme
export reconstruction_order

# Chemistry
export AbstractChemistry, NoChemistry, RadioactiveDecay, CompositeChemistry
export apply_chemistry!

# Cubed-sphere advection
export fill_panel_halos!, strang_split_cs!, CSAdvectionWorkspace

# NOTE: sweep_x!, sweep_y!, sweep_z!, max_cfl_*, minmod, van_leer_slope
# are intentionally NOT exported. They accept raw arrays without basis
# checking and are implementation details of strang_split!.
# Access via AtmosTransport.Operators.Advection.sweep_x! if needed.

# MetDrivers
export AbstractDriver, AbstractClosure
export AbstractMetDriver, PreprocessedERA5Driver
export ERA5BinaryReader, ERA5BinaryHeader
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export TransportBinaryDriver, AbstractTransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas
export StructuredTransportWindow, FaceIndexedTransportWindow
export load_window!, load_qv_window!, load_flux_delta_window!
export load_qv_pair_window!, load_grid, load_transport_window
export driver_grid, air_mass_basis, has_humidity_endpoints
export interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc
export has_surface, has_tm5conv, has_temperature
export grid_type, horizontal_topology
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
export A_ifc, B_ifc
export build_dry_fluxes!, build_air_mass!
export total_windows, window_dt, steps_per_window
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh
export DiagnoseVerticalFromHorizontal, PressureTendencyClosure
export supports_diffusion, supports_convection
export CubedSphereBinaryReader, CubedSphereBinaryHeader
export load_cs_window, cs_window_count

# Models
export TransportModel, Simulation, DrivenSimulation, SurfaceFluxSource
export step!, run!, run_window!
export window_index, substep_index, current_qv

# Offline regridding (preprocessing only)
export build_regridder, save_regridder, load_regridder
export save_esmf_weights, apply_regridder!
export cubed_sphere_face_corners

end # module AtmosTransport
