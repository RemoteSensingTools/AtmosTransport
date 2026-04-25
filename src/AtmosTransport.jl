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

# ---- Diagnostic NetCDF output ----
include("Output/Output.jl")
using .Output

# ---- Met-data adapters ----
include("MetDrivers/MetDrivers.jl")
using .MetDrivers: AbstractDriver, AbstractClosure, AbstractMetDriver,
                   PreprocessedERA5Driver, current_time,
                   ERA5BinaryReader, ERA5BinaryHeader,
                   TransportBinaryReader, TransportBinaryHeader, write_transport_binary,
                   TransportBinaryDriver, AbstractTransportWindow,
                   StructuredFluxDeltas, FaceIndexedFluxDeltas,
                   StructuredTransportWindow, FaceIndexedTransportWindow,
                   CubedSphereTransportWindow, CubedSphereTransportDriver,
                   load_window!, load_qv_window!, load_flux_delta_window!,
                   load_qv_pair_window!, load_grid, load_transport_window,
                   driver_grid, air_mass_basis, has_humidity_endpoints,
                   interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!,
                   load_cmfmc_window!, load_surface_window!, load_tm5conv_window!,
                   load_temperature_window!,
                   ConvectionForcing, has_convection_forcing,
                   copy_convection_forcing!, allocate_convection_forcing_like,
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
                   load_cs_window, cs_window_count, mesh_convention,
                   StreamingTransportBinaryWriter,
                   open_streaming_transport_binary, write_streaming_window!,
                   close_streaming_transport_binary!,
                   binary_capabilities, inspect_binary   # plan 40 Commit 5

# ---- Physics operators ----
include("Operators/Operators.jl")
using .Operators

# ---- Kernel patterns ----
include("Kernels/Kernels.jl")
using .Kernels

# ---- Offline regridding glue (CR.jl + JLD2) ----
# Loaded before Models so `Models.InitialConditionIO` (plan 40 Commit 1c) can
# directly `using ..Regridding` and `using ..Preprocessing.CSHelpers` for the
# CS file-based IC path. Regridding and Preprocessing have no back-references
# to Models (verified by grep), so reordering is safe.
include("Regridding/Regridding.jl")
using .Regridding

# ---- Preprocessing pipeline (spectral/gridded → transport binary) ----
include("Preprocessing/Preprocessing.jl")
using .Preprocessing

# ---- Topology-aware snapshot visualization data layer ----
include("Visualization/Visualization.jl")
using .Visualization

# ---- Minimal runtime/model layer ----
include("Models/Models.jl")
using .Models

# ---- Download pipeline (TOML-driven met/emissions data download) ----
include("Downloads/Downloads.jl")
using .DataDownloads

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
export PanelEdge, PanelConnectivity
export default_panel_connectivity, gnomonic_panel_connectivity, panel_connectivity_for
export panel_cell_center_lonlat, panel_cell_corner_lonlat, panel_cell_local_tangent_basis
export reciprocal_edge
export EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
export n_levels, pressure_at_interface, pressure_at_level, level_thickness

# State -- types and basis tags
export AbstractMassBasis, MoistBasis, DryBasis, mass_basis
export AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis, flux_basis
export DryCellState, MoistCellState
export DryCubedSphereState, MoistCubedSphereState
export DryStructuredFluxState, MoistStructuredFluxState
export DryCubedSphereFluxState, MoistCubedSphereFluxState
export CellState
export CubedSphereState
export AbstractFaceFluxState, AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState, CubedSphereFaceFluxState, FluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export MetState, allocate_face_fluxes, allocate_tracers
export mixing_ratio, total_mass, total_air_mass, tracer_names
export ntracers, tracer_index, tracer_name, get_tracer, eachtracer

# Output diagnostics
export SnapshotFrame, SnapshotWriteOptions
export capture_snapshot, write_snapshot_netcdf
export column_mean_mixing_ratio, layer_mass_per_area, column_mass_per_area

# Time-varying field abstraction (plan 16a, extended in 16b/17)
export AbstractTimeVaryingField, AbstractCubedSphereField,
       ConstantField, ProfileKzField, PreComputedKzField, CubedSphereField,
       DerivedKzField, PBLPhysicsParameters, StepwiseField,
       field_value, update_field!, integral_between, panel_field

# Operators -- public transport API
export AdvectionWorkspace, strang_split!, strang_split_mt!, apply!
export TracerView
export diagnose_cm!

# Diffusion solver infrastructure + operator types (plan 16b Commits 2-4)
export solve_tridiagonal!, build_diffusion_coefficients
export AbstractDiffusionOperator, NoDiffusion, ImplicitVerticalDiffusion
export apply_vertical_diffusion!

# Surface flux data types + operator hierarchy (plan 17 Commits 2-3).
# SurfaceFluxSource is re-exported above (from Models) for backward compat.
export PerTracerFluxMap, flux_for
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!

# Convection operator hierarchy (plan 18 + plan 23).
# NoConvection, CMFMCConvection live since plan 18; TM5Convection
# lands via plan 23 (Commit 1: types + dispatch stubs; Commit 4:
# real kernels on all three topologies).
# `ConvectionForcing` lives in MetDrivers and is re-exported below.
export AbstractConvectionOperator, NoConvection
export CMFMCConvection                          # plan 18 Commit 3
export CMFMCWorkspace, invalidate_cmfmc_cache!  # plan 18 Commit 3
export TM5Convection                            # plan 23 Commit 1
export TM5Workspace                             # plan 23 Commit 1
export apply_convection!

# Advection scheme hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme
export reconstruction_order, required_halo_width

# Chemistry
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export chemistry_block!

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs!, CSAdvectionWorkspace

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, CSLinRoodAdvectionWorkspace
export fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export fillz_q!, apply_divergence_damping_cs!

# NOTE: sweep_x!, sweep_y!, sweep_z!, max_cfl_*
# are intentionally NOT exported. They accept raw arrays without basis
# checking and are implementation details of strang_split!.
# Access via AtmosTransport.Operators.Advection.sweep_x! if needed.

# MetDrivers
export AbstractDriver, AbstractClosure
export AbstractMetDriver, PreprocessedERA5Driver
export current_time
export ERA5BinaryReader, ERA5BinaryHeader
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export TransportBinaryDriver, AbstractTransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas
export StructuredTransportWindow, FaceIndexedTransportWindow
export CubedSphereTransportWindow, CubedSphereTransportDriver
export load_window!, load_qv_window!, load_flux_delta_window!
export load_qv_pair_window!, load_grid, load_transport_window
export driver_grid, air_mass_basis, has_humidity_endpoints
export interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export ConvectionForcing, has_convection_forcing
export copy_convection_forcing!, allocate_convection_forcing_like
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
export load_cs_window, cs_window_count, mesh_convention
export StreamingTransportBinaryWriter
export open_streaming_transport_binary, write_streaming_window!, close_streaming_transport_binary!

# Models
export TransportModel, Simulation, DrivenSimulation, SurfaceFluxSource
export step!, run!, run_window!
export window_index, substep_index, current_qv
export with_chemistry, with_diffusion, with_emissions
export with_convection, with_convection_forcing
export RuntimePhysicsRecipe, CSPhysicsRecipe
export build_runtime_advection, build_runtime_diffusion, build_runtime_convection
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export configured_halo_width
export build_cs_advection, build_cs_diffusion, build_cs_convection
export build_cs_physics_recipe, validate_cs_physics_recipe
export configured_cs_halo_width
export FileInitialConditionSource, build_initial_mixing_ratio, pack_initial_tracer_mass
export FileSurfaceFluxField, build_surface_flux_source, build_surface_flux_sources
export expand_binary_paths
export binary_capabilities, inspect_binary   # plan 40 Commit 5
export run_driven_simulation, TransportTracerSpec   # plan 40 Commit 6a

# Offline regridding (preprocessing only)
export build_regridder, save_regridder, load_regridder
export save_esmf_weights, apply_regridder!
export cubed_sphere_face_corners

# Visualization data layer. Makie plotting methods load from the optional
# AtmosTransportMakieExt extension when a Makie backend is available.
export SnapshotDataset, HorizontalField, RasterField, SnapshotRegridCache, PlotSpec
export open_snapshot, available_variables, snapshot_times, snapshot_topology
export fieldview, frame_indices, as_raster, robust_colorrange
export mapplot, mapplot!, snapshot_grid, movie, movie_grid

end # module AtmosTransport
