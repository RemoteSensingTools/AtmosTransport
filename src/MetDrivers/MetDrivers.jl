"""
    MetDrivers

Meteorological data adapters for the basis-explicit transport architecture.

Provides:
- Abstract driver types with capability traits
- Mass closure strategies (diagnose cm, pressure tendency, native vertical)
- topology-generic transport binary readers
- ERA5 preprocessed-binary readers and native reduced-Gaussian GRIB geometry helpers
"""
module MetDrivers

using Adapt
using Printf: @sprintf
using ..Architectures: array_adapter_for
using ..State
using ..Grids

include("AbstractMetDriver.jl")
include("MassClosure.jl")
include("DryFluxBuilder.jl")
include("ConvectionForcing.jl")
include("SurfaceForcing.jl")
include("TransportBinary.jl")
include("ReplayContinuity.jl")
include("TransportBinaryDriver.jl")
include("CubedSphereBinaryReader.jl")
include("CubedSphereTransportDriver.jl")
include("ERA5/ERA5.jl")
using .ERA5

# ERA5.BinaryReader lives in a nested module. Names that already exist in
# MetDrivers (for CS/generic transport readers) are not imported by `using
# .ERA5`, so add forwarding methods on the public MetDrivers generics.
window_count(r::ERA5BinaryReader) = ERA5.window_count(r)
has_qv(r::ERA5BinaryReader) = ERA5.has_qv(r)
has_flux_delta(r::ERA5BinaryReader) = ERA5.has_flux_delta(r)
has_cmfmc(r::ERA5BinaryReader) = ERA5.has_cmfmc(r)
has_surface(r::ERA5BinaryReader) = ERA5.has_surface(r)
has_tm5conv(r::ERA5BinaryReader) = ERA5.has_tm5conv(r)
has_temperature(r::ERA5BinaryReader) = ERA5.has_temperature(r)
mass_basis(r::ERA5BinaryReader) = ERA5.mass_basis(r)
A_ifc(r::ERA5BinaryReader) = ERA5.A_ifc(r)
B_ifc(r::ERA5BinaryReader) = ERA5.B_ifc(r)
load_window!(r::ERA5BinaryReader, win::Int; kwargs...) =
    ERA5.load_window!(r, win; kwargs...)
load_qv_window!(r::ERA5BinaryReader, win::Int; kwargs...) =
    ERA5.load_qv_window!(r, win; kwargs...)
load_flux_delta_window!(r::ERA5BinaryReader, win::Int; kwargs...) =
    ERA5.load_flux_delta_window!(r, win; kwargs...)
load_surface_window!(r::ERA5BinaryReader, win::Int; kwargs...) =
    ERA5.load_surface_window!(r, win; kwargs...)

# Re-export reader and adapter types
export PreprocessedERA5Driver
export ERA5BinaryReader, ERA5BinaryHeader
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export TransportBinaryContract, canonical_window_constant_contract,
       validate_transport_contract!
export StreamingTransportBinaryWriter
export open_streaming_transport_binary, write_streaming_window!, close_streaming_transport_binary!
export TransportBinaryDriver, AbstractTransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas
export StructuredTransportWindow, FaceIndexedTransportWindow
export CubedSphereTransportWindow, CubedSphereTransportDriver
export load_window!, load_qv_window!, load_flux_delta_window!
export load_tm5_convection_window!, has_tm5_convection
export load_qv_pair_window!, load_grid, load_transport_window
export driver_grid, air_mass_basis, has_humidity_endpoints
export interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export ConvectionForcing, has_convection_forcing
export copy_convection_forcing!, allocate_convection_forcing_like
export PBLSurfaceForcing, has_pbl_surface_forcing
export window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc
export binary_capabilities, inspect_binary   # plan 40 Commit 5
export has_surface, has_tm5conv, has_temperature
export mass_basis, grid_type, horizontal_topology, A_ifc, B_ifc
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export recompute_cm_from_dm_target!, recompute_faceindexed_cm_from_dm_target!
export verify_window_continuity, verify_window_continuity_ll, verify_window_continuity_rg,
       verify_window_continuity_cs
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh

# Cubed-sphere binary reader
export CubedSphereBinaryReader, CubedSphereBinaryHeader
export load_cs_window, cs_window_count, mesh_convention, mesh_definition

end # module MetDrivers
