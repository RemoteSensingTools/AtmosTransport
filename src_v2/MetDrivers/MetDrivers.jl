"""
    MetDrivers (v2)

Meteorological data adapters for the basis-explicit transport architecture.

Provides:
- Abstract driver types with capability traits
- Mass closure strategies (diagnose cm, pressure tendency, native vertical)
- topology-generic transport binary readers
- ERA5 preprocessed-binary readers and native reduced-Gaussian GRIB geometry helpers
"""
module MetDrivers

using Adapt
using ..State
using ..Grids

include("AbstractMetDriver.jl")
include("MassClosure.jl")
include("DryFluxBuilder.jl")
include("TransportBinary.jl")
include("TransportBinaryDriver.jl")
include("CubedSphereBinaryReader.jl")
include("ERA5/ERA5.jl")
using .ERA5

# Re-export reader and adapter types
export PreprocessedERA5Driver
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
export mass_basis, grid_type, horizontal_topology, A_ifc, B_ifc
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh

# Cubed-sphere binary reader
export CubedSphereBinaryReader, CubedSphereBinaryHeader
export load_cs_window, cs_window_count

end # module MetDrivers
