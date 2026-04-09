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

using ..State
using ..Grids

include("AbstractMetDriver.jl")
include("MassClosure.jl")
include("DryFluxBuilder.jl")
include("TransportBinary.jl")
include("ERA5/ERA5.jl")
using .ERA5

# Re-export reader and adapter types
export PreprocessedERA5Driver
export ERA5BinaryReader, ERA5BinaryHeader
export TransportBinaryReader, TransportBinaryHeader, write_transport_binary
export load_window!, load_qv_window!, load_flux_delta_window!
export load_qv_pair_window!, load_grid
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export window_count, has_qv, has_qv_endpoints, has_flux_delta, has_cmfmc
export has_surface, has_tm5conv, has_temperature
export mass_basis, grid_type, horizontal_topology, A_ifc, B_ifc
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!
export ERA5ReducedGaussianGeometry
export read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh

end # module MetDrivers
