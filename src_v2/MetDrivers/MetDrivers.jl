"""
    MetDrivers (v2)

Meteorological data adapters for the dry-mass transport architecture.

Provides:
- Abstract driver types with capability traits
- Mass closure strategies (diagnose cm, pressure tendency, native vertical)
- `build_dry_fluxes!` interface — the key met→transport boundary
- ERA5 dry flux builder (preprocessed binary path)
"""
module MetDrivers

using ..State
using ..Grids

include("AbstractMetDriver.jl")
include("MassClosure.jl")
include("DryFluxBuilder.jl")
include("ERA5/ERA5.jl")
using .ERA5

# Re-export ERA5 types
export PreprocessedERA5Driver
export ERA5BinaryReader, ERA5BinaryHeader
export load_window!, load_qv_window!, load_flux_delta_window!
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export window_count, has_qv, has_flux_delta, has_cmfmc
export has_surface, has_tm5conv, has_temperature
export mass_basis, A_ifc, B_ifc
export diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!
export diagnose_cm_from_continuity_ka!

end # module MetDrivers
