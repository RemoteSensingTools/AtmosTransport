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

end # module MetDrivers
