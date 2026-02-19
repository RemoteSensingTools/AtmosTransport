"""
    Sources

Surface emission and flux injection for atmospheric tracers.

Supports gridded emission inventories (EDGAR, etc.) with conservative
regridding from the inventory resolution to the model grid.

# Interface

    apply_surface_flux!(tracers, source, grid, dt)
"""
module Sources

using DocStringExtensions
using NCDatasets
using ..Grids: AbstractGrid, LatitudeLongitudeGrid, cell_area, Δz, floattype
using ..Grids: grid_size

export AbstractSource, GriddedEmission, NoEmission
export load_edgar_co2, regrid_emissions!, apply_surface_flux!

"""
$(TYPEDEF)

Supertype for surface emission and flux sources.
"""
abstract type AbstractSource end

"""
$(TYPEDEF)

No emission (pass-through).
"""
struct NoEmission <: AbstractSource end
apply_surface_flux!(tracers, ::NoEmission, grid, dt) = nothing

include("gridded_emission.jl")
include("edgar_reader.jl")

end # module Sources
