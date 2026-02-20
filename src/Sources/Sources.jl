"""
    Sources

Surface emission and flux injection for atmospheric tracers.

Supports gridded emission inventories (EDGAR, CT-NRT, GFAS, Jena CarboScope)
with conservative regridding from native resolution to the model grid.

Multi-component fluxes are handled via `CompositeEmission` (combines multiple
sources) and `TimeVaryingEmission` (time-dependent fields like 3-hourly biosphere).

# Interface

    apply_surface_flux!(tracers, source, grid, dt)
"""
module Sources

using DocStringExtensions
using NCDatasets
using Dates
using ..Grids: AbstractGrid, LatitudeLongitudeGrid, cell_area, Δz, floattype
using ..Grids: grid_size

export AbstractSource, GriddedEmission, NoEmission
export CompositeEmission, TimeVaryingEmission, update_time_index!
export load_edgar_co2, load_carbontracker_fluxes
export load_jena_ocean_flux, load_gfas_fire_flux
export regrid_emissions!, apply_surface_flux!

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
include("composite_emission.jl")
include("edgar_reader.jl")
include("carbontracker_reader.jl")
include("ocean_flux_reader.jl")
include("gfas_reader.jl")

end # module Sources
