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
using ..Grids: AbstractGrid, LatitudeLongitudeGrid, CubedSphereGrid
using ..Grids: cell_area, Δz, floattype, grid_size

export AbstractSource, AbstractGriddedEmission, GriddedEmission, NoEmission
export CubedSphereEmission
export CompositeEmission, TimeVaryingEmission, update_time_index!
export load_edgar_co2, load_carbontracker_fluxes
export load_jena_ocean_flux, load_gfas_fire_flux
export regrid_emissions!, apply_surface_flux!
export apply_emissions_window!
export regrid_edgar_to_cs

# Inventory source types and load_inventory dispatch
export AbstractInventorySource
export EdgarSource, CarbonTrackerSource, GFASSource, JenaCarboScopeSource, CATRINESource
export load_inventory

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
include("cubed_sphere_emission.jl")
include("gpu_emission_kernels.jl")
include("emission_regridding.jl")
include("composite_emission.jl")
include("edgar_reader.jl")
include("carbontracker_reader.jl")
include("ocean_flux_reader.jl")
include("gfas_reader.jl")
include("inventory_sources.jl")

end # module Sources
