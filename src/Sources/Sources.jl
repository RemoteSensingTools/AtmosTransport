"""
    Sources

Surface flux injection for atmospheric tracers.

Supports gridded flux inventories (EDGAR, CT-NRT, GFAS, Jena CarboScope)
with conservative regridding from native resolution to the model grid.

Multi-component fluxes are handled via `CombinedFlux` (combines multiple
sources) and `TimeVaryingSurfaceFlux` (time-dependent fields like 3-hourly biosphere).

# Interface

    apply_surface_flux!(tracers, source, grid, dt)
"""
module Sources

using DocStringExtensions
using NCDatasets
using Dates
using ..Grids: AbstractGrid, LatitudeLongitudeGrid, CubedSphereGrid
using ..Grids: cell_area, Δz, floattype, grid_size

export AbstractSurfaceFlux, SurfaceFlux, TimeVaryingSurfaceFlux
export LatLonLayout, CubedSphereLayout
export CombinedFlux, NoFlux
export update_time_index!, flux_data
export load_edgar_co2, load_carbontracker_fluxes
export load_jena_ocean_flux, load_gfas_fire_flux
export load_cams_co2, load_lmdz_co2, load_gridfed_fossil_co2, load_edgar_sf6, load_zhang_rn222
export regrid_emissions!, apply_surface_flux!
export apply_emissions_window!
export regrid_edgar_to_cs
export regrid_latlon_to_cs, build_latlon_to_cs_map, build_conservative_cs_map
export ConservativeCSMap
export conservative_regrid_ll
export remap_lon_neg180_to_0_360, ensure_south_to_north, normalize_lons_lats
export latlon_cell_areas, latlon_cell_area
export compute_areas_from_corners, generate_cs_gridspec
export M_AIR, M_CO2, M_SF6, M_RN222, molar_mass_for_species
export SECONDS_PER_YEAR, SECONDS_PER_MONTH, KG_PER_TONNE, KGC_TO_KGCO2
export tonnes_per_year_to_kgm2s, kgC_per_m2s_to_kgCO2_per_m2s, kgCO2_per_month_m2_to_kgm2s

# Inventory source types and load_inventory dispatch
export AbstractInventorySource
export EdgarSource, CarbonTrackerSource, GFASSource, JenaCarboScopeSource, CATRINESource
export load_inventory, log_flux_integral

# New type hierarchy (must come before files that use these types)
include("flux_types.jl")

# Emission unit conventions and Unitful-verified constants
include("emission_units.jl")

# Consolidated regridding utilities (before all readers)
include("regrid_utils.jl")

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
include("catrine_reader.jl")

end # module Sources
