# ---------------------------------------------------------------------------
# Typed emission inventory sources for intercomparison studies
#
# Each inventory is a concrete type carrying metadata (version, species, sector).
# `load_inventory` dispatches on (inventory type, grid type) to produce either
# a `SurfaceFlux` (lat-lon) or `SurfaceFlux` (cubed-sphere).
#
# This enables clean multi-inventory comparisons (e.g. CATRINE project):
#   sources = [
#       load_inventory(EdgarSource(; version="v8.0"), grid; year=2022),
#       load_inventory(GFASSource(), grid; year=2022),
#   ]
# ---------------------------------------------------------------------------

using ..Grids: CubedSphereGrid
using JSON3

"""
$(TYPEDEF)

Supertype for named emission inventories. Each subtype carries metadata
about the inventory (version, default file paths, species, sector info).

Use `load_inventory(source, grid; kwargs...)` to load and regrid.
"""
abstract type AbstractInventorySource <: AbstractSurfaceFlux end

# No-op apply — inventories are loaded via load_inventory, not applied directly
apply_surface_flux!(tracers, ::AbstractInventorySource, grid, dt) = nothing

"""
$(TYPEDEF)

EDGAR (Emissions Database for Global Atmospheric Research) inventory.
Anthropogenic fossil + industrial CO2 emissions.

$(FIELDS)
"""
struct EdgarSource <: AbstractInventorySource
    "EDGAR version string (e.g. \"v8.0\")"
    version  :: String
    "path to EDGAR NetCDF file (optional — can be passed to load_inventory)"
    filepath :: String
    "species symbol (e.g. :co2, :ch4)"
    species  :: Symbol
end

EdgarSource(; version="v8.0", filepath="", species=:co2) =
    EdgarSource(version, filepath, species)

"""
$(TYPEDEF)

CarbonTracker Near-Real-Time (CT-NRT) biosphere + fire + ocean fluxes.

$(FIELDS)
"""
struct CarbonTrackerSource <: AbstractInventorySource
    version  :: String
    filepath :: String
end

CarbonTrackerSource(; version="CT2022", filepath="") =
    CarbonTrackerSource(version, filepath)

"""
$(TYPEDEF)

GFAS (Global Fire Assimilation System) fire emission inventory.

$(FIELDS)
"""
struct GFASSource <: AbstractInventorySource
    version  :: String
    filepath :: String
end

GFASSource(; version="1.2", filepath="") = GFASSource(version, filepath)

"""
$(TYPEDEF)

Jena CarboScope ocean CO2 flux.

$(FIELDS)
"""
struct JenaCarboScopeSource <: AbstractInventorySource
    version  :: String
    filepath :: String
end

JenaCarboScopeSource(; version="oc_v2024", filepath="") =
    JenaCarboScopeSource(version, filepath)

"""
$(TYPEDEF)

CATRINE intercomparison inventory (placeholder for future datasets).
See: https://www.catrine-project.eu/

$(FIELDS)
"""
struct CATRINESource <: AbstractInventorySource
    dataset  :: String
    version  :: String
    filepath :: String
end

CATRINESource(; dataset="default", version="1.0", filepath="") =
    CATRINESource(dataset, version, filepath)

# =====================================================================
# load_inventory — multi-dispatch on (inventory, grid)
# =====================================================================

"""
    load_inventory(source, grid; kwargs...) → AbstractSurfaceFlux

Load an emission inventory and regrid to `grid`. Dispatches on both
the inventory type and the grid type:

    load_inventory(::EdgarSource, ::LatitudeLongitudeGrid; year, file)  → SurfaceFlux
    load_inventory(::EdgarSource, ::CubedSphereGrid; year, file)        → SurfaceFlux
"""
function load_inventory end

# --- EDGAR on lat-lon grid ---
function load_inventory(src::EdgarSource, grid::LatitudeLongitudeGrid{FT};
                        year::Int=2022, file::String=src.filepath) where FT
    filepath = isempty(file) ? _default_edgar_path(year, src.species) : expanduser(file)
    return load_edgar_co2(filepath, grid; year, species=src.species)
end

# --- EDGAR on cubed-sphere grid ---
function load_inventory(src::EdgarSource, grid::CubedSphereGrid{FT};
                        year::Int=2022, file::String=src.filepath,
                        binary_file::String="",
                        file_lons=nothing, file_lats=nothing) where FT
    Nc = grid.Nc
    ft_tag = FT == Float32 ? "float32" : "float64"

    # Try preprocessed binary first
    sp_tag = lowercase(string(src.species))
    bin_path = if !isempty(binary_file)
        expanduser(binary_file)
    else
        joinpath(homedir(), "data", "emissions", "edgar_v8",
                 "edgar_$(sp_tag)_cs_c$(Nc)_$(ft_tag).bin")
    end

    if isfile(bin_path)
        @info "  Loading preprocessed EDGAR binary: $bin_path"
        flux_panels = load_edgar_cs_binary(bin_path, FT)
    else
        # Fall back to regridding from NetCDF
        filepath = isempty(file) ? _default_edgar_path(year, src.species) : expanduser(file)
        @info "  Regridding EDGAR from NetCDF to C$Nc..."
        isfile(filepath) || error("EDGAR file not found: $filepath")
        ds = NCDataset(filepath)
        edgar_lons = FT.(ds["lon"][:])
        edgar_lats = FT.(ds["lat"][:])
        edgar_flux = FT.(replace(ds["emissions"][:, :], missing => zero(FT)))
        close(ds)
        flux_panels = regrid_edgar_to_cs(edgar_flux, edgar_lons, edgar_lats, grid;
                                          file_lons, file_lats)
        @info "  EDGAR regridded to C$Nc"
    end

    sf = SurfaceFlux(flux_panels, src.species,
        "EDGAR $(src.version) $(uppercase(string(src.species))) $year")
    log_flux_integral(sf, grid)
    return sf
end

# --- CarbonTracker on lat-lon grid ---
function load_inventory(src::CarbonTrackerSource, grid::LatitudeLongitudeGrid{FT};
                        year::Int=2022, file::String=src.filepath) where FT
    filepath = isempty(file) ? _default_ct_path() : expanduser(file)
    return load_carbontracker_fluxes(filepath, grid)
end

# --- GFAS on lat-lon grid ---
function load_inventory(src::GFASSource, grid::LatitudeLongitudeGrid{FT};
                        year::Int=2022, file::String=src.filepath) where FT
    filepath = isempty(file) ? _default_gfas_path(year) : expanduser(file)
    return load_gfas_fire_flux(filepath, grid)
end

# --- Jena CarboScope on lat-lon grid ---
function load_inventory(src::JenaCarboScopeSource, grid::LatitudeLongitudeGrid{FT};
                        year::Int=2022, file::String=src.filepath) where FT
    filepath = isempty(file) ? _default_jena_path() : expanduser(file)
    return load_jena_ocean_flux(filepath, grid)
end

# CATRINE load_inventory dispatch is in catrine_reader.jl

# =====================================================================
# EDGAR cubed-sphere binary reader (self-contained, no IO dependency)
# =====================================================================

const _EDGAR_HEADER_SIZE = 4096

"""
    load_edgar_cs_binary(bin_path, FT) → NTuple{6, Matrix{FT}}

Read a preprocessed EDGAR emission file on cubed-sphere panels.
File layout: [4096-byte JSON header | panel₁ data | … | panel₆ data]
"""
function load_edgar_cs_binary(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, _EDGAR_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), _EDGAR_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))
    Nc = Int(hdr.Nc)
    flux_panels = ntuple(6) do _
        arr = Array{FT}(undef, Nc, Nc)
        read!(io, arr)
        arr
    end
    close(io)
    return flux_panels
end

# =====================================================================
# Default file path helpers
# =====================================================================

"""
    log_flux_integral(sf::SurfaceFlux, grid)

Compute and log the global integral of a SurfaceFlux in useful units.
For CS grids: sum(flux × Aᶜ) over all 6 panels.
For lat-lon grids: sum(flux × cell_areas).
"""
function log_flux_integral(sf::SurfaceFlux{CubedSphereLayout}, grid::CubedSphereGrid)
    total_kg_s = zero(Float64)
    for p in 1:6
        total_kg_s += sum(Float64.(sf.flux[p]) .* Float64.(grid.Aᶜ[p]))
    end
    _print_flux_integral(sf.label, sf.species, total_kg_s)
end

function log_flux_integral(sf::SurfaceFlux{LatLonLayout}, grid::LatitudeLongitudeGrid)
    areas = latlon_cell_areas(Float64.(grid.λᶠ), Float64.(grid.φᶠ))
    total_kg_s = sum(Float64.(sf.flux) .* areas)
    _print_flux_integral(sf.label, sf.species, total_kg_s)
end

function log_flux_integral(sf::TimeVaryingSurfaceFlux{CubedSphereLayout}, grid::CubedSphereGrid)
    # Log integral of first snapshot
    panels = sf.flux_data[1]
    total_kg_s = zero(Float64)
    for p in 1:6
        total_kg_s += sum(Float64.(panels[p]) .* Float64.(grid.Aᶜ[p]))
    end
    _print_flux_integral("$(sf.label) [t=1]", sf.species, total_kg_s)
end

function _print_flux_integral(label, species, total_kg_s)
    total_per_yr = total_kg_s * SECONDS_PER_YEAR
    if species === :co2 || species === :fossil_co2
        @info "  Flux integral: $label → $(round(total_per_yr / 1e12, digits=3)) PgCO2/yr " *
              "($(round(total_per_yr / 1e12 / KGC_TO_KGCO2, digits=3)) PgC/yr)"
    elseif species === :sf6
        @info "  Flux integral: $label → $(round(total_per_yr / 1e6, digits=3)) kt/yr"
    elseif species === :rn222
        @info "  Flux integral: $label → $(round(total_per_yr, digits=3)) kg/yr"
    else
        @info "  Flux integral: $label → $(round(total_per_yr, sigdigits=4)) kg/yr"
    end
end

function _default_edgar_path(year::Int, species::Symbol=:co2)
    sp = uppercase(string(species))
    joinpath(homedir(), "data", "emissions", "edgar_v8",
             "v8.0_FT2022_GHG_$(sp)_$(year)_TOTALS_emi.nc")
end

function _default_ct_path()
    joinpath(homedir(), "data", "emissions", "carbontracker",
             "CT2022.flux1x1.nc")
end

function _default_gfas_path(year::Int)
    joinpath(homedir(), "data", "emissions", "gfas",
             "gfas_fire_$(year).nc")
end

function _default_jena_path()
    joinpath(homedir(), "data", "emissions", "jena",
             "oc_v2024_daily_1x1.nc")
end
