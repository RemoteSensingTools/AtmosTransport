# ---------------------------------------------------------------------------
# CompositeEmission — multiple flux components applied together
# TimeVaryingEmission — flux that changes with time (3-hourly, daily, etc.)
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Holds multiple emission sources that are applied together.
Useful for combining anthropogenic, biosphere, ocean, and fire CO2 fluxes.

$(FIELDS)
"""
struct CompositeEmission{S <: Tuple} <: AbstractSource
    "tuple of AbstractSource components"
    components :: S
    "human-readable label"
    label      :: String
end

function CompositeEmission(sources::AbstractSource...; label="composite")
    CompositeEmission(sources, label)
end

function apply_surface_flux!(tracers::NamedTuple, source::CompositeEmission,
                             grid::LatitudeLongitudeGrid, dt)
    for component in source.components
        apply_surface_flux!(tracers, component, grid, dt)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TimeVaryingEmission — time-dependent surface flux with periodic updates
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

A surface emission source whose flux field changes over time.
Stores a stack of flux snapshots and selects the appropriate one based on
elapsed simulation time.

$(FIELDS)
"""
mutable struct TimeVaryingEmission{FT, A <: AbstractArray{FT, 3}} <: AbstractSource
    "flux snapshots [kg/m²/s], shape (Nx, Ny, Nt_flux)"
    flux_stack  :: A
    "time axis: elapsed hours from simulation start for each snapshot"
    time_hours  :: Vector{Float64}
    "tracer name"
    species     :: Symbol
    "human-readable label"
    label       :: String
    "molar mass of emitted species [kg/mol]"
    molar_mass  :: FT
    "currently active time index (cached to avoid re-searching)"
    current_idx :: Int
end

function TimeVaryingEmission(flux_stack::AbstractArray{FT,3},
                             time_hours::Vector{Float64},
                             species::Symbol;
                             label::String="time-varying",
                             molar_mass::Real=molar_mass_for_species(species)) where FT
    @assert size(flux_stack, 3) == length(time_hours)
    TimeVaryingEmission{FT, typeof(flux_stack)}(
        flux_stack, time_hours, species, label, FT(molar_mass), 1)
end

"""
Update the active flux snapshot index based on simulation time.
Uses the most recent snapshot that does not exceed `sim_hours`.
"""
function update_time_index!(source::TimeVaryingEmission, sim_hours::Float64)
    idx = searchsortedlast(source.time_hours, sim_hours)
    idx = clamp(idx, 1, length(source.time_hours))
    source.current_idx = idx
    return idx
end

function apply_surface_flux!(tracers::NamedTuple, source::TimeVaryingEmission{FT},
                             grid::LatitudeLongitudeGrid{FT}, dt) where FT
    name = source.species
    haskey(tracers, name) || return nothing

    c = tracers[name]
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = grid.gravity
    ti = source.current_idx
    mol_ratio = FT(1e6 * M_AIR / source.molar_mass)

    @inbounds for j in 1:Ny, i in 1:Nx
        flux_ij = source.flux_stack[i, j, ti]
        flux_ij == zero(FT) && continue

        area_ij = cell_area(i, j, grid)
        Δp_sfc  = Δz(Nz, grid)
        m_air   = Δp_sfc * area_ij / g

        ΔM_kg   = flux_ij * FT(dt) * area_ij
        Δc_ppm  = ΔM_kg * mol_ratio / m_air
        c[i, j, Nz] += Δc_ppm
    end
    return nothing
end
