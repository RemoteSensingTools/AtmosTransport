# =====================================================================
# Surface flux types — grid-parameterized surface mass flux sources
#
# Replaces the old GriddedEmission/CubedSphereEmission/TimeVaryingEmission/
# TimeVaryingCubedSphereEmission types with a cleaner 2-type hierarchy
# parameterized on layout tag (LatLonLayout or CubedSphereLayout).
# =====================================================================

# ---- Molar mass constants and lookup ----

const M_AIR   = 28.97e-3    # kg/mol  (dry air)
const M_CO2   = 44.01e-3    # kg/mol
const M_SF6   = 146.06e-3   # kg/mol
const M_RN222 = 222.0e-3    # kg/mol  (radon-222)

"""
    molar_mass_for_species(species::Symbol) → Float64

Look up the default molar mass (kg/mol) for a tracer species.
Falls back to `M_CO2` for unknown species.
"""
function molar_mass_for_species(species::Symbol)
    species === :co2       && return M_CO2
    species === :fossil_co2 && return M_CO2
    species === :sf6       && return M_SF6
    species === :rn222     && return M_RN222
    return M_CO2  # default
end

# ---- Layout tags ----

"""Layout tag for lat-lon data: flux stored as `Matrix{FT}` (Nx × Ny)."""
struct LatLonLayout end

"""Layout tag for cubed-sphere data: flux stored as `NTuple{6, Matrix{FT}}` (6 panels)."""
struct CubedSphereLayout end

# ---- Abstract type ----

"""
$(TYPEDEF)

Supertype for all surface mass flux sources.
"""
abstract type AbstractSurfaceFlux end

# ---- SurfaceFlux (constant in time) ----

"""
$(TYPEDEF)

Constant-in-time surface mass flux field.

Type parameters:
- `G`: Layout tag (`LatLonLayout` or `CubedSphereLayout`)
- `FT`: Float type
- `D`: Data container — `Matrix{FT}` for lat-lon, `NTuple{6, Matrix{FT}}` for cubed-sphere

$(FIELDS)
"""
struct SurfaceFlux{G, FT, D} <: AbstractSurfaceFlux
    "surface flux [kg/m²/s]"
    flux       :: D
    "tracer name (e.g. :co2)"
    species    :: Symbol
    "human-readable label"
    label      :: String
    "molar mass of emitted species [kg/mol]"
    molar_mass :: FT
end

# Lat-lon constructor (infers G from Matrix)
function SurfaceFlux(flux::A, species::Symbol, label::String;
                     molar_mass::Real=molar_mass_for_species(species)) where {FT, A <: AbstractMatrix{FT}}
    SurfaceFlux{LatLonLayout, FT, A}(flux, species, label, FT(molar_mass))
end

# Cubed-sphere constructor (infers G from NTuple{6})
function SurfaceFlux(flux_panels::NTuple{6, A}, species::Symbol, label::String;
                     molar_mass::Real=molar_mass_for_species(species)) where {FT, A <: AbstractMatrix{FT}}
    SurfaceFlux{CubedSphereLayout, FT, NTuple{6, A}}(flux_panels, species, label, FT(molar_mass))
end

# ---- TimeVaryingSurfaceFlux ----

"""
$(TYPEDEF)

Time-varying surface mass flux with periodic snapshot updates.

Type parameters:
- `G`: Layout tag (`LatLonLayout` or `CubedSphereLayout`)
- `FT`: Float type
- `S`: Stack container — `Array{FT,3}` for lat-lon, `Vector{NTuple{6, Matrix{FT}}}` for cubed-sphere

$(FIELDS)
"""
mutable struct TimeVaryingSurfaceFlux{G, FT, S} <: AbstractSurfaceFlux
    "flux snapshots [kg/m²/s]"
    flux_data   :: S
    "time axis: elapsed hours from simulation start for each snapshot"
    time_hours  :: Vector{Float64}
    "tracer name"
    species     :: Symbol
    "human-readable label"
    label       :: String
    "molar mass of emitted species [kg/mol]"
    molar_mass  :: FT
    "currently active time index"
    current_idx :: Int
    "if true, wrap sim_hours modulo the time span (for climatological data)"
    cyclic      :: Bool
end

# Lat-lon constructor (infers G from 3D array)
function TimeVaryingSurfaceFlux(flux_data::A, time_hours::Vector{Float64},
                                species::Symbol;
                                label::String="time-varying",
                                molar_mass::Real=molar_mass_for_species(species),
                                cyclic::Bool=false) where {FT, A <: AbstractArray{FT, 3}}
    @assert size(flux_data, 3) == length(time_hours)
    _warn_suspicious_time_hours(time_hours, label)
    TimeVaryingSurfaceFlux{LatLonLayout, FT, A}(
        flux_data, time_hours, species, label, FT(molar_mass), 1, cyclic)
end

# Cubed-sphere constructor (infers G from Vector of NTuple{6})
function TimeVaryingSurfaceFlux(flux_data::Vector{NTuple{6, A}}, time_hours::Vector{Float64},
                                species::Symbol;
                                label::String="time-varying CS",
                                molar_mass::Real=molar_mass_for_species(species),
                                cyclic::Bool=false) where {FT, A <: AbstractMatrix{FT}}
    @assert length(flux_data) == length(time_hours)
    _warn_suspicious_time_hours(time_hours, label)
    TimeVaryingSurfaceFlux{CubedSphereLayout, FT, Vector{NTuple{6, A}}}(
        flux_data, time_hours, species, label, FT(molar_mass), 1, cyclic)
end

"""Warn if time_hours look like sequential indices rather than real hours."""
function _warn_suspicious_time_hours(time_hours::Vector{Float64}, label::String)
    length(time_hours) > 1 || return
    max_gap = maximum(abs, diff(time_hours))
    if max_gap < 2.0
        @warn "$(label): time_hours look like sequential indices " *
              "($(time_hours[1:min(5,end)])...), not hours. " *
              "Snapshot selection will likely be wrong."
    end
end

# ---- Time index management ----

"""Update the active flux snapshot index based on simulation time (hours).

For `cyclic=true` sources (e.g. climatological monthly data), wraps `sim_hours`
modulo the time span so the snapshots repeat every cycle.
"""
function update_time_index!(source::TimeVaryingSurfaceFlux, sim_hours::Float64)
    th = source.time_hours
    Nt = length(th)

    if source.cyclic && Nt > 1
        # Wrap sim_hours into the cyclic time range [th[1], th[1] + span)
        span = th[end] - th[1] + (th[end] - th[end-1])  # assume last step same duration as second-to-last
        wrapped = th[1] + mod(sim_hours - th[1], span)
        idx = searchsortedlast(th, wrapped)
    else
        idx = searchsortedlast(th, sim_hours)
    end
    new_idx = clamp(idx, 1, Nt)
    if new_idx != source.current_idx
        cycle_info = source.cyclic ? " [cyclic]" : ""
        @info "$(source.label): snapshot $(source.current_idx) → $(new_idx)$cycle_info " *
              "(sim_hours=$(round(sim_hours, digits=1)), " *
              "snap_hours=$(th[new_idx]))"
        source.current_idx = new_idx
    end
    return source.current_idx
end

# ---- Flux data accessors ----

"""Return the currently active flux data for any surface flux type."""
flux_data(s::SurfaceFlux) = s.flux
flux_data(s::TimeVaryingSurfaceFlux{LatLonLayout}) = @view s.flux_data[:, :, s.current_idx]
flux_data(s::TimeVaryingSurfaceFlux{CubedSphereLayout}) = s.flux_data[s.current_idx]

# ---- CombinedFlux ----

"""
$(TYPEDEF)

Multiple flux sources applied together (e.g. anthropogenic + biosphere + ocean CO₂).

$(FIELDS)
"""
struct CombinedFlux{S <: Tuple} <: AbstractSurfaceFlux
    "tuple of AbstractSurfaceFlux components"
    components :: S
    "human-readable label"
    label      :: String
end

function CombinedFlux(sources::AbstractSurfaceFlux...; label="combined")
    CombinedFlux(sources, label)
end

# ---- NoFlux ----

"""No flux (pass-through)."""
struct NoFlux <: AbstractSurfaceFlux end

apply_surface_flux!(tracers, ::NoFlux, grid, dt) = nothing
