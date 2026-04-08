# ---------------------------------------------------------------------------
# CellState — prognostic cell-centered fields for transport
#
# This is the primary state container that advection operates on.
# It carries dry air mass and tracer mass (or mixing ratio) per cell.
# ---------------------------------------------------------------------------

"""
    CellState{A <: AbstractArray, Tr}

Cell-centered prognostic state for transport.

# Fields
- `air_dry_mass :: A` — dry air mass per cell [kg]. Layout matches grid:
  `(Nx, Ny, Nz)` for structured, `(ncells, Nz)` for unstructured.
- `tracers :: Tr` — `NamedTuple` of tracer mass arrays, same layout as `air_dry_mass`.
  Each value is tracer mass [kg] = mixing_ratio × air_dry_mass.

# Invariant
After each transport step, `sum(air_dry_mass)` and `sum(tracers.X)` for each
tracer X must be conserved (within floating-point tolerance).
"""
struct CellState{A <: AbstractArray, Tr}
    air_dry_mass :: A
    tracers      :: Tr
end

"""
    CellState(m::AbstractArray; tracers...)

Convenience constructor: `CellState(m; CO2=rm_co2, SF6=rm_sf6)`.
"""
function CellState(m::AbstractArray; tracers...)
    tr = NamedTuple(tracers)
    return CellState(m, tr)
end

"""
    mixing_ratio(state::CellState, name::Symbol)

Compute mixing ratio `q = tracer_mass / air_dry_mass` for the named tracer.
Returns a lazy or materialized array depending on the backend.
"""
function mixing_ratio(state::CellState, name::Symbol)
    return getfield(state.tracers, name) ./ state.air_dry_mass
end

"""
    total_mass(state::CellState, name::Symbol) -> scalar

Sum of tracer mass across all cells and levels.
"""
function total_mass(state::CellState, name::Symbol)
    return sum(getfield(state.tracers, name))
end

"""
    total_air_mass(state::CellState) -> scalar

Sum of dry air mass across all cells and levels.
"""
total_air_mass(state::CellState) = sum(state.air_dry_mass)

"""
    tracer_names(state::CellState) -> Tuple of Symbols

Names of all tracers in the state.
"""
tracer_names(::CellState{A, <:NamedTuple{names}}) where {A, names} = names

export CellState, mixing_ratio, total_mass, total_air_mass, tracer_names
