# ---------------------------------------------------------------------------
# CellState — prognostic cell-centered fields for transport
#
# This is the primary state container that advection operates on.
# It carries air mass and tracer mass (or mixing ratio) per cell, with an
# explicit moist/dry basis tag at the type level.
# ---------------------------------------------------------------------------

"""
    CellState{Basis, A <: AbstractArray, Tr}

Cell-centered prognostic state for transport.

# Fields
- `air_mass :: A` — air mass per cell [kg] on the basis carried by `Basis`.
  Layout matches grid:
  `(Nx, Ny, Nz)` for structured, `(ncells, Nz)` for unstructured.
- `tracers :: Tr` — `NamedTuple` of tracer mass arrays, same layout as `air_mass`.
  Each value is tracer mass [kg] = mixing_ratio × air_mass.

# Invariant
After each transport step, `sum(air_mass)` and `sum(tracers.X)` for each
tracer X must be conserved (within floating-point tolerance).
"""
struct CellState{Basis <: AbstractMassBasis, A <: AbstractArray, Tr}
    air_mass :: A
    tracers  :: Tr
end

"""
    CellState(m::AbstractArray; tracers...)

Convenience constructor: `CellState(m; CO2=rm_co2, SF6=rm_sf6)`.
Defaults to `DryBasis`.
"""
function CellState(m::AbstractArray; tracers...)
    tr = NamedTuple(tracers)
    return CellState{DryBasis, typeof(m), typeof(tr)}(m, tr)
end

function CellState(::Type{B}, m::AbstractArray; tracers...) where {B <: AbstractMassBasis}
    tr = NamedTuple(tracers)
    return CellState{B, typeof(m), typeof(tr)}(m, tr)
end

mass_basis(::CellState{B}) where {B <: AbstractMassBasis} = B()

const DryCellState = CellState{DryBasis}
const MoistCellState = CellState{MoistBasis}

function Base.getproperty(state::CellState, name::Symbol)
    if name === :air_dry_mass
        return getfield(state, :air_mass)
    else
        return getfield(state, name)
    end
end

"""
    mixing_ratio(state::CellState, name::Symbol)

Compute mixing ratio `q = tracer_mass / air_dry_mass` for the named tracer.
Returns a lazy or materialized array depending on the backend.
"""
function mixing_ratio(state::CellState, name::Symbol)
    return getfield(state.tracers, name) ./ state.air_mass
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

Sum of air mass across all cells and levels.
"""
total_air_mass(state::CellState) = sum(state.air_mass)

"""
    tracer_names(state::CellState) -> Tuple of Symbols

Names of all tracers in the state.
"""
tracer_names(::CellState{B, A, <:NamedTuple{names}}) where {B, A, names} = names

export CellState, DryCellState, MoistCellState
export mixing_ratio, total_mass, total_air_mass, tracer_names
