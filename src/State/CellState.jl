# ---------------------------------------------------------------------------
# CellState — prognostic cell-centered fields for transport
#
# This is the primary state container that advection operates on.
# It carries air mass and tracer mass (or mixing ratio) per cell, with an
# explicit moist/dry basis tag at the type level.
# ---------------------------------------------------------------------------

"""
    CellState{Basis, A <: AbstractArray, Tr, Raw, Names}

Cell-centered prognostic state for transport.

# Fields
- `air_mass :: A` — air mass per cell [kg] on the basis carried by `Basis`.
  Layout matches grid:
  `(Nx, Ny, Nz)` for structured, `(ncells, Nz)` for unstructured.
- `tracers :: Tr` — `NamedTuple` of tracer mass arrays, same layout as
  `air_mass`. Each value is tracer mass [kg] = mixing_ratio × air_mass.
  *(Being retired — see Commit 4 of plan 14.)*
- `tracers_raw :: Raw` — parallel packed tracer storage: same element
  layout as `air_mass` plus a trailing tracer axis. Structured grids
  have `(Nx, Ny, Nz, Nt)`; face-indexed grids `(ncells, Nz, Nt)`.
  Kept in sync with `tracers` by the constructors. **Commit 3 adds this
  as a parallel (currently unused) allocation**; Commit 4 switches the
  pipeline to it and drops `tracers`.
- `tracer_names :: Names` — `NTuple{Nt, Symbol}` tracer name list.

# Invariant
After each transport step, `sum(air_mass)` and `sum(tracers.X)` for each
tracer X must be conserved (within floating-point tolerance).
"""
struct CellState{Basis <: AbstractMassBasis, A <: AbstractArray, Tr, Raw <: AbstractArray, Names <: Tuple}
    air_mass     :: A
    tracers      :: Tr
    tracers_raw  :: Raw
    tracer_names :: Names
end

# ---------------------------------------------------------------------------
# Construction helpers — allocate a parallel packed buffer alongside the
# existing NamedTuple of 3D arrays.
#
# Commit 3 discipline: `tracers` is still the primary storage for advection
# and other callers. `tracers_raw` is a **dead parallel allocation** that
# tests can read but nothing in the pipeline writes. Commit 4 flips the
# pipeline to `tracers_raw` and drops the NamedTuple field. Because the
# NamedTuple values remain the caller's arrays (shared reference), existing
# test helpers that rely on `CellState(m; X=arr)` keeping
# `state.tracers.X === arr` continue to work during Commit 3.
# ---------------------------------------------------------------------------

function _pack_tracers_raw(m::AbstractArray{FT, N}, tr::NamedTuple) where {FT, N}
    Nt = length(tr)
    raw = similar(m, FT, size(m)..., Nt)
    if Nt == 0
        return raw
    end
    @inbounds for (i, rm) in enumerate(values(tr))
        selectdim(raw, N + 1, i) .= rm
    end
    return raw
end

"""
    CellState(m::AbstractArray; tracers...)

Convenience constructor: `CellState(m; CO2=rm_co2, SF6=rm_sf6)`.
Defaults to `DryBasis`.
"""
function CellState(m::AbstractArray; tracers...)
    return CellState(DryBasis, m; tracers...)
end

function CellState(::Type{B}, m::AbstractArray; tracers...) where {B <: AbstractMassBasis}
    tr = NamedTuple(tracers)
    raw = _pack_tracers_raw(m, tr)
    names = keys(tr)  # NTuple{Nt, Symbol}
    return CellState{B, typeof(m), typeof(tr), typeof(raw), typeof(names)}(
        m, tr, raw, names)
end

mass_basis(::CellState{B}) where {B <: AbstractMassBasis} = B()

function Adapt.adapt_structure(to, state::CellState{B}) where {B <: AbstractMassBasis}
    air_mass    = Adapt.adapt(to, state.air_mass)
    tracers     = Adapt.adapt(to, state.tracers)
    tracers_raw = Adapt.adapt(to, state.tracers_raw)
    names       = state.tracer_names
    return CellState{B, typeof(air_mass), typeof(tracers), typeof(tracers_raw), typeof(names)}(
        air_mass, tracers, tracers_raw, names)
end

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
    return _get_tracer_impl(state.tracers, name) ./ state.air_mass
end

"""
    total_mass(state::CellState, name::Symbol) -> scalar

Sum of tracer mass across all cells and levels.
"""
function total_mass(state::CellState, name::Symbol)
    return sum(_get_tracer_impl(state.tracers, name))
end

"""
    total_air_mass(state::CellState) -> scalar

Sum of air mass across all cells and levels.
"""
total_air_mass(state::CellState) = sum(state.air_mass)

"""
    tracer_names(state::CellState) -> NTuple{Nt, Symbol}

Names of all tracers in `state`, in stored order. Reads the
`tracer_names` field directly.
"""
tracer_names(state::CellState) = getfield(state, :tracer_names)

export CellState, DryCellState, MoistCellState
export mixing_ratio, total_mass, total_air_mass, tracer_names
