# ---------------------------------------------------------------------------
# Mass basis tags shared by cell state and flux state.
# ---------------------------------------------------------------------------

"""
    AbstractMassBasis

Supertype for mass-basis tags carried by `CellState` and `FluxState`.
"""
abstract type AbstractMassBasis end

"""
    MoistBasis <: AbstractMassBasis

Tag for total-air / moist mass.
"""
struct MoistBasis <: AbstractMassBasis end

"""
    DryBasis <: AbstractMassBasis

Tag for dry-air mass.
"""
struct DryBasis <: AbstractMassBasis end

mass_basis(::Type{B}) where {B <: AbstractMassBasis} = B()

"""
    mass_basis_type(s::Symbol) -> Type{<:AbstractMassBasis}

Map a header/driver basis `Symbol` (`:dry` / `:moist`) to the basis tag type.
Throws `ArgumentError` for anything else — callers must not default an
unknown basis to moist.
"""
@inline function mass_basis_type(s::Symbol)
    s === :dry && return DryBasis
    s === :moist && return MoistBasis
    throw(ArgumentError("unknown mass-basis symbol $(s); expected :dry or :moist"))
end

"""
    mass_basis_symbol(b::AbstractMassBasis) -> Symbol

Inverse of [`mass_basis_type`](@ref): the header `Symbol` for a basis tag.
"""
@inline mass_basis_symbol(::DryBasis) = :dry
@inline mass_basis_symbol(::MoistBasis) = :moist

const AbstractMassFluxBasis = AbstractMassBasis
const MoistMassFluxBasis = MoistBasis
const DryMassFluxBasis = DryBasis

export AbstractMassBasis, MoistBasis, DryBasis, mass_basis,
    mass_basis_type, mass_basis_symbol
export AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis
