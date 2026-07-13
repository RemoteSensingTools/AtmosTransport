# ---------------------------------------------------------------------------
# Mass basis tags shared by cell state and flux state.
# ---------------------------------------------------------------------------

"""
    AbstractMassBasis

Supertype for mass-basis tags carried by cell and face-flux states.
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

export AbstractMassBasis, MoistBasis, DryBasis, mass_basis
