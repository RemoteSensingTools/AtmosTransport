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

const AbstractMassFluxBasis = AbstractMassBasis
const MoistMassFluxBasis = MoistBasis
const DryMassFluxBasis = DryBasis

export AbstractMassBasis, MoistBasis, DryBasis, mass_basis
export AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis
