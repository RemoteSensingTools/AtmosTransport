# ---------------------------------------------------------------------------
# MetState — upstream meteorological fields (before flux construction)
#
# MetState holds the raw meteorological data that the met driver reads.
# The DryFluxBuilder consumes MetState and produces AbstractFaceFluxState +
# updated CellState.air_dry_mass.  Transport operators never see MetState.
# ---------------------------------------------------------------------------

"""
    MetState{A <: AbstractArray, M}

Container for meteorological fields upstream of the transport core.

# Fields
- `ps :: A` — surface pressure [Pa]. Layout: `(Nx, Ny)` or `(ncells,)`.
- `q  :: A` — specific humidity [kg/kg]. Layout matches grid 3D shape.
- `metvars :: M` — additional met-specific fields (winds, omega, diffusivities, etc.)
  as a `NamedTuple`. Content depends on the met driver.

Transport operators never receive MetState directly. It is consumed by
`build_dry_fluxes!` to produce `AbstractFaceFluxState` and `CellState.air_dry_mass`.
"""
struct MetState{A <: AbstractArray, M}
    ps      :: A
    q       :: A
    metvars :: M
end

function MetState(ps::AbstractArray, q::AbstractArray; metvars...)
    return MetState(ps, q, NamedTuple(metvars))
end

export MetState
