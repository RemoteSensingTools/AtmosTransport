# ---------------------------------------------------------------------------
# MetState — upstream meteorological fields (before flux construction)
#
# MetState holds the raw meteorological data that the met driver reads.
# The DryFluxBuilder consumes MetState and produces AbstractFaceFluxState +
# updated CellState.air_dry_mass.  Transport operators never see MetState.
# ---------------------------------------------------------------------------

"""
    MetState{PA <: AbstractArray, QA <: AbstractArray, M}

Container for meteorological fields upstream of the transport core.

# Fields
- `ps :: PA` — surface pressure [Pa]. Layout: `(Nx, Ny)` or `(ncells,)`.
- `q  :: QA` — specific humidity [kg/kg]. Layout is the matching horizontal
  shape plus a trailing vertical dimension.
- `metvars :: M` — additional met-specific fields (winds, omega, diffusivities, etc.)
  as a `NamedTuple`. Content depends on the met driver.

Transport operators never receive MetState directly. It is consumed by
`build_dry_fluxes!` to produce `AbstractFaceFluxState` and `CellState.air_dry_mass`.
"""
struct MetState{PA <: AbstractArray, QA <: AbstractArray, M}
    ps      :: PA
    q       :: QA
    metvars :: M

    function MetState(ps::PA, q::QA, metvars::M) where
            {PA <: AbstractArray, QA <: AbstractArray, M}
        ndims(q) == ndims(ps) + 1 || throw(DimensionMismatch(
            "MetState q must have exactly one more dimension than ps; got ndims(ps)=$(ndims(ps)), ndims(q)=$(ndims(q))"))
        size(q)[1:ndims(ps)] == size(ps) || throw(DimensionMismatch(
            "MetState q horizontal shape $(size(q)[1:ndims(ps)]) must match ps shape $(size(ps))"))
        return new{PA, QA, M}(ps, q, metvars)
    end
end

function MetState(ps::AbstractArray, q::AbstractArray; metvars...)
    return MetState(ps, q, NamedTuple(metvars))
end

export MetState
