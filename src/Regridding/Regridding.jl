"""
    Regridding

Interpolation between met data grids and model grids.

When the met data grid differs from the model grid (e.g. ERA5 lat-lon 0.25°
to model CubedSphereGrid C90), a regridder maps fields between them.

# Regridder types

- `ConservativeRegridder` — mass-preserving remapping (for tracers, mass fluxes)
- `BilinearRegridder` — smooth interpolation (for winds, temperature)
- `IdentityRegridder` — no-op when grids match

# Interface contract

    regrid!(target_field, source_data, regridder)
"""
module Regridding

using DocStringExtensions
using SparseArrays: SparseMatrixCSC

export AbstractRegridder
export ConservativeRegridder, BilinearRegridder, IdentityRegridder
export regrid!, compute_weights

"""
$(TYPEDEF)

Supertype for all regridding methods.
"""
abstract type AbstractRegridder end

"""
$(TYPEDEF)

No-op regridder for when met data and model grids match.
"""
struct IdentityRegridder <: AbstractRegridder end

regrid!(target, source, ::IdentityRegridder) = copyto!(target, source)

"""
$(TYPEDEF)

Conservative (mass-preserving) regridding using precomputed sparse weights.

$(FIELDS)
"""
struct ConservativeRegridder{FT} <: AbstractRegridder
    "precomputed remapping weights"
    weights :: SparseMatrixCSC{FT, Int}
end

"""
$(TYPEDEF)

Bilinear interpolation using precomputed sparse weights.
Fast but not mass-conserving; suitable for smooth fields (wind, temperature).

$(FIELDS)
"""
struct BilinearRegridder{FT} <: AbstractRegridder
    "precomputed interpolation weights"
    weights :: SparseMatrixCSC{FT, Int}
end

"""
$(SIGNATURES)

Interpolate `source` data onto `target` field using `regridder`.
"""
function regrid!(target, source, r::Union{ConservativeRegridder{FT}, BilinearRegridder{FT}}) where FT
    # target[:] = r.weights * source[:]
    # Actual implementation depends on data layout; stub for now.
    error("regrid! not yet implemented for $(typeof(r))")
end

"""
$(SIGNATURES)

Precompute interpolation weights for mapping from `source_grid` to `target_grid`.
Implementation stub.
"""
function compute_weights(source_grid, target_grid, ::Type{<:AbstractRegridder})
    error("compute_weights not yet implemented")
end

end # module Regridding
