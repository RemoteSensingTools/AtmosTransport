# ---------------------------------------------------------------------------
# ReducedGaussianMesh — stub for Phase 2+
#
# Will hold the ECMWF reduced Gaussian grid with variable nlon per latitude
# ring, explicit face-cell connectivity (CSR), and precomputed face normals
# and cell areas.
# ---------------------------------------------------------------------------

"""
    ReducedGaussianMesh{FT} <: AbstractHorizontalMesh

Reduced Gaussian mesh for native ERA5 spectral model resolution.

NOTE: This is NOT a structured mesh — it uses explicit connectivity.
Transport kernels dispatch on `ReducedGaussianMesh` for face-based
advection (one thread per face), unlike the (i,j)-indexed structured path.

Phase 2+ implementation.
"""
struct ReducedGaussianMesh{FT <: AbstractFloat} <: AbstractHorizontalMesh
    latitudes     :: Vector{FT}
    nlon_per_ring :: Vector{Int}
    _ncells       :: Int
    _nfaces       :: Int
end

ncells(m::ReducedGaussianMesh) = m._ncells
nfaces(m::ReducedGaussianMesh) = m._nfaces

export ReducedGaussianMesh
