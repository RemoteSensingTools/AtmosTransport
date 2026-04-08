# ---------------------------------------------------------------------------
# CubedSphereMesh — stub for Phase 2+
#
# Will hold gnomonic cubed-sphere mesh with 6 panels, panel connectivity,
# and precomputed cell areas / edge lengths from GMAO grid specs.
# ---------------------------------------------------------------------------

"""
    CubedSphereMesh{FT} <: AbstractStructuredMesh

Gnomonic cubed-sphere mesh with `Nc` cells per panel edge and 6 panels.
Phase 2+ implementation.
"""
struct CubedSphereMesh{FT <: AbstractFloat} <: AbstractStructuredMesh
    Nc :: Int
    radius :: FT
end

nx(m::CubedSphereMesh) = m.Nc
ny(m::CubedSphereMesh) = m.Nc
ncells(m::CubedSphereMesh) = 6 * m.Nc^2
nfaces(m::CubedSphereMesh) = 6 * 2 * m.Nc * (m.Nc + 1)

export CubedSphereMesh
