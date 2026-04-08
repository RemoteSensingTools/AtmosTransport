"""
    Grids (v2)

Geometry layer for the dry-mass transport architecture.

Provides:
- `AbstractHorizontalMesh` hierarchy with face/cell oriented API
- `AbstractVerticalCoordinate` with `HybridSigmaPressure`
- `AtmosGrid` composite type (mesh + vertical + architecture)
- Concrete meshes: `LatLonMesh` (Phase 1), `CubedSphereMesh` / `ReducedGaussianMesh` (stubs)
"""
module Grids

using DocStringExtensions

include("AbstractMeshes.jl")
include("VerticalCoordinates.jl")
include("GeometryOps.jl")
include("LatLonMesh.jl")
include("CubedSphereMesh.jl")
include("ReducedGaussianMesh.jl")

end # module Grids
