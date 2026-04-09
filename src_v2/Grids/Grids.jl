"""
    Grids (v2)

Geometry layer for the dry-mass transport architecture.

Provides:
- `AbstractHorizontalMesh` hierarchy with face/cell oriented API
- `AbstractVerticalCoordinate` with `HybridSigmaPressure`
- `AtmosGrid` composite type (mesh + vertical + architecture)
- Concrete meshes: `LatLonMesh` (structured fast path), `ReducedGaussianMesh`
  (variable-ring native ERA5 / IFS path), and `CubedSphereMesh`
  (structured GEOS/FV3 metadata with explicit panel conventions)
"""
module Grids

using ..Architectures: CPU
using ..Parameters: PlanetParameters, earth_parameters

include("AbstractMeshes.jl")
include("VerticalCoordinates.jl")
include("GeometryOps.jl")
include("LatLonMesh.jl")
include("CubedSphereMesh.jl")
include("ReducedGaussianMesh.jl")

end # module Grids
