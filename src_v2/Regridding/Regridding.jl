"""
    Regridding (src_v2)

Offline conservative regridding between v2 mesh types, built on
[ConservativeRegridding.jl](https://github.com/JuliaGeo/ConservativeRegridding.jl).

Designed for the preprocessing stage: build a sparse weights matrix once per
`(source_mesh, target_mesh)` pair, cache it to disk, and reuse at every
subsequent run. The runtime transport core never calls into this module —
`TransportBinaryReader` / `CubedSphereBinaryReader` consume binaries that are
already on the target grid.

## Workflow

```julia
using AtmosTransportV2: LatLonMesh, CubedSphereMesh
using AtmosTransportV2.Regridding

src = LatLonMesh(Nx=1440, Ny=721)
dst = CubedSphereMesh(Nc=90)

# Build weights (expensive — minutes for C90, cached to disk)
r = build_regridder(src, dst; cache_dir="/tmp/atmos_regrid_cache")

# Apply to a 2D field
src_field = zeros(src.Nx * src.Ny)
dst_field = zeros(6 * dst.Nc * dst.Nc)
apply_regridder!(dst_field, r, src_field)

# Persist / export
save_regridder("weights.jld2", r)
save_esmf_weights("weights_esmf.nc", r)
```

## Status (Tier 1)

- `LatLonMesh` and `CubedSphereMesh` are fully wired via `Trees.treeify`.
- `ReducedGaussianMesh` is handled by a `FlatNoTree` fallback — correct but
  slow at N320 scale until an STR-tree path is added.
- `CubedSphereMesh` uses the **analytical gnomonic** projection from
  `src_v2/Grids/CubedSphereMesh.jl`. Real GEOS-FP native cubed-sphere data
  has panel-4/5 axis rotations and small (~1–2°) corner offsets relative to
  the gnomonic projection; GMAO coordinate loading must be ported to v2 for
  bit-exact parity with production GEOS-FP binaries.

See also: `/home/cfranken/.claude/plans/luminous-prancing-firefly.md`.
"""
module Regridding

using ..Grids: AbstractHorizontalMesh, AbstractStructuredMesh,
               LatLonMesh, CubedSphereMesh, ReducedGaussianMesh,
               AbstractCubedSpherePanelConvention,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               nrings, nboundaries, ring_cell_count, cell_index,
               nx, ny, ncells

using ConservativeRegridding
using ConservativeRegridding: Regridder
using ConservativeRegridding.Trees

import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI
using GeometryOps.UnitSpherical: UnitSphericalPoint

using JLD2
using NCDatasets
using SparseArrays
using LinearAlgebra
using Dates
using SHA
using StaticArrays: SA

export build_regridder, save_regridder, load_regridder
export save_esmf_weights, apply_regridder!
export cubed_sphere_face_corners

include("treeify_meshes.jl")
include("weights_io.jl")

end # module Regridding
