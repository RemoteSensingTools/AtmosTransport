# ---------------------------------------------------------------------------
# Trees.treeify methods for mesh types
#
# Each mesh type is converted into a ConservativeRegridding spatial tree that
# implements the SpatialTreeInterface. The tree is consumed by CR.jl's dual
# DFS during Regridder construction.
# ---------------------------------------------------------------------------

# --- best_manifold overloads --------------------------------------------------
#
# GeometryOpsCore.best_manifold tells CR.jl which manifold (Planar or
# Spherical) a grid lives on, and at what radius. All AtmosTransport
# meshes are spherical with radius = mesh.radius (default 6.371e6 m).

"""Return `Spherical(radius=mesh.radius)` for a `LatLonMesh`."""
GOCore.best_manifold(mesh::LatLonMesh)          = GO.Spherical(; radius = mesh.radius)
"""Return `Spherical(radius=mesh.radius)` for a `CubedSphereMesh`."""
GOCore.best_manifold(mesh::CubedSphereMesh)     = GO.Spherical(; radius = mesh.radius)
"""Return `Spherical(radius=mesh.radius)` for a `ReducedGaussianMesh`."""
GOCore.best_manifold(mesh::ReducedGaussianMesh) = GO.Spherical(; radius = mesh.radius)

# ---------------------------------------------------------------------------
# LatLonMesh → CellBasedGrid(UnitSphericalPoint) → TopDownQuadtreeCursor
#
# Uses the cell-face vectors λᶠ, φᶠ to build an (Nx+1) × (Ny+1) matrix of
# corner points on the unit sphere. If the mesh covers the full sphere we
# skip extent computation via KnownFullSphereExtentWrapper.
# ---------------------------------------------------------------------------

"""
    _latlon_full_sphere(mesh::LatLonMesh) -> Bool

Return `true` when the mesh's face vectors span the full sphere
(longitude covers 360°, latitude from −90 to +90).
"""
function _latlon_full_sphere(mesh::LatLonMesh)
    Δλ = last(mesh.λᶠ) - first(mesh.λᶠ)
    return isapprox(Δλ, 360; atol = 1e-6) &&
           isapprox(first(mesh.φᶠ), -90; atol = 1e-6) &&
           isapprox(last(mesh.φᶠ),   90; atol = 1e-6)
end

"""
    _latlon_corner_matrix(mesh::LatLonMesh) -> Matrix{UnitSphericalPoint}

Build the `(Nx+1) × (Ny+1)` matrix of cell corners, mapped to unit-sphere
Cartesian coordinates. This matches the Oceananigans pattern in
`ConservativeRegriddingOceananigansExt.jl`.
"""
function _latlon_corner_matrix(mesh::LatLonMesh)
    Nx1 = length(mesh.λᶠ)
    Ny1 = length(mesh.φᶠ)
    pts = Matrix{UnitSphericalPoint{Float64}}(undef, Nx1, Ny1)
    # GO.UnitSphereFromGeographic() expects (lon, lat) degrees
    to_sphere = GO.UnitSphereFromGeographic()
    @inbounds for j in 1:Ny1, i in 1:Nx1
        pts[i, j] = to_sphere((Float64(mesh.λᶠ[i]), Float64(mesh.φᶠ[j])))
    end
    return pts
end

"""
    Trees.treeify(::Spherical, mesh::LatLonMesh)

Build a spatial tree for a `LatLonMesh` on the sphere.

## Algorithm

1. Build an `(Nx+1) × (Ny+1)` matrix of cell-face corner points as
   `UnitSphericalPoint`s, from the mesh's face vectors `λᶠ` (longitude)
   and `φᶠ` (latitude). Each corner `(i, j)` is at geographic coordinate
   `(λᶠ[i], φᶠ[j])`, converted to Cartesian `(x, y, z)` on the unit
   sphere via `GO.UnitSphereFromGeographic()`.

2. Wrap in `CellBasedGrid{Spherical}`, which builds cell polygons on the
   fly as quadrilaterals from 4 adjacent corner points. Cell `(i, j)` has
   vertices SW=`(i,j)`, SE=`(i+1,j)`, NE=`(i+1,j+1)`, NW=`(i,j+1)`.

3. Wrap in `TopDownQuadtreeCursor` for O(log(Nx·Ny)) spatial pruning.

4. If the mesh covers the full sphere (360° longitude, ±90° latitude),
   wrap in `KnownFullSphereExtentWrapper` to skip expensive extent
   computation at the root.

## Longitude convention

The mesh's `λᶠ` vector defines cell-face longitudes. The default
`LatLonMesh()` uses `[-180, 180]`. The UnitSphericalPoint representation
is the same physical location regardless of whether the source convention
is `[-180, 180]` or `[0, 360]` — the Cartesian `(x, y, z)` coordinates
are unambiguous.

## Cell indexing

CR.jl's `CellBasedGrid` uses column-major linear indexing:
`linear_idx = i + (j-1) * Nx`, where `i` is the longitude index (1:Nx)
and `j` is the latitude index (1:Ny). This matches Julia's default
`vec(reshape(field, Nx, Ny))` memory layout.
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::LatLonMesh)
    corners = _latlon_corner_matrix(mesh)
    grid    = Trees.CellBasedGrid(manifold, corners)
    tree    = Trees.TopDownQuadtreeCursor(grid)
    return _latlon_full_sphere(mesh) ? Trees.KnownFullSphereExtentWrapper(tree) : tree
end

Trees.treeify(mesh::LatLonMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)

# ---------------------------------------------------------------------------
# CubedSphereMesh → CubedSphereToplevelTree of 6 CellBasedGrids
#
# CubedSphereMesh does not store face-corner coordinates; we regenerate
# them from the analytical gnomonic projection. Panel index remapping between
# conventions is handled by _gnomonic_panel_id.
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel) -> NTuple{3, Float64}

Local copy of the gnomonic projection helper in
`src/Grids/CubedSphereMesh.jl`. Panel ids are the gnomonic convention:
1=X+, 2=Y+, 3=X−, 4=Y−, 5=Z+ (north), 6=Z− (south).

Reproduced here so this module does not depend on a non-exported name from
`Grids`.
"""
@inline function _gnomonic_xyz(ξ::Float64, η::Float64, panel::Int)
    d = 1.0 / sqrt(1.0 + ξ^2 + η^2)
    if     panel == 1; return ( d,    ξ*d,  η*d)
    elseif panel == 2; return (-ξ*d,  d,    η*d)
    elseif panel == 3; return (-d,   -ξ*d,  η*d)
    elseif panel == 4; return ( ξ*d, -d,    η*d)
    elseif panel == 5; return (-η*d,  ξ*d,  d  )
    else               return ( η*d,  ξ*d, -d  )
    end
end

"""
    _gnomonic_panel_id(conv, p) -> Int

Map a user-visible panel index `p` (1..6) under convention `conv` to the
gnomonic panel id consumed by `_gnomonic_xyz`.

- `GnomonicPanelConvention` → identity
- `GEOSNativePanelConvention` → (1→1, 2→2, 3→5, 4→3, 5→4, 6→6)

Based on the label interpretation in `CubedSphereMesh.jl`:
```
GnomonicPanelConvention:   (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
GEOSNativePanelConvention: (:equatorial_1, :equatorial_2, :north_pole,
                            :equatorial_4, :equatorial_5, :south_pole)
```

**Caveat:** GEOS panels 4 and 5 have local (X=south, Y=east) axes — a 90° CW
rotation relative to the gnomonic `(ξ, η)` parameterization. The current
mapping places polygons at the right physical location but assumes gnomonic
(i, j) ordering within each panel. Real GEOS-FP binary data uses the rotated
(i, j) ordering, and for bit-exact parity with production binaries this
function must be paired with a per-panel (i, j) rotation. TODO when GMAO
coordinate loading is ported .
"""
@inline function _gnomonic_panel_id(::GnomonicPanelConvention, p::Int)
    return p
end

@inline function _gnomonic_panel_id(::GEOSNativePanelConvention, p::Int)
    if     p == 1; return 1
    elseif p == 2; return 2
    elseif p == 3; return 5   # north pole
    elseif p == 4; return 3
    elseif p == 5; return 4
    elseif p == 6; return 6   # south pole
    else error("invalid GEOS native panel id $p")
    end
end

"""
    cubed_sphere_face_corners(mesh::CubedSphereMesh)
        -> NTuple{6, Matrix{UnitSphericalPoint{Float64}}}

Return a 6-tuple of `(Nc+1) × (Nc+1)` corner-point matrices, one per panel,
with corners on the unit sphere. Panel ordering follows `mesh.convention`;
for each user-visible panel `p`, corner `(i, j)` corresponds to the cell
corner at angular coordinates `(α_faces[i], α_faces[j])` on the gnomonic
panel identified by `_gnomonic_panel_id(mesh.convention, p)`.

Used by `Trees.treeify(::Spherical, ::CubedSphereMesh)` to assemble a
`CubedSphereToplevelTree`. Cached externally (the regridder cache includes
the mesh hash) so this is only called once per `(Nc, convention)`.
"""
function cubed_sphere_face_corners(mesh::CubedSphereMesh)
    Nc  = mesh.Nc
    Np  = Nc + 1
    dα  = π / (2 * Nc)
    α_faces = Vector{Float64}(undef, Np)
    @inbounds for i in 1:Np
        α_faces[i] = -π/4 + (i - 1) * dα
    end
    ξ_faces = tan.(α_faces)
    # Cache one corner matrix per panel
    panels = ntuple(6) do user_panel
        g = _gnomonic_panel_id(mesh.convention, user_panel)
        m = Matrix{UnitSphericalPoint{Float64}}(undef, Np, Np)
        @inbounds for j in 1:Np, i in 1:Np
            xyz = _gnomonic_xyz(ξ_faces[i], ξ_faces[j], g)
            m[i, j] = UnitSphericalPoint(xyz[1], xyz[2], xyz[3])
        end
        m
    end
    return panels
end

"""
    Trees.treeify(::Spherical, mesh::CubedSphereMesh)

Build a spatial tree for a `CubedSphereMesh` on the sphere.

## Algorithm

1. Call [`cubed_sphere_face_corners`](@ref) to produce 6 matrices of
   `(Nc+1) × (Nc+1)` `UnitSphericalPoint` corner coordinates, one per
   panel. The panel ordering follows `mesh.convention` (gnomonic or
   GEOS-native), with panel-index remapping via `_gnomonic_panel_id`.

2. For each panel `p`, build a `CellBasedGrid{Spherical}` from the
   corner matrix, and wrap it in an `IndexOffsetQuadtreeCursor` with
   offset `(p-1) × Nc²`. This ensures the global cell index is:

       global_idx = (panel - 1) × Nc² + local_linear_idx

   where `local_linear_idx = i + (j-1) × Nc` (column-major within
   the panel, `i` = local ξ-index, `j` = local η-index).

3. Assemble all 6 per-panel cursors into a `CubedSphereToplevelTree`,
   whose top-level extent is the full sphere.

## Cell indexing

The global flat index space is `1 : 6·Nc²`. Panel `p` occupies indices
`(p-1)·Nc² + 1 : p·Nc²`. Within a panel, cells are column-major in
`(i, j)` gnomonic indices: `i` varies fastest (ξ direction), `j` second
(η direction). This layout matches `vec(panel_data[:, :])` in Julia.

## Known limitation

Uses analytical gnomonic coordinates only. GEOS panels 4/5 have a 90° CW
local-axis rotation relative to the gnomonic `(ξ, η)` parameterization.
The current implementation places polygons at the correct physical
location but uses gnomonic `(i, j)` ordering within each panel. For
bit-exact parity with production GEOS-FP panel data, GMAO coordinate
loading must be ported and the per-panel `(i, j)` rotation applied.
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::CubedSphereMesh)
    corners = cubed_sphere_face_corners(mesh)
    Nc = mesh.Nc
    N_per_panel = Nc * Nc
    quadtrees = [
        Trees.IndexOffsetQuadtreeCursor(
            Trees.CellBasedGrid(manifold, corners[p]),
            (p - 1) * N_per_panel,
        )
        for p in 1:6
    ]
    return Trees.CubedSphereToplevelTree(quadtrees)
end

Trees.treeify(mesh::CubedSphereMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)

# ---------------------------------------------------------------------------
# ReducedGaussianMesh → per-ring CellBasedGrid trees → MultiTreeWrapper
#
# A reduced Gaussian mesh has variable `nlon` per latitude ring, so it
# cannot be directly represented as a single rectangular
# AbstractCurvilinearGrid. Instead, we treat each ring as an independent
# `CellBasedGrid{Spherical}` with shape `(nlon+1, 2)` — the +1 in the
# longitude dimension gives face edges, and the 2 latitudes are the
# ring's south and north boundaries.
#
# Each ring tree is an `IndexOffsetQuadtreeCursor` that maps the ring's
# local cell indices (1:nlon) to the mesh's global flat index:
#   global_idx = ring_offsets[j] + local_idx - 1
# via offset = ring_offsets[j] - 1.
#
# The per-ring trees are combined into a `MultiTreeWrapper` (from
# CR.jl's wrappers.jl), which uses cumulative cell offsets for O(1)
# ring lookup given a global index. The whole thing is wrapped in
# `KnownFullSphereExtentWrapper` since a reduced Gaussian mesh
# (ERA5 / IFS native) always covers the full sphere.
#
# This approach:
#   - Reuses CellBasedGrid{Spherical}'s `cell_range_extent`, which
#     correctly produces SphericalCap extents (avoids the
#     SphericalCap vs Extent{(:X,:Y,:Z)} type mismatch that crashes
#     FlatNoTree on spherical manifolds)
#   - Gives O(log nlon) spatial pruning within each ring via
#     IndexOffsetQuadtreeCursor's recursive subdivision
#   - Gives O(nrings) lookup at the top level, acceptable for ERA5
#     (nrings ~ 320)
#   - Uses only existing CR.jl infrastructure: CellBasedGrid,
#     IndexOffsetQuadtreeCursor, MultiTreeWrapper,
#     KnownFullSphereExtentWrapper
#
# Cell index convention:
#   Cells are flattened ring-by-ring south→north, west→east within
#   each ring, starting at longitude 0° with uniform spacing
#   Δlon = 360° / nlon. This matches ReducedGaussianMesh's own
#   `cell_index(mesh, i, j) = ring_offsets[j] + i - 1`.
#
# Corner-point construction:
#   For ring j with nlon cells:
#     - Longitude face edges: lon_k = (k-1) * 360/nlon for k = 1:nlon+1
#       (first face at 0°, last at 360° — numerically equal as
#       UnitSphericalPoint to ~1e-15)
#     - Latitude face edges: lat_faces[j] (south), lat_faces[j+1] (north)
#     - Corner matrix pts[k, ℓ] where k = 1:nlon+1 (lon faces),
#       ℓ = 1 (south lat) or 2 (north lat)
#     - All corners are UnitSphericalPoint via GO.UnitSphereFromGeographic()
#   CellBasedGrid.getcell(grid, i, 1) builds the polygon:
#     pts[i,1] (SW) → pts[i+1,1] (SE) → pts[i+1,2] (NE) → pts[i,2] (NW)
#   Winding: counter-clockwise when viewed from outside the sphere. ✓
# ---------------------------------------------------------------------------

"""
    Trees.treeify(::Spherical, mesh::ReducedGaussianMesh) -> KnownFullSphereExtentWrapper{MultiTreeWrapper}

Build a spatial tree for a `ReducedGaussianMesh` by treating each latitude
ring as an independent `CellBasedGrid{Spherical}` and combining all rings
into a `MultiTreeWrapper`.

Each ring `j` (south to north) becomes a `(nlon+1, 2)` corner-point grid
wrapped in an `IndexOffsetQuadtreeCursor` with offset `ring_offsets[j] - 1`,
so that the cursor's global cell indices match the mesh's flat cell indexing:

    global_cell_index = ring_offsets[j] + in_ring_index - 1

The resulting tree correctly produces `SphericalCap` extents at every level
(via `CellBasedGrid{Spherical}`'s specialized `cell_range_extent`) and
provides O(log nlon) pruning within each ring via the cursor's recursive
subdivision.

## Longitude convention

Within each ring, the `nlon` cells span `[0°, 360°)` uniformly:
- Cell `i` covers longitude `[(i-1) × 360/nlon, i × 360/nlon]`
- Cell centers are at `(i - 0.5) × 360/nlon` (matching
  `ReducedGaussianMesh.ring_longitudes`)

## Latitude convention

Ring boundaries come from `mesh.lat_faces` (south to north, with
`lat_faces[1] = -90°` and `lat_faces[end] = +90°`).
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::ReducedGaussianMesh)
    to_sphere = GO.UnitSphereFromGeographic()
    nr = nrings(mesh)

    ring_trees = map(1:nr) do j
        nlon = mesh.nlon_per_ring[j]
        dlon = 360.0 / nlon
        φ_s  = Float64(mesh.lat_faces[j])
        φ_n  = Float64(mesh.lat_faces[j + 1])

        # Epsilon-clamp polar latitudes to avoid degenerate polygons.
        #
        # At exactly ±90°, all longitude face edges map to the same
        # UnitSphericalPoint (the pole). This makes the polygon a
        # degenerate quadrilateral (two vertices coincide), and
        # GO.area(Spherical(), polygon) can underestimate the area by
        # up to ~6% for large polar cells. Clamping by ε = 0.001°
        # (~111 m at the pole) creates a tiny but non-degenerate
        # quadrilateral. The omitted polar cap area is
        # π × (R × ε_rad)² ≈ 4e4 m² — negligible relative to the
        # full sphere (5.1e14 m²), ratio ~8e-11.
        φ_s = max(φ_s, -90.0 + 0.001)
        φ_n = min(φ_n,  90.0 - 0.001)

        # Build (nlon+1) × 2 corner-point matrix.
        # Dimension 1: longitude face edges (0° to 360° inclusive).
        # Dimension 2: latitude face edges (south=1, north=2).
        pts = Matrix{UnitSphericalPoint{Float64}}(undef, nlon + 1, 2)
        @inbounds for k in 1:(nlon + 1)
            lon = (k - 1) * dlon
            pts[k, 1] = to_sphere((lon, φ_s))
            pts[k, 2] = to_sphere((lon, φ_n))
        end

        ring_grid = Trees.CellBasedGrid(manifold, pts)
        # IndexOffsetQuadtreeCursor maps local index → global:
        #   global = local + offset, where offset = ring_offsets[j] - 1
        Trees.IndexOffsetQuadtreeCursor(ring_grid, mesh.ring_offsets[j] - 1)
    end

    # Cumulative cell counts for MultiTreeWrapper's searchsortedfirst.
    # offsets[j] = total cells in rings 1:j = ring_offsets[j+1] - 1.
    offsets = [mesh.ring_offsets[j + 1] - 1 for j in 1:nr]

    tree = Trees.MultiTreeWrapper(ring_trees, offsets)
    return Trees.KnownFullSphereExtentWrapper(tree)
end

Trees.treeify(mesh::ReducedGaussianMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)
