# ---------------------------------------------------------------------------
# Trees.treeify methods for v2 mesh types
#
# Each mesh type is converted into a ConservativeRegridding spatial tree that
# implements the SpatialTreeInterface. The tree is consumed by CR.jl's dual
# DFS during Regridder construction.
# ---------------------------------------------------------------------------

# --- best_manifold overloads --------------------------------------------------

GOCore.best_manifold(mesh::LatLonMesh)          = GO.Spherical(; radius = mesh.radius)
GOCore.best_manifold(mesh::CubedSphereMesh)     = GO.Spherical(; radius = mesh.radius)
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
# v2 CubedSphereMesh does not store face-corner coordinates; we regenerate
# them from the analytical gnomonic projection. Panel index remapping between
# conventions is handled by _gnomonic_panel_id.
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel) -> NTuple{3, Float64}

Local copy of the gnomonic projection helper in
`src_v2/Grids/CubedSphereMesh.jl`. Panel ids are the gnomonic convention:
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
coordinate loading is ported to v2.
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
# ReducedGaussianMesh
#
# TODO (Tier 1 follow-up): ReducedGaussianMesh treeify is a design choice
# between two approaches, neither of which is trivial:
#
#   A) Padded-matrix ExplicitPolygonGrid with a custom index-remapping
#      wrapper to translate the padded `(max_nlon, nrings)` linear index
#      back to the mesh's natural ring-flattened index. Reuses
#      TopDownQuadtreeCursor's spherical cell_range_extent for fast pruning
#      but requires writing a new cursor.
#
#   B) Flat Vector of polygons in a custom spherical spatial tree
#      (SphericalCap-based bulk-loaded tree, e.g. recursive ring grouping).
#      More work upfront but arguably the "right" solution.
#
#   The obvious Option C — `FlatNoTree(polys_with_unit_spherical_vertices)` —
#   does not currently work: FlatNoTree's per-cell extents fall back to
#   planar `Extents.Extent{(:X, :Y, :Z)}`, which CR.jl's spherical
#   `_intersects` does not know how to handle against the `SphericalCap`
#   extents produced by the other side of a regridder build. See
#   error trace in `/home/cfranken/.claude/plans/luminous-prancing-firefly.md`.
#
# For Tier 1 we stub this path with a clear error and ship the LatLon and
# CubedSphere paths, which cover the immediate ERA5 → C* preprocessing
# use case. N320-scale reduced Gaussian sources will be addressed in a
# dedicated follow-up.
# ---------------------------------------------------------------------------

function Trees.treeify(::GOCore.Spherical, ::ReducedGaussianMesh)
    error("""
    Trees.treeify(::Spherical, ::ReducedGaussianMesh) is not yet implemented.

    Tier 1 implementation (LatLonMesh + CubedSphereMesh) ships without this
    path because CR.jl's dual DFS on a spherical manifold requires
    per-cell `SphericalCap` extents, and `FlatNoTree` over polygons with
    UnitSphericalPoint vertices falls back to planar (X, Y, Z) `Extent`s
    that trip the intersection check.

    Workaround for ERA5 until this is implemented: regrid ERA5 reduced
    Gaussian → ERA5 regular LatLon upstream in the preprocessing pipeline,
    then use `build_regridder(::LatLonMesh, ::CubedSphereMesh)`.

    See /home/cfranken/.claude/plans/luminous-prancing-firefly.md for the
    planned follow-up.
    """)
end

Trees.treeify(mesh::ReducedGaussianMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)
