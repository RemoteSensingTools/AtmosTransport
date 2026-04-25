# ---------------------------------------------------------------------------
# CubedSphereMesh — structured cubed-sphere grid for src
#
# Equidistant gnomonic cubed-sphere with computed geometry (cell areas,
# edge lengths) and panel connectivity. All panels share identical
# metrics by symmetry; connectivity follows the GEOS-FP convention.
#
# Geometry is computed at construction time from the gnomonic projection
# using CubedSphere.SphericalGeometry for exact spherical area/distance.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere grid
#   Martin et al. (2022, GMD) — GCHP v13
# ---------------------------------------------------------------------------

using CubedSphere.SphericalGeometry: spherical_area_quadrilateral, spherical_distance

abstract type AbstractCubedSpherePanelConvention end

"""
    GnomonicPanelConvention <: AbstractCubedSpherePanelConvention

Classical gnomonic panel numbering:
1-4 are equatorial panels, 5 is the north-pole panel, 6 is the south-pole panel.
"""
struct GnomonicPanelConvention <: AbstractCubedSpherePanelConvention end

"""
    GEOSNativePanelConvention <: AbstractCubedSpherePanelConvention

Panel numbering used by native GEOS-FP / GEOS-IT cubed-sphere files:
panels 1-2 are equatorial, 3 is north-pole, 4-5 are equatorial, 6 is south-pole.
"""
struct GEOSNativePanelConvention <: AbstractCubedSpherePanelConvention end

"""
    panel_connectivity_for(convention) -> PanelConnectivity

Return the `PanelConnectivity` that matches the given panel-numbering convention.
Dispatches to `gnomonic_panel_connectivity()` or `default_panel_connectivity()`
so that the mesh and all downstream code automatically use the right edge table.
"""
panel_connectivity_for(::GnomonicPanelConvention)   = gnomonic_panel_connectivity()
panel_connectivity_for(::GEOSNativePanelConvention) = default_panel_connectivity()

# ---------------------------------------------------------------------------
# Gnomonic projection
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel) -> (x, y, z)

Map local tangent-plane coordinates `(ξ, η)` to Cartesian `(x, y, z)` on the
unit sphere via the gnomonic (central) projection for the given `panel`.

The gnomonic projection maps from a tangent plane touching the sphere at the
panel center to the sphere surface via a straight line through the sphere's
center. In angular coordinates, `α` and `β` are the local east and north
angles from the panel center, both in `[-π/4, π/4]`. The tangent-plane
coordinates are `ξ = tan(α)` and `η = tan(β)`.

The normalisation factor `d = 1 / √(1 + ξ² + η²)` projects back onto the
unit sphere (ensures `x² + y² + z² = 1`).

## Panel orientation (gnomonic convention)

| Panel | Center face | (x, y, z) at ξ=η=0 | Local ξ → global | Local η → global |
|-------|------------|---------------------|-----------------|-----------------|
| 1     | +x face    | (1, 0, 0)           | ξ → +y          | η → +z          |
| 2     | +y face    | (0, 1, 0)           | ξ → −x          | η → +z          |
| 3     | −x face    | (−1, 0, 0)          | ξ → −y          | η → +z          |
| 4     | −y face    | (0, −1, 0)          | ξ → +x          | η → +z          |
| 5     | +z (N pole)| (0, 0, 1)           | ξ → +y          | η → −x          |
| 6     | −z (S pole)| (0, 0, −1)          | ξ → +y          | η → +x          |

Panels 1-4 are the equatorial belt; panels 5, 6 are the polar caps.
"""
@inline function _gnomonic_xyz(ξ::FT, η::FT, panel::Int) where FT
    d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)   # gnomonic normalisation
    if     panel == 1;  return ( d,  ξ*d,  η*d)   # +x face
    elseif panel == 2;  return (-ξ*d,  d,  η*d)   # +y face
    elseif panel == 3;  return (-d, -ξ*d,  η*d)   # −x face
    elseif panel == 4;  return ( ξ*d, -d,  η*d)   # −y face
    elseif panel == 5;  return (-η*d,  ξ*d,  d)   # +z (north pole)
    else;               return ( η*d,  ξ*d, -d)   # −z (south pole)
    end
end

# ---------------------------------------------------------------------------
# CubedSphereMesh
# ---------------------------------------------------------------------------

"""
    CubedSphereMesh{FT, C} <: AbstractStructuredMesh

Equidistant gnomonic cubed-sphere with `Nc` cells per panel edge and 6 panels.

Cell areas and edge lengths are computed once at construction time from the
gnomonic projection. All 6 panels share identical metrics by symmetry, so
only one set is stored.

# Fields
- `Nc :: Int` — cells per panel edge
- `Hp :: Int` — halo padding width (1 for slopes, 3 for PPM)
- `radius :: FT` — planet radius [m]
- `convention :: C` — panel numbering convention
- `connectivity :: PanelConnectivity` — edge-to-edge panel connectivity
- `cell_areas :: Matrix{FT}` — `(Nc, Nc)` cell areas [m²] (one panel, all identical)
- `Δx :: Matrix{FT}` — `(Nc, Nc)` x-direction cell widths [m]
- `Δy :: Matrix{FT}` — `(Nc, Nc)` y-direction cell widths [m]
"""
struct CubedSphereMesh{FT <: AbstractFloat,
                       C <: AbstractCubedSpherePanelConvention} <: AbstractStructuredMesh
    Nc           :: Int
    Hp           :: Int
    radius       :: FT
    convention   :: C
    connectivity :: PanelConnectivity
    cell_areas   :: Matrix{FT}
    Δx           :: Matrix{FT}
    Δy           :: Matrix{FT}
end

Base.eltype(::CubedSphereMesh{FT}) where FT = FT
Base.eltype(::Type{<:CubedSphereMesh{FT}}) where FT = FT

"""Number of panels (always 6 for a cubed sphere)."""
@inline panel_count(::CubedSphereMesh) = 6

"""Return the panel-numbering convention struct (Gnomonic or GEOSNative)."""
@inline panel_convention(m::CubedSphereMesh) = m.convention

"""
    panel_labels(convention) -> NTuple{6, Symbol}

Return symbolic labels for each panel under the given convention.

- **Gnomonic**: `(:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)`
  — panels 1-4 are equatorial (centred on ±x, ±y axes), 5 and 6 are polar.
- **GEOS native**: `(:equatorial_1, ..., :north_pole, ..., :south_pole)`
  — panels 1-2 and 4-5 are equatorial, 3 is north pole, 6 is south pole.
  This matches the file-panel ordering in GEOS-FP/GEOS-IT NetCDF variables.
"""
panel_labels(::GnomonicPanelConvention) =
    (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
panel_labels(::GEOSNativePanelConvention) =
    (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)
panel_labels(m::CubedSphereMesh) = panel_labels(panel_convention(m))

"""Cells per panel edge (same in both x and y by construction)."""
nx(m::CubedSphereMesh) = m.Nc
ny(m::CubedSphereMesh) = m.Nc

"""Total number of cells across all 6 panels: `6 × Nc²`."""
ncells(m::CubedSphereMesh) = panel_count(m) * m.Nc^2

"""Total number of faces across all 6 panels: `6 × 2 × Nc × (Nc + 1)` (x + y faces per panel)."""
nfaces(m::CubedSphereMesh) = panel_count(m) * 2 * m.Nc * (m.Nc + 1)

"""Cell area [m²] at local panel indices `(i, j)`. Same on all 6 panels by gnomonic symmetry."""
cell_area(m::CubedSphereMesh, i::Integer, j::Integer) = m.cell_areas[i, j]

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

function _validate_cubed_sphere_size(Nc::Integer)
    Nc > 0 || throw(ArgumentError("Nc must be positive"))
    return Int(Nc)
end

"""
    CubedSphereMesh(; Nc, FT=Float64, Hp=1, radius=6.371e6,
                      convention=GnomonicPanelConvention())

Construct an equidistant gnomonic cubed-sphere mesh with `Nc` cells per panel
edge. Cell areas and metric terms are computed from the gnomonic projection.

The default convention is `GnomonicPanelConvention()` (panels 1-4 equatorial,
5=north pole, 6=south pole), which matches all ERA5-CS binaries written by
the preprocessing pipeline. Pass `convention=GEOSNativePanelConvention()` only
when reading legacy GEOS-IT/FP binaries that use GEOS panel ordering.

# Keyword arguments
- `Nc :: Int` — cells per panel edge (e.g. 48, 90, 180, 720)
- `FT :: Type` — floating-point type (default `Float64`)
- `Hp :: Int` — halo padding width (default 1 for slopes; use 3 for PPM)
- `radius` — planet radius [m] (default Earth)
- `convention` — panel numbering convention (default gnomonic)
"""
function CubedSphereMesh(; FT::Type{<:AbstractFloat} = Float64,
                           Nc::Int,
                           Hp::Int = 1,
                           radius = FT(6.371e6),
                           convention::AbstractCubedSpherePanelConvention = GnomonicPanelConvention())
    Nc = _validate_cubed_sphere_size(Nc)
    Hp >= 0 || throw(ArgumentError("Hp must be non-negative, got $Hp"))

    R = FT(radius)
    conn = panel_connectivity_for(convention)

    # Angular grid: uniform spacing in α ∈ [-π/4, π/4]
    dα = FT(π) / (2 * Nc)
    α_faces   = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    α_centers = [(α_faces[i] + α_faces[i + 1]) / 2 for i in 1:Nc]

    areas = zeros(FT, Nc, Nc)
    dx    = zeros(FT, Nc, Nc)
    dy    = zeros(FT, Nc, Nc)

    # Compute geometry for panel 1 — all panels are identical by symmetry
    p = 1
    for j in 1:Nc, i in 1:Nc
        ξ1, ξ2 = tan(α_faces[i]), tan(α_faces[i + 1])
        η1, η2 = tan(α_faces[j]), tan(α_faces[j + 1])

        # Cell area: exact spherical quadrilateral area via Girard's theorem
        v1 = _gnomonic_xyz(FT(ξ1), FT(η1), p)
        v2 = _gnomonic_xyz(FT(ξ2), FT(η1), p)
        v3 = _gnomonic_xyz(FT(ξ2), FT(η2), p)
        v4 = _gnomonic_xyz(FT(ξ1), FT(η2), p)
        Ω = spherical_area_quadrilateral(v1, v2, v3, v4)
        areas[i, j] = R^2 * FT(Ω)

        # Δx: great-circle distance between midpoints of west and east edges
        mid_w = _gnomonic_xyz(FT(ξ1), FT(tan(α_centers[j])), p)
        mid_e = _gnomonic_xyz(FT(ξ2), FT(tan(α_centers[j])), p)
        dx[i, j] = R * FT(spherical_distance(mid_w, mid_e))

        # Δy: great-circle distance between midpoints of south and north edges
        mid_s = _gnomonic_xyz(FT(tan(α_centers[i])), FT(η1), p)
        mid_n = _gnomonic_xyz(FT(tan(α_centers[i])), FT(η2), p)
        dy[i, j] = R * FT(spherical_distance(mid_s, mid_n))
    end

    return CubedSphereMesh{FT, typeof(convention)}(
        Nc, Hp, R, convention, conn, areas, dx, dy)
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.summary(m::CubedSphereMesh)
    FT = eltype(m)
    conv = nameof(typeof(panel_convention(m)))
    return string("C", m.Nc, " CubedSphereMesh{", FT, ", ", conv, "}")
end

function Base.show(io::IO, m::CubedSphereMesh)
    print(io, summary(m), "\n",
          "├── panels:    ", panel_count(m), " with ", m.Nc, "×", m.Nc, " cells each\n",
          "├── halo:      ", m.Hp, " cells\n",
          "├── ordering:  ", join(string.(panel_labels(m)), ", "), "\n",
          "├── Δx range:  ", round(minimum(m.Δx)/1e3, digits=2), "–",
                             round(maximum(m.Δx)/1e3, digits=2), " km\n",
          "└── radius:    ", m.radius, " m")
end

# ---------------------------------------------------------------------------
# Visualization helpers — lat/lon coordinates for cell centers and corners
# ---------------------------------------------------------------------------

"""Convert Cartesian `(x, y, z)` on the unit sphere to `(lon, lat)` in degrees, lon in [0, 360)."""
@inline function _xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = asind(z / sqrt(x^2 + y^2 + z^2))
    lon < 0 && (lon += 360)
    return lon, lat
end

@inline function _rotate_z_lon_offset(x::FT, y::FT, z::FT, offset_deg::Real) where FT
    θ = FT(deg2rad(offset_deg))
    c = cos(θ)
    s = sin(θ)
    return (c * x - s * y, s * x + c * y, z)
end

@inline function _panel_xyz(::GnomonicPanelConvention, ξ::FT, η::FT, panel::Int) where FT
    return _gnomonic_xyz(ξ, η, panel)
end

"""
    _panel_xyz(::GEOSNativePanelConvention, ξ, η, panel)

GEOS-FP/GEOS-IT native cubed-sphere coordinates for arrays exposed by
NCDatasets as `(Xdim, Ydim, nf, ...)`.

GEOS native files use panel order `1, 2, north, 4, 5, south` and a global
longitude offset of `-10°` relative to the classical gnomonic cube. Panels 4
and 5 are stored with their `Ydim` direction reversed relative to the
corresponding gnomonic equatorial panels. Panel 3 (north pole) is stored with a
quarter-turn so its lower-left corner starts at ~35°E, matching the
`corner_lons/corner_lats` arrays in GEOS C180/C720 grid files.
"""
@inline function _panel_xyz(::GEOSNativePanelConvention, ξ::FT, η::FT, panel::Int) where FT
    ξg, ηg, gpanel = if panel == 1
        (ξ, η, 1)
    elseif panel == 2
        (ξ, η, 2)
    elseif panel == 3
        (-η, ξ, 5)
    elseif panel == 4
        (ξ, -η, 3)
    elseif panel == 5
        (ξ, -η, 4)
    elseif panel == 6
        (ξ, η, 6)
    else
        throw(ArgumentError("invalid GEOS native panel id $panel"))
    end
    x, y, z = _gnomonic_xyz(ξg, ηg, gpanel)
    return _rotate_z_lon_offset(x, y, z, -10)
end

"""
    panel_cell_center_lonlat(Nc, panel, FT) -> (lons, lats)
    panel_cell_center_lonlat(mesh::CubedSphereMesh, panel) -> (lons, lats)

Return `(Nc, Nc)` arrays of cell-center longitudes and latitudes in degrees
for the given `panel`.

The `Nc` method is the classical gnomonic convention. The mesh method honors
`panel_convention(mesh)`, including GEOS-native panel ordering and orientation.
"""
function panel_cell_center_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat})
    return _panel_cell_center_lonlat(Nc, panel, FT, GnomonicPanelConvention())
end

function panel_cell_center_lonlat(mesh::CubedSphereMesh{FT}, panel::Int) where FT
    return _panel_cell_center_lonlat(mesh.Nc, panel, FT, mesh.convention)
end

function _panel_cell_center_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   convention::AbstractCubedSpherePanelConvention)
    dα = FT(π) / (2 * Nc)
    α_centers = [FT(-π/4) + (i - FT(0.5)) * dα for i in 1:Nc]
    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        x, y, z = _panel_xyz(convention, tan(α_centers[i]), tan(α_centers[j]), panel)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

"""
    panel_cell_corner_lonlat(Nc, panel, FT) -> (lons, lats)
    panel_cell_corner_lonlat(mesh::CubedSphereMesh, panel) -> (lons, lats)

Return `(Nc+1, Nc+1)` arrays of cell-corner longitudes and latitudes in
degrees. The mesh method honors `panel_convention(mesh)`.
"""
function panel_cell_corner_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat})
    return _panel_cell_corner_lonlat(Nc, panel, FT, GnomonicPanelConvention())
end

function panel_cell_corner_lonlat(mesh::CubedSphereMesh{FT}, panel::Int) where FT
    return _panel_cell_corner_lonlat(mesh.Nc, panel, FT, mesh.convention)
end

function _panel_cell_corner_lonlat(Nc::Int, panel::Int, FT::Type{<:AbstractFloat},
                                   convention::AbstractCubedSpherePanelConvention)
    dα = FT(π) / (2 * Nc)
    α_faces = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    lons = zeros(FT, Nc + 1, Nc + 1)
    lats = zeros(FT, Nc + 1, Nc + 1)
    for j in 1:(Nc + 1), i in 1:(Nc + 1)
        x, y, z = _panel_xyz(convention, tan(α_faces[i]), tan(α_faces[j]), panel)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

@inline _dot3(ax, ay, az, bx, by, bz) = ax * bx + ay * by + az * bz

@inline function _normalize3(x, y, z)
    n = sqrt(x^2 + y^2 + z^2)
    invn = inv(max(n, eps(typeof(n))))
    return (x * invn, y * invn, z * invn)
end

@inline function _panel_unit_xyz(convention::AbstractCubedSpherePanelConvention,
                                α::Float64, β::Float64, panel::Int)
    x, y, z = _panel_xyz(convention, tan(α), tan(β), panel)
    return _normalize3(x, y, z)
end

@inline function _east_north_basis(x::Float64, y::Float64, z::Float64)
    rxy = hypot(x, y)
    if rxy > 1e-14
        east = (-y / rxy, x / rxy, 0.0)
    else
        # Longitude is singular at the pole. Pick a stable tangent basis; no
        # production C-grid has a cell center exactly at the pole, but odd Nc
        # synthetic tests can.
        east = (0.0, 1.0, 0.0)
    end
    north = (-z * east[2], z * east[1], x * east[2] - y * east[1])
    north = _normalize3(north...)
    return east, north
end

"""
    panel_cell_local_tangent_basis(mesh, panel)
        -> (x_east, x_north, y_east, y_north)

Return four `(Nc, Nc)` matrices describing the local panel-coordinate unit
vectors at cell centers in geographic `(east, north)` components.

For each cell, `(x_east[i,j], x_north[i,j])` is the unit vector for increasing
local X (`i`) and `(y_east[i,j], y_north[i,j])` is the unit vector for
increasing local Y (`j`). The helper honors `panel_convention(mesh)`, including
GEOS-native Y-reversed panels, and is the geometry contract used by
preprocessing wind rotation.
"""
function panel_cell_local_tangent_basis(mesh::CubedSphereMesh{FT}, panel::Int) where FT
    Nc = mesh.Nc
    dα = π / (2 * Nc)
    h = min(1.0e-6, 0.01 * dα)

    x_east  = zeros(FT, Nc, Nc)
    x_north = zeros(FT, Nc, Nc)
    y_east  = zeros(FT, Nc, Nc)
    y_north = zeros(FT, Nc, Nc)

    for j in 1:Nc, i in 1:Nc
        α = -π / 4 + (i - 0.5) * dα
        β = -π / 4 + (j - 0.5) * dα

        x0, y0, z0 = _panel_unit_xyz(mesh.convention, α, β, panel)
        xp, yp, zp = _panel_unit_xyz(mesh.convention, α + h, β, panel)
        xm, ym, zm = _panel_unit_xyz(mesh.convention, α - h, β, panel)
        xq, yq, zq = _panel_unit_xyz(mesh.convention, α, β + h, panel)
        xr, yr, zr = _panel_unit_xyz(mesh.convention, α, β - h, panel)

        dx = ((xp - xm) / (2h), (yp - ym) / (2h), (zp - zm) / (2h))
        dy = ((xq - xr) / (2h), (yq - yr) / (2h), (zq - zr) / (2h))

        # Remove radial roundoff before normalizing to tangent unit vectors.
        rx = _dot3(dx[1], dx[2], dx[3], x0, y0, z0)
        ry = _dot3(dy[1], dy[2], dy[3], x0, y0, z0)
        ex = _normalize3(dx[1] - rx * x0, dx[2] - rx * y0, dx[3] - rx * z0)
        ey = _normalize3(dy[1] - ry * x0, dy[2] - ry * y0, dy[3] - ry * z0)

        east, north = _east_north_basis(x0, y0, z0)

        x_east[i, j]  = FT(_dot3(ex[1], ex[2], ex[3], east[1], east[2], east[3]))
        x_north[i, j] = FT(_dot3(ex[1], ex[2], ex[3], north[1], north[2], north[3]))
        y_east[i, j]  = FT(_dot3(ey[1], ey[2], ey[3], east[1], east[2], east[3]))
        y_north[i, j] = FT(_dot3(ey[1], ey[2], ey[3], north[1], north[2], north[3]))
    end

    return (x_east, x_north, y_east, y_north)
end

export AbstractCubedSpherePanelConvention
export GnomonicPanelConvention, GEOSNativePanelConvention, panel_connectivity_for
export CubedSphereMesh, panel_count, panel_convention, panel_labels
export panel_cell_center_lonlat, panel_cell_corner_lonlat
export panel_cell_local_tangent_basis
