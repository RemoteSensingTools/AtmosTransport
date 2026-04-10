# ---------------------------------------------------------------------------
# CubedSphereMesh — structured cubed-sphere grid for src_v2
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

# ---------------------------------------------------------------------------
# Gnomonic projection
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel)

Map local tangent-plane coordinates `(ξ, η)` to Cartesian `(x, y, z)` on the
unit sphere via the gnomonic (central) projection for the given `panel`.

`ξ = tan(α)`, `η = tan(β)` where `α, β ∈ [-π/4, π/4]`.
"""
@inline function _gnomonic_xyz(ξ::FT, η::FT, panel::Int) where FT
    d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
    if     panel == 1;  return ( d,  ξ*d,  η*d)
    elseif panel == 2;  return (-ξ*d,  d,  η*d)
    elseif panel == 3;  return (-d, -ξ*d,  η*d)
    elseif panel == 4;  return ( ξ*d, -d,  η*d)
    elseif panel == 5;  return (-η*d,  ξ*d,  d)
    else;               return ( η*d,  ξ*d, -d)
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

@inline panel_count(::CubedSphereMesh) = 6
@inline panel_convention(m::CubedSphereMesh) = m.convention

panel_labels(::GnomonicPanelConvention) =
    (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
panel_labels(::GEOSNativePanelConvention) =
    (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)
panel_labels(m::CubedSphereMesh) = panel_labels(panel_convention(m))

nx(m::CubedSphereMesh) = m.Nc
ny(m::CubedSphereMesh) = m.Nc
ncells(m::CubedSphereMesh) = panel_count(m) * m.Nc^2
nfaces(m::CubedSphereMesh) = panel_count(m) * 2 * m.Nc * (m.Nc + 1)

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
                      convention=GEOSNativePanelConvention())

Construct an equidistant gnomonic cubed-sphere mesh with `Nc` cells per panel
edge. Cell areas and metric terms are computed from the gnomonic projection.

# Keyword arguments
- `Nc :: Int` — cells per panel edge (e.g. 48, 90, 180, 720)
- `FT :: Type` — floating-point type (default `Float64`)
- `Hp :: Int` — halo padding width (default 1 for slopes; use 3 for PPM)
- `radius` — planet radius [m] (default Earth)
- `convention` — panel numbering convention (default GEOS-FP native)
"""
function CubedSphereMesh(; FT::Type{<:AbstractFloat} = Float64,
                           Nc::Int,
                           Hp::Int = 1,
                           radius = FT(6.371e6),
                           convention::AbstractCubedSpherePanelConvention = GEOSNativePanelConvention())
    Nc = _validate_cubed_sphere_size(Nc)
    Hp >= 0 || throw(ArgumentError("Hp must be non-negative, got $Hp"))

    R = FT(radius)
    conn = default_panel_connectivity()

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

export AbstractCubedSpherePanelConvention
export GnomonicPanelConvention, GEOSNativePanelConvention
export CubedSphereMesh, panel_count, panel_convention, panel_labels
