# ---------------------------------------------------------------------------
# CubedSphereMesh -- structured cubed-sphere metadata for src_v2
#
# This is the structured horizontal-mesh descriptor for GEOS/FV3-style
# equidistant gnomonic cubed-sphere grids. The geometry-heavy implementation
# still lives in src/, but src_v2 can already carry the key resolution and
# panel-order conventions explicitly instead of burying them in comments.
# ---------------------------------------------------------------------------

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
    CubedSphereMesh{FT, C} <: AbstractStructuredMesh

Equidistant gnomonic cubed-sphere with `Nc` cells per panel edge and 6 panels.

This type currently records the structured mesh metadata, resolution naming,
and panel-order convention. The full metric and connectivity implementation
from the existing GEOS/FV3 path can be layered onto this interface next.
"""
struct CubedSphereMesh{FT <: AbstractFloat, C <: AbstractCubedSpherePanelConvention} <: AbstractStructuredMesh
    Nc         :: Int
    radius     :: FT
    convention :: C
end

Base.eltype(::CubedSphereMesh{FT}) where FT = FT
Base.eltype(::Type{<:CubedSphereMesh{FT}}) where FT = FT

@inline function panel_count(::CubedSphereMesh)
    return 6
end

@inline panel_convention(m::CubedSphereMesh) = m.convention

panel_labels(::GnomonicPanelConvention) =
    (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)

panel_labels(::GEOSNativePanelConvention) =
    (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)

panel_labels(m::CubedSphereMesh) = panel_labels(panel_convention(m))

function _validate_cubed_sphere_size(Nc::Integer)
    Nc > 0 || throw(ArgumentError("Nc must be positive"))
    return Int(Nc)
end

function CubedSphereMesh(; FT::Type{<:AbstractFloat} = Float64,
                           Nc::Int,
                           radius = FT(6.371e6),
                           convention::AbstractCubedSpherePanelConvention = GEOSNativePanelConvention())
    Nc = _validate_cubed_sphere_size(Nc)
    return CubedSphereMesh{FT, typeof(convention)}(Nc, FT(radius), convention)
end

nx(m::CubedSphereMesh) = m.Nc
ny(m::CubedSphereMesh) = m.Nc
ncells(m::CubedSphereMesh) = panel_count(m) * m.Nc^2
nfaces(m::CubedSphereMesh) = panel_count(m) * 2 * m.Nc * (m.Nc + 1)

function Base.summary(m::CubedSphereMesh)
    FT = eltype(m)
    convention = nameof(typeof(panel_convention(m)))
    return string("C", m.Nc, " CubedSphereMesh{", FT, ", ", convention, "}")
end

function Base.show(io::IO, m::CubedSphereMesh)
    print(io, summary(m), "\n",
          "├── panels:    ", panel_count(m), " with ", m.Nc, "×", m.Nc, " cells each\n",
          "├── ordering:  ", join(string.(panel_labels(m)), ", "), "\n",
          "└── radius:    ", m.radius, " m")
end

export AbstractCubedSpherePanelConvention
export GnomonicPanelConvention, GEOSNativePanelConvention
export CubedSphereMesh, panel_count, panel_convention, panel_labels
