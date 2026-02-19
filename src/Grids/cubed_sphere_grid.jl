# ---------------------------------------------------------------------------
# CubedSphereGrid
#
# Gnomonic equidistant cubed-sphere grid, compatible with GEOS-FP/FV3 and
# MERRA-2 conventions.  Six panels, each Nc × Nc cells.
#
# Resolutions follow the GEOS convention: C48, C90, C180, C360, C720.
# Panel numbering (GEOS): 1=front (0°E), 2=east (90°E), 3=back (180°E),
# 4=west (90°W), 5=north, 6=south.
#
# References:
#   Putman & Lin (2007) — FV3 cubed-sphere advection
#   Ronchi et al. (1996) — gnomonic projection
#   Martin et al. (2022, GMD) — GCHP v13, native CS mass fluxes
# ---------------------------------------------------------------------------

using CubedSphere.SphericalGeometry: spherical_area_quadrilateral, spherical_distance

"""
$(TYPEDEF)

Describes how the 6 panels of a cubed-sphere connect at their edges.
Each panel has 4 neighbors (north, south, east, west), identified by
panel index and orientation (rotation needed to align axes).

$(FIELDS)
"""
struct PanelConnectivity
    "per-panel neighbor info: `(north, south, east, west)` as `(panel, orientation)` tuples"
    neighbors :: NTuple{6, NTuple{4, @NamedTuple{panel::Int, orientation::Int}}}
end

"""
$(TYPEDEF)

Gnomonic equidistant cubed-sphere grid with 6 panels of `Nc × Nc` cells.
Uses the same projection as NASA GEOS-FP/FV3 (Putman & Lin, 2007).

$(FIELDS)
"""
struct CubedSphereGrid{FT, Arch, VZ} <: AbstractStructuredGrid{FT, Arch}
    "compute architecture (CPU or GPU)"
    architecture :: Arch

    "cells per panel edge (e.g. 90 for C90)"
    Nc :: Int
    "number of vertical levels"
    Nz :: Int
    "panel-edge halo width"
    Hp :: Int
    "vertical halo width"
    Hz :: Int

    "cell-center longitudes per panel [degrees]"
    λᶜ :: NTuple{6, Matrix{FT}}
    "cell-center latitudes per panel [degrees]"
    φᶜ :: NTuple{6, Matrix{FT}}
    "cell-face longitudes per panel (Nc+1 × Nc+1) [degrees]"
    λᶠ :: NTuple{6, Matrix{FT}}
    "cell-face latitudes per panel (Nc+1 × Nc+1) [degrees]"
    φᶠ :: NTuple{6, Matrix{FT}}
    "cell areas per panel [m²]"
    Aᶜ :: NTuple{6, Matrix{FT}}
    "local x metric terms (edge lengths) per panel [m]"
    Δxᶜ :: NTuple{6, Matrix{FT}}
    "local y metric terms (edge lengths) per panel [m]"
    Δyᶜ :: NTuple{6, Matrix{FT}}

    "planet radius [m]"
    radius :: FT
    "gravitational acceleration [m/s²]"
    gravity :: FT
    "reference surface pressure [Pa]"
    reference_pressure :: FT

    "panel connectivity"
    connectivity :: PanelConnectivity
    "vertical coordinate"
    vertical     :: VZ
end

# ---------------------------------------------------------------------------
# Gnomonic projection utilities
# ---------------------------------------------------------------------------

"""
    _gnomonic_xyz(ξ, η, panel)

Map local tangent-plane coordinates `(ξ, η)` to Cartesian `(x, y, z)` on the
unit sphere via the gnomonic (central) projection for the given `panel`.

`ξ = tan(α)`, `η = tan(β)` where `α, β ∈ [-π/4, π/4]` are the angular
coordinates on the cube face.
"""
function _gnomonic_xyz(ξ::FT, η::FT, panel::Int) where FT
    d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
    # Panel rotations following GEOS convention
    if     panel == 1;  return ( d,  ξ*d,  η*d)  # X+ face → 0°E equator
    elseif panel == 2;  return (-ξ*d,  d,  η*d)  # Y+ face → 90°E equator
    elseif panel == 3;  return (-d, -ξ*d,  η*d)  # X- face → 180°E equator
    elseif panel == 4;  return ( ξ*d, -d,  η*d)  # Y- face → 270°E equator
    elseif panel == 5;  return (-η*d,  ξ*d,  d)  # Z+ face → north pole
    else;               return ( η*d,  ξ*d, -d)  # Z- face → south pole
    end
end

"""
    _xyz_to_lonlat(x, y, z)

Convert Cartesian `(x, y, z)` on the unit sphere to `(lon, lat)` in degrees.
"""
@inline function _xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = asind(z / sqrt(x^2 + y^2 + z^2))
    return (lon, lat)
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Construct a gnomonic equidistant cubed-sphere grid with `Nc` cells per
panel edge and `vertical` coordinate.

Grid coordinates, cell areas, and metric terms are computed from the gnomonic
(central) projection following the GEOS/FV3 convention. Cell areas are exact
spherical areas via Girard's theorem.

# Keyword arguments
- `FT`: floating-point type (default `Float64`)
- `Nc`: cells per panel edge (e.g. 48, 90, 180)
- `vertical`: vertical coordinate (e.g. `HybridSigmaPressure`)
- `halo`: `(Hp, Hz)` panel-edge and vertical halo widths (default `(3, 1)`)
- `radius`: planet radius in meters (default Earth)
- `gravity`: gravitational acceleration (default 9.81 m/s²)
- `reference_pressure`: reference surface pressure in Pa (default 101325)
"""
function CubedSphereGrid(arch::AbstractArchitecture;
                         FT::Type = Float64,
                         Nc::Int,
                         vertical::AbstractVerticalCoordinate,
                         halo = (3, 1),
                         radius = FT(6.371e6),
                         gravity = FT(9.81),
                         reference_pressure = FT(101325.0))
    Hp, Hz = halo
    Nz = n_levels(vertical)

    # Angular grid: uniform spacing in α ∈ [-π/4, π/4]
    dα = FT(π) / (2 * Nc)
    α_faces = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    α_centers = [(α_faces[i] + α_faces[i + 1]) / 2 for i in 1:Nc]

    # Pre-allocate per-panel arrays
    λᶜ_panels = ntuple(6) do _; zeros(FT, Nc, Nc) end
    φᶜ_panels = ntuple(6) do _; zeros(FT, Nc, Nc) end
    λᶠ_panels = ntuple(6) do _; zeros(FT, Nc + 1, Nc + 1) end
    φᶠ_panels = ntuple(6) do _; zeros(FT, Nc + 1, Nc + 1) end
    Aᶜ_panels = ntuple(6) do _; zeros(FT, Nc, Nc) end
    Δx_panels  = ntuple(6) do _; zeros(FT, Nc, Nc) end
    Δy_panels  = ntuple(6) do _; zeros(FT, Nc, Nc) end

    R = FT(radius)

    for p in 1:6
        # Face (vertex) coordinates
        for jf in 1:(Nc + 1), if_ in 1:(Nc + 1)
            ξ = tan(α_faces[if_])
            η = tan(α_faces[jf])
            xyz = _gnomonic_xyz(FT(ξ), FT(η), p)
            lon, lat = _xyz_to_lonlat(xyz...)
            λᶠ_panels[p][if_, jf] = FT(lon)
            φᶠ_panels[p][if_, jf] = FT(lat)
        end

        # Cell centers, areas, and metric terms
        for j in 1:Nc, i in 1:Nc
            # Center coordinates
            ξc = tan(α_centers[i])
            ηc = tan(α_centers[j])
            xyz_c = _gnomonic_xyz(FT(ξc), FT(ηc), p)
            lon_c, lat_c = _xyz_to_lonlat(xyz_c...)
            λᶜ_panels[p][i, j] = FT(lon_c)
            φᶜ_panels[p][i, j] = FT(lat_c)

            # Cell area via spherical quadrilateral (exact on unit sphere)
            ξ1, ξ2 = tan(α_faces[i]), tan(α_faces[i + 1])
            η1, η2 = tan(α_faces[j]), tan(α_faces[j + 1])
            v1 = _gnomonic_xyz(FT(ξ1), FT(η1), p)
            v2 = _gnomonic_xyz(FT(ξ2), FT(η1), p)
            v3 = _gnomonic_xyz(FT(ξ2), FT(η2), p)
            v4 = _gnomonic_xyz(FT(ξ1), FT(η2), p)
            Ω = spherical_area_quadrilateral(v1, v2, v3, v4)
            Aᶜ_panels[p][i, j] = R^2 * FT(Ω)

            # Metric terms: great-circle distance between edge midpoints
            # Δx: distance between midpoints of west and east edges
            mid_w = _gnomonic_xyz(FT(ξ1), FT(tan(α_centers[j])), p)
            mid_e = _gnomonic_xyz(FT(ξ2), FT(tan(α_centers[j])), p)
            Δx_panels[p][i, j] = R * FT(spherical_distance(mid_w, mid_e))

            # Δy: distance between midpoints of south and north edges
            mid_s = _gnomonic_xyz(FT(tan(α_centers[i])), FT(η1), p)
            mid_n = _gnomonic_xyz(FT(tan(α_centers[i])), FT(η2), p)
            Δy_panels[p][i, j] = R * FT(spherical_distance(mid_s, mid_n))
        end
    end

    conn = default_panel_connectivity()

    @info "CubedSphereGrid C$Nc: 6×$(Nc)×$(Nc)×$(Nz) cells, " *
          "Δx range ≈ $(round(minimum(minimum.(Δx_panels))/1e3, digits=1))–" *
          "$(round(maximum(maximum.(Δx_panels))/1e3, digits=1)) km"

    return CubedSphereGrid{FT, typeof(arch), typeof(vertical)}(
        arch, Nc, Nz, Hp, Hz,
        λᶜ_panels, φᶜ_panels, λᶠ_panels, φᶠ_panels,
        Aᶜ_panels, Δx_panels, Δy_panels,
        R, FT(gravity), FT(reference_pressure),
        conn, vertical)
end

# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

topology(::CubedSphereGrid) = (CubedPanel(), CubedPanel())

grid_size(g::CubedSphereGrid) = (Nc=g.Nc, Nz=g.Nz, Npanels=6)
halo_size(g::CubedSphereGrid) = (Hp=g.Hp, Hz=g.Hz)

xnode(i, j, g::CubedSphereGrid, ::Center; panel::Int) = g.λᶜ[panel][i, j]
xnode(i, j, g::CubedSphereGrid, ::Face;   panel::Int) = g.λᶠ[panel][i, j]

ynode(i, j, g::CubedSphereGrid, ::Center; panel::Int) = g.φᶜ[panel][i, j]
ynode(i, j, g::CubedSphereGrid, ::Face;   panel::Int) = g.φᶠ[panel][i, j]

function cell_area(i, j, g::CubedSphereGrid; panel::Int)
    return g.Aᶜ[panel][i, j]
end

function Δx(i, j, g::CubedSphereGrid; panel::Int)
    return g.Δxᶜ[panel][i, j]
end

function Δy(i, j, g::CubedSphereGrid; panel::Int)
    return g.Δyᶜ[panel][i, j]
end

"""
$(SIGNATURES)

Pressure at vertical level or interface `k` for reference surface pressure.
"""
function znode(k, g::CubedSphereGrid{FT}, ::Center;
               p_surface::FT = g.reference_pressure) where FT
    return pressure_at_level(g.vertical, k, p_surface)
end

function znode(k, g::CubedSphereGrid{FT}, ::Face;
               p_surface::FT = g.reference_pressure) where FT
    return pressure_at_interface(g.vertical, k, p_surface)
end

"""
$(SIGNATURES)

Pressure thickness (Pa) of vertical level `k`.
"""
function Δz(k, g::CubedSphereGrid{FT};
            p_surface::FT = g.reference_pressure) where FT
    return level_thickness(g.vertical, k, p_surface)
end

"""
$(SIGNATURES)

Volume of cell `(i, j, k)` on panel `panel` as `area × Δp / g`.
"""
function cell_volume(i, j, k, g::CubedSphereGrid{FT};
                     panel::Int,
                     p_surface::FT = g.reference_pressure) where FT
    A = cell_area(i, j, g; panel=panel)
    Δp = level_thickness(g.vertical, k, p_surface)
    return A * Δp / g.gravity
end
