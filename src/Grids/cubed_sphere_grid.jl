# ---------------------------------------------------------------------------
# CubedSphereGrid
#
# Conformal cubed-sphere grid (via CubedSphere.jl), compatible with GEOS-FP
# and MERRA-2 conventions. Six panels, each Nc × Nc cells. Avoids polar singularity.
#
# Resolutions follow the GEOS convention: C48, C90, C180, C360, C720.
# Panel numbering (GEOS): 1=front (0°E), 2=east (90°E), 3=back (180°E),
# 4=west (90°W), 5=north, 6=south.
# ---------------------------------------------------------------------------

using CubedSphere
using CubedSphere.SphericalGeometry: conformal_cubed_sphere_mapping, cartesian_to_lat_lon,
    compute_cell_areas, spherical_distance

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

Gnomonic cubed-sphere grid with 6 panels of `Nc × Nc` cells.

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

    "cell-center longitudes per panel"
    λᶜ :: NTuple{6, Matrix{FT}}
    "cell-center latitudes per panel"
    φᶜ :: NTuple{6, Matrix{FT}}
    "cell areas per panel"
    Aᶜ :: NTuple{6, Matrix{FT}}
    "local x metric terms per panel"
    Δxᶜ :: NTuple{6, Matrix{FT}}
    "local y metric terms per panel"
    Δyᶜ :: NTuple{6, Matrix{FT}}

    "panel connectivity"
    connectivity :: PanelConnectivity
    "vertical coordinate"
    vertical     :: VZ
end

# ---------------------------------------------------------------------------
# Placeholder constructor — to be implemented with CubedSphere.jl
# ---------------------------------------------------------------------------

function CubedSphereGrid(arch::AbstractArchitecture;
                         FT::Type = Float64,
                         Nc::Int,
                         vertical::AbstractVerticalCoordinate,
                         halo = (3, 1))
    error("CubedSphereGrid constructor not yet implemented. " *
          "This will use CubedSphere.jl for grid generation.")
end

# ---------------------------------------------------------------------------
# Accessor stubs — to be implemented by parallel agent
# ---------------------------------------------------------------------------

topology(::CubedSphereGrid) = (CubedPanel(), CubedPanel())

grid_size(g::CubedSphereGrid) = (Nc=g.Nc, Nz=g.Nz, Npanels=6)
halo_size(g::CubedSphereGrid) = (Hp=g.Hp, Hz=g.Hz)

function xnode(i, j, g::CubedSphereGrid, loc; panel::Int)
    return g.λᶜ[panel][i, j]  # simplified; real version needs loc dispatch
end

function ynode(i, j, g::CubedSphereGrid, loc; panel::Int)
    return g.φᶜ[panel][i, j]
end

function cell_area(i, j, g::CubedSphereGrid; panel::Int)
    return g.Aᶜ[panel][i, j]
end

function Δx(i, j, g::CubedSphereGrid; panel::Int)
    return g.Δxᶜ[panel][i, j]
end

function Δy(i, j, g::CubedSphereGrid; panel::Int)
    return g.Δyᶜ[panel][i, j]
end
