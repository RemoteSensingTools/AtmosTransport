# ---------------------------------------------------------------------------
# CubedSphereGrid
#
# Gnomonic equidistant cubed-sphere grid, native to GEOS-FP and MERRA-2.
# Six panels, each Nc × Nc cells. Avoids polar singularity.
#
# Resolutions follow the GEOS convention: C48, C90, C180, C360.
# Grid generation will use CubedSphere.jl (CliMA).
#
# This file defines the type and accessor stubs. The full constructor
# (with CubedSphere.jl integration) will be filled by a parallel agent.
# ---------------------------------------------------------------------------

"""
    PanelConnectivity

Describes how the 6 panels of a cubed-sphere connect at their edges.
Each panel has 4 neighbors (north, south, east, west), identified by
panel index and orientation (rotation needed to align axes).
"""
struct PanelConnectivity
    neighbors :: NTuple{6, NTuple{4, @NamedTuple{panel::Int, orientation::Int}}}
end

"""
    CubedSphereGrid{FT, Arch, VZ} <: AbstractStructuredGrid{FT, Arch}

Gnomonic cubed-sphere grid with 6 panels of `Nc × Nc` cells.

# Fields
- `architecture` — CPU or GPU
- `Nc` — cells per panel edge (e.g. 90 for C90)
- `Nz` — number of vertical levels
- `Hp` — panel-edge halo width
- `Hz` — vertical halo width
- `λᶜ` — cell-center longitudes per panel, `NTuple{6, Matrix{FT}}`
- `φᶜ` — cell-center latitudes per panel, `NTuple{6, Matrix{FT}}`
- `Aᶜ` — cell areas per panel, `NTuple{6, Matrix{FT}}`
- `Δxᶜ, Δyᶜ` — local metric terms per panel
- `connectivity` — `PanelConnectivity`
- `vertical` — `AbstractVerticalCoordinate`
"""
struct CubedSphereGrid{FT, Arch, VZ} <: AbstractStructuredGrid{FT, Arch}
    architecture :: Arch

    Nc :: Int    # cells per panel edge
    Nz :: Int
    Hp :: Int    # panel-edge halo
    Hz :: Int    # vertical halo

    λᶜ :: NTuple{6, Matrix{FT}}
    φᶜ :: NTuple{6, Matrix{FT}}
    Aᶜ :: NTuple{6, Matrix{FT}}
    Δxᶜ :: NTuple{6, Matrix{FT}}
    Δyᶜ :: NTuple{6, Matrix{FT}}

    connectivity :: PanelConnectivity
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
