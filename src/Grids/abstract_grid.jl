# ---------------------------------------------------------------------------
# Abstract grid hierarchy and generic accessor functions
#
# ALL physics code must access grid properties through these functions.
# Never index raw grid arrays directly — that breaks grid-agnosticism.
# ---------------------------------------------------------------------------

"""
    AbstractGrid{FT, Arch}

Supertype for all grids. Parametric on:
- `FT`: floating-point type (Float32, Float64)
- `Arch`: architecture (CPU, GPU)
"""
abstract type AbstractGrid{FT, Arch} end

"""
    AbstractStructuredGrid{FT, Arch} <: AbstractGrid{FT, Arch}

Grids with a logically rectangular structure per region/panel.
Both `LatitudeLongitudeGrid` and `CubedSphereGrid` are structured.
"""
abstract type AbstractStructuredGrid{FT, Arch} <: AbstractGrid{FT, Arch} end

# ---------------------------------------------------------------------------
# Required accessor functions — must be implemented for every concrete grid
# ---------------------------------------------------------------------------

"""
    xnode(i, j, grid, loc)

Horizontal x-coordinate at index `(i, j)` and location `loc` (Center or Face).
On lat-lon grids this is longitude; on cubed-sphere it is the local panel x-coordinate.
"""
function xnode end

"""
    ynode(i, j, grid, loc)

Horizontal y-coordinate at index `(i, j)` and location `loc`.
On lat-lon grids this is latitude; on cubed-sphere it is the local panel y-coordinate.
"""
function ynode end

"""
    znode(k, grid, loc)

Vertical coordinate at level `k` and location `loc`.
Returns pressure, sigma, or height depending on the vertical coordinate type.
"""
function znode end

"""
    cell_area(i, j, grid)

Horizontal area of cell `(i, j)` in m².
"""
function cell_area end

"""
    cell_volume(i, j, k, grid)

Volume of cell `(i, j, k)` in m³.
"""
function cell_volume end

"""
    Δx(i, j, grid)

Horizontal grid spacing in the x-direction at `(i, j)` in meters.
"""
function Δx end

"""
    Δy(i, j, grid)

Horizontal grid spacing in the y-direction at `(i, j)` in meters.
"""
function Δy end

"""
    Δz(k, grid)

Vertical grid spacing at level `k`.
Units depend on vertical coordinate (Pa for pressure-based, m for height-based).
"""
function Δz end

# ---------------------------------------------------------------------------
# Required metadata functions
# ---------------------------------------------------------------------------

"""
    topology(grid)

Return a tuple of `AbstractTopology` types describing each dimension.
E.g. `(Periodic(), Bounded())` for a lat-lon grid.
"""
function topology end

"""
    halo_size(grid)

Return a `NamedTuple` `(Hx=..., Hy=..., Hz=...)` with halo widths.
"""
function halo_size end

"""
    grid_size(grid)

Return a `NamedTuple` `(Nx=..., Ny=..., Nz=...)` with interior grid dimensions.
"""
function grid_size end

# ---------------------------------------------------------------------------
# Fallback: architecture extraction
# ---------------------------------------------------------------------------

"""Extract architecture from a grid via its type parameter."""
architecture(grid::AbstractGrid{FT, Arch}) where {FT, Arch} = Arch()

"""Extract float type from a grid via its type parameter."""
floattype(::AbstractGrid{FT, Arch}) where {FT, Arch} = FT
