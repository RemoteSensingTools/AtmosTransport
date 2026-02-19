# ---------------------------------------------------------------------------
# Abstract grid hierarchy and generic accessor functions
#
# ALL physics code must access grid properties through these functions.
# Never index raw grid arrays directly — that breaks grid-agnosticism.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for all grids. Parametric on:
- `FT`: floating-point type (Float32, Float64)
- `Arch`: architecture (CPU, GPU)
"""
abstract type AbstractGrid{FT, Arch} end

"""
$(TYPEDEF)

Grids with a logically rectangular structure per region/panel.
Both `LatitudeLongitudeGrid` and `CubedSphereGrid` are structured.
"""
abstract type AbstractStructuredGrid{FT, Arch} <: AbstractGrid{FT, Arch} end

# ---------------------------------------------------------------------------
# Required accessor functions — must be implemented for every concrete grid
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Horizontal x-coordinate at index `(i, j)` and location `loc` (Center or Face).
On lat-lon grids this is longitude; on cubed-sphere it is the local panel x-coordinate.
"""
function xnode end

"""
$(SIGNATURES)

Horizontal y-coordinate at index `(i, j)` and location `loc`.
On lat-lon grids this is latitude; on cubed-sphere it is the local panel y-coordinate.
"""
function ynode end

"""
$(SIGNATURES)

Vertical coordinate at level `k` and location `loc`.
Returns pressure, sigma, or height depending on the vertical coordinate type.
"""
function znode end

"""
$(SIGNATURES)

Horizontal area of cell `(i, j)` in m².
"""
function cell_area end

"""
$(SIGNATURES)

Volume of cell `(i, j, k)` in m³.
"""
function cell_volume end

"""
$(SIGNATURES)

Horizontal grid spacing in the x-direction at `(i, j)` in meters.
"""
function Δx end

"""
$(SIGNATURES)

Horizontal grid spacing in the y-direction at `(i, j)` in meters.
"""
function Δy end

"""
$(SIGNATURES)

Vertical grid spacing at level `k`.
Units depend on vertical coordinate (Pa for pressure-based, m for height-based).
"""
function Δz end

# ---------------------------------------------------------------------------
# Required metadata functions
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Return a tuple of `AbstractTopology` types describing each dimension.
E.g. `(Periodic(), Bounded())` for a lat-lon grid.
"""
function topology end

"""
$(SIGNATURES)

Return a `NamedTuple` `(Hx=..., Hy=..., Hz=...)` with halo widths.
"""
function halo_size end

"""
$(SIGNATURES)

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
