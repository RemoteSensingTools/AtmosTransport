# ---------------------------------------------------------------------------
# LatitudeLongitudeGrid
#
# Regular latitude-longitude grid, native to ERA5 and TM5.
# Supports TM5 resolutions: 6°×4°, 3°×2°, 1°×1° and arbitrary spacing.
#
# Topology: Periodic in longitude, Bounded in latitude.
# Halo regions wrap periodically in longitude; latitude halos use extrapolation
# or reflection depending on the boundary condition.
#
# Vertical coordinate: Uses AbstractVerticalCoordinate (e.g. HybridSigmaPressure)
# for pressure-based levels. Pressure at level k depends on reference surface
# pressure via the vertical coordinate's interface.
# ---------------------------------------------------------------------------

"""Earth radius in meters (WGS84 equatorial)."""
const EARTH_RADIUS = 6.371e6

"""Standard surface pressure in Pascals (1013.25 hPa)."""
const REFERENCE_SURFACE_PRESSURE = 101325.0

"""Gravitational acceleration in m/s²."""
const GRAVITY = 9.81

"""
    LatitudeLongitudeGrid{FT, Arch, VZ} <: AbstractStructuredGrid{FT, Arch}

Regular latitude-longitude grid with hybrid sigma-pressure vertical coordinate.

# Constructor

    LatitudeLongitudeGrid(arch::AbstractArchitecture;
        FT = Float64,
        size,                     # (Nx, Ny, Nz)
        longitude = (-180, 180),  # (λ_west, λ_east)
        latitude  = (-90, 90),    # (φ_south, φ_north)
        vertical,                 # AbstractVerticalCoordinate
        halo = (3, 3, 1))        # (Hx, Hy, Hz)

# Fields
- `architecture` — CPU or GPU
- `Nx, Ny, Nz` — number of interior cells
- `Hx, Hy, Hz` — halo widths
- `λᶜ, λᶠ` — longitude at cell centers / faces
- `φᶜ, φᶠ` — latitude at cell centers / faces
- `Δλ` — longitude spacing (uniform)
- `Δφ` — latitude spacing (uniform)
- `vertical` — vertical coordinate (e.g. HybridSigmaPressure)

# Accessor functions
- `xnode`, `ynode` — horizontal coordinates (longitude, latitude)
- `znode` — vertical coordinate (pressure at level/interface center)
- `cell_area`, `cell_volume` — horizontal area and 3D cell volume
- `Δx`, `Δy`, `Δz` — grid spacing in x, y, and vertical (pressure thickness)
"""
struct LatitudeLongitudeGrid{FT, Arch, VZ} <: AbstractStructuredGrid{FT, Arch}
    architecture :: Arch

    Nx :: Int
    Ny :: Int
    Nz :: Int

    Hx :: Int
    Hy :: Int
    Hz :: Int

    λᶜ :: Vector{FT}    # cell center longitudes  (length Nx)
    λᶠ :: Vector{FT}    # cell face longitudes    (length Nx+1)
    φᶜ :: Vector{FT}    # cell center latitudes   (length Ny)
    φᶠ :: Vector{FT}    # cell face latitudes     (length Ny+1)

    Δλ :: FT             # uniform lon spacing (degrees)
    Δφ :: FT             # uniform lat spacing (degrees)

    vertical :: VZ
end

# Placeholder constructor — implementation will be filled by a parallel agent
function LatitudeLongitudeGrid(arch::AbstractArchitecture;
                               FT::Type = Float64,
                               size::Tuple{Int,Int,Int},
                               longitude = (-180, 180),
                               latitude = (-90, 90),
                               vertical::AbstractVerticalCoordinate,
                               halo = (3, 3, 1))
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    λ_west, λ_east   = FT.(longitude)
    φ_south, φ_north  = FT.(latitude)

    Δλ = (λ_east - λ_west) / Nx
    Δφ = (φ_north - φ_south) / Ny

    λᶠ = collect(range(λ_west,  λ_east,  length=Nx+1) .|> FT)
    λᶜ = [(λᶠ[i] + λᶠ[i+1]) / 2 for i in 1:Nx] .|> FT
    φᶠ = collect(range(φ_south, φ_north, length=Ny+1) .|> FT)
    φᶜ = [(φᶠ[j] + φᶠ[j+1]) / 2 for j in 1:Ny] .|> FT

    return LatitudeLongitudeGrid{FT, typeof(arch), typeof(vertical)}(
        arch, Nx, Ny, Nz, Hx, Hy, Hz,
        λᶜ, λᶠ, φᶜ, φᶠ, Δλ, Δφ, vertical)
end

# ---------------------------------------------------------------------------
# Accessor function implementations
# ---------------------------------------------------------------------------

"""Topology: periodic in longitude, bounded in latitude."""
topology(::LatitudeLongitudeGrid) = (Periodic(), Bounded())

"""Interior dimensions (Nx, Ny, Nz)."""
grid_size(g::LatitudeLongitudeGrid) = (Nx=g.Nx, Ny=g.Ny, Nz=g.Nz)

"""Halo widths (Hx, Hy, Hz) on each side of the interior."""
halo_size(g::LatitudeLongitudeGrid) = (Hx=g.Hx, Hy=g.Hy, Hz=g.Hz)

"""Longitude at (i, j). Center → cell center; Face → cell face."""
xnode(i, j, g::LatitudeLongitudeGrid, ::Center) = g.λᶜ[i]
xnode(i, j, g::LatitudeLongitudeGrid, ::Face)   = g.λᶠ[i]

"""Latitude at (i, j). Center → cell center; Face → cell face."""
ynode(i, j, g::LatitudeLongitudeGrid, ::Center) = g.φᶜ[j]
ynode(i, j, g::LatitudeLongitudeGrid, ::Face)   = g.φᶠ[j]

"""Horizontal x-spacing (m) at cell center (i, j). Depends on latitude via cos(φ)."""
function Δx(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    R = FT(EARTH_RADIUS)
    return R * cosd(g.φᶜ[j]) * deg2rad(g.Δλ)
end

"""Horizontal y-spacing (m) at cell center (i, j). Uniform in latitude."""
function Δy(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    R = FT(EARTH_RADIUS)
    return R * deg2rad(g.Δφ)
end

"""Horizontal cell area (m²) at (i, j). Varies with latitude; larger near equator."""
function cell_area(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    R = FT(EARTH_RADIUS)
    return R^2 * deg2rad(g.Δλ) * abs(sind(g.φᶠ[j+1]) - sind(g.φᶠ[j]))
end

# ---------------------------------------------------------------------------
# Vertical coordinate and cell volume
# ---------------------------------------------------------------------------

"""
    znode(k, g::LatitudeLongitudeGrid, loc; p_surface=REFERENCE_SURFACE_PRESSURE)

Pressure (Pa) at vertical level or interface `k` for reference surface pressure.

- `Center()`: pressure at level center k (average of bounding interfaces)
- `Face()`: pressure at interface k (k = 1 is top, k = Nz+1 is surface)
"""
function znode(k, g::LatitudeLongitudeGrid{FT}, ::Center; p_surface::FT=FT(REFERENCE_SURFACE_PRESSURE)) where FT
    return pressure_at_level(g.vertical, k, p_surface)
end

function znode(k, g::LatitudeLongitudeGrid{FT}, ::Face; p_surface::FT=FT(REFERENCE_SURFACE_PRESSURE)) where FT
    return pressure_at_interface(g.vertical, k, p_surface)
end

"""
    Δz(k, g::LatitudeLongitudeGrid; p_surface=REFERENCE_SURFACE_PRESSURE)

Pressure thickness (Pa) of vertical level k.
"""
function Δz(k, g::LatitudeLongitudeGrid{FT}; p_surface::FT=FT(REFERENCE_SURFACE_PRESSURE)) where FT
    return level_thickness(g.vertical, k, p_surface)
end

"""
    cell_volume(i, j, k, g::LatitudeLongitudeGrid; p_surface=REFERENCE_SURFACE_PRESSURE)

Volume of cell (i, j, k) as pressure thickness × area / g.

Under hydrostatic balance this gives mass per cell (kg). Uses the vertical
coordinate's level thickness and horizontal cell area.
"""
function cell_volume(i, j, k, g::LatitudeLongitudeGrid{FT}; p_surface::FT=FT(REFERENCE_SURFACE_PRESSURE)) where FT
    area = cell_area(i, j, g)
    Δp = level_thickness(g.vertical, k, p_surface)
    return area * Δp / FT(GRAVITY)
end
