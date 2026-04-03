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

"""
$(TYPEDEF)

Regular latitude-longitude grid with hybrid sigma-pressure vertical coordinate.

Planet constants (radius, gravity, reference pressure) are stored in the grid
so that all accessor functions are self-contained and type-stable.  Values come
from `config/defaults.toml` via `ModelParameters`, or can be overridden in the
constructor.

# Accessor functions
- `xnode`, `ynode` — horizontal coordinates (longitude, latitude)
- `znode` — vertical coordinate (pressure at level/interface center)
- `cell_area`, `cell_volume` — horizontal area and 3D cell volume
- `Δx`, `Δy`, `Δz` — grid spacing in x, y, and vertical (pressure thickness)

$(FIELDS)
"""
struct LatitudeLongitudeGrid{FT, Arch, VZ, V, RG} <: AbstractStructuredGrid{FT, Arch}
    "compute architecture (CPU or GPU)"
    architecture :: Arch

    "number of grid cells in x (longitude)"
    Nx :: Int
    "number of grid cells in y (latitude)"
    Ny :: Int
    "number of vertical levels"
    Nz :: Int

    "halo width in x"
    Hx :: Int
    "halo width in y"
    Hy :: Int
    "halo width in z"
    Hz :: Int

    "cell-center longitudes — device array (length Nx)"
    λᶜ :: V
    "cell-face longitudes — device array (length Nx+1)"
    λᶠ :: V
    "cell-center latitudes — device array (length Ny)"
    φᶜ :: V
    "cell-face latitudes — device array (length Ny+1)"
    φᶠ :: V

    "uniform longitude spacing [degrees]"
    Δλ :: FT
    "uniform latitude spacing [degrees]"
    Δφ :: FT

    "vertical coordinate (e.g. HybridSigmaPressure)"
    vertical :: VZ

    "planet radius [m]"
    radius             :: FT
    "gravitational acceleration [m/s²]"
    gravity            :: FT
    "reference surface pressure [Pa]"
    reference_pressure :: FT

    "reduced grid specification for polar CFL handling, or `nothing`"
    reduced_grid :: RG

    "CPU-cached cell-center longitudes for host-side scalar access"
    λᶜ_cpu :: Vector{FT}
    "CPU-cached cell-face longitudes for host-side scalar access"
    λᶠ_cpu :: Vector{FT}
    "CPU-cached cell-center latitudes for host-side scalar access"
    φᶜ_cpu :: Vector{FT}
    "CPU-cached cell-face latitudes for host-side scalar access"
    φᶠ_cpu :: Vector{FT}
end

# Placeholder constructor — implementation will be filled by a parallel agent
function LatitudeLongitudeGrid(arch::AbstractArchitecture;
                               FT::Type = Float64,
                               size::Tuple{Int,Int,Int},
                               longitude = (-180, 180),
                               latitude = (-90, 90),
                               vertical::AbstractVerticalCoordinate,
                               halo = (3, 3, 1),
                               radius             = FT(6.371e6),
                               gravity             = FT(9.81),
                               reference_pressure  = FT(101325.0),
                               use_reduced_grid    = :auto)
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    λ_west, λ_east   = FT.(longitude)
    φ_south, φ_north  = FT.(latitude)

    Δλ = (λ_east - λ_west) / Nx
    Δφ = (φ_north - φ_south) / Ny

    ArrayType = array_type(arch)

    λᶠ_cpu = collect(range(λ_west,  λ_east,  length=Nx+1) .|> FT)
    λᶜ_cpu = FT[(λᶠ_cpu[i] + λᶠ_cpu[i+1]) / 2 for i in 1:Nx]
    φᶠ_cpu = collect(range(φ_south, φ_north, length=Ny+1) .|> FT)
    φᶜ_cpu = FT[(φᶠ_cpu[j] + φᶠ_cpu[j+1]) / 2 for j in 1:Ny]

    rg = if use_reduced_grid === :auto
        Δλ <= 1.0 ? compute_reduced_grid_tm5(Nx, φᶜ_cpu) : nothing
    elseif use_reduced_grid === true
        compute_reduced_grid_tm5(Nx, φᶜ_cpu)
    else
        nothing
    end

    if rg !== nothing
        n_reduced = count(>(1), rg.cluster_sizes)
        max_r = maximum(rg.cluster_sizes)
        @info "Reduced grid enabled: $n_reduced of $Ny latitudes reduced (max cluster=$max_r)"
    end

    λᶠ = ArrayType(λᶠ_cpu)
    λᶜ = ArrayType(λᶜ_cpu)
    φᶠ = ArrayType(φᶠ_cpu)
    φᶜ = ArrayType(φᶜ_cpu)

    return LatitudeLongitudeGrid{FT, typeof(arch), typeof(vertical), typeof(λᶜ), typeof(rg)}(
        arch, Nx, Ny, Nz, Hx, Hy, Hz,
        λᶜ, λᶠ, φᶜ, φᶠ, Δλ, Δφ, vertical,
        FT(radius), FT(gravity), FT(reference_pressure), rg,
        λᶜ_cpu, λᶠ_cpu, φᶜ_cpu, φᶠ_cpu)
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
xnode(i, j, g::LatitudeLongitudeGrid, ::Center) = @inbounds g.λᶜ_cpu[i]
xnode(i, j, g::LatitudeLongitudeGrid, ::Face)   = @inbounds g.λᶠ_cpu[i]

"""Latitude at (i, j). Center → cell center; Face → cell face."""
ynode(i, j, g::LatitudeLongitudeGrid, ::Center) = @inbounds g.φᶜ_cpu[j]
ynode(i, j, g::LatitudeLongitudeGrid, ::Face)   = @inbounds g.φᶠ_cpu[j]

"""Horizontal x-spacing (m) at cell center (i, j). Uses cell-center latitude φᶜ (TM5 convention).
At the poles the grid cell collapses in x; clamped to Δy to prevent CFL violation
and division-by-zero. Physical fluxes near poles are small due to v·cos(φ) → 0."""
function Δx(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    R = g.radius
    dx = R * cosd(g.φᶜ_cpu[j]) * deg2rad(g.Δλ)
    dx_min = R * deg2rad(g.Δφ)
    return max(dx, dx_min)
end

"""Horizontal y-spacing (m) at cell center (i, j). Uniform in latitude."""
function Δy(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    return g.radius * deg2rad(g.Δφ)
end

"""Horizontal cell area (m²) at (i, j). Varies with latitude; larger near equator."""
function cell_area(i, j, g::LatitudeLongitudeGrid{FT}) where FT
    R = g.radius
    return R^2 * deg2rad(g.Δλ) * abs(sind(g.φᶠ_cpu[j+1]) - sind(g.φᶠ_cpu[j]))
end

# ---------------------------------------------------------------------------
# Vertical coordinate and cell volume
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Pressure (Pa) at vertical level or interface `k` for reference surface pressure.
"""
function znode(k, g::LatitudeLongitudeGrid{FT}, ::Center;
               p_surface::FT = g.reference_pressure) where FT
    return pressure_at_level(g.vertical, k, p_surface)
end

function znode(k, g::LatitudeLongitudeGrid{FT}, ::Face;
               p_surface::FT = g.reference_pressure) where FT
    return pressure_at_interface(g.vertical, k, p_surface)
end

"""
$(TYPEDSIGNATURES)

Pressure thickness (Pa) of vertical level k.
"""
function Δz(k, g::LatitudeLongitudeGrid{FT};
            p_surface::FT = g.reference_pressure) where FT
    return level_thickness(g.vertical, k, p_surface)
end

"""
$(TYPEDSIGNATURES)

Volume of cell (i, j, k) as pressure thickness × area / g.
Under hydrostatic balance this gives mass per cell (kg).
"""
function cell_volume(i, j, k, g::LatitudeLongitudeGrid{FT};
                     p_surface::FT = g.reference_pressure) where FT
    area = cell_area(i, j, g)
    Δp = level_thickness(g.vertical, k, p_surface)
    return area * Δp / g.gravity
end
