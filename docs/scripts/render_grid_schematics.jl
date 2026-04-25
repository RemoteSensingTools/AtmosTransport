#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Render the three grid-topology schematics shipped with the docs:
#
#   docs/src/assets/grids/latlon.png             — small LL cell grid
#   docs/src/assets/grids/reduced_gaussian.png   — RG rings with variable cells
#   docs/src/assets/grids/cubed_sphere.png       — six panels under both
#                                                  GnomonicPanelConvention and
#                                                  GEOSNativePanelConvention.
#
# Run from the repo root with the docs project active:
#
#   julia --project=docs docs/scripts/render_grid_schematics.jl
#
# Re-run only when the underlying topology / labelling changes — the PNGs
# are committed.
# ---------------------------------------------------------------------------

using CairoMakie
using AtmosTransport
using AtmosTransport.Grids: LatLonMesh, CubedSphereMesh,
                              GnomonicPanelConvention, GEOSNativePanelConvention

const ASSET_DIR = joinpath(@__DIR__, "..", "src", "assets", "grids")
mkpath(ASSET_DIR)

const FT = Float64

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _draw_cell_rect!(ax, x0, x1, y0, y1; color = :white, strokecolor = :gray40)
    poly!(ax, [Point2f(x0, y0), Point2f(x1, y0),
               Point2f(x1, y1), Point2f(x0, y1)];
          color = color, strokecolor = strokecolor, strokewidth = 0.8)
end

# ---------------------------------------------------------------------------
# 1. LatLon — small 12 × 6 grid
# ---------------------------------------------------------------------------

function render_latlon!(path::AbstractString)
    Nx, Ny = 12, 6
    fig = Figure(size = (720, 360))
    ax  = Axis(fig[1, 1];
               title  = "LatLonMesh: $(Nx) × $(Ny) regular lat-lon (cells)",
               xlabel = "longitude (°)", ylabel = "latitude (°)",
               aspect = DataAspect())
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    for i in 1:Nx, j in 1:Ny
        _draw_cell_rect!(ax, -180 + (i-1)*Δλ, -180 + i*Δλ,
                              -90  + (j-1)*Δφ, -90 + j*Δφ)
    end
    # Highlight cell (i,j) = (5,4) and label its center
    i, j = 5, 4
    cx = -180 + (i - 0.5) * Δλ
    cy = -90  + (j - 0.5) * Δφ
    _draw_cell_rect!(ax, -180 + (i-1)*Δλ, -180 + i*Δλ,
                          -90 + (j-1)*Δφ, -90 + j*Δφ;
                     color = (:steelblue, 0.6))
    scatter!(ax, [Point2f(cx, cy)]; color = :black, markersize = 7)
    text!(ax, "(i, j) = (5, 4)\nλᶜ=$(round(cx; digits=1))°  φᶜ=$(round(cy; digits=1))°";
          position = Point2f(cx + 8, cy), align = (:left, :center), fontsize = 11)
    xlims!(ax, -190, 190); ylims!(ax, -100, 100)
    save(path, fig; px_per_unit = 2)
    return path
end

# ---------------------------------------------------------------------------
# 2. Reduced Gaussian — synthetic small N (rings of variable longitude count)
# ---------------------------------------------------------------------------
#
# We do not need a real ECMWF reduced-Gaussian table for the schematic —
# the visual point is "more cells per ring near the equator, fewer near
# the poles". A small synthetic distribution is clearer than an O90
# profile.
# ---------------------------------------------------------------------------

function render_reduced_gaussian!(path::AbstractString)
    nlon_per_ring = [4, 8, 14, 18, 20, 18, 14, 8, 4]
    nrings = length(nlon_per_ring)
    lats = collect(range(-80.0, 80.0; length = nrings))     # ring centers
    Δlat = abs(lats[2] - lats[1])

    fig = Figure(size = (720, 360))
    ax = Axis(fig[1, 1];
              title  = "ReducedGaussianMesh: 9 rings, " *
                       "nlon_per_ring = $(nlon_per_ring)",
              xlabel = "longitude (°)", ylabel = "latitude (°)",
              aspect = DataAspect())

    # Light gray ring boundaries for context
    for j in 1:nrings
        y0 = lats[j] - Δlat/2
        y1 = lats[j] + Δlat/2
        lines!(ax, [-180.0, 180.0], [y0, y0]; color = :gray70, linewidth = 0.5)
        # Cells in this ring
        Δλ = 360 / nlon_per_ring[j]
        for i in 1:nlon_per_ring[j]
            _draw_cell_rect!(ax, -180 + (i-1)*Δλ, -180 + i*Δλ, y0, y1;
                             color = :white)
        end
        # Equator gets a stronger highlight
        if abs(lats[j]) < 5
            for i in 1:nlon_per_ring[j]
                Δλ = 360 / nlon_per_ring[j]
                _draw_cell_rect!(ax, -180 + (i-1)*Δλ, -180 + i*Δλ, y0, y1;
                                 color = (:steelblue, 0.4))
            end
        end
        text!(ax, "$(nlon_per_ring[j]) cells";
              position = Point2f(195, lats[j]),
              align = (:left, :center), fontsize = 9)
    end
    xlims!(ax, -190, 260); ylims!(ax, -100, 100)
    save(path, fig; px_per_unit = 2)
    return path
end

# ---------------------------------------------------------------------------
# 3. Cubed-sphere — 6 panels in an unfolded "cross" layout, two conventions
#    side by side.
# ---------------------------------------------------------------------------
#
# Layout (cross), face index 1..6:
#
#               +----+
#               |  N |
#       +----+--+----+----+----+
#       |  4 |  1 |  2 |  3 |   each ⊂ equator
#       +----+--+----+----+----+
#               |  S |
#               +----+
#
# We render two such crosses with the panel labels chosen by each
# convention (gnomonic vs GEOS-native) and overlay the index number.
# The label on each tile is "panel <p>: <role>".

function _gnomonic_role(p::Int)
    p == 5 && return "north"
    p == 6 && return "south"
    return "equatorial"
end

function _geos_role(p::Int)
    p == 3 && return "north"
    p == 6 && return "south"
    return "equatorial"
end

function _draw_cross!(ax, indices, roles; tilesize = 1.0,
                      color_eq = (:steelblue, 0.25),
                      color_n  = (:firebrick, 0.25),
                      color_s  = (:goldenrod, 0.25))
    # indices: NTuple{6,Int} — physical layout (panel index per slot)
    # slot order: (north, west, center1, center2, east, south)
    slot_xy = [
        (1.0, 2.0),   # north (slot 1)
        (-1.0, 1.0),  # west  (slot 2)
        (0.0, 1.0),   # center1
        (1.0, 1.0),   # center2
        (2.0, 1.0),   # east
        (1.0, 0.0),   # south
    ]
    color_for(role) = role == "north" ? color_n :
                       role == "south" ? color_s :
                       color_eq
    for (slot, (x, y)) in enumerate(slot_xy)
        p = indices[slot]
        role = roles[slot]
        c = color_for(role)
        _draw_cell_rect!(ax, x*tilesize, (x+1)*tilesize, y*tilesize, (y+1)*tilesize;
                         color = c, strokecolor = :black)
        text!(ax, "panel $(p)\n($(role))";
              position = Point2f((x + 0.5)*tilesize, (y + 0.5)*tilesize),
              align = (:center, :center), fontsize = 11)
    end
end

function render_cubed_sphere!(path::AbstractString)
    fig = Figure(size = (820, 380))

    # Slot order: (north, west, center1, center2, east, south)
    # Gnomonic: equatorial = panels 1,2,3,4 (panels 1+2 in centers,
    # panel 4 west, panel 3 east); 5 north; 6 south.
    gnomonic_indices = (5, 4, 1, 2, 3, 6)
    gnomonic_roles   = ("north", "equatorial", "equatorial",
                        "equatorial", "equatorial", "south")

    # GEOS-native: equatorial = panels 1,2,4,5 (1+2 centers, 5 east,
    # 4 west); 3 north; 6 south.
    geos_indices     = (3, 4, 1, 2, 5, 6)
    geos_roles       = ("north", "equatorial", "equatorial",
                        "equatorial", "equatorial", "south")

    ax1 = Axis(fig[1, 1];
               title = "CubedSphereMesh — GnomonicPanelConvention",
               aspect = DataAspect())
    ax2 = Axis(fig[1, 2];
               title = "CubedSphereMesh — GEOSNativePanelConvention",
               aspect = DataAspect())

    for ax in (ax1, ax2)
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, -1.2, 3.2); ylims!(ax, -0.2, 3.2)
    end
    _draw_cross!(ax1, gnomonic_indices, gnomonic_roles)
    _draw_cross!(ax2, geos_indices,     geos_roles)
    save(path, fig; px_per_unit = 2)
    return path
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

paths = (
    render_latlon!(            joinpath(ASSET_DIR, "latlon.png")),
    render_reduced_gaussian!(  joinpath(ASSET_DIR, "reduced_gaussian.png")),
    render_cubed_sphere!(      joinpath(ASSET_DIR, "cubed_sphere.png")),
)

println.(paths)
