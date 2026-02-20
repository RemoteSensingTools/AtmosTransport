#!/usr/bin/env julia
#
# Generate diagrams for grid documentation page.
# Output: docs/src/assets/grid_latlon.png
#         docs/src/assets/grid_cubedsphere.png
#         docs/src/assets/grid_hybrid_sigma.png
#         docs/src/assets/grid_reduced.png

using CairoMakie

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

# =========================================================================
# Diagram 1: Latitude-Longitude grid
# =========================================================================
function diagram_latlon()
    fig = Figure(size=(700, 400), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Longitude [°]", ylabel="Latitude [°]",
        title="LatitudeLongitudeGrid (1° × 1° shown)",
        aspect=DataAspect())

    Δλ, Δφ = 15, 15
    lons = -180:Δλ:180
    lats = -90:Δφ:90

    for λ in lons
        lines!(ax, [λ, λ], [-90, 90]; color=(:gray60, 0.5), linewidth=0.8)
    end
    for φ in lats
        lines!(ax, [-180, 180], [φ, φ]; color=(:gray60, 0.5), linewidth=0.8)
    end

    # Highlight one cell
    cx, cy = 30, 45
    poly!(ax, Rect(cx, cy, Δλ, Δφ); color=(:steelblue, 0.3))
    lines!(ax, [cx, cx+Δλ, cx+Δλ, cx, cx],
               [cy, cy, cy+Δφ, cy+Δφ, cy]; color=:steelblue, linewidth=2)
    scatter!(ax, [cx + Δλ/2], [cy + Δφ/2]; color=:red, markersize=8)
    text!(ax, cx + Δλ/2 + 2, cy + Δφ/2 + 2; text="(i, j)", fontsize=11,
          color=:red)

    # Annotations
    text!(ax, -170, -80; text="Periodic in λ", fontsize=11, color=:blue)
    text!(ax, -170, -70; text="Bounded in φ", fontsize=11, color=:blue)

    # Pole convergence arrows
    for λ in -150:30:150
        arrows!(ax, [λ], [85.0], [0.0], [3.0]; color=(:red, 0.4), linewidth=1.5)
        arrows!(ax, [λ], [-85.0], [0.0], [-3.0]; color=(:red, 0.4), linewidth=1.5)
    end
    text!(ax, 60, 80; text="Cells converge\nat poles", fontsize=10,
          color=(:red, 0.7))

    xlims!(ax, -185, 185); ylims!(ax, -95, 95)

    save(joinpath(ASSETS, "grid_latlon.png"), fig; px_per_unit=2.5)
    @info "Saved grid_latlon.png"
end

# =========================================================================
# Diagram 2: Cubed-sphere grid (unfolded)
# =========================================================================
function diagram_cubedsphere()
    fig = Figure(size=(800, 600), fontsize=13)
    ax = Axis(fig[1, 1]; title="CubedSphereGrid — Unfolded Layout (6 panels)",
              aspect=DataAspect())
    hidedecorations!(ax)

    # Cross-shaped layout: top=5, left=4, center=1, right=2, bottom=6, back=3
    panel_positions = [
        (1, 1, "Panel 1\n(Front)"),     # center
        (2, 1, "Panel 2\n(East)"),      # right of center
        (1, 2, "Panel 5\n(North)"),     # top of center
        (0, 1, "Panel 4\n(West)"),      # left of center
        (1, 0, "Panel 6\n(South)"),     # bottom of center
        (3, 1, "Panel 3\n(Back)"),      # far right
    ]

    colors = [:steelblue, :teal, :coral, :mediumpurple, :goldenrod, :seagreen]
    s = 1.0  # panel size

    for (idx, (px, py, label)) in enumerate(panel_positions)
        x0, y0 = px * s, py * s
        poly!(ax, Rect(x0, y0, s, s); color=(colors[idx], 0.25))
        lines!(ax, [x0, x0+s, x0+s, x0, x0],
                   [y0, y0, y0+s, y0+s, y0]; color=colors[idx], linewidth=2.5)

        # Internal grid lines
        n = 4
        for k in 1:n-1
            frac = k / n
            lines!(ax, [x0 + frac*s, x0 + frac*s], [y0, y0+s];
                   color=(colors[idx], 0.3), linewidth=0.7)
            lines!(ax, [x0, x0+s], [y0 + frac*s, y0 + frac*s];
                   color=(colors[idx], 0.3), linewidth=0.7)
        end

        text!(ax, x0 + s/2, y0 + s/2; text=label, align=(:center, :center),
              fontsize=11, color=colors[idx], font=:bold)
    end

    # Edge connectivity arrows
    arrows!(ax, [1.95], [1.5], [0.1], [0.0]; color=:gray40, linewidth=1.5)
    arrows!(ax, [2.05], [1.5], [-0.1], [0.0]; color=:gray40, linewidth=1.5)
    text!(ax, 2.0, 1.35; text="shared\nedge", fontsize=9, color=:gray40,
          align=(:center, :center))

    xlims!(ax, -0.3, 4.3); ylims!(ax, -0.3, 3.3)

    save(joinpath(ASSETS, "grid_cubedsphere.png"), fig; px_per_unit=2.5)
    @info "Saved grid_cubedsphere.png"
end

# =========================================================================
# Diagram 3: Hybrid sigma-pressure vertical coordinate
# =========================================================================
function diagram_hybrid_sigma()
    fig = Figure(size=(600, 450), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Horizontal distance (schematic)",
        ylabel="Pressure [hPa]",
        title="HybridSigmaPressure Vertical Coordinate",
        yreversed=true)

    # Schematic terrain
    x = range(0, 1, length=100)
    terrain = @. 1000 - 200 * exp(-((x - 0.4) / 0.15)^2) -
                 100 * exp(-((x - 0.7) / 0.1)^2)

    # Hybrid levels: pure pressure at top, terrain-following at bottom
    n_levels = 12
    p_top = 10.0
    p_ref = 1013.0

    for k in 0:n_levels
        sigma = k / n_levels
        alpha = sigma^2  # transition parameter

        p_level = @. p_top + (1 - alpha) * (p_ref - p_top) * sigma +
                     alpha * (terrain - p_top) * sigma

        color = k == 0 || k == n_levels ? :black : (:gray50, 0.6)
        lw = k == 0 || k == n_levels ? 2.0 : 0.8
        lines!(ax, x, p_level; color, linewidth=lw)
    end

    # Terrain fill
    band!(ax, x, terrain, fill(1100, length(x)); color=(:saddlebrown, 0.3))
    lines!(ax, x, terrain; color=:saddlebrown, linewidth=2)

    # Labels
    text!(ax, 0.05, 50; text="Pure pressure\n(A dominates)", fontsize=10,
          color=:blue)
    text!(ax, 0.05, 800; text="Terrain-following\n(B dominates)", fontsize=10,
          color=:blue)

    arrows!(ax, [0.5], [300.0], [0.0], [-50.0]; color=(:blue, 0.5))
    text!(ax, 0.52, 280; text="Transition\nregion", fontsize=9, color=(:blue, 0.6))

    ylims!(ax, 1100, 0)

    save(joinpath(ASSETS, "grid_hybrid_sigma.png"), fig; px_per_unit=2.5)
    @info "Saved grid_hybrid_sigma.png"
end

# =========================================================================
# Diagram 4: Reduced grid near poles
# =========================================================================
function diagram_reduced()
    fig = Figure(size=(600, 400), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Longitude [°]", ylabel="Latitude [°]",
        title="Reduced Grid (CFL-adaptive polar coarsening)",
        aspect=DataAspect())

    Δλ = 15
    lons = -180:Δλ:180

    # Draw latitude bands with increasing reduction toward poles
    for (lat, red) in [(75, 1), (80, 2), (85, 3),
                       (-75, 1), (-80, 2), (-85, 3)]
        effective_Δλ = Δλ * (1 + red)
        eff_lons = -180:effective_Δλ:180
        for λ in eff_lons[1:end-1]
            rect = Rect(λ, lat, effective_Δλ, 5.0 * sign(lat))
            color = red == 1 ? (:orange, 0.2) : red == 2 ? (:red, 0.2) : (:darkred, 0.2)
            poly!(ax, rect; color)
            bcolor = red == 1 ? :orange : red == 2 ? :red : :darkred
            lines!(ax, [λ, λ+effective_Δλ, λ+effective_Δλ, λ, λ],
                [lat, lat, lat + 5.0*sign(lat), lat + 5.0*sign(lat), lat];
                color=bcolor, linewidth=1)
        end
    end

    # Normal cells in mid-latitudes
    for lat in 0:Δλ:60
        for λ in lons[1:end-1]
            lines!(ax, [λ, λ+Δλ, λ+Δλ, λ, λ],
                       [lat, lat, lat+Δλ, lat+Δλ, lat];
                   color=(:gray50, 0.3), linewidth=0.5)
        end
    end

    # Legend-style annotations
    text!(ax, -170, 87; text="3× coarsened", fontsize=10, color=:darkred)
    text!(ax, -170, 82; text="2× coarsened", fontsize=10, color=:red)
    text!(ax, -170, 77; text="1× coarsened", fontsize=10, color=:orange)
    text!(ax, -170, 65; text="Native resolution", fontsize=10, color=:gray50)

    xlims!(ax, -185, 185); ylims!(ax, -5, 95)

    save(joinpath(ASSETS, "grid_reduced.png"), fig; px_per_unit=2.5)
    @info "Saved grid_reduced.png"
end

diagram_latlon()
diagram_cubedsphere()
diagram_hybrid_sigma()
diagram_reduced()

@info "All grid diagrams generated in $ASSETS"
