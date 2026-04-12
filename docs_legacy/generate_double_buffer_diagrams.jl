#!/usr/bin/env julia
#
# Generate Gantt-style timeline diagrams for GPU double-buffering documentation.
# Output: docs/src/assets/gpu_sync_pipeline.png
#         docs/src/assets/gpu_double_buffer_era5.png
#         docs/src/assets/gpu_double_buffer_cs.png

using CairoMakie

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

function draw_bar!(ax, y, x_start, width, label;
                   color=:steelblue, fontsize=11, height=0.6)
    rect = Rect(x_start, y - height/2, width, height)
    poly!(ax, rect; color=color)
    text!(ax, x_start + width/2, y; text=label, align=(:center, :center),
          fontsize=fontsize, color=:white, font=:bold)
end

# =========================================================================
# Diagram 1: Current synchronous pipeline
# =========================================================================
function diagram_sync()
    fig = Figure(size=(900, 280), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Time →",
        yticks=([2, 1], ["CPU", "GPU"]),
        title="Current Synchronous Pipeline — GPU idle during CPU work")
    hidexdecorations!(ax; label=false)
    xlims!(ax, -0.5, 19)
    ylims!(ax, 0.3, 2.7)

    # Step N
    draw_bar!(ax, 2, 0.0, 2.0, "Load\nNetCDF"; color=(:royalblue, 0.8))
    draw_bar!(ax, 2, 2.0, 1.5, "Preprocess"; color=(:slateblue, 0.8))
    draw_bar!(ax, 2, 3.5, 1.0, "H→D"; color=(:orange, 0.85))
    draw_bar!(ax, 1, 4.5, 4.5, "Compute step N"; color=(:green4, 0.9))

    # Step N+1
    draw_bar!(ax, 2, 9.0, 2.0, "Load\nNetCDF"; color=(:royalblue, 0.8))
    draw_bar!(ax, 2, 11.0, 1.5, "Preprocess"; color=(:slateblue, 0.8))
    draw_bar!(ax, 2, 12.5, 1.0, "H→D"; color=(:orange, 0.85))
    draw_bar!(ax, 1, 13.5, 4.5, "Compute step N+1"; color=(:green4, 0.9))

    # Idle markers
    text!(ax, 2.25, 1; text="idle", align=(:center, :center),
          fontsize=11, color=(:gray50, 0.8), font=:italic)
    text!(ax, 11.25, 1; text="idle", align=(:center, :center),
          fontsize=11, color=(:gray50, 0.8), font=:italic)

    # Step brackets
    bracket!(ax, 0.0, 2.55, 8.5, 2.55; text="Step N", fontsize=10, color=:gray50)
    bracket!(ax, 9.0, 2.55, 17.5, 2.55; text="Step N+1", fontsize=10, color=:gray50)

    save(joinpath(ASSETS, "gpu_sync_pipeline.png"), fig; px_per_unit=2.5)
    @info "Saved gpu_sync_pipeline.png"
end

# =========================================================================
# Diagram 2: Double-buffered ERA5 pipeline
# =========================================================================
function diagram_double_buffer_era5()
    fig = Figure(size=(900, 350), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Time →",
        yticks=([3, 2, 1], ["CPU", "Transfer\nstream", "Compute\nstream"]),
        title="Double-Buffered ERA5 Pipeline — overlap I/O with compute")
    hidexdecorations!(ax; label=false)
    xlims!(ax, -0.5, 19)
    ylims!(ax, 0.3, 3.7)

    # Initial load step N
    draw_bar!(ax, 3, 0.0, 2.0, "Load N"; color=(:royalblue, 0.8))
    draw_bar!(ax, 3, 2.0, 1.0, "Prep N"; color=(:slateblue, 0.8))
    draw_bar!(ax, 2, 3.0, 1.0, "H→D buf A"; color=(:orange, 0.85), fontsize=9)

    # Compute N on buf A | CPU loads N+1 | transfer to B
    draw_bar!(ax, 1, 4.0, 4.5, "Compute N  (buf A)"; color=(:green4, 0.9))
    draw_bar!(ax, 3, 4.0, 2.0, "Load N+1"; color=(:royalblue, 0.8))
    draw_bar!(ax, 3, 6.0, 1.0, "Prep"; color=(:slateblue, 0.8), fontsize=10)
    draw_bar!(ax, 2, 7.0, 1.0, "H→D buf B"; color=(:orange, 0.85), fontsize=9)

    # Compute N+1 on buf B | CPU loads N+2 | transfer to A
    draw_bar!(ax, 1, 8.5, 4.5, "Compute N+1  (buf B)"; color=(:teal, 0.9))
    draw_bar!(ax, 3, 8.5, 2.0, "Load N+2"; color=(:royalblue, 0.8))
    draw_bar!(ax, 3, 10.5, 1.0, "Prep"; color=(:slateblue, 0.8), fontsize=10)
    draw_bar!(ax, 2, 11.5, 1.0, "H→D buf A"; color=(:orange, 0.85), fontsize=9)

    # Compute N+2 on A
    draw_bar!(ax, 1, 13.0, 4.5, "Compute N+2  (buf A)"; color=(:green4, 0.9))

    # Annotation
    text!(ax, 9.5, 3.55; text="CPU load + transfer overlapped with GPU compute",
          align=(:center, :center), fontsize=11, color=:gray45, font=:italic)

    save(joinpath(ASSETS, "gpu_double_buffer_era5.png"), fig; px_per_unit=2.5)
    @info "Saved gpu_double_buffer_era5.png"
end

# =========================================================================
# Diagram 3: Double-buffered cubed-sphere panel pipeline
# =========================================================================
function diagram_double_buffer_cs()
    fig = Figure(size=(960, 350), fontsize=13)
    ax = Axis(fig[1, 1];
        xlabel="Time →",
        yticks=([3, 2, 1], ["Host", "Transfer\nstream", "Compute\nstream"]),
        title="Double-Buffered Cubed-Sphere Panel Pipeline (6 panels)")
    hidexdecorations!(ax; label=false)
    xlims!(ax, -0.5, 22.5)
    ylims!(ax, 0.3, 3.7)

    panel_colors = [
        (:green4, 0.9), (:teal, 0.9), (:green4, 0.9),
        (:teal, 0.9), (:green4, 0.9), (:teal, 0.9)
    ]

    # Initial upload of panel 1
    draw_bar!(ax, 2, 0.0, 1.0, "P1→A"; color=(:orange, 0.85), fontsize=9)

    x = 1.2
    pw = 3.0
    gap = 0.3

    for p in 1:6
        buf = p % 2 == 1 ? "A" : "B"
        next_buf = p % 2 == 1 ? "B" : "A"

        draw_bar!(ax, 1, x, pw, "Panel $p  (buf $buf)";
                  color=panel_colors[p], fontsize=10)

        if p < 6
            draw_bar!(ax, 2, x + 0.3, 1.0, "P$(p+1)→$next_buf";
                      color=(:orange, 0.85), fontsize=9)
        end

        x += pw + gap
    end

    text!(ax, 11.0, 3.55;
          text="Upload panel P+1 overlaps with compute on panel P",
          align=(:center, :center), fontsize=11, color=:gray45, font=:italic)

    save(joinpath(ASSETS, "gpu_double_buffer_cs.png"), fig; px_per_unit=2.5)
    @info "Saved gpu_double_buffer_cs.png"
end

diagram_sync()
diagram_double_buffer_era5()
diagram_double_buffer_cs()

@info "All diagrams generated in $ASSETS"
