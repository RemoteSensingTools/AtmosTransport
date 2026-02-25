#!/usr/bin/env julia
# Quick visualization of GEOS-IT C180 June 2023 output
# Usage: julia --project=. scripts/visualize_geosit.jl

using NCDatasets, CairoMakie

const INFILE = expanduser("~/data/output/geosit_c180_june2023_05deg.nc")
const SNAP_PNG = "/tmp/geosit_c180_june2023_snapshots.png"
const GIF_FILE = "/tmp/geosit_c180_june2023.gif"

# Load data
ds = NCDataset(INFILE, "r")
lon = ds["lon"][:]
lat = ds["lat"][:]
co2 = ds["co2"][:, :, :]   # (720, 361, Nt)
Nt = size(co2, 3)
close(ds)

per_day = Nt ÷ 30  # outputs per day
println("Loaded: $(size(co2)), Nt=$Nt, per_day=$per_day")

# --- 4-panel snapshot ---
days_to_show = [1, 7, 15, 30]
fig = Figure(size=(1400, 800), fontsize=14)
Label(fig[0, 1:2], "GEOS-IT C180 Column-Mean CO₂ (June 2023, zero start)",
      fontsize=18, font=:bold)

vmax = Float64(maximum(co2[:, :, end])) * 0.85

for (idx, day) in enumerate(days_to_show)
    ti = min(day * per_day, Nt)
    row = (idx - 1) ÷ 2 + 1
    col = (idx - 1) % 2 + 1

    ax = Axis(fig[row, col];
        title="Day $day",
        ylabel= col == 1 ? "Latitude" : "",
        aspect=DataAspect())

    data = permutedims(Float64.(co2[:, :, ti]))  # (lat, lon) for correct orientation
    heatmap!(ax, Float64.(lon), Float64.(lat), data;
        colorrange=(0, vmax), colormap=:thermal)
end

Colorbar(fig[1:2, 3]; colorrange=(0, vmax), colormap=:thermal,
         label="CO₂ [ppm]", height=Relative(0.8))

save(SNAP_PNG, fig, px_per_unit=2)
println("Saved snapshots: $SNAP_PNG")

# --- Animation (every 6 hours) ---
stride = max(1, per_day ÷ 4)
frames = collect(1:stride:Nt)
println("Animating $(length(frames)) frames...")

fig2 = Figure(size=(900, 500), fontsize=14)
ax = Axis(fig2[1, 1]; xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
Colorbar(fig2[1, 2]; colorrange=(0, vmax), colormap=:thermal, label="CO₂ column mean [ppm]")

record(fig2, GIF_FILE, frames; framerate=8) do t
    empty!(ax)
    data = permutedims(Float64.(co2[:, :, t]))
    heatmap!(ax, Float64.(lon), Float64.(lat), data;
        colorrange=(0, vmax), colormap=:thermal)
    day = t / per_day
    ax.title = "GEOS-IT C180 Column-Mean CO₂ — Day $(round(day, digits=1))"
end
println("Saved animation: $GIF_FILE")
