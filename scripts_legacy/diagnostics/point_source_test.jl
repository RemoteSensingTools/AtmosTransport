#!/usr/bin/env julia
# ===========================================================================
# Point Source Transport Test — Impulse IC, pure advection, per-window diagnostics
#
# Injects a single-cell tracer pulse at the equator (surface level),
# then runs 3 days of pure advection (no emissions, no physics).
#
# Diagnostics logged per window:
#   - Total tracer mass and conservation error
#   - Max/min rm and their locations (lat/lon/level)
#   - Plume centroid (lat/lon) and effective wind speed
#   - Plume spatial extent (number of cells > 1% of max)
#   - Any NaN or negative rm values
#
# Usage:
#   julia --threads=2 --project=. scripts/diagnostics/point_source_test.jl
#
# For MCP interactive use, evaluate sections between the # === markers.
# ===========================================================================

using Logging

import TOML
config_path = joinpath(@__DIR__, "..", "..", "config", "runs", "era5_point_source_test.toml")
config = TOML.parsefile(config_path)
@info "Configuration: $config_path"

# === Load GPU ===
if get(get(config, "architecture", Dict()), "use_gpu", false)
    if Sys.isapple()
        using Metal
    else
        using CUDA
        CUDA.allowscalar(false)
    end
end

using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!
using Statistics
using Printf

@info "Building model..."
model = build_model_from_config(config)
grid = model.grid
@info "Model built: $(grid.Nx)×$(grid.Ny)×$(grid.Nz)"

# === Inject point source pulse ===
# Grid coordinates:
#   longitude: cell centers at -179.75, -179.25, ..., 179.75 (720 cells, Δλ=0.5°)
#   latitude:  cell centers at ~-89.75, ..., ~89.75 (361 cells, Δφ≈0.499°)
#   levels:    k=1 TOA, k=Nz surface (ERA5 convention)
#
# Source location: ~0°E, ~0°N, surface
#   i_src = 361  → lon ≈ 0.25°E   (center of cell spanning -0.25° to 0.75°)
#   j_src = 181  → lat ≈ 0°N      (equator)
#   k_src = Nz   → surface level

Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
i_src = 361
j_src = 181
k_src = Nz

# Verify coordinates from grid
lon_src = grid.λᶜ_cpu[i_src]
lat_src = grid.φᶜ_cpu[j_src]
@info @sprintf("Point source at: lon=%.2f°, lat=%.2f°, level=%d (surface)", lon_src, lat_src, k_src)

# Inject pulse: set rm = 1.0 at the source cell (concentration ≈ 1/m_air at that cell)
# This is tracer mass in the same units as m (air mass).
# A value of 1e-6 gives ~1 ppm equivalent if m_air ~ 1.0 at that cell.
pulse_value = Float32(1e-6)

# Access the tracer array (GPU array)
rm_co2 = model.tracers.co2
# Set the pulse — need scalar indexing
if rm_co2 isa Array
    rm_co2[i_src, j_src, k_src] = pulse_value
else
    # GPU array: use allowscalar or copy to CPU, modify, copy back
    rm_cpu = Array(rm_co2)
    rm_cpu[i_src, j_src, k_src] = pulse_value
    copyto!(rm_co2, rm_cpu)
end

initial_mass = sum(Array(rm_co2))
@info @sprintf("Injected pulse: rm=%.2e at (%d,%d,%d), total mass=%.6e", pulse_value, i_src, j_src, k_src, initial_mass)

# === Diagnostic state ===
struct PlumeStats
    window      :: Int
    sim_hours   :: Float64
    total_mass  :: Float64
    mass_err_pct:: Float64
    max_rm      :: Float64
    max_loc     :: Tuple{Int,Int,Int}
    min_rm      :: Float64
    has_nan     :: Bool
    has_neg     :: Bool
    n_cells_1pct:: Int         # cells with rm > 1% of max
    centroid_lon:: Float64
    centroid_lat:: Float64
    speed_ms    :: Float64     # effective wind speed from centroid displacement
end

function compute_plume_stats(rm_gpu, grid, window, sim_hours, initial_mass, prev_centroid)
    rm = Array(rm_gpu)
    Nx, Ny, Nz = size(rm)

    total_mass = sum(rm)
    mass_err = (total_mass - initial_mass) / initial_mass * 100

    max_rm = maximum(rm)
    max_idx = argmax(rm)
    max_loc = Tuple(max_idx)
    min_rm = minimum(rm)

    has_nan = any(isnan, rm)
    has_neg = min_rm < 0

    threshold = max_rm * 0.01
    n_cells_1pct = count(>(threshold), rm)

    # Plume centroid (mass-weighted lat/lon) — only for surface level
    rm_sfc = @view rm[:, :, Nz]
    total_sfc = sum(rm_sfc)
    if total_sfc > 0 && !any(isnan, rm_sfc)
        lons = grid.λᶜ_cpu  # length Nx
        lats = grid.φᶜ_cpu  # length Ny
        clon = sum(rm_sfc[i,j] * lons[i] for i in 1:Nx, j in 1:Ny) / total_sfc
        clat = sum(rm_sfc[i,j] * lats[j] for i in 1:Nx, j in 1:Ny) / total_sfc
    else
        clon, clat = NaN, NaN
    end

    # Transport speed from centroid displacement
    if prev_centroid !== nothing && !isnan(clon) && !isnan(prev_centroid[1])
        R = 6.371e6  # Earth radius [m]
        dt_hours = 1.0  # one window = 1 hour
        Δlon = (clon - prev_centroid[1]) * π / 180
        Δlat = (clat - prev_centroid[2]) * π / 180
        lat_avg = (clat + prev_centroid[2]) / 2 * π / 180
        dx = Δlon * cos(lat_avg) * R
        dy = Δlat * R
        dist = sqrt(dx^2 + dy^2)
        speed = dist / (dt_hours * 3600)
    else
        speed = NaN
    end

    return PlumeStats(window, sim_hours, total_mass, mass_err, max_rm, max_loc,
                      min_rm, has_nan, has_neg, n_cells_1pct, clon, clat, speed)
end

function print_stats(s::PlumeStats)
    status = s.has_nan ? "NaN!" : s.has_neg ? "NEG!" : "OK"
    @info @sprintf(
        "Win %3d (%5.1fh) [%s] mass=%.4e (Δ=%+.4f%%) max=%.2e@(%d,%d,%d) min=%.2e cells>1%%=%d centroid=(%.1f°,%.1f°) v=%.1f m/s",
        s.window, s.sim_hours, status, s.total_mass, s.mass_err_pct,
        s.max_rm, s.max_loc..., s.min_rm, s.n_cells_1pct,
        s.centroid_lon, s.centroid_lat, s.speed_ms)
end

# === Run simulation ===
@info "Starting 3-day point source transport test (72 windows)..."
@info "Expected: plume advects at ~5-10 m/s (ERA5 surface winds), mass conserved, no NaN"
@info ""

# Run the full simulation — diagnostics come from output file analysis afterward.
# For per-window inspection, use the MCP approach below.
run!(model)

@info ""
@info "Simulation complete. Analyzing output..."

# === Post-run analysis ===
# Read the NetCDF output and compute per-timestep diagnostics
using NCDatasets
output_file = expanduser(config["output"]["filename"])

if isfile(output_file)
    ds = NCDataset(output_file, "r")
    times = ds["time"][:]
    nt = length(times)
    @info "Output file: $output_file ($nt timesteps)"

    if haskey(ds, "co2_surface")
        sfc = ds["co2_surface"]  # (lon, lat, time)
        for t in 1:nt
            slice = sfc[:, :, t]
            mx = maximum(filter(!isnan, slice))
            mn = minimum(filter(!isnan, slice))
            tot = sum(filter(!isnan, slice))
            has_nan = any(isnan, slice)
            has_neg = mn < 0

            # Centroid
            lons = ds["lon"][:]
            lats = ds["lat"][:]
            if tot > 0 && !has_nan
                clon = sum(slice[i,j] * lons[i] for i in axes(slice,1), j in axes(slice,2)) / tot
                clat = sum(slice[i,j] * lats[j] for i in axes(slice,1), j in axes(slice,2)) / tot
            else
                clon = clat = NaN
            end

            status = has_nan ? "NaN!" : has_neg ? "NEG!" : " OK "
            @info @sprintf("t=%3d [%s] sfc: max=%.2e min=%.2e sum=%.2e centroid=(%.1f°,%.1f°)",
                           t, status, mx, mn, tot, clon, clat)
        end
    end

    if haskey(ds, "co2_column_mean")
        col = ds["co2_column_mean"]
        for t in 1:nt
            slice = col[:, :, t]
            mx = maximum(filter(!isnan, slice))
            @info @sprintf("t=%3d column_mean: max=%.2e", t, mx)
        end
    end

    close(ds)
else
    @warn "Output file not found: $output_file"
end

@info "Point source test complete."
