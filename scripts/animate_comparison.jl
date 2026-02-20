#!/usr/bin/env julia
# ===========================================================================
# Three-panel comparison animation: ERA5 | GEOS-FP CS | Difference
#
# Reads output NetCDF files from both runs and produces a side-by-side
# column-mean CO2 animation showing how the ERA5 lat-lon and GEOS-FP
# cubed-sphere simulations compare over time.
#
# Usage:
#   julia --project=. scripts/animate_comparison.jl
#
# Environment:
#   ERA5_OUTPUT     — path to ERA5 output NetCDF
#   GEOSFP_OUTPUT   — path to GEOS-FP CS output NetCDF
#   OUT_GIF         — output GIF path (default: comparison.gif)
# ===========================================================================

using NCDatasets
using CairoMakie

const ERA5_FILE   = get(ENV, "ERA5_OUTPUT",
    joinpath(homedir(), "data", "output", "era5_edgar_june2024", "era5_edgar_f32.nc"))
const GEOSFP_FILE = get(ENV, "GEOSFP_OUTPUT",
    joinpath(homedir(), "data", "output", "geosfp_cs_edgar_june2024", "geosfp_cs_edgar_f32.nc"))
const OUT_GIF     = get(ENV, "OUT_GIF", "comparison_era5_geosfp.gif")

function load_output(filepath::String)
    ds = NCDataset(filepath, "r")
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    time_h = ds["time"][:]
    col = ds["co2_column_mean"][:, :, :]
    close(ds)
    return (; lon, lat, time_h, col)
end

function make_animation(era5, geosfp; fps=4)
    Nt = min(size(era5.col, 3), size(geosfp.col, 3))
    @info "Animating $Nt frames"

    vmin, vmax = 0.0, max(
        maximum(era5.col[:, :, end]),
        maximum(geosfp.col[:, :, end])
    ) * 1.1

    fig = Figure(size=(1600, 500), fontsize=14)
    ax1 = Axis(fig[1, 1]; title="ERA5 (lat-lon)",
               xlabel="Longitude", ylabel="Latitude")
    ax2 = Axis(fig[1, 2]; title="GEOS-FP (cubed-sphere)",
               xlabel="Longitude")
    ax3 = Axis(fig[1, 3]; title="Difference (GEOS-FP − ERA5)",
               xlabel="Longitude")

    linkaxes!(ax1, ax2, ax3)
    hideydecorations!(ax2); hideydecorations!(ax3)

    record(fig, OUT_GIF, 1:Nt; framerate=fps) do t
        empty!(ax1); empty!(ax2); empty!(ax3)

        heatmap!(ax1, era5.lon, era5.lat, era5.col[:, :, t]';
                 colorrange=(vmin, vmax), colormap=:thermal)
        heatmap!(ax2, geosfp.lon, geosfp.lat, geosfp.col[:, :, t]';
                 colorrange=(vmin, vmax), colormap=:thermal)

        diff = geosfp.col[:, :, t] .- era5.col[:, :, t]
        dmax = max(abs(minimum(diff)), abs(maximum(diff)), 1e-10)
        heatmap!(ax3, era5.lon, era5.lat, diff';
                 colorrange=(-dmax, dmax), colormap=:balance)

        hours = era5.time_h[t]
        fig.suptitle[] = "Column-mean CO₂ [ppm] — t = $(round(hours/24, digits=1)) days"
    end

    @info "Saved animation: $OUT_GIF"
end

function main()
    isfile(ERA5_FILE) || error("ERA5 output not found: $ERA5_FILE")
    isfile(GEOSFP_FILE) || error("GEOS-FP output not found: $GEOSFP_FILE")

    @info "Loading ERA5 output: $ERA5_FILE"
    era5 = load_output(ERA5_FILE)

    @info "Loading GEOS-FP output: $GEOSFP_FILE"
    geosfp = load_output(GEOSFP_FILE)

    make_animation(era5, geosfp)
end

main()
