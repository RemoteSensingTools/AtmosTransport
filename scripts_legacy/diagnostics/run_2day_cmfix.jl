#!/usr/bin/env julia
using CUDA; CUDA.allowscalar(false)
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!
import TOML

config = TOML.parsefile(joinpath(@__DIR__, "../../config/runs/era5_point_source_test.toml"))
config["advection"] = Dict("scheme" => "slopes")
config["met_data"]["max_windows"] = 8
config["grid"]["merge_min_thickness_Pa"] = 1000
config["initial_conditions"] = Dict(
    "co2" => Dict(
        "file" => expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc"),
        "variable" => "CO2"
    )
)
config["tracers"] = Dict("co2" => Dict(
    "emission" => "uniform_surface",
    "rate" => 1.0e-8,
    "species" => "co2"
))
config["output"]["filename"] = "/tmp/era5_2day_fixed.nc"
config["output"]["interval"] = 10800

model = build_model_from_config(config)
@info "Running..."
run!(model)

# Generate plots using GR backend (fast, no CairoMakie compilation)
using NCDatasets, Statistics, Printf
using Plots; gr()

ds = NCDataset("/tmp/era5_2day_fixed.nc", "r")
sfc = ds["co2_surface"][:,:,:]
lons = ds["lon"][:]
lats = ds["lat"][:]
times = ds["time"][:]
close(ds)

outdir = expanduser("~/www/catrina")
mkpath(outdir)

for t in 1:size(sfc, 3)
    slice = sfc[:, :, t]
    has_nan = any(isnan, slice)
    if has_nan
        @warn "NaN at t=$t, skipping"
        continue
    end

    p = heatmap(lons, lats, slice' .* 1e6;
        clims=(390, 440), color=:RdYlBu_r,
        title="Surface CO₂ — $(times[t])",
        xlabel="Longitude", ylabel="Latitude",
        colorbar_title="ppm", size=(1000, 400), dpi=150)

    fname = joinpath(outdir, @sprintf("sfc_co2_%02d.png", t))
    savefig(p, fname)
    @info "Saved $fname"
end

@info "All plots saved to $outdir"
