using CUDA; CUDA.allowscalar(false)
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!
import TOML

config = TOML.parsefile(joinpath(@__DIR__, "../../config/runs/era5_point_source_test.toml"))
config["advection"] = Dict("scheme" => "slopes")
config["met_data"]["max_windows"] = 8
config["initial_conditions"] = Dict(
    "co2" => Dict(
        "file" => expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc"),
        "variable" => "CO2"
    )
)
config["tracers"] = Dict("co2" => Dict(
    "emission" => "uniform_surface", "rate" => 1.0e-8, "species" => "co2"
))
config["output"]["filename"] = "/tmp/era5_2day_nomerge.nc"
config["output"]["interval"] = 10800

model = build_model_from_config(config)
@info "Running 2 days (8 windows), no merging, 137L..."
run!(model)
@info "Simulation done"

using NCDatasets, Statistics, Printf, Plots; gr()
ds = NCDataset("/tmp/era5_2day_nomerge.nc", "r")
sfc = ds["co2_surface"][:,:,:]
lons = ds["lon"][:]
lats = ds["lat"][:]
times = ds["time"][:]
close(ds)

outdir = expanduser("~/www/catrina")
mkpath(outdir)
for t in axes(sfc, 3)
    s = sfc[:, :, t]
    any(isnan, s) && (@warn "NaN at t=$t, skipping"; continue)
    p = heatmap(lons, lats, permutedims(s) .* 1e6;
        clims=(390, 440), color=:RdYlBu_r,
        title="Surface CO₂ — $(times[t])",
        xlabel="Longitude", ylabel="Latitude",
        colorbar_title="ppm", size=(1000, 400), dpi=150)
    fname = joinpath(outdir, @sprintf("sfc_co2_%02d.png", t))
    savefig(p, fname)
    @info "Saved $fname"
end
@info "All plots saved to $outdir"
