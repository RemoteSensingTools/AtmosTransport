#!/usr/bin/env julia
# Quick test: run 18L balanced 2-day with pole fix, report SH std + mass drift
using CUDA; CUDA.allowscalar(false)
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!
import TOML

config = TOML.parsefile(joinpath(@__DIR__, "../../config/runs/era5_v4_advonly_2day.toml"))
config["met_data"]["directory"] = "/temp1/atmos_transport/era5_v4_5000Pa_balanced"
config["met_data"]["max_windows"] = 48  # full 2 days
config["advection"]["enable_flux_delta"] = true
config["output"]["filename"] = "/tmp/era5_v4_pole_fix_test.nc"

model = build_model_from_config(config)
@info "Running 18L balanced, 2 days (48 windows) — pole fix test"
run!(model)

# Analyze output
using NCDatasets, Statistics, Printf
ds = NCDataset("/tmp/era5_v4_pole_fix_test.nc")
co2_sfc = ds["co2_surface"][:]  # lon × lat × time
close(ds)

n_times = size(co2_sfc, 3)
@printf("\n=== Pole Fix Test Results (18L balanced, 2 days) ===\n")
for t in 1:n_times
    slab = co2_sfc[:, :, t]
    sh = @view slab[:, 1:181]
    gl_std = std(slab)
    sh_std = std(sh)
    @printf("t=%2d: global_mean=%.2f  SH_std=%.4f  global_std=%.4f ppm\n",
            t, mean(slab) * 1e6, sh_std * 1e6, gl_std * 1e6)
end

init_mean = mean(co2_sfc[:, :, 1])
final_mean = mean(co2_sfc[:, :, end])
drift_pct = (final_mean - init_mean) / init_mean * 100
@printf("\nMass drift: %.4f%%\n", drift_pct)
@printf("Baseline (before pole fix): SH_std=0.34 ppm, drift=-0.002%%\n")
