#!/usr/bin/env julia
# Run 1-month fullphys simulation, generate animation, deploy to web
cd(joinpath(@__DIR__, ".."))

using AtmosTransport
using CUDA
using AtmosTransport.Models: run!, MOIST_DIAG
using AtmosTransport.IO: build_model_from_config

MOIST_DIAG[] = nothing

# --- Run simulation (skip if final output exists) ---
last_day_file = "/temp1/catrine/output/catrine_gchp_v4_21d_fullphys_20220101.nc"
if isfile(last_day_file)
    @info "Simulation output exists ($last_day_file), skipping run."
else
    @info "Building 1-month fullphys model..."
    tm = build_model_from_config("config/runs/catrine_geosit_c180_gchp_v4_21d_fullphys.toml")
    @info "Starting 1-month run..."
    run!(tm)
    @info "Simulation complete!"
end

# --- Generate animation ---
@info "Generating animation..."
ENV["AT_PATTERN"] = "catrine_gchp_v4_21d_fullphys"
ENV["OUT_GIF"] = "/temp1/catrine/output/gchp_v4_1month_fullphys_vs_geoschem.gif"
ENV["DATE_END"] = "2021-12-31T21:00:00"
include(joinpath("scripts", "visualization", "animate_gchp_v4_fullphys.jl"))

# --- Deploy ---
@info "Copying to web directory..."
cp("/temp1/catrine/output/gchp_v4_1month_fullphys_vs_geoschem.gif",
   expanduser("~/www/catrina/gchp_v4_1month_fullphys_vs_geoschem.gif"); force=true)
@info "Deployed to ~/www/catrina/"

# --- Send email ---
@info "Sending notification email..."
run(`bash -c "echo 'CATRINE 1-month animation available at https://gps.caltech.edu/~cfranken/catrina/gchp_v4_1month_fullphys_vs_geoschem.gif

AtmosTransport (GCHP v4 + RAS + PBL, dry basis, per-step remap) vs GEOS-Chem reference.
Dec 2021, C180, 3-hourly output.

Panels: Surface CO2, 750 hPa CO2, dp-weighted column-avg XCO2.

Key changes in this run:
- Operator order matches GCHP: advection → emissions → BL mixing
- Per-step vertical remap enabled
- Kahan compensated Float32 PE accumulation' | mail -s 'CATRINE: 1-month fullphys animation ready' cfranken@caltech.edu"`)
@info "All done!"
