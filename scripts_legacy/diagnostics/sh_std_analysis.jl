#!/usr/bin/env julia
# =========================================================================
# SH Standard Deviation Analysis
#
# Runs transport for N windows and tracks SH surface CO2 std at each output
# step. This is the key metric for transport quality — SH std should stay
# small (<5 ppm) because CO2 is well-mixed in the SH.
#
# Also tracks: mass conservation, mean VMR, pole metrics.
#
# Usage:
#   julia --project=. scripts/diagnostics/sh_std_analysis.jl <config.toml> [max_windows]
# =========================================================================

using CUDA
using AtmosTransport
using AtmosTransport.IO: MassFluxBinaryReader, load_window!
using Statistics
using Printf

config_file = length(ARGS) >= 1 ? ARGS[1] : "config/runs/era5_v4_advonly_2day.toml"
max_win = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 48

println("=== SH Std Analysis ===")
println("Config: $config_file")
println("Max windows: $max_win")

config = AtmosTransport.IO.load_configuration(config_file)
config["met_data"]["max_windows"] = max_win
model = AtmosTransport.IO.build_model_from_config(config)

driver = model.met_data
Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
n_sub = AtmosTransport.IO.steps_per_window(driver)
dt_sub = Float64(driver.dt)
dt_window = dt_sub * n_sub
n_win = min(max_win, AtmosTransport.IO.total_windows(driver))

println("Grid: $(Nx)×$(Ny)×$(Nz), n_sub=$n_sub, dt=$dt_sub s, windows=$n_win")
println()

# Run the model
t0 = time()
AtmosTransport.Models.run!(model)
elapsed = time() - t0

co2 = Array(model.tracers.co2)
n_nan = sum(isnan, co2)
n_neg = sum(co2 .< 0)

println("\n=== Final Results ===")
println("Time: $(round(elapsed, digits=1))s ($(round(elapsed/n_win, digits=2)) s/win)")
println("NaN: $n_nan / $(length(co2))")
println("Neg: $n_neg / $(length(co2))")

if n_nan > 0
    println("FAILED: NaN detected")
    exit(1)
end

# Load final window's mass for VMR computation
files = sort([joinpath(dirname(driver.files[1]), f) for f in readdir(dirname(driver.files[1])) if endswith(f, ".bin")])
r = MassFluxBinaryReader(files[end], Float32)
m_last = Array{Float32}(undef, Nx, Ny, Nz)
am_t = Array{Float32}(undef, Nx+1, Ny, Nz)
bm_t = Array{Float32}(undef, Nx, Ny+1, Nz)
cm_t = Array{Float32}(undef, Nx, Ny, Nz+1)
ps_t = Array{Float32}(undef, Nx, Ny)
load_window!(m_last, am_t, bm_t, cm_t, ps_t, r, r.Nt)
close(r)

vmr = co2 ./ max.(m_last, 1f-30) .* 1f6  # ppm
sfc = vmr[:, :, Nz]  # surface level

# Latitude bands
j_sh_pole = 1:10      # 90S-85S (reduced grid)
j_sh_high = 11:60     # 85S-60S
j_sh_mid  = 61:150    # 60S-15S
j_eq      = 151:211   # 15S-15N
j_nh_mid  = 212:302   # 15N-60N
j_nh_high = 303:352   # 60N-85N
j_nh_pole = 353:361   # 85N-90N (reduced grid)

println("\n=== Surface VMR by latitude band (ppm) ===")
for (jr, name) in [(j_sh_pole, "SH pole 85-90°S"),
                    (j_sh_high, "SH high 60-85°S"),
                    (j_sh_mid,  "SH mid  15-60°S"),
                    (j_eq,      "Equator 15S-15N"),
                    (j_nh_mid,  "NH mid  15-60°N"),
                    (j_nh_high, "NH high 60-85°N"),
                    (j_nh_pole, "NH pole 85-90°N")]
    band = sfc[:, jr]
    @printf("  %-20s: mean=%7.1f  std=%7.2f  min=%7.1f  max=%9.1f\n",
            name, mean(band), std(band), minimum(band), maximum(band))
end

# Key metrics
sh_mid_std = std(sfc[:, j_sh_mid])
sh_all_std = std(sfc[:, 11:180])  # all SH excluding pole rows
nh_mid_std = std(sfc[:, j_nh_mid])
global_mean = mean(vmr)

println("\n=== Key Metrics ===")
@printf("Global mean VMR:     %7.1f ppm\n", global_mean)
@printf("SH mid-lat std:      %7.2f ppm\n", sh_mid_std)
@printf("SH (excl poles) std: %7.2f ppm\n", sh_all_std)
@printf("NH mid-lat std:      %7.2f ppm\n", nh_mid_std)
@printf("Mass drift:          %+.4e%%\n",
        100 * (sum(Float64.(co2)) - sum(Float64.(411f-6 .* m_last))) /
        sum(Float64.(411f-6 .* m_last)))

# Column mean std (more robust than surface)
col_mean = dropdims(mean(vmr, dims=3), dims=3)
sh_col_std = std(col_mean[:, 11:180])
println(@sprintf("SH column-mean std:  %7.2f ppm", sh_col_std))
