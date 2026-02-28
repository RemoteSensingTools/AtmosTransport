#!/usr/bin/env julia
# Compute exact expected emission per window and compare against run outputs.
# Usage: julia --project=. scripts/diagnose_mass_budget.jl config/runs/geosit_c180_june2023.toml

using AtmosTransport
using AtmosTransport.IO: build_model_from_config
using AtmosTransport.Sources: M_AIR, CubedSphereEmission
using Printf, TOML, Dates

config_file = length(ARGS) >= 1 ? ARGS[1] : "config/runs/geosit_c180_june2023.toml"
config = TOML.parsefile(config_file)
config["architecture"]["use_gpu"] = false

model = build_model_from_config(config)
grid = model.grid
Nc = grid.Nc

dt_window = Float64(get(config["met_data"], "met_interval", 3600))
start = Date(config["met_data"]["start_date"])
stop  = Date(config["met_data"]["end_date"])
n_days = Dates.value(stop - start)  # end_date is inclusive last day
n_win = n_days * 24

println("Grid: C$(Nc), Nz=$(grid.Nz)")
println("dt_window=$dt_window s, n_win=$n_win ($n_days days: $start to $stop)\n")

# Find CubedSphereEmission sources and compute Σ(flux × area)
for src in model.sources
    if src isa CubedSphereEmission
        mol_ratio_f64 = 1e6 * M_AIR / src.molar_mass
        mol_ratio_f32 = Float64(Float32(mol_ratio_f64))

        total_fa = 0.0
        for p in 1:6
            flux = src.flux_panels[p]
            area = grid.Aᶜ[p]
            for j in 1:Nc, i in 1:Nc
                total_fa += Float64(flux[i, j]) * Float64(area[i, j])
            end
        end

        E_window_f64 = dt_window * mol_ratio_f64 * total_fa
        E_window_f32 = dt_window * mol_ratio_f32 * total_fa

        # Simulate F32 GPU emission kernel: each cell does F32 multiply then F64 sum
        E_window_gpu = 0.0
        dt_f32 = Float32(dt_window)
        mr_f32 = Float32(mol_ratio_f64)
        for p in 1:6
            flux = src.flux_panels[p]
            area = grid.Aᶜ[p]
            for j in 1:Nc, i in 1:Nc
                # GPU kernel: f * dt_window * area[i,j] * mol_ratio  (all F32)
                cell_emit = Float64(Float32(flux[i,j]) * dt_f32 * Float32(area[i,j]) * mr_f32)
                E_window_gpu += cell_emit
            end
        end

        n_inj = n_win - 1  # budget at win w is before emission → w-1 injections

        @printf("Emission: %s\n", src.label)
        @printf("  molar_mass     = %.4e kg/mol\n", src.molar_mass)
        @printf("  mol_ratio(F64) = %.6f\n", mol_ratio_f64)
        @printf("  mol_ratio(F32) = %.6f\n", mol_ratio_f32)
        @printf("  Σ(flux×area)   = %.6e  kg/s\n", total_fa)
        @printf("  E_window(F64)  = %.6e  ppm·kg_air / window\n", E_window_f64)
        @printf("  E_window(F32)  = %.6e  ppm·kg_air / window\n", E_window_f32)
        @printf("  E_window(GPU)  = %.6e  ppm·kg_air / window (simulated F32 kernel)\n\n",
                E_window_gpu)

        expected = n_inj * E_window_gpu

        slopes_720 = 1.9989e+18
        ppm_720    = 2.1040e+18

        @printf("Expected total_rm at win %d (%d injections): %.6e\n\n", n_win, n_inj, expected)
        @printf("%-12s  %14s  %10s\n", "Scheme", "total_rm", "deviation")
        @printf("%-12s  %14.4e  %9.3f%%\n", "Slopes", slopes_720, 100(slopes_720 - expected)/expected)
        @printf("%-12s  %14.4e  %9.3f%%\n", "PPM", ppm_720, 100(ppm_720 - expected)/expected)

        println("\n── Early-window cross-check ──")
        slopes_24 = 6.7466e+16
        ppm_24    = 6.7504e+16
        @printf("Slopes  win 24 / 23 inj = %.6e / win\n", slopes_24 / 23)
        @printf("PPM     win 24 / 23 inj = %.6e / win\n", ppm_24 / 23)
        @printf("Analytical E_window(GPU)= %.6e / win\n", E_window_gpu)
    end
end
