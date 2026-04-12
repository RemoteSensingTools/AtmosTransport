#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Step 2: Real moist→dry conversion using QV from ERA5 thermo NetCDF
#
# Tests the full v2 dry-flux pipeline:
#   1. Load moist binary (m, am, bm, ps) at merged 34 levels
#   2. Load specific humidity (q) from thermo NetCDF at native 137 levels
#   3. Mass-weighted remap QV 137→34 using dp-weighting and the embedded merge_map
#   4. Call build_dry_fluxes! to get dry air mass + dry fluxes
#   5. Run one Strang split on the dry state
#   6. Verify: total dry air mass, per-column dry mass closure, tracer mass
#
# NOT a CI test — requires real ERA5 binary + thermo data.
# ---------------------------------------------------------------------------

using Test
using Printf
using NCDatasets
using TOML
using JSON3
using KernelAbstractions: CPU

const BIN_PATH = expanduser(
    "~/data/AtmosTransport/met/era5/spectral_v4_tropo34_dec2021/" *
    "era5_v4_20211201_merged1000Pa_float32.bin")
const THERMO_PATH = expanduser(
    "~/data/AtmosTransport/met/era5/physics/era5_thermo_ml_20211201.nc")
const COEFF_PATH = joinpath(@__DIR__, "..", "config", "era5_L137_coefficients.toml")

for (name, path) in [("binary", BIN_PATH), ("thermo", THERMO_PATH), ("coefficients", COEFF_PATH)]
    if !isfile(path)
        @warn "Skipping real-data test: $name not found at $path"
        exit(0)
    end
end

ENV["ATMOSTR_NO_STALE_CHECK"] = "1"
ENV["ATMOSTR_NO_CM_CHECK"] = "1"

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

const FT = Float64
const WIN = 1  # hour 0 = window 1

# ===================================================================
# Helpers
# ===================================================================

"""
    load_native_ab_coefficients(coeff_path) -> (A_native, B_native)

Load the full L137 interface A/B coefficients (138 values each).
"""
function load_native_ab_coefficients(coeff_path::String)
    cfg = TOML.parsefile(coeff_path)
    a = Float64.(cfg["coefficients"]["a"])  # 138 values
    b = Float64.(cfg["coefficients"]["b"])
    return a, b
end

"""
    compute_dp_native(A_native, B_native, ps, Nx, Ny) -> dp[Nx, Ny, 137]

Pressure thickness of each native level: dp[k] = dA[k] + dB[k] × ps(i,j).
"""
function compute_dp_native(A_native, B_native, ps, Nx, Ny)
    Nz_native = length(A_native) - 1
    dA = diff(A_native)
    dB = diff(B_native)
    dp = Array{FT}(undef, Nx, Ny, Nz_native)
    @inbounds for k in 1:Nz_native, j in 1:Ny, i in 1:Nx
        dp[i, j, k] = FT(dA[k]) + FT(dB[k]) * ps[i, j]
    end
    return dp
end

"""
    merge_qv_mass_weighted!(qv_merged, qv_native, dp_native, merge_map, Nx, Ny, Nz)

Mass-weighted (dp-weighted) merge of QV from native to merged levels.
Identical to the preprocessor's merge_qv! — uses dp as the weight, which is
proportional to layer air mass (area/g cancels).
"""
function merge_qv_mass_weighted!(qv_merged::Array{FT,3}, qv_native::Array{<:Real,3},
                                  dp_native::Array{FT,3}, merge_map::Vector{Int},
                                  Nx::Int, Ny::Int, Nz::Int) where FT
    fill!(qv_merged, zero(FT))
    dp_sum = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:length(merge_map)
        km = merge_map[k]
        for j in 1:Ny, i in 1:Nx
            qv_merged[i, j, km] += FT(qv_native[i, j, k]) * dp_native[i, j, k]
            dp_sum[i, j, km]    += dp_native[i, j, k]
        end
    end
    @inbounds for km in 1:Nz, j in 1:Ny, i in 1:Nx
        qv_merged[i, j, km] /= max(dp_sum[i, j, km], FT(1))
    end
    return nothing
end

"""
    read_qv_from_thermo(path, hour_idx, Nx, Ny, Nz_native) -> Array{Float32,3}

Read specific humidity from the ERA5 thermo NetCDF for a given hour.
Flips latitude to S→N if needed.
"""
function read_qv_from_thermo(path::String, hour_idx::Int, Nx::Int, Ny::Int, Nz_native::Int)
    NCDataset(path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            q = Float32.(q_var[:, :, :, hour_idx])
        else
            q_raw = Float32.(q_var[hour_idx, :, :, :])
            q = permutedims(q_raw, (3, 2, 1))
        end
        if size(q, 2) == Ny && ds["latitude"][1] > ds["latitude"][end]
            q = q[:, end:-1:1, :]
        end
        return q
    end
end

# ===================================================================
# Part D: QV remapping — 137→34 mass-weighted
# ===================================================================
"""Parse the merge_map from the raw JSON header of a v4 binary."""
function _read_merge_map(bin_path::String)
    io = open(bin_path, "r")
    raw = read(io, 16384)
    json_end = something(findfirst(==(0x00), raw), 16385) - 1
    hdr = JSON3.read(String(raw[1:json_end]))
    close(io)
    return Int.(collect(hdr.merge_map))
end

@testset "Part D: QV remapping (137→34)" begin
    reader = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m_moist, ps, fluxes_moist = AtmosTransport.load_window!(reader, WIN)
    Nx, Ny, Nz = reader.header.Nx, reader.header.Ny, reader.header.Nz
    Nz_native = 137

    merge_map = _read_merge_map(BIN_PATH)
    @test length(merge_map) == 137
    @test minimum(merge_map) == 1
    @test maximum(merge_map) == 34

    # Load native A/B
    A_native, B_native = load_native_ab_coefficients(COEFF_PATH)
    @test length(A_native) == 138

    # Load QV from thermo NetCDF (window 1 = hour 1)
    qv_native = read_qv_from_thermo(THERMO_PATH, WIN, Nx, Ny, Nz_native)
    @test size(qv_native) == (Nx, Ny, Nz_native)

    # Compute native dp
    reader = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m_moist2, ps2, _ = AtmosTransport.load_window!(reader, WIN)
    dp_native = compute_dp_native(A_native, B_native, ps2, Nx, Ny)
    @test size(dp_native) == (Nx, Ny, 137)

    # Mass-weighted remap
    qv_merged = Array{FT}(undef, Nx, Ny, Nz)
    merge_qv_mass_weighted!(qv_merged, qv_native, dp_native, merge_map, Nx, Ny, Nz)

    # Sanity: QV should be in [0, ~0.03] range (max ~3% specific humidity)
    @printf("  QV merged: min=%.6f  max=%.6f  mean=%.6f\n",
            minimum(qv_merged), maximum(qv_merged), sum(qv_merged) / length(qv_merged))
    @test all(qv_merged .>= 0)
    @test maximum(qv_merged) < 0.05
    @test sum(qv_merged) > 0  # not all zeros

    # Conservation check: total moisture mass
    # Σ dp_native × qv_native ≈ Σ dp_merged × qv_merged for each column
    # where dp_merged can be derived from the merged A/B and ps
    B_merged = AtmosTransport.B_ifc(reader)
    A_merged = reader.header.A_ifc
    dp_merged = Array{FT}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_merged[i, j, k] = (A_merged[k+1] - A_merged[k]) + (B_merged[k+1] - B_merged[k]) * ps2[i, j]
    end

    # Per-column moisture mass: Σ_k dp[k] × qv[k]
    q_mass_native = zeros(FT, Nx, Ny)
    @inbounds for k in 1:Nz_native, j in 1:Ny, i in 1:Nx
        q_mass_native[i, j] += dp_native[i, j, k] * FT(qv_native[i, j, k])
    end
    q_mass_merged = zeros(FT, Nx, Ny)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        q_mass_merged[i, j] += dp_merged[i, j, k] * qv_merged[i, j, k]
    end

    # The merge_map remaps dp_native → dp_merged exactly (sum is preserved),
    # so the dp-weighted QV merge should conserve column moisture mass
    max_col_diff = maximum(abs.(q_mass_native .- q_mass_merged))
    total_native = sum(q_mass_native)
    total_merged = sum(q_mass_merged)
    rel_global = abs(total_native - total_merged) / total_native

    @printf("  Column moisture: max_abs_diff=%.4e  global_rel_diff=%.4e\n",
            max_col_diff, rel_global)
    @test rel_global < 1e-6

    @info "Part D: QV remapping ✓"
    close(reader)
end

# ===================================================================
# Part E: Full moist→dry conversion + advection + conservation
# ===================================================================
@testset "Part E: Dry conversion + advection + conservation" begin
    reader = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m_moist, ps, fluxes_moist = AtmosTransport.load_window!(reader, WIN)
    am_moist, bm_moist = fluxes_moist.am, fluxes_moist.bm
    Nx, Ny, Nz = reader.header.Nx, reader.header.Ny, reader.header.Nz

    # --- Load and merge QV ---
    merge_map = _read_merge_map(BIN_PATH)

    A_native, B_native = load_native_ab_coefficients(COEFF_PATH)
    dp_native = compute_dp_native(A_native, B_native, ps, Nx, Ny)
    qv_native = read_qv_from_thermo(THERMO_PATH, WIN, Nx, Ny, 137)
    qv_merged = Array{FT}(undef, Nx, Ny, Nz)
    merge_qv_mass_weighted!(qv_merged, qv_native, dp_native, merge_map, Nx, Ny, Nz)

    # --- Build dry fluxes ---
    mesh = AtmosTransport.LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vc = AtmosTransport.HybridSigmaPressure(
        FT.(collect(reader.header.A_ifc)),
        FT.(collect(reader.header.B_ifc)))
    grid = AtmosTransport.AtmosGrid(mesh, vc, CPU())
    driver = AtmosTransport.PreprocessedERA5Driver(
        AtmosTransport.window_count(reader), reader.header.dt_seconds,
        reader.header.steps_per_met)
    closure = AtmosTransport.DiagnoseVerticalFromHorizontal()

    cell_mass_dry = Array{FT}(undef, Nx, Ny, Nz)
    am_dry = copy(am_moist)
    bm_dry = copy(bm_moist)
    cm_dry = zeros(FT, Nx, Ny, Nz+1)
    dry_fluxes = AtmosTransport.StructuredFaceFluxState{AtmosTransport.DryMassFluxBasis}(
        am_dry, bm_dry, cm_dry)

    AtmosTransport.build_dry_fluxes!(dry_fluxes, cell_mass_dry, fluxes_moist,
                                        m_moist, qv_merged, grid, driver, closure)

    # --- Diagnostic 1: Dry air mass ---
    total_moist = sum(m_moist)
    total_dry = sum(cell_mass_dry)
    moisture_fraction = 1.0 - total_dry / total_moist

    @printf("\n  === Dry conversion diagnostics ===\n")
    @printf("  Total moist air mass:  %.10e kg\n", total_moist)
    @printf("  Total dry air mass:    %.10e kg\n", total_dry)
    @printf("  Moisture fraction:     %.6f %%\n", moisture_fraction * 100)
    @printf("  QV mean:               %.6f\n", sum(qv_merged .* m_moist) / total_moist)

    @test total_dry < total_moist  # dry < moist
    @test moisture_fraction > 0.001  # > 0.1% moisture
    @test moisture_fraction < 0.010  # < 1% moisture

    # --- Diagnostic 2: Per-column dry mass closure ---
    # cm should satisfy continuity: cm[1]=0, cm[Nz+1]≈0
    max_cm_toa = maximum(abs.(cm_dry[:, :, 1]))
    max_cm_sfc = maximum(abs.(cm_dry[:, :, Nz+1]))
    @printf("  CM TOA boundary:       %.4e\n", max_cm_toa)
    @printf("  CM surface residual:   %.4e\n", max_cm_sfc)
    @test max_cm_toa == 0.0
    @test max_cm_sfc < 1e-3

    # --- Diagnostic 3: CFL on dry fluxes ---
    max_cfl = zero(FT)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mi = max(cell_mass_dry[i,j,k], FT(1))
        cx = max(abs(am_dry[i,j,k]), abs(am_dry[i+1,j,k])) / mi
        cy = max(abs(bm_dry[i,j,k]), abs(bm_dry[i,j+1,k])) / mi
        max_cfl = max(max_cfl, cx, cy)
    end
    @printf("  Max CFL (dry):         %.2f\n", max_cfl)

    # --- Scale for CFL safety and run advection ---
    scale = FT(0.8) / max_cfl
    am_dry .*= scale
    bm_dry .*= scale

    # Re-diagnose cm from scaled dry fluxes
    dB = FT[reader.header.B_ifc[k+1] - reader.header.B_ifc[k] for k in 1:Nz]
    AtmosTransport.diagnose_cm_from_continuity!(cm_dry, am_dry, bm_dry, dB, Nx, Ny, Nz)

    # Set up tracer at 411 ppm on DRY basis
    vmr0 = FT(411e-6)
    rm0 = cell_mass_dry .* vmr0
    total_dry_mass0 = sum(cell_mass_dry)
    total_tracer0 = sum(rm0)

    cell_state = AtmosTransport.CellState(copy(cell_mass_dry); CO2=copy(rm0))
    dry_fluxes_scaled = AtmosTransport.StructuredFaceFluxState{AtmosTransport.DryMassFluxBasis}(
        am_dry, bm_dry, cm_dry)
    ws = AtmosTransport.AdvectionWorkspace(cell_mass_dry)
    scheme = AtmosTransport.RussellLernerAdvection(use_limiter=true)

    AtmosTransport.strang_split!(cell_state, dry_fluxes_scaled, grid, scheme;
                                    workspace=ws)

    total_dry_mass1 = sum(cell_state.air_dry_mass)
    total_tracer1 = sum(cell_state.tracers.CO2)
    dry_mass_drift = (total_dry_mass1 - total_dry_mass0) / total_dry_mass0
    tracer_drift = (total_tracer1 - total_tracer0) / total_tracer0

    # VMR after advection
    vmr_after = cell_state.tracers.CO2 ./ cell_state.air_dry_mass
    vmr_min = minimum(vmr_after)
    vmr_max = maximum(vmr_after)
    vmr_rms = sqrt(sum((vmr_after .- vmr0).^2) / length(vmr_after))

    @printf("\n  === Advection on dry state ===\n")
    @printf("  Dry air mass drift:    %.4e\n", dry_mass_drift)
    @printf("  Tracer mass drift:     %.4e\n", tracer_drift)
    @printf("  VMR: min=%.6e  max=%.6e  rms_dev=%.4e\n", vmr_min, vmr_max, vmr_rms)

    @test abs(dry_mass_drift) < 1e-12
    @test abs(tracer_drift)   < 1e-12

    # VMR should stay close to 411e-6 for uniform initial condition
    @test vmr_min > 400e-6
    @test vmr_max < 420e-6

    @info "Part E: Dry conversion + advection + conservation ✓"
    close(reader)
end

@info "All dry-conversion tests passed"
