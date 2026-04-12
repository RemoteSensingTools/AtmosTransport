#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Real-data gating test: direct dry-basis binary -> advection
#
# Validates the new preprocessed dry-basis ERA5 binary path by:
#   1. Loading one real window from a dry-basis binary with embedded qv
#   2. Verifying the reader dispatches to DryMassFluxBasis
#   3. Verifying embedded qv matches the thermo source timing convention
#   4. Running one CFL-scaled Strang split directly on the dry state
#   5. Checking dry-mass and tracer-mass conservation
#
# NOT a CI test — requires real ERA5 binary + thermo data.
# ---------------------------------------------------------------------------

using Test
using Printf
using NCDatasets
using KernelAbstractions: CPU

const BIN_PATH = expanduser(
    "~/data/AtmosTransport/met/era5/0.5x0.5/preprocessed/massflux/" *
    "v4_137L_smoketest_dry_qv/era5_v4_20211201_merged1Pa_float32.bin")
const THERMO_PATH = expanduser(
    "~/data/AtmosTransport/met/era5/0.5x0.5/physics/era5_thermo_ml_20211201.nc")

for (name, path) in [("binary", BIN_PATH), ("thermo", THERMO_PATH)]
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
const WIN = 1

"""
    read_qv_from_thermo(path, hour_idx, Nx, Ny, Nz) -> Array{Float64, 3}

Read ERA5 model-level specific humidity using the same dimension and latitude
handling as the preprocessor.
"""
function read_qv_from_thermo(path::String, hour_idx::Int, Nx::Int, Ny::Int, Nz::Int)
    NCDataset(path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        q = if dims[1] == "longitude"
            FT.(q_var[:, :, :, hour_idx])
        else
            q_raw = FT.(q_var[hour_idx, :, :, :])
            permutedims(q_raw, (3, 2, 1))
        end
        if size(q, 2) == Ny && ds["latitude"][1] > ds["latitude"][end]
            q = q[:, end:-1:1, :]
        end
        size(q) == (Nx, Ny, Nz) || error("Unexpected qv shape $(size(q))")
        return q
    end
end

"""
    max_cfl(cell_mass, am, bm) -> Float64

Return the maximum horizontal CFL-like mass-flux ratio used for the real-data
stability smoke test.
"""
function max_cfl(cell_mass, am, bm)
    Nx, Ny, Nz = size(cell_mass)
    c = zero(FT)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mi = max(cell_mass[i, j, k], FT(1))
        cx = max(abs(am[i, j, k]), abs(am[i + 1, j, k])) / mi
        cy = max(abs(bm[i, j, k]), abs(bm[i, j + 1, k])) / mi
        c = max(c, cx, cy)
    end
    return c
end

@testset "Direct dry-basis binary -> advection" begin
    reader = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    try
        @test AtmosTransport.mass_basis(reader) === :dry
        @test AtmosTransport.has_qv(reader) == true
        @test AtmosTransport.window_count(reader) == 24

        m_dry, ps, dry_fluxes = AtmosTransport.load_window!(reader, WIN)
        qv = AtmosTransport.load_qv_window!(reader, WIN)

        Nx, Ny, Nz = size(m_dry)
        @test size(ps) == (Nx, Ny)
        @test dry_fluxes isa AtmosTransport.StructuredFaceFluxState{AtmosTransport.DryMassFluxBasis}
        @test qv !== nothing
        @test size(qv) == (Nx, Ny, Nz)
        @test all(m_dry .> 0)

        # Verify the embedded qv payload uses the documented time alignment.
        qv_src = read_qv_from_thermo(THERMO_PATH, WIN, Nx, Ny, Nz)
        max_qv_diff = maximum(abs.(qv .- qv_src))
        rms_qv_diff = sqrt(sum(abs2, qv .- qv_src) / length(qv))
        @printf("  QV payload check: max_abs=%.6e  rms=%.6e\n", max_qv_diff, rms_qv_diff)
        @test max_qv_diff < 1e-8

        mesh = AtmosTransport.LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
        vc = AtmosTransport.HybridSigmaPressure(
            FT.(reader.header.A_ifc),
            FT.(reader.header.B_ifc),
        )
        grid = AtmosTransport.AtmosGrid(mesh, vc, CPU())

        raw_cfl = max_cfl(m_dry, dry_fluxes.am, dry_fluxes.bm)
        scale = FT(0.8) / raw_cfl
        @printf("  Max CFL before scaling: %.3f  scale: %.6f\n", raw_cfl, scale)
        @test raw_cfl > 1

        am = copy(dry_fluxes.am)
        bm = copy(dry_fluxes.bm)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        am .*= scale
        bm .*= scale

        dB = FT[reader.header.B_ifc[k + 1] - reader.header.B_ifc[k] for k in 1:Nz]
        AtmosTransport.diagnose_cm_from_continuity!(cm, am, bm, dB, Nx, Ny, Nz)
        @test maximum(abs.(cm[:, :, 1])) == 0.0
        @test maximum(abs.(cm[:, :, Nz + 1])) < 1e-5

        vmr0 = FT(411e-6)
        tracer0 = m_dry .* vmr0
        state = AtmosTransport.CellState(copy(m_dry); CO2=copy(tracer0))
        fluxes_scaled = AtmosTransport.StructuredFaceFluxState{AtmosTransport.DryMassFluxBasis}(am, bm, cm)
        ws = AtmosTransport.AdvectionWorkspace(m_dry)
        scheme = AtmosTransport.RussellLernerAdvection(use_limiter=true)

        total_m0 = sum(state.air_dry_mass)
        total_t0 = sum(state.tracers.CO2)
        AtmosTransport.strang_split!(state, fluxes_scaled, grid, scheme; workspace=ws)
        total_m1 = sum(state.air_dry_mass)
        total_t1 = sum(state.tracers.CO2)

        dry_mass_drift = (total_m1 - total_m0) / total_m0
        tracer_drift = (total_t1 - total_t0) / total_t0
        vmr = state.tracers.CO2 ./ state.air_dry_mass

        @printf("  Dry mass drift:   %.6e\n", dry_mass_drift)
        @printf("  Tracer drift:     %.6e\n", tracer_drift)
        @printf("  VMR min/max:      %.6e / %.6e\n", minimum(vmr), maximum(vmr))

        @test abs(dry_mass_drift) < 1e-12
        @test abs(tracer_drift) < 1e-12
        @test minimum(vmr) > 410e-6
        @test maximum(vmr) < 412e-6
    finally
        close(reader)
    end
end

@info "Real dry-binary test passed"
