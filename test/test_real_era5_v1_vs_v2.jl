#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Real-data gating test: v1 vs v2 on ERA5 LatLon preprocessed binary
#
# Proves the v2 refactoring preserves the existing moist-basis path by
# loading one real window and running identical advection through both
# src/ (v1) and src/ (v2), then comparing field-level diagnostics.
#
# NOT a CI test — requires real ERA5 binary data.
# ---------------------------------------------------------------------------

using Test
using Printf
using KernelAbstractions

const BIN_PATH = expanduser(
    "~/data/AtmosTransport/met/era5/spectral_v4_tropo34_dec2021/" *
    "era5_v4_20211201_merged1000Pa_float32.bin")

if !isfile(BIN_PATH)
    @warn "Skipping real-data test: binary not found at $BIN_PATH"
    exit(0)
end

ENV["ATMOSTR_NO_STALE_CHECK"] = "1"
ENV["ATMOSTR_NO_CM_CHECK"] = "1"

# Load both modules
include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))

using .AtmosTransport
using .AtmosTransport

using KernelAbstractions: CPU

const FT = Float64
const WIN = 1

# ===================================================================
# Part A: Raw data loading — both readers produce identical arrays
# ===================================================================
@testset "Part A: Binary reader parity" begin
    # --- v2 reader ---
    r2 = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m2, ps2, fluxes2 = AtmosTransport.load_window!(r2, WIN)
    am2, bm2, cm2_raw = fluxes2.am, fluxes2.bm, fluxes2.cm

    Nx, Ny, Nz = r2.header.Nx, r2.header.Ny, r2.header.Nz
    @test (Nx, Ny, Nz) == (720, 361, 34)

    # --- v1 reader ---
    r1 = AtmosTransport.IO.MassFluxBinaryReader(BIN_PATH, FT)
    m1  = Array{FT}(undef, Nx, Ny, Nz)
    am1 = Array{FT}(undef, Nx+1, Ny, Nz)
    bm1 = Array{FT}(undef, Nx, Ny+1, Nz)
    cm1_raw = Array{FT}(undef, Nx, Ny, Nz+1)
    ps1 = Array{FT}(undef, Nx, Ny)
    AtmosTransport.IO.load_window!(m1, am1, bm1, cm1_raw, ps1, r1, WIN)

    @test m1 == m2
    @test am1 == am2
    @test bm1 == bm2
    @test cm1_raw == cm2_raw
    @test ps1 == ps2

    @info "Part A: Reader parity ✓ — all fields identical"

    close(r2)
    close(r1.io)
end

# ===================================================================
# Part B: CM diagnosis — v1 (Neumaier) vs v2 (naive) on real data
# ===================================================================
@testset "Part B: CM diagnosis parity" begin
    r2 = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m, ps, fluxes = AtmosTransport.load_window!(r2, WIN)
    am, bm = fluxes.am, fluxes.bm
    Nx, Ny, Nz = r2.header.Nx, r2.header.Ny, r2.header.Nz

    B_ifc = AtmosTransport.B_ifc(r2)
    dB = FT[B_ifc[k+1] - B_ifc[k] for k in 1:Nz]

    # --- v2 cm diagnosis (naive summation) ---
    cm_v2 = zeros(FT, Nx, Ny, Nz+1)
    AtmosTransport.diagnose_cm_from_continuity!(cm_v2, am, bm, dB, Nx, Ny, Nz)

    # --- v1 cm diagnosis (Neumaier compensated) ---
    cm_v1 = zeros(FT, Nx, Ny, Nz+1)
    dB_vec = FT.(dB)
    AtmosTransport.Models.recompute_cm_runtime!(cm_v1, am, bm, dB_vec)

    max_abs_diff = maximum(abs.(cm_v1 .- cm_v2))
    cm_range = maximum(abs.(cm_v1))
    overall_rel = max_abs_diff / cm_range

    @printf("  CM max absolute diff:  %.4e\n", max_abs_diff)
    @printf("  CM range (max |cm|):   %.4e\n", cm_range)
    @printf("  CM overall relative:   %.4e  (abs_diff / range)\n", overall_rel)

    # v1 uses Neumaier compensated summation, v2 uses naive. In Float64 the
    # difference is O(1e-6) absolute, which is ~1e-12 relative to the cm range.
    @test overall_rel < 1e-10

    # Check surface closure: cm[Nz+1] should be ≈ 0
    # v1 forces cm[Nz+1]=0; v2 lets it float as a residual
    sfc_residual_v2 = maximum(abs.(cm_v2[:, :, Nz+1]))
    sfc_residual_v1 = maximum(abs.(cm_v1[:, :, Nz+1]))  # forced to 0
    @printf("  Surface residual v2: %.4e\n", sfc_residual_v2)
    @printf("  Surface residual v1: %.4e (forced to 0)\n", sfc_residual_v1)
    @test sfc_residual_v2 < 1e-3  # should be small by construction
    @test sfc_residual_v1 == 0.0

    @info "Part B: CM diagnosis parity ✓"
    close(r2)
end

# ===================================================================
# Part C: Advection comparison — one Strang split, CFL-scaled fluxes
# ===================================================================
@testset "Part C: Strang split parity (CFL-scaled)" begin
    r2 = AtmosTransport.ERA5BinaryReader(BIN_PATH; FT=FT)
    m_raw, ps, fluxes_raw = AtmosTransport.load_window!(r2, WIN)
    am_raw, bm_raw = copy(fluxes_raw.am), copy(fluxes_raw.bm)
    Nx, Ny, Nz = r2.header.Nx, r2.header.Ny, r2.header.Nz

    B_ifc = AtmosTransport.B_ifc(r2)
    dB = FT[B_ifc[k+1] - B_ifc[k] for k in 1:Nz]

    # Find max CFL and scale fluxes so max CFL < 0.8
    max_cfl = zero(FT)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mi = max(m_raw[i,j,k], FT(1))
        cx = max(abs(am_raw[i,j,k]), abs(am_raw[i+1,j,k])) / mi
        cy = max(abs(bm_raw[i,j,k]), abs(bm_raw[i,j+1,k])) / mi
        max_cfl = max(max_cfl, cx, cy)
    end
    scale_factor = FT(0.8) / max_cfl
    @printf("  Max CFL before scaling: %.2f, scale factor: %.6f\n", max_cfl, scale_factor)

    am_scaled = am_raw .* scale_factor
    bm_scaled = bm_raw .* scale_factor

    # Re-diagnose cm from scaled fluxes
    cm_diag = zeros(FT, Nx, Ny, Nz+1)
    AtmosTransport.diagnose_cm_from_continuity!(cm_diag, am_scaled, bm_scaled, dB, Nx, Ny, Nz)

    # Uniform tracer at 411 ppm
    vmr0 = FT(411e-6)
    m0 = copy(m_raw)
    total_m0 = sum(m0)
    rm0 = m0 .* vmr0
    total_rm0 = sum(rm0)

    # --- v1 path ---
    m_v1  = copy(m0)
    rm_v1 = copy(rm0)
    am_v1 = copy(am_scaled)
    bm_v1 = copy(bm_scaled)
    cm_v1 = copy(cm_diag)

    vc = AtmosTransport.Grids.HybridSigmaPressure(
        FT.(collect(r2.header.A_ifc)),
        FT.(collect(r2.header.B_ifc)))
    grid_v1 = AtmosTransport.Grids.LatitudeLongitudeGrid(
        AtmosTransport.Architectures.CPU();
        FT=FT, size=(Nx, Ny, Nz), vertical=vc,
        use_reduced_grid=false)

    tracers_v1 = (; CO2 = rm_v1)
    ws_v1 = AtmosTransport.Advection.allocate_massflux_workspace(m_v1, am_v1, bm_v1, cm_v1)
    AtmosTransport.Advection.strang_split_massflux!(
        tracers_v1, m_v1, am_v1, bm_v1, cm_v1,
        grid_v1, true, ws_v1; cfl_limit=FT(1.0))

    # --- v2 path ---
    m_v2  = copy(m0)
    rm_v2 = copy(rm0)
    am_v2 = copy(am_scaled)
    bm_v2 = copy(bm_scaled)
    cm_v2 = copy(cm_diag)

    mesh_v2 = AtmosTransport.LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vc_v2 = AtmosTransport.HybridSigmaPressure(
        FT.(collect(r2.header.A_ifc)),
        FT.(collect(r2.header.B_ifc)))
    grid_v2 = AtmosTransport.AtmosGrid(mesh_v2, vc_v2, CPU())

    cell_state = AtmosTransport.CellState(m_v2; CO2 = rm_v2)
    dry_fluxes = AtmosTransport.StructuredFaceFluxState{AtmosTransport.DryMassFluxBasis}(
        am_v2, bm_v2, cm_v2)
    ws_v2 = AtmosTransport.AdvectionWorkspace(m_v2)
    scheme_v2 = AtmosTransport.SlopesScheme(AtmosTransport.MonotoneLimiter())
    AtmosTransport.strang_split!(cell_state, dry_fluxes, grid_v2, scheme_v2;
                                    workspace=ws_v2)

    # --- Comparison ---
    diff_m  = m_v1 .- m_v2
    diff_rm = rm_v1 .- rm_v2
    max_diff_m  = maximum(abs.(diff_m))
    max_diff_rm = maximum(abs.(diff_rm))
    rms_diff_m  = sqrt(sum(diff_m.^2) / length(diff_m))
    rms_diff_rm = sqrt(sum(diff_rm.^2) / length(diff_rm))

    total_m_v1  = sum(m_v1)
    total_m_v2  = sum(m_v2)
    total_rm_v1 = sum(rm_v1)
    total_rm_v2 = sum(rm_v2)

    @printf("\n  === Field comparison (v1 vs v2) ===\n")
    @printf("  Air mass:  max_diff=%.4e  rms_diff=%.4e\n", max_diff_m, rms_diff_m)
    @printf("  Tracer:    max_diff=%.4e  rms_diff=%.4e\n", max_diff_rm, rms_diff_rm)
    @printf("  Total m:   v1=%.10e  v2=%.10e  diff=%.4e\n", total_m_v1, total_m_v2, total_m_v1 - total_m_v2)
    @printf("  Total rm:  v1=%.10e  v2=%.10e  diff=%.4e\n", total_rm_v1, total_rm_v2, total_rm_v1 - total_rm_v2)

    # Mass conservation
    m_drift_v1 = (total_m_v1 - total_m0) / total_m0
    m_drift_v2 = (total_m_v2 - total_m0) / total_m0
    rm_drift_v1 = (total_rm_v1 - total_rm0) / total_rm0
    rm_drift_v2 = (total_rm_v2 - total_rm0) / total_rm0

    @printf("\n  === Mass conservation ===\n")
    @printf("  Air mass drift:   v1=%.4e  v2=%.4e\n", m_drift_v1, m_drift_v2)
    @printf("  Tracer drift:     v1=%.4e  v2=%.4e\n", rm_drift_v1, rm_drift_v2)

    # v1 and v2 should match to machine precision
    @test max_diff_m  < 1e-10
    @test max_diff_rm < 1e-10

    # Both should conserve mass
    @test abs(m_drift_v1) < 1e-12
    @test abs(m_drift_v2) < 1e-12
    @test abs(rm_drift_v1) < 1e-12
    @test abs(rm_drift_v2) < 1e-12

    @info "Part C: Strang split parity ✓ — v1 and v2 match on real data"
    close(r2)
end

@info "All real-data tests passed"
