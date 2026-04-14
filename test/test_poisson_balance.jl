#!/usr/bin/env julia
#
# Unit tests for the Poisson mass-flux balance solvers.
# Tests synthetic cases where the exact solution is known.

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
Prep = AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------
# LL FFT Poisson balance
# ---------------------------------------------------------------------------

@testset "LL FFT Poisson balance" begin
    @testset "Already-balanced fluxes → no change" begin
        Nx, Ny, Nz = 24, 12, 2
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        am0 = copy(am); bm0 = copy(bm)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        @test am ≈ am0 atol=1e-14
        @test bm ≈ bm0 atol=1e-14
    end

    @testset "Uniform divergence → corrected to match dm_dt" begin
        Nx, Ny, Nz = 36, 18, 4
        # Create fluxes with known divergence
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        # Put some divergence in the fluxes
        for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
            am[i, j, k] = 0.01 * sin(2π * i / Nx) * cos(2π * j / Ny)
        end
        for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
            bm[i, j, k] = 0.01 * cos(2π * i / Nx) * sin(2π * j / Ny)
        end

        # Target: zero mass tendency
        fill!(dm_dt, 0.0)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        # After balance, convergence should match dm_dt at each cell
        max_residual = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            conv = (am[i,j,k] - am[i+1,j,k]) + (bm[i,j,k] - bm[i,j+1,k])
            residual = abs(conv - dm_dt[i,j,k])
            max_residual = max(max_residual, residual)
        end

        # Should be at machine precision
        @test max_residual < 1e-12
    end

    @testset "Non-zero dm_dt → convergence matches target" begin
        Nx, Ny, Nz = 48, 24, 2
        am = zeros(Float64, Nx+1, Ny, Nz)
        bm = zeros(Float64, Nx, Ny+1, Nz)
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        # Non-zero mass tendency pattern
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            dm_dt[i,j,k] = 1e-3 * sin(4π * i / Nx) * sin(2π * j / Ny)
        end
        # Subtract global mean (Poisson can't fix the mean)
        for k in 1:Nz
            dm_dt[:,:,k] .-= sum(dm_dt[:,:,k]) / (Nx * Ny)
        end

        ws = Prep.LLPoissonWorkspace(Nx, Ny)
        Prep.balance_mass_fluxes!(am, bm, dm_dt, ws)

        max_residual = 0.0
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            conv = (am[i,j,k] - am[i+1,j,k]) + (bm[i,j,k] - bm[i,j+1,k])
            residual = abs(conv - dm_dt[i,j,k])
            max_residual = max(max_residual, residual)
        end

        @test max_residual < 1e-12
    end

    @testset "Workspace reuse across levels" begin
        Nx, Ny, Nz = 24, 12, 8
        am = rand(Float64, Nx+1, Ny, Nz) .* 0.01
        bm = rand(Float64, Nx, Ny+1, Nz) .* 0.01
        dm_dt = zeros(Float64, Nx, Ny, Nz)

        ws = Prep.LLPoissonWorkspace(Nx, Ny)

        # Run twice — workspace should produce same result both times
        am1 = copy(am); bm1 = copy(bm)
        Prep.balance_mass_fluxes!(am1, bm1, dm_dt, ws)

        am2 = copy(am); bm2 = copy(bm)
        Prep.balance_mass_fluxes!(am2, bm2, dm_dt, ws)

        @test am1 ≈ am2 atol=1e-15
        @test bm1 ≈ bm2 atol=1e-15
    end
end

# ---------------------------------------------------------------------------
# CS global Poisson balance
# ---------------------------------------------------------------------------

@testset "CS global Poisson balance" begin
    if isdefined(Prep, :balance_cs_global_mass_fluxes!)
        using .AtmosTransport.Grids: CubedSphereMesh, GnomonicPanelConvention,
            default_panel_connectivity

        @testset "Zero fluxes → zero correction" begin
            Nc = 8; Nz = 2
            mesh = CubedSphereMesh(; Nc=Nc, FT=Float64, convention=GnomonicPanelConvention())
            conn = default_panel_connectivity()
            ft = Prep.build_cs_global_face_table(Nc, conn)
            degree = Prep.cs_cell_face_degree(ft)
            scratch = Prep.CSPoissonScratch(ft.nc)

            m = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
            m_next = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
            am = ntuple(_ -> zeros(Float64, Nc+1, Nc, Nz), 6)
            bm = ntuple(_ -> zeros(Float64, Nc, Nc+1, Nz), 6)

            diag = Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next, ft, degree, 4, scratch)

            for p in 1:6
                @test maximum(abs, am[p]) < 1e-14
                @test maximum(abs, bm[p]) < 1e-14
            end
        end

        @testset "Random fluxes → post-balance residual near machine precision" begin
            Nc = 12; Nz = 2; steps = 4
            mesh = CubedSphereMesh(; Nc=Nc, FT=Float64, convention=GnomonicPanelConvention())
            conn = default_panel_connectivity()
            ft = Prep.build_cs_global_face_table(Nc, conn)
            degree = Prep.cs_cell_face_degree(ft)
            scratch = Prep.CSPoissonScratch(ft.nc)

            # Same mass at both endpoints (zero target) with random initial fluxes
            m = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
            am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
            bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

            diag = Prep.balance_cs_global_mass_fluxes!(am, bm, m, m,
                                                        ft, degree, steps, scratch;
                                                        tol=1e-14, max_iter=10000)

            # Post-balance: convergence matches zero target to machine precision
            @test diag.max_post_residual < 1e-10
        end

        @testset "All cells have degree 4" begin
            for Nc in [4, 8, 16]
                conn = default_panel_connectivity()
                ft = Prep.build_cs_global_face_table(Nc, conn)
                degree = Prep.cs_cell_face_degree(ft)
                @test all(degree .== 4)
                @test length(degree) == 6 * Nc^2
            end
        end

        @testset "cm diagnosis gives zero bottom boundary" begin
            Nc = 8; Nz = 4
            am = ntuple(_ -> zeros(Float64, Nc+1, Nc, Nz), 6)
            bm = ntuple(_ -> zeros(Float64, Nc, Nc+1, Nz), 6)
            dm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
            m  = ntuple(_ -> ones(Float64, Nc, Nc, Nz), 6)
            cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz+1), 6)

            Prep.diagnose_cs_cm!(cm, am, bm, dm, m, Nc, Nz)

            # With zero fluxes and zero dm, all cm should be zero
            for p in 1:6
                @test maximum(abs, cm[p]) < 1e-14
            end
        end
    else
        @test true  # skip if CS balance not available
    end
end
