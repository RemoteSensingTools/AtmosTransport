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
        @testset "mirror_sign: inflow positions get +1, outflow get -1" begin
            for Nc in [4, 8, 12]
                conn = default_panel_connectivity()
                ft = Prep.build_cs_global_face_table(Nc, conn)

                for f in 1:ft.nf
                    mq = Int(ft.mirror_panel[f])
                    mq == 0 && continue  # interior face

                    mdir = Int(ft.mirror_dir[f])
                    mi   = Int(ft.mirror_idx_i[f])
                    mj   = Int(ft.mirror_idx_j[f])
                    ms   = Int(ft.mirror_sign[f])

                    # Mirror at outflow position (index Nc+1) → sign must be -1
                    # Mirror at inflow position (index 1 or interior) → sign must be +1
                    at_outflow = (mdir == 1 && mi == Nc + 1) ||
                                 (mdir == 2 && mj == Nc + 1)
                    expected = at_outflow ? -1 : 1
                    @test ms == expected
                end
            end
        end

        @testset "mirror_sign: per-panel div matches global face-table div" begin
            Nc = 8; Nz = 3; steps = 4
            conn = default_panel_connectivity()
            ft = Prep.build_cs_global_face_table(Nc, conn)
            degree = Prep.cs_cell_face_degree(ft)
            scratch = Prep.CSPoissonScratch(ft.nc)
            nc = 6 * Nc^2

            # Non-trivial mass fields to create a real balance problem
            m      = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
            m_next = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
            am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
            bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

            Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next,
                ft, degree, steps, scratch; tol=1e-14, max_iter=20000)

            # After balance (which calls _sync_cs_mirrors!), compare divergences.
            for k in 1:Nz
                # Global face-table divergence
                div_global = zeros(Float64, nc)
                @inbounds for f in 1:ft.nf
                    p   = Int(ft.face_panel[f])
                    dir = Int(ft.face_dir[f])
                    i   = Int(ft.face_idx_i[f])
                    j   = Int(ft.face_idx_j[f])
                    flux = dir == 1 ? am[p][i, j, k] : bm[p][i, j, k]

                    left  = Int(ft.face_left[f])
                    right = Int(ft.face_right[f])
                    div_global[left]  += flux
                    div_global[right] -= flux
                end

                # Per-panel divergence (outflow convention) from am/bm arrays.
                # div = (am[i+1] - am[i]) + (bm[j+1] - bm[j]) = net outflow.
                # Boundary faces use mirror entries — this is what mirror_sign
                # must get right.
                div_panel = zeros(Float64, nc)
                @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
                    c = (p - 1) * Nc^2 + (j - 1) * Nc + i
                    div_panel[c] = (am[p][i+1, j, k] - am[p][i, j, k]) +
                                   (bm[p][i, j+1, k] - bm[p][i, j, k])
                end

                # They must match at every cell — this fails without mirror_sign
                max_diff = maximum(abs, div_global .- div_panel)
                @test max_diff < 1e-13
            end
        end

        @testset "mirror_sign: balance + cm diagnosis conserves mass per column" begin
            Nc = 8; Nz = 4; steps = 4
            conn = default_panel_connectivity()
            ft = Prep.build_cs_global_face_table(Nc, conn)
            degree = Prep.cs_cell_face_degree(ft)
            scratch = Prep.CSPoissonScratch(ft.nc)

            m      = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)
            m_next = ntuple(p -> 1.0 .+ 0.1 .* rand(Float64, Nc, Nc, Nz), 6)

            # Make per-level global mass match (Poisson requires Σ target = 0 per level)
            for k in 1:Nz
                total_cur  = sum(p -> sum(m[p][:,:,k]), 1:6)
                total_next = sum(p -> sum(m_next[p][:,:,k]), 1:6)
                offset = (total_cur - total_next) / (6 * Nc^2)
                for p in 1:6; m_next[p][:,:,k] .+= offset; end
            end

            am = ntuple(_ -> rand(Float64, Nc+1, Nc, Nz) .* 0.01, 6)
            bm = ntuple(_ -> rand(Float64, Nc, Nc+1, Nz) .* 0.01, 6)

            Prep.balance_cs_global_mass_fluxes!(am, bm, m, m_next,
                ft, degree, steps, scratch; tol=1e-14, max_iter=20000)

            # Compute dm and diagnose cm
            dm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz), 6)
            cm = ntuple(_ -> zeros(Float64, Nc, Nc, Nz+1), 6)
            inv_scale = 1.0 / (2 * steps)
            for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
                dm[p][i, j, k] = (m_next[p][i, j, k] - m[p][i, j, k]) * inv_scale
            end
            Prep.diagnose_cs_cm!(cm, am, bm, dm, m, Nc, Nz)

            # cm[k=1] = 0 by construction, cm[k=Nz+1] should be near zero
            # after residual redistribution
            max_bottom = maximum(p -> maximum(abs, cm[p][:,:,Nz+1]), 1:6)
            @test max_bottom < 1e-12
        end

    else
        @test true  # skip if CS balance not available
    end
end
