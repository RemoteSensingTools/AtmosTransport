#!/usr/bin/env julia
#
# Regression test for cs_global_poisson_balance.jl
#
# Tests:
#   1. Face table: correct count, all-degree-4, no duplicates
#   2. CG solver: round-trip L*psi → solve → psi ≈ psi_true
#   3. Full balance: random fluxes → global div matches dm_dt (up to null-space)
#   4. Mirror consistency: canonical = mirror for all cross-panel faces
#   5. Per-panel ↔ global divergence agreement after balance
#   6. Scaling: C24 × 34 levels completes in reasonable time

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

include(joinpath(@__DIR__, "cs_global_poisson_balance.jl"))

using Test
using Printf
using LinearAlgebra: dot

@testset "CS Global Poisson Balance" begin

    conn = default_panel_connectivity()

    @testset "Face table (C4)" begin
        Nc = 4
        ft = build_cs_global_face_table(Nc, conn)
        @test ft.nf == 12 * Nc^2
        @test ft.nc == 6 * Nc^2
        @test all(1 .<= ft.face_left  .<= ft.nc)
        @test all(1 .<= ft.face_right .<= ft.nc)
        @test all(ft.face_left .!= ft.face_right)

        # No duplicates
        pairs = Set{Tuple{Int32, Int32}}()
        for f in 1:ft.nf
            pair = minmax(ft.face_left[f], ft.face_right[f])
            @test pair ∉ pairs
            push!(pairs, pair)
        end

        # All degree 4
        deg = cs_cell_face_degree(ft)
        @test all(deg .== 4)

        # 48 cross-panel faces
        @test count(ft.mirror_panel .> 0) == 12 * Nc
    end

    @testset "CG solver round-trip (C4)" begin
        Nc = 4
        ft = build_cs_global_face_table(Nc, conn)
        deg = cs_cell_face_degree(ft)
        scratch = CSPoissonScratch(ft.nc)

        # Smooth potential → L*psi → solve back
        psi_true = zeros(Float64, ft.nc)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            gc = _global_cell(i, j, p, Nc)
            psi_true[gc] = sin(2π * i / Nc) * cos(2π * j / Nc) * Float64(p)
        end
        _project_mean_zero!(psi_true)

        rhs = zeros(Float64, ft.nc)
        _cs_graph_laplacian_mul!(rhs, psi_true, ft, deg)

        psi_sol = zeros(Float64, ft.nc)
        cgs = (r = scratch.r, p = scratch.p, Ap = scratch.Ap, z = scratch.z)
        resid, iters = solve_cs_poisson_pcg!(psi_sol, copy(rhs), ft, deg, cgs;
                                              tol=1e-14, max_iter=5000)
        @test resid < 1e-10
        @test iters < 200

        # Round-trip accuracy
        _project_mean_zero!(psi_sol)
        @test maximum(abs, psi_sol - psi_true) < 1e-10
    end

    @testset "Full balance + mirror + div consistency (C4)" begin
        Nc = 4
        ft = build_cs_global_face_table(Nc, conn)
        deg = cs_cell_face_degree(ft)
        Nz = 3
        spw = 4

        panels_am = ntuple(_ -> randn(Float64, Nc + 1, Nc, Nz), 6)
        panels_bm = ntuple(_ -> randn(Float64, Nc, Nc + 1, Nz), 6)
        panels_m  = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)
        panels_mn = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)

        for p in 1:6
            panels_am[p][1, :, :] .= 0.0
            panels_am[p][Nc + 1, :, :] .= 0.0
            panels_bm[p][:, 1, :] .= 0.0
            panels_bm[p][:, Nc + 1, :] .= 0.0
        end

        scratch = CSPoissonScratch(ft.nc)
        diag = balance_cs_global_mass_fluxes!(
            panels_am, panels_bm, panels_m, panels_mn,
            ft, deg, spw, scratch; tol=1e-14, max_iter=5000)

        # CG converged on range(L)
        @test diag.max_post_projected < 1e-10

        # Post-balance residual bounded by null-space
        @test diag.max_post_residual ≤ diag.max_rhs_mean + 1e-10

        # Mirror consistency: canonical = mirror for all cross-panel faces
        max_mirror_diff = 0.0
        n_cross = 0
        for f in 1:ft.nf
            Int(ft.mirror_panel[f]) == 0 && continue
            n_cross += 1
            for k in 1:Nz
                p   = Int(ft.face_panel[f])
                dir = Int(ft.face_dir[f])
                ii  = Int(ft.face_idx_i[f])
                jj  = Int(ft.face_idx_j[f])
                can = dir == 1 ? panels_am[p][ii, jj, k] : panels_bm[p][ii, jj, k]

                mq  = Int(ft.mirror_panel[f])
                md  = Int(ft.mirror_dir[f])
                mi  = Int(ft.mirror_idx_i[f])
                mj  = Int(ft.mirror_idx_j[f])
                mir = md == 1 ? panels_am[mq][mi, mj, k] : panels_bm[mq][mi, mj, k]
                max_mirror_diff = max(max_mirror_diff, abs(can - mir))
            end
        end
        @test n_cross == 12 * Nc
        @test max_mirror_diff < 1e-12

        # Per-panel ↔ global divergence
        max_div_diff = 0.0
        for k in 1:Nz
            div_panel = zeros(Float64, ft.nc)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                conv = panels_am[p][i, j, k] - panels_am[p][i + 1, j, k] +
                       panels_bm[p][i, j, k] - panels_bm[p][i, j + 1, k]
                div_panel[_global_cell(i, j, p, Nc)] = -conv
            end
            div_global = zeros(Float64, ft.nc)
            for f in 1:ft.nf
                p2 = Int(ft.face_panel[f])
                d2 = Int(ft.face_dir[f])
                i2 = Int(ft.face_idx_i[f])
                j2 = Int(ft.face_idx_j[f])
                flux = d2 == 1 ? Float64(panels_am[p2][i2, j2, k]) :
                                 Float64(panels_bm[p2][i2, j2, k])
                div_global[ft.face_left[f]]  += flux
                div_global[ft.face_right[f]] -= flux
            end
            max_div_diff = max(max_div_diff, maximum(abs, div_panel - div_global))
        end
        @test max_div_diff < 1e-12
    end

    @testset "Non-zero boundary fluxes (C4)" begin
        # Test that balance works with pre-existing inconsistent boundary fluxes
        # (not just zeroed boundaries). This exercises the canonical-only divergence
        # computation for cross-panel faces.
        Nc = 4
        ft = build_cs_global_face_table(Nc, conn)
        deg = cs_cell_face_degree(ft)
        Nz = 2
        spw = 4

        panels_am = ntuple(_ -> randn(Float64, Nc + 1, Nc, Nz), 6)
        panels_bm = ntuple(_ -> randn(Float64, Nc, Nc + 1, Nz), 6)
        panels_m  = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)
        panels_mn = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)

        # Leave boundary fluxes as random (deliberately inconsistent)
        scratch = CSPoissonScratch(ft.nc)
        diag = balance_cs_global_mass_fluxes!(
            panels_am, panels_bm, panels_m, panels_mn,
            ft, deg, spw, scratch; tol=1e-14, max_iter=5000)

        # CG should still converge
        @test diag.max_post_projected < 1e-10

        # Mirrors must be consistent after balance
        max_md = 0.0
        for f in 1:ft.nf
            Int(ft.mirror_panel[f]) == 0 && continue
            for k in 1:Nz
                p2 = Int(ft.face_panel[f])
                d2 = Int(ft.face_dir[f])
                i2 = Int(ft.face_idx_i[f])
                j2 = Int(ft.face_idx_j[f])
                can = d2 == 1 ? panels_am[p2][i2, j2, k] : panels_bm[p2][i2, j2, k]
                mq2 = Int(ft.mirror_panel[f])
                md2 = Int(ft.mirror_dir[f])
                mi2 = Int(ft.mirror_idx_i[f])
                mj2 = Int(ft.mirror_idx_j[f])
                mir = md2 == 1 ? panels_am[mq2][mi2, mj2, k] : panels_bm[mq2][mi2, mj2, k]
                max_md = max(max_md, abs(can - mir))
            end
        end
        @test max_md < 1e-12

        # Per-panel ↔ global divergence must still agree
        max_dd = 0.0
        for k in 1:Nz
            dp = zeros(Float64, ft.nc)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                conv = panels_am[p][i, j, k] - panels_am[p][i + 1, j, k] +
                       panels_bm[p][i, j, k] - panels_bm[p][i, j + 1, k]
                dp[_global_cell(i, j, p, Nc)] = -conv
            end
            dg = zeros(Float64, ft.nc)
            for f in 1:ft.nf
                p2 = Int(ft.face_panel[f])
                d2 = Int(ft.face_dir[f])
                i2 = Int(ft.face_idx_i[f])
                j2 = Int(ft.face_idx_j[f])
                flux = d2 == 1 ? Float64(panels_am[p2][i2, j2, k]) :
                                 Float64(panels_bm[p2][i2, j2, k])
                dg[ft.face_left[f]]  += flux
                dg[ft.face_right[f]] -= flux
            end
            max_dd = max(max_dd, maximum(abs, dp - dg))
        end
        @test max_dd < 1e-12
    end

    @testset "C24 scaling" begin
        Nc = 24
        Nz = 34
        ft = build_cs_global_face_table(Nc, conn)
        @test ft.nf == 12 * Nc^2
        deg = cs_cell_face_degree(ft)
        @test all(deg .== 4)

        am = ntuple(_ -> randn(Float64, Nc + 1, Nc, Nz), 6)
        bm = ntuple(_ -> randn(Float64, Nc, Nc + 1, Nz), 6)
        m  = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)
        mn = ntuple(_ -> abs.(randn(Float64, Nc, Nc, Nz)) .+ 1.0, 6)
        for p in 1:6
            am[p][1, :, :] .= 0; am[p][Nc+1, :, :] .= 0
            bm[p][:, 1, :] .= 0; bm[p][:, Nc+1, :] .= 0
        end

        s = CSPoissonScratch(ft.nc)
        t0 = time()
        d = balance_cs_global_mass_fluxes!(am, bm, m, mn, ft, deg, 4, s;
                                            tol=1e-14, max_iter=5000)
        elapsed = time() - t0

        @test d.max_post_projected < 1e-3 * d.max_pre_residual
        @test d.max_cg_iter < 500
        @test elapsed < 10.0  # should finish in < 1s on any modern machine
        @printf("  C24×34: %.2fs, %d CG iters, proj residual %.2e\n",
                elapsed, d.max_cg_iter, d.max_post_projected)
    end
end

println("\nAll CS global Poisson balance tests passed.")
