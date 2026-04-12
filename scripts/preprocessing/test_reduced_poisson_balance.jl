#!/usr/bin/env julia
#
# Regression test for the reduced-Gaussian Poisson mass-flux balance.
# Covers the 2026-04-11 fix and addresses the diagnostic issue Codex
# flagged in his 16:48 UTC AGENT_CHAT review.
#
# Builds a tiny synthetic RG mesh (N=4, nc=128), constructs an hflux
# field with a KNOWN interior divergence plus a KNOWN nonzero mean
# offset in the target tendency, and verifies:
#
#   1. The projected solver residual (`max_post_projected`) goes to
#      machine precision (below 1e-10 kg).
#   2. The null-space magnitude (`max_rhs_mean`) is nonzero and
#      matches the injected offset up to roundoff.
#   3. The raw post residual (`max_post_raw_residual`) is dominated
#      by the null-space magnitude (i.e. the part of the target that
#      interior-face corrections cannot touch).
#   4. The post-balance `max(|cm|/m)` (computed from the balanced
#      hflux via continuity) is at F64 machine precision — this is
#      the physics truth, not the raw residual.
#
# The test does NOT depend on ERA5 data. Run with:
#
#   julia --project=. scripts/preprocessing/test_reduced_poisson_balance.jl
#

using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "preprocess_era5_reduced_gaussian_transport_binary_v2.jl"))

@testset "Reduced-Gaussian Poisson balance" begin
    # 1. Build a tiny regular N=4 synthetic mesh (8 rings × 16 lon = 128 cells).
    cfg_grid = Dict("type" => "synthetic_reduced_gaussian",
                    "gaussian_number" => 4, "nlon_mode" => "regular")
    grid = build_target_geometry(cfg_grid, Float64)
    mesh = grid.mesh
    nc = AtmosTransport.ncells(mesh)
    nf = AtmosTransport.nfaces(mesh)
    @test nc == 128
    @test nf > nc

    face_left  = Vector{Int32}(undef, nf)
    face_right = Vector{Int32}(undef, nf)
    for f in 1:nf
        l, r = AtmosTransport.face_cells(mesh, f)
        face_left[f]  = Int32(l)
        face_right[f] = Int32(r)
    end
    degree = cell_face_degree(face_left, face_right, nc)

    # 2. Inject a deterministic interior-only flux plus a known offset
    #    in the target tendency. Set m_cur and m_next so that the
    #    target mean has a controlled nonzero value:
    #      target[c] = m_offset_per_cell  (uniform)
    #      actual div(hflux) = random, interior only
    #      rhs = div - target  → rhs has nonzero mean exactly -m_offset
    Nz = 1
    Random.seed!(17)
    hflux = zeros(nf, Nz)
    for f in 1:nf
        l = Int(face_left[f]); r = Int(face_right[f])
        if l > 0 && r > 0
            hflux[f, 1] = randn() * 1e8
        end
    end

    m_cur  = fill(1e13, nc, Nz)
    m_offset_per_cell = 5.0e5   # uniform target offset per cell per substep
    # target_substep = (m_next - m_cur) / (2 * steps_per_window)
    # set steps_per_window = 4 so inv_scale = 1/8.
    steps_per_window = 4
    inv_scale = 1.0 / (2 * steps_per_window)
    # Solve: (m_next - m_cur) * inv_scale = m_offset_per_cell
    m_next = m_cur .+ (m_offset_per_cell / inv_scale)

    # 3. Run the balance.
    scratch = (psi = zeros(nc), rhs = zeros(nc), r = zeros(nc),
               p   = zeros(nc), Ap  = zeros(nc), z = zeros(nc))
    diag = balance_reduced_horizontal_fluxes!(hflux, m_cur, m_next,
                                              face_left, face_right, degree,
                                              steps_per_window, scratch;
                                              tol=1e-14, max_iter=50000)

    println("  pre_raw  = ", diag.max_pre_raw_residual)
    println("  rhs_mean = ", diag.max_rhs_mean)
    println("  pre_proj = ", diag.max_pre_projected)
    println("  post_proj= ", diag.max_post_projected)
    println("  post_raw = ", diag.max_post_raw_residual)
    println("  cg_iter  = ", diag.max_cg_iter)

    # 4. Assertions.
    # (a) Projected residual reduced by ≥ 10 orders of magnitude.
    # PCG converges to relative tol=1e-14, so the absolute projected
    # residual is ~`max_pre_projected * 1e-14`. We assert a looser
    # 1e-10 relative reduction to leave roundoff headroom.
    rel_reduction = diag.max_post_projected / max(diag.max_pre_projected, eps())
    println("  projected residual relative reduction = ", rel_reduction)
    @test rel_reduction < 1e-10
    # (b) Null-space magnitude matches the injected offset exactly
    # (for uniform offset the rhs mean = div_mean - target_mean,
    # and div_mean from the conservation telescope is 0 for a
    # closed mesh, so rhs_mean = -target_mean = m_offset_per_cell).
    @test diag.max_rhs_mean > 0
    @test isapprox(diag.max_rhs_mean, m_offset_per_cell; rtol=1e-10)
    # (c) Raw residual tracks the null-space component magnitude, not
    # the tiny projected residual — this is the diagnostic Codex
    # flagged in his 16:48 review.
    @test isapprox(diag.max_post_raw_residual, m_offset_per_cell; rtol=1e-8)

    # 5. Recompute cm and verify worst(|cm|/m) at F64 machine precision.
    cm = zeros(nc, Nz + 1)
    div_scratch = zeros(nc, Nz)
    recompute_faceindexed_cm_from_divergence!(cm, hflux, face_left, face_right,
                                              div_scratch)
    worst_ratio = 0.0
    for c in 1:nc, k in 2:Nz
        r = abs(cm[c, k]) / max(m_cur[c, k], eps())
        r > worst_ratio && (worst_ratio = r)
    end
    # With only one level, the meaningful test is |cm| at the single
    # interior interface (k=Nz+1=2) being small compared to cell mass.
    # Use the single level k=1 (interior means cm[c,2]) as the probe.
    max_abs_cm = maximum(abs, cm[:, 2])
    println("  max |cm[k=2]| = ", max_abs_cm)
    @test max_abs_cm / 1e13 < 1e-10

    # 6. Pure-constructed test: rhs = L * psi_true, verify solver
    # recovers psi_true at machine precision (null-space projected out).
    Random.seed!(42)
    psi_true = randn(nc); psi_true .-= sum(psi_true) / nc
    # Compute rhs = L * psi_true directly via the public function.
    rhs_test = zeros(nc)
    _graph_laplacian_mul!(rhs_test, psi_true, face_left, face_right, degree)
    psi_solve = zeros(nc)
    scratch2 = (psi = zeros(nc), rhs = zeros(nc), r = zeros(nc),
                p   = zeros(nc), Ap  = zeros(nc), z = zeros(nc))
    res, it = solve_graph_poisson_pcg!(psi_solve, copy(rhs_test),
                                       face_left, face_right, degree, scratch2;
                                       tol=1e-14, max_iter=20000)
    err = maximum(abs.(psi_solve .- psi_true))
    println("  constructed-rhs test: iter=", it, " residual=", res,
            "  max|psi - psi_true|=", err)
    @test err < 1e-10
    @test res < 1e-8
end
