#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection
const TapeMod = AtmosTransport.Tape

# ---------------------------------------------------------------------------
# Plan 26 Phase A.3 — StrideCheckpoint{K} parity vs FullCheckpoint.
#
# Verifies:
#   * `FullCheckpoint()` (explicit) is bit-exact with the no-kwarg default.
#   * `StrideCheckpoint(K)` produces bit-identical footprints to
#     `FullCheckpoint()` on the linear-scheme mass-tape paths, over
#     K = 1, 2, 3 and K > nsteps. Same forward kernels, same reverse
#     records, same order — deterministic by construction on CPU
#     Float64.
#   * `(tape_storage, schedule)` is a clean cross-product: `:device`
#     and `:mmap` agree under both `FullCheckpoint()` and
#     `StrideCheckpoint(K)`.
#   * Stride co-exists with `ImplicitVerticalDiffusion`.
#   * Argument validation: stride with nonlinear PPM / LinRood and
#     `cs_surface_emission_footprint_from_seed` reject with
#     ArgumentError; `StrideCheckpoint(0)` rejects with ArgumentError.
#   * `checkpoint_window_count` / `checkpoint_window_range` math.
# ---------------------------------------------------------------------------

function _stride_problem(; Nc = 4, Nz = 3, nsteps = 6, FT = Float64)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        @inbounds for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p + 0.0001(i + j))
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.013) * sin(FT(0.21step + 0.31p + 0.71i + 0.41j + 0.23k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.011) * cos(FT(0.33step + 0.51p + 0.41i + 0.61j + 0.13k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            @inbounds for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.009) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

# Pick a single column on panel 1, midway into the active mesh so it's a
# stable interior cell. Objective indices are 1:Nc (not haloed).
_column_mean_objective(mesh::AT.CubedSphereMesh) =
    AT.CSColumnMeanObjective(1, max(2, mesh.Nc ÷ 2), max(2, mesh.Nc ÷ 2))

function _footprints_equal(a, b)
    length(a.footprints) == length(b.footprints) || return false
    for step in eachindex(a.footprints)
        for p in 1:6
            a.footprints[step][p] == b.footprints[step][p] || return false
        end
    end
    return true
end

@testset "checkpoint_window_count / checkpoint_window_range" begin
    @test TapeMod.checkpoint_window_count(TapeMod.FullCheckpoint(), 10) == 1
    @test TapeMod.checkpoint_window_range(TapeMod.FullCheckpoint(), 1, 10) == 1:10

    s2 = TapeMod.StrideCheckpoint(2)
    @test TapeMod.checkpoint_window_count(s2, 6) == 3
    @test TapeMod.checkpoint_window_range(s2, 1, 6) == 1:2
    @test TapeMod.checkpoint_window_range(s2, 2, 6) == 3:4
    @test TapeMod.checkpoint_window_range(s2, 3, 6) == 5:6

    s3 = TapeMod.StrideCheckpoint(3)
    @test TapeMod.checkpoint_window_count(s3, 7) == 3
    @test TapeMod.checkpoint_window_range(s3, 3, 7) == 7:7  # tail is short

    s_big = TapeMod.StrideCheckpoint(100)
    @test TapeMod.checkpoint_window_count(s_big, 7) == 1
    @test TapeMod.checkpoint_window_range(s_big, 1, 7) == 1:7

    @test_throws ArgumentError TapeMod.StrideCheckpoint(0)
    @test_throws ArgumentError TapeMod.StrideCheckpoint(-3)
    @test_throws BoundsError TapeMod.checkpoint_window_range(s2, 4, 6)
    @test_throws BoundsError TapeMod.checkpoint_window_range(s2, 0, 6)
end

@testset "StrideCheckpoint vs FullCheckpoint — linear PPM, transport flow" begin
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 3, nsteps = 6, FT = Float64)
    objective = _column_mean_objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    ref = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = scheme, dt = 1.0)

    ref_explicit_full = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = scheme, dt = 1.0,
        checkpoint = AT.FullCheckpoint())
    @test _footprints_equal(ref, ref_explicit_full)

    for K in (1, 2, 3, 5, 6, 12)
        @testset "StrideCheckpoint(K=$K), :device" begin
            stride = AT.cs_surface_emission_footprint(
                panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
                scheme = scheme, dt = 1.0,
                checkpoint = AT.StrideCheckpoint(K))
            @test _footprints_equal(ref, stride)
        end

        @testset "StrideCheckpoint(K=$K), :mmap" begin
            stride = AT.cs_surface_emission_footprint(
                panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
                scheme = scheme, dt = 1.0,
                tape_storage = :mmap,
                checkpoint = AT.StrideCheckpoint(K))
            @test _footprints_equal(ref, stride)
        end
    end
end

@testset "StrideCheckpoint with CMFMC convection" begin
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 5, nsteps = 6, FT = Float64)
    objective = _column_mean_objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    # Build a minimal CMFMC convection forcing inline (same pattern as
    # `_cs_cmfmc_convection_context` in test_cs_ppm_adjoint_footprint.jl).
    FT = eltype(panels_m[1])
    Nc = mesh.Nc
    Nz = size(panels_m[1], 3)
    cmfmc = ntuple(_ -> begin
        c = similar(panels_m[1], FT, Nc, Nc, Nz + 1)
        fill!(c, zero(FT))
        Nz >= 2 && (c[:, :, 2] .= FT(0.012))
        Nz >= 3 && (c[:, :, 3] .= FT(0.020))
        Nz >= 4 && (c[:, :, 4] .= FT(0.015))
        c
    end, 6)
    dtrain = ntuple(_ -> begin
        d = similar(panels_m[1], FT, Nc, Nc, Nz)
        fill!(d, zero(FT))
        Nz >= 2 && (d[:, :, 2] .= FT(0.006))
        Nz >= 3 && (d[:, :, 3] .= FT(0.005))
        d
    end, 6)
    forcing = AT.ConvectionForcing(cmfmc, dtrain, nothing)
    metrics = ntuple(_ -> begin
        a = similar(panels_m[1], FT, Nc, Nc)
        fill!(a, one(FT))
        a
    end, 6)
    ws_conv = AT.CMFMCWorkspace(panels_m; cell_metrics = metrics)
    conv_op = AT.CMFMCConvection()

    ref = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = scheme, dt = 1.0,
        convection_op = conv_op,
        convection_forcing = forcing,
        convection_workspace = ws_conv)

    for K in (2, 3, 6)
        stride = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0,
            convection_op = conv_op,
            convection_forcing = forcing,
            convection_workspace = ws_conv,
            checkpoint = AT.StrideCheckpoint(K))
        @test _footprints_equal(ref, stride)
    end
end

@testset "StrideCheckpoint with ImplicitVerticalDiffusion" begin
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 3, nsteps = 6, FT = Float64)
    objective = _column_mean_objective(mesh)
    scheme = AT.PPMScheme(AT.NoLimiter())

    # Build the diffusion context. Same as test_cs_ppm_adjoint_footprint.jl's
    # `_cs_diffusion_context`, inlined here for self-containment.
    FT = eltype(panels_m[1])
    ws = AT.CSAdvectionWorkspace(mesh, panels_m[1])
    for p in 1:6
        fill!(ws.dz_scratch[p], FT(50.0))
    end
    kz_field = AT.CubedSphereField(ntuple(_ -> AT.ConstantField{FT, 3}(FT(2.0)), 6))
    op = AT.ImplicitVerticalDiffusion(; kz_field)

    ref = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = scheme, dt = 1.0,
        diffusion_op = op, diffusion_workspace = ws)

    for K in (2, 3, 6)
        stride = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0,
            diffusion_op = op, diffusion_workspace = ws,
            checkpoint = AT.StrideCheckpoint(K))
        @test _footprints_equal(ref, stride)
    end
end

@testset "argument validation" begin
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 3, nsteps = 4, FT = Float64)
    objective = _column_mean_objective(mesh)
    nonlinear = AT.PPMScheme(AT.MonotoneLimiter())
    linrood = AT.LinRoodPPMScheme()
    schedule = AT.StrideCheckpoint(2)

    # Nonlinear PPM with stride.
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = nonlinear, dt = 1.0,
        checkpoint = schedule)

    # LinRood with stride.
    @test_throws ArgumentError AT.cs_surface_emission_footprint(
        panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
        scheme = linrood, dt = 1.0,
        checkpoint = schedule)

    # cs_surface_emission_footprint_from_seed with stride.
    FT = eltype(panels_m[1])
    seed = ntuple(p -> begin
        a = similar(panels_rm[p])
        fill!(a, zero(FT))
        a
    end, 6)
    @test_throws ArgumentError AT.cs_surface_emission_footprint_from_seed(
        seed, panels_m, am_steps, bm_steps, cm_steps, mesh;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = 1.0,
        checkpoint = schedule)
end
