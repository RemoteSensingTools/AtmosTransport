#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C3 — `CSIterationLog` per-iteration diagnostic capture.
#
# Coverage:
#   * Default behavior: `log = false` keeps `solve.log === nothing`
#     (pre-C3 baseline).
#   * `log = true`: `solve.log isa CSIterationLog` and contains one
#     `CSIterationLogEntry` per iteration (plus the iteration-0
#     initial-probe row).
#   * Consistency: log entries' `cost` field matches `cost_history`;
#     `gradient_norm` matches `gradient_norm_history`; `step_size`
#     matches `step_history` (GD only — L-BFGS leaves step at 0).
#   * Cost decomposition: `observation_cost + background_cost == cost`
#     bit-exact (the cost-decomposition invariant).
#   * Wall-clock monotonicity: `elapsed_seconds` is monotonically
#     non-decreasing across iterations.
#   * Coverage spans both backends: `CSGradientDescent` and
#     `CSLBFGS`.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

function _constant_cs_problem(; Nc=3, Nz=3, nsteps=2, FT=Float64)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p)
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir=0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir=0)
    panels_am = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
    panels_bm = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
    panels_cm = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    return mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm
end

function _scenario()
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
    dt = 2.0
    zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
        zero_panel)
    observations = [
        AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.05, 0.2),
    ]
    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt)
    end
    return (control = control, cost_fn = cost_fn)
end

# ---------------------------------------------------------------------------
# Default behavior: log = false ⇒ solve.log === nothing
# ---------------------------------------------------------------------------

@testset "CS4DVarSolveResult — log defaults to nothing" begin
    s = _scenario()
    opt = AT.CSGradientDescent(iterations = 3, initial_step = 0.25)
    solve = AT.cs_surface_flux_4dvar_solve(opt, s.cost_fn, s.control)
    @test solve.log === nothing

    opt_lbfgs = AT.CSLBFGS(iterations = 3)
    solve_lbfgs = AT.cs_surface_flux_4dvar_solve(opt_lbfgs, s.cost_fn, s.control)
    @test solve_lbfgs.log === nothing
end

# ---------------------------------------------------------------------------
# GD with log enabled
# ---------------------------------------------------------------------------

@testset "CSGradientDescent — log captures every iteration" begin
    s = _scenario()
    opt = AT.CSGradientDescent(iterations = 4, initial_step = 0.25,
                                log = true)
    solve = AT.cs_surface_flux_4dvar_solve(opt, s.cost_fn, s.control)

    @test solve.log isa AT.CSIterationLog
    log = solve.log

    # Log row 1 is the iteration-0 initial probe; subsequent rows
    # correspond to accepted descent steps.
    @test length(log) == length(solve.cost_history)
    @test length(log) == 1 + length(solve.step_history)
    @test log[1].iteration == 0
    @test log[end].iteration == solve.iterations

    # Consistency with the existing history vectors.
    for (k, entry) in enumerate(log)
        @test entry.cost ≈ solve.cost_history[k] atol = 1e-12
        @test entry.gradient_norm ≈ solve.gradient_norm_history[k] atol = 1e-12
    end

    # Step size: entry 1 is the initial probe (0); subsequent entries
    # carry the accepted line-search step.
    @test log[1].step_size == 0.0
    for k in 2:length(log)
        @test log[k].step_size == solve.step_history[k - 1]
    end

    # Cost-decomposition invariant.
    for entry in log
        @test entry.cost ≈ entry.observation_cost + entry.background_cost atol = 1e-10
    end

    # Wall-clock monotonicity.
    @test issorted([entry.elapsed_seconds for entry in log])
end

# ---------------------------------------------------------------------------
# LBFGS with log enabled
# ---------------------------------------------------------------------------

@testset "CSLBFGS — log captures every iteration" begin
    s = _scenario()
    opt = AT.CSLBFGS(iterations = 6, gradient_tolerance = 1e-12, log = true)
    solve = AT.cs_surface_flux_4dvar_solve(opt, s.cost_fn, s.control)

    @test solve.log isa AT.CSIterationLog
    log = solve.log

    # Iteration-0 initial probe row is present.
    @test !isempty(log)
    @test log[1].iteration == 0

    # All log iterations are non-negative and monotone.
    iters = [entry.iteration for entry in log]
    @test issorted(iters)
    @test all(iters .>= 0)

    # Cost-decomposition invariant — Optim's callback fires after
    # `g!` is called at the new iterate, so the cached cost result
    # is at the same iterate as `state.value`.
    for entry in log
        @test entry.cost ≈ entry.observation_cost + entry.background_cost atol = 1e-10
    end

    # Wall-clock monotonicity.
    @test issorted([entry.elapsed_seconds for entry in log])

    # Step size is always 0 for L-BFGS — Optim's line-search step
    # is not exposed in the trace.
    @test all(entry.step_size == 0.0 for entry in log)
end

# ---------------------------------------------------------------------------
# Public-entrypoint integration: log flag flows through.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public-entrypoint integration: the log flag carried by the optimizer
# survives the legacy-kwargs vs explicit-optimizer split in
# `cs_surface_flux_4dvar_optimize`.
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar_optimize — log via explicit optimizer" begin
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
    dt = 2.0
    zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
        zero_panel)
    observations = [
        AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.05, 0.2),
    ]

    solve = AT.cs_surface_flux_4dvar_optimize(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, observations, control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt,
        optimizer = AT.CSGradientDescent(iterations = 2,
                                          initial_step = 0.25,
                                          log = true))
    @test solve.log isa AT.CSIterationLog
    @test length(solve.log) == length(solve.cost_history)
end
