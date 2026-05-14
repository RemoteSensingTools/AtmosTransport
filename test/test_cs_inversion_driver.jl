#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C4 — end-to-end inversion driver smoke test.
#
# Loads `config/inversions/example_synthetic.toml`, runs the
# driver's `run_inversion`, and verifies the result. This is the
# Plan 26 Phase C "wraps" gate — every B + C component (covariance,
# preconditioner, optimizer, iteration log, preconditioned 4D-Var
# entrypoint) participates.
# ---------------------------------------------------------------------------

using Test

# The driver is a script, not a module — include it so `run_inversion`
# is exposed at the top level.
include(joinpath(@__DIR__, "..", "scripts", "inversions", "cs_4dvar.jl"))

const CONFIG = joinpath(@__DIR__, "..", "config", "inversions",
                         "example_synthetic.toml")

@testset "cs_4dvar.jl driver — synthetic smoke test" begin
    @test isfile(CONFIG)

    result = run_inversion(CONFIG)
    @test result isa AT.CS4DVarSolveResult

    # The synthetic problem has zero air mass flux (constant flow),
    # so the observation residual is whatever the initial mass field
    # produces minus the configured observation value. The optimizer
    # should reduce the cost from initial.
    initial_cost = result.cost_history[1]
    @test result.last.cost <= initial_cost
    @test result.last.cost < initial_cost  # strict — the example is
                                            # not at a local min at χ=0.

    # Cost-decomposition invariant: total cost == obs + background.
    @test result.last.cost ≈ result.last.observation_cost +
                              result.last.background_cost atol = 1e-9
    # Preconditioned mode: at χ = 0 the background term is exactly 0
    # (0.5 ‖0‖² = 0). The initial cost recorded in cost_history[1]
    # therefore equals the initial observation cost.
    @test result.cost_history[1] >= 0

    # Iteration log was requested via `log = true` in the TOML.
    @test result.log isa AT.CSIterationLog
    @test !isempty(result.log)
    @test result.log[1].iteration == 0

    # Cost-decomposition invariant on every log row.
    for entry in result.log
        @test entry.cost ≈ entry.observation_cost + entry.background_cost atol = 1e-9
    end

    # L-BFGS produces a monotone non-increasing cost trajectory.
    @test issorted(result.cost_history, rev = true) ||
          all(diff(result.cost_history) .<= 1e-12)
end

@testset "cs_4dvar.jl driver — config dispatch coverage" begin
    # Each TOML knob the driver dispatches on must produce a working
    # solve. We override the default config in-place via a tempdir
    # copy.
    base = read(CONFIG, String)

    function _run_with_override(override::AbstractString)
        mktempdir() do dir
            path = joinpath(dir, "config.toml")
            write(path, override)
            return run_inversion(path)
        end
    end

    # Switch optimizer to gradient descent.
    gd_cfg = replace(base,
        "kind = \"lbfgs\"" => "kind = \"gradient_descent\"\ninitial_step = 0.25")
    result_gd = _run_with_override(gd_cfg)
    @test result_gd isa AT.CS4DVarSolveResult

    # Switch covariance to diagonal.
    diag_cfg = replace(base,
        "kind = \"isotropic_gaussian\"" => "kind = \"diagonal\"")
    result_diag = _run_with_override(diag_cfg)
    @test result_diag isa AT.CS4DVarSolveResult

    # Unconditioned mode (preconditioner.enabled = false).
    uncond_cfg = replace(base,
        "enabled = true" => "enabled = false")
    result_uncond = _run_with_override(uncond_cfg)
    @test result_uncond isa AT.CS4DVarSolveResult
end

@testset "cs_4dvar.jl driver — rejects bad TOML" begin
    # Missing required fields, unknown enum values, etc.
    base = read(CONFIG, String)

    function _expect_throw(override::AbstractString)
        mktempdir() do dir
            path = joinpath(dir, "bad.toml")
            write(path, override)
            @test_throws Exception run_inversion(path)
        end
    end

    _expect_throw(replace(base,
        "kind = \"isotropic_gaussian\"" => "kind = \"mystery_kernel\""))
    _expect_throw(replace(base,
        "kind = \"lbfgs\"" => "kind = \"newton_raphson\""))
    _expect_throw(replace(base,
        "objective = \"layer_mean\"" => "objective = \"point_sample\""))
    _expect_throw(replace(base,
        "optim_type = \"linear\"" => "optim_type = \"sigmoid\""))

    # Missing config file.
    @test_throws ArgumentError run_inversion(joinpath(tempdir(),
        "does_not_exist_$(rand(UInt64)).toml"))
end
