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

# ---------------------------------------------------------------------------
# Regression: control.initial = "background" in preconditioned mode must
# set χ = 0 (so the physical starting point is x_b). The previous build
# copied prec.background into the χ slot, which seeded χ = x_b and
# leaked a nonzero 0.5‖χ‖² prior cost on iteration 0.
# ---------------------------------------------------------------------------

@testset "cs_4dvar.jl driver — initial='background' in preconditioned mode → χ=0" begin
    base = read(CONFIG, String)
    # Force a non-zero `preconditioner.background_value` so seeding
    # χ from `prec.background` (the buggy behavior) would produce a
    # detectably nonzero initial 0.5‖χ‖² cost.
    cfg = replace(base,
        "background_value = 0.0" => "background_value = 0.7",
        "initial = \"zeros\""    => "initial = \"background\"")
    mktempdir() do dir
        path = joinpath(dir, "bg.toml")
        write(path, cfg)
        result = run_inversion(path)
        # χ = 0 at iteration 0 ⇒ background term 0.5‖χ‖² = 0. The
        # pre-fix copy of `x_b = 0.7` into the χ slot would have
        # produced 0.5 · 6 · (Nc²) · 0.7² > 0 as the initial bg
        # cost.
        @test result.log isa AT.CSIterationLog
        @test result.log[1].iteration == 0
        @test result.log[1].background_cost == 0.0
        # And the iteration-0 total cost equals the observation
        # cost since the background term is exactly zero.
        @test result.log[1].cost ≈ result.log[1].observation_cost atol = 1e-12
    end
end

# ---------------------------------------------------------------------------
# Regression: [preconditioner].enabled = true with no [covariance]
# section used to silently degrade to unconditioned mode (prec became
# nothing). Now throws a clear ArgumentError.
# ---------------------------------------------------------------------------

@testset "cs_4dvar.jl driver — preconditioner.enabled w/o covariance rejected" begin
    cfg = """
    [mesh]
    Nc = 3
    Hp = 3
    float_type = "Float64"

    [time]
    nsteps = 2
    dt_seconds = 2.0

    [meteo]
    source = "synthetic_constant"

    [[observations.entries]]
    step = 2
    objective = "layer_mean"
    panel = 1
    i = 2
    j = 2
    level = 3
    value = 0.05
    sigma = 0.2

    [control]
    name = "both_steps"
    steps = [1, 2]
    normalize = true
    initial = "zeros"

    [preconditioner]
    enabled = true
    optim_type = "linear"
    background_value = 0.0

    [optimizer]
    kind = "lbfgs"
    iterations = 2
    """
    mktempdir() do dir
        path = joinpath(dir, "no_cov.toml")
        write(path, cfg)
        @test_throws ArgumentError run_inversion(path)
    end
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
