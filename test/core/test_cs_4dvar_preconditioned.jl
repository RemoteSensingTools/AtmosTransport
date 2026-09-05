#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.B3 — preconditioned 4D-Var cost + gradient.
#
# Wires the B2 `CSSurfaceFluxPreconditioner` into `cs_surface_flux_4dvar`
# via the `preconditioner` kwarg. The cost becomes
#     J(χ) = 0.5 ‖χ‖²  +  J_obs(T(χ)),
# the reported gradient is
#     ∇_χ J = χ  +  T'(χ)^T ∇_x J_obs.
#
# Coverage:
#   * Sanity at χ = 0: physical-side `T(0) = x_b`, observation cost
#     matches the unconditioned cost with controls = x_b, background
#     cost is 0 (since ‖χ‖ = 0), gradient is `T'(0)^T ∇_x J_obs`.
#   * Background term equals `0.5 ‖χ‖²` for any χ (Linear + LogNormal
#     × Diagonal + Gaussian).
#   * FD-identity: central-difference of `J(χ ± ε δχ)` matches
#     `⟨∇_χ J(χ), δχ⟩` for both optim types and both covariance types
#     — the B3 acceptance gate.
#   * Argument validation: length mismatch between controls and
#     preconditioners is rejected.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Constant-flow CS test problem, copied from `test_cs_ppm_adjoint_footprint.jl`
# to keep this test file self-contained (each test file runs in an
# isolated anonymous module per `runtests.jl`).
function _constant_cs_problem(; Nc=4, Nz=3, nsteps=2, FT=Float64)
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

    panels_am_steps = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
    panels_bm_steps = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
    panels_cm_steps = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

function _dot_control_gradient(gradients, directions)
    total = 0.0
    for c in eachindex(gradients), p in 1:6
        total += sum(gradients[c][p] .* directions[c][p])
    end
    return total
end

function _shift_control(control, direction, scale)
    value = ntuple(p -> control.value[p] .+ scale .* direction[p], 6)
    return AT.CSSurfaceFluxControl(control.window, value)
end

# Two distinct controls + two observations + reasonable σ + small Gaussian
# correlation length so the spectral filter is well-conditioned at Nc = 3.
function _preconditioned_scenario(; optim_type, cov_kind = :gaussian, Nc = 3)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(; Nc = Nc, Nz = 3, nsteps = 3)
    dt = 2.0

    # χ = 0 baseline. x_b = 1.0 everywhere (positive — valid for both
    # optim types).
    chi_zero = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    x_b = ntuple(_ -> fill(1.0, mesh.Nc, mesh.Nc), 6)

    sigma = ntuple(_ -> fill(0.3, mesh.Nc, mesh.Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    prec_a = AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)
    prec_b = AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)

    controls = [
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:first_step, 1), chi_zero),
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:late_window, 2:3; normalize=true),
            chi_zero),
    ]
    preconditioners = [prec_a, prec_b]

    observations = [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.03, 0.4),
        AT.CSObservation(3, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3),
    ]

    return (mesh = mesh,
            panels_rm = panels_rm, panels_m = panels_m,
            panels_am = panels_am, panels_bm = panels_bm, panels_cm = panels_cm,
            dt = dt,
            x_b = x_b,
            controls = controls,
            preconditioners = preconditioners,
            observations = observations,
            optim_type = optim_type,
            cov = cov)
end

# Convenience: call cs_surface_flux_4dvar with a custom chi.
function _eval_at_chi(s, chi_panels_per_control)
    new_controls = [
        AT.CSSurfaceFluxControl(s.controls[i].window, chi_panels_per_control[i])
        for i in eachindex(s.controls)
    ]
    return AT.cs_surface_flux_4dvar(
        s.panels_rm, s.panels_m,
        s.panels_am, s.panels_bm, s.panels_cm,
        s.mesh, s.observations, new_controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = s.dt,
        preconditioner = s.preconditioners)
end

# ---------------------------------------------------------------------------
# Sanity at χ = 0 — background cost is exactly 0; physical-side cost
# matches the unconditioned path called with `controls = x_b`.
# ---------------------------------------------------------------------------

@testset "preconditioned 4D-Var — sanity at χ = 0" begin
    for optim_type in (AT.LinearOptimType(), AT.LogNormalOptimType()),
        cov_kind in (:diagonal, :gaussian)

        s = _preconditioned_scenario(; optim_type = optim_type,
                                     cov_kind = cov_kind, Nc = 3)
        result = AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m,
            s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, s.controls;
            scheme = AT.PPMScheme(AT.NoLimiter()),
            dt = s.dt,
            preconditioner = s.preconditioners)

        # Background term comes from 0.5 ‖χ‖² = 0 when χ = 0.
        @test result.background_cost == 0.0

        # Total = observation cost.
        @test result.cost ≈ result.observation_cost atol = 1e-12

        # Compare against unconditioned-mode evaluation at x_b.
        x_b_controls = [
            AT.CSSurfaceFluxControl(s.controls[i].window, s.x_b)
            for i in eachindex(s.controls)
        ]
        baseline = AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m,
            s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, x_b_controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt)

        @test result.observation_cost ≈ baseline.observation_cost atol = 1e-10
        @test result.residuals ≈ baseline.residuals atol = 1e-10

        # Reported controls are the original χ inputs (not the
        # internal physical_controls).
        @test result.controls[1].value === s.controls[1].value
        @test result.controls[2].value === s.controls[2].value
    end
end

# ---------------------------------------------------------------------------
# 0.5 ‖χ‖² literal — non-zero χ produces the expected background cost.
# ---------------------------------------------------------------------------

@testset "preconditioned 4D-Var — background_cost = 0.5 ‖χ‖²" begin
    optim_type = AT.LinearOptimType()
    s = _preconditioned_scenario(; optim_type = optim_type,
                                 cov_kind = :diagonal, Nc = 3)

    # Deterministic non-zero χ.
    chi = [
        ntuple(p -> [0.05 * sin(0.1p + 0.2i - 0.3j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [0.04 * cos(0.2p + 0.3i + 0.1j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]
    expected_bg = 0.5 * sum(sum(chi[c][p] .^ 2) for c in 1:2, p in 1:6)

    result = _eval_at_chi(s, chi)
    @test result.background_cost ≈ expected_bg atol = 1e-12
end

# ---------------------------------------------------------------------------
# Finite-difference identity — the B3 acceptance gate.
# ---------------------------------------------------------------------------

function _fd_identity_check(s; chi_scale = 0.02, eps_dir = 1e-6,
                            rtol = 5e-4, atol = 1e-8)
    # Non-zero base χ so we exercise the full chain rule (LogNormal's
    # tangent depends on x = T(χ)).
    chi_base = [
        ntuple(p -> [chi_scale * sin(0.11p + 0.17i - 0.13j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [chi_scale * cos(0.07p + 0.19i + 0.23j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]
    directions = [
        ntuple(p -> [sin(0.13p + 0.29i - 0.41j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [cos(0.31p + 0.37i + 0.43j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]

    base = _eval_at_chi(s, chi_base)
    plus = _eval_at_chi(s, [
        ntuple(p -> chi_base[c][p] .+ eps_dir .* directions[c][p], 6)
        for c in 1:2
    ])
    minus = _eval_at_chi(s, [
        ntuple(p -> chi_base[c][p] .- eps_dir .* directions[c][p], 6)
        for c in 1:2
    ])

    fd = (plus.cost - minus.cost) / (2 * eps_dir)
    predicted = _dot_control_gradient(base.gradients, directions)
    @test isapprox(predicted, fd; rtol = rtol, atol = atol)
end

@testset "preconditioned 4D-Var — FD identity, Linear × Diagonal" begin
    s = _preconditioned_scenario(; optim_type = AT.LinearOptimType(),
                                 cov_kind = :diagonal, Nc = 3)
    _fd_identity_check(s)
end

@testset "preconditioned 4D-Var — FD identity, Linear × Gaussian" begin
    s = _preconditioned_scenario(; optim_type = AT.LinearOptimType(),
                                 cov_kind = :gaussian, Nc = 3)
    _fd_identity_check(s)
end

@testset "preconditioned 4D-Var — FD identity, LogNormal × Diagonal" begin
    s = _preconditioned_scenario(; optim_type = AT.LogNormalOptimType(),
                                 cov_kind = :diagonal, Nc = 3)
    _fd_identity_check(s)
end

@testset "preconditioned 4D-Var — FD identity, LogNormal × Gaussian" begin
    s = _preconditioned_scenario(; optim_type = AT.LogNormalOptimType(),
                                 cov_kind = :gaussian, Nc = 3)
    _fd_identity_check(s)
end

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@testset "preconditioned 4D-Var — preconditioner / control length mismatch" begin
    s = _preconditioned_scenario(; optim_type = AT.LinearOptimType(),
                                 cov_kind = :diagonal, Nc = 3)
    # Mismatched lengths that are neither 1 nor `length(controls)` are
    # rejected. `[prec_a]` (length 1 with 2 controls) is now the
    # documented broadcast case — see the optimizer-dispatch suite —
    # so the failing case here is a length-3 vector against 2 controls.
    @test_throws ArgumentError AT.cs_surface_flux_4dvar(
        s.panels_rm, s.panels_m,
        s.panels_am, s.panels_bm, s.panels_cm,
        s.mesh, s.observations, s.controls;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt,
        preconditioner = [s.preconditioners[1],
                           s.preconditioners[1],
                           s.preconditioners[1]])
end
