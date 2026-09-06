#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# TM5 4DVAR adjoint test in preconditioned space (Eqs. 2.6-2.7 of
# the TM5 4DVAR adjoint-and-gradient-test memo).
#
# Identity under test (per pair of random χ₁, χ₂ and a random
# departure d, with L AND H linear in their arguments):
#
#     dᵀ (y₁ − y₂) == (χ₁ − χ₂)ᵀ Lᵀ Hᵀ d        where  x_i = L χ_i + x_b
#                                                       y_i = H x_i
#
# Requirements: (a) `H` is linear in x — so we restrict to
# `PPMScheme(NoLimiter())`, the same gate as the model-space test;
# (b) `L` is linear in χ — so we use `LinearOptimType`. The
# log-normal preconditioner is non-linear in χ; the PDF explicitly
# notes the identity does not hold for non-linear preconditioners,
# even if their tangent and adjoint are coded correctly. We exercise
# that as a negative control at the bottom of this file.
#
# The trick to extract `Hᵀ d`: same σ=1 / synthetic-obs path as in
# `test_adjoint_identity_model_space.jl`. We compute `Hᵀ d` per
# control in physical space with `preconditioner = nothing`, then
# apply `Lᵀ` per control via `apply_preconditioner_adjoint!`. We do
# NOT route the gradient through `cs_surface_flux_4dvar`'s
# preconditioned mode because that adds the `+ χ` background-term
# contribution; we want pure `Lᵀ Hᵀ d`.
# ---------------------------------------------------------------------------

using Test
using Random
using LinearAlgebra

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

function _transport_cs_problem(; Nc=3, Nz=3, nsteps=3, FT=Float64)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

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

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.015) * sin(FT(0.2step + 0.3p + 0.7i + 0.4j + 0.2k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.012) * cos(FT(0.3step + 0.5p + 0.4i + 0.6j + 0.1k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.010) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

# A control per step. Each control holds the χ-space vector for that
# step — `value` is sized `(Nc, Nc)` per panel.
function _per_step_chi_controls(rng, mesh, nsteps; scale = 0.5)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                ntuple(p -> scale .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6))
            for k in 1:nsteps]
end

# Apply the preconditioner per-control to get physical-space x = L χ + x_b.
# Returns a Vector{CSSurfaceFluxControl} ready to feed the
# unconditioned `cs_surface_flux_4dvar` path.
function _physical_from_chi(prec, chi_controls)
    out = Vector{AT.CSSurfaceFluxControl}(undef, length(chi_controls))
    for k in eachindex(chi_controls)
        x_val = ntuple(p -> similar(chi_controls[k].value[p]), 6)
        AT.apply_preconditioner!(x_val, prec, chi_controls[k].value)
        out[k] = AT.CSSurfaceFluxControl(chi_controls[k].window, x_val)
    end
    return out
end

function _build_observations(nsteps)
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.0, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSColumnMeanObjective(4, 2, 1), 0.0, 1.0),
    ]
end

function _run_4dvar(mesh, panels_rm0, panels_m0,
                    panels_am, panels_bm, panels_cm,
                    observations, controls;
                    dt = 2.0)
    return AT.cs_surface_flux_4dvar(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = dt,
        preconditioner = nothing)
end

# `Hᵀ d` per (physical) control at linearization `x_ref` — same σ=1
# trick as in the model-space test.
function _extract_Hadjoint_d(mesh, panels_rm0, panels_m0,
                             panels_am, panels_bm, panels_cm,
                             observations, x_ref, d; dt = 2.0)
    y_ref = _run_4dvar(mesh, panels_rm0, panels_m0,
                       panels_am, panels_bm, panels_cm,
                       observations, x_ref; dt = dt).simulated
    obs_d = [AT.CSObservation(observations[k].step, observations[k].objective,
                              y_ref[k] - d[k], 1.0)
             for k in eachindex(observations)]
    result_d = _run_4dvar(mesh, panels_rm0, panels_m0,
                          panels_am, panels_bm, panels_cm,
                          obs_d, x_ref; dt = dt)
    @test all(isapprox.(result_d.residuals, d; atol = 1e-12, rtol = 1e-12))
    return result_d.gradients
end

# η[c] = Lᵀ_c · f_phys[c]. For `LinearOptimType` the base point
# `x_lin` is ignored by `apply_preconditioner_adjoint!`; we pass it
# anyway to keep the call signature uniform.
function _apply_Ltranspose(prec, x_lin_controls, f_phys)
    out = Vector{NTuple{6, Matrix{Float64}}}(undef, length(f_phys))
    for c in eachindex(f_phys)
        eta = ntuple(p -> similar(f_phys[c][p]), 6)
        AT.apply_preconditioner_adjoint!(eta, prec, x_lin_controls[c].value, f_phys[c])
        out[c] = eta
    end
    return out
end

# <χ_1 - χ_2, η> over per-control panels.
function _chi_inner(chi_a, chi_b, eta)
    total = 0.0
    @inbounds for c in eachindex(chi_a)
        for p in 1:6
            total += sum((chi_a[c].value[p] .- chi_b[c].value[p]) .* eta[c][p])
        end
    end
    return total
end

# Build a Linear preconditioner of the requested covariance kind.
# Background `x_b = 1.0` on every panel — non-zero so the L χ + x_b
# composition is exercised.
function _build_linear_preconditioner(mesh; cov_kind::Symbol)
    Nc = mesh.Nc
    sigma = ntuple(_ -> fill(0.3, Nc, Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    x_b = ntuple(_ -> fill(1.0, Nc, Nc), 6)
    return AT.CSSurfaceFluxPreconditioner(cov, x_b, AT.LinearOptimType()), x_b
end

# Run a single identity trial and return (lhs, rhs) so the assertion
# can be made by the caller (either as a passing check for linear L
# or as a negative-control divergence check for non-linear L).
function _run_identity_trial(mesh, panels_rm, panels_m,
                              panels_am, panels_bm, panels_cm,
                              observations, prec, rng; dt = 2.0)
    chi_1 = _per_step_chi_controls(rng, mesh, 3)
    chi_2 = _per_step_chi_controls(rng, mesh, 3)
    d     = rand(rng, Float64, length(observations)) .- 0.5

    x_1 = _physical_from_chi(prec, chi_1)
    x_2 = _physical_from_chi(prec, chi_2)

    # Linearization point for Hᵀ and Lᵀ. For linear (H, L) it does
    # not matter; we pick `x_1` for concreteness so the LogNormal
    # negative control evaluates Lᵀ at a non-zero point.
    f_phys = _extract_Hadjoint_d(mesh, panels_rm, panels_m,
                                  panels_am, panels_bm, panels_cm,
                                  observations, x_1, d; dt = dt)
    eta = _apply_Ltranspose(prec, x_1, f_phys)

    y_1 = _run_4dvar(mesh, panels_rm, panels_m,
                     panels_am, panels_bm, panels_cm,
                     observations, x_1; dt = dt).simulated
    y_2 = _run_4dvar(mesh, panels_rm, panels_m,
                     panels_am, panels_bm, panels_cm,
                     observations, x_2; dt = dt).simulated

    lhs = dot(d, y_1 .- y_2)
    rhs = _chi_inner(chi_1, chi_2, eta)
    return lhs, rhs
end

@testset "Adjoint identity in preconditioned space (memo Eqs. 2.6-2.7)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations(3)

    @testset "Linear L × $(cov_kind) covariance" for cov_kind in (:diagonal, :gaussian)
        prec, _x_b = _build_linear_preconditioner(mesh; cov_kind = cov_kind)
        @testset "Random trial $trial" for trial in 1:3
            lhs, rhs = _run_identity_trial(mesh, panels_rm, panels_m,
                                            panels_am, panels_bm, panels_cm,
                                            observations, prec, rng; dt = dt)
            scale = max(abs(lhs), abs(rhs), 1e-30)
            @test abs(lhs - rhs) / scale < 1e-12
            @test isfinite(lhs) && isfinite(rhs)
        end
    end

    # ---------------------------------------------------------------------
    # Negative control: LogNormalOptimType.
    #
    # The PDF explicitly states that the preconditioned-space identity
    # only holds for LINEAR L. The log-normal change-of-variables
    # x = x_b ⊙ exp(B^(1/2) χ) is non-linear in χ, so the algebraic
    # identity must FAIL at the same tolerance — even though
    # `apply_preconditioner_tangent!` / `apply_preconditioner_adjoint!`
    # are individually correct (their per-pair consistency is checked
    # by `test_cs_preconditioning.jl`). This guard prevents someone
    # from accidentally extending the linear test to non-linear L and
    # interpreting a "small" violation as success.
    # ---------------------------------------------------------------------
    @testset "LogNormal L — algebraic identity must fail" begin
        Nc = mesh.Nc
        sigma = ntuple(_ -> fill(0.2, Nc, Nc), 6)
        cov = AT.DiagonalCSCovariance(sigma)
        x_b = ntuple(_ -> fill(1.0, Nc, Nc), 6)
        prec = AT.CSSurfaceFluxPreconditioner(cov, x_b, AT.LogNormalOptimType())

        lhs, rhs = _run_identity_trial(mesh, panels_rm, panels_m,
                                        panels_am, panels_bm, panels_cm,
                                        observations, prec, rng; dt = dt)
        scale = max(abs(lhs), abs(rhs), 1e-30)
        # We expect a non-zero residual; the threshold is well above the
        # linear-case tolerance to avoid a flaky test if LogNormal
        # happens to be near-linear for a particular random draw.
        @test abs(lhs - rhs) / scale > 1e-6
        @test isfinite(lhs) && isfinite(rhs)
    end
end
