#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# TM5 4DVAR adjoint test in model/flux space (Eqs. 2.4-2.5 of the
# TM5 4DVAR adjoint-and-gradient-test memo).
#
# Identity under test (per pair of random flux vectors x1, x2 and a
# random departure vector d, with H linear in x):
#
#     dᵀ (y₁ − y₂) == (x₁ − x₂)ᵀ Hᵀ d            where  y_i = H x_i
#
# This is an exact algebraic identity to ~1e-12 double-precision, so
# we restrict the scheme here to `PPMScheme(NoLimiter())`. The
# LinRood scheme is non-linear in the emissions through its donor-cell
# denominator and the `c = rm/m` chain rule and is therefore covered
# by the local FD tangent-adjoint checks in
# `test_linrood_adjoint_integration.jl` instead.
#
# How `Hᵀ d` is extracted from the production cost path:
# With `preconditioner = nothing`, σ = 1 on every observation, and no
# `.background` / `.sigma` on the controls, `cs_surface_flux_4dvar`
# computes the gradient
#     ∇_x J = Σ_k (residual_k / σ_k²) · footprint_k = Σ_k residual_k · footprint_k
# where residual_k = simulated_k − y_obs_k. Setting y_obs_k = simulated_k(x_ref) − d_k
# makes the residual exactly `d_k`, so the returned `.gradients` are
# the per-control window-aggregated `Hᵀ d`.
# ---------------------------------------------------------------------------

using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Non-zero-flow CS test problem, copied from
# `test_cs_ppm_adjoint_footprint.jl` because each test file in
# `runtests.jl` runs in its own anonymous module — `include`-ing that
# file here would rerun its testsets.
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

# Per-step controls, one CSSurfaceFluxControl per step over a single
# step (weight = 1). This is the cleanest x → emission_rates mapping:
# emission_rates[step] = controls[step].value.
function _per_step_controls(rng, mesh, nsteps; scale = 0.01)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                ntuple(p -> scale .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6))
            for k in 1:nsteps]
end

# Random departure vector in observation space.
function _random_departures(rng, K)
    return rand(rng, Float64, K) .- 0.5
end

# Single observation set sampled across panels, ij, steps, and
# objective kinds. σ = 1 on every observation so residuals pass
# through unattenuated.
function _build_observations(nsteps)
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.0, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSColumnMeanObjective(4, 2, 1), 0.0, 1.0),
    ]
end

# Run the production cost/gradient path with σ=1, no background, and
# arbitrary obs values. We use `.simulated` for y values and
# `.gradients` for `Hᵀ d` (after rewriting obs values to make
# residuals equal `d` — see `_extract_Hadjoint_d`).
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

# Extract `Hᵀ d` per control at linearization point `x_ref` (a
# Vector{CSSurfaceFluxControl}) and departure `d`. The trick: with
# σ = 1 and `obs.value = simulated_k(x_ref) − d_k`, the production
# residual equals `d_k` and the production gradient equals `Hᵀ d`.
function _extract_Hadjoint_d(mesh, panels_rm0, panels_m0,
                             panels_am, panels_bm, panels_cm,
                             observations, x_ref, d; dt = 2.0)
    # Step 1 — forward at x_ref to learn simulated values.
    y_ref = _run_4dvar(mesh, panels_rm0, panels_m0,
                      panels_am, panels_bm, panels_cm,
                      observations, x_ref; dt = dt).simulated
    # Step 2 — synthesize observations whose residual at x_ref equals d.
    obs_d = [AT.CSObservation(observations[k].step, observations[k].objective,
                              y_ref[k] - d[k], 1.0)
             for k in eachindex(observations)]
    # Step 3 — read back gradients = Hᵀ d.
    result_d = _run_4dvar(mesh, panels_rm0, panels_m0,
                          panels_am, panels_bm, panels_cm,
                          obs_d, x_ref; dt = dt)
    # Sanity: residuals match `d` to roundoff (round-trip y_ref - obs_d).
    @test all(isapprox.(result_d.residuals, d; atol = 1e-12, rtol = 1e-12))
    return result_d.gradients, y_ref
end

# ⟨x1 - x2, f⟩ over per-step controls (Vector{CSSurfaceFluxControl}).
function _flux_inner(controls_a, controls_b, f_per_control)
    total = 0.0
    @inbounds for c in eachindex(controls_a)
        for p in 1:6
            total += sum((controls_a[c].value[p] .- controls_b[c].value[p]) .*
                         f_per_control[c][p])
        end
    end
    return total
end

@testset "Adjoint identity in model space (memo Eqs. 2.4-2.5)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations(3)
    K = length(observations)

    @testset "Degenerate case: x1 == x2 ⇒ both sides zero" begin
        x_ref = _per_step_controls(rng, mesh, 3)
        d = _random_departures(rng, K)
        f, _ = _extract_Hadjoint_d(mesh, panels_rm, panels_m,
                                    panels_am, panels_bm, panels_cm,
                                    observations, x_ref, d; dt = dt)
        # x1 = x2 = x_ref ⇒ y1 = y2, both sides exactly 0.
        rhs = _flux_inner(x_ref, x_ref, f)
        @test rhs == 0.0
    end

    # N=5 random trials. Each draws fresh (x_ref, x1, x2, d), extracts
    # Hᵀd via the σ=1 trick, forward-simulates y1 and y2 through the
    # same observation operator, and checks the algebraic identity to
    # ~1e-12 relative.
    @testset "Random trial $trial" for trial in 1:5
        x_ref = _per_step_controls(rng, mesh, 3)
        x_1   = _per_step_controls(rng, mesh, 3)
        x_2   = _per_step_controls(rng, mesh, 3)
        d     = _random_departures(rng, K)

        # Hᵀ d at the linearization point x_ref. For linear H the
        # linearization point is irrelevant; we still pick one
        # explicitly so the call signature is honest.
        f, _ = _extract_Hadjoint_d(mesh, panels_rm, panels_m,
                                    panels_am, panels_bm, panels_cm,
                                    observations, x_ref, d; dt = dt)

        # y1, y2 from the same observation operator (.simulated field
        # of the production cost path).
        y_1 = _run_4dvar(mesh, panels_rm, panels_m,
                         panels_am, panels_bm, panels_cm,
                         observations, x_1; dt = dt).simulated
        y_2 = _run_4dvar(mesh, panels_rm, panels_m,
                         panels_am, panels_bm, panels_cm,
                         observations, x_2; dt = dt).simulated

        lhs = dot(d, y_1 .- y_2)
        rhs = _flux_inner(x_1, x_2, f)
        scale = max(abs(lhs), abs(rhs), 1e-30)

        # Both sides should agree to ~10 ulps of double precision.
        # The empirical probe in the design review gave ~5e-15
        # for PPM(NoLimiter); 1e-12 is comfortable headroom for
        # mesh / panel-edge accumulation.
        @test abs(lhs - rhs) / scale < 1e-12
        @test isfinite(lhs) && isfinite(rhs)
    end
end
