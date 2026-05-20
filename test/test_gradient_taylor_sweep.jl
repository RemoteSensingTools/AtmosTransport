#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# TM5 4DVAR gradient test — Taylor-sweep convergence (Eqs. 2.8-2.11 of
# the TM5 4DVAR adjoint-and-gradient-test memo).
#
# Given the cost J(χ) and gradient g₀ = ∇_χ J(χ₀), the Taylor expansion
# at χ₀ along direction −g₀ gives
#
#     J(χ₀ − α g₀) = J(χ₀) − α ⟨g₀, g₀⟩ + ½ α² ⟨g₀, J''(χ₀) g₀⟩ + O(α³)
#
# Define DJ₁ = ⟨g₀, g₀⟩ and DJ₂(α) = (J(χ₀) − J(χ₀ − α g₀)) / α. Then
#
#     r(α) ≡ 1 − DJ₂(α) / DJ₁ = ½ α ⟨g₀, J''(χ₀) g₀⟩ / DJ₁ + O(α²)
#
# is linear in α to leading order. Reducing α by a factor f should
# reduce |r(α)| by f. We sweep α geometrically by f=2 and assert:
#
#   (1) `|r(α)| → 0` along the sweep (smallest clean |r| ≤ tol₁).
#   (2) The ratio `|r_{k+1}| / |r_k|` lands in `[0.4, 0.6]` (≈ 1/f)
#       for the clean middle of the sweep.
#   (3) Cancellation noise (where `J₀ − J_k` runs out of digits) is
#       detected and the sweep is truncated before evaluating (2).
#
# Unlike the algebraic identity tests, this check holds for any
# preconditioner whose tangent and adjoint are coded correctly —
# Linear or LogNormal. It is the strongest end-to-end check of
# (preconditioner, transport, observation operator) consistency:
# any one of them being wrong will break the linear convergence.
# ---------------------------------------------------------------------------

using Test
using Random
using Printf

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
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

# Three per-step χ-space controls aligned with `nsteps = 3`.
function _make_chi_controls(chi_panels_per_step)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                chi_panels_per_step[k])
            for k in eachindex(chi_panels_per_step)]
end

# Observations chosen with large enough `value` that J(χ₀=0) — i.e.
# at x = x_b — is well above roundoff, so the Taylor sweep has
# headroom before cancellation kicks in.
function _build_observations()
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.50, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2),   0.40, 1.0),
        AT.CSObservation(3, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.30, 1.0),
        AT.CSObservation(3, AT.CSColumnMeanObjective(4, 2, 1),   0.20, 1.0),
    ]
end

function _build_preconditioner(mesh; cov_kind::Symbol = :diagonal,
                                optim_type::AT.AbstractCSOptimType = AT.LinearOptimType())
    Nc = mesh.Nc
    sigma = ntuple(_ -> fill(0.3, Nc, Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    x_b = ntuple(_ -> fill(1.0, Nc, Nc), 6)
    return AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)
end

# One-shot `cs_surface_flux_4dvar` wrapper. Returns the same result
# object so callers can read `.cost` and (for χ₀ only) `.gradients`.
# Yes, every sweep call also evaluates the adjoint — we accept that
# overhead rather than re-deriving a cost-only path, because the
# alternative would be a parallel forward-evaluation routine that
# could silently drift from the production observation operator.
function _eval_at_chi(mesh, panels_rm0, panels_m0,
                     panels_am, panels_bm, panels_cm,
                     observations, chi_controls, preconditioner; dt = 2.0)
    return AT.cs_surface_flux_4dvar(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, observations, chi_controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = dt,
        preconditioner = preconditioner)
end

# Vector-space helpers over `Vector{NTuple{6, Matrix{Float64}}}`.
function _flat_inner(a, b)
    total = 0.0
    @inbounds for c in eachindex(a), p in 1:6
        total += sum(a[c][p] .* b[c][p])
    end
    return total
end

# χ_k = χ₀ - α g₀, returned as a Vector{NTuple{6, Matrix}}.
function _step_chi(chi0, g, alpha)
    return [ntuple(p -> chi0[c].value[p] .- alpha .* g[c][p], 6)
            for c in eachindex(chi0)]
end

# Find the longest geometrically-decreasing prefix of |r| that
# survives cancellation, then return ratios `|r_{k+1}|/|r_k|` over
# that prefix. The classical signature of a passing gradient test is
# that these ratios cluster near `1/f`.
function _clean_ratios(r_seq)
    abs_r = abs.(r_seq)
    # Truncate at the first index where |r| stops decreasing — that's
    # where Taylor's linear term and roundoff noise are comparable.
    clean_end = length(abs_r)
    for k in 2:length(abs_r)
        if abs_r[k] >= abs_r[k-1]
            clean_end = k - 1
            break
        end
    end
    return [abs_r[k+1] / abs_r[k] for k in firstindex(abs_r):clean_end-1], clean_end, abs_r
end

# Single Taylor-sweep trial for a given `(χ₀, preconditioner)`. Returns
# the sweep diagnostics so the testset can report them on failure.
function _taylor_sweep(mesh, panels_rm, panels_m,
                       panels_am, panels_bm, panels_cm,
                       observations, chi0_controls, prec;
                       alpha0 = 1e-1, factor = 2.0, M = 10, dt = 2.0)
    base = _eval_at_chi(mesh, panels_rm, panels_m,
                        panels_am, panels_bm, panels_cm,
                        observations, chi0_controls, prec; dt = dt)
    J0 = base.cost
    g0 = base.gradients
    DJ1 = _flat_inner(g0, g0)
    DJ1 > 0 || error("Taylor sweep requires DJ1 > 0; got $DJ1 — pick a different χ₀ " *
                     "or boost observation magnitude")

    alphas = [alpha0 / factor^(k-1) for k in 1:M]
    r_seq = Vector{Float64}(undef, M)
    for k in eachindex(alphas)
        chi_k_panels = _step_chi(chi0_controls, g0, alphas[k])
        chi_k_controls = _make_chi_controls(chi_k_panels)
        Jk = _eval_at_chi(mesh, panels_rm, panels_m,
                          panels_am, panels_bm, panels_cm,
                          observations, chi_k_controls, prec; dt = dt).cost
        DJ2 = (J0 - Jk) / alphas[k]
        r_seq[k] = 1 - DJ2 / DJ1
    end
    return alphas, r_seq, J0, DJ1
end

# Pretty-print the sweep for diagnostic output on failure.
function _print_sweep(alphas, r_seq, abs_r, clean_end)
    println("    α               r                |r|              clean?")
    for k in eachindex(alphas)
        mark = k <= clean_end ? "✓" : "·"
        @printf("    %.3e       %+.3e        %.3e       %s\n",
                alphas[k], r_seq[k], abs_r[k], mark)
    end
end

@testset "Gradient Taylor sweep (memo Eqs. 2.8-2.11)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations()

    # Three fixtures covering both the χ₀ placement and the
    # preconditioner family. The PDF's gradient test holds for any
    # preconditioner whose tangent and adjoint are coded correctly,
    # so we exercise both `LinearOptimType` (x = x_b + B^(1/2) χ)
    # and `LogNormalOptimType` (x = x_b ⊙ exp(B^(1/2) χ)) — the
    # latter being non-linear in χ. A bug in either preconditioner's
    # adjoint would break the f-factor convergence signature here.
    chi_zero = [ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6) for _ in 1:3]
    chi_rand = [ntuple(_ -> 0.2 .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6)
                for _ in 1:3]

    fixtures = (
        ("Linear L, χ₀ = 0 (prior)",   chi_zero, AT.LinearOptimType()),
        ("Linear L, χ₀ random",         chi_rand, AT.LinearOptimType()),
        ("LogNormal L, χ₀ random",      chi_rand, AT.LogNormalOptimType()),
    )

    @testset "$(label)" for (label, chi0_panels, optim_type) in fixtures
        prec = _build_preconditioner(mesh;
                                     cov_kind = :diagonal,
                                     optim_type = optim_type)
        chi0_controls = _make_chi_controls(chi0_panels)

        alphas, r_seq, J0, DJ1 = _taylor_sweep(
            mesh, panels_rm, panels_m,
            panels_am, panels_bm, panels_cm,
            observations, chi0_controls, prec;
            alpha0 = 1e-1, factor = 2.0, M = 10, dt = dt)

        ratios, clean_end, abs_r = _clean_ratios(r_seq)

        # Diagnostic dump — visible when the testset fails.
        @info "Taylor sweep" label J0 DJ1 clean_end

        # (1) The sweep must reach a clean prefix of at least 4 points;
        # otherwise either DJ1 is too small or α₀ is too small to
        # exercise the linear term before cancellation hits.
        @test clean_end >= 4

        # (2) Smallest clean |r| must be small in absolute terms —
        # confirms DJ₂(α) → DJ₁ as α → 0, i.e. g₀ is the true
        # directional derivative.
        @test abs_r[clean_end] < 1e-3

        # (3) Ratios in the clean middle of the sweep should cluster
        # near 1/f = 0.5. We allow [0.4, 0.6]; the leading α is large
        # enough that O(α²) terms in r(α) bias the first ratio
        # slightly, so we skip the first one if there are enough
        # samples.
        ratio_window = length(ratios) >= 4 ? ratios[2:end] : ratios
        @test all(0.4 .<= ratio_window .<= 0.6)

        # Failure-time diagnostics.
        if !all(0.4 .<= ratio_window .<= 0.6) || abs_r[clean_end] >= 1e-3
            _print_sweep(alphas, r_seq, abs_r, clean_end)
            @info "Taylor sweep ratios" ratios
        end
    end
end
