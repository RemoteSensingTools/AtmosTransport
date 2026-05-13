#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.B2 — preconditioner (χ ↔ x change of variables) for the
# CS 4D-Var surface-flux path.
#
# Coverage:
#   * Constructor validation: shape mismatch, non-finite background,
#     non-positive background under LogNormal.
#   * Forward correctness:
#       - Linear/Diagonal: `x = x_b + σ ⊙ χ` (literal arithmetic).
#       - LogNormal/Diagonal: `x = x_b ⊙ exp(σ ⊙ χ)`, always > 0,
#         `x = x_b` at `χ = 0`.
#   * `B^(-1/2)` correctness (added to Covariance.jl for this commit):
#       - Diagonal: literal `y = x ⊙ inv(σ)`.
#       - Gaussian round-trip: `B^(1/2) · B^(-1/2) x ≈ x` for smooth `x`.
#   * Bijection (`T^(-1) ∘ T = id` and `T ∘ T^(-1) = id`):
#       - Linear/Diagonal and Linear/Gaussian (round-trip exact up to
#         FFT roundoff).
#       - LogNormal/Diagonal and LogNormal/Gaussian — the user-
#         specified B2 bijection requirement.
#   * Adjoint identity: `⟨T'(χ) δχ, g_phys⟩ = ⟨δχ, T'(χ)^T g_phys⟩`
#     for both optim types crossed with both covariance types.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

const FT_TEST = Float64
const NC_SMALL = 4

# Deterministic pseudo-random tuple from sin/cos seeding.
_random_tuple(Nc, seed) = ntuple(p ->
    [FT_TEST(sin(0.7 * seed * (i + 13 * j + 41 * p)))
     for i in 1:Nc, j in 1:Nc], 6)

_unit_sigma(Nc) = ntuple(_ -> ones(FT_TEST, Nc, Nc), 6)
_constant_sigma(Nc, σ) = ntuple(_ -> fill(FT_TEST(σ), Nc, Nc), 6)
_constant_background(Nc, b) = ntuple(_ -> fill(FT_TEST(b), Nc, Nc), 6)
_zeros_tuple(Nc) = ntuple(_ -> zeros(FT_TEST, Nc, Nc), 6)

_diag_cov(Nc, σ) = AT.DiagonalCSCovariance(_constant_sigma(Nc, σ))
_gauss_cov(Nc, σ, L) = AT.IsotropicGaussianCSCovariance(_constant_sigma(Nc, σ), L)

_panel_inner(a, b) = sum(p -> sum(a[p] .* b[p]), 1:6)

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

@testset "CSSurfaceFluxPreconditioner — constructor validation" begin
    cov = _diag_cov(NC_SMALL, 0.5)
    bg = _constant_background(NC_SMALL, 1.0)

    @test AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType()) isa
          AT.CSSurfaceFluxPreconditioner
    @test AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType()) isa
          AT.CSSurfaceFluxPreconditioner

    # Shape mismatch with covariance's Nc.
    bad_shape = ntuple(_ -> ones(FT_TEST, NC_SMALL + 1, NC_SMALL + 1), 6)
    @test_throws DimensionMismatch AT.CSSurfaceFluxPreconditioner(
        cov, bad_shape, AT.LinearOptimType())

    # Non-finite background.
    nan_bg = ntuple(_ -> fill(FT_TEST(NaN), NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, nan_bg, AT.LinearOptimType())

    # Non-positive background under LogNormal — rejected.
    neg_bg = ntuple(p -> p == 3 ?
                            fill(FT_TEST(-1.0), NC_SMALL, NC_SMALL) :
                            ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, neg_bg, AT.LogNormalOptimType())
    zero_bg = ntuple(_ -> zeros(FT_TEST, NC_SMALL, NC_SMALL), 6)
    @test_throws ArgumentError AT.CSSurfaceFluxPreconditioner(
        cov, zero_bg, AT.LogNormalOptimType())

    # Non-positive background under Linear is OK (linear allows
    # any sign).
    @test AT.CSSurfaceFluxPreconditioner(cov, neg_bg, AT.LinearOptimType()) isa
          AT.CSSurfaceFluxPreconditioner
end

# ---------------------------------------------------------------------------
# B^(-1/2) — added to Covariance.jl by this commit
# ---------------------------------------------------------------------------

@testset "apply_B_half_inverse! — diagonal literal" begin
    σ = ntuple(p -> reshape(FT_TEST.((1:(NC_SMALL * NC_SMALL))) .+ p * 10,
                            NC_SMALL, NC_SMALL), 6)
    cov = AT.DiagonalCSCovariance(σ)
    x = _random_tuple(NC_SMALL, 1)

    y = _zeros_tuple(NC_SMALL)
    AT.apply_B_half_inverse!(y, cov, x)
    for p in 1:6
        @test y[p] ≈ x[p] ./ σ[p] atol = 1e-12 rtol = 1e-12
    end
end

@testset "apply_B_half_inverse! — Gaussian B^(1/2) · B^(-1/2) = I round-trip" begin
    cov = _gauss_cov(NC_SMALL, 0.5, 1.5)
    x = _random_tuple(NC_SMALL, 2)
    inv_x = _zeros_tuple(NC_SMALL)
    round_trip = _zeros_tuple(NC_SMALL)
    AT.apply_B_half_inverse!(inv_x, cov, x)
    AT.apply_B_half!(round_trip, cov, inv_x)
    for p in 1:6
        @test round_trip[p] ≈ x[p] atol = 1e-9 rtol = 1e-9
    end
end

# ---------------------------------------------------------------------------
# Forward T(χ): correctness
# ---------------------------------------------------------------------------

@testset "apply_preconditioner! — Linear/Diagonal literal" begin
    σ_scalar = FT_TEST(0.5)
    cov = _diag_cov(NC_SMALL, σ_scalar)
    bg_value = FT_TEST(420.0)
    bg = _constant_background(NC_SMALL, bg_value)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

    chi = _random_tuple(NC_SMALL, 3)
    x = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x, prec, chi)

    # x = x_b + σ ⊙ χ — literal check.
    for p in 1:6
        @test x[p] ≈ bg_value .+ σ_scalar .* chi[p] atol = 1e-12 rtol = 1e-12
    end
end

@testset "apply_preconditioner! — LogNormal/Diagonal literal + positivity" begin
    σ_scalar = FT_TEST(0.1)
    cov = _diag_cov(NC_SMALL, σ_scalar)
    bg_value = FT_TEST(420.0)
    bg = _constant_background(NC_SMALL, bg_value)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

    # χ = 0 → x = x_b.
    chi0 = _zeros_tuple(NC_SMALL)
    x0 = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x0, prec, chi0)
    for p in 1:6
        @test x0[p] ≈ bg[p] atol = 1e-12 rtol = 1e-12
    end

    # Arbitrary χ (including negative) → x = x_b ⊙ exp(σ ⊙ χ), always > 0.
    chi = _random_tuple(NC_SMALL, 4)
    x = _zeros_tuple(NC_SMALL)
    AT.apply_preconditioner!(x, prec, chi)
    for p in 1:6
        @test x[p] ≈ bg_value .* exp.(σ_scalar .* chi[p]) atol = 1e-10 rtol = 1e-10
        @test all(x[p] .> 0)
    end
end

# ---------------------------------------------------------------------------
# Bijection: T^(-1) ∘ T = id  and  T ∘ T^(-1) = id
# ---------------------------------------------------------------------------

@testset "Linear preconditioner — bijection (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.5),
                _gauss_cov(NC_SMALL, 0.5, 1.2))
        bg = _constant_background(NC_SMALL, 420.0)
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

        # Round trip from χ.
        chi = _random_tuple(NC_SMALL, 5)
        x = _zeros_tuple(NC_SMALL)
        chi_back = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)
        AT.apply_preconditioner_inverse!(chi_back, prec, x)
        for p in 1:6
            @test chi_back[p] ≈ chi[p] atol = 1e-9 rtol = 1e-9
        end

        # Round trip from x (use a smooth x: T of a smooth chi).
        x_in = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x_in, prec, _random_tuple(NC_SMALL, 6))
        chi_mid = _zeros_tuple(NC_SMALL)
        x_back = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_inverse!(chi_mid, prec, x_in)
        AT.apply_preconditioner!(x_back, prec, chi_mid)
        for p in 1:6
            @test x_back[p] ≈ x_in[p] atol = 1e-9 rtol = 1e-9
        end
    end
end

@testset "LogNormal preconditioner — bijection (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.1),
                _gauss_cov(NC_SMALL, 0.1, 1.2))
        bg = _constant_background(NC_SMALL, 1.0)   # x_b > 0
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

        chi = _random_tuple(NC_SMALL, 7)
        x = _zeros_tuple(NC_SMALL)
        chi_back = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)

        # Sanity: forward gives positivity.
        for p in 1:6
            @test all(x[p] .> 0)
        end

        AT.apply_preconditioner_inverse!(chi_back, prec, x)
        for p in 1:6
            @test chi_back[p] ≈ chi[p] atol = 1e-9 rtol = 1e-9
        end

        # T ∘ T^(-1) = id on x produced by T.
        x_in = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x_in, prec, _random_tuple(NC_SMALL, 8))
        chi_mid = _zeros_tuple(NC_SMALL)
        x_back = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_inverse!(chi_mid, prec, x_in)
        AT.apply_preconditioner!(x_back, prec, chi_mid)
        for p in 1:6
            @test x_back[p] ≈ x_in[p] atol = 1e-9 rtol = 1e-9
        end
    end
end

@testset "LogNormal inverse — rejects non-positive x" begin
    cov = _diag_cov(NC_SMALL, 0.1)
    bg = _constant_background(NC_SMALL, 1.0)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

    bad_x = ntuple(p -> p == 2 ?
                            fill(FT_TEST(-1.0), NC_SMALL, NC_SMALL) :
                            ones(FT_TEST, NC_SMALL, NC_SMALL), 6)
    chi = _zeros_tuple(NC_SMALL)
    @test_throws ArgumentError AT.apply_preconditioner_inverse!(chi, prec, bad_x)
end

# ---------------------------------------------------------------------------
# Adjoint identity — ⟨T'(χ) δχ, g_phys⟩ = ⟨δχ, T'(χ)^T g_phys⟩
# ---------------------------------------------------------------------------

@testset "Adjoint identity — Linear (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.5),
                _gauss_cov(NC_SMALL, 0.5, 1.5))
        bg = _constant_background(NC_SMALL, 420.0)
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

        chi = _random_tuple(NC_SMALL, 9)
        delta_chi = _random_tuple(NC_SMALL, 10)
        g_phys = _random_tuple(NC_SMALL, 11)

        # Base point x = T(χ) — Linear's tangent doesn't actually use x.
        x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)

        delta_x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_tangent!(delta_x, prec, x, delta_chi)

        g_chi = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_adjoint!(g_chi, prec, x, g_phys)

        lhs = _panel_inner(delta_x, g_phys)
        rhs = _panel_inner(delta_chi, g_chi)
        @test isapprox(lhs, rhs; atol = 1e-10, rtol = 1e-10)
    end
end

@testset "Adjoint identity — LogNormal (Diagonal + Gaussian)" begin
    for cov in (_diag_cov(NC_SMALL, 0.1),
                _gauss_cov(NC_SMALL, 0.1, 1.5))
        bg = _constant_background(NC_SMALL, 1.0)
        prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LogNormalOptimType())

        chi = _random_tuple(NC_SMALL, 12)
        delta_chi = _random_tuple(NC_SMALL, 13)
        g_phys = _random_tuple(NC_SMALL, 14)

        # Base point x = T(χ) — LogNormal's tangent uses x.
        x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner!(x, prec, chi)

        delta_x = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_tangent!(delta_x, prec, x, delta_chi)

        g_chi = _zeros_tuple(NC_SMALL)
        AT.apply_preconditioner_adjoint!(g_chi, prec, x, g_phys)

        lhs = _panel_inner(delta_x, g_phys)
        rhs = _panel_inner(delta_chi, g_chi)
        @test isapprox(lhs, rhs; atol = 1e-10, rtol = 1e-10)
    end
end
