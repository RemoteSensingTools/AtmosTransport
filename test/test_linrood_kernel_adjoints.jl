#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 — LinRood CS adjoint, kernel-level transposition tests.
#
# Verifies the discrete transpose of each forward LinRood kernel by the
# adjoint identity
#     ⟨y, F·x⟩ = ⟨Fᵀ·y, x⟩
# evaluated on random inputs. The identity must hold to floating-point
# precision because each forward kernel is linear in the differentiated
# inputs once velocities (`am`, `bm`) are fixed.
#
# Commit 1 covers `_linrood_update_kernel!` (the averaged-flux update).
# Commits 2 and 3 will add `_pre_advect_*_kernel!` and the four face
# kernels to this file as they ship.
# ---------------------------------------------------------------------------

using Test
using Random
using KernelAbstractions: get_backend, synchronize

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Sum of element-wise products over the interior region `[Hp+1 : Hp+Nc,
# Hp+1 : Hp+Nc, :]` of two haloed cubed-sphere arrays.
function _inner_interior(a::AbstractArray{FT, 3}, b::AbstractArray{FT, 3},
                          Nc::Int, Hp::Int) where {FT}
    s = zero(FT)
    @inbounds for k in axes(a, 3)
        for j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            s += a[i, j, k] * b[i, j, k]
        end
    end
    return s
end

# Sum of element-wise products over the full (already non-haloed) face
# array.
function _inner_full(a::AbstractArray{FT}, b::AbstractArray{FT}) where {FT}
    s = zero(FT)
    @inbounds for I in eachindex(a)
        s += a[I] * b[I]
    end
    return s
end

# Build a tiny CS panel layout for one face: random rm, m, am, bm,
# fx_in, fx_out, fy_in, fy_out arrays sized exactly the way
# `_linrood_update_kernel!` expects.
function _linrood_update_panel_inputs(; Nc=4, Hp=3, Nz=3, FT=Float64, seed=1)
    rng = MersenneTwister(seed)
    N = Nc + 2Hp
    rm = randn(rng, FT, N, N, Nz)
    m  = FT(2) .+ rand(rng, FT, N, N, Nz)  # strictly positive
    am = FT(0.01) .* randn(rng, FT, Nc + 1, Nc, Nz)
    bm = FT(0.01) .* randn(rng, FT, Nc, Nc + 1, Nz)
    fx_in  = randn(rng, FT, Nc + 1, Nc, Nz)
    fx_out = randn(rng, FT, Nc + 1, Nc, Nz)
    fy_in  = randn(rng, FT, Nc, Nc + 1, Nz)
    fy_out = randn(rng, FT, Nc, Nc + 1, Nz)
    return (; rm, m, am, bm, fx_in, fx_out, fy_in, fy_out)
end

# Run the forward `_linrood_update_kernel!` directly on the inputs and
# return the (rm_new, m_new) interior result, written into freshly
# allocated haloed arrays so the inner-product helpers can be used
# without adjustment.
function _run_linrood_update_forward(inputs, mesh::AT.CubedSphereMesh{FT}) where {FT}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(inputs.rm, 3)
    N = Nc + 2Hp
    rm_new = zeros(FT, N, N, Nz)
    m_new  = zeros(FT, N, N, Nz)
    backend = get_backend(rm_new)
    k! = Adv._linrood_update_kernel!(backend, 256)
    k!(rm_new, m_new,
       inputs.rm, inputs.m, inputs.am, inputs.bm,
       inputs.fx_in, inputs.fx_out, inputs.fy_in, inputs.fy_out, Hp;
       ndrange=(Nc, Nc, Nz))
    synchronize(backend)
    return (rm_new, m_new)
end

@testset "Plan 25 Commit 1 — LinRood update kernel adjoint" begin
    @testset "transposition identity ($(FT))" for FT in (Float64, Float32)
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        inputs = _linrood_update_panel_inputs(; Nc, Hp, Nz, FT, seed=42)

        # Forward
        rm_new, m_new = _run_linrood_update_forward(inputs, mesh)

        # `_linrood_update_kernel!` is affine in `(rm, m, fx_*, fy_*)`
        # for fixed velocities: `m_new = m + (am_w - am_e) + (bm_s - bm_n)`
        # carries a constant velocity-divergence offset that depends
        # only on the meteo tape. The adjoint identity holds for the
        # **linear** part of the map only, so we subtract the
        # zero-tracer-state baseline before forming the LHS inner
        # product.
        zero_inputs = (
            rm     = zero(inputs.rm),
            m      = zero(inputs.m),
            am     = inputs.am,
            bm     = inputs.bm,
            fx_in  = zero(inputs.fx_in),
            fx_out = zero(inputs.fx_out),
            fy_in  = zero(inputs.fy_in),
            fy_out = zero(inputs.fy_out),
        )
        rm_new_zero, m_new_zero = _run_linrood_update_forward(zero_inputs, mesh)
        rm_new_linear = rm_new .- rm_new_zero
        m_new_linear  = m_new  .- m_new_zero

        # Random adjoint seed on the outputs. Halo cells of the seed are
        # irrelevant — the adjoint kernel only reads interior `[ii, jj]`
        # — but we keep them zero so the LHS inner product is restricted
        # to the interior region by construction.
        rng = MersenneTwister(43)
        lambda_rm_new = zeros(FT, N, N, Nz)
        lambda_m_new  = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            lambda_rm_new[i, j, k] = randn(rng, FT)
            lambda_m_new[i, j, k]  = randn(rng, FT)
        end

        # Adjoint outputs (initialised to zero — the kernel only
        # accumulates).
        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fx_in  = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fx_out = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fy_in  = zeros(FT, Nc, Nc + 1, Nz)
        lambda_fy_out = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_linrood_update_adjoint!(
            lambda_rm, lambda_m,
            lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
            lambda_rm_new, lambda_m_new,
            inputs.am, inputs.bm, mesh,
        )

        # LHS: ⟨y, F·x⟩ — sum over interior outputs only, with the
        # affine constant subtracted (see comment above).
        lhs = _inner_interior(lambda_rm_new, rm_new_linear, Nc, Hp) +
              _inner_interior(lambda_m_new,  m_new_linear,  Nc, Hp)

        # RHS: ⟨Fᵀ·y, x⟩ — sum the six adjoint inputs against the
        # forward inputs. Cell-centred terms restricted to interior;
        # face terms span the full face array.
        rhs = _inner_interior(lambda_rm, inputs.rm, Nc, Hp) +
              _inner_interior(lambda_m,  inputs.m,  Nc, Hp) +
              _inner_full(lambda_fx_in,  inputs.fx_in)  +
              _inner_full(lambda_fx_out, inputs.fx_out) +
              _inner_full(lambda_fy_in,  inputs.fy_in)  +
              _inner_full(lambda_fy_out, inputs.fy_out)

        tol = FT === Float64 ? 1e-12 : 1f-5
        @test isapprox(lhs, rhs; atol=tol, rtol=tol)
    end

    @testset "adjoint zeros on zero seed" begin
        # Sanity check: if the seed is zero, no adjoint accumulators
        # should be touched.
        FT = Float64
        Nc = 4; Hp = 3; Nz = 3
        mesh = AT.CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp

        inputs = _linrood_update_panel_inputs(; Nc, Hp, Nz, FT, seed=7)

        lambda_rm_new = zeros(FT, N, N, Nz)
        lambda_m_new  = zeros(FT, N, N, Nz)

        lambda_rm = zeros(FT, N, N, Nz)
        lambda_m  = zeros(FT, N, N, Nz)
        lambda_fx_in  = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fx_out = zeros(FT, Nc + 1, Nc, Nz)
        lambda_fy_in  = zeros(FT, Nc, Nc + 1, Nz)
        lambda_fy_out = zeros(FT, Nc, Nc + 1, Nz)

        Adv.apply_linrood_update_adjoint!(
            lambda_rm, lambda_m,
            lambda_fx_in, lambda_fx_out, lambda_fy_in, lambda_fy_out,
            lambda_rm_new, lambda_m_new,
            inputs.am, inputs.bm, mesh,
        )

        @test all(iszero, lambda_rm)
        @test all(iszero, lambda_m)
        @test all(iszero, lambda_fx_in)
        @test all(iszero, lambda_fx_out)
        @test all(iszero, lambda_fy_in)
        @test all(iszero, lambda_fy_out)
    end
end
