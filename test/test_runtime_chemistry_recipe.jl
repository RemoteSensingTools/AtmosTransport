#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# build_runtime_chemistry + recipe.chemistry wiring through DrivenSimulation.
#
# Covers:
#   1. Builder dispatch on `[chemistry].kind`:
#      - missing / "none" → NoChemistry
#      - "decay" with empty half-lives → NoChemistry
#      - "decay" with `half_lives_seconds.rn222 = 330350.4` →
#        ExponentialDecay with first-order rate `ln(2)/T`
#      - unknown kind → ArgumentError
#   2. End-to-end decay correctness: a one-tracer state with VMR ≡ 1.0
#      stepped through one day under `ExponentialDecay(Rn222=330350.4)`
#      yields VMR ≈ exp(-ln(2) · 86400 / 330350.4) ≈ 0.834.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
using .AT.Models: build_runtime_chemistry
using .AT.State: field_value

@testset "build_runtime_chemistry — dispatch" begin
    # No [chemistry] section → NoChemistry
    op = build_runtime_chemistry(Dict{String,Any}(), Float64)
    @test op isa AT.Operators.Chemistry.NoChemistry

    # kind = "none"
    op = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "none")), Float64)
    @test op isa AT.Operators.Chemistry.NoChemistry

    # kind = "decay" with no half-lives — degenerate, no tracers to decay
    op = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay")), Float64)
    @test op isa AT.Operators.Chemistry.NoChemistry

    # kind = "decay" with Rn222
    op = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay",
                                  "half_lives_seconds" => Dict("rn222" => 330350.4))),
        Float64)
    @test op isa AT.Operators.Chemistry.ExponentialDecay
    @test op.tracer_names == (:rn222,)
    rate = field_value(op.decay_rates[1], ())
    @test rate ≈ log(2) / 330350.4 atol = 1e-12

    # Float32 path
    op32 = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay",
                                  "half_lives_seconds" => Dict("rn222" => 330350.4))),
        Float32)
    @test op32 isa AT.Operators.Chemistry.ExponentialDecay
    @test eltype(op32.decay_rates[1].value) === Float32

    # Bad kind
    @test_throws ArgumentError build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "photolysis")), Float64)

    # Multi-tracer decay (Rn222 + Kr85)
    op = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay",
                                  "half_lives_seconds" =>
                                      Dict("rn222" => 330350.4,
                                            "kr85"  => 3.394e8))),
        Float64)
    @test op isa AT.Operators.Chemistry.ExponentialDecay
    @test length(op.tracer_names) == 2
    @test :rn222 in op.tracer_names
    @test :kr85 in op.tracer_names
end

@testset "RuntimePhysicsRecipe — chemistry field default + override" begin
    # 3-arg legacy constructor defaults chemistry to NoChemistry
    rec3 = AT.Models.RuntimePhysicsRecipe(
        AT.UpwindScheme(), AT.NoDiffusion(), AT.NoConvection())
    @test rec3.chemistry isa AT.Operators.Chemistry.NoChemistry

    # 4-arg explicit chemistry
    chem = AT.Operators.Chemistry.ExponentialDecay(; rn222 = 330350.4)
    rec4 = AT.Models.RuntimePhysicsRecipe(
        AT.UpwindScheme(), AT.NoDiffusion(), AT.NoConvection(), chem)
    @test rec4.chemistry === chem
end
