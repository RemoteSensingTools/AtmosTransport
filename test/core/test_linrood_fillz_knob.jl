#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# [advection] fillz knob (LinRood positivity fixer on/off)
#
# fillz's F32 round-trip is the LinRood scheme's only mass non-conservation
# (attribution: fillz/surplus = 1.000 on a sharp IC=0 tracer). The knob keeps
# the GCHP-faithful default (fillz = true) and offers an exactly-conservative
# mode (fillz = false) for budget/4D-Var work. Full tradeoff:
# docs/reference/ADVECTION_SCHEMES.md.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Models: advection_spec, materialize,
    CubedSphereRuntimeRecipeStyle, LinRoodAdvectionSpec

@testset "scheme constructors carry fillz (back-compat default true)" begin
    @test LinRoodPPMScheme().fillz === true            # fieldless-era forms
    @test LinRoodPPMScheme{7}().fillz === true
    @test LinRoodPPMScheme(7).fillz === true
    @test LinRoodPPMScheme(7; fillz = false).fillz === false
    @test_throws ArgumentError LinRoodPPMScheme(6)
end

@testset "[advection] fillz parse + materialize" begin
    # default: linrood without the key -> fillz=true
    spec = advection_spec(Dict{String, Any}("scheme" => "linrood"))
    @test spec isa LinRoodAdvectionSpec && spec.fillz === true
    sch = materialize(spec, CubedSphereRuntimeRecipeStyle())
    @test sch isa LinRoodPPMScheme{5} && sch.fillz === true

    # explicit off, with order
    spec = advection_spec(Dict{String, Any}("scheme" => "linrood",
                                            "ppm_order" => 7, "fillz" => false))
    @test spec.order == 7 && spec.fillz === false
    sch = materialize(spec, CubedSphereRuntimeRecipeStyle())
    @test sch isa LinRoodPPMScheme{7} && sch.fillz === false

    # linrood-only knob rejected on EVERY other scheme (typo guard)
    for other in ("ppm", "upwind", "slopes", "none")
        @test_throws ArgumentError advection_spec(
            Dict{String, Any}("scheme" => other, "fillz" => false))
    end
    # non-bool value rejected
    @test_throws ArgumentError advection_spec(
        Dict{String, Any}("scheme" => "linrood", "fillz" => "off"))
end

@testset "fillz=false skips the fixer (wiring)" begin
    # The palindrome guards each _fillz_rm_panels! call with the flag; pin the
    # wiring the same way as the start_time tripwire (a behavioral end-to-end
    # check is the 1-day fossil A/B: fillz-injected mass reads exactly 0).
    src = read(normpath(joinpath(@__DIR__, "..", "..", "src", "Operators",
                                 "Advection", "LinRood.jl")), String)
    @test count("fillz && _fillz_rm_panels!", src) == 4
    src2 = read(normpath(joinpath(@__DIR__, "..", "..", "src", "Operators",
                                  "Advection", "StrangSplitting.jl")), String)
    @test occursin("fillz = scheme.fillz", src2)
end

println("test_linrood_fillz_knob.jl OK")
