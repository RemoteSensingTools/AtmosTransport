#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "curated top-level public API" begin
    exported = Set(names(AtmosTransport))

    @test length(exported) < 100
    @test :run_driven_simulation in exported
    @test :validate_config in exported
    @test :inspect_binary in exported
    @test :TransportModel in exported
    @test :open_snapshot in exported

    # Advanced internals stay reachable by qualification but no longer fill
    # `using AtmosTransport` completions.
    @test isdefined(AtmosTransport, :CSTapeSlot)
    @test isdefined(AtmosTransport, :apply_B_half!)
    @test isdefined(AtmosTransport, :set_streaming_steps_per_window_schedule!)
    @test :CSTapeSlot ∉ exported
    @test :apply_B_half! ∉ exported
    @test :set_streaming_steps_per_window_schedule! ∉ exported
end
