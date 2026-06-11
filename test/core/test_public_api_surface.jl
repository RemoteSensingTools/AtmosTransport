#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "curated top-level public API" begin
    exported = Set(names(AtmosTransport))

    # Soft cap on top-level export count. Original threshold was 100; the
    # 2026-05 additions (`NoAdvection`, `CMFMCMatrixConvection`,
    # `AbstractConvection`, `ConvectionForcing`, `apply_convection!`,
    # `has_convection_forcing`, `AbstractMetDriver`, `lonlat_to_panel_xy`)
    # took the count to 107. Bump to 120 to give breathing room. Next API
    # audit should bring it back down (good candidates: internal accessors
    # that escaped through `using .Submodule` re-exports).
    # 2026-06 (plan 45): the reference-state accessor split adds deliberate
    # public API (`get_tracer_raw`/`get_tracer_full`/`total_mass_full`/
    # `mixing_ratio_full`) — physical-field readers that downstream
    # diagnostics MUST use instead of the bare accessors. Count 148; bump
    # the cap to 160. The next audit's trim candidates stand.
    @test length(exported) < 160
    @test :get_tracer_full in exported     # plan-45 physical-field reader
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
