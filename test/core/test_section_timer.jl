#!/usr/bin/env julia

using Test

import AtmosTransport

const ST = AtmosTransport.SectionTimer

@testset "SectionTimer records concurrent samples safely" begin
    ST.enable!(allocations=true)
    tasks = [Threads.@spawn begin
                 for _ in 1:100
                     ST.record_sample!(:concurrent, 10.0, Int64(2))
                 end
             end for _ in 1:8]
    fetch.(tasks)
    ST.disable!()

    snapshot = ST._timer_snapshot()
    @test length(snapshot.timings[:concurrent]) == 800
    @test length(snapshot.allocations[:concurrent]) == 800
    @test sum(snapshot.timings[:concurrent]) == 8_000.0
    @test sum(snapshot.allocations[:concurrent]) == 1_600

    report_buffer = IOBuffer()
    ST.report(report_buffer)
    @test occursin("concurrent", String(take!(report_buffer)))

    mktempdir() do tmp
        path = joinpath(tmp, "timer.csv")
        @test ST.write_csv(path) == path
        csv = read(path, String)
        @test occursin("concurrent,800", csv)
    end
end

@testset "SectionTimer rejects samples from disabled epochs" begin
    ST.enable!()
    started = Channel{Nothing}(1)
    release = Channel{Nothing}(1)
    late_task = Threads.@spawn ST.@section :late begin
        put!(started, nothing)
        take!(release)
    end
    take!(started)
    ST.disable!()
    before_release = ST._timer_snapshot()
    put!(release, nothing)
    fetch(late_task)
    after_release = ST._timer_snapshot()
    @test !haskey(before_release.timings, :late)
    @test after_release.timings == before_release.timings

    ST.enable!()
    started_next = Channel{Nothing}(1)
    release_next = Channel{Nothing}(1)
    stale_task = Threads.@spawn ST.@section :stale begin
        put!(started_next, nothing)
        take!(release_next)
    end
    take!(started_next)
    ST.disable!()
    ST.enable!()
    ST.record_sample!(:current, 1.0)
    put!(release_next, nothing)
    fetch(stale_task)
    ST.disable!()
    snapshot = ST._timer_snapshot()
    @test !haskey(snapshot.timings, :stale)
    @test snapshot.timings[:current] == [1.0]
end
