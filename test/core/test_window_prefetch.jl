using Test, AtmosTransport
include(joinpath(@__DIR__, "..", "fixtures", "window_prefetch.jl"))
using .WindowPrefetchFixtures
@testset "CPU startup loads one forcing window" begin
    check_prefetch_startup(Array)
    check_prefetch_startup(Array; enabled=false)
    check_prefetch_startup(Array; stop_window=1)
end
