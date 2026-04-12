#!/usr/bin/env julia
#
# Top-level test runner for the offline regridding glue
# (`AtmosTransport.Regridding`). Invoked either directly
# (`julia --project test_v2/regridding/runtests.jl`) or included
# from `test/runtests.jl`.
#
# Tier 4 of /home/cfranken/.claude/plans/luminous-prancing-firefly.md.

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Regridding

@testset "Regridding (Tier 4)" begin
    include("test_cubed_sphere_corners.jl")
    include("test_conservation.jl")
    include("test_transpose.jl")
    include("test_serialization.jl")
    include("test_reduced_gaussian_stub.jl")
end
