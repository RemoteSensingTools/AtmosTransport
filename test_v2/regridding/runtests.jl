#!/usr/bin/env julia
#
# Top-level test runner for the offline regridding glue
# (`AtmosTransportV2.Regridding`). Invoked either directly
# (`julia --project test_v2/regridding/runtests.jl`) or included
# from `test/runtests.jl`.
#
# Tier 4 of /home/cfranken/.claude/plans/luminous-prancing-firefly.md.

using Test

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2.Regridding

@testset "Regridding (Tier 4)" begin
    include("test_cubed_sphere_corners.jl")
    include("test_conservation.jl")
    include("test_transpose.jl")
    include("test_serialization.jl")
    include("test_reduced_gaussian_stub.jl")
end
