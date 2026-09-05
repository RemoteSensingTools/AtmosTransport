using Test
include(joinpath(@__DIR__,"..","fixtures","cs_multifile.jl"))
using .CSDriverHandoffFixtures

@testset "CS transport, diffusion, convection, and decay agree across files" begin
    mktempdir() do dir
        combined, first, second = joinpath.(dir,["combined.bin","first.bin","second.bin"])
        cs_handoff_fixture(combined,[1.0,1.0,8.0,8.0])
        cs_handoff_fixture(first,[1.0,1.0])
        cs_handoff_fixture(second,[8.0,8.0])
        for (advection,convection) in (("upwind","cmfmc"),("ppm","cmfmc_matrix"),("linrood","tm5"))
            continuous = cs_handoff_run([combined],advection,convection)
            split = cs_handoff_run([first,second],advection,convection)
            for p in 1:6
                @test split.state.air_mass[p] == continuous.state.air_mass[p]
                @test split.state.tracers_raw[p] ≈ continuous.state.tracers_raw[p] rtol=1e-12
                @test all(isfinite,split.state.tracers_raw[p])
            end
        end
    end
end
