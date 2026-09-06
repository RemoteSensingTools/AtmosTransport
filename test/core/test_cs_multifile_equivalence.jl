using Test
include(joinpath(@__DIR__,"..","fixtures","cs_multifile.jl"))
using .CSDriverHandoffFixtures

@testset "CS transport, diffusion, convection, and decay agree across files" begin
    mktempdir() do dir
        combined, first, second = joinpath.(dir,["combined.bin","first.bin","second.bin"])
        cs_handoff_fixture(combined,[1.0,1.0,8.0,8.0])
        cs_handoff_fixture(first,[1.0,1.0])
        cs_handoff_fixture(second,[8.0,8.0])
        invalid_cfg = Dict{String,Any}("input"=>Dict("binary_paths"=>[first]),
            "run"=>Dict("start_window"=>2),
            "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>400e-6))))
        failure = try
            CSDriverHandoffFixtures.AtmosTransport.Models.run_driven_simulation(invalid_cfg)
        catch err
            err
        end
        @test failure isa ArgumentError
        @test occursin("start_window=1",sprint(showerror,failure))
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
