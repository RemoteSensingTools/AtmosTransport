using Test
using AtmosTransportModel.TimeSteppers

@testset "TimeSteppers" begin
    @testset "Clock" begin
        c = Clock()
        @test c.time == 0.0
        @test c.iteration == 0
        tick!(c, 3600.0)
        @test c.time == 3600.0
        @test c.iteration == 1
        tick_backward!(c, 3600.0)
        @test c.time == 0.0
        @test c.iteration == 0
    end
end
