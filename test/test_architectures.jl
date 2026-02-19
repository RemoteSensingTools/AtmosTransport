using Test
using AtmosTransportModel.Architectures

@testset "Architectures" begin
    @test CPU() isa AbstractArchitecture
    @test GPU() isa AbstractArchitecture
    @test array_type(CPU()) === Array
    @test_throws MethodError array_type(GPU())  # no CUDA loaded
end
