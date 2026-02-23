using Test
using AtmosTransport.Regridding

@testset "Regridding" begin
    @test IdentityRegridder() isa AbstractRegridder

    # Identity regridder should just copy
    src = rand(10)
    dst = zeros(10)
    regrid!(dst, src, IdentityRegridder())
    @test dst == src
end
