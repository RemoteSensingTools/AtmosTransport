using Test
using AtmosTransport.Callbacks

@testset "Callbacks" begin
    triggered = Ref(false)
    cb = DiscreteCallback(
        (model, t) -> t > 100.0,
        (model) -> triggered[] = true)

    @test cb isa AbstractCallback

    # Should not trigger at t=0
    execute_callbacks!([cb], nothing, 0.0)
    @test triggered[] == false

    # Should trigger at t=200
    execute_callbacks!([cb], nothing, 200.0)
    @test triggered[] == true
end
