using Test
using AtmosTransport

const DOCUMENTED_CORE_API = (
    :TransportModel,
    :CubedSphereState,
    :StructuredFaceFluxState,
    :FaceIndexedFluxState,
    :CubedSphereFaceFluxState,
    :CMFMCConvection,
    :TM5Convection,
    :TransportBinaryReader,
    :TransportBinaryHeader,
    :CubedSphereBinaryGeometry,
    :TransportBinaryDriver,
    :CubedSphereTransportDriver,
    :StructuredTransportWindow,
    :FaceIndexedTransportWindow,
    :CubedSphereTransportWindow,
    :TransportTracerSpec,
    :write_transport_binary,
    :build_runtime_physics_recipe,
    :validate_runtime_physics_recipe,
    :run!,
    :run_window!,
)

@testset "core public API has attached docstrings" begin
    for name in DOCUMENTED_CORE_API
        @test Base.Docs.hasdoc(AtmosTransport, name)
    end
end
