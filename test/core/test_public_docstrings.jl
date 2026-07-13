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
    :TransportWindow,
    :TransportTracerSpec,
    :write_transport_binary,
    :build_runtime_physics_recipe,
    :validate_runtime_physics_recipe,
    :run!,
    :run_window!,
)

const DOCUMENTED_MET_DRIVER_API = (
    :TransportBinaryContract,
    :StreamingTransportBinaryWriter,
    :canonical_window_constant_contract,
    :validate_transport_contract!,
    :load_window!,
    :load_qv_pair_window!,
    :load_flux_delta_window!,
    :load_transport_window,
    :binary_capabilities,
    :inspect_binary,
    :has_qv_endpoints,
    :air_mass_basis,
    :driver_grid,
)

@testset "core public API has attached docstrings" begin
    for name in DOCUMENTED_CORE_API
        @test Base.Docs.hasdoc(AtmosTransport, name)
    end
end

@testset "met-driver API has attached docstrings" begin
    for name in DOCUMENTED_MET_DRIVER_API
        @test Base.Docs.hasdoc(AtmosTransport.MetDrivers, name)
    end
end
