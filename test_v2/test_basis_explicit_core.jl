#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

@testset "PlanetParameters and AtmosGrid" begin
    params = PlanetParameters(; FT=Float32, radius=6.0f6, gravity=9.8f0, reference_pressure=1.0f5)
    mesh = LatLonMesh(; FT=Float32, Nx=4, Ny=3)
    vc = HybridSigmaPressure(Float32[0, 100, 300], Float32[0, 0, 1])
    grid = @inferred AtmosGrid(mesh, vc, AtmosTransportV2.CPU(); planet=params)

    @test planet_parameters(grid) == params
    @test radius(grid) == params.radius
    @test gravity(grid) == params.gravity
    @test reference_pressure(grid) == params.reference_pressure
    @test grid.radius == params.radius
    @test floattype(grid) === Float32
end

@testset "Basis-explicit core types" begin
    m = ones(Float64, 4, 3, 2)
    state_dry = @inferred CellState(DryBasis, m; CO2=copy(m) .* 400e-6)
    state_moist = @inferred CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)

    @test mass_basis(state_dry) isa DryBasis
    @test mass_basis(state_moist) isa MoistBasis

    mesh = LatLonMesh(; Nx=4, Ny=3)
    vc = HybridSigmaPressure([0.0, 100.0, 300.0], [0.0, 0.0, 1.0])
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.CPU())
    flux_dry = allocate_face_fluxes(mesh, 2; FT=Float64, basis=DryBasis)
    flux_moist = allocate_face_fluxes(StructuredTopology(), 4, 3, 2; FT=Float64, basis=MoistBasis)

    @test mass_basis(flux_dry) isa DryBasis
    @test mass_basis(flux_moist) isa MoistBasis

    ws = AdvectionWorkspace(state_dry.air_mass)

    @test_throws MethodError apply!(state_dry, flux_moist, grid, FirstOrderUpwindAdvection(), 1800.0; workspace=ws)
end

@testset "Standalone src_v2 runtime smoke test" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.CPU(); FT=FT)

    m = ones(FT, Nx, Ny, Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(StructuredTopology(), Nx, Ny, Nz; FT=FT, basis=DryBasis)

    m0 = total_air_mass(state)
    rm0 = total_mass(state, :CO2)

    model = @inferred TransportModel(state, fluxes, grid, FirstOrderUpwindAdvection())
    sim = Simulation(model; Δt=FT(1800), stop_time=FT(3600))
    run!(sim)

    @test sim.iteration == 2
    @test total_air_mass(sim.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10
end

@testset "Honest metadata-only CubedSphere API" begin
    mesh = CubedSphereMesh(; FT=Float64, Nc=4)
    @test_throws ArgumentError cell_area(mesh, 1)
    @test_throws ArgumentError face_cells(mesh, 1)
end

@testset "Face-connected reduced-Gaussian smoke test" begin
    FT = Float64
    Nz = 2
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.CPU(); FT=FT)

    m = ones(FT, ncells(mesh), Nz)
    state = CellState(DryBasis, copy(m); CO2=copy(m) .* FT(400e-6))
    fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)

    @test fluxes isa FaceIndexedFluxState{DryBasis}
    @test mass_basis(fluxes) isa DryBasis

    m0 = total_air_mass(state)
    rm0 = total_mass(state, :CO2)

    model = @inferred TransportModel(state, fluxes, grid, FirstOrderUpwindAdvection())
    sim = Simulation(model; Δt=FT(1800), stop_time=FT(3600))
    run!(sim)

    @test sim.iteration == 2
    @test total_air_mass(sim.model.state) ≈ m0 atol=eps(FT) * m0 * 10
    @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(FT) * rm0 * 10
end
