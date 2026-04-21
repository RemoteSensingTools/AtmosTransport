#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

const _REALISTIC_AIR_MASS_KG = 1e16

function _make_convection_grid(; FT = Float64, Nx = 4, Ny = 3, Nz = 5)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    return AtmosGrid(mesh, vertical, CPU(); FT = FT)
end

function _make_convection_model(; FT = Float64, Nx = 4, Ny = 3, Nz = 5,
                                 convection::AbstractConvectionOperator = NoConvection(),
                                 convection_forcing::ConvectionForcing = ConvectionForcing())
    grid = _make_convection_grid(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    tracer = zeros(FT, Nx, Ny, Nz)
    tracer[:, :, Nz] .= FT(1e-6) .* air_mass[:, :, Nz]
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          convection = convection,
                          convection_forcing = convection_forcing)
end

function _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    cmfmc = zeros(FT, Nx, Ny, Nz + 1)
    cmfmc[:, :, 4] .= peak * FT(0.5)
    cmfmc[:, :, 3] .= peak
    cmfmc[:, :, 2] .= peak * FT(0.5)

    dtrain = zeros(FT, Nx, Ny, Nz)
    dtrain[:, :, 1] .= top_detrain
    return ConvectionForcing(cmfmc, dtrain, nothing)
end

function _make_rg_cmfmc_forcing(FT, ncell, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    cmfmc = zeros(FT, ncell, Nz + 1)
    cmfmc[:, 4] .= peak * FT(0.5)
    cmfmc[:, 3] .= peak
    cmfmc[:, 2] .= peak * FT(0.5)

    dtrain = zeros(FT, ncell, Nz)
    dtrain[:, 1] .= top_detrain
    return ConvectionForcing(cmfmc, dtrain, nothing)
end

function _make_rg_convection_model(; FT = Float64, Nz = 5,
                                    convection::AbstractConvectionOperator = NoConvection(),
                                    convection_forcing::ConvectionForcing = ConvectionForcing())
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    ncell = ncells(mesh)

    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), ncell, Nz)
    tracer = zeros(FT, ncell, Nz)
    tracer[:, Nz] .= FT(1e-6) .* air_mass[:, Nz]
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
    return TransportModel(state, fluxes, grid, UpwindScheme();
                          convection = convection,
                          convection_forcing = convection_forcing)
end

struct _ConvectionWindowDriver{FT, GridT, WindowT} <: AbstractMetDriver
    grid    :: GridT
    windows :: Vector{WindowT}
    dt      :: FT
    steps   :: Int
end

AtmosTransport.total_windows(driver::_ConvectionWindowDriver) = length(driver.windows)
AtmosTransport.window_dt(driver::_ConvectionWindowDriver) = driver.dt
AtmosTransport.steps_per_window(driver::_ConvectionWindowDriver) = driver.steps
AtmosTransport.load_transport_window(driver::_ConvectionWindowDriver, win::Int) = driver.windows[win]
AtmosTransport.driver_grid(driver::_ConvectionWindowDriver) = driver.grid
AtmosTransport.air_mass_basis(::_ConvectionWindowDriver) = :dry
AtmosTransport.MetDrivers.flux_interpolation_mode(::_ConvectionWindowDriver) = :constant
AtmosTransport.MetDrivers.supports_native_vertical_flux(::_ConvectionWindowDriver) = true
AtmosTransport.supports_convection(::_ConvectionWindowDriver) = true

function _make_convection_window_driver(; FT = Float64)
    grid = _make_convection_grid(FT = FT)
    Nx, Ny, Nz = 4, 3, 5
    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    ps = fill(FT(95_000), Nx, Ny)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)

    forcing_a = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    forcing_b = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.5), top_detrain = FT(0.1))

    window_a = StructuredTransportWindow(air_mass, ps, fluxes; convection = forcing_a)
    window_b = StructuredTransportWindow(air_mass, ps, fluxes; convection = forcing_b)
    driver = _ConvectionWindowDriver{FT, typeof(grid), typeof(window_a)}(
        grid, [window_a, window_b], FT(1800), 1)
    return driver, forcing_a, forcing_b
end

@testset "TransportModel convection runtime" begin
    @testset "default model carries NoConvection and no convection workspace" begin
        model = _make_convection_model()
        @test model.convection isa NoConvection
        @test model.workspace.convection_ws === nothing
    end

    @testset "with_convection installs structured CMFMC workspace" begin
        base = _make_convection_model()
        updated = with_convection(base, CMFMCConvection())

        @test updated.convection isa CMFMCConvection
        @test updated.convection_forcing === base.convection_forcing
        @test updated.state === base.state
        @test updated.fluxes === base.fluxes
        @test updated.grid === base.grid
        @test updated.advection === base.advection
        @test updated.chemistry === base.chemistry
        @test updated.diffusion === base.diffusion
        @test updated.emissions === base.emissions
        @test updated.workspace !== base.workspace
        @test updated.workspace.advection_ws === base.workspace.advection_ws
        @test updated.workspace.convection_ws isa CMFMCWorkspace
        @test updated.workspace.dz_scratch === updated.workspace.advection_ws.dz_scratch
    end

    @testset "step! with default NoConvection stays bit-exact" begin
        model_a = _make_convection_model()
        model_b = _make_convection_model()

        for _ in 1:3
            step!(model_a, 1800.0)
            step!(model_b, 1800.0)
        end

        @test model_a.state.tracers_raw == model_b.state.tracers_raw
    end

    @testset "step! with CMFMCConvection redistributes vertically and conserves mass" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 5
        forcing = _make_cmfmc_forcing(FT, Nx, Ny, Nz)

        model_ctrl = _make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
        model_conv = _make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz)
        model_conv = with_convection(model_conv, CMFMCConvection())
        model_conv = with_convection_forcing(model_conv, forcing)

        rm_before = copy(model_conv.state.tracers_raw)
        total_before = sum(rm_before)

        step!(model_ctrl, FT(1800))
        step!(model_conv, FT(1800))

        @test model_ctrl.state.tracers_raw == rm_before
        @test model_conv.state.tracers_raw != rm_before
        @test sum(model_conv.state.tracers_raw) ≈ total_before rtol = 1e-12
        @test model_conv.state.tracers.CO2[1, 1, Nz] < rm_before[1, 1, Nz, 1]
        @test maximum(model_conv.state.tracers.CO2[:, :, 1:(Nz - 1)]) > 0
    end

    @testset "step! reads convection forcing from the model" begin
        FT = Float64
        Nx, Ny, Nz = 4, 3, 5

        weak = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
        strong = _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.5), top_detrain = FT(0.1))

        model_weak = with_convection(_make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz), CMFMCConvection())
        model_strong = with_convection(_make_convection_model(FT = FT, Nx = Nx, Ny = Ny, Nz = Nz), CMFMCConvection())
        model_weak = with_convection_forcing(model_weak, weak)
        model_strong = with_convection_forcing(model_strong, strong)

        step!(model_weak, FT(1800))
        step!(model_strong, FT(1800))

        @test model_weak.state.tracers_raw != model_strong.state.tracers_raw
    end

    @testset "ReducedGaussian step! with CMFMCConvection redistributes vertically" begin
        FT = Float64
        Nz = 5
        ncell = ncells(ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT))
        forcing = _make_rg_cmfmc_forcing(FT, ncell, Nz; peak = FT(0.02), top_detrain = FT(0.01))

        model = with_convection(_make_rg_convection_model(FT = FT, Nz = Nz), CMFMCConvection())
        model = with_convection_forcing(model, forcing)
        rm_before = copy(model.state.tracers_raw)

        step!(model, FT(1800))

        @test abs(sum(model.state.tracers_raw) - sum(rm_before)) / sum(rm_before) < 1e-12
        @test model.state.tracers_raw != rm_before
    end
end

@testset "DrivenSimulation convection runtime" begin
    FT = Float64
    driver, forcing_a, forcing_b = _make_convection_window_driver(FT = FT)

    state = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                      CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes = allocate_face_fluxes(driver.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                           convection = CMFMCConvection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 2)

    @test sim.model.workspace.convection_ws isa CMFMCWorkspace
    @test sim.model.convection_forcing !== forcing_a
    @test sim.model.convection_forcing.cmfmc !== forcing_a.cmfmc
    @test sim.model.convection_forcing.cmfmc == forcing_a.cmfmc

    step!(sim)
    @test sim.model.workspace.convection_ws.cached_n_sub[] == 1
    @test sim.model.workspace.convection_ws.cache_valid[] == true

    step!(sim)
    @test window_index(sim) == 2
    @test sim.model.convection_forcing.cmfmc == forcing_b.cmfmc
    @test sim.model.workspace.convection_ws.cached_n_sub[] > 1
    @test sim.model.workspace.convection_ws.cache_valid[] == true
end

@testset "DrivenSimulation keeps convection runtime on model FT" begin
    FT = Float32
    driver, forcing_a, _ = _make_convection_window_driver(FT = FT)

    state = CellState(fill(FT(_REALISTIC_AIR_MASS_KG), 4, 3, 5);
                      CO2 = fill(FT(1e-6 * _REALISTIC_AIR_MASS_KG), 4, 3, 5))
    fluxes = allocate_face_fluxes(driver.grid.horizontal, 5; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                           convection = CMFMCConvection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)

    @test typeof(sim.Δt) === FT
    @test typeof(sim.window_dt) === FT
    @test eltype(sim.model.convection_forcing.cmfmc) === FT
    @test sim.model.convection_forcing.cmfmc == forcing_a.cmfmc
end
