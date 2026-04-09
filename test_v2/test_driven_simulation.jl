#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

function write_driven_latlon_binary(path::AbstractString; FT::Type{<:AbstractFloat}=Float64)
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

    windows = [
        begin
            m = ones(FT, Nx, Ny, Nz)
            am = zeros(FT, Nx + 1, Ny, Nz)
            bm = zeros(FT, Nx, Ny + 1, Nz)
            cm = zeros(FT, Nx, Ny, Nz + 1)
            ps = fill(FT(95000 + 100win), Nx, Ny)
            qv_start = fill(FT(0.01win), Nx, Ny, Nz)
            qv_end = fill(FT(0.01win + 0.01), Nx, Ny, Nz)
            dam = fill(FT(0.2win), Nx + 1, Ny, Nz)
            dbm = fill(FT(0.4win), Nx, Ny + 1, Nz)
            dcm = fill(FT(0.6win), Nx, Ny, Nz + 1)
            dm = fill(FT(0.8win), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
        end for win in 1:2
    ]

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist)
    return grid
end

@testset "DrivenSimulation window-forcing runtime" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_driven_latlon_binary(path; FT=Float64)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        grid = driver_grid(driver)
        @test grid.horizontal isa LatLonMesh
        @test nx(grid.horizontal) == nx(grid_ref.horizontal)
        @test ny(grid.horizontal) == ny(grid_ref.horizontal)

        state = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model = @inferred TransportModel(state, fluxes, grid, UpwindAdvection())
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=2)

        @test sim.Δt == 1800.0
        @test sim.window_dt == 3600.0
        @test sim.steps_per_window == 2
        @test window_index(sim) == 1
        @test substep_index(sim) == 1
        @test current_qv(sim) !== nothing
        @test all(current_qv(sim) .== 0.01)

        m0 = total_air_mass(sim.model.state)
        rm0 = total_mass(sim.model.state, :CO2)

        state_window = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2=fill(400e-6, 4, 3, 2))
        fluxes_window = allocate_face_fluxes(grid.horizontal, 2; FT=Float64, basis=MoistBasis)
        model_window = TransportModel(state_window, fluxes_window, grid, UpwindAdvection())
        sim_window = DrivenSimulation(model_window, driver; start_window=1, stop_window=2)
        run_window!(sim_window)
        @test sim_window.iteration == 2
        @test sim_window.time == 3600.0
        @test window_index(sim_window) == 1

        step!(sim)
        @test sim.iteration == 1
        @test sim.time == 1800.0
        @test window_index(sim) == 1
        @test substep_index(sim) == 2
        @test all(isapprox.(sim.model.fluxes.am, 0.05; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.bm, 0.1; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.cm, 0.15; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.expected_air_mass, 1.2; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0125; atol=eps(Float64) * 10))

        step!(sim)
        @test sim.iteration == 2
        @test sim.time == 3600.0
        @test window_index(sim) == 1
        @test substep_index(sim) == 1
        @test all(isapprox.(sim.model.fluxes.am, 0.15; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.bm, 0.3; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.cm, 0.45; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.expected_air_mass, 1.6; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0175; atol=eps(Float64) * 10))

        step!(sim)
        @test sim.iteration == 3
        @test sim.time == 5400.0
        @test window_index(sim) == 2
        @test substep_index(sim) == 2
        @test all(isapprox.(sim.model.fluxes.am, 0.1; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.bm, 0.2; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.model.fluxes.cm, 0.3; atol=eps(Float64) * 10))
        @test all(isapprox.(sim.expected_air_mass, 1.4; atol=eps(Float64) * 10))
        @test all(isapprox.(current_qv(sim), 0.0225; atol=eps(Float64) * 10))

        run!(sim)
        @test sim.iteration == 4
        @test sim.time == 7200.0
        @test window_index(sim) == 2
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
        @test_throws ArgumentError step!(sim)

        close(driver)
    end
end
