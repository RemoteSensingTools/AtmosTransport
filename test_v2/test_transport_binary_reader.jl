#!/usr/bin/env julia

using Test
using JSON3

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

function write_test_transport_binary(path::AbstractString; FT::Type{<:AbstractFloat}=Float64)
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)

    windows = [
        begin
            m = reshape(FT.(1:(ncell*nlevel)) .* FT(win), ncell, nlevel)
            hflux = zeros(FT, nface_h, nlevel)
            cm = zeros(FT, ncell, nlevel + 1)
            ps = fill(FT(90000 + 1000win), ncell)
            qv_start = fill(FT(0.01win), ncell, nlevel)
            qv_end = fill(FT(0.01win + 0.001), ncell, nlevel)
            (; m, hflux, cm, ps, qv_start, qv_end)
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

@testset "TransportBinaryReader reduced-Gaussian path" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_test_transport_binary(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :reduced_gaussian
        @test horizontal_topology(reader) == :faceindexed
        @test window_count(reader) == 2
        @test mass_basis(reader) == :moist
        @test has_qv(reader)
        @test has_qv_endpoints(reader)

        grid = load_grid(reader; FT=Float64, arch=CPU())
        @test grid.horizontal isa ReducedGaussianMesh
        @test ncells(grid.horizontal) == ncells(grid_ref.horizontal)
        @test nfaces(grid.horizontal) == reader.header.nface_h

        m, ps, fluxes = load_window!(reader, 1)
        @test size(m) == (8, 2)
        @test size(ps) == (8,)
        @test size(fluxes.horizontal_flux) == (reader.header.nface_h, 2)
        @test size(fluxes.cm) == (8, 3)
        @test mass_basis(fluxes) isa MoistBasis

        qv_pair = load_qv_pair_window!(reader, 1)
        @test qv_pair !== nothing
        @test all(qv_pair.qv_start .== 0.01)
        @test all(qv_pair.qv_end .== 0.011)

        state = CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)
        model = TransportModel(state, fluxes, grid, FirstOrderUpwindAdvection())
        sim = Simulation(model; Δt=1800.0, stop_time=3600.0)
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :CO2)
        run!(sim)
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
    end
end
