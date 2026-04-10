#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

function write_test_transport_binary_reduced(path::AbstractString; FT::Type{<:AbstractFloat}=Float64, source_flux_sampling::Symbol=:window_start_endpoint, binary_kwargs...)
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)

    windows = [
        begin
            m = reshape(FT.(1:(ncell * nlevel)) .* FT(win), ncell, nlevel)
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
                           mass_basis=:moist,
                           source_flux_sampling=source_flux_sampling,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ),
                           binary_kwargs...)
    return grid
end

function write_test_transport_binary_latlon(path::AbstractString;
                                            FT::Type{<:AbstractFloat}=Float64,
                                            source_flux_sampling::Symbol=:window_start_endpoint,
                                            flux_sampling::Symbol=:window_constant,
                                            include_poisson_metadata::Bool=true,
                                            binary_kwargs...)
    Nx, Ny, Nz = 6, 4, 3
    mesh = LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300, 1000], FT[0, 0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

    windows = [
        begin
            m = reshape(FT.(1:(Nx * Ny * Nz)) .* FT(win), Nx, Ny, Nz)
            am = zeros(FT, Nx + 1, Ny, Nz)
            bm = zeros(FT, Nx, Ny + 1, Nz)
            cm = zeros(FT, Nx, Ny, Nz + 1)
            ps = fill(FT(95000 + 1000win), Nx, Ny)
            qv_start = fill(FT(0.002win), Nx, Ny, Nz)
            qv_end = fill(FT(0.002win + 0.0005), Nx, Ny, Nz)
            dam = fill(FT(0.1win), Nx + 1, Ny, Nz)
            dbm = fill(FT(0.2win), Nx, Ny + 1, Nz)
            dcm = fill(FT(0.3win), Nx, Ny, Nz + 1)
            dm = fill(FT(0.4win), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
        end for win in 1:2
    ]

    extra_header = include_poisson_metadata ? Dict(
        "poisson_balance_target_scale" => 0.25,
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
    ) : Dict{String, Any}()

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist,
                           source_flux_sampling=source_flux_sampling,
                           flux_sampling=flux_sampling,
                           extra_header=extra_header,
                           binary_kwargs...)
    return grid
end

@testset "TransportBinaryReader structured lat-lon path" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_test_transport_binary_latlon(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :latlon
        @test horizontal_topology(reader) == :structureddirectional
        @test window_count(reader) == 2
        @test mass_basis(reader) == :moist
        @test has_qv(reader)
        @test has_qv_endpoints(reader)
        @test has_flux_delta(reader)
        @test source_flux_sampling(reader) == :window_start_endpoint
        @test air_mass_sampling(reader) == :window_start_endpoint
        @test flux_sampling(reader) == :window_constant
        @test flux_kind(reader) == :substep_mass_amount
        @test humidity_sampling(reader) == :window_endpoints
        @test delta_semantics(reader) == :forward_window_endpoint_difference
        @test reader.header.poisson_balance_target_scale == 0.25
        @test reader.header.poisson_balance_target_semantics == "forward_window_mass_difference / (2 * steps_per_window)"

        header_repr = sprint(show, reader.header)
        @test occursin("TransportBinaryHeader", header_repr)
        @test occursin("qv_start/qv_end", header_repr)
        @test occursin("substep_mass_amount", header_repr)

        reader_repr = sprint(show, reader)
        @test occursin("TransportBinaryReader", reader_repr)
        @test occursin("latlon/structureddirectional", reader_repr)

        grid = load_grid(reader; FT=Float64, arch=CPU())
        @test grid.horizontal isa LatLonMesh
        @test nx(grid.horizontal) == nx(grid_ref.horizontal)
        @test ny(grid.horizontal) == ny(grid_ref.horizontal)

        m, ps, fluxes = load_window!(reader, 1)
        @test size(m) == (6, 4, 3)
        @test size(ps) == (6, 4)
        @test size(fluxes.am) == (7, 4, 3)
        @test size(fluxes.bm) == (6, 5, 3)
        @test size(fluxes.cm) == (6, 4, 4)
        @test mass_basis(fluxes) isa MoistBasis

        qv_pair = load_qv_pair_window!(reader, 1)
        @test qv_pair !== nothing
        @test size(qv_pair.qv_start) == (6, 4, 3)
        @test size(qv_pair.qv_end) == (6, 4, 3)
        @test all(qv_pair.qv_start .== 0.002)
        @test all(qv_pair.qv_end .== 0.0025)

        deltas = load_flux_delta_window!(reader, 1)
        @test deltas !== nothing
        @test size(deltas.dam) == (7, 4, 3)
        @test size(deltas.dbm) == (6, 5, 3)
        @test size(deltas.dcm) == (6, 4, 4)
        @test size(deltas.dm) == (6, 4, 3)
        @test all(deltas.dam .== 0.1)
        @test all(deltas.dbm .== 0.2)
        @test all(deltas.dcm .== 0.3)
        @test all(deltas.dm .== 0.4)

        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        @test steps_per_window(driver) == 2
        @test air_mass_basis(driver) == :moist
        @test driver_grid(driver).horizontal isa LatLonMesh

        driver_repr = sprint(show, driver)
        @test occursin("TransportBinaryDriver", driver_repr)
        @test occursin("steps/window=2", driver_repr)

        window = load_transport_window(driver, 1)
        @test window isa StructuredTransportWindow{MoistBasis}
        @test has_humidity_endpoints(window)
        @test has_flux_delta(window)

        flux_interp = allocate_face_fluxes(driver_grid(driver).horizontal, 3; FT=Float64, basis=MoistBasis)
        interpolate_fluxes!(flux_interp, window, 0.5)
        @test all(flux_interp.am .== 0.05)
        @test all(flux_interp.bm .== 0.1)
        @test all(flux_interp.cm .== 0.15)

        m_interp = similar(window.air_mass)
        expected_air_mass!(m_interp, window, 0.5)
        @test all(m_interp .== window.air_mass .+ 0.2)

        qv_interp = similar(window.qv_start)
        interpolate_qv!(qv_interp, window, 0.5)
        @test all(isapprox.(qv_interp, 0.00225; atol=eps(Float64) * 10))

        close(driver)

        state = CellState(MoistBasis, copy(m); CO2=copy(m) .* 400e-6)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = Simulation(model; Δt=1800.0, stop_time=3600.0)
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :CO2)
        run!(sim)
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
    end
end

@testset "TransportBinaryReader reduced-Gaussian path" begin
    mktemp() do path, io
        close(io)
        grid_ref = write_test_transport_binary_reduced(path; FT=Float64)

        reader = TransportBinaryReader(path; FT=Float64)
        @test grid_type(reader) == :reduced_gaussian
        @test horizontal_topology(reader) == :faceindexed
        @test window_count(reader) == 2
        @test mass_basis(reader) == :moist
        @test has_qv(reader)
        @test has_qv_endpoints(reader)
        @test !has_flux_delta(reader)
        @test source_flux_sampling(reader) == :window_start_endpoint
        @test delta_semantics(reader) == :none

        reduced_reader_repr = sprint(show, reader)
        @test occursin("reduced_gaussian/faceindexed", reduced_reader_repr)
        @test occursin("qv_start/qv_end", reduced_reader_repr)

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
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = Simulation(model; Δt=1800.0, stop_time=3600.0)
        m0 = total_air_mass(state)
        rm0 = total_mass(state, :CO2)
        run!(sim)
        @test total_air_mass(sim.model.state) ≈ m0 atol=eps(Float64) * m0 * 10
        @test total_mass(sim.model.state, :CO2) ≈ rm0 atol=eps(Float64) * rm0 * 10
    end
end


@testset "TransportBinaryDriver timing semantics validation" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64, flux_kind=:mass_rate)
        @test_throws ArgumentError TransportBinaryDriver(path; FT=Float64, arch=CPU())
    end
end

@testset "TransportBinaryDriver accepts legacy delta-bearing headers without poisson metadata" begin
    mktemp() do path, io
        close(io)
        write_test_transport_binary_latlon(path; FT=Float64, include_poisson_metadata=false)
        driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
        @test total_windows(driver) == 2
        @test window_dt(driver) == 3600.0
        close(driver)
    end
end
