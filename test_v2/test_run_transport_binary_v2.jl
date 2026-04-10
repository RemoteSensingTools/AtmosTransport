#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
include(joinpath(@__DIR__, "..", "scripts", "run_transport_binary_v2.jl"))

function write_sequence_binary(path::AbstractString; FT::Type{<:AbstractFloat}=Float64, scale::Real=1)
    Nx, Ny, Nz = 4, 3, 2
    mesh = AtmosTransportV2.LatLonMesh(; FT=FT, Nx=Nx, Ny=Ny)
    vertical = AtmosTransportV2.HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosTransportV2.AtmosGrid(mesh, vertical, AtmosTransportV2.CPU(); FT=FT)

    windows = [begin
        m = fill(FT(scale), Nx, Ny, Nz)
        am = zeros(FT, Nx + 1, Ny, Nz)
        bm = zeros(FT, Nx, Ny + 1, Nz)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        ps = fill(FT(95000), Nx, Ny)
        qv_start = fill(FT(0.01), Nx, Ny, Nz)
        qv_end = fill(FT(0.01), Nx, Ny, Nz)
        dam = zeros(FT, Nx + 1, Ny, Nz)
        dbm = zeros(FT, Nx, Ny + 1, Nz)
        dcm = zeros(FT, Nx, Ny, Nz + 1)
        dm = zeros(FT, Nx, Ny, Nz)
        (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
    end]

    AtmosTransportV2.write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=450.0,
                           steps_per_window=1,
                           mass_basis=:moist,
                           source_flux_sampling=:window_start_endpoint,
                           flux_sampling=:window_constant,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.5,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return nothing
end

@testset "transport-binary sequence runner" begin
    mktempdir() do dir
        path1 = joinpath(dir, "day1.bin")
        path2 = joinpath(dir, "day2.bin")
        write_sequence_binary(path1; scale=1)
        write_sequence_binary(path2; scale=1)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path1, path2]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("scheme" => "upwind", "tracer_name" => "CO2", "start_window" => 1),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        model = run_sequence([path1, path2], cfg)
        @test AtmosTransportV2.total_air_mass(model.state) ≈ 24.0 atol=eps(Float64) * 100
        @test AtmosTransportV2.total_mass(model.state, :CO2) ≈ 24.0 * 4.0e-4 atol=eps(Float64) * 100
    end
end
