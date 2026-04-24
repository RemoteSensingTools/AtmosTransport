#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "scripts", "run_transport_binary.jl"))
using .AtmosTransport

const _RUNTIME_RECIPE_AIR_MASS = 1e16

function _make_runtime_recipe_tm5_fields(::Type{FT}, Nx, Ny, Nz) where FT
    entu = zeros(FT, Nx, Ny, Nz)
    detu = zeros(FT, Nx, Ny, Nz)
    entd = zeros(FT, Nx, Ny, Nz)
    detd = zeros(FT, Nx, Ny, Nz)
    return (; entu, detu, entd, detd)
end

function write_runtime_recipe_binary(path::AbstractString; FT::Type{<:AbstractFloat} = Float64)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m = fill(FT(_RUNTIME_RECIPE_AIR_MASS), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ps = fill(FT(95_000), Nx, Ny)

    windows = [(
        m = m,
        am = am,
        bm = bm,
        cm = cm,
        ps = ps,
        tm5_fields = _make_runtime_recipe_tm5_fields(FT, Nx, Ny, Nz),
    )]

    write_transport_binary(path, grid, windows;
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 1,
                           mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return nothing
end

@testset "run_transport_binary builds structured runtime recipes" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "ppm"),
            "diffusion" => Dict("kind" => "constant", "value" => 2.0),
            "convection" => Dict("kind" => "tm5"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        # Plan 40 Commit 6a: `make_model` is gone; run through the
        # unified library entry point. The model is returned at the end of
        # the loop, so for a 1-window input we can inspect it directly.
        model = run_driven_simulation(cfg)

        @test model.advection isa PPMScheme
        @test model.diffusion isa ImplicitVerticalDiffusion
        @test model.convection isa TM5Convection
        @test model.workspace.convection_ws isa TM5Workspace{Float64}
    end
end

@testset "run_transport_binary runtime recipe wiring" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "advection" => Dict("scheme" => "ppm"),
            "convection" => Dict("kind" => "tm5"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        model = run_driven_simulation(cfg)

        @test model.convection_forcing.tm5_fields !== nothing
        @test total_air_mass(model.state) ≈ 60 * _RUNTIME_RECIPE_AIR_MASS rtol = 1e-12
        @test total_mass(model.state, :CO2) ≈ 60 * _RUNTIME_RECIPE_AIR_MASS * 4.0e-4 rtol = 1e-12
    end
end

@testset "run_transport_binary rejects unsupported recipe capabilities" begin
    mktempdir() do dir
        path = joinpath(dir, "tm5_runtime.bin")
        write_runtime_recipe_binary(path)

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [path]),
            "numerics" => Dict("float_type" => "Float64"),
            "run" => Dict("start_window" => 1),
            "convection" => Dict("kind" => "cmfmc"),
            "init" => Dict("kind" => "uniform", "background" => 4.0e-4),
        )

        @test_throws ArgumentError run_driven_simulation(cfg)
    end
end
