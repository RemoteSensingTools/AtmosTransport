#!/usr/bin/env julia

using Test
using Dates
using TOML

include(joinpath(@__DIR__, "..", "scripts", "preprocessing", "preprocess_spectral_v4_binary.jl"))

@testset "Preprocessor setup helpers" begin
    cfg = TOML.parsefile(joinpath(@__DIR__, "..", "config", "preprocessing", "era5_spectral_v4.toml"))

    grid = build_target_geometry(cfg["grid"], Float64)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range, settings.min_dp, cfg["grid"])

    @test parse_day_filter(["config.toml"]) === nothing
    @test parse_day_filter(["config.toml", "--day", "2021-12-01"]) == Date(2021, 12, 1)

    @test grid isa LatLonTargetGeometry
    @test nlon(grid) == 720
    @test nlat(grid) == 361
    @test target_spectral_truncation(grid) == 359
    @test occursin("TM5 convention", target_summary(grid))

    @test settings.output_float_type == Float32
    @test settings.mass_basis == :moist
    @test settings.T_target == 359

    @test vertical.Nz_native == 137
    @test vertical.Nz > 0

    counts = window_element_counts(grid, vertical.Nz; include_qv=false)
    byte_sizes = window_byte_sizes(counts, settings.output_float_type, 24)
    @test counts.n_qv == 0
    @test byte_sizes.bytes_per_window > 0
    @test byte_sizes.total_bytes > HEADER_SIZE
end

@testset "Unsupported reduced Gaussian preprocessing path" begin
    mesh = ReducedGaussianMesh([-45.0, 0.0, 45.0], [4, 8, 4]; FT=Float64)
    grid = ReducedGaussianTargetGeometry{Float64, typeof(mesh)}(
        mesh,
        "dummy_native_q.grib",
        2,
        [4, 8, 4],
        [-45.0, 0.0, 45.0],
        [ring_longitudes(mesh, j) for j in 1:nrings(mesh)],
    )

    err = try
        ensure_supported_target(grid)
        nothing
    catch caught
        caught
    end

    @test err isa ErrorException
    @test occursin("native reduced-Gaussian", sprint(showerror, err))
end
