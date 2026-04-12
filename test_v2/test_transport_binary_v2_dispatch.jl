#!/usr/bin/env julia

using Test
using JSON3

include(joinpath(@__DIR__, "..", "scripts", "preprocessing", "preprocess_era5_cs_conservative_v2.jl"))
include(joinpath(@__DIR__, "..", "scripts", "preprocessing", "transport_binary_v2_cs_bilinear.jl"))
using .AtmosTransportV2

struct DummyTransportBinaryTarget{FT <: AbstractFloat} <: AbstractTransportBinaryV2Target
    input_path :: String
    output_path :: String
    label :: String
end

target_input_path(target::DummyTransportBinaryTarget) = target.input_path
target_output_path(target::DummyTransportBinaryTarget) = target.output_path
target_float_type(::DummyTransportBinaryTarget{FT}) where FT = FT
target_summary(target::DummyTransportBinaryTarget) = target.label

prepare_transport_binary_v2_target(target::DummyTransportBinaryTarget, reader) =
    (header_windows = reader.header.nwindow, output_path = target.output_path)

collect_transport_binary_v2_windows(::DummyTransportBinaryTarget, ctx, reader) =
    fill("window", min(ctx.header_windows, reader.header.nwindow))

build_transport_binary_v2_header(::DummyTransportBinaryTarget, ctx, reader, windows) =
    Dict(
        "source_windows" => reader.header.nwindow,
        "collected_windows" => length(windows),
        "output_path" => ctx.output_path,
    )

function write_transport_binary_v2_output(::DummyTransportBinaryTarget, ctx, reader, header, windows)
    text = join([
        "summary=$(length(windows))",
        "reader_windows=$(reader.header.nwindow)",
        "output=$(ctx.output_path)",
        "collected=$(header["collected_windows"])",
    ], "\n") * "\n"
    write(ctx.output_path, text)
    return sizeof(text)
end

function write_tiny_structured_transport_binary(path::AbstractString;
                                                FT::Type{<:AbstractFloat} = Float64,
                                                nwindow::Int = 1)
    mesh = LatLonMesh(; FT=FT, Nx=6, Ny=4)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)
    Nx, Ny, Nz = 6, 4, 2

    windows = [begin
        m = fill(FT(1 + 0.1win), Nx, Ny, Nz)
        am = zeros(FT, Nx + 1, Ny, Nz)
        bm = zeros(FT, Nx, Ny + 1, Nz)
        cm = zeros(FT, Nx, Ny, Nz + 1)
        ps = fill(FT(95000 + win), Nx, Ny)
        (; m, am, bm, cm, ps)
    end for win in 1:nwindow]

    write_transport_binary(path, grid, windows;
                           FT=FT,
                           dt_met_seconds=3600.0,
                           half_dt_seconds=1800.0,
                           steps_per_window=2,
                           mass_basis=:moist,
                           source_flux_sampling=:window_start_endpoint,
                           flux_sampling=:window_constant,
                           extra_header=Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return path
end

function read_padded_header_json(path::AbstractString)
    raw = open(path, "r") do io
        read(io, 262144)
    end
    nul = findfirst(==(0x00), raw)
    nul === nothing && error("Expected null-padded JSON header in $path")
    return JSON3.read(String(raw[1:nul-1]))
end

@testset "transport-binary v2 stable API contract" begin
    mktempdir() do dir
        input_path = joinpath(dir, "input.bin")
        output_path = joinpath(dir, "dummy.out")
        write_tiny_structured_transport_binary(input_path; nwindow=1)

        target = DummyTransportBinaryTarget{Float64}(input_path, output_path, "dummy stable target")
        result = run_transport_binary_v2_preprocessor(target)

        @test result.path == output_path
        @test result.nwindow == 1
        @test result.bytes_written == filesize(output_path)
        @test occursin("reader_windows=1", read(output_path, String))
        @test target_summary(target) == "dummy stable target"
    end
end

@testset "build_transport_binary_v2_target stable front door" begin
    target = build_transport_binary_v2_target(
        :cubed_sphere_conservative,
        ["--input", "latlon.bin", "--output", "cs.bin", "--Nc", "24", "--cache-dir", "/tmp/cs-cache"];
        FT=Float32,
    )
    bilinear_target = build_transport_binary_v2_target(
        :cubed_sphere_bilinear,
        ["--input", "latlon.bin", "--output", "cs.bin", "--Nc", "24"];
        FT=Float64,
    )

    @test target isa CubedSphereConservativeTransportBinaryTarget{Float32}
    @test target.Nc == 24
    @test target.cache_dir == "/tmp/cs-cache"
    @test target_summary(target) == "ERA5 LatLon -> C24 (CR.jl conservative regridding)"
    @test bilinear_target isa CubedSphereBilinearTransportBinaryTarget{Float64}
    @test bilinear_target.Nc == 24
    @test target_summary(bilinear_target) == "ERA5 LatLon -> C24 (bilinear cell-center regridding)"
    @test_throws ErrorException build_transport_binary_v2_target(:does_not_exist, String[]; FT=Float64)
end

@testset "cubed-sphere conservative preprocessor tiny end-to-end" begin
    mktempdir() do dir
        input_path = joinpath(dir, "latlon.bin")
        output_path = joinpath(dir, "cs.bin")
        cache_dir = joinpath(dir, "cache")
        write_tiny_structured_transport_binary(input_path; nwindow=1)

        target = build_transport_binary_v2_target(
            :cubed_sphere_conservative,
            ["--input", input_path, "--output", output_path, "--Nc", "4", "--cache-dir", cache_dir];
            FT=Float64,
        )
        result = run_transport_binary_v2_preprocessor(target)
        header = read_padded_header_json(output_path)

        @test isfile(output_path)
        @test result.path == output_path
        @test result.nwindow == 1
        @test result.bytes_written == filesize(output_path)
        @test String(header.grid_type) == "cubed_sphere"
        @test Int(header.nwindow) == 1
        @test Int(header.Nc) == 4
        @test String(header.regrid_method) == "conservative_crjl"
        @test String(header.poisson_balance_method) == "global_cg_graph_laplacian"
    end
end

@testset "cubed-sphere bilinear preprocessor tiny end-to-end" begin
    mktempdir() do dir
        input_path = joinpath(dir, "latlon.bin")
        output_path = joinpath(dir, "cs.bin")
        write_tiny_structured_transport_binary(input_path; nwindow=1)

        target = build_transport_binary_v2_target(
            :cubed_sphere_bilinear,
            ["--input", input_path, "--output", output_path, "--Nc", "4"];
            FT=Float64,
        )
        result = run_transport_binary_v2_preprocessor(target)
        header = read_padded_header_json(output_path)

        @test isfile(output_path)
        @test result.path == output_path
        @test result.nwindow == 1
        @test result.bytes_written == filesize(output_path)
        @test String(header.grid_type) == "cubed_sphere"
        @test Int(header.nwindow) == 1
        @test Int(header.Nc) == 4
        @test String(header.regrid_method) == "bilinear_cell_center"
        @test String(header.poisson_balance_method) == "per_panel_fft_periodic_gnomonic"
    end
end

@testset "preprocessing docs anchor is linked from root docs" begin
    philosophy = read(joinpath(@__DIR__, "..", "docs", "PREPROCESSING_PHILOSOPHY.md"), String)
    quickstart = read(joinpath(@__DIR__, "..", "docs", "QUICKSTART.md"), String)
    meteo = read(joinpath(@__DIR__, "..", "docs", "METEO_PREPROCESSING.md"), String)
    architecture = read(joinpath(@__DIR__, "..", "docs", "ARCHITECTURE.md"), String)
    spectral_dispatch = read(joinpath(@__DIR__, "..", "scripts", "preprocessing", "spectral_transport_binary_v2_dispatch.jl"), String)
    latlon_wrapper = read(joinpath(@__DIR__, "..", "scripts", "preprocessing", "preprocess_era5_latlon_transport_binary_v2.jl"), String)
    reduced_wrapper = read(joinpath(@__DIR__, "..", "scripts", "preprocessing", "preprocess_era5_reduced_gaussian_transport_binary_v2.jl"), String)

    @test occursin("build_transport_binary_v2_target", philosophy)
    @test occursin("cubed_sphere_bilinear", philosophy)
    @test occursin("build_spectral_transport_binary_v2_target", philosophy)
    @test occursin("run_spectral_transport_binary_v2_preprocessor", philosophy)
    @test occursin("CONSERVATIVE_REGRIDDING.md", philosophy)
    @test occursin("BINARY_FORMAT.md", philosophy)
    @test occursin("AbstractSpectralTransportBinaryV2Target", spectral_dispatch)
    @test occursin("build_spectral_transport_binary_v2_target", spectral_dispatch)
    @test occursin("run_spectral_transport_binary_v2_preprocessor", spectral_dispatch)
    @test occursin("LatLonSpectralTransportBinaryV2Target", latlon_wrapper)
    @test occursin("ReducedGaussianSpectralTransportBinaryV2Target", reduced_wrapper)
    @test occursin("PREPROCESSING_PHILOSOPHY.md", quickstart)
    @test occursin("PREPROCESSING_PHILOSOPHY.md", meteo)
    @test occursin("PREPROCESSING_PHILOSOPHY.md", architecture)
end
