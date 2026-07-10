#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const MD = AT.MetDrivers
const TapeMod = AT.Tape
const DD = AT.DataDownloads
const PP = AT.Preprocessing

using .AtmosTransport.Architectures: CPU
using .AtmosTransport.Grids: AtmosGrid, HybridSigmaPressure, ncells, nfaces

function _rg_fixture(::Type{FT}=Float32) where FT
    mesh = AT.ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT=FT)
    window = (
        m = ones(FT, ncells(mesh), 1),
        hflux = zeros(FT, nfaces(mesh), 1),
        cm = zeros(FT, ncells(mesh), 2),
        ps = fill(FT(90_000), ncells(mesh)),
    )
    return grid, window
end

function _open_rg(path, nwindow=1; extra_header=Dict{String,Any}())
    grid, window = _rg_fixture()
    writer = MD.open_streaming_transport_binary(
        path, grid, nwindow, window;
        FT=Float32, header_bytes=4096, steps_per_window=1,
        source_flux_sampling=:window_start_endpoint,
        humidity_sampling=:none, delta_semantics=:none, mass_basis=:dry,
        extra_header=extra_header,
    )
    return writer, window
end

@testset "download verification sidecars preserve safe resume semantics" begin
    mktempdir() do dir
        path = joinpath(dir, "download.dat")
        write(path, Vector{UInt8}(codeunits("complete-payload")))
        env = DD.PythonEnvironment("python3", false, false, false, false,
                                   false, false)
        protocol = DD.CDSProtocol(env)
        task = DD.DownloadTask("fixture", "dataset/request", path,
                               Dict{String,Any}("variable" => "co2"), 1.0)

        @test DD._existing_task_status(task, protocol) == (:unverifiable, 0)
        manifest_path = DD._write_download_manifest(task, protocol)
        @test isfile(manifest_path)
        @test DD._existing_task_status(task, protocol) ==
              (:verified, filesize(path))
        different_environment = DD.CDSProtocol(DD.PythonEnvironment(
            "/different/python", true, true, true, true, true, true))
        @test DD._existing_task_status(task, different_environment) ==
              (:verified, filesize(path))

        changed_request = DD.DownloadTask(
            "fixture", "dataset/request", path,
            Dict{String,Any}("variable" => "temperature"), 1.0)
        @test first(DD._existing_task_status(changed_request, protocol)) == :corrupt

        bytes = read(path)
        bytes[1] = bytes[1] == 0x00 ? 0x01 : bytes[1] - 0x01
        write(path, bytes)
        @test first(DD._existing_task_status(task, protocol)) == :corrupt
    end
end

@testset "generation fingerprints cover source and preprocessing settings" begin
    provenance = (script_path="preprocess.jl", script_mtime=1.0,
                  git_commit="abc123", git_dirty=false,
                  creation_time="ignored")
    kwargs = (spectral_resolution=127, source_paths=String[],
              next_day_hour0=nothing, provenance=provenance)
    base = PP.generation_fingerprint(
        ; settings=(T_target=127, source_dir="a"), kwargs...)
    changed_resolution = PP.generation_fingerprint(
        ; settings=(T_target=255, source_dir="a"), kwargs...)
    changed_source = PP.generation_fingerprint(
        ; settings=(T_target=127, source_dir="b"), kwargs...)
    @test base != changed_resolution
    @test base != changed_source
end

@testset "streaming binary writes fail closed" begin
    mktempdir() do dir
        @testset "generic windows retain the sample shape contract" begin
            path = joinpath(dir, "rg-shape.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-stream"))
            write(path, sentinel)
            writer, window = _open_rg(path)
            bad = merge(window, (; m=zeros(Float32, length(window.m))))
            @test_throws DimensionMismatch MD.write_streaming_window!(writer, bad)
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test read(path) == sentinel
            @test !ispath(writer.staging_path)
        end

        @testset "CS validates arguments, panel count, and every panel shape" begin
            path = joinpath(dir, "cs-shape.bin")
            Nc, npanel, Nz = 2, 6, 1
            vc = HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
            writer = MD.open_streaming_cs_transport_binary(
                path, Nc, npanel, Nz, 1, vc;
                FT=Float32, header_bytes=4096, steps_per_window=1,
                mass_basis=:dry,
            )
            window = (
                m=ntuple(_ -> ones(Float32, Nc, Nc, Nz), npanel),
                am=ntuple(_ -> zeros(Float32, Nc + 1, Nc, Nz), npanel),
                bm=ntuple(_ -> zeros(Float32, Nc, Nc + 1, Nz), npanel),
                cm=ntuple(_ -> zeros(Float32, Nc, Nc, Nz + 1), npanel),
                ps=ntuple(_ -> fill(90_000f0, Nc, Nc), npanel),
            )
            @test_throws DimensionMismatch MD.write_streaming_cs_window!(writer, window, Nc + 1, npanel)
            bad_panels = Base.setindex(window.m, zeros(Float32, Nc + 1, Nc, Nz), 3)
            @test_throws DimensionMismatch MD.write_streaming_cs_window!(
                writer, merge(window, (; m=bad_panels)), Nc, npanel)
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test !ispath(path)
            @test !ispath(writer.staging_path)
        end

        @testset "extra metadata cannot rewrite structural fields" begin
            path = joinpath(dir, "override.bin")
            @test_throws ArgumentError _open_rg(path, 1; extra_header=Dict("ncell" => 999))
            @test !ispath(path)
        end

        @testset "header layout metadata is internally consistent" begin
            path = joinpath(dir, "header-contract.bin")
            writer, window = _open_rg(path)
            header = deepcopy(writer.header)
            for (key, value) in (("float_type", "Float128"),
                                 ("mass_basis", "unknown"),
                                 ("elems_per_window", header["elems_per_window"] + 1),
                                 ("dt_met_seconds", Inf),
                                 ("include_qv", true),
                                 ("nface_h", header["nface_h"] + 1))
                bad = deepcopy(header)
                bad[key] = value
                @test_throws ArgumentError MD.validate_transport_contract!(bad)
            end
            bad = deepcopy(header)
            push!(bad["payload_sections"], "qv_start")
            @test_throws ArgumentError MD.validate_transport_contract!(bad)
            MD.write_streaming_window!(writer, window)
            MD.close_streaming_transport_binary!(writer)
        end

        @testset "failed final header publication preserves the destination" begin
            path = joinpath(dir, "header-publication.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-stream"))
            write(path, sentinel)
            writer, window = _open_rg(path)
            MD.write_streaming_window!(writer, window)
            writer.header["oversized_metadata"] = "x"^writer.header_bytes
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test read(path) == sentinel
            @test !ispath(writer.staging_path)
        end

        @testset "eager writes replace the destination only after success" begin
            path = joinpath(dir, "atomic-eager.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-artifact"))
            write(path, sentinel)
            grid, window = _rg_fixture()
            bad = merge(window, (; m=fill("not-a-number", size(window.m))))
            @test_throws Exception MD.write_transport_binary(
                path, grid, [bad];
                FT=Float32, header_bytes=4096, steps_per_window=1,
                source_flux_sampling=:window_start_endpoint,
                humidity_sampling=:none, delta_semantics=:none,
                mass_basis=:dry,
            )
            @test read(path) == sentinel
            @test !ispath(path * ".tmp")
        end
    end
end

@testset "binary readers reject size mismatches and close failed opens" begin
    mktempdir() do dir
        valid = joinpath(dir, "valid.bin")
        writer, window = _open_rg(valid)
        MD.write_streaming_window!(writer, window)
        MD.close_streaming_transport_binary!(writer)
        reader = MD.TransportBinaryReader(valid)
        close(reader)

        open(valid, "a") do io
            write(io, UInt8(0xff))
        end
        @test_throws ArgumentError MD.TransportBinaryReader(valid)

        malformed = joinpath(dir, "malformed.bin")
        open(malformed, "w") do io
            write(io, "{not-json")
        end
        if isdir("/proc/self/fd")
            before = length(readdir("/proc/self/fd"))
            for _ in 1:100
                @test_throws Exception MD.TransportBinaryReader(malformed)
                @test_throws Exception MD.CubedSphereBinaryReader(malformed)
            end
            after = length(readdir("/proc/self/fd"))
            @test after <= before + 2
        end
    end
end

@testset "mmap tape manifest is atomic and cannot outlive its records" begin
    mktempdir() do dir
        panels = ntuple(_ -> ones(Float32, 2, 2, 1), 6)
        first_storage = TapeMod.MmapCSTapeStorage(
            dir=dir, cleanup_on_finalize=false)
        TapeMod._stage_panels(first_storage, panels)
        TapeMod.finalize_tape!(first_storage)
        @test isfile(joinpath(dir, "manifest.toml"))

        second_storage = TapeMod.MmapCSTapeStorage(
            dir=dir, cleanup_on_finalize=false)
        @test !ispath(joinpath(dir, "manifest.toml"))
        @test filesize(joinpath(dir, "records.bin")) == 0
        @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        TapeMod.finalize_tape!(second_storage)

        open(joinpath(dir, "records.bin"), "a") do io
            write(io, UInt8(0x01))
        end
        @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
    end
end
