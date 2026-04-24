#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_binary_inspector.jl — plan 40 Commit 5
#
# Exercises `binary_capabilities(reader)` + `inspect_binary(path)` against
# a tiny LL fixture written via `write_transport_binary`. Covers:
#   - capability accessors return sensible booleans for m/am/bm/cm/ps/qv
#   - `inspect_binary` prints a capability report and returns the NamedTuple
#   - `has_tm5_convection` reported correctly when entu/detu/entd/detd
#     are present vs absent
#
# CS fixture coverage is deferred to Commit 3's `regrid_ll_to_cs` script
# (which will produce a CS binary we can inspect end-to-end).
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

const _INSPECTOR_AIR_MASS = 1e16

function _inspector_fixture_binary(path::AbstractString;
                                   FT::Type{<:AbstractFloat} = Float64,
                                   with_tm5::Bool = false)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m  = fill(FT(_INSPECTOR_AIR_MASS), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ps = fill(FT(95_000), Nx, Ny)

    window = if with_tm5
        tm5 = (entu = zeros(FT, Nx, Ny, Nz),
               detu = zeros(FT, Nx, Ny, Nz),
               entd = zeros(FT, Nx, Ny, Nz),
               detd = zeros(FT, Nx, Ny, Nz))
        (m = m, am = am, bm = bm, cm = cm, ps = ps, tm5_fields = tm5)
    else
        (m = m, am = am, bm = bm, cm = cm, ps = ps)
    end

    write_transport_binary(path, grid, [window];
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 1,
                           mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return nothing
end

@testset "plan 40 Commit 5 — binary_capabilities + inspect_binary" begin

    @testset "basic LL fixture (no TM5)" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_base.bin")
            _inspector_fixture_binary(path; with_tm5 = false)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.advection === true
            @test caps.tm5_convection === false
            @test caps.cmfmc_convection === false
            @test caps.surface_pressure === true
            @test caps.mass_basis === :dry
            @test caps.grid_type === :latlon
            @test :m in caps.payload_sections
            @test :am in caps.payload_sections
            close(reader)
        end
    end

    @testset "LL fixture with TM5 convection sections" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_tm5.bin")
            _inspector_fixture_binary(path; with_tm5 = true)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.tm5_convection === true
            @test :entu in caps.payload_sections
            @test :detu in caps.payload_sections
            @test :entd in caps.payload_sections
            @test :detd in caps.payload_sections
            close(reader)
        end
    end

    @testset "inspect_binary returns capability NamedTuple + prints report" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_inspect.bin")
            _inspector_fixture_binary(path; with_tm5 = true)

            io = IOBuffer()
            caps = inspect_binary(path; io = io)
            report = String(take!(io))

            # Programmatic summary
            @test caps.advection === true
            @test caps.tm5_convection === true
            @test caps.surface_pressure === true
            @test caps.grid_type === :latlon

            # Human report contains capability rows with check marks
            @test occursin("Capabilities:", report)
            @test occursin("✓ advection", report)
            @test occursin("✓ TM5 convection", report)
            @test occursin("✓ surface pressure", report)
            @test occursin("mass_basis       = dry", report)
            @test occursin("grid_type        = latlon", report)
        end
    end

    @testset "inspect_binary errors on missing file" begin
        @test_throws ArgumentError inspect_binary("/nonexistent/xyz.bin")
    end

end
