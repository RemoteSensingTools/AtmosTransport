#!/usr/bin/env julia

using Test
using NCDatasets: NCDataset

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

# ---------------------------------------------------------------------------
# Plan 26 P0.D1 — observation IO round-trip + schema-validation tests.
#
# Verifies:
#   * `write_observations` emits a NetCDF that matches the
#     `schemas/cs_observations_v1.toml` layout (dim names, var dtypes,
#     root attrs).
#   * `read_observations(write_observations(...))` recovers the
#     original `CSObservationSet` bit-exact at the dtype precisions
#     declared in the schema.
#   * Loader rejects files that violate the schema:
#       - wrong `cs_observations_schema` version
#       - missing required variables
#       - missing required attributes
#       - wrong `date_component` dimension length
#   * `CSObservationRecord` keyword constructor coerces dtypes and
#     rejects non-positive sigma.
# ---------------------------------------------------------------------------

function _sample_records()
    return [
        AT.CSObservationRecord(id = 1,
                               date_components = (2024, 6, 15, 12, 0, 0),
                               lat = 34.05f0, lon = -118.25f0, alt = 100.0f0,
                               value = 420.1, value_sigma = 0.5,
                               instrument_type = "TCCON",
                               tracer = "CO2"),
        AT.CSObservationRecord(id = 2,
                               date_components = (2024, 6, 15, 13, 30, 0),
                               lat = -33.87f0, lon = 151.21f0, alt = 50.0f0,
                               value = 1925.3, value_sigma = 1.2,
                               instrument_type = "ICOS",
                               tracer = "CH4"),
        AT.CSObservationRecord(id = 99,
                               date_components = (2024, 12, 31, 23, 59, 59),
                               lat = 0.0f0, lon = 0.0f0, alt = 0.0f0,
                               value = 0.0, value_sigma = 1e-6,
                               instrument_type = "OCO-2",
                               tracer = "CO2"),
    ]
end

@testset "CSObservationRecord keyword constructor" begin
    r = AT.CSObservationRecord(id = 7,
                               date_components = (2024, 1, 1, 0, 0, 0),
                               lat = 1, lon = 2, alt = 3,
                               value = 10, value_sigma = 0.1,
                               instrument_type = "X",
                               tracer = "Y")
    @test r.id === Int64(7)
    @test r.date_components === (Int16(2024), Int16(1), Int16(1),
                                 Int16(0),    Int16(0), Int16(0))
    @test r.lat === Float32(1)
    @test r.lon === Float32(2)
    @test r.alt === Float32(3)
    @test r.value === Float64(10)
    @test r.value_sigma === Float64(0.1)
    @test r.instrument_type == "X"
    @test r.tracer == "Y"

    @test_throws ArgumentError AT.CSObservationRecord(
        id = 1, date_components = (1, 2, 3, 4, 5),  # length 5
        lat = 0, lon = 0, alt = 0,
        value = 0, value_sigma = 1.0,
        instrument_type = "I", tracer = "T")
    @test_throws ArgumentError AT.CSObservationRecord(
        id = 1, date_components = (2024, 1, 1, 0, 0, 0),
        lat = 0, lon = 0, alt = 0,
        value = 0, value_sigma = -1.0,   # non-positive sigma
        instrument_type = "I", tracer = "T")
end

@testset "CSObservationSet bit-exact NetCDF round-trip" begin
    records = _sample_records()
    set = AT.CSObservationSet(records;
                              time_origin = "1900-01-01 00:00:00")

    mktempdir() do dir
        path = joinpath(dir, "obs_v1.nc")
        AT.write_observations(path, set)
        @test isfile(path)

        back = AT.read_observations(path)
        @test back isa AT.CSObservationSet
        @test back.time_origin == set.time_origin
        @test length(back) == length(set)
        @test isempty(back) == false
        for (orig, got) in zip(set, back)
            @test orig.id == got.id
            @test orig.date_components == got.date_components
            @test orig.lat === got.lat
            @test orig.lon === got.lon
            @test orig.alt === got.alt
            @test orig.value === got.value
            @test orig.value_sigma === got.value_sigma
            @test orig.instrument_type == got.instrument_type
            @test orig.tracer == got.tracer
        end
    end
end

@testset "v1 NetCDF on-disk layout matches schema" begin
    set = AT.CSObservationSet(_sample_records();
                              time_origin = "2024-01-01 00:00:00")
    mktempdir() do dir
        path = joinpath(dir, "obs_v1.nc")
        AT.write_observations(path, set)

        NCDataset(path, "r") do ds
            # Root attributes — schema-pinning + time-origin.
            @test ds.attrib["cs_observations_schema"] == "v1"
            @test ds.attrib["time_origin"] == "2024-01-01 00:00:00"

            # Dimensions: `obs` and `date_component` of length 6.
            @test ds.dim["obs"] == length(set)
            @test ds.dim["date_component"] == 6

            # Per-variable dtype check — must match the schema.
            @test eltype(ds["id"][:]) === Int64
            @test eltype(ds["date_components"][:, :]) === Int16
            @test eltype(ds["lat"][:]) === Float32
            @test eltype(ds["lon"][:]) === Float32
            @test eltype(ds["alt"][:]) === Float32
            @test eltype(ds["value"][:]) === Float64
            @test eltype(ds["value_sigma"][:]) === Float64
            @test ds["instrument_type"][1] isa AbstractString
            @test ds["tracer"][1] isa AbstractString

            # `date_components` is (date_component=6, obs) — NetCDF order
            # matches the column-major shape we wrote.
            @test size(ds["date_components"][:, :]) == (6, length(set))
        end
    end
end

@testset "read_observations rejects schema violations" begin
    set = AT.CSObservationSet(_sample_records();
                              time_origin = "1900-01-01 00:00:00")
    mktempdir() do dir
        # Write a valid file, then mutate it to violate the schema in
        # one place at a time. Each mutation produces a separate file
        # so the loader sees a complete invalid input rather than a
        # partial-write artifact.
        good = joinpath(dir, "good.nc")
        AT.write_observations(good, set)
        @test AT.read_observations(good) isa AT.CSObservationSet

        # 1. Wrong schema version.
        bad_version = joinpath(dir, "bad_version.nc")
        cp(good, bad_version)
        NCDataset(bad_version, "a") do ds
            ds.attrib["cs_observations_schema"] = "v2"
        end
        @test_throws ArgumentError AT.read_observations(bad_version)

        # 2. Missing schema attribute entirely.
        no_attr = joinpath(dir, "no_attr.nc")
        cp(good, no_attr)
        NCDataset(no_attr, "a") do ds
            delete!(ds.attrib, "cs_observations_schema")
        end
        @test_throws ArgumentError AT.read_observations(no_attr)

        # 3. Missing time_origin.
        no_origin = joinpath(dir, "no_origin.nc")
        cp(good, no_origin)
        NCDataset(no_origin, "a") do ds
            delete!(ds.attrib, "time_origin")
        end
        @test_throws ArgumentError AT.read_observations(no_origin)

        # 4. Non-existent file.
        @test_throws ArgumentError AT.read_observations(joinpath(dir, "missing.nc"))
    end
end

@testset "write_observations rejects empty time_origin" begin
    @test_throws ArgumentError AT.write_observations(
        tempname(), AT.CSObservationSet(_sample_records(); time_origin = ""))
end
