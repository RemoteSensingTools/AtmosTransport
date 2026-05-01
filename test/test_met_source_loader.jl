#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 4 — TOML descriptor → typed settings
#
# Verifies that `load_met_settings(toml_path; root_dir, ...)` reads a
# `config/met_sources/*.toml` file and returns the correct concrete
# `AbstractMetSettings` subtype with [preprocessing]-derived defaults
# applied and explicit kwargs overriding.
# ---------------------------------------------------------------------------

using Test
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Preprocessing: load_met_settings, GEOSITSettings, GEOSFPSettings,
                                      AbstractGEOSSettings, AbstractMetSettings,
                                      geosfp_native_hourly_ctm_path
using Dates

const REPO_ROOT = joinpath(@__DIR__, "..")

@testset "Met source loader" begin

    @testset "GEOS-IT TOML → GEOSITSettings" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosit.toml")
        s = load_met_settings(toml; root_dir = "/tmp/geosit_test")
        @test s isa GEOSITSettings
        @test s isa AbstractGEOSSettings
        @test s isa AbstractMetSettings
        @test s.root_dir          == "/tmp/geosit_test"
        @test s.Nc                == 180
        @test s.mass_flux_dt      == 450.0
        @test s.level_orientation === :auto
        @test s.include_surface === false
        @test s.include_convection === false
        @test endswith(s.coefficients_file, "geos_L72_coefficients.toml")
    end

    @testset "GEOS-FP TOML → native C720 GEOSFPSettings" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosfp.toml")
        tmp = mktempdir()
        daydir = joinpath(tmp, "20211201")
        mkpath(daydir)
        fname = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20211201_1330.V01.nc4"
        touch(joinpath(daydir, fname))
        s = load_met_settings(toml; root_dir = tmp)
        @test s isa GEOSFPSettings
        @test s.Nc == 720
        @test s.mass_flux_dt == 450.0
        @test s.include_surface === false
        @test s.include_convection === false
        @test s.physics_dir == ""
        @test s.physics_layout === :auto
        @test geosfp_native_hourly_ctm_path(s, Date(2021, 12, 1), 13) ==
              joinpath(daydir, fname)

        legacy = "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20211201_1400.V01.nc4"
        touch(joinpath(daydir, legacy))
        @test geosfp_native_hourly_ctm_path(s, Date(2021, 12, 1), 14) ==
              joinpath(daydir, legacy)
    end

    @testset "kwargs override TOML defaults" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosit.toml")
        s = load_met_settings(toml;
                              root_dir = "/tmp/geosit_test",
                              mass_flux_dt = 900.0,
                              level_orientation = :bottom_up,
                              include_surface = true,
                              include_convection = true,
                              physics_dir = "/tmp/geosfp_physics",
                              physics_layout = :latlon_025)
        @test s.mass_flux_dt       == 900.0
        @test s.level_orientation  === :bottom_up
        @test s.include_surface    === true
        @test s.include_convection === true
        @test s.physics_dir        == "/tmp/geosfp_physics"
        @test s.physics_layout     === :latlon_025
    end

    @testset "unsupported source name errors loudly" begin
        # Synthesize a tiny TOML with an unknown source name.
        path = tempname() * ".toml"
        open(path, "w") do io
            print(io, """
                [source]
                name = "FAKE-SOURCE"

                [grid]
                Nc = 8
                """)
        end
        @test_throws ErrorException load_met_settings(path; root_dir = "/tmp/x")
    end

    @testset "missing TOML errors with file path" begin
        @test_throws ErrorException load_met_settings("/nonexistent/path.toml";
                                                       root_dir = "/tmp")
    end
end
