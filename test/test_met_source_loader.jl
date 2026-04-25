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
                                      AbstractGEOSSettings, AbstractMetSettings

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
        @test s.include_convection === false
        @test endswith(s.coefficients_file, "geos_L72_coefficients.toml")
    end

    @testset "kwargs override TOML defaults" begin
        toml = joinpath(REPO_ROOT, "config", "met_sources", "geosit.toml")
        s = load_met_settings(toml;
                              root_dir = "/tmp/geosit_test",
                              mass_flux_dt = 900.0,
                              level_orientation = :bottom_up,
                              include_convection = true)
        @test s.mass_flux_dt       == 900.0
        @test s.level_orientation  === :bottom_up
        @test s.include_convection === true
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
