#!/usr/bin/env julia
# Smoke test for scripts/run_cs_driven.jl operator builders. Exercises the
# TOML→operator path without requiring a real CS binary (that's what the
# full end-to-end test in test_cubed_sphere_runtime.jl covers).

using Test

include(joinpath(@__DIR__, "..", "scripts", "run_cs_driven.jl"))
using .AtmosTransport

@testset "run_cs_driven builders" begin

    @testset "build_scheme dispatch" begin
        @test build_scheme(Dict("run" => Dict("scheme" => "upwind"))) isa UpwindScheme
        @test build_scheme(Dict("advection" => Dict("scheme" => "slopes"))) isa SlopesScheme
        # ppm_order is ignored with a warning; scheme construction still succeeds.
        @test (@test_logs (:warn,) match_mode=:any build_scheme(
                   Dict("advection" => Dict("scheme" => "ppm", "ppm_order" => 7)))
               ) isa PPMScheme
        @test build_scheme(Dict("advection" => Dict("scheme" => "ppm"))) isa PPMScheme
        @test build_scheme(Dict{String,Any}()) isa UpwindScheme   # default
        @test_throws ErrorException build_scheme(Dict("advection" => Dict("scheme" => "xyz")))
    end

    @testset "build_diffusion dispatch" begin
        # default (no section) → NoDiffusion
        @test build_diffusion(Dict{String,Any}(), Float64) isa NoDiffusion

        # kind = "none" → NoDiffusion
        @test build_diffusion(Dict("diffusion" => Dict("kind" => "none")), Float64) isa NoDiffusion

        # kind = "constant" → ImplicitVerticalDiffusion
        op = build_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                       "value" => 2.5)), Float64)
        @test op isa ImplicitVerticalDiffusion
        # kz_field is a CubedSphereField wrapping 6 ConstantField
        kz = op.kz_field
        @test kz isa CubedSphereField
        # Check the per-panel field types round-trip the value.
        @test all(field_value(panel_field(kz, p), (1, 1, 1)) == 2.5 for p in 1:6)

        # F32 propagates to the Kz value
        op32 = build_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                         "value" => 1.0)), Float32)
        @test op32 isa ImplicitVerticalDiffusion
        @test eltype(field_value(panel_field(op32.kz_field, 1), (1, 1, 1))) === Float32 ||
              field_value(panel_field(op32.kz_field, 1), (1, 1, 1)) isa Float32

        # Unknown kind → error
        @test_throws ErrorException build_diffusion(
            Dict("diffusion" => Dict("kind" => "magic")), Float64)
    end

    @testset "build_convection dispatch" begin
        # A stub reader with a controllable has_cmfmc answer.
        struct StubReader
            has_cmfmc :: Bool
        end
        AtmosTransport.MetDrivers.has_cmfmc(r::StubReader) = r.has_cmfmc

        no_cmfmc = StubReader(false)
        yes_cmfmc = StubReader(true)

        # default (no section) → NoConvection
        @test build_convection(Dict{String,Any}(), no_cmfmc) isa NoConvection

        # kind = "none" → NoConvection
        @test build_convection(Dict("convection" => Dict("kind" => "none")), no_cmfmc) isa NoConvection

        # kind = "tm5" → TM5Convection (doesn't require cmfmc flag)
        @test build_convection(Dict("convection" => Dict("kind" => "tm5")), no_cmfmc) isa TM5Convection

        # kind = "cmfmc" — requires reader to expose cmfmc
        @test build_convection(Dict("convection" => Dict("kind" => "cmfmc")), yes_cmfmc) isa CMFMCConvection
        @test_throws ErrorException build_convection(
            Dict("convection" => Dict("kind" => "cmfmc")), no_cmfmc)

        # Unknown kind → error
        @test_throws ErrorException build_convection(
            Dict("convection" => Dict("kind" => "ras")), no_cmfmc)
    end

    @testset "build_tracer_panels produces matching-shape initial fields" begin
        FT = Float64
        air_mass = ntuple(_ -> fill(FT(1e9), 4, 4, 3), 6)

        uni = build_tracer_panels(Dict("kind" => "uniform", "background" => 1e-6), air_mass, FT)
        @test length(uni) == 6
        @test size(uni[1]) == size(air_mass[1])
        @test all(uni[p] ≈ air_mass[p] .* 1e-6 for p in 1:6)

        cat = build_tracer_panels(Dict("kind" => "catrine_co2"), air_mass, FT)
        @test all(cat[p] ≈ air_mass[p] .* 4.11e-4 for p in 1:6)

        # zero-filled fossil placeholder
        zero_init = build_tracer_panels(Dict("kind" => "uniform", "background" => 0.0), air_mass, FT)
        @test all(iszero, zero_init[1])

        @test_throws ErrorException build_tracer_panels(
            Dict("kind" => "file", "file" => "x.nc"), air_mass, FT)
    end
end
