#!/usr/bin/env julia
# Smoke test for scripts/run_cs_driven.jl operator builders. Exercises the
# TOML→operator path without requiring a real CS binary (that's what the
# full end-to-end test in test_cubed_sphere_runtime.jl covers).

using Test

include(joinpath(@__DIR__, "..", "scripts", "run_cs_driven.jl"))
using .AtmosTransport

struct StubReader
    has_cmfmc :: Bool
    has_tm5   :: Bool
end
AtmosTransport.MetDrivers.has_cmfmc(r::StubReader) = r.has_cmfmc
AtmosTransport.MetDrivers.has_tm5conv(r::StubReader) = r.has_tm5
AtmosTransport.Models._runtime_recipe_style(::StubReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5conv(r::StubReader) = r.has_tm5
AtmosTransport.Models._runtime_has_cmfmc(r::StubReader) = r.has_cmfmc

struct StubStructuredReader
    has_tm5 :: Bool
end
AtmosTransport.Models._runtime_recipe_style(::StubStructuredReader) =
    AtmosTransport.Models.LatLonRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5conv(r::StubStructuredReader) = r.has_tm5
AtmosTransport.Models._runtime_has_cmfmc(::StubStructuredReader) = false

@testset "run_cs_driven builders" begin

    latlon_grid = AtmosGrid(
        LatLonMesh(; FT = Float64, Nx = 2, Ny = 2),
        HybridSigmaPressure(Float64[0, 1000], Float64[0, 1]),
        CPU();
        FT = Float64,
    )
    reduced_grid = AtmosGrid(
        ReducedGaussianMesh(Float64[-45, 45], [4, 4]; FT = Float64),
        HybridSigmaPressure(Float64[0, 1000], Float64[0, 1]),
        CPU();
        FT = Float64,
    )

    @testset "build_runtime_advection dispatch" begin
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "upwind")), latlon_grid) isa UpwindScheme
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "slopes")), latlon_grid) isa SlopesScheme
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "ppm")), latlon_grid) isa PPMScheme
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "linrood")), latlon_grid)
    end

    @testset "build_cs_advection dispatch" begin
        @test build_cs_advection(Dict("run" => Dict("scheme" => "upwind"))) isa UpwindScheme
        @test build_cs_advection(Dict("advection" => Dict("scheme" => "slopes"))) isa SlopesScheme
        @test build_cs_advection(Dict("advection" => Dict("scheme" => "ppm"))) isa PPMScheme
        @test build_cs_advection(Dict("advection" => Dict("scheme" => "linrood"))) isa LinRoodPPMScheme
        @test build_cs_advection(Dict("advection" => Dict("scheme" => "linrood", "ppm_order" => 7))) isa LinRoodPPMScheme
        @test build_cs_advection(Dict{String,Any}()) isa UpwindScheme
        @test_throws ArgumentError build_cs_advection(
            Dict("advection" => Dict("scheme" => "ppm", "ppm_order" => 7)))
        @test_throws ArgumentError build_cs_advection(Dict("advection" => Dict("scheme" => "xyz")))
    end

    @testset "configured_cs_halo_width dispatch" begin
        @test configured_cs_halo_width(Dict{String,Any}(), UpwindScheme()) == 1
        @test configured_cs_halo_width(Dict("advection" => Dict("scheme" => "ppm")), PPMScheme()) == 3
        @test configured_cs_halo_width(Dict("run" => Dict("halo_padding" => 5)), SlopesScheme()) == 5
        @test configured_cs_halo_width(Dict("run" => Dict("Hp" => 4)), LinRoodPPMScheme()) == 4
        @test_throws ArgumentError configured_cs_halo_width(
            Dict("run" => Dict("Hp" => 3, "halo_padding" => 4)), UpwindScheme())
    end

    @testset "build_cs_diffusion dispatch" begin
        # default (no section) → NoDiffusion
        @test build_cs_diffusion(Dict{String,Any}(), Float64) isa NoDiffusion

        # kind = "none" → NoDiffusion
        @test build_cs_diffusion(Dict("diffusion" => Dict("kind" => "none")), Float64) isa NoDiffusion

        # kind = "constant" → ImplicitVerticalDiffusion
        op = build_cs_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                          "value" => 2.5)), Float64)
        @test op isa ImplicitVerticalDiffusion
        # kz_field is a CubedSphereField wrapping 6 ConstantField
        kz = op.kz_field
        @test kz isa CubedSphereField
        # Check the per-panel field types round-trip the value.
        @test all(field_value(panel_field(kz, p), (1, 1, 1)) == 2.5 for p in 1:6)

        # F32 propagates to the Kz value
        op32 = build_cs_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                            "value" => 1.0)), Float32)
        @test op32 isa ImplicitVerticalDiffusion
        @test eltype(field_value(panel_field(op32.kz_field, 1), (1, 1, 1))) === Float32 ||
              field_value(panel_field(op32.kz_field, 1), (1, 1, 1)) isa Float32

        # Unknown kind → error
        @test_throws ArgumentError build_cs_diffusion(
            Dict("diffusion" => Dict("kind" => "magic")), Float64)

        # Section B (codex) P0: legacy `type = "..."` schema must NOT
        # silently fall through to NoDiffusion. It must error with a
        # migration hint so old configs that expected diffusion are caught.
        @test_throws ArgumentError build_cs_diffusion(
            Dict("diffusion" => Dict("type" => "pbl")), Float64)
        @test_throws ArgumentError build_cs_diffusion(
            Dict("diffusion" => Dict("type" => "nonlocal_pbl")), Float64)

        # `[diffusion]` present but neither `type` nor `kind` → error.
        @test_throws ArgumentError build_cs_diffusion(
            Dict("diffusion" => Dict("value" => 1.0)), Float64)
    end

    @testset "build_runtime_diffusion chooses layout-aware field rank" begin
        op_ll = build_runtime_diffusion(
            Dict("diffusion" => Dict("kind" => "constant", "value" => 2.5)),
            latlon_grid,
            Float64)
        @test op_ll isa ImplicitVerticalDiffusion
        @test field_value(op_ll.kz_field, (1, 1, 1)) == 2.5

        op_rg = build_runtime_diffusion(
            Dict("diffusion" => Dict("kind" => "constant", "value" => 1.5)),
            reduced_grid,
            Float64)
        @test op_rg isa ImplicitVerticalDiffusion
        @test field_value(op_rg.kz_field, (1, 1)) == 1.5
    end

    @testset "build_cs_convection + recipe validation" begin
        no_conv   = StubReader(false, false)
        only_tm5  = StubReader(false, true)
        only_cmfmc = StubReader(true, false)
        full_conv = StubReader(true, true)

        # default (no section) → NoConvection
        @test build_cs_convection(Dict{String,Any}()) isa NoConvection

        # kind = "none" → NoConvection
        @test build_cs_convection(Dict("convection" => Dict("kind" => "none"))) isa NoConvection

        @test build_cs_convection(Dict("convection" => Dict("kind" => "tm5"))) isa TM5Convection
        @test build_cs_convection(Dict("convection" => Dict("kind" => "cmfmc"))) isa CMFMCConvection

        @test build_cs_physics_recipe(Dict("convection" => Dict("kind" => "tm5")), only_tm5, Float64).convection isa TM5Convection
        @test build_cs_physics_recipe(Dict("convection" => Dict("kind" => "cmfmc")), only_cmfmc, Float64).convection isa CMFMCConvection
        @test build_cs_physics_recipe(Dict("convection" => Dict("kind" => "cmfmc")), full_conv, Float64).convection isa CMFMCConvection

        @test_throws ArgumentError build_cs_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), no_conv, Float64)
        @test_throws ArgumentError build_cs_physics_recipe(
            Dict("convection" => Dict("kind" => "cmfmc")), no_conv, Float64)
        @test_throws ArgumentError build_cs_convection(
            Dict("convection" => Dict("kind" => "ras")))
    end

    @testset "build_runtime_physics_recipe validates structured convection capabilities" begin
        tm5_reader = StubStructuredReader(true)
        dry_reader = StubStructuredReader(false)

        @test build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), tm5_reader, Float64).convection isa TM5Convection

        @test_throws ArgumentError build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), dry_reader, Float64)

        @test_throws ArgumentError build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "cmfmc")), tm5_reader, Float64)
    end

    @testset "build_cs_physics_recipe validates halo width" begin
        reader = StubReader(false, false)

        recipe = build_cs_physics_recipe(
            Dict("advection" => Dict("scheme" => "linrood")), reader, Float64; halo_width = 3)
        @test recipe.advection isa LinRoodPPMScheme

        @test_throws ArgumentError build_cs_physics_recipe(
            Dict("advection" => Dict("scheme" => "linrood")), reader, Float64; halo_width = 2)
    end

    # Plan 40 Commit 2 removed `build_cs_tracer_panels` (it was a flat-411
    # stub). CS tracers now flow through the unified pipeline:
    # `build_initial_mixing_ratio` + `pack_initial_tracer_mass`. Those are
    # tested in detail in `test/test_initial_condition_io.jl` (plan 40
    # Commits 1b–1d). The CS runner `scripts/run_cs_driven.jl` wires
    # exactly that pipeline through from TOML at line 130.
    @testset "CS tracer IC flows through unified pipeline" begin
        FT = Float64
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1e9), Nc + 2Hp, Nc + 2Hp, Nz), 6)

        # uniform IC path
        vmr = build_initial_mixing_ratio(air_mass,
                                         grid,
                                         Dict("kind" => "uniform",
                                              "background" => 1e-6))
        rm  = pack_initial_tracer_mass(grid, air_mass, vmr;
                                       mass_basis = DryBasis())
        @test length(rm) == 6
        @test size(rm[1]) == size(air_mass[1])
        # Interior should be vmr * air_mass = 1e-6 * 1e9 = 1e3
        for p in 1:6
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            @test all(interior .≈ 1e3)
            # Halo stays zero (populated at runtime by halo exchanges)
            @test rm[p][1, 1, 1] == zero(FT)
        end

        # zero-filled fossil placeholder
        vmr_zero = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "uniform",
                                                   "background" => 0.0))
        rm_zero  = pack_initial_tracer_mass(grid, air_mass, vmr_zero;
                                            mass_basis = DryBasis())
        @test all(iszero, rm_zero[1])

        # Unsupported kind errors
        @test_throws ArgumentError build_initial_mixing_ratio(
            air_mass, grid, Dict("kind" => "gaussian_blob"))
    end
end
