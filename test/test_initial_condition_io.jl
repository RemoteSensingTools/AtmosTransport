#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_initial_condition_io.jl — plan 40 Commit 1b regression tests.
#
# Covers:
#   - `build_initial_mixing_ratio` bit-exact behaviour post-hoist on LL
#     (uniform + gaussian_blob) and RG (uniform + gaussian_blob) — uses
#     the same algebra as the pre-hoist path at scripts/run_transport_binary.jl:
#     {570,593}, so equivalence is proven by "same output for the same inputs".
#   - `pack_initial_tracer_mass` basis-aware packing per
#     feedback_vmr_to_mass_basis_aware:
#       * DryBasis   → rm = vmr .* air_mass
#       * MoistBasis → rm = vmr .* air_mass .* (1 .- qv)  (+ error without qv)
#
# Does NOT exercise the file-based IC path (needs Catrine NetCDF); that
# is covered end-to-end via test_run_transport_binary_recipe.jl.
# CS dispatch is added in plan 40 Commit 1c.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

const FT = Float64

@testset "plan 40 Commit 1b — InitialConditionIO hoist" begin

    # ---------------------------- LL uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — LatLonMesh" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        Nz = 2
        air_mass = ones(FT, 4, 3, Nz)   # shape-only; builder does not read values

        # uniform
        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 1.5e-4))
        @test size(q) == size(air_mass)
        @test all(q .== FT(1.5e-4))

        # gaussian_blob — non-uniform profile centered at (0, 0)
        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 10.0,
                                             "sigma_lat_deg" => 10.0,
                                             "amplitude" => 1.0e-3,
                                             "background" => 4.0e-4))
        @test size(q2) == size(air_mass)
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)   # non-trivial profile
    end

    # ---------------------------- RG uniform + blob ------------------------
    @testset "build_initial_mixing_ratio — ReducedGaussianMesh" begin
        latitudes = [-75.0, -25.0, 25.0, 75.0]
        nlon_per_ring = [4, 8, 8, 4]
        mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
        Nz = 2
        ncells_ = ncells(mesh)
        air_mass = ones(FT, ncells_, Nz)

        q = build_initial_mixing_ratio(air_mass, mesh,
                                       Dict("kind" => "uniform",
                                            "background" => 2.0e-4))
        @test size(q) == (ncells_, Nz)
        @test all(q .== FT(2.0e-4))

        q2 = build_initial_mixing_ratio(air_mass, mesh,
                                        Dict("kind" => "gaussian_blob",
                                             "lon0_deg" => 0.0, "lat0_deg" => 0.0,
                                             "sigma_lon_deg" => 15.0,
                                             "sigma_lat_deg" => 15.0,
                                             "amplitude" => 5.0e-4,
                                             "background" => 4.0e-4))
        @test minimum(q2) ≥ FT(4.0e-4)
        @test maximum(q2) > minimum(q2)
    end

    # ---------------------------- LL grid-level dispatch -------------------
    @testset "build_initial_mixing_ratio — AtmosGrid{LatLonMesh} forwards" begin
        # A non-file kind should forward to the mesh method and be bit-exact.
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        Nz = 2
        air_mass = ones(FT, 4, 3, Nz)
        cfg = Dict("kind" => "uniform", "background" => 3.14e-4)
        q_mesh = build_initial_mixing_ratio(air_mass, mesh, cfg)
        q_grid = build_initial_mixing_ratio(air_mass, grid, cfg)
        @test q_mesh == q_grid   # bit-exact
    end

    # ---------------------------- pack_initial_tracer_mass ------------------
    @testset "pack_initial_tracer_mass — DryBasis (LL)" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = fill(FT(1.2e10), 4, 3, 2)
        vmr      = fill(FT(4.11e-4), 4, 3, 2)
        rm = pack_initial_tracer_mass(grid, air_mass, vmr; mass_basis = DryBasis())
        @test rm == vmr .* air_mass
        # qv is ignored on DryBasis
        qv_ignored = fill(FT(0.5), 4, 3, 2)   # absurd humidity — would break MoistBasis
        rm2 = pack_initial_tracer_mass(grid, air_mass, vmr;
                                       mass_basis = DryBasis(), qv = qv_ignored)
        @test rm2 == rm
    end

    @testset "pack_initial_tracer_mass — MoistBasis (LL)" begin
        mesh = LatLonMesh(; Nx = 4, Ny = 3,
                          longitude = (0.0, 360.0),
                          latitude  = (-90.0, 90.0))
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = fill(FT(1.2e10), 4, 3, 2)
        vmr      = fill(FT(4.11e-4), 4, 3, 2)
        qv       = fill(FT(0.02), 4, 3, 2)   # 2 % humidity
        rm = pack_initial_tracer_mass(grid, air_mass, vmr;
                                      mass_basis = MoistBasis(), qv = qv)
        @test rm == vmr .* air_mass .* (1 .- qv)
        # Missing qv errors loudly
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr;
                                                            mass_basis = MoistBasis())
        # Shape mismatch errors
        @test_throws DimensionMismatch pack_initial_tracer_mass(grid, air_mass, vmr;
                                                                mass_basis = MoistBasis(),
                                                                qv = fill(FT(0.02), 3, 3, 2))
    end

    @testset "pack_initial_tracer_mass — RG both bases" begin
        latitudes = [-75.0, -25.0, 25.0, 75.0]
        nlon_per_ring = [4, 8, 8, 4]
        mesh = ReducedGaussianMesh(latitudes, nlon_per_ring; FT = FT)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        ncells_ = ncells(mesh)
        air_mass = fill(FT(1.2e10), ncells_, 2)
        vmr      = fill(FT(4.11e-4), ncells_, 2)

        rm_dry = pack_initial_tracer_mass(grid, air_mass, vmr;
                                          mass_basis = DryBasis())
        @test rm_dry == vmr .* air_mass

        qv = fill(FT(0.01), ncells_, 2)
        rm_moist = pack_initial_tracer_mass(grid, air_mass, vmr;
                                            mass_basis = MoistBasis(), qv = qv)
        @test rm_moist == vmr .* air_mass .* (1 .- qv)
    end

    # --------- plan 40 Commit 1c: CubedSphere IC + packer ---------------
    @testset "build_initial_mixing_ratio — CubedSphereMesh uniform" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        # Halo-padded 6-panel air_mass tuple (matches CubedSphereTransportDriver layout).
        air_mass = ntuple(_ -> fill(FT(1e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)

        vmr = build_initial_mixing_ratio(air_mass, grid,
                                         Dict("kind" => "uniform",
                                              "background" => 4.11e-4))
        @test vmr isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr[p]) == (Nc, Nc, Nz) for p in 1:6)   # interior only
        @test all(all(vmr[p] .== FT(4.11e-4)) for p in 1:6)

        # Unsupported kind errors with a helpful message
        @test_throws ArgumentError build_initial_mixing_ratio(air_mass, grid,
                                                              Dict("kind" => "gaussian_blob"))
    end

    @testset "pack_initial_tracer_mass — CubedSphereMesh (DryBasis)" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1.2e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)
        vmr_interior = ntuple(_ -> fill(FT(4.11e-4), Nc, Nc, Nz), 6)

        rm = pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                      mass_basis = DryBasis())
        @test rm isa NTuple{6, Array{FT, 3}}
        for p in 1:6
            @test size(rm[p]) == (Nc + 2 * Hp, Nc + 2 * Hp, Nz)
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            expected = vmr_interior[p] .* FT(1.2e10)   # air_mass × vmr (interior)
            @test interior == expected
            # Halo ring is zero — halo exchanges at runtime populate it.
            @test rm[p][1, 1, 1] == zero(FT)
            @test rm[p][end, end, end] == zero(FT)
        end
    end

    @testset "pack_initial_tracer_mass — CubedSphereMesh (MoistBasis)" begin
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1.2e10), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)
        vmr_interior = ntuple(_ -> fill(FT(4.11e-4), Nc, Nc, Nz), 6)
        qv = ntuple(_ -> fill(FT(0.02), Nc + 2 * Hp, Nc + 2 * Hp, Nz), 6)

        rm = pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                      mass_basis = MoistBasis(), qv = qv)
        for p in 1:6
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            expected = vmr_interior[p] .* FT(1.2e10) .* (1 - FT(0.02))
            @test interior == expected
        end

        # MoistBasis without qv errors
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                            mass_basis = MoistBasis())
        # MoistBasis with wrong qv type errors
        qv_flat = fill(FT(0.02), Nc + 2 * Hp, Nc + 2 * Hp, Nz)   # not an NTuple{6}
        @test_throws ArgumentError pack_initial_tracer_mass(grid, air_mass, vmr_interior;
                                                            mass_basis = MoistBasis(),
                                                            qv = qv_flat)
    end

end
