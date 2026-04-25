#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 2 — IdentityRegrid passthrough
#
# When source and target meshes are structurally equivalent, build_regridder
# must return an IdentityRegrid sentinel and apply_regridder! must short-
# circuit to copyto! (bitwise equality, no allocation, no `if` in hot path).
#
# Coverage:
#   1. meshes_equivalent on structurally-equal but distinct LL/CS/RG instances.
#   2. meshes_equivalent rejects different types and different shapes.
#   3. build_regridder returns IdentityRegrid for the equivalent case (CS, LL).
#   4. apply_regridder! on IdentityRegrid produces bitwise-equal output.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: LatLonMesh, CubedSphereMesh, ReducedGaussianMesh,
                              GnomonicPanelConvention, GEOSNativePanelConvention
using .AtmosTransport.Regridding: build_regridder, apply_regridder!,
                                   IdentityRegrid, meshes_equivalent

@testset "IdentityRegrid passthrough" begin

    @testset "meshes_equivalent: structural equality" begin
        # Two LatLonMesh built independently with the same parameters.
        ll1 = LatLonMesh(Nx=4, Ny=3)
        ll2 = LatLonMesh(Nx=4, Ny=3)
        @test ll1 !== ll2                  # distinct objects
        @test meshes_equivalent(ll1, ll2)  # but structurally equal

        # CubedSphere — same Nc + same convention type.
        cs1 = CubedSphereMesh(Nc=8)
        cs2 = CubedSphereMesh(Nc=8)
        @test cs1 !== cs2
        @test meshes_equivalent(cs1, cs2)

        # ReducedGaussianMesh — same per-ring nlon + same latitudes.
        rg1 = ReducedGaussianMesh(Float64[-45, 45], [4, 4]; FT=Float64)
        rg2 = ReducedGaussianMesh(Float64[-45, 45], [4, 4]; FT=Float64)
        @test rg1 !== rg2
        @test meshes_equivalent(rg1, rg2)
    end

    @testset "meshes_equivalent: rejects mismatch" begin
        ll  = LatLonMesh(Nx=4, Ny=3)
        ll2 = LatLonMesh(Nx=4, Ny=4)        # Ny differs
        cs  = CubedSphereMesh(Nc=8)
        cs2 = CubedSphereMesh(Nc=12)        # Nc differs
        cs_geos = CubedSphereMesh(Nc=8, convention=GEOSNativePanelConvention())
        cs_gnom = CubedSphereMesh(Nc=8, convention=GnomonicPanelConvention())

        @test !meshes_equivalent(ll, ll2)     # shape mismatch
        @test !meshes_equivalent(cs, cs2)     # Nc mismatch
        @test !meshes_equivalent(cs_geos, cs_gnom)  # convention mismatch
        @test !meshes_equivalent(ll, cs)      # different mesh types
    end

    @testset "build_regridder returns IdentityRegrid when equivalent" begin
        cs1 = CubedSphereMesh(Nc=8)
        cs2 = CubedSphereMesh(Nc=8)
        r = build_regridder(cs1, cs2)
        @test r isa IdentityRegrid
        @test r.mesh === cs1                  # holds the source mesh

        ll1 = LatLonMesh(Nx=4, Ny=3)
        ll2 = LatLonMesh(Nx=4, Ny=3)
        rll = build_regridder(ll1, ll2)
        @test rll isa IdentityRegrid
    end

    @testset "apply_regridder! on IdentityRegrid is bitwise copyto!" begin
        FT = Float64
        Nc = 8
        npanel = 6
        Nz = 3
        n_h = npanel * Nc * Nc
        src = rand(FT, n_h, Nz)
        dst = zeros(FT, n_h, Nz)
        cs1 = CubedSphereMesh(Nc=Nc)
        cs2 = CubedSphereMesh(Nc=Nc)
        r = build_regridder(cs1, cs2)
        apply_regridder!(dst, r, src)
        @test dst == src                     # bitwise equality
        @test dst !== src                    # but distinct buffers (copyto!, not aliasing)
    end
end
