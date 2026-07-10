#!/usr/bin/env julia

using Test
using Adapt

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

function _derived_field_fixture(::Type{FT}=Float32) where FT
    Nx, Ny, Nz = 3, 2, 4
    pblh_samples = cat(fill(FT(500), Nx, Ny), fill(FT(900), Nx, Ny); dims=3)
    pblh = AT.StepwiseField(pblh_samples, FT[0, 1, 2])
    surface = (
        pblh = pblh,
        ustar = AT.ConstantField{FT, 2}(FT(0.3)),
        hflux = AT.ConstantField{FT, 2}(FT(80)),
        t2m = AT.ConstantField{FT, 2}(FT(290)))
    delp = AT.ConstantField{FT, 3}(FT(3000))
    return AT.DerivedKzField(; surface, delp, cache=zeros(FT, Nx, Ny, Nz))
end

@testset "adapted time-varying fields remain host-updatable" begin
    field = _derived_field_fixture()
    AT.update_field!(field, 1.5f0)
    reference = copy(field.cache)
    @test field.surface.pblh.current_window == [2]

    array_field = Adapt.adapt(Array, field)
    AT.update_field!(array_field, 0.5f0)
    @test array_field.surface.pblh.current_window == [1]

    has_cuda = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end

    if has_cuda
        device_field = Adapt.adapt(CUDA.CuArray, field)
        @test device_field.cache isa CUDA.CuArray
        AT.update_field!(device_field, 1.5f0)
        @test Array(device_field.surface.pblh.current_window) == [2]
        @test Array(device_field.cache) ≈ reference rtol=2f-5 atol=2f-5

        op = AT.ImplicitVerticalDiffusion(; kz_field=field)
        device_op = Adapt.adapt(CUDA.CuArray, op)
        @test device_op.kz_field.cache isa CUDA.CuArray
        AT.update_field!(device_op.kz_field, 1.5f0)
        @test all(isfinite, Array(device_op.kz_field.cache))
    else
        @test_skip false
    end
end

@testset "DerivedKz analytic free-troposphere branch" begin
    FT = Float64
    Nx, Ny, Nz = 2, 2, 10
    surface = (
        pblh = AT.ConstantField{FT, 2}(500.0),
        ustar = AT.ConstantField{FT, 2}(0.3),
        hflux = AT.ConstantField{FT, 2}(100.0),
        t2m = AT.ConstantField{FT, 2}(295.0))
    delp = AT.PreComputedKzField(fill(10132.5, Nx, Ny, Nz))
    field = AT.DerivedKzField(; surface, delp, cache=zeros(FT, Nx, Ny, Nz))
    AT.update_field!(field, 0.0)

    # Top cells lie above 1.2*pblh, where the closure is analytically Kz_bg.
    @test field.cache[1, 1, 1] === field.params.Kz_bg
    @test field.cache[1, 1, 2] === field.params.Kz_bg
    @test all(field.params.Kz_min .<= field.cache .<= field.params.Kz_max)
end
