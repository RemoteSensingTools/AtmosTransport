#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "LatLonMesh style and validation" begin
    mesh = LatLonMesh(; FT=Float64, size=(720, 360), longitude=(-180, 180), latitude=(-90, 90))

    @test nx(mesh) == 720
    @test ny(mesh) == 360
    @test ncells(mesh) == 720 * 360
    @test nfaces(mesh) == (721 * 360) + (720 * 361)
    @test summary(mesh) == "720×360 LatLonMesh{Float64}"

    shown = sprint(show, mesh)
    @test occursin("longitude:", shown)
    @test occursin("latitude:", shown)
    @test occursin("Δλ=0.5", shown)
    @test occursin("Δφ=0.5", shown)

    @test_throws ArgumentError LatLonMesh(; FT=Float64, size=(0, 360))
    @test_throws ArgumentError LatLonMesh(; FT=Float64, size=(720, 360), latitude=(-95, 90))
    @test_throws ArgumentError LatLonMesh(; FT=Float64, Nx=720, Ny=360, size=(360, 180))
end

@testset "CubedSphereMesh conventions" begin
    default_mesh = CubedSphereMesh(; FT=Float64, Nc=180)
    gnomonic_mesh = CubedSphereMesh(; FT=Float64, Nc=180, convention=GnomonicPanelConvention())
    geos_mesh = CubedSphereMesh(; FT=Float64, Nc=180, convention=GEOSNativePanelConvention())

    @test panel_count(default_mesh) == 6
    @test nx(default_mesh) == 180
    @test ny(default_mesh) == 180
    @test ncells(default_mesh) == 6 * 180^2
    @test nfaces(default_mesh) == 6 * 2 * 180 * 181

    @test panel_convention(default_mesh) isa GnomonicPanelConvention
    @test panel_convention(geos_mesh) isa GEOSNativePanelConvention
    @test panel_convention(gnomonic_mesh) isa GnomonicPanelConvention
    @test panel_labels(default_mesh) == (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)
    @test panel_labels(geos_mesh) == (:equatorial_1, :equatorial_2, :north_pole, :equatorial_4, :equatorial_5, :south_pole)
    @test panel_labels(gnomonic_mesh) == (:x_plus, :y_plus, :x_minus, :y_minus, :north_pole, :south_pole)

    @test occursin("C180 CubedSphereMesh", summary(default_mesh))
    shown = sprint(show, geos_mesh)
    @test occursin("equatorial_1", shown)
    @test occursin("north_pole", shown)

    @test_throws ArgumentError CubedSphereMesh(; FT=Float64, Nc=0)
end
