# Quick numerics implementation tests for TM5-aligned schemes
#
# Checks that our Russell-Lerner slopes and Tiedtke convection implementations
# behave as expected (conservation, simple analytic cases) before full 3D/TM5 comparison.

using Test
using AtmosTransport
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Grids
using LinearAlgebra: dot

@testset "Numerics (TM5-aligned)" begin
    # -------------------------------------------------------------------------
    # 1D Russell-Lerner slopes: mass conservation and translation (no limiter)
    # -------------------------------------------------------------------------
    @testset "Slopes 1D mass conservation (no limiter)" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 101325.0],
            [0.0, 0.0, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (32, 1, 1),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        scheme = SlopesAdvection(use_limiter = false)

        u = fill(5.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)
        velocities = (; u, v, w)

        # Smooth initial profile (Gaussian in x)
        c = zeros(Nx, Ny, Nz)
        for i in 1:Nx
            c[i, 1, 1] = exp(-((i - Nx/2 - 0.5)^2) / 8.0)
        end
        tracers = (; c = copy(c))
        mass0 = sum(tracers.c)

        Δt = 50.0
        advect_x!(tracers, velocities, grid, scheme, Δt)
        @test sum(tracers.c) ≈ mass0 rtol = 1e-10
    end

    @testset "Slopes 1D mass conservation (with limiter)" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 101325.0],
            [0.0, 0.0, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (24, 1, 1),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        scheme = SlopesAdvection(use_limiter = true)

        u = fill(3.0, Nx + 1, Ny, Nz)
        v = zeros(Nx, Ny + 1, Nz)
        w = zeros(Nx, Ny, Nz + 1)
        velocities = (; u, v, w)

        c = rand(Nx, Ny, Nz) .+ 0.1
        tracers = (; c = copy(c))
        mass0 = sum(tracers.c)

        advect_x!(tracers, velocities, grid, scheme, 30.0)
        @test sum(tracers.c) ≈ mass0 rtol = 1e-10
    end

    # -------------------------------------------------------------------------
    # Single-column Tiedtke: mass conservation (already in test_convection;
    # here we check one column gives expected vertical redistribution)
    # -------------------------------------------------------------------------
    @testset "Tiedtke single column vertical redistribution" begin
        vc = HybridSigmaPressure(
            [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
            [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
        grid = LatitudeLongitudeGrid(CPU();
            size = (2, 2, 5),
            longitude = (-180, 180),
            latitude = (-90, 90),
            vertical = vc)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        conv = TiedtkeConvection()
        Δt = 3600.0

        # One column: tracer only in bottom layer; uniform upward flux
        c = zeros(Nx, Ny, Nz)
        c[:, :, Nz] .= 1.0
        tracers = (; c)

        mass_flux = zeros(Nx, Ny, Nz + 1)
        mass_flux[:, :, 2:Nz] .= 0.01
        met = (; conv_mass_flux = mass_flux)

        mass_before = sum(Δz(k, grid) * c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        convect!(tracers, met, grid, conv, Δt)
        mass_after = sum(Δz(k, grid) * tracers.c[i, j, k] for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        @test mass_after ≈ mass_before rtol = 1e-10

        # Tracer should have moved up: layer above gained mass
        @test sum(tracers.c[:, :, Nz-1]) > 0
    end
end
