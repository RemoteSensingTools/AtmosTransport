using Test
using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Advection
using AtmosTransportModel.Grids

@testset "Mass Correction (Pressure Fixer)" begin

    # Shared grid setup: realistic hybrid-sigma with varying Δp
    vc = HybridSigmaPressure(
        [0.0, 5000.0, 10000.0, 20000.0, 50000.0, 101325.0],
        [0.0, 0.0, 0.1, 0.3, 0.7, 1.0])
    grid = LatitudeLongitudeGrid(CPU();
        size = (36, 18, 5),
        longitude = (-180, 180),
        latitude = (-90, 90),
        vertical = vc)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    FT = Float64

    function build_Δp(grid, FT)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = Array{FT}(undef, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δp[i, j, k] = level_thickness(grid.vertical, k, grid.reference_pressure)
        end
        return Δp
    end

    function pressure_weighted_mass(c, Δp, grid)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mass = zero(eltype(c))
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            mass += c[i, j, k] * Δp[i, j, k] * cell_area(i, j, grid) / grid.gravity
        end
        return mass
    end

    @testset "update_pressure_x! conserves column mass" begin
        u = fill(15.0, Nx + 1, Ny, Nz)
        Δp = build_Δp(grid, FT)
        total_before = sum(Δp)
        update_pressure_x!(Δp, u, grid, 60.0)
        total_after = sum(Δp)
        @test total_after ≈ total_before rtol = 1e-12
        @test all(Δp .> 0)
    end

    @testset "update_pressure_y! conserves global mass" begin
        v = fill(8.0, Nx, Ny + 1, Nz)
        v[:, 1, :]   .= 0.0  # zero at south pole
        v[:, end, :] .= 0.0  # zero at north pole
        Δp = build_Δp(grid, FT)
        total_before = sum(Δp)
        update_pressure_y!(Δp, v, grid, 60.0)
        total_after = sum(Δp)
        @test total_after ≈ total_before rtol = 1e-10
        @test all(Δp .> 0)
    end

    @testset "update_pressure_z! conserves column Δp" begin
        w = zeros(FT, Nx, Ny, Nz + 1)
        for k in 2:Nz, j in 1:Ny, i in 1:Nx
            w[i, j, k] = 0.5 * sin(2π * k / Nz)
        end
        Δp = build_Δp(grid, FT)
        col_sums_before = [sum(Δp[i, j, :]) for i in 1:Nx, j in 1:Ny]
        update_pressure_z!(Δp, w, 60.0)
        col_sums_after = [sum(Δp[i, j, :]) for i in 1:Nx, j in 1:Ny]
        @test col_sums_after ≈ col_sums_before rtol = 1e-12
        @test all(Δp .> 0)
    end

    @testset "Uniform field stays uniform with mass correction (x)" begin
        scheme = SlopesAdvection(use_limiter = true)
        c = fill(FT(420.0), Nx, Ny, Nz)
        tracers = (; c)
        u = fill(FT(15.0), Nx + 1, Ny, Nz)
        v = zeros(FT, Nx, Ny + 1, Nz)
        w = zeros(FT, Nx, Ny, Nz + 1)
        vel = (; u, v, w)
        Δp = build_Δp(grid, FT)

        advect_x_mass_corrected!(tracers, vel, grid, scheme, 60.0, Δp)

        @test minimum(tracers.c) ≈ 420.0 rtol = 1e-10
        @test maximum(tracers.c) ≈ 420.0 rtol = 1e-10
    end

    @testset "Uniform field stays uniform with mass correction (y)" begin
        scheme = SlopesAdvection(use_limiter = true)
        c = fill(FT(420.0), Nx, Ny, Nz)
        tracers = (; c)
        u = zeros(FT, Nx + 1, Ny, Nz)
        v = fill(FT(8.0), Nx, Ny + 1, Nz)
        v[:, 1, :]   .= 0.0
        v[:, end, :] .= 0.0
        w = zeros(FT, Nx, Ny, Nz + 1)
        vel = (; u, v, w)
        Δp = build_Δp(grid, FT)

        advect_y_mass_corrected!(tracers, vel, grid, scheme, 60.0, Δp)

        @test minimum(tracers.c) ≈ 420.0 rtol = 1e-10
        @test maximum(tracers.c) ≈ 420.0 rtol = 1e-10
    end

    @testset "Uniform field stays uniform with mass correction (z)" begin
        scheme = SlopesAdvection(use_limiter = true)
        c = fill(FT(420.0), Nx, Ny, Nz)
        tracers = (; c)
        u = zeros(FT, Nx + 1, Ny, Nz)
        v = zeros(FT, Nx, Ny + 1, Nz)
        w = zeros(FT, Nx, Ny, Nz + 1)
        for k in 2:Nz, j in 1:Ny, i in 1:Nx
            w[i, j, k] = 0.3
        end
        vel = (; u, v, w)
        Δp = build_Δp(grid, FT)

        advect_z_mass_corrected!(tracers, vel, grid, scheme, 60.0, Δp)

        @test minimum(tracers.c) ≈ 420.0 rtol = 1e-10
        @test maximum(tracers.c) ≈ 420.0 rtol = 1e-10
    end

    @testset "Mass conservation with full Strang split + mass correction" begin
        scheme = SlopesAdvection(use_limiter = true)
        c = zeros(FT, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            c[i, j, k] = 400.0 + 40.0 * sin(2π * i / Nx) * cos(π * j / Ny)
        end
        tracers = (; c)

        u = zeros(FT, Nx + 1, Ny, Nz)
        v = zeros(FT, Nx, Ny + 1, Nz)
        w = zeros(FT, Nx, Ny, Nz + 1)
        for k in 1:Nz, j in 1:Ny
            for i in 1:(Nx+1)
                u[i, j, k] = 10.0 * cosd(grid.φᶜ[j])
            end
        end
        for k in 1:Nz, j in 2:Ny, i in 1:Nx
            v[i, j, k] = 3.0 * sind(2 * grid.φᶠ[j])
        end
        for k in 2:Nz, j in 1:Ny, i in 1:Nx
            w[i, j, k] = 0.1 * sin(2π * k / Nz)
        end
        vel = (; u, v, w)

        Δp = build_Δp(grid, FT)
        mass_before = pressure_weighted_mass(tracers.c, Δp, grid)

        Δt_half = 300.0

        # X→Y→Z→Z→Y→X Strang split
        Δp_strang = copy(Δp)
        advect_x_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)
        advect_y_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)
        advect_z_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)
        advect_z_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)
        advect_y_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)
        advect_x_mass_corrected!(tracers, vel, grid, scheme, Δt_half, Δp_strang)

        mass_after = pressure_weighted_mass(tracers.c, Δp_strang, grid)
        Δmass_pct = abs(mass_after - mass_before) / abs(mass_before) * 100

        @test Δmass_pct < 0.5
    end

    @testset "Δp stays positive with subcycled mass correction" begin
        scheme = SlopesAdvection(use_limiter = true)
        c = fill(FT(420.0), Nx, Ny, Nz)
        tracers = (; c)

        u = zeros(FT, Nx + 1, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:(Nx+1)
            u[i, j, k] = 25.0 * cosd(grid.φᶜ[min(j, Ny)])
        end
        v = zeros(FT, Nx, Ny + 1, Nz)
        w = zeros(FT, Nx, Ny, Nz + 1)
        vel = (; u, v, w)

        Δp = build_Δp(grid, FT)
        advect_x_mass_corrected_subcycled!(tracers, vel, grid, scheme, 600.0, Δp;
                                            cfl_limit = 0.95)

        @test all(Δp .> 0)
        @test minimum(tracers.c) > 0
    end

    @testset "Mass correction eliminates operator-splitting extremes" begin
        scheme = SlopesAdvection(use_limiter = true)
        c_with    = fill(FT(420.0), Nx, Ny, Nz)
        c_without = fill(FT(420.0), Nx, Ny, Nz)

        u = zeros(FT, Nx + 1, Ny, Nz)
        v = zeros(FT, Nx, Ny + 1, Nz)
        w = zeros(FT, Nx, Ny, Nz + 1)
        for k in 1:Nz, j in 1:Ny
            for i in 1:(Nx+1)
                u[i, j, k] = 15.0 * cosd(grid.φᶜ[min(j, Ny)]) + 5.0 * sin(2π * i / Nx)
            end
        end
        for k in 1:Nz, j in 2:Ny, i in 1:Nx
            v[i, j, k] = 5.0 * sind(2 * grid.φᶠ[j])
        end
        for k in 2:Nz, j in 1:Ny, i in 1:Nx
            w[i, j, k] = 0.2 * sin(2π * k / Nz)
        end
        vel = (; u, v, w)

        Δt_half = 300.0
        Δp_mc = build_Δp(grid, FT)
        tracers_mc = (; c = c_with)
        tracers_no = (; c = c_without)

        for _ in 1:5
            # With mass correction
            advect_x_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)
            advect_y_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)
            advect_z_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)
            advect_z_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)
            advect_y_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)
            advect_x_mass_corrected!(tracers_mc, vel, grid, scheme, Δt_half, Δp_mc)

            # Without mass correction
            advect_x!(tracers_no, vel, grid, scheme, Δt_half)
            advect_y!(tracers_no, vel, grid, scheme, Δt_half)
            advect_z!(tracers_no, vel, grid, scheme, Δt_half)
            advect_z!(tracers_no, vel, grid, scheme, Δt_half)
            advect_y!(tracers_no, vel, grid, scheme, Δt_half)
            advect_x!(tracers_no, vel, grid, scheme, Δt_half)
        end

        range_with    = maximum(tracers_mc.c) - minimum(tracers_mc.c)
        range_without = maximum(tracers_no.c) - minimum(tracers_no.c)

        @test range_with < range_without
    end
end
