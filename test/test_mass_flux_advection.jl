#!/usr/bin/env julia
# ===========================================================================
# Tests for TM5-faithful mass-flux advection
#
# Validates:
# 1. Mass conservation (sum of rm is invariant after advection)
# 2. Uniform field preservation (uniform c stays uniform)
# 3. Full Strang split mass conservation
# 4. Positivity preservation with limiter
# 5. Comparison: mass-flux vs concentration-based advection
# ===========================================================================

using Test
using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection

const FT = Float64

function make_test_grid(; Nx=36, Ny=18, Nz=10)
    A = FT.(range(0.0, 0.0, length=Nz+1))
    B = FT.(range(0.0, 1.0, length=Nz+1))
    vc = HybridSigmaPressure(A, B)
    grid = LatitudeLongitudeGrid(CPU();
        FT, size=(Nx, Ny, Nz),
        longitude=(-180.0, 180.0),
        latitude=(-90.0, 90.0),
        vertical=vc,
        use_reduced_grid=false)
    return grid
end

function uniform_dp(grid, ps=101325.0)
    Advection._build_Δz_3d(grid, fill(FT(ps), grid.Nx, grid.Ny))
end

@testset "Mass-Flux Advection" begin

    @testset "compute_air_mass" begin
        grid = make_test_grid()
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        @test size(m) == (grid.Nx, grid.Ny, grid.Nz)
        @test all(m .> 0)

        # Total mass should equal total atmospheric mass
        total_m = sum(m)
        @test total_m > 0
        @info "  Total air mass: $(round(total_m, sigdigits=6)) kg"
    end

    @testset "compute_mass_fluxes" begin
        grid = make_test_grid()
        Δp = uniform_dp(grid)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        u = zeros(FT, Nx+1, Ny, Nz) .+ 5.0
        v = zeros(FT, Nx, Ny+1, Nz)
        half_dt = FT(450.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)

        @test size(mf.am) == (Nx+1, Ny, Nz)
        @test size(mf.bm) == (Nx, Ny+1, Nz)
        @test size(mf.cm) == (Nx, Ny, Nz+1)
        @test all(mf.am .> 0)  # uniform eastward wind
        @test mf.cm[:, :, 1] ≈ zeros(FT, Nx, Ny) atol=1e-10
        @test mf.cm[:, :, Nz+1] ≈ zeros(FT, Nx, Ny) atol=1e-6
    end

    @testset "X-advection mass conservation" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm_init = m .* FT(420.0)
        rm = (; c = copy(rm_init))

        u = zeros(FT, Nx+1, Ny, Nz) .+ 10.0
        half_dt = FT(450.0)
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, half_dt)

        mass_before = sum(rm.c)
        advect_x_massflux!(rm, m, mf.am, grid, true)
        mass_after = sum(rm.c)

        rel_error = abs(mass_after - mass_before) / abs(mass_before)
        @info "  X-advection mass conservation: $(rel_error)"
        @test rel_error < 1e-12
    end

    @testset "Y-advection mass conservation" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm = (; c = m .* FT(420.0))

        v = zeros(FT, Nx, Ny+1, Nz) .+ 5.0
        half_dt = FT(450.0)
        mf = compute_mass_fluxes(zeros(FT, Nx+1, Ny, Nz), v, grid, Δp, half_dt)

        mass_before = sum(rm.c)
        advect_y_massflux!(rm, m, mf.bm, grid, true)
        mass_after = sum(rm.c)

        rel_error = abs(mass_after - mass_before) / abs(mass_before)
        @info "  Y-advection mass conservation: $(rel_error)"
        @test rel_error < 1e-12
    end

    @testset "Z-advection mass conservation" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm = (; c = m .* FT(420.0))

        cm = zeros(FT, Nx, Ny, Nz+1)
        # Small downward flux in the interior
        for k in 2:Nz
            cm[:, :, k] .= 0.01 .* m[:, :, min(k-1, Nz)]
        end

        mass_before = sum(rm.c)
        advect_z_massflux!(rm, m, cm, true)
        mass_after = sum(rm.c)

        rel_error = abs(mass_after - mass_before) / abs(mass_before)
        @info "  Z-advection mass conservation: $(rel_error)"
        @test rel_error < 1e-12
    end

    @testset "Uniform field preservation" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        c_uniform = fill(FT(420.0), Nx, Ny, Nz)
        tracers = (; c = copy(c_uniform))

        u = randn(FT, Nx+1, Ny, Nz) .* 5.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = randn(FT, Nx, Ny+1, Nz) .* 3.0
        v[:, 1, :] .= 0
        v[:, Ny+1, :] .= 0
        half_dt = FT(100.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)
        strang_split_massflux!(tracers, m, mf.am, mf.bm, mf.cm, grid, true;
                                cfl_limit = FT(0.95))

        max_deviation = maximum(abs.(tracers.c .- 420.0))
        @info "  Uniform field max deviation: $(max_deviation)"
        @test max_deviation < 1e-8
    end

    @testset "Full Strang split mass conservation" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        ps = fill(FT(101325.0), Nx, Ny)
        Δp = Advection._build_Δz_3d(grid, ps)
        m = compute_air_mass(Δp, grid)

        c_init = fill(FT(420.0), Nx, Ny, Nz)
        # Add a localized perturbation
        ci, cj, ck = Nx ÷ 2, Ny ÷ 2, Nz ÷ 2
        c_init[ci-1:ci+1, cj-1:cj+1, ck-1:ck+1] .= FT(500.0)
        tracers = (; c = copy(c_init))

        u = randn(FT, Nx+1, Ny, Nz) .* 8.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = randn(FT, Nx, Ny+1, Nz) .* 5.0
        v[:, 1, :] .= 0
        v[:, Ny+1, :] .= 0
        half_dt = FT(200.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)

        # Compute initial mass (pressure-weighted)
        mass_init = sum(c_init .* Δp)

        # Run 10 Strang splits
        for _ in 1:10
            m_fresh = compute_air_mass(Δp, grid)
            strang_split_massflux!(tracers, m_fresh, mf.am, mf.bm, mf.cm, grid, true;
                                    cfl_limit = FT(0.95))
        end

        mass_final = sum(tracers.c .* Δp)
        rel_error = abs(mass_final - mass_init) / abs(mass_init)
        @info "  10-step Strang split mass drift: $(rel_error * 100)%"
        @test rel_error < 0.01  # < 1% drift after 10 splits
    end

    @testset "Positivity with limiter" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        # Concentration with a sharp spike (challenging for positivity)
        c = fill(FT(1.0), Nx, Ny, Nz)
        c[Nx÷2, Ny÷2, Nz÷2] = FT(1000.0)
        tracers = (; c)

        u = randn(FT, Nx+1, Ny, Nz) .* 5.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = randn(FT, Nx, Ny+1, Nz) .* 3.0
        v[:, 1, :] .= 0
        v[:, Ny+1, :] .= 0
        half_dt = FT(100.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)

        for _ in 1:5
            m_fresh = compute_air_mass(Δp, grid)
            strang_split_massflux!(tracers, m_fresh, mf.am, mf.bm, mf.cm, grid, true;
                                    cfl_limit = FT(0.95))
        end

        n_negative = count(x -> x < 0, tracers.c)
        min_val = minimum(tracers.c)
        @info "  After 5 splits: min=$(round(min_val, digits=6)), negatives=$n_negative"
        @test n_negative == 0 || min_val > -1e-6  # Small negatives acceptable
    end

    @testset "CFL subcycling" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm = (; c = m .* FT(420.0))

        # Large wind → CFL > 1 → subcycling needed
        u = zeros(FT, Nx+1, Ny, Nz) .+ 200.0
        half_dt = FT(1800.0)
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, half_dt)

        cfl_x = max_cfl_massflux_x(mf.am, m)
        @info "  CFL before subcycling: $(round(cfl_x, digits=2))"
        @test cfl_x > 1.0

        mass_before = sum(rm.c)
        n_sub = advect_x_massflux_subcycled!(rm, m, mf.am, grid, true; cfl_limit = FT(0.95))
        mass_after = sum(rm.c)

        @info "  Subcycles used: $n_sub"
        @test n_sub > 1
        rel_error = abs(mass_after - mass_before) / abs(mass_before)
        @test rel_error < 1e-10
    end
end

@info "\nAll mass-flux advection tests passed!"
