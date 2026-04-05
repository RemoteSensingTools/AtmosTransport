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
using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
const ModelPhases = AtmosTransport.Models

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

    @testset "Z donor convention and clamp invariants" begin
        m = reshape(FT[2.0, 5.0, 7.0], 1, 1, 3)

        cm_pos = zeros(FT, 1, 1, 4)
        cm_pos[1, 1, 2] = FT(1.6)
        @test max_cfl_massflux_z(cm_pos, m) ≈ FT(0.8) atol=1e-12

        cm_neg = zeros(FT, 1, 1, 4)
        cm_neg[1, 1, 2] = FT(-2.0)
        @test max_cfl_massflux_z(cm_neg, m) ≈ FT(0.4) atol=1e-12

        cm_clamped = zeros(FT, 1, 1, 4)
        cm_clamped[1, 1, 1] = FT(3.0)
        cm_clamped[1, 1, 2] = FT(4.0)
        cm_clamped[1, 1, 3] = FT(-7.0)
        cm_clamped[1, 1, 4] = FT(-2.0)
        ModelPhases._clamp_cm_cfl!(cm_clamped, m, FT(0.95))
        @test cm_clamped[1, 1, 1] == FT(0)
        @test cm_clamped[1, 1, 2] ≈ FT(1.9) atol=1e-12
        @test cm_clamped[1, 1, 3] ≈ FT(-6.65) atol=1e-12
        @test cm_clamped[1, 1, 4] == FT(0)
        @test max_cfl_massflux_z(cm_clamped, m) <= FT(0.95) + 1e-12

        mass_before_air = sum(m)
        m_work = copy(m)
        rm = (; c = copy(m_work))
        mass_before_tracer = sum(rm.c)
        advect_z_massflux!(rm, m_work, cm_clamped, true)
        @test sum(m_work) ≈ mass_before_air atol=1e-12
        @test sum(rm.c) ≈ mass_before_tracer atol=1e-12
        @test minimum(m_work) > 0
        @test minimum(rm.c) > 0
    end

    @testset "Z clamp preserves column mass balance" begin
        m = reshape(FT[
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
            2.5, 3.5, 4.5,
            5.5, 6.5, 7.5,
        ], 2, 2, 3)
        c = reshape(FT[
            410.0, 415.0, 420.0,
            425.0, 430.0, 435.0,
            440.0, 445.0, 450.0,
            455.0, 460.0, 465.0,
        ], 2, 2, 3)
        rm_init = c .* m

        cm = zeros(FT, 2, 2, 4)
        cm[:, :, 1] .= FT(9.0)
        cm[:, :, 2] .= FT.([7.0  -12.0; -4.0  10.0])
        cm[:, :, 3] .= FT.([-20.0  14.0; 11.0  -18.0])
        cm[:, :, 4] .= FT(-6.0)

        ModelPhases._clamp_cm_cfl!(cm, m, FT(0.95))

        @test all(cm[:, :, 1] .== 0)
        @test all(cm[:, :, 4] .== 0)
        @test max_cfl_massflux_z(cm, m) <= FT(0.95) + 1e-12

        air_before_cols = dropdims(sum(m, dims=3), dims=3)
        tracer_before_cols = dropdims(sum(rm_init, dims=3), dims=3)
        m_work = copy(m)
        rm = (; c = copy(rm_init))
        advect_z_massflux!(rm, m_work, cm, true)

        air_after_cols = dropdims(sum(m_work, dims=3), dims=3)
        tracer_after_cols = dropdims(sum(rm.c, dims=3), dims=3)
        @test maximum(abs.(air_after_cols .- air_before_cols)) < 1e-12
        @test maximum(abs.(tracer_after_cols .- tracer_before_cols)) < 1e-12
        @test sum(m_work) ≈ sum(m) atol=1e-12
        @test sum(rm.c) ≈ sum(rm_init) atol=1e-12
    end

    @testset "Limiter degrades safely for negative rm" begin
        @test Advection._limit_mass_slope(FT(3), FT(-2)) == FT(0)
        @test Advection._limit_mass_slope(FT(-3), FT(-2)) == FT(0)
        @test Advection._limit_mass_slope(FT(3), FT(2)) == FT(2)
        @test Advection._limit_mass_slope(FT(-3), FT(2)) == FT(-2)
    end

    @testset "Uniform field preservation (rm form)" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        # Tracers stored as rm = c × m (TM5-style)
        c_uniform = FT(420.0)
        rm_init = fill(c_uniform, Nx, Ny, Nz) .* m
        tracers = (; c = copy(rm_init))

        u = randn(FT, Nx+1, Ny, Nz) .* 5.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = randn(FT, Nx, Ny+1, Nz) .* 3.0
        v[:, 1, :] .= 0
        v[:, Ny+1, :] .= 0
        half_dt = FT(100.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)
        ws = allocate_massflux_workspace(m, mf.am, mf.bm, mf.cm)
        m_work = copy(m)
        strang_split_massflux!(tracers, m_work, mf.am, mf.bm, mf.cm, grid, true, ws;
                                cfl_limit = FT(0.95))

        # After advection, c = rm / m_evolved should still be ~uniform
        c_after = tracers.c ./ m_work
        max_deviation = maximum(abs.(c_after .- c_uniform))
        @info "  Uniform field max deviation: $(max_deviation)"
        @test max_deviation < 1e-8
    end

    @testset "Full Strang split mass conservation (rm form)" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        ps = fill(FT(101325.0), Nx, Ny)
        Δp = Advection._build_Δz_3d(grid, ps)
        m = compute_air_mass(Δp, grid)

        c_init = fill(FT(420.0), Nx, Ny, Nz)
        ci, cj, ck = Nx ÷ 2, Ny ÷ 2, Nz ÷ 2
        c_init[ci-1:ci+1, cj-1:cj+1, ck-1:ck+1] .= FT(500.0)
        # Store as rm = c × m (TM5-style)
        rm_init = c_init .* m
        tracers = (; c = copy(rm_init))

        u = randn(FT, Nx+1, Ny, Nz) .* 8.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = randn(FT, Nx, Ny+1, Nz) .* 5.0
        v[:, 1, :] .= 0
        v[:, Ny+1, :] .= 0
        half_dt = FT(200.0)

        mf = compute_mass_fluxes(u, v, grid, Δp, half_dt)
        ws = allocate_massflux_workspace(m, mf.am, mf.bm, mf.cm)

        # Initial total tracer mass = sum(rm)
        mass_init = sum(rm_init)

        # Run 10 Strang splits — rm persists, m resets each time (like TM5)
        for _ in 1:10
            m_fresh = compute_air_mass(Δp, grid)
            strang_split_massflux!(tracers, m_fresh, mf.am, mf.bm, mf.cm, grid, true, ws;
                                    cfl_limit = FT(0.95))
        end

        mass_final = sum(tracers.c)
        rel_error = abs(mass_final - mass_init) / abs(mass_init)
        @info "  10-step Strang split mass drift: $(rel_error * 100)%"
        # With rm as prognostic + workspace, conservation is machine-epsilon
        @test rel_error < 1e-12
    end

    @testset "Positivity with limiter (rm form)" begin
        grid = make_test_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        # Tracer mass with a sharp spike
        c = fill(FT(1.0), Nx, Ny, Nz)
        c[Nx÷2, Ny÷2, Nz÷2] = FT(1000.0)
        rm = c .* m
        tracers = (; c = rm)

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

@testset "Reduced-grid mass-flux X-advection" begin

    function make_reduced_grid(; Nx=360, Ny=180, Nz=5)
        A = FT.(range(0.0, 0.0, length=Nz+1))
        B = FT.(range(0.0, 1.0, length=Nz+1))
        vc = HybridSigmaPressure(A, B)
        grid = LatitudeLongitudeGrid(CPU();
            FT, size=(Nx, Ny, Nz),
            longitude=(-180.0, 180.0),
            latitude=(-90.0, 90.0),
            vertical=vc,
            use_reduced_grid=true)
        return grid
    end

    @testset "Reduced grid is activated at 1-degree" begin
        grid = make_reduced_grid()
        @test grid.reduced_grid !== nothing
        rg = grid.reduced_grid
        @test any(rg.cluster_sizes .> 1)
        n_reduced = count(>(1), rg.cluster_sizes)
        max_r = maximum(rg.cluster_sizes)
        @info "  Reduced grid: $n_reduced/$(grid.Ny) rows reduced, max cluster=$max_r"
    end

    @testset "Mass conservation (reduced-grid x-advection)" begin
        grid = make_reduced_grid(Nx=72, Ny=36, Nz=5)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm_init = m .* FT(420.0)
        rm = (; c = copy(rm_init))

        u = zeros(FT, Nx+1, Ny, Nz) .+ 10.0
        half_dt = FT(450.0)
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, half_dt)

        mass_before = sum(rm.c)
        advect_x_massflux_reduced!(rm, m, mf.am, grid, true)
        mass_after = sum(rm.c)

        rel_error = abs(mass_after - mass_before) / abs(mass_before)
        @info "  Reduced-grid x-advection mass conservation: $(rel_error)"
        @test rel_error < 1e-10
    end

    @testset "Uniform field preservation (reduced-grid x-advection)" begin
        grid = make_reduced_grid(Nx=72, Ny=36, Nz=5)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        rm = (; c = m .* FT(420.0))

        u = randn(FT, Nx+1, Ny, Nz) .* 5.0
        u[Nx+1, :, :] .= u[1, :, :]
        half_dt = FT(200.0)
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, half_dt)

        advect_x_massflux_reduced!(rm, m, mf.am, grid, true)

        c_after = rm.c ./ m
        max_deviation = maximum(abs.(c_after .- 420.0))
        @info "  Reduced-grid uniform preservation: max_dev=$(max_deviation)"
        @test max_deviation < 1e-8
    end

    @testset "CFL reduction on reduced grid" begin
        grid = make_reduced_grid(Nx=360, Ny=180, Nz=3)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp(grid)
        m = compute_air_mass(Δp, grid)

        u = zeros(FT, Nx+1, Ny, Nz) .+ 10.0
        half_dt = FT(1800.0)
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, half_dt)

        cfl_full = max_cfl_massflux_x(mf.am, m)
        @info "  CFL on full grid (no reduction): $(round(cfl_full, digits=2))"
        @test cfl_full > 1.0

        rg = grid.reduced_grid
        if rg !== nothing
            max_cfl_reduced = zero(FT)
            for j in 1:Ny
                r = rg.cluster_sizes[j]
                if r > 1
                    Nx_red = rg.reduced_counts[j]
                    m_red = zeros(FT, Nx_red)
                    am_red = zeros(FT, Nx_red + 1)
                    reduce_row_mass!(m_red, m, j, 1, r, Nx)
                    reduce_am_row!(am_red, mf.am, j, 1, r, Nx)
                    for i_r in 1:Nx_red
                        cfl_r = abs(am_red[i_r]) / m_red[i_r]
                        max_cfl_reduced = max(max_cfl_reduced, cfl_r)
                    end
                end
            end
            @info "  Max CFL on reduced rows: $(round(max_cfl_reduced, digits=3))"
            @test max_cfl_reduced < cfl_full
            @test max_cfl_reduced < 1.0
        end
    end
end

@info "\nAll mass-flux advection tests passed!"
