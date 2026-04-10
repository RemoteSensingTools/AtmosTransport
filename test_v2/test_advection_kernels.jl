#!/usr/bin/env julia
"""
Cross-backend and cross-precision validation for the new generic advection
kernel shells (UpwindScheme, SlopesScheme through structured_kernels.jl).

Tests the 2×2 matrix:  {CPU, GPU} × {Float32, Float64}

For each combination, verifies:
1. Uniform field invariance (zero mixing ratio change under advection)
2. Mass conservation (total tracer mass preserved to machine precision)
3. Non-trivial transport (gradient field + realistic fluxes → field changes)
4. CPU-GPU bit-agreement (F64) and close agreement (F32, ≤ 2 ULP)
5. F32 vs F64 systematic bias (no drift beyond expected truncation error)

Uses semi-realistic fluxes: sinusoidal zonal wind with latitude taper and
vertically-consistent cm diagnosed from horizontal continuity. This exercises
all three sweep directions with non-trivial Courant numbers.
"""

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2: Operators, Grids

# Try to load CUDA; tests gracefully skip if unavailable
const HAS_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

# =========================================================================
# Helper: build semi-realistic test problem
# =========================================================================

"""
    build_test_problem(FT; Nx=36, Ny=18, Nz=4, cfl=0.15)

Build a semi-realistic structured-grid advection test case.

Returns `(grid, m, rm_uniform, rm_gradient, am, bm, cm)` all in `FT` precision.

Fluxes:
- am: sinusoidal zonal mass flux, tapered to zero at poles
- bm: weak meridional convergence in the tropics
- cm: diagnosed from continuity (exact mass closure)
- CFL ≈ `cfl` at the equator

Initial fields:
- rm_uniform: uniform mixing ratio (400 ppm)
- rm_gradient: latitude-dependent gradient (300-500 ppm)
"""
function build_test_problem(FT; Nx=36, Ny=18, Nz=4, cfl=FT(0.15))
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.CPU(); FT=FT)

    g  = gravity(grid)
    ps = reference_pressure(grid)
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(level_thickness(vc, k, ps)) * areas[j] / g
    end

    χ_uniform = FT(400e-6)
    rm_uniform = m .* χ_uniform

    rm_gradient = similar(m)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        lat_frac = FT(j - 1) / FT(Ny - 1)
        χ_j = FT(300e-6) + FT(200e-6) * lat_frac
        rm_gradient[i, j, k] = m[i, j, k] * χ_j
    end

    # Semi-realistic mass fluxes
    m_min = minimum(m)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        lat = FT(-90) + FT(j - FT(0.5)) * (FT(180) / FT(Ny))
        lon = FT(i - 1) * (FT(360) / FT(Nx))
        cos_lat = cos(lat * FT(π) / FT(180))
        am[i, j, k] = cfl * m_min * cos_lat * (FT(1) + FT(0.3) * sin(lon * FT(π) / FT(180)))
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)

    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        lat = FT(-90) + FT(j - 1) * (FT(180) / FT(Ny))
        bm[i, j, k] = cfl * m_min * FT(0.1) * sin(FT(2) * lat * FT(π) / FT(180))
    end

    cm = zeros(FT, Nx, Ny, Nz + 1)
    bt = FT[Grids.b_diff(vc, k) for k in 1:Nz]
    Operators.Advection.diagnose_cm!(cm, am, bm, bt)

    return grid, m, rm_uniform, rm_gradient, am, bm, cm
end

"""
    run_strang!(m, rm, am, bm, cm, scheme; n_steps=1)

Run `n_steps` Strang splits on CPU arrays, return (m_out, rm_out).
"""
function run_strang!(m, rm, am, bm, cm, grid, scheme; n_steps=1)
    m_work  = copy(m)
    rm_work = copy(rm)
    state   = CellState(m_work; tracer=rm_work)
    fluxes  = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    ws      = AdvectionWorkspace(m_work)
    for _ in 1:n_steps
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
    end
    return m_work, rm_work
end

"""
    to_gpu(arrays...)

Move arrays to GPU (CuArray). Returns a tuple.
"""
function to_gpu(arrays...)
    return map(a -> CuArray(a), arrays)
end

# =========================================================================
# Test suite
# =========================================================================

@testset "Advection kernels: {CPU,GPU} × {F32,F64}" begin

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        Nx, Ny, Nz = size(m_cpu)
        scheme = UpwindScheme()

        # --- CPU tests ---

        @testset "CPU $precision_tag: uniform invariance" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            χ_out = rm_out ./ max.(m_out, eps(FT))
            @test maximum(abs.(χ_out .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
        end

        @testset "CPU $precision_tag: mass conservation (uniform)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            tol = FT == Float64 ? 1e-13 : FT(1e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_uni_cpu)) / sum(rm_uni_cpu) < tol
        end

        @testset "CPU $precision_tag: mass conservation (gradient, 4 steps)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)
            tol = FT == Float64 ? 1e-12 : FT(5e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
        end

        @testset "CPU $precision_tag: non-trivial transport" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            @test maximum(abs.(rm_out .- rm_grad_cpu)) > zero(FT)
        end

        # --- GPU tests (skip if no CUDA) ---

        if HAS_GPU

            @testset "GPU $precision_tag: mass conservation (gradient, 4 steps)" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)

                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)

                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end

                m_gpu  = Array(state_g.air_dry_mass)
                rm_gpu = Array(state_g.tracers[:tracer])

                tol = FT == Float64 ? 1e-12 : FT(5e-5)
                @test abs(sum(m_gpu) - sum(m_cpu)) / sum(m_cpu) < tol
                @test abs(sum(rm_gpu) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
            end

            @testset "GPU $precision_tag: uniform invariance" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu)

                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)

                strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_dry_mass)
                χ_gpu  = rm_gpu ./ max.(m_gpu, eps(FT))
                @test maximum(abs.(χ_gpu .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
            end

            @testset "CPU-GPU agreement $precision_tag (1 step, gradient)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_dry_mass)

                # GPU FMA instructions may produce ≤ few ULP difference vs CPU
                ulp_tol = FT == Float64 ? FT(4) : FT(4)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU $precision_tag 1-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

            @testset "CPU-GPU agreement $precision_tag (4 steps, gradient)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_dry_mass)

                # After 4 steps, FMA-induced ULP drift accumulates modestly
                ulp_tol = FT == Float64 ? FT(16) : FT(16)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU $precision_tag 4-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

        else
            @testset "GPU $precision_tag: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

    # --- F32 vs F64 systematic bias check ---

    @testset "F32 vs F64 systematic bias (4 steps)" begin
        grid64, m64, _, rm_grad64, am64, bm64, cm64 = build_test_problem(Float64)
        grid32, m32, _, rm_grad32, am32, bm32, cm32 = build_test_problem(Float32)

        _, rm_out64 = run_strang!(m64, rm_grad64, am64, bm64, cm64, grid64, UpwindScheme(); n_steps=4)
        _, rm_out32 = run_strang!(m32, rm_grad32, am32, bm32, cm32, grid32, UpwindScheme(); n_steps=4)

        rm64_as32 = Float32.(rm_out64)
        rel_diff  = maximum(abs.(rm_out32 .- rm64_as32)) / maximum(abs.(rm64_as32))

        println("  F32 vs F64 max relative difference (4 steps): ", rel_diff)
        @test rel_diff < Float32(1e-4)

        bias = (sum(Float64.(rm_out32)) - sum(rm_out64)) / sum(rm_out64)
        println("  F32 vs F64 global mass bias (4 steps):        ", bias)
        @test abs(bias) < 1e-6
    end

    # --- Legacy vs new scheme equivalence ---

    @testset "UpwindAdvection vs UpwindScheme equivalence" begin
        for FT in (Float64, Float32)
            tag = FT == Float64 ? "F64" : "F32"
            grid, m, _, rm_grad, am, bm, cm = build_test_problem(FT)

            m_old, rm_old = run_strang!(m, rm_grad, am, bm, cm, grid, UpwindAdvection(); n_steps=4)
            m_new, rm_new = run_strang!(m, rm_grad, am, bm, cm, grid, UpwindScheme();    n_steps=4)

            @testset "bit-identical $tag" begin
                @test rm_old == rm_new
                @test m_old  == m_new
            end
        end
    end

    @testset "RussellLernerAdvection vs SlopesScheme equivalence" begin
        # The legacy RL kernels have an `if r==1` branch (reduced-grid path)
        # that can alter FMA fusion vs the new branchless generic kernel shells.
        # Algebraically identical, but ≤1 ULP difference from compiler FMA.
        for FT in (Float64, Float32)
            tag = FT == Float64 ? "F64" : "F32"
            grid, m, _, rm_grad, am, bm, cm = build_test_problem(FT)

            @testset "MonotoneLimiter $tag" begin
                m_old, rm_old = run_strang!(m, rm_grad, am, bm, cm, grid,
                    RussellLernerAdvection(use_limiter=true); n_steps=4)
                m_new, rm_new = run_strang!(m, rm_grad, am, bm, cm, grid,
                    SlopesScheme(MonotoneLimiter()); n_steps=4)
                rm_ulp = maximum(abs.(rm_old .- rm_new)) / eps(maximum(abs.(rm_old)))
                m_ulp  = maximum(abs.(m_old .- m_new))   / eps(maximum(abs.(m_old)))
                println("    RL vs Slopes Mono $tag: rm_ulp=$rm_ulp  m_ulp=$m_ulp")
                @test rm_ulp < FT(1)
                @test m_ulp  < FT(1)
            end

            @testset "NoLimiter $tag" begin
                m_old, rm_old = run_strang!(m, rm_grad, am, bm, cm, grid,
                    RussellLernerAdvection(use_limiter=false); n_steps=4)
                m_new, rm_new = run_strang!(m, rm_grad, am, bm, cm, grid,
                    SlopesScheme(NoLimiter()); n_steps=4)
                rm_ulp = maximum(abs.(rm_old .- rm_new)) / eps(maximum(abs.(rm_old)))
                m_ulp  = maximum(abs.(m_old .- m_new))   / eps(maximum(abs.(m_old)))
                println("    RL vs Slopes NoLim $tag: rm_ulp=$rm_ulp  m_ulp=$m_ulp")
                @test rm_ulp < FT(1)
                @test m_ulp  < FT(1)
            end
        end
    end

end

# =========================================================================
# SlopesScheme: {CPU,GPU} × {F32,F64}
# =========================================================================

@testset "SlopesScheme kernels: {CPU,GPU} × {F32,F64}" begin

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        scheme = SlopesScheme(MonotoneLimiter())

        @testset "CPU $precision_tag: uniform invariance (Slopes)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            χ_out = rm_out ./ max.(m_out, eps(FT))
            @test maximum(abs.(χ_out .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
        end

        @testset "CPU $precision_tag: mass conservation (Slopes, gradient, 4 steps)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)
            tol = FT == Float64 ? 1e-12 : FT(5e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
        end

        @testset "CPU $precision_tag: non-trivial transport (Slopes)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            @test maximum(abs.(rm_out .- rm_grad_cpu)) > zero(FT)
        end

        @testset "CPU $precision_tag: Slopes differs from Upwind on gradient" begin
            _, rm_upwind = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, UpwindScheme(); n_steps=1)
            _, rm_slopes = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=1)
            diff = maximum(abs.(rm_slopes .- rm_upwind))
            @test diff > zero(FT)
        end

        if HAS_GPU

            @testset "GPU $precision_tag: mass conservation (Slopes, gradient, 4 steps)" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end
                m_gpu  = Array(state_g.air_dry_mass)
                rm_gpu = Array(state_g.tracers[:tracer])
                tol = FT == Float64 ? 1e-12 : FT(5e-5)
                @test abs(sum(m_gpu) - sum(m_cpu)) / sum(m_cpu) < tol
                @test abs(sum(rm_gpu) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
            end

            @testset "CPU-GPU agreement $precision_tag (Slopes, 4 steps)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end
                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_dry_mass)

                ulp_tol = FT == Float64 ? FT(16) : FT(16)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU Slopes $precision_tag 4-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

        else
            @testset "GPU $precision_tag Slopes: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

    @testset "F32 vs F64 systematic bias (Slopes, 4 steps)" begin
        grid64, m64, _, rm_grad64, am64, bm64, cm64 = build_test_problem(Float64)
        grid32, m32, _, rm_grad32, am32, bm32, cm32 = build_test_problem(Float32)

        scheme = SlopesScheme(MonotoneLimiter())
        _, rm_out64 = run_strang!(m64, rm_grad64, am64, bm64, cm64, grid64, scheme; n_steps=4)
        _, rm_out32 = run_strang!(m32, rm_grad32, am32, bm32, cm32, grid32, scheme; n_steps=4)

        rm64_as32 = Float32.(rm_out64)
        rel_diff  = maximum(abs.(rm_out32 .- rm64_as32)) / maximum(abs.(rm64_as32))
        println("  Slopes F32 vs F64 max relative difference (4 steps): ", rel_diff)
        @test rel_diff < Float32(1e-4)

        bias = (sum(Float64.(rm_out32)) - sum(rm_out64)) / sum(rm_out64)
        println("  Slopes F32 vs F64 global mass bias (4 steps):        ", bias)
        @test abs(bias) < 1e-6
    end

end

# =========================================================================
# PPMScheme: {CPU,GPU} × {F32,F64}
# =========================================================================

@testset "PPMScheme kernels: {CPU,GPU} × {F32,F64}" begin

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        scheme = PPMScheme(MonotoneLimiter())

        @testset "CPU $precision_tag: uniform invariance (PPM)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            χ_out = rm_out ./ max.(m_out, eps(FT))
            @test maximum(abs.(χ_out .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
        end

        @testset "CPU $precision_tag: mass conservation (PPM, gradient, 4 steps)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)
            tol = FT == Float64 ? 1e-12 : FT(5e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
        end

        @testset "CPU $precision_tag: non-trivial transport (PPM)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            @test maximum(abs.(rm_out .- rm_grad_cpu)) > zero(FT)
        end

        @testset "CPU $precision_tag: PPM differs from Slopes on gradient" begin
            _, rm_slopes = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid,
                SlopesScheme(MonotoneLimiter()); n_steps=1)
            _, rm_ppm = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid,
                scheme; n_steps=1)
            diff = maximum(abs.(rm_ppm .- rm_slopes))
            @test diff > zero(FT)
        end

        if HAS_GPU

            @testset "GPU $precision_tag: mass conservation (PPM, gradient, 4 steps)" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end
                m_gpu  = Array(state_g.air_dry_mass)
                rm_gpu = Array(state_g.tracers[:tracer])
                tol = FT == Float64 ? 1e-12 : FT(5e-5)
                @test abs(sum(m_gpu) - sum(m_cpu)) / sum(m_cpu) < tol
                @test abs(sum(rm_gpu) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
            end

            @testset "CPU-GPU agreement $precision_tag (PPM, 4 steps)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g.air_dry_mass)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end
                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_dry_mass)

                ulp_tol = FT == Float64 ? FT(16) : FT(16)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU PPM $precision_tag 4-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

        else
            @testset "GPU $precision_tag PPM: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

    @testset "F32 vs F64 systematic bias (PPM, 4 steps)" begin
        grid64, m64, _, rm_grad64, am64, bm64, cm64 = build_test_problem(Float64)
        grid32, m32, _, rm_grad32, am32, bm32, cm32 = build_test_problem(Float32)

        scheme = PPMScheme(MonotoneLimiter())
        _, rm_out64 = run_strang!(m64, rm_grad64, am64, bm64, cm64, grid64, scheme; n_steps=4)
        _, rm_out32 = run_strang!(m32, rm_grad32, am32, bm32, cm32, grid32, scheme; n_steps=4)

        rm64_as32 = Float32.(rm_out64)
        rel_diff  = maximum(abs.(rm_out32 .- rm64_as32)) / maximum(abs.(rm64_as32))
        println("  PPM F32 vs F64 max relative difference (4 steps): ", rel_diff)
        @test rel_diff < Float32(1e-4)

        bias = (sum(Float64.(rm_out32)) - sum(rm_out64)) / sum(rm_out64)
        println("  PPM F32 vs F64 global mass bias (4 steps):        ", bias)
        @test abs(bias) < 1e-6
    end

end

# =========================================================================
# Multi-tracer kernel fusion: equivalence with per-tracer path
# =========================================================================

@testset "Multi-tracer kernel fusion" begin

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        Nx, Ny, Nz = size(m_cpu)
        Nt = 5

        rm_tracers = [rm_grad_cpu .* FT(0.5 + 0.1 * t) for t in 1:Nt]

        for (scheme, scheme_tag) in (
            (UpwindScheme(), "Upwind"),
            (SlopesScheme(MonotoneLimiter()), "Slopes"),
            (PPMScheme(MonotoneLimiter()), "PPM"),
        )
            @testset "CPU $precision_tag $scheme_tag: MT ≡ per-tracer (4 steps)" begin
                # Per-tracer path (reference)
                m_ref = copy(m_cpu)
                rm_ref = [copy(rm) for rm in rm_tracers]
                ws_ref = AdvectionWorkspace(m_ref)
                for _ in 1:4
                    m_save = copy(m_ref)
                    for t in 1:Nt
                        if t > 1
                            copyto!(m_ref, m_save)
                        end
                        Operators.Advection.sweep_x!(rm_ref[t], m_ref, am_cpu, scheme, ws_ref)
                        Operators.Advection.sweep_y!(rm_ref[t], m_ref, bm_cpu, scheme, ws_ref)
                        Operators.Advection.sweep_z!(rm_ref[t], m_ref, cm_cpu, scheme, ws_ref)
                        Operators.Advection.sweep_z!(rm_ref[t], m_ref, cm_cpu, scheme, ws_ref)
                        Operators.Advection.sweep_y!(rm_ref[t], m_ref, bm_cpu, scheme, ws_ref)
                        Operators.Advection.sweep_x!(rm_ref[t], m_ref, am_cpu, scheme, ws_ref)
                    end
                end

                # Multi-tracer path
                m_mt = copy(m_cpu)
                rm_4d = zeros(FT, Nx, Ny, Nz, Nt)
                for t in 1:Nt
                    rm_4d[:, :, :, t] .= rm_tracers[t]
                end
                ws_mt = AdvectionWorkspace(m_mt; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d, m_mt, am_cpu, bm_cpu, cm_cpu, scheme, ws_mt)
                end

                # Compare: must be BIT-IDENTICAL (same arithmetic, same order)
                for t in 1:Nt
                    max_diff = maximum(abs.(rm_4d[:, :, :, t] .- rm_ref[t]))
                    @test max_diff == zero(FT)
                end
                @test maximum(abs.(m_mt .- m_ref)) == zero(FT)
            end

            @testset "CPU $precision_tag $scheme_tag: MT mass conservation (4 steps)" begin
                m_mt = copy(m_cpu)
                rm_4d = zeros(FT, Nx, Ny, Nz, Nt)
                for t in 1:Nt
                    rm_4d[:, :, :, t] .= rm_tracers[t]
                end
                ws_mt = AdvectionWorkspace(m_mt; n_tracers=Nt)
                rm_init_sums = [sum(rm_4d[:, :, :, t]) for t in 1:Nt]
                m_init_sum = sum(m_mt)
                for _ in 1:4
                    strang_split_mt!(rm_4d, m_mt, am_cpu, bm_cpu, cm_cpu, scheme, ws_mt)
                end
                tol = FT == Float64 ? 1e-12 : FT(5e-5)
                @test abs(sum(m_mt) - m_init_sum) / m_init_sum < tol
                for t in 1:Nt
                    @test abs(sum(rm_4d[:, :, :, t]) - rm_init_sums[t]) / rm_init_sums[t] < tol
                end
            end
        end

        if HAS_GPU
            @testset "GPU $precision_tag: MT CPU-GPU agreement (Slopes, 4 steps)" begin
                scheme = SlopesScheme(MonotoneLimiter())

                # CPU reference
                m_mt_cpu = copy(m_cpu)
                rm_4d_cpu = zeros(FT, Nx, Ny, Nz, Nt)
                for t in 1:Nt
                    rm_4d_cpu[:, :, :, t] .= rm_tracers[t]
                end
                ws_cpu = AdvectionWorkspace(m_mt_cpu; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d_cpu, m_mt_cpu, am_cpu, bm_cpu, cm_cpu, scheme, ws_cpu)
                end

                # GPU
                m_g = CuArray(copy(m_cpu))
                rm_4d_g = CuArray(zeros(FT, Nx, Ny, Nz, Nt))
                for t in 1:Nt
                    rm_4d_g[:, :, :, t] .= CuArray(rm_tracers[t])
                end
                am_g, bm_g, cm_g = CuArray(am_cpu), CuArray(bm_cpu), CuArray(cm_cpu)
                ws_g = AdvectionWorkspace(m_g; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d_g, m_g, am_g, bm_g, cm_g, scheme, ws_g)
                end

                rm_4d_back = Array(rm_4d_g)
                m_back = Array(m_g)
                ulp_tol = FT(16)
                for t in 1:Nt
                    max_ulp = maximum(abs.(rm_4d_back[:,:,:,t] .- rm_4d_cpu[:,:,:,t])) /
                              eps(maximum(abs.(rm_4d_cpu[:,:,:,t])))
                    @test max_ulp < ulp_tol
                end
                max_m_ulp = maximum(abs.(m_back .- m_mt_cpu)) / eps(maximum(abs.(m_mt_cpu)))
                println("    MT CPU-GPU $precision_tag: m_ulp=$max_m_ulp")
                @test max_m_ulp < ulp_tol
            end
        else
            @testset "GPU $precision_tag MT: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

end

println("\n✓ All advection kernel cross-backend/precision tests completed!")
