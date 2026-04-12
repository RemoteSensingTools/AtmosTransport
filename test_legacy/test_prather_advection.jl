#!/usr/bin/env julia

using Test
using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection

const FT = Float64

function make_prather_grid(; Nx=8, Ny=6, Nz=5)
    A = FT.(range(0.0, 0.0, length=Nz+1))
    B = FT.(range(0.0, 1.0, length=Nz+1))
    vc = HybridSigmaPressure(A, B)
    LatitudeLongitudeGrid(CPU();
        FT, size=(Nx, Ny, Nz),
        longitude=(-180.0, 180.0),
        latitude=(-90.0, 90.0),
        vertical=vc,
        use_reduced_grid=false)
end

uniform_dp_prather(grid, ps=101325.0) =
    AtmosTransport.Advection._build_Δz_3d(grid, fill(FT(ps), grid.Nx, grid.Ny))

const _advect_x_prather! = getfield(AtmosTransport.Advection, Symbol("_advect_x_prather!"))
const _advect_y_prather! = getfield(AtmosTransport.Advection, Symbol("_advect_y_prather!"))
const _advect_z_prather! = getfield(AtmosTransport.Advection, Symbol("_advect_z_prather!"))

@testset "Prather Advection" begin
    @testset "Slope initialization from rm/m" begin
        grid = make_prather_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp_prather(grid)
        m = compute_air_mass(Δp, grid)

        c = Array{FT}(undef, Nx, Ny, Nz)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            c[i, j, k] = FT(10 + 2i + 3j + 4k)
        end
        rm = c .* m

        pw = allocate_prather_workspace(m)
        initialize_prather_workspace!(pw, rm, m, true)

        i, j, k = 4, 3, 3
        @test pw.rxm[i, j, k] ≈ FT(2) * m[i, j, k] atol=1e-10
        @test pw.rym[i, j, k] ≈ FT(3) * m[i, j, k] atol=1e-10
        @test pw.rzm[i, j, k] ≈ FT(4) * m[i, j, k] atol=1e-10
        @test all(Array(pw.rym[:, 1, :]) .== 0)
        @test all(Array(pw.rym[:, Ny, :]) .== 0)
        @test all(Array(pw.rzm[:, :, 1]) .== 0)
        @test all(Array(pw.rzm[:, :, Nz]) .== 0)
    end

    @testset "Mass conservation and uniform preservation" begin
        grid = make_prather_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp_prather(grid)
        m = compute_air_mass(Δp, grid)
        rm = (; c = m .* FT(420.0))

        u = zeros(FT, Nx+1, Ny, Nz) .+ FT(2.0)
        u[Nx+1, :, :] .= u[1, :, :]
        mf = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, FT(40.0))
        cm = zeros(FT, Nx, Ny, Nz+1)

        m_work = copy(m)
        pw = (; c = allocate_prather_workspace(m))
        mass_before = sum(rm.c)

        strang_split_prather!(rm, m_work, mf.am, mf.bm, cm, grid, pw, true)

        mass_after = sum(rm.c)
        c_after = rm.c ./ m_work
        @test abs(mass_after - mass_before) / abs(mass_before) < 1e-12
        @test maximum(abs.(c_after .- FT(420.0))) < 1e-8
        @test all(isfinite, rm.c)
        @test all(isfinite, m_work)
    end

    @testset "Directional one-step cases" begin
        grid = make_prather_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        Δp = uniform_dp_prather(grid)
        m = compute_air_mass(Δp, grid)

        c0 = Array{FT}(undef, Nx, Ny, Nz)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            c0[i, j, k] = FT(100 + i + 2j + 3k)
        end
        rm0 = c0 .* m

        u = zeros(FT, Nx+1, Ny, Nz)
        u .= 3.0
        u[Nx+1, :, :] .= u[1, :, :]
        v = zeros(FT, Nx, Ny+1, Nz)
        v[:, 2:Ny, :] .= 1.5
        cm = zeros(FT, Nx, Ny, Nz+1)
        cm[:, :, 2:Nz] .= FT(0.2) .* m[:, :, 1:Nz-1]

        mf_x = compute_mass_fluxes(u, zeros(FT, Nx, Ny+1, Nz), grid, Δp, FT(40.0))
        mf_y = compute_mass_fluxes(zeros(FT, Nx+1, Ny, Nz), v, grid, Δp, FT(40.0))

        for (label, op!, flux) in (
            ("x", _advect_x_prather!, mf_x.am),
            ("y", _advect_y_prather!, mf_y.bm),
            ("z", _advect_z_prather!, cm),
        )
            rm = copy(rm0)
            m_work = copy(m)
            pw = allocate_prather_workspace(m)
            initialize_prather_workspace!(pw, rm, m_work, true)
            mass_before = sum(rm)

            if label == "x"
                op!(rm, m_work, flux, pw, Nx, true)
            elseif label == "y"
                op!(rm, m_work, flux, pw, Ny, true)
            else
                op!(rm, m_work, flux, pw, Nz, true)
            end

            @test abs(sum(rm) - mass_before) / abs(mass_before) < 1e-12
            @test all(isfinite, rm)
            @test all(isfinite, m_work)
            @test all(isfinite, pw.rxm)
            @test all(isfinite, pw.rym)
            @test all(isfinite, pw.rzm)
        end
    end
end
