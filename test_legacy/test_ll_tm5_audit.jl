#!/usr/bin/env julia

using Test
using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
const ModelPhases = AtmosTransport.Models

const FT_AUDIT = Float64

function make_audit_grid(; Nx=8, Ny=5, Nz=4)
    A = FT_AUDIT.(zeros(Nz + 1))
    B = FT_AUDIT.(range(0.0, 1.0, length=Nz + 1))
    vc = HybridSigmaPressure(A, B)
    LatitudeLongitudeGrid(CPU();
        FT=FT_AUDIT, size=(Nx, Ny, Nz),
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
        vertical=vc,
        use_reduced_grid=false)
end

@testset "LL TM5 Audit Helpers" begin
    @testset "TM5 cm reference matches Julia divergence path" begin
        grid = make_audit_grid()
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        am = reshape(FT_AUDIT.(1:(Nx + 1) * Ny * Nz), Nx + 1, Ny, Nz) ./ 100
        bm = reshape(FT_AUDIT.(1:Nx * (Ny + 1) * Nz), Nx, Ny + 1, Nz) ./ 150
        m = fill(FT_AUDIT(5.0), Nx, Ny, Nz)

        cm_ref = ModelPhases._ll_tm5_massflow_reference(am, bm, grid).cm
        cm_julia = zeros(FT_AUDIT, Nx, Ny, Nz + 1)
        ModelPhases._compute_cm_from_divergence_gpu!(cm_julia, am, bm, m, grid)

        @test maximum(abs.(cm_ref .- cm_julia)) < 1e-12
    end

    @testset "TM5 z gamma matches donor-based CFL" begin
        m = reshape(FT_AUDIT[2.0, 5.0, 7.0], 1, 1, 3)
        cm = zeros(FT_AUDIT, 1, 1, 4)
        cm[1, 1, 2] = FT_AUDIT(1.6)
        cm[1, 1, 3] = FT_AUDIT(-2.8)

        gamma_tm5 = ModelPhases._ll_tm5_z_gamma_max(cm, m)
        gamma_julia = max_cfl_massflux_z(cm, m)

        @test gamma_tm5 ≈ gamma_julia atol=1e-12
        @test gamma_tm5 ≈ FT_AUDIT(0.8) atol=1e-12
    end

    @testset "TM5 x row nloop helper" begin
        m_row = FT_AUDIT[1.0, 1.0, 1.0, 1.0]
        am_row_ok = FT_AUDIT[0.2, 0.5, 0.1, -0.3, 0.2]
        am_row_bad = FT_AUDIT[-1.2, -1.2, -1.2, -1.2, -1.2]

        @test ModelPhases._ll_tm5_advectx_get_nloop_row(m_row, am_row_ok) == 1
        @test ModelPhases._ll_tm5_advectx_get_nloop_row(m_row, am_row_bad) == 2
    end
end
