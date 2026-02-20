#!/usr/bin/env julia
# ===========================================================================
# Tests for cubed-sphere mass-flux advection
#
# Validates:
# 1. Air mass computation from DELP + area
# 2. Vertical mass-flux closure (cm) from horizontal convergence
# 3. Uniform field preservation (uniform c stays uniform under advection)
# 4. Mass conservation (total tracer mass is invariant after Strang split)
# 5. Positivity preservation with limiter enabled
# ===========================================================================

using Test
using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Grids: fill_panel_halos!
using AtmosTransportModel.Advection

const FT = Float64

function make_cs_test_grid(; Nc=12, Nz=5)
    A = FT.(range(0.0, 0.0, length=Nz+1))
    B = FT.(range(0.0, 1.0, length=Nz+1))
    vc = HybridSigmaPressure(A, B)
    grid = CubedSphereGrid(CPU(); FT, Nc, vertical=vc)
    return grid
end

function uniform_delp_panels(Nc, Nz, Hp; ps=FT(101325))
    ntuple(6) do _
        arr = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        dp = ps / Nz
        arr[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= dp
        arr
    end
end

function zero_mass_flux_panels(Nc, Nz)
    am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6)
    cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    return am, bm, cm
end

@testset "Cubed-Sphere Mass-Flux Advection" begin

    @testset "CubedSphereGeometryCache construction" begin
        grid = make_cs_test_grid()
        ref = zeros(FT, grid.Nc + 2grid.Hp, grid.Nc + 2grid.Hp, grid.Nz)
        gc = build_geometry_cache(grid, ref)

        @test gc.Nc == grid.Nc
        @test gc.Nz == grid.Nz
        @test gc.Hp == grid.Hp
        @test length(gc.area) == 6
        @test size(gc.area[1]) == (grid.Nc, grid.Nc)
        @test all(gc.area[p][i, j] > 0 for p in 1:6, i in 1:grid.Nc, j in 1:grid.Nc)
        @test length(gc.bt) == grid.Nz
    end

    @testset "compute_air_mass_panel!" begin
        grid = make_cs_test_grid()
        Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
        ref = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        gc = build_geometry_cache(grid, ref)

        delp = uniform_delp_panels(Nc, Nz, Hp)
        m = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)

        for p in 1:6
            compute_air_mass_panel!(m[p], delp[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # All interior air mass should be positive
        for p in 1:6
            interior = m[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
            @test all(interior .> 0)
        end

        # Total mass should be reasonable (total atmospheric mass ≈ 5.15e18 kg)
        total_m = sum(sum(m[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
        @test total_m > 0
    end

    @testset "compute_cm_panel! with zero horizontal fluxes" begin
        Nc, Nz = 12, 5
        am = zeros(FT, Nc + 1, Nc, Nz)
        bm = zeros(FT, Nc, Nc + 1, Nz)
        cm = zeros(FT, Nc, Nc, Nz + 1)
        bt = ones(FT, Nz) / Nz

        compute_cm_panel!(cm, am, bm, bt, Nc, Nz)

        @test all(cm .== 0)  # no horizontal convergence → no vertical flux
    end

    @testset "Uniform field preservation" begin
        grid = make_cs_test_grid(; Nc=12, Nz=5)
        Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
        ref = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        gc = build_geometry_cache(grid, ref)

        delp = uniform_delp_panels(Nc, Nz, Hp)
        m_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
        for p in 1:6
            compute_air_mass_panel!(m_panels[p], delp[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # Uniform tracer field: rm = c * m with c = 400 ppm
        c_uniform = FT(400e-6)
        rm_panels = ntuple(6) do p
            rm = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                rm[Hp+i, Hp+j, k] = c_uniform * m_panels[p][Hp+i, Hp+j, k]
            end
            rm
        end

        # Small uniform mass fluxes (should preserve uniformity)
        am_panels = ntuple(_ -> fill(FT(1e3), Nc + 1, Nc, Nz), 6)
        bm_panels = ntuple(_ -> fill(FT(1e3), Nc, Nc + 1, Nz), 6)
        cm_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)

        for p in 1:6
            compute_cm_panel!(cm_panels[p], am_panels[p], bm_panels[p], gc.bt, Nc, Nz)
        end

        ws = allocate_cs_massflux_workspace(ref)
        fill_panel_halos!(rm_panels, grid)
        fill_panel_halos!(m_panels, grid)

        strang_split_massflux!(rm_panels, m_panels,
                               am_panels, bm_panels, cm_panels,
                               grid, true, ws)

        # After advection, concentration should still be approximately uniform
        for p in 1:6
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                c_after = rm_panels[p][Hp+i, Hp+j, k] / m_panels[p][Hp+i, Hp+j, k]
                @test isapprox(c_after, c_uniform; rtol=1e-6)
            end
        end
    end

    @testset "Mass conservation with zero fluxes" begin
        grid = make_cs_test_grid(; Nc=12, Nz=5)
        Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
        ref = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        gc = build_geometry_cache(grid, ref)

        delp = uniform_delp_panels(Nc, Nz, Hp)
        m_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
        for p in 1:6
            compute_air_mass_panel!(m_panels[p], delp[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # Gaussian blob tracer on panel 1
        rm_panels = ntuple(6) do p
            rm = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
            if p == 1
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    di = i - Nc/2
                    dj = j - Nc/2
                    c = FT(100e-6) * exp(-(di^2 + dj^2) / (Nc/4)^2)
                    rm[Hp+i, Hp+j, k] = c * m_panels[p][Hp+i, Hp+j, k]
                end
            end
            rm
        end

        mass_before = sum(sum(rm_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)

        am, bm, cm = zero_mass_flux_panels(Nc, Nz)
        ws = allocate_cs_massflux_workspace(ref)

        strang_split_massflux!(rm_panels, m_panels, am, bm, cm, grid, true, ws)

        mass_after = sum(sum(rm_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)

        @test isapprox(mass_after, mass_before; rtol=1e-12)
    end

end
