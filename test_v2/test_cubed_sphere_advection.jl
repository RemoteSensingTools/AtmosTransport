#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2.Operators.Advection: fill_panel_halos!, strang_split_cs!,
    CSAdvectionWorkspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function total_interior(panels, Nc, Hp, Nz)
    s = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        s += panels[p][i, j, k]
    end
    return s
end

function max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, target)
    dev = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        vmr = panels_rm[p][i, j, k] / panels_m[p][i, j, k]
        dev = max(dev, abs(vmr - target))
    end
    return dev
end

function make_cs_test_state(; Nc=12, Hp=1, Nz=4, FT=Float64, vmr=411.0)
    mesh = CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    panels_m  = ntuple(_ -> ones(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> fill!(zeros(FT, N, N, Nz), FT(vmr)), 6)
    fill_panel_halos!(panels_m, mesh; dir=0)
    fill_panel_halos!(panels_rm, mesh; dir=0)
    return mesh, panels_m, panels_rm
end

# ---------------------------------------------------------------------------
# Panel connectivity
# ---------------------------------------------------------------------------

@testset "CubedSphereMesh geometry" begin
    @testset "Construction and area" begin
        for Nc in [8, 24, 48]
            mesh = CubedSphereMesh(Nc=Nc)
            @test ncells(mesh) == 6 * Nc^2
            total_area = 6 * sum(mesh.cell_areas)
            expected = 4π * mesh.radius^2
            @test abs(total_area - expected) / expected < 1e-12
        end
    end

    @testset "F32 construction" begin
        mesh = CubedSphereMesh(Nc=12, FT=Float32)
        @test eltype(mesh) == Float32
        total_area = 6 * sum(mesh.cell_areas)
        expected = 4f0 * Float32(π) * mesh.radius^2
        @test abs(total_area - expected) / expected < 1f-5
    end

    @testset "Connectivity reciprocal" begin
        mesh = CubedSphereMesh(Nc=8)
        conn = mesh.connectivity
        for p in 1:6, e in 1:4
            nb = conn.neighbors[p][e]
            re = reciprocal_edge(conn, p, e)
            back = conn.neighbors[nb.panel][re]
            @test back.panel == p
        end
    end

    @testset "Metric symmetry" begin
        mesh = CubedSphereMesh(Nc=24)
        # All panels should have the same areas by gnomonic symmetry
        # (areas are computed for panel 1 and shared)
        @test all(mesh.cell_areas .> 0)
        @test all(mesh.Δx .> 0)
        @test all(mesh.Δy .> 0)
        # Cell area should be larger at panel center than edges
        mid = div(mesh.Nc, 2)
        @test mesh.cell_areas[mid, mid] > mesh.cell_areas[1, 1]
    end
end

# ---------------------------------------------------------------------------
# Halo exchange
# ---------------------------------------------------------------------------

@testset "Halo exchange" begin
    @testset "Edge fill — no zeros" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        panels = ntuple(6) do p
            q = zeros(Float64, N, N, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                q[Hp+i, Hp+j, k] = 1000.0*p + 100.0*k + 10.0*j + i
            end
            q
        end

        fill_panel_halos!(panels, mesh; dir=0)

        for p in 1:6, k in 1:Nz, s in 1:Nc, d in 1:Hp
            @test panels[p][Hp+s, Hp+Nc+d, k] != 0.0  # north
            @test panels[p][Hp+s, Hp+1-d, k] != 0.0    # south
            @test panels[p][Hp+Nc+d, Hp+s, k] != 0.0   # east
            @test panels[p][Hp+1-d, Hp+s, k] != 0.0    # west
        end
    end

    @testset "Edge consistency — P1 east ↔ P2 west (aligned)" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        panels = ntuple(6) do p
            q = zeros(Float64, N, N, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                q[Hp+i, Hp+j, k] = 1000.0*p + 100.0*k + 10.0*j + i
            end
            q
        end

        fill_panel_halos!(panels, mesh; dir=0)

        # P1 east halo should match P2 west interior
        for k in 1:Nz, s in 1:Nc
            @test panels[1][Hp+Nc+1, Hp+s, k] == panels[2][Hp+1, Hp+s, k]
        end
    end

    @testset "Corner fill — no zeros (dir=1 and dir=2)" begin
        mesh = CubedSphereMesh(Nc=8, Hp=1)
        Nc, Hp = mesh.Nc, mesh.Hp
        N = Nc + 2Hp; Nz = 2

        for dir in [1, 2]
            panels = ntuple(6) do p
                q = zeros(Float64, N, N, Nz)
                for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    q[Hp+i, Hp+j, k] = 1000.0*p + 10.0*j + i
                end
                q
            end
            fill_panel_halos!(panels, mesh; dir=dir)

            for p in 1:6, k in 1:Nz, dj in 1:Hp, di in 1:Hp
                @test panels[p][Hp+1-di, Hp+1-dj, k] != 0.0      # SW
                @test panels[p][Hp+Nc+di, Hp+1-dj, k] != 0.0     # SE
                @test panels[p][Hp+Nc+di, Hp+Nc+dj, k] != 0.0    # NE
                @test panels[p][Hp+1-di, Hp+Nc+dj, k] != 0.0     # NW
            end
        end
    end
end

# ---------------------------------------------------------------------------
# CS Strang splitting — Upwind scheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — UpwindScheme" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Small uniform eastward flux
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
            base_am[i, j, k] = 0.03
        end

        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        scheme = UpwindScheme()
        ws = CSAdvectionWorkspace(mesh, Nz)

        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=100.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+3):(Hp+Nc-2), i in (Hp+3):(Hp+Nc-1)
                am[i, j, k] = 0.04 * sin(Float64(i)*0.7 + Float64(j)*1.3)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+3):(Hp+Nc-1), i in (Hp+3):(Hp+Nc-2)
                bm[i, j, k] = 0.04 * cos(Float64(i)*1.1 + Float64(j)*0.9)
            end
            bm
        end
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, UpwindScheme(), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
    end
end

# ---------------------------------------------------------------------------
# CS Strang splitting — SlopesScheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — SlopesScheme{MonotoneLimiter}" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                am[i, j, k] = 0.05 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                bm[i, j, k] = 0.05 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.025 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        ws = CSAdvectionWorkspace(mesh, Nz)
        scheme = SlopesScheme(MonotoneLimiter())
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, scheme, ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=16, Hp=1, Nz=4, vmr=350.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        margin = 3
        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
                for i in (Hp+margin):(Hp+Nc-margin+2)
                    am[i, j, k] = 0.05 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
                end
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+2)
                for i in (Hp+margin):(Hp+Nc-margin+1)
                    bm[i, j, k] = 0.05 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
                end
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.025 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        m0  = total_interior(panels_m, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)
        m1  = total_interior(panels_m, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
        @test abs(m1 - m0) / m0 < 1e-13
    end

    @testset "Cross-panel conservation — uniform flux" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=100.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Identical uniform eastward flux for all panels
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
            base_am[i, j, k] = 0.02
        end
        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
    end

    @testset "Panel interior symmetry — central cells match across panels" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        # Non-uniform initial tracer (same pattern on all panels)
        for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
            panels_rm[p][Hp+i, Hp+j, k] = 411.0 + 10.0*sin(Float64(i)/Nc*π) * cos(Float64(j)/Nc*π)
        end
        fill_panel_halos!(panels_rm, mesh; dir=0)

        # Identical fluxes, zero at boundaries
        margin = 3
        base_am = zeros(Float64, N+1, N, Nz)
        for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
            for i in (Hp+margin):(Hp+Nc-margin+2)
                base_am[i, j, k] = 0.02
            end
        end
        panels_am = ntuple(_ -> copy(base_am), 6)
        panels_bm = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        panels_cm = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, SlopesScheme(MonotoneLimiter()), ws)

        # Central cells (far from panel edges) should be identical across panels
        # because they don't see the halo differences
        center = (Hp+margin+1):(Hp+Nc-margin)
        for p in 2:6
            @test panels_rm[p][center, center, :] ≈ panels_rm[1][center, center, :] atol=1e-14
        end
    end
end

# ---------------------------------------------------------------------------
# F32 precision
# ---------------------------------------------------------------------------

@testset "CS advection — Float32" begin
    mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=1, Nz=4, FT=Float32, vmr=411f0)
    Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
    N = Nc + 2Hp

    panels_am = ntuple(_ -> zeros(Float32, N+1, N, Nz), 6)
    panels_bm = ntuple(_ -> zeros(Float32, N, N+1, Nz), 6)
    panels_cm = ntuple(_ -> zeros(Float32, N, N, Nz+1), 6)

    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
        panels_am[p][i, j, k] = 0.02f0 * sin(Float32(i)*0.7f0)
    end

    rm0 = Float64(total_interior(panels_rm, Nc, Hp, Nz))
    ws = CSAdvectionWorkspace(mesh, Nz; FT=Float32)
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, SlopesScheme(MonotoneLimiter()), ws)

    dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
    @test dev < 1f-4  # F32 has ~7 digits
end
