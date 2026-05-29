#!/usr/bin/env julia

using Test
using Logging

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: reciprocal_edge
using .AtmosTransport.Operators: MonotoneLimiter, required_halo_width
using .AtmosTransport.Operators.Advection: fill_panel_halos!, strang_split_cs!,
    strang_split_cs_mt!, strang_split!, CSAdvectionWorkspace, VerticalRemapWorkspace,
    compute_target_pressure_from_mass_direct!, vertical_remap_cs!

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

function max_interior_absdiff(a, b, Nc, Hp, Nz)
    dev = 0.0
    for p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        dev = max(dev, abs(a[p][i, j, k] - b[p][i, j, k]))
    end
    return dev
end

function max_interior_absdiff_4d(a, b, Nc, Hp, Nz, Nt)
    dev = 0.0
    for t in 1:Nt, p in 1:6, k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
        dev = max(dev, abs(a[p][i, j, k, t] - b[p][i, j, k, t]))
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

function total_cs_surface_rate(rates)
    s = zero(eltype(rates[1]))
    for p in 1:6
        s += sum(rates[p])
    end
    return s
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

    @testset "Packed multi-tracer path matches single-tracer reference" begin
        for scheme in (UpwindScheme(), PPMScheme())
            Hp = required_halo_width(scheme)
            mesh, panels_m0, panels_rm0 = make_cs_test_state(Nc=8, Hp=Hp, Nz=3, vmr=100.0)
            Nc, Nz = mesh.Nc, 3
            N = Nc + 2Hp
            panels_rm2 = ntuple(p -> panels_rm0[p] .* 1.7, 6)
            panels_am = ntuple(6) do p
                am = zeros(Float64, N + 1, N, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                    am[i, j, k] = 0.015 * sin(0.2p + 0.7i - 0.4j + 0.3k)
                end
                am
            end
            panels_bm = ntuple(6) do p
                bm = zeros(Float64, N, N + 1, Nz)
                for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                    bm[i, j, k] = 0.012 * cos(0.3p - 0.5i + 0.6j + 0.2k)
                end
                bm
            end
            panels_cm = ntuple(6) do p
                cm = zeros(Float64, N, N, Nz + 1)
                for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                    cm[i, j, k] = 0.01 * sin(0.1p + 0.2i + 0.3j - 0.7k)
                end
                cm
            end

            m_ref0 = deepcopy(panels_m0)
            m_ref = deepcopy(panels_m0)
            rm_ref1 = deepcopy(panels_rm0)
            rm_ref2 = deepcopy(panels_rm2)
            ws_ref = CSAdvectionWorkspace(mesh, Nz)
            strang_split_cs!(rm_ref1, m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)
            copyto!.(m_ref, m_ref0)
            strang_split_cs!(rm_ref2, m_ref, panels_am, panels_bm, panels_cm,
                             mesh, scheme, ws_ref; subcycle_count = 1)

            rm_mt = ntuple(p -> cat(panels_rm0[p], panels_rm2[p]; dims = 4), 6)
            m_mt = deepcopy(panels_m0)
            ws_mt = CSAdvectionWorkspace(mesh, Nz; n_tracers = 2)
            strang_split_cs_mt!(rm_mt, m_mt, panels_am, panels_bm, panels_cm,
                                mesh, scheme, ws_mt; subcycle_count = 1)
            rm_ref = ntuple(p -> cat(rm_ref1[p], rm_ref2[p]; dims = 4), 6)

            @test max_interior_absdiff_4d(rm_mt, rm_ref, Nc, Hp, Nz, 2) < 1e-12
            @test max_interior_absdiff(m_mt, m_ref, Nc, Hp, Nz) < 1e-12
        end
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
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=411.0)
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
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=16, Hp=2, Nz=4, vmr=350.0)
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
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=100.0)
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
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, vmr=411.0)
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

# ---------------------------------------------------------------------------
# CS Strang splitting — PPMScheme
# ---------------------------------------------------------------------------

@testset "CS Strang splitting — PPMScheme" begin
    @testset "Uniform field invariance" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=3, Nz=4, vmr=411.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc+1)
                am[i, j, k] = 0.03 * sin(Float64(i)*0.5 + Float64(j)*1.1 + Float64(k)*0.4)
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc+1), i in (Hp+1):(Hp+Nc)
                bm[i, j, k] = 0.03 * cos(Float64(i)*1.3 + Float64(j)*0.7 + Float64(k)*0.2)
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.015 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, PPMScheme(), ws)

        dev = max_vmr_deviation(panels_rm, panels_m, Nc, Hp, Nz, 411.0)
        @test dev < 1e-10
    end

    @testset "Mass conservation — interior fluxes" begin
        mesh, panels_m, panels_rm = make_cs_test_state(Nc=16, Hp=3, Nz=4, vmr=350.0)
        Nc, Hp, Nz = mesh.Nc, mesh.Hp, 4
        N = Nc + 2Hp

        margin = 4  # PPM has wider stencil
        panels_am = ntuple(6) do _
            am = zeros(Float64, N+1, N, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+1)
                for i in (Hp+margin):(Hp+Nc-margin+2)
                    am[i, j, k] = 0.04 * sin(Float64(i)*0.7 + Float64(j)*1.3 + Float64(k)*0.5)
                end
            end
            am
        end
        panels_bm = ntuple(6) do _
            bm = zeros(Float64, N, N+1, Nz)
            for k in 1:Nz, j in (Hp+margin):(Hp+Nc-margin+2)
                for i in (Hp+margin):(Hp+Nc-margin+1)
                    bm[i, j, k] = 0.04 * cos(Float64(i)*1.1 + Float64(j)*0.9 + Float64(k)*0.3)
                end
            end
            bm
        end
        panels_cm = ntuple(6) do _
            cm = zeros(Float64, N, N, Nz+1)
            for k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                cm[i, j, k] = 0.02 * sin(Float64(i)*0.3 + Float64(k)*2.1)
            end
            cm
        end

        rm0 = total_interior(panels_rm, Nc, Hp, Nz)
        m0  = total_interior(panels_m, Nc, Hp, Nz)
        ws = CSAdvectionWorkspace(mesh, Nz)
        strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                         mesh, PPMScheme(), ws)
        rm1 = total_interior(panels_rm, Nc, Hp, Nz)
        m1  = total_interior(panels_m, Nc, Hp, Nz)

        @test abs(rm1 - rm0) / rm0 < 1e-13
        @test abs(m1 - m0) / m0 < 1e-13
    end
end

@testset "CS source mass closure — PPMScheme" begin
    @testset "Surface source stays equal to integrated source through transport" begin
        FT = Float64
        Nc, Hp, Nz = 12, 3, 4
        mesh = CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
        N = Nc + 2Hp
        vertical = HybridSigmaPressure(FT[0, 100, 300, 600, 1000],
                                       FT[0, 0, 0, 0.5, 1])
        grid = AtmosGrid(mesh, vertical, CPU(); FT=FT)

        panels_m = ntuple(6) do p
            m = zeros(FT, N, N, Nz)
            for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
                m[i, j, k] = FT(1.0e6) * (1 + FT(0.05) * k + FT(0.001) * p)
            end
            m
        end
        tracer = ntuple(_ -> zeros(FT, N, N, Nz), 6)
        fill_panel_halos!(panels_m, mesh; dir=0)
        fill_panel_halos!(tracer, mesh; dir=0)
        state = CubedSphereState(DryBasis, mesh, panels_m; FossilCO2=tracer)

        fluxes = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
        margin = 4
        for p in 1:6, k in 1:Nz
            for j in (Hp+margin):(Hp+Nc-margin+1), i in (Hp+margin):(Hp+Nc-margin+2)
                fluxes.am[p][i, j, k] = FT(25.0) * sin(FT(0.17) * i + FT(0.11) * j + FT(0.13) * k + p)
            end
            for j in (Hp+margin):(Hp+Nc-margin+2), i in (Hp+margin):(Hp+Nc-margin+1)
                fluxes.bm[p][i, j, k] = FT(20.0) * cos(FT(0.09) * i - FT(0.14) * j + FT(0.07) * k + p)
            end
        end
        for p in 1:6, k in 2:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
            fluxes.cm[p][i, j, k] = FT(10.0) * sin(FT(0.05) * i + FT(0.08) * j - FT(0.19) * k + p)
        end

        rates = ntuple(6) do p
            [FT(0.02) * (1 + FT(0.01) * p + FT(0.001) * i + FT(0.002) * j)
             for i in 1:Nc, j in 1:Nc]
        end
        source_rate = total_cs_surface_rate(rates)
        emissions = SurfaceFluxOperator(SurfaceFluxSource(:FossilCO2, rates))
        ws = CSAdvectionWorkspace(mesh, Nz; n_tracers=1)
        dt = FT(600)

        initial_air = total_air_mass(state)
        expected = zero(FT)
        for _ in 1:6
            strang_split!(state, fluxes, grid, PPMScheme();
                          workspace=ws,
                          cfl_limit=FT(0.95),
                          diffusion_op=NoDiffusion(),
                          emissions_op=emissions,
                          dt=dt)
            expected += source_rate * dt
            storage = total_mass(state, :FossilCO2)
            @test isapprox(storage, expected; rtol=1e-12, atol=1e-8)
            @test isapprox(total_air_mass(state), initial_air; rtol=1e-13, atol=1e-6)
        end
    end
end

# ---------------------------------------------------------------------------
# Halo validation
# ---------------------------------------------------------------------------

@testset "CS halo validation" begin
    @testset "SlopesScheme requires Hp ≥ 2" begin
        mesh_hp1 = CubedSphereMesh(Nc=8, Hp=1)
        N = 8 + 2; Nz = 2
        ws = CSAdvectionWorkspace(mesh_hp1, Nz)
        pr = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs!(pr, pm, pa, pb, pc,
                                                      mesh_hp1, SlopesScheme(), ws)
    end

    @testset "PPMScheme requires Hp ≥ 3" begin
        mesh_hp2 = CubedSphereMesh(Nc=8, Hp=2)
        N = 8 + 4; Nz = 2
        ws = CSAdvectionWorkspace(mesh_hp2, Nz)
        pr = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pm = ntuple(_ -> ones(Float64, N, N, Nz), 6)
        pa = ntuple(_ -> zeros(Float64, N+1, N, Nz), 6)
        pb = ntuple(_ -> zeros(Float64, N, N+1, Nz), 6)
        pc = ntuple(_ -> zeros(Float64, N, N, Nz+1), 6)

        @test_throws ErrorException strang_split_cs!(pr, pm, pa, pb, pc,
                                                      mesh_hp2, PPMScheme(), ws)
    end
end

# ---------------------------------------------------------------------------
# CS Poisson balance (LLPoissonWorkspace zero-allocation)
# ---------------------------------------------------------------------------

@testset "LLPoissonWorkspace zero-allocation balance" begin
    Prep = AtmosTransport.Preprocessing
    if isdefined(Prep, :LLPoissonWorkspace) && isdefined(Prep, :balance_mass_fluxes!)
        Nx, Ny, Nz = 24, 12, 4
        am = rand(Float64, Nx+1, Ny, Nz) .* 0.01
        bm = rand(Float64, Nx, Ny+1, Nz) .* 0.01
        dm = rand(Float64, Nx, Ny, Nz) .* 1e-6

        ws = Prep.LLPoissonWorkspace(Nx, Ny)

        # Warm up
        Prep.balance_mass_fluxes!(copy(am), copy(bm), copy(dm), ws)
        prev_logger = global_logger(NullLogger())
        try
            Prep.balance_mass_fluxes!(copy(am), copy(bm), copy(dm), ws)

            # Measure allocation
            am2 = copy(am); bm2 = copy(bm); dm2 = copy(dm)
            alloc = @allocated Prep.balance_mass_fluxes!(am2, bm2, dm2, ws)
            # Should be near-zero once logging allocations are removed.
            @test alloc < 10_000  # < 10 KB
        finally
            global_logger(prev_logger)
        end
    else
        @test true  # skip if Preprocessing not available
    end
end

# ---------------------------------------------------------------------------
# F32 precision
# ---------------------------------------------------------------------------

@testset "CS advection — Float32" begin
    mesh, panels_m, panels_rm = make_cs_test_state(Nc=12, Hp=2, Nz=4, FT=Float32, vmr=411f0)
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

@testset "CS vertical remap identity — Float32" begin
    FT = Float32
    Nc, Hp, Nz = 4, 1, 5
    mesh = CubedSphereMesh(Nc=Nc, Hp=Hp, FT=FT)
    N = Nc + 2Hp
    gravity = FT(9.80665)
    q = FT(3e-6)

    panels_m = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        ii, jj = Hp + i, Hp + j
        dp = FT(1000 + 25k + i + 2j)
        panels_m[p][ii, jj, k] = dp * mesh.cell_areas[i, j] / gravity
        panels_rm[p][ii, jj, k] = q * panels_m[p][ii, jj, k]
    end
    fill_panel_halos!(panels_m, mesh; dir=0)
    fill_panel_halos!(panels_rm, mesh; dir=0)

    panels_rm0 = deepcopy(panels_rm)
    ak = fill(FT(1), Nz + 1)
    bk = collect(range(zero(FT), one(FT); length=Nz + 1))
    ws = VerticalRemapWorkspace(mesh, Nz, ak, bk; FT=FT)

    compute_target_pressure_from_mass_direct!(ws, panels_m, mesh.cell_areas,
                                              gravity, Nc, Hp, Nz)
    vertical_remap_cs!(panels_rm, panels_m, ws, similar(panels_rm[1]),
                       mesh.cell_areas, gravity, Nc, Hp, Nz)

    max_rel = zero(FT)
    for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        ii, jj = Hp + i, Hp + j
        denom = max(abs(panels_rm0[p][ii, jj, k]), eps(FT))
        max_rel = max(max_rel, abs(panels_rm[p][ii, jj, k] -
                                   panels_rm0[p][ii, jj, k]) / denom)
    end
    @test max_rel < FT(2e-5)
end
