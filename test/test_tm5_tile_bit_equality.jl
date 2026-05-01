#!/usr/bin/env julia
# Storage redesign Commit 4 — load-bearing safety check.
#
# The bounded tile workspace + tile launch loop is bit-equal to
# the legacy "one tile per launch" path *at any tile size*. This
# test runs each topology's `apply_convection!` twice on the same
# fixture — once with `tile_columns = N_total` (single big tile,
# equivalent to the pre-Commit-4 per-cell allocator) and once with
# `tile_columns = small` (many tiles) — and asserts bitwise
# equality of `state.tracers_raw` afterwards.
#
# Coverage spans LL, RG, and CS (Codex Issue 3 guardrail). A
# kernel index-decode bug on any topology can only be caught by
# running that topology's test; a CS-only check would silently
# accept a broken LL kernel.

using Test
using Random

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: Operators, Grids, State, MetDrivers, Models
using .AtmosTransport.State: DryBasis, CellState, CubedSphereState
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh,
                             CubedSphereMesh, HybridSigmaPressure
using .AtmosTransport.Operators: TM5Convection, TM5Workspace
using .AtmosTransport.MetDrivers: ConvectionForcing
using .AtmosTransport.Operators.Convection: apply!

const FT = Float32

# Build forcing arrays with non-trivial structure that varies by
# (i, j) so any kernel index-decode mistake produces a detectable
# diff. Forcing magnitude depends on the cell index, not just the
# vertical level.
function _ll_forcing(Nx, Ny, Nz, ::Type{T}) where T
    entu = zeros(T, Nx, Ny, Nz)
    detu = zeros(T, Nx, Ny, Nz)
    entd = zeros(T, Nx, Ny, Nz)
    detd = zeros(T, Nx, Ny, Nz)
    for j in 1:Ny, i in 1:Nx
        s = T(0.01) * (i + 2j)               # cell-dependent magnitude
        entu[i, j, 3:6] .= s * T(3.0)
        detu[i, j, 3:6] .= s * T(2.0)
        entd[i, j, 4:6] .= s * T(1.0)
        detd[i, j, 4:6] .= s * T(0.5)
    end
    return ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
end

function _rg_forcing(ncells, Nz, ::Type{T}) where T
    entu = zeros(T, ncells, Nz)
    detu = zeros(T, ncells, Nz)
    entd = zeros(T, ncells, Nz)
    detd = zeros(T, ncells, Nz)
    for c in 1:ncells
        s = T(0.01) * c
        entu[c, 2:4] .= s * T(3.0)
        detu[c, 2:4] .= s * T(2.0)
        entd[c, 3:4] .= s * T(1.0)
        detd[c, 3:4] .= s * T(0.5)
    end
    return ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd))
end

function _cs_forcing(Nc, Nz, ::Type{T}) where T
    mk(scale) = ntuple(p -> begin
        a = zeros(T, Nc, Nc, Nz)
        for j in 1:Nc, i in 1:Nc
            a[i, j, 2:4] .= T(0.01) * (i + 2j + 5p) * scale
        end
        a
    end, 6)
    return ConvectionForcing(nothing, nothing,
                              (; entu = mk(T(3.0)), detu = mk(T(2.0)),
                                 entd = mk(T(1.0)), detd = mk(T(0.5))))
end

function _make_ll_state(Nx, Ny, Nz, Nt)
    m = fill(FT(5e3), Nx, Ny, Nz)
    Random.seed!(20260501)
    tr = ntuple(_ -> rand(FT, Nx, Ny, Nz) .* FT(1e-3) .* m, Nt)
    if Nt == 2
        return CellState(m; CO2 = tr[1], CH4 = tr[2])
    else
        return CellState(m; CO2 = tr[1])
    end
end

function _make_rg_state(ncells, Nz)
    m = fill(FT(5e3), ncells, Nz)
    Random.seed!(20260501)
    co2 = rand(FT, ncells, Nz) .* FT(1e-3) .* m
    return CellState(m; CO2 = co2)
end

function _make_cs_state(mesh, Nz)
    Hp = mesh.Hp
    Nc = mesh.Nc
    air_mass = ntuple(_ -> fill(FT(5e3), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    Random.seed!(20260501)
    co2 = ntuple(p -> begin
        t = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
        t[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] = rand(FT, Nc, Nc, Nz) .* FT(1e-3) .* FT(5e3)
        t
    end, 6)
    return CubedSphereState(DryBasis, mesh, air_mass; CO2 = co2)
end

@testset "TM5 tile bit-equality (storage Commit 4)" begin

    # ----------------------------------------------------------------
    # LatLon: one big tile vs many small tiles must be bit-equal.
    # ----------------------------------------------------------------
    @testset "LatLon" begin
        Nx, Ny, Nz, Nt = 8, 5, 8, 2
        mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
        A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
        B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
        forcing = _ll_forcing(Nx, Ny, Nz, FT)

        # Path A: single tile covering all cells (Nx*Ny = 40).
        state_A = _make_ll_state(Nx, Ny, Nz, Nt)
        ws_A    = TM5Workspace(state_A.air_mass; tile_columns = Nx * Ny)
        apply!(state_A, forcing, grid, TM5Convection(), FT(600); workspace = ws_A)

        # Path B: small tile (7) so the launch is broken into 6 tiles
        # with a trailing remainder of 5. Stress-tests the bias logic
        # on uneven splits.
        state_B = _make_ll_state(Nx, Ny, Nz, Nt)
        ws_B    = TM5Workspace(state_B.air_mass; tile_columns = 7)
        apply!(state_B, forcing, grid, TM5Convection(), FT(600); workspace = ws_B)

        @test state_A.tracers_raw == state_B.tracers_raw
    end

    # ----------------------------------------------------------------
    # Reduced Gaussian: same comparison.
    # ----------------------------------------------------------------
    @testset "ReducedGaussian" begin
        # 3 latitude bands × 4 cells = 12 cells.
        mesh = ReducedGaussianMesh(FT[-0.9, 0.0, 0.9], [4, 4, 4]; FT = FT)
        Nz = 6
        A_ifc = collect(FT, range(0, 5f4; length = Nz + 1))
        B_ifc = collect(FT, range(1, 0;   length = Nz + 1))
        A_ifc[1] = 0; B_ifc[end] = 0; B_ifc[1] = 0; A_ifc[end] = 0
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
        ncells = AtmosTransport.Grids.ncells(mesh)
        forcing = _rg_forcing(ncells, Nz, FT)

        state_A = _make_rg_state(ncells, Nz)
        ws_A    = TM5Workspace(state_A.air_mass; tile_columns = ncells)
        apply!(state_A, forcing, grid, TM5Convection(), FT(600); workspace = ws_A)

        state_B = _make_rg_state(ncells, Nz)
        ws_B    = TM5Workspace(state_B.air_mass; tile_columns = 5)
        apply!(state_B, forcing, grid, TM5Convection(), FT(600); workspace = ws_B)

        @test state_A.tracers_raw == state_B.tracers_raw
    end

    # ----------------------------------------------------------------
    # CubedSphere: most demanding case — six panel launches share a
    # single tile workspace, so the bit-equality also covers the
    # workspace-reuse-across-panels invariant.
    # ----------------------------------------------------------------
    @testset "CubedSphere" begin
        Nc = 6
        Nz = 6
        mesh = CubedSphereMesh(; Nc = Nc, Hp = 1, FT = FT)
        A_ifc = collect(FT, range(0, 5f4; length = Nz + 1))
        B_ifc = collect(FT, range(1, 0;   length = Nz + 1))
        A_ifc[1] = 0; B_ifc[end] = 0; B_ifc[1] = 0; A_ifc[end] = 0
        vc = HybridSigmaPressure(A_ifc, B_ifc)
        grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
        forcing = _cs_forcing(Nc, Nz, FT)

        # Path A: single tile per panel (Nc*Nc = 36).
        state_A = _make_cs_state(mesh, Nz)
        ws_A    = TM5Workspace(state_A.air_mass; tile_columns = Nc * Nc)
        apply!(state_A, forcing, grid, TM5Convection(), FT(600); workspace = ws_A)

        # Path B: small tile, leaves an uneven remainder per panel.
        state_B = _make_cs_state(mesh, Nz)
        ws_B    = TM5Workspace(state_B.air_mass; tile_columns = 11)
        apply!(state_B, forcing, grid, TM5Convection(), FT(600); workspace = ws_B)

        for p in 1:6
            @test state_A.tracers_raw[p] == state_B.tracers_raw[p]
        end
    end
end
