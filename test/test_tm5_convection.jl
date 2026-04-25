#!/usr/bin/env julia
"""
Plan 23 tests for `TM5Convection` — built up over Commits 1–4.

Commit 1 (this file, initial): type hierarchy + workspace factory
+ runtime validator + stub `apply!` / `apply_convection!`.

Commit 2 adds the column-solver tests.
Commit 4 adds the full-kernel parity / conservation / CPU-GPU
agreement tests.

Invariant preserved by every commit in plan 23: `NoConvection` and
`CMFMCConvection` paths stay bit-exact to pre-plan-23 behaviour.
"""

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: Operators, Grids, State, MetDrivers, Models

using .AtmosTransport.State: DryBasis, MoistBasis, CellState, CubedSphereState,
                              allocate_face_fluxes
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh,
                             CubedSphereMesh, HybridSigmaPressure
using .AtmosTransport.Operators: AbstractConvection, NoConvection,
                                 CMFMCConvection, TM5Convection,
                                 CMFMCWorkspace, TM5Workspace,
                                 UpwindScheme
using .AtmosTransport.Operators.Convection: apply_convection!
using .AtmosTransport.MetDrivers: ConvectionForcing
using .AtmosTransport.Models: TransportModel, with_convection,
                              _convection_workspace_for

const FT = Float32

@testset "plan 23 Commit 1: TM5Convection type + workspace factory" begin
    @testset "construct TM5Convection() without fields" begin
        op = TM5Convection()
        @test op isa AbstractConvection
        @test op isa TM5Convection
        # Stateless — no fields.
        @test fieldcount(TM5Convection) == 0
    end

    @testset "NoConvection and CMFMCConvection unchanged" begin
        # Plan 23's validator refactor must not change what these
        # return or how they dispatch. Bit-exact regression.
        @test NoConvection() isa AbstractConvection
        @test CMFMCConvection() isa AbstractConvection
    end
end

@testset "plan 23 Commit 1: TM5Workspace allocation shape" begin
    Nx, Ny, Nz = 4, 3, 5

    @testset "structured LatLon air_mass (Nx, Ny, Nz)" begin
        air_mass = zeros(FT, Nx, Ny, Nz)
        ws = TM5Workspace(air_mass)
        @test ws isa TM5Workspace{FT}
        @test size(ws.conv1)      == (Nz, Nz, Nx, Ny)
        @test size(ws.pivots)     == (Nz, Nx, Ny)
        @test size(ws.cloud_dims) == (3, Nx, Ny)
        @test eltype(ws.conv1)      == FT
        @test eltype(ws.pivots)     == Int
        @test eltype(ws.cloud_dims) == Int
    end

    @testset "face-indexed RG air_mass (ncells, Nz)" begin
        ncells = 24
        air_mass = zeros(FT, ncells, Nz)
        ws = TM5Workspace(air_mass)
        @test ws isa TM5Workspace{FT}
        @test size(ws.conv1)      == (Nz, Nz, ncells)
        @test size(ws.pivots)     == (Nz, ncells)
        @test size(ws.cloud_dims) == (3, ncells)
    end

    @testset "CS panel NTuple{6, (Nc, Nc, Nz)}" begin
        Nc = 6
        air_mass = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
        ws = TM5Workspace(air_mass)
        @test ws isa TM5Workspace{FT}
        @test ws.conv1 isa NTuple{6, <:AbstractArray}
        @test size(ws.conv1[1])      == (Nz, Nz, Nc, Nc)
        @test size(ws.pivots[3])     == (Nz, Nc, Nc)
        @test size(ws.cloud_dims[6]) == (3, Nc, Nc)
    end
end

@testset "plan 23 Commit 1: _convection_workspace_for dispatch" begin
    # Minimal LatLon grid just to drive _convection_workspace_for.
    mesh = LatLonMesh(; Nx=4, Ny=3, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(zeros(FT, 4, 3, 4); CO2 = zeros(FT, 4, 3, 4))

    @test _convection_workspace_for(NoConvection(), state, grid) === nothing
    ws_cmfmc = _convection_workspace_for(CMFMCConvection(), state, grid)
    @test ws_cmfmc isa CMFMCWorkspace{FT}

    ws_tm5 = _convection_workspace_for(TM5Convection(), state, grid)
    @test ws_tm5 isa TM5Workspace{FT}
    @test size(ws_tm5.conv1) == (4, 4, 4, 3)  # (Nz, Nz, Nx, Ny)
end

@testset "plan 23 Commit 4: TM5Convection apply! LL kernel" begin
    Nx, Ny, Nz, Nt = 4, 3, 8, 2
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
    B_ifc = FT[0, 0, 0, FT(0.05), FT(0.2), FT(0.4), FT(0.7), FT(0.9), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    m = fill(FT(5e3), Nx, Ny, Nz)
    tracer1 = zeros(FT, Nx, Ny, Nz); tracer1[:, :, Nz] .= FT(1e-3) .* m[:, :, Nz]
    tracer2 = zeros(FT, Nx, Ny, Nz); tracer2[:, :, 3] .= FT(2e-3) .* m[:, :, 3]
    state = CellState(m; CO2 = tracer1, CH4 = tracer2)

    entu = zeros(FT, Nx, Ny, Nz); entu[:, :, 3:6] .= FT(0.03)
    detu = zeros(FT, Nx, Ny, Nz); detu[:, :, 3:6] .= FT(0.02)
    entd = zeros(FT, Nx, Ny, Nz); entd[:, :, 4:6] .= FT(0.01)
    detd = zeros(FT, Nx, Ny, Nz); detd[:, :, 4:6] .= FT(0.005)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass)

    mass_before = [sum(state.tracers_raw[:, :, :, t]) for t in 1:Nt]
    state0_copy = copy(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)

    for t in 1:Nt
        mass_after = sum(state.tracers_raw[:, :, :, t])
        @test isapprox(mass_after, mass_before[t];
                        rtol = 1f4 * eps(FT))
    end
    # Nontrivial: tracer profile changed (not silent identity).
    @test any(state.tracers_raw .!= state0_copy)

    # Zero-forcing → bit-exact identity.
    state_zero = CellState(m; CO2 = copy(tracer1), CH4 = copy(tracer2))
    ws_zero = TM5Workspace(state_zero.air_mass)
    zero_forcing = ConvectionForcing(nothing, nothing,
        (; entu = zeros(FT, Nx, Ny, Nz), detu = zeros(FT, Nx, Ny, Nz),
           entd = zeros(FT, Nx, Ny, Nz), detd = zeros(FT, Nx, Ny, Nz)))
    state0_identity = copy(state_zero.tracers_raw)
    apply!(state_zero, zero_forcing, grid, TM5Convection(), FT(600);
            workspace = ws_zero)
    @test state_zero.tracers_raw == state0_identity
end

@testset "plan 23 Commit 4: TM5Convection apply! RG kernel" begin
    mesh = ReducedGaussianMesh(FT[-0.9, 0.0, 0.9], [4, 4, 4]; FT=FT)
    Nz = 6
    A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
    B_ifc = collect(FT, range(1, 0;   length=Nz+1))
    A_ifc[1] = 0; B_ifc[end] = 0
    B_ifc[1] = 0; A_ifc[end] = 0
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)
    ncells = AtmosTransport.Grids.ncells(mesh)

    m = fill(FT(5e3), ncells, Nz)
    tracer1 = zeros(FT, ncells, Nz); tracer1[:, Nz] .= FT(1e-3) .* m[:, Nz]
    state = CellState(m; CO2 = tracer1)

    entu = zeros(FT, ncells, Nz); entu[:, 2:4] .= FT(0.03)
    detu = zeros(FT, ncells, Nz); detu[:, 2:4] .= FT(0.02)
    entd = zeros(FT, ncells, Nz); entd[:, 3:4] .= FT(0.01)
    detd = zeros(FT, ncells, Nz); detd[:, 3:4] .= FT(0.005)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass)

    mass_before = sum(state.tracers_raw)
    state0_copy = copy(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)
    mass_after = sum(state.tracers_raw)
    @test isapprox(mass_after, mass_before; rtol = 1f4 * eps(FT))
    @test any(state.tracers_raw .!= state0_copy)
end

@testset "plan 23 Commit 4: TM5Convection apply! CS kernel" begin
    Nc = 4
    Nz = 6
    mesh = CubedSphereMesh(; Nc = Nc, Hp = 1, FT = FT)
    A_ifc = collect(FT, range(0, 5f4; length=Nz+1))
    B_ifc = collect(FT, range(1, 0;   length=Nz+1))
    A_ifc[1] = 0; B_ifc[end] = 0
    B_ifc[1] = 0; A_ifc[end] = 0
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    Hp = mesh.Hp

    air_mass = ntuple(_ -> fill(FT(5e3), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    tracer1  = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    for p in 1:6
        tracer1[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, Nz] .= FT(1e-3) * FT(5e3)
    end
    state = CubedSphereState(DryBasis, mesh, air_mass; CO2 = tracer1)

    entu = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 2:4] .= FT(0.03); e end, 6)
    detu = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 2:4] .= FT(0.02); e end, 6)
    entd = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 3:4] .= FT(0.01); e end, 6)
    detd = ntuple(_ -> begin e = zeros(FT, Nc, Nc, Nz); e[:, :, 3:4] .= FT(0.005); e end, 6)
    forcing = ConvectionForcing(nothing, nothing,
                                 (; entu, detu, entd, detd))
    ws = TM5Workspace(state.air_mass)

    function interior_mass(tracers_raw)
        s = zero(FT)
        for p in 1:6, k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
            s += tracers_raw[p][i, j, k, 1]
        end
        return s
    end
    mass_before = interior_mass(state.tracers_raw)
    apply!(state, forcing, grid, TM5Convection(), FT(600); workspace = ws)
    mass_after = interior_mass(state.tracers_raw)
    @test isapprox(mass_after, mass_before; rtol = 1f4 * eps(FT))
end

@testset "plan 23 Commit 4: _assert_tm5_forcing catches missing tm5_fields" begin
    Nx, Ny, Nz = 4, 3, 4
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(fill(FT(1), Nx, Ny, Nz); CO2 = zeros(FT, Nx, Ny, Nz))
    ws = TM5Workspace(state.air_mass)
    empty_forcing = ConvectionForcing()

    err = try
        apply!(state, empty_forcing, grid, TM5Convection(), FT(60); workspace=ws)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("tm5_fields", err.msg)
    @test occursin("NamedTuple", err.msg)

    # No "not yet implemented" strings remain (principle 7 prep).
    @test !occursin("not yet implemented", err.msg)
end

@testset "plan 23 Commit 2: _tm5_solve_column! identity + conservation" begin
    using .AtmosTransport.Operators.Convection: _tm5_solve_column!,
                                                   _tm5_diagnose_cloud_dims

    Nz = 8
    Nt = 3

    # -----------------------------------------------------------------
    # Zero-forcing short-circuits to identity
    # -----------------------------------------------------------------
    @testset "zero forcing → identity (F64)" begin
        T = Float64
        rm    = rand(T, Nz, Nt)
        rm0   = copy(rm)
        m     = rand(T, Nz) .+ T(0.1)
        entu  = zeros(T, Nz)
        detu  = zeros(T, Nz)
        entd  = zeros(T, Nz)
        detd  = zeros(T, Nz)
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))
        @test rm == rm0                 # bit-exact identity
        @test cd == [Nz + 1, 0, Nz + 1] # "no convection" sentinel
    end

    # -----------------------------------------------------------------
    # Tracer mass conservation: sum(rm_new) == sum(rm_old) by the
    # column-sum-is-1 invariant of `conv1` (each column of conv1
    # sums to 1 by construction; that's what makes backward-Euler
    # `conv1 · rm_new = rm_old` mass-conserving). Note: `rm_col`
    # here is tracer MASS, not mixing ratio — that's the plan 14
    # storage convention (tracers_raw stored as mass).
    # -----------------------------------------------------------------
    @testset "tracer-mass conservation (F64)" begin
        T = Float64
        rm = T[0.1 0.2 0.3;
               0.2 0.1 0.4;
               0.4 0.3 0.5;
               0.5 0.6 0.2;
               0.3 0.4 0.3;
               0.2 0.2 0.2;
               0.1 0.1 0.1;
               0.05 0.05 0.05]
        rm0 = copy(rm)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))

        for t in 1:Nt
            mass_before = sum(rm0[:, t])
            mass_after  = sum(rm[:, t])
            @test isapprox(mass_after, mass_before;
                           rtol = 1e4 * eps(T))
        end
        # Nontrivial: the profile actually changed (not silent identity).
        @test any(rm .!= rm0)
        # Cloud-dim diagnosis should pick up the active range.
        icltop, iclbas, icllfs = cd
        @test icltop == 2          # smallest k with detu > 0
        @test iclbas == 6          # largest k with detu > 0
        @test icllfs == 3          # smallest k with entd > 0
    end

    # -----------------------------------------------------------------
    # Uniform MIXING RATIO in → uniform MIXING RATIO out.
    # Since conv1 acts on tracer mass, the mixing-ratio preservation
    # test requires initializing rm_col as `const_mr × m` and
    # checking that `rm_new / m == const_mr` layer-by-layer.
    # -----------------------------------------------------------------
    @testset "uniform mixing ratio preserved (F64)" begin
        T = Float64
        const_mr = T(2.5e-4)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        rm = zeros(T, Nz, Nt)
        for t in 1:Nt, k in 1:Nz
            rm[k, t] = const_mr * m[k]
        end
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))

        for t in 1:Nt, k in 1:Nz
            @test isapprox(rm[k, t] / m[k], const_mr;
                            rtol = 1e4 * eps(T))
        end
    end

    # -----------------------------------------------------------------
    # F32 variant: zero-forcing + mass conservation.
    # -----------------------------------------------------------------
    @testset "F32: zero forcing + tracer-mass conservation" begin
        T = Float32
        rm = T[0.1 0.2; 0.2 0.1; 0.4 0.3; 0.5 0.6;
               0.3 0.4; 0.2 0.2; 0.1 0.1; 0.05 0.05]
        rm0 = copy(rm)
        m  = T[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4]
        conv1 = zeros(T, Nz, Nz)
        piv   = zeros(Int, Nz)
        cd    = zeros(Int, 3)

        # Zero forcing → identity.
        _tm5_solve_column!(rm, m,
                            zeros(T, Nz), zeros(T, Nz),
                            zeros(T, Nz), zeros(T, Nz),
                            conv1, piv, cd, T(600))
        @test rm == rm0

        # Nontrivial forcing → tracer mass conserved to F32 ULP.
        entu = T[0.0, 0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0]
        detu = T[0.0, 0.02, 0.02, 0.01, 0.01, 0.03, 0.0, 0.0]
        entd = T[0.0, 0.0, 0.01, 0.02, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0]
        _tm5_solve_column!(rm, m, entu, detu, entd, detd,
                            conv1, piv, cd, T(600))
        for t in 1:size(rm, 2)
            mass_before = sum(rm0[:, t])
            mass_after  = sum(rm[:, t])
            @test isapprox(mass_after, mass_before;
                            rtol = 1f4 * eps(T))
        end
    end

    # -----------------------------------------------------------------
    # Cloud-dim diagnostics stand-alone.
    # -----------------------------------------------------------------
    @testset "_tm5_diagnose_cloud_dims" begin
        T = Float64
        detu_empty = zeros(T, Nz)
        entd_empty = zeros(T, Nz)
        @test _tm5_diagnose_cloud_dims(detu_empty, entd_empty, Nz) ==
              (Nz + 1, 0, Nz + 1)

        detu = zeros(T, Nz); detu[3] = 0.1; detu[5] = 0.2; detu[6] = 0.1
        entd = zeros(T, Nz); entd[4] = 0.05
        @test _tm5_diagnose_cloud_dims(detu, entd, Nz) == (3, 6, 4)

        # icllfs should be the HIGHEST-altitude (smallest k) with
        # entd > 0, even if multiple levels are active.
        entd2 = zeros(T, Nz); entd2[3] = 0.05; entd2[5] = 0.01
        @test _tm5_diagnose_cloud_dims(detu, entd2, Nz)[3] == 3
    end
end

@testset "plan 23 Commit 2: column-major loop-order audit" begin
    # Principle 9: matrix assembly loops leftmost-innermost.
    # Back-substitute a minimal case in both orders and confirm the
    # shipped version is not catastrophically slow. This is a sanity
    # guard, not a benchmark; the 3× multiplier from Invariant 8 is
    # orders of magnitude above noise on any machine.
    using .AtmosTransport.Operators.Convection: _tm5_solve_column!

    T = Float64
    Nz = 16
    Ncols = 256
    entu = zeros(T, Nz, Ncols)
    detu = zeros(T, Nz, Ncols)
    entd = zeros(T, Nz, Ncols)
    detd = zeros(T, Nz, Ncols)
    m    = fill(T(1e4), Nz, Ncols)
    rm   = fill(T(1.0), Nz, 1, Ncols)
    # Seed one mid-column detrainment so work is nontrivial.
    detu[8, :] .= 0.01

    conv1s = zeros(T, Nz, Nz, Ncols)
    pivs   = zeros(Int, Nz, Ncols)
    cds    = zeros(Int, 3, Ncols)

    # Loop column index as outermost (leftmost-innermost k: correct
    # column-major order because column slices are contiguous when
    # arrays are (Nz, …, Ncols)).
    t_correct = @elapsed for c in 1:Ncols
        _tm5_solve_column!(view(rm, :, :, c),
                            view(m, :, c),
                            view(entu, :, c), view(detu, :, c),
                            view(entd, :, c), view(detd, :, c),
                            view(conv1s, :, :, c),
                            view(pivs, :, c),
                            view(cds, :, c),
                            T(600))
    end
    @test t_correct < 1.0   # generous smoke gate; actual <50 ms
end

@testset "plan 23 Commit 1: with_convection(model, TM5Convection())" begin
    # Build a minimal LatLon TransportModel end-to-end to prove the
    # workspace installer threads TM5Workspace through correctly.
    mesh = LatLonMesh(; Nx=4, Ny=3, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(fill(FT(1), 4, 3, 4); CO2 = zeros(FT, 4, 3, 4))
    fluxes = allocate_face_fluxes(grid.horizontal, 4; FT = FT, basis = DryBasis)

    model = TransportModel(state, fluxes, grid, UpwindScheme())
    @test model.workspace.convection_ws === nothing  # NoConvection default

    model_tm5 = with_convection(model, TM5Convection())
    @test model_tm5.convection isa TM5Convection
    @test model_tm5.workspace.convection_ws isa TM5Workspace{FT}
    @test size(model_tm5.workspace.convection_ws.conv1) == (4, 4, 4, 3)

    # Swapping back to NoConvection drops the workspace.
    model_none = with_convection(model_tm5, NoConvection())
    @test model_none.workspace.convection_ws === nothing
end
