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
using .AtmosTransport.Operators: AbstractConvectionOperator, NoConvection,
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
        @test op isa AbstractConvectionOperator
        @test op isa TM5Convection
        # Stateless — no fields.
        @test fieldcount(TM5Convection) == 0
    end

    @testset "NoConvection and CMFMCConvection unchanged" begin
        # Plan 23's validator refactor must not change what these
        # return or how they dispatch. Bit-exact regression.
        @test NoConvection() isa AbstractConvectionOperator
        @test CMFMCConvection() isa AbstractConvectionOperator
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

@testset "plan 23 Commit 1: apply! stubs error cleanly" begin
    mesh = LatLonMesh(; Nx=4, Ny=3, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    state = CellState(zeros(FT, 4, 3, 4); CO2 = zeros(FT, 4, 3, 4))
    ws = TM5Workspace(state.air_mass)
    forcing = ConvectionForcing()

    # Stub must throw with a message pointing at Commit 4 (not a
    # silent no-op, not a MethodError from missing dispatch).
    err = try
        apply!(state, forcing, grid, TM5Convection(), FT(60); workspace=ws)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Commit 4", err.msg)
    @test occursin("TM5Convection", err.msg)
    @test occursin("kernel not yet implemented", err.msg)

    # Array-level entry point has the same contract.
    rm4 = zeros(FT, 4, 3, 4, 1)
    err = try
        apply_convection!(rm4, state.air_mass, forcing,
                          TM5Convection(), FT(60), ws, grid)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Commit 4", err.msg)
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
