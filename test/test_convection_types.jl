#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 18 Commit 1 — AbstractConvectionOperator + NoConvection
#
# Minimal: type hierarchy and no-op operator. No kernels yet. Tests cover
# what's shipped in this commit:
#
# 1. Type hierarchy: `NoConvection() isa AbstractConvectionOperator`.
# 2. State-level identity: `apply!(state, ConvectionForcing(), grid,
#    NoConvection(), dt; workspace=nothing)` returns state bit-exact.
# 3. Array-level identity: `apply_convection!(q_raw, air_mass,
#    ConvectionForcing(), NoConvection(), dt, ws, grid) === nothing`.
# 4. Dispatch correctness: `NoConvection` resolves to the no-op, not to
#    any future concrete convection operator.
# 5. Exported symbols visible at `AtmosTransport` level.
# 6. Works on both `CellState{DryBasis}` and `CellState{MoistBasis}`.
#
# Concrete operators (`CMFMCConvection`, `TM5Convection`) land in
# plan 18 Commits 3 and 4.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "Type hierarchy: NoConvection <: AbstractConvectionOperator" begin
    @test NoConvection <: AbstractConvectionOperator
    @test NoConvection() isa AbstractConvectionOperator
    @test AbstractConvectionOperator isa DataType || AbstractConvectionOperator isa UnionAll ||
          AbstractConvectionOperator === AbstractConvectionOperator  # just verify the type exists
    # `NoConvection` is a singleton — no fields.
    @test fieldnames(NoConvection) === ()
end

@testset "ConvectionForcing default construction" begin
    f = ConvectionForcing()
    @test f.cmfmc === nothing
    @test f.dtrain === nothing
    @test f.tm5_fields === nothing
    @test !has_convection_forcing(f)

    # Plan 18 Commit 1 also ships the 3-arg positional (auto-generated
    # by Julia from the struct definition). Commit 2 will add validating
    # inner constructors.
    g = ConvectionForcing(nothing, nothing, nothing)
    @test g.cmfmc === nothing
    @test g.dtrain === nothing
    @test g.tm5_fields === nothing
    @test !has_convection_forcing(g)

    # Non-nothing payloads work (Commit 2 will enforce invariants).
    cmfmc_arr = zeros(Float64, 3, 2, 5)    # (Nx, Ny, Nz+1)
    dtrain_arr = zeros(Float64, 3, 2, 4)   # (Nx, Ny, Nz)
    h = ConvectionForcing(cmfmc_arr, dtrain_arr, nothing)
    @test h.cmfmc === cmfmc_arr
    @test h.dtrain === dtrain_arr
    @test h.tm5_fields === nothing
    @test has_convection_forcing(h)
end

@testset "State-level apply!(::NoConvection) is identity" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    for Basis in (DryBasis, MoistBasis)
        m = ones(FT, Nx, Ny, Nz)
        rm_init = FT(400e-6) .* m
        state = CellState(Basis, m; CO2 = copy(rm_init))

        # Snapshot before apply!
        m_before = copy(state.air_mass)
        rm_before = copy(state.tracers_raw)

        result = apply!(state, ConvectionForcing(), grid, NoConvection(),
                        FT(1800); workspace = nothing)

        # Return value is the same state object
        @test result === state

        # Bit-exact: NoConvection is a dead branch, no FP work.
        @test state.air_mass == m_before
        @test state.tracers_raw == rm_before
    end
end

@testset "Array-level apply_convection!(::NoConvection) is no-op" begin
    FT = Float64
    Nx, Ny, Nz, Nt = 4, 3, 2, 1

    q_raw = zeros(FT, Nx, Ny, Nz, Nt) .+ FT(400e-6)
    air_mass = ones(FT, Nx, Ny, Nz)
    q_before = copy(q_raw)
    m_before = copy(air_mass)

    result = apply_convection!(q_raw, air_mass, ConvectionForcing(),
                                NoConvection(), FT(1800), nothing, nothing)

    @test result === nothing
    @test q_raw == q_before
    @test air_mass == m_before

    # Even with non-nothing payload, NoConvection still no-ops.
    cmfmc = fill(FT(0.1), Nx, Ny, Nz + 1)
    dtrain = fill(FT(0.05), Nx, Ny, Nz)
    forcing = ConvectionForcing(cmfmc, dtrain, nothing)
    q_raw2 = copy(q_before)
    result2 = apply_convection!(q_raw2, air_mass, forcing, NoConvection(),
                                 FT(1800), nothing, nothing)
    @test result2 === nothing
    @test q_raw2 == q_before    # still unchanged: NoConvection ignores payload
end

@testset "Dispatch correctness: NoConvection dead branch" begin
    # Exercises the `!(op isa NoConvection)` check that TransportModel.step!
    # (Commit 6) uses to skip the convection block entirely when no
    # operator is installed. Here we confirm that `op isa NoConvection`
    # works on the singleton.
    @test NoConvection() isa NoConvection
    @test !(NoConvection() isa AbstractConvectionOperator) == false  # double-negation clarity

    # Type-stability check — no allocations for the no-op on a fresh
    # state. `@allocated` on the repeated call should be 0 after warmup.
    FT = Float64
    mesh = LatLonMesh(; FT = FT, Nx = 2, Ny = 2)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    m = ones(FT, 2, 2, 2)
    state = CellState(m; CO2 = FT(400e-6) .* m)
    forcing = ConvectionForcing()

    # Warmup
    apply!(state, forcing, grid, NoConvection(), FT(1); workspace = nothing)
    # Measure
    allocs = @allocated apply!(state, forcing, grid, NoConvection(),
                                FT(1); workspace = nothing)
    @test allocs == 0
end

@testset "Exported symbols visible at AtmosTransport" begin
    # All new plan-18-Commit-1 symbols are reachable as plain names from
    # `using .AtmosTransport` callers.
    @test isdefined(AtmosTransport, :AbstractConvectionOperator)
    @test isdefined(AtmosTransport, :NoConvection)
    @test isdefined(AtmosTransport, :apply_convection!)
    @test isdefined(AtmosTransport, :ConvectionForcing)
    @test isdefined(AtmosTransport, :has_convection_forcing)
end
