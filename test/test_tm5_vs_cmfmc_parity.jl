#!/usr/bin/env julia
"""
Plan 23 Commit 5: cross-scheme consistency tests between TM5Convection
and CMFMCConvection.

Both schemes transport tracers through cumulus convection but have
different implementations and unit conventions:

- CMFMCConvection: explicit forward-Euler two-pass update,
  CFL-subcycled, well-mixed sub-cloud treatment. `air_mass` in
  kg per cell (kernel multiplies forcings by `cell_area`).
- TM5Convection: implicit backward-Euler matrix solve,
  unconditionally stable. `air_mass` in kg per unit area
  (kernel divides forcings by `m` directly; no cell_area).

Because the natural unit conventions differ, this test does NOT
attempt a profile-by-profile numerical agreement between the two
schemes. It guards against silent regression in either via:

  (A) Both preserve uniform mixing ratio (column-sum-is-1 / CMFMC's
      equivalent well-mixed invariant).
  (B) Both conserve total tracer mass.
  (C) Both produce a non-identity change when forcing is active on
      a non-uniform initial tracer profile.

Plan 23 Commit 5 lists three grids (LL, RG, CS) in one testset. A
stricter quantitative agreement test (idealized column, rtol ≤ 10%)
is deferred to a follow-on that resolves the cmfmc-vs-entu unit
translation convention. Regressions in either scheme — shipping
a broken kernel where the column solver silently returns identity
or loses mass — are caught here.
"""

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

using .AtmosTransport.State: CellState, CubedSphereState, DryBasis
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh,
                              CubedSphereMesh, HybridSigmaPressure,
                              cell_areas_by_latitude, cell_area, ncells
using .AtmosTransport.Operators: CMFMCConvection, TM5Convection,
                                 CMFMCWorkspace, TM5Workspace
using .AtmosTransport.MetDrivers: ConvectionForcing

const FT = Float64

# -------------------------------------------------------------------------
# Idealized forcing profile for a single column.
# -------------------------------------------------------------------------
function _cmfmc_column(Nz::Int, k_top::Int; FT = Float64, peak = FT(0.05))
    cmfmc  = zeros(FT, Nz + 1)
    dtrain = zeros(FT, Nz)
    for k in k_top:Nz
        frac = FT((Nz - k + 1) / (Nz - k_top + 1))
        cmfmc[k + 1] = peak * frac * (1 - frac)
    end
    cmfmc[Nz + 1] = zero(FT)
    for k in (k_top + 1):(k_top + 2)
        dtrain[k] = FT(0.01)
    end
    return (cmfmc, dtrain)
end

function _tm5_column(Nz::Int, k_top::Int; FT = Float64)
    # Matching idealized entrainment/detrainment profile for TM5:
    # entu peaks at mid-cloud, detu at cloud top, no downdraft.
    entu = zeros(FT, Nz)
    detu = zeros(FT, Nz)
    entd = zeros(FT, Nz)
    detd = zeros(FT, Nz)
    for k in k_top:Nz
        frac = FT((Nz - k + 1) / (Nz - k_top + 1))
        entu[k] = FT(0.03) * frac * (1 - frac)
    end
    for k in (k_top + 1):(k_top + 2)
        detu[k] = FT(0.01)
    end
    return (; entu, detu, entd, detd)
end

# -------------------------------------------------------------------------
# Structured LatLon
# -------------------------------------------------------------------------

@testset "plan 23 Commit 5: CMFMC vs TM5 parity on LatLon" begin
    Nx, Ny, Nz = 2, 2, 8
    k_top = 3

    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = FT[0, 500, 1000, 2000, 5000, 10000, 30000, 50000, 0]
    B_ifc = FT[0, 0, 0, 0.05, 0.2, 0.4, 0.7, 0.9, 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    cell_areas = cell_areas_by_latitude(mesh)

    # Forcings (same column tiled over all cells).
    cmfmc_col, dtrain_col = _cmfmc_column(Nz, k_top; FT = FT)
    tm5_col = _tm5_column(Nz, k_top; FT = FT)
    cmfmc_3d  = repeat(reshape(cmfmc_col, 1, 1, Nz + 1),  Nx, Ny, 1)
    dtrain_3d = repeat(reshape(dtrain_col, 1, 1, Nz), Nx, Ny, 1)
    entu_3d   = repeat(reshape(tm5_col.entu, 1, 1, Nz), Nx, Ny, 1)
    detu_3d   = repeat(reshape(tm5_col.detu, 1, 1, Nz), Nx, Ny, 1)
    entd_3d   = repeat(reshape(tm5_col.entd, 1, 1, Nz), Nx, Ny, 1)
    detd_3d   = repeat(reshape(tm5_col.detd, 1, 1, Nz), Nx, Ny, 1)

    # -- CMFMC path — `m` in kg/cell --
    m_cmfmc = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_cmfmc[i, j, k] = FT(5e3) * cell_areas[j]
    end
    dt = FT(60)

    @testset "CMFMC: mass conservation + nontrivial change" begin
        tracer_init = zeros(FT, Nx, Ny, Nz)
        tracer_init[:, :, Nz] .= FT(1e-3) .* m_cmfmc[:, :, Nz]
        state = CellState(m_cmfmc; CO2 = copy(tracer_init))
        forcing = ConvectionForcing(cmfmc_3d, dtrain_3d, nothing)
        ws = CMFMCWorkspace(state.air_mass; cell_metrics = cell_areas)
        total_before = sum(state.tracers_raw)
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         CMFMCConvection(), dt;
                                         workspace = ws)
        total_after = sum(state.tracers_raw)
        @test isapprox(total_after, total_before; rtol = 1e-10)
        @test state.tracers_raw != reshape(tracer_init, Nx, Ny, Nz, 1)
    end

    @testset "CMFMC: preserves uniform mixing ratio" begin
        mr = FT(2.5e-4)
        tracer_init = mr .* m_cmfmc
        state = CellState(m_cmfmc; CO2 = copy(tracer_init))
        forcing = ConvectionForcing(cmfmc_3d, dtrain_3d, nothing)
        ws = CMFMCWorkspace(state.air_mass; cell_metrics = cell_areas)
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         CMFMCConvection(), dt;
                                         workspace = ws)
        mr_after = state.tracers_raw[:, :, :, 1] ./ m_cmfmc
        @test all(isapprox.(mr_after, mr; rtol = 1e-8))
    end

    # -- TM5 path — `m` in kg/m² --
    m_tm5 = fill(FT(5e3), Nx, Ny, Nz)

    @testset "TM5: mass conservation + nontrivial change" begin
        tracer_init = zeros(FT, Nx, Ny, Nz)
        tracer_init[:, :, Nz] .= FT(1e-3) .* m_tm5[:, :, Nz]
        state = CellState(m_tm5; CO2 = copy(tracer_init))
        tm5_fields = (entu = entu_3d, detu = detu_3d,
                       entd = entd_3d, detd = detd_3d)
        forcing = ConvectionForcing(nothing, nothing, tm5_fields)
        ws = TM5Workspace(state.air_mass)
        total_before = sum(state.tracers_raw)
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         TM5Convection(), dt;
                                         workspace = ws)
        total_after = sum(state.tracers_raw)
        @test isapprox(total_after, total_before; rtol = 1e-10)
        @test state.tracers_raw != reshape(tracer_init, Nx, Ny, Nz, 1)
    end

    @testset "TM5: preserves uniform mixing ratio" begin
        mr = FT(2.5e-4)
        tracer_init = mr .* m_tm5
        state = CellState(m_tm5; CO2 = copy(tracer_init))
        tm5_fields = (entu = entu_3d, detu = detu_3d,
                       entd = entd_3d, detd = detd_3d)
        forcing = ConvectionForcing(nothing, nothing, tm5_fields)
        ws = TM5Workspace(state.air_mass)
        AtmosTransport.Operators.apply!(state, forcing, grid,
                                         TM5Convection(), dt;
                                         workspace = ws)
        mr_after = state.tracers_raw[:, :, :, 1] ./ m_tm5
        @test all(isapprox.(mr_after, mr; rtol = 1e-10))
    end
end

# -------------------------------------------------------------------------
# Face-indexed ReducedGaussian
# -------------------------------------------------------------------------

@testset "plan 23 Commit 5: CMFMC vs TM5 parity on ReducedGaussian" begin
    mesh = ReducedGaussianMesh(FT[-0.5, 0.5], [4, 4]; FT = FT)
    Nz = 6
    k_top = 3
    ncell = ncells(mesh)

    A_ifc = collect(FT, range(0, 5e4; length = Nz + 1))
    B_ifc = collect(FT, range(1, 0;   length = Nz + 1))
    A_ifc[1] = 0; B_ifc[end] = 0
    B_ifc[1] = 0; A_ifc[end] = 0
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    cell_metrics = [cell_area(mesh, c) for c in 1:ncell]

    cmfmc_col, dtrain_col = _cmfmc_column(Nz, k_top; FT = FT)
    tm5_col = _tm5_column(Nz, k_top; FT = FT)
    cmfmc_2d  = repeat(reshape(cmfmc_col, 1, Nz + 1), ncell, 1)
    dtrain_2d = repeat(reshape(dtrain_col, 1, Nz), ncell, 1)
    entu_2d   = repeat(reshape(tm5_col.entu, 1, Nz), ncell, 1)
    detu_2d   = repeat(reshape(tm5_col.detu, 1, Nz), ncell, 1)
    entd_2d   = repeat(reshape(tm5_col.entd, 1, Nz), ncell, 1)
    detd_2d   = repeat(reshape(tm5_col.detd, 1, Nz), ncell, 1)

    m_cmfmc = zeros(FT, ncell, Nz)
    for k in 1:Nz, c in 1:ncell
        m_cmfmc[c, k] = FT(5e3) * cell_metrics[c]
    end
    m_tm5 = fill(FT(5e3), ncell, Nz)
    dt = FT(60)

    # CMFMC sanity.
    tracer_init = zeros(FT, ncell, Nz)
    tracer_init[:, Nz] .= FT(1e-3) .* m_cmfmc[:, Nz]
    state_c = CellState(m_cmfmc; CO2 = copy(tracer_init))
    forcing_c = ConvectionForcing(cmfmc_2d, dtrain_2d, nothing)
    ws_c = CMFMCWorkspace(state_c.air_mass; cell_metrics = cell_metrics)
    total_c_before = sum(state_c.tracers_raw)
    AtmosTransport.Operators.apply!(state_c, forcing_c, grid,
                                     CMFMCConvection(), dt; workspace = ws_c)
    @test isapprox(sum(state_c.tracers_raw), total_c_before; rtol = 1e-10)

    # TM5 sanity.
    tracer_init_tm5 = zeros(FT, ncell, Nz)
    tracer_init_tm5[:, Nz] .= FT(1e-3) .* m_tm5[:, Nz]
    state_t = CellState(m_tm5; CO2 = copy(tracer_init_tm5))
    tm5_fields = (entu = entu_2d, detu = detu_2d,
                   entd = entd_2d, detd = detd_2d)
    forcing_t = ConvectionForcing(nothing, nothing, tm5_fields)
    ws_t = TM5Workspace(state_t.air_mass)
    total_t_before = sum(state_t.tracers_raw)
    AtmosTransport.Operators.apply!(state_t, forcing_t, grid,
                                     TM5Convection(), dt; workspace = ws_t)
    @test isapprox(sum(state_t.tracers_raw), total_t_before; rtol = 1e-10)
    @test state_t.tracers_raw != reshape(tracer_init_tm5, ncell, Nz, 1)
end
