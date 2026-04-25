#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan indexed-baking-valiant Commit 1 — AbstractMetSettings trait + RawWindow
#
# Verifies the source-axis abstraction:
#   1. RawWindow{FT,A2,A3} carries both window endpoints + integrated fluxes.
#   2. has_convection defaults to false on any AbstractMetSettings subtype.
#   3. The undefined interface methods are callable as generic functions
#      (i.e. read_window!, source_grid, windows_per_day exist and dispatch
#      will route to subtype methods once GEOS lands in Commit 3).
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Preprocessing: AbstractMetSettings, RawWindow,
                                      read_window!, source_grid,
                                      windows_per_day, has_convection

struct DummyMetSettings <: AbstractMetSettings end

@testset "Met source trait" begin
    FT = Float64
    Nx, Ny, Nz = 4, 4, 3

    @testset "RawWindow holds both endpoints + window-integrated fluxes" begin
        m       = zeros(FT, Nx, Ny, Nz)
        ps      = zeros(FT, Nx, Ny)
        m_next  = zeros(FT, Nx, Ny, Nz)
        ps_next = zeros(FT, Nx, Ny)
        am      = zeros(FT, Nx + 1, Ny, Nz)   # x-face flux
        bm      = zeros(FT, Nx, Ny + 1, Nz)   # y-face flux

        raw = RawWindow{FT, typeof(ps), typeof(m)}(
            m, ps, nothing,
            m_next, ps_next, nothing,
            am, bm,
            nothing, nothing,
            nothing, nothing,
        )

        @test size(raw.m)       == (Nx, Ny, Nz)
        @test size(raw.m_next)  == (Nx, Ny, Nz)
        @test size(raw.ps)      == (Nx, Ny)
        @test size(raw.ps_next) == (Nx, Ny)
        @test size(raw.am)      == (Nx + 1, Ny, Nz)
        @test size(raw.bm)      == (Nx, Ny + 1, Nz)
        @test raw.qv === nothing && raw.qv_next === nothing
        @test raw.u  === nothing && raw.v       === nothing
        @test raw.cmfmc === nothing && raw.dtrain === nothing
    end

    @testset "RawWindow with optional fields populated" begin
        m       = zeros(FT, Nx, Ny, Nz)
        ps      = zeros(FT, Nx, Ny)
        am      = zeros(FT, Nx + 1, Ny, Nz)
        bm      = zeros(FT, Nx, Ny + 1, Nz)
        qv      = zeros(FT, Nx, Ny, Nz)
        u       = zeros(FT, Nx, Ny, Nz)
        v       = zeros(FT, Nx, Ny, Nz)

        raw = RawWindow{FT, typeof(ps), typeof(m)}(
            m, ps, qv,
            copy(m), copy(ps), copy(qv),
            am, bm,
            u, v,
            nothing, nothing,
        )

        @test raw.qv      isa AbstractArray{FT, 3}
        @test raw.qv_next isa AbstractArray{FT, 3}
        @test raw.u       isa AbstractArray{FT, 3}
        @test raw.v       isa AbstractArray{FT, 3}
    end

    @testset "Interface defaults" begin
        @test has_convection(DummyMetSettings()) === false
    end

    @testset "Interface methods exist as generic functions" begin
        @test read_window!     isa Function
        @test source_grid      isa Function
        @test windows_per_day  isa Function
        @test has_convection   isa Function
    end
end
