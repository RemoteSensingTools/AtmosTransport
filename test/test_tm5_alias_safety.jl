#!/usr/bin/env julia
# Lock the `f_scratch === conv1` aliasing invariant. The TM5
# workspace stores `f_scratch` as the same array as `conv1`
# (`convection_workspace.jl` `TM5Workspace` constructor), saving an
# `(Nz, Nz)` slab per column. The aliasing is safe only because
# `_tm5_build_conv1!` overwrites row `k_atm` of `conv1` strictly
# after every read of `f[k_atm+1, ...]` for the same column. This
# test asserts bitwise equality between the aliased path and a
# separate-buffer path across the column-geometry cases that span
# the builder's behaviour: `icltop = 1`, `icltop = Nz/2`,
# `icltop > Nz`, and `icllfs < icltop`.

using Test
using Random

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Operators.Convection: _tm5_solve_column!,
                                             _tm5_diagnose_cloud_dims

# Returns (rm_out, conv1_out) for the two execution paths so the
# caller can compare both. `rm_in` and the four forcing vectors are
# copied internally; `m` is shared (read-only).
function _run_alias_pair(rm_in::AbstractMatrix{T},
                          m::AbstractVector{T},
                          entu::AbstractVector{T},
                          detu::AbstractVector{T},
                          entd::AbstractVector{T},
                          detd::AbstractVector{T},
                          dt::T) where {T}
    Nz = length(m)
    # --- Path A: separate buffers (f_buf is (Nz+1, Nz)).
    rmA    = copy(rm_in)
    conv1A = zeros(T, Nz, Nz)
    pivA   = zeros(Int, Nz)
    cdA    = zeros(Int, 3)
    fA     = zeros(T, Nz + 1, Nz)
    amuA   = zeros(T, Nz + 1)
    amdA   = zeros(T, Nz + 1)
    _tm5_solve_column!(rmA, m, entu, detu, entd, detd,
                        conv1A, pivA, cdA, dt;
                        f_buf = fA,
                        amu_buf = amuA, amd_buf = amdA)

    # --- Path B: aliased — f_buf shares storage with conv1_buf.
    # Production passes f_buf as a view into the same (Nz, Nz)
    # slab as conv1_buf. The build loop's `f[Nz+1, :]` row is
    # never read (the loop uses `zero(FT)` at k_atm == Nz).
    rmB    = copy(rm_in)
    conv1B = zeros(T, Nz, Nz)
    pivB   = zeros(Int, Nz)
    cdB    = zeros(Int, 3)
    fB     = conv1B  # alias
    amuB   = zeros(T, Nz + 1)
    amdB   = zeros(T, Nz + 1)
    _tm5_solve_column!(rmB, m, entu, detu, entd, detd,
                        conv1B, pivB, cdB, dt;
                        f_buf = fB,
                        amu_buf = amuB, amd_buf = amdB)

    return (rmA, conv1A, cdA), (rmB, conv1B, cdB)
end

@testset "TM5 alias safety: f_scratch === conv1" begin
    Nz = 12
    Nt = 3

    # Mass-per-area profile (kg/m²), surface heavy. Used by every fixture.
    m = Float64[1.0e4, 1.2e4, 1.5e4, 1.7e4, 2.0e4, 2.3e4,
                2.6e4, 2.9e4, 3.2e4, 3.5e4, 3.8e4, 4.1e4]

    Random.seed!(20260501)
    rm_seed = rand(Float64, Nz, Nt) .+ 0.01

    # ----------------------------------------------------------------
    # Fixture 1 — full-depth convection: detu nonzero starting at k=1
    # → icltop = 1, no upper cushion.
    # ----------------------------------------------------------------
    @testset "icltop = 1 (full depth)" begin
        T = Float64
        entu = T[0.04, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.0]
        detu = T[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.0]
        entd = T[0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.01, 0.0, 0.0]
        (rmA, c1A, cdA), (rmB, c1B, cdB) = _run_alias_pair(
            T.(rm_seed), m, entu, detu, entd, detd, T(600))
        @test cdA == cdB
        @test cdA[1] == 1                   # icltop = 1
        @test rmA == rmB                    # bitwise equal tracer output
        @test c1A == c1B                    # bitwise equal post-LU matrix
    end

    # ----------------------------------------------------------------
    # Fixture 2 — typical mid-trop convection: icltop ≈ Nz/2,
    # icllfs > icltop (downdraft LFS surface-ward of cloud top).
    # ----------------------------------------------------------------
    @testset "icltop = Nz/2 (typical), icllfs > icltop" begin
        T = Float64
        entu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
        detu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.03, 0.03, 0.02, 0.01, 0.0]
        entd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0.02, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.0]
        (rmA, c1A, cdA), (rmB, c1B, cdB) = _run_alias_pair(
            T.(rm_seed), m, entu, detu, entd, detd, T(600))
        @test cdA == cdB
        @test cdA[1] == 6                   # icltop
        @test cdA[3] > cdA[1]               # icllfs surface-ward
        @test rmA == rmB
        @test c1A == c1B
    end

    # ----------------------------------------------------------------
    # Fixture 3 — no convection: detu == 0 everywhere. Solver
    # short-circuits via the `icltop > Nz` early return; both paths
    # must produce the identity transform.
    # ----------------------------------------------------------------
    @testset "icltop > Nz (no convection)" begin
        T = Float64
        zeros_v = zeros(T, Nz)
        (rmA, c1A, cdA), (rmB, c1B, cdB) = _run_alias_pair(
            T.(rm_seed), m, zeros_v, zeros_v, zeros_v, zeros_v, T(600))
        @test cdA == cdB
        @test cdA == [Nz + 1, 0, Nz + 1]
        @test rmA == T.(rm_seed)            # identity
        @test rmB == T.(rm_seed)
        @test c1A == c1B                    # both paths leave conv1 untouched
    end

    # ----------------------------------------------------------------
    # Fixture 4 — pathological geometry icllfs < icltop. The
    # downdraft LFS is TOA-ward of the updraft cloud top: the
    # downdraft pass writes `f[r, c]` rows that overlap with the
    # combine pass's own writes, exercising the order-of-operations
    # constraint that makes the alias safe.
    # ----------------------------------------------------------------
    @testset "icllfs < icltop (downdraft LFS above updraft)" begin
        T = Float64
        # Updraft confined to lower half, detu only at k=8..10.
        entu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.02, 0.01, 0.0]
        detu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.01, 0.01, 0.0]
        # Downdraft enters at k=4 (entd nonzero), well above k=8.
        entd = T[0.0, 0.0, 0.0, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.01, 0.01, 0.0, 0.0]
        (rmA, c1A, cdA), (rmB, c1B, cdB) = _run_alias_pair(
            T.(rm_seed), m, entu, detu, entd, detd, T(600))
        @test cdA == cdB
        @test cdA[1] == 8                   # icltop
        @test cdA[3] == 4                   # icllfs (TOA-ward of icltop)
        @test cdA[3] < cdA[1]               # the failure-mode geometry
        @test rmA == rmB
        @test c1A == c1B
    end

    # ----------------------------------------------------------------
    # F32 spot check on the typical-convection fixture. The alias
    # safety is a structural property of the build loop and does
    # not depend on precision; the F32 case must hold bit-equal too.
    # ----------------------------------------------------------------
    @testset "F32 typical fixture" begin
        T = Float32
        entu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
        detu = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.03, 0.03, 0.02, 0.01, 0.0]
        entd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0.02, 0.0, 0.0]
        detd = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.0]
        m32 = Float32.(m)
        rm32 = Float32.(rm_seed)
        (rmA, c1A, cdA), (rmB, c1B, cdB) = _run_alias_pair(
            rm32, m32, entu, detu, entd, detd, T(600))
        @test cdA == cdB
        @test rmA == rmB
        @test c1A == c1B
    end
end
