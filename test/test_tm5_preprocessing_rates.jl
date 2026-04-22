#!/usr/bin/env julia
"""
Plan 24 Commit 1 — `ec2tm_from_rates!` + `dz_hydrostatic_virtual!`
unit tests.

The full-fidelity rate-input variant mirrors TM5's F90
`ECconv_to_TMconv` (deps/tm5/base/src/phys_convec_ec2tm.F90:87-237)
line by line: small-value clip + dz integration + uptop/dotop
search + mass-budget closure + symmetric negative redistribution.
Tests here:

1. Byte-for-byte parity with a hand-computed reference for a
   nontrivial 6-level column.
2. Zero-forcing → zero output (no-convection short-circuit).
3. Bad-data stress — injected negatives, sub-threshold noise,
   empty-direction columns — all handled per TM5's rules.
4. `TM5CleanupStats` counters increment as expected.
5. `dz_hydrostatic_virtual!` vs `dz_hydrostatic_constT!` divergence
   is bounded in the troposphere.
6. No regressions against plan 23's `ec2tm!` tests.

All tests run core (no real data).
"""

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

using .AtmosTransport.Preprocessing: ec2tm!, ec2tm_from_rates!,
                                       TM5CleanupStats,
                                       dz_hydrostatic_virtual!,
                                       dz_hydrostatic_constT!

# ─────────────────────────────────────────────────────────────────────────
# Reference column used across several testsets
# ─────────────────────────────────────────────────────────────────────────

# 6-level column with a convective cloud band in the middle.
# AtmosTransport orientation: k=1=TOA, k=6=surface.  Interface
# k has layer k below (closer to surface) if we follow the
# F90→Julia 1-based shift: interface i=1..Nz+1, layer k=1..Nz,
# interface i=k is TOP of layer k, interface i=k+1 is BOTTOM.
const _NZ_REF     = 6
const _UDMF_REF   = Float64[0.00, 0.00, 0.02, 0.05, 0.03, 0.00, 0.00]
const _DDMF_REF   = Float64[0.00, 0.00, 0.00, -0.01, -0.02, 0.00, 0.00]
const _UDRF_RATE  = Float64[0.0, 0.0, 1.0e-4, 2.0e-4, 1.0e-4, 0.0]
const _DDRF_RATE  = Float64[0.0, 0.0, 0.0, 5.0e-5, 1.0e-4, 0.0]
const _DZ_REF     = Float64[200.0, 200.0, 200.0, 150.0, 120.0, 80.0]

# ─────────────────────────────────────────────────────────────────────────
# Helper: hand-computed reference
# ─────────────────────────────────────────────────────────────────────────

"""
Reproduce F90 `ECconv_to_TMconv` step by step, independently of the
Julia port, for cross-check.  The two must agree bit-for-bit.
"""
function _f90_reference_column(udmf_in, ddmf_in, udrf_rate, ddrf_rate, dz, Nz)
    # Make local copies; the F90 algorithm mutates its scratch.
    mflu = copy(udmf_in)
    mfld = copy(ddmf_in)
    detu = copy(udrf_rate)
    detd = copy(ddrf_rate)
    entu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz)

    # Clipping (F90 L141-144).
    @. mflu = ifelse(mflu < 1e-6,  0.0, mflu)
    @. detu = ifelse(detu < 1e-10, 0.0, detu)
    @. mfld = ifelse(mfld > -1e-6, 0.0, mfld)
    @. detd = ifelse(detd < 1e-10, 0.0, detd)

    # dz integration (F90 L146-151).
    for k in 1:Nz
        detu[k] *= dz[k]
        detd[k] *= dz[k]
    end

    # uptop/dotop (F90 L153-173) — scan 1..Nz+1 for first nonzero.
    uptop = 0
    for k in 1:(Nz + 1)
        if mflu[k] > 0.0
            uptop = k
            break
        end
    end
    dotop = 0
    for k in 1:(Nz + 1)
        if mfld[k] < 0.0
            dotop = k
            break
        end
    end

    # Updraft closure (F90 L175-192).
    if uptop > 0 && uptop <= Nz
        for k in 1:(uptop - 1)
            entu[k] = 0.0; detu[k] = 0.0
        end
        for k in uptop:Nz
            entu[k] = mflu[k] - mflu[k + 1] + detu[k]
        end
    else
        fill!(entu, 0.0); fill!(detu, 0.0)
    end
    if dotop > 0 && dotop <= Nz
        for k in 1:(dotop - 1)
            entd[k] = 0.0; detd[k] = 0.0
        end
        for k in dotop:Nz
            entd[k] = mfld[k] - mfld[k + 1] + detd[k]
        end
    else
        fill!(entd, 0.0); fill!(detd, 0.0)
    end

    # Negative redistribution (F90 L214-232).
    for k in 1:Nz
        if entu[k] < 0
            detu[k] -= entu[k]; entu[k] = 0.0
        end
        if detu[k] < 0
            entu[k] -= detu[k]; detu[k] = 0.0
        end
        if entd[k] < 0
            detd[k] -= entd[k]; entd[k] = 0.0
        end
        if detd[k] < 0
            entd[k] -= detd[k]; detd[k] = 0.0
        end
    end
    return (entu, detu, entd, detd)
end

# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

@testset "plan 24 Commit 1: ec2tm_from_rates! F90 parity" begin
    Nz = _NZ_REF
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    udmf = copy(_UDMF_REF); ddmf = copy(_DDMF_REF)
    udrf = copy(_UDRF_RATE); ddrf = copy(_DDRF_RATE)
    dz   = copy(_DZ_REF)

    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz)

    ref_entu, ref_detu, ref_entd, ref_detd = _f90_reference_column(
        _UDMF_REF, _DDMF_REF, _UDRF_RATE, _DDRF_RATE, _DZ_REF, Nz)

    # Bit-exact match.
    @test entu == ref_entu
    @test detu == ref_detu
    @test entd == ref_entd
    @test detd == ref_detd

    # All outputs non-negative (F90 invariant).
    @test all(entu .>= 0)
    @test all(detu .>= 0)
    @test all(entd .>= 0)
    @test all(detd .>= 0)

    # In the updraft window only, entu - detu matches the flux
    # divergence -(udmf[k+1] - udmf[k]) = udmf[k] - udmf[k+1].
    udmf_clipped = copy(_UDMF_REF)
    @. udmf_clipped = ifelse(udmf_clipped < 1e-6, 0.0, udmf_clipped)
    for k in 3:5   # the active updraft levels by construction
        @test isapprox(entu[k] - detu[k],
                        udmf_clipped[k] - udmf_clipped[k + 1];
                        atol = 1e-12)
    end
end

@testset "plan 24 Commit 1: ec2tm_from_rates! zero forcing" begin
    Nz = 5
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    udmf = zeros(Float64, Nz + 1); ddmf = zeros(Float64, Nz + 1)
    udrf = zeros(Float64, Nz); ddrf = zeros(Float64, Nz)
    dz   = fill(100.0, Nz)

    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz)
    @test all(entu .== 0)
    @test all(detu .== 0)
    @test all(entd .== 0)
    @test all(detd .== 0)
end

@testset "plan 24 Commit 1: ec2tm_from_rates! bad-data cleanup" begin
    Nz = 5
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)

    # Sub-threshold updraft noise: should all get clipped to 0 ⇒
    # no updraft at all ⇒ all outputs zero.
    udmf = fill(0.5e-6, Nz + 1)     # all < 1e-6 → zeroed
    ddmf = fill(-0.5e-6, Nz + 1)    # all > -1e-6 → zeroed
    udrf = fill(0.5e-10, Nz)        # all < 1e-10 → zeroed
    ddrf = fill(0.5e-10, Nz)
    dz   = fill(100.0, Nz)

    stats = TM5CleanupStats()
    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz; stats = stats)

    @test all(entu .== 0) && all(detu .== 0)
    @test all(entd .== 0) && all(detd .== 0)
    @test stats.levels_udmf_clipped[] == Nz + 1
    @test stats.levels_ddmf_clipped[] == Nz + 1
    @test stats.levels_udrf_clipped[] == Nz
    @test stats.levels_ddrf_clipped[] == Nz
    @test stats.no_updraft[]    == 1
    @test stats.no_downdraft[]  == 1
    @test stats.columns_processed[] == 1
end

@testset "plan 24 Commit 1: ec2tm_from_rates! negative redistribution" begin
    # Construct a column where the naive mass-budget closure produces
    # a NEGATIVE entu at some level.  Redistribution should swap the
    # negative into detu, preserving (entu - detu) in the ACTIVE
    # window (uptop..Nz).
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    # udmf increases upward (interfaces 2→3): 0.03 → 0.05.  Small detu
    # so entu[2] = udmf[2] - udmf[3] + small > 0 is NOT satisfied.
    # With udrf=1e-5 and dz=200 → detu[2]*dz = 2e-3, far less than
    # the 0.02 flux divergence → entu[2] negative.
    udmf = Float64[0.0, 0.03, 0.05, 0.00, 0.00]
    ddmf = zeros(Float64, Nz + 1)
    udrf = Float64[0.0, 1.0e-5, 5.0e-4, 0.0]   # kg/m³/s, small
    ddrf = zeros(Float64, Nz)
    dz   = fill(200.0, Nz)

    stats = TM5CleanupStats()
    # We keep a copy of the pre-redistribution (entu_raw - detu_raw)
    # via the reference column so we can assert invariance.
    ref_entu, ref_detu, _, _ = _f90_reference_column(
        udmf, ddmf, udrf, ddrf, dz, Nz)

    ec2tm_from_rates!(entu, detu, entd, detd,
                       udmf, ddmf, udrf, ddrf, dz, Nz; stats = stats)

    @test all(entu .>= 0)
    @test all(detu .>= 0)
    @test stats.levels_entu_neg[] >= 1

    # Match the F90-reference outputs byte-for-byte.
    @test entu == ref_entu
    @test detu == ref_detu

    # Mass-budget preserved in the ACTIVE updraft window (uptop..Nz).
    # Levels above uptop are explicitly zeroed by the algorithm;
    # don't test mass balance there (udmf can have a zero above a
    # nonzero and that's the physical "boundary" of the cloud).
    uptop_k = findfirst(udmf .> 0.0)   # = 2
    for k in uptop_k:Nz
        @test isapprox(entu[k] - detu[k],
                        udmf[k] - udmf[k + 1];
                        atol = 1e-12)
    end
end

@testset "plan 24 Commit 1: dz_hydrostatic_virtual! sanity" begin
    # Typical ERA5 L137-like column: realistic T/Q + ps.
    Nz = 5
    # Descending ak/bk so p_top < p_bot at each level.
    ak = Float64[0.0, 100.0, 500.0, 1000.0, 5000.0, 30000.0]
    bk = Float64[0.0, 0.0,   0.05,  0.2,    0.6,    1.0]
    ps = 101325.0
    T  = Float64[210, 220, 240, 260, 280]  # K
    Q  = Float64[0.0, 0.0, 0.001, 0.005, 0.010]  # kg/kg
    dz_virt  = zeros(Float64, Nz)
    dz_const = zeros(Float64, Nz)

    dz_hydrostatic_virtual!(dz_virt,  T, Q, ps, ak, bk, Nz)
    dz_hydrostatic_constT!(dz_const, ps, ak, bk, Nz; T_ref = 260)

    # All dz strictly positive.
    @test all(dz_virt .> 0)
    @test all(dz_const .> 0)

    # Virtual-temperature variant gives larger dz where air is warm/moist
    # (lower density → more height per hPa).  Surface layer k=Nz has
    # highest T and Q.
    @test dz_virt[Nz] > dz_const[Nz]    # surface warmer than 260 → more dz

    # Divergence bounded in troposphere: within 30% everywhere, typical
    # 10-20% at surface.
    for k in 1:Nz
        rel = abs(dz_virt[k] - dz_const[k]) / dz_const[k]
        @test rel < 0.30
    end
end

@testset "plan 24 Commit 1: dz_hydrostatic_virtual! shape guards" begin
    ak = Float64[0, 100, 500]; bk = Float64[0, 0.1, 1.0]
    ps = 101325.0
    T = Float64[220, 260]; Q = Float64[0.001, 0.005]
    dz = zeros(Float64, 2)

    # Correct shapes → no throw.
    dz_hydrostatic_virtual!(dz, T, Q, ps, ak, bk, 2)
    @test all(dz .> 0)

    # Mismatched sizes → clear errors.
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        zeros(3), T, Q, ps, ak, bk, 2)
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        dz, T, Float64[0.001], ps, ak, bk, 2)
    @test_throws ArgumentError dz_hydrostatic_virtual!(
        dz, T, Q, ps, Float64[0, 100], bk, 2)
end

@testset "plan 24 Commit 1: ec2tm! (plan-23 minimal) unchanged" begin
    # Regression: the plan-23 ec2tm! signature and behaviour must stay
    # identical.  One zero-forcing test is enough to catch accidental
    # behaviour drift from the new ec2tm_from_rates! additions.
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    mflu = zeros(Float64, Nz + 1); mfld = zeros(Float64, Nz + 1)
    detu_ec = zeros(Float64, Nz); detd_ec = zeros(Float64, Nz)

    ec2tm!(entu, detu, entd, detd, mflu, mfld, detu_ec, detd_ec)
    @test all(entu .== 0) && all(detu .== 0)
    @test all(entd .== 0) && all(detd .== 0)
end

@testset "plan 24 Commit 1: ec2tm_from_rates! shape guards" begin
    Nz = 4
    entu = zeros(Float64, Nz); detu = zeros(Float64, Nz)
    entd = zeros(Float64, Nz); detd = zeros(Float64, Nz)
    udmf = zeros(Float64, Nz + 1); ddmf = zeros(Float64, Nz + 1)
    udrf = zeros(Float64, Nz); ddrf = zeros(Float64, Nz)
    dz   = fill(100.0, Nz)

    # Wrong entu length → clear error.
    @test_throws ArgumentError ec2tm_from_rates!(
        zeros(Float64, 3), detu, entd, detd,
        udmf, ddmf, udrf, ddrf, dz, Nz)

    # Wrong udmf length → clear error.
    @test_throws ArgumentError ec2tm_from_rates!(
        entu, detu, entd, detd,
        zeros(Float64, Nz), ddmf, udrf, ddrf, dz, Nz)
end
