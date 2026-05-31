#!/usr/bin/env julia
# Quantify how much the convection CADENCE (per-window vs per-advection-substep)
# changes the tracer fields, and compare that magnitude to the convection
# GROUPING approximation (n_merge=1 bit-exact vs n_merge=3 layer-aggregation).
#
#   A = per-window,  n_merge=1   (the fix; reference)
#   B = per-substep, n_merge=1   (old cadence — isolates the cadence change)
#   C = per-window,  n_merge=3   (isolates the grouping approximation)
#
# Reports, per tracer, the A↔B (cadence) and A↔C (grouping) differences at the
# final snapshot as max|Δ|, RMS(Δ), and RMS(Δ) relative to the field RMS — so we
# can see whether the cadence change moves the answer more or less than grouping.
#
# Usage: julia --project=. scripts/diagnostics/compare_convection_cadence_experiment.jl \
#            A.nc B.nc C.nc

using NCDatasets, Statistics, Printf

const TRACERS = ("co2_natural", "co2_fossil", "sf6", "rn222")

_last(v) = Float64.(coalesce.(v[:, :, :, :, end], NaN))   # final time slice, 4D → fill missing

function _stats(ref, other)
    d = other .- ref
    keep = .!(isnan.(d))
    dd = d[keep]; rr = ref[keep]
    rms_ref = sqrt(mean(abs2, rr))
    rms_d   = sqrt(mean(abs2, dd))
    (maxabs = maximum(abs, dd), rms = rms_d, rel = rms_d / rms_ref, rms_ref = rms_ref)
end

function main(a, b, c)
    dsA = NCDataset(a); dsB = NCDataset(b); dsC = NCDataset(c)
    @printf "%-12s | %-34s | %-34s | verdict\n" "tracer" "A↔B  CADENCE (per-sub vs per-win)" "A↔C  GROUPING (n_merge 1 vs 3)"
    println(repeat("-", 104))
    for t in TRACERS
        haskey(dsA, t) || continue
        A = _last(dsA[t]); B = _last(dsB[t]); C = _last(dsC[t])
        sB = _stats(A, B); sC = _stats(A, C)
        ratio = sB.rms / max(sC.rms, eps())
        verdict = ratio > 2 ? "CADENCE dominates ($(round(ratio,digits=1))×)" :
                  ratio < 0.5 ? "grouping dominates ($(round(1/ratio,digits=1))×)" :
                  "comparable"
        @printf "%-12s | rel=%.2e rms=%.3g max=%.3g | rel=%.2e rms=%.3g max=%.3g | %s\n" t sB.rel sB.rms sB.maxabs sC.rel sC.rms sC.maxabs verdict
    end
    println()
    println("rel = RMS(Δ) / RMS(field); units are dry VMR. CADENCE column isolates the ",
            "2026-05-31 fix (per-window) vs the old per-substep behaviour on the SAME binary.")
    close(dsA); close(dsB); close(dsC)
end

length(ARGS) == 3 || error("usage: A.nc B.nc C.nc")
main(ARGS[1], ARGS[2], ARGS[3])
