# Per-direction (x / y / z separately) substep-requirement analysis for a CS
# transport binary — the same palindrome outgoing-budget math as
# verify_substep_positivity_cs!, computed offline from reader windows.
# Usage: julia --project=. split_substep_analysis.jl <binary.bin>
using Printf
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: CubedSphereBinaryReader, load_cs_window,
    flux_kind, flux_storage_substep_scale

path = ARGS[1]
reader = CubedSphereBinaryReader(path; FT = Float32)
h = reader.header
sched = h.steps_per_window_by_window
fk = flux_kind(reader)
target = Float64(get(h.raw_header, "substep_cfl_target", 0.85))
Nt = h.nwindow
@printf("%s\n  C%d  windows=%d  flux_kind=%s  cfl_target=%.2f  schedule=%d..%d\n",
        basename(path), h.Nc, Nt, fk, target, minimum(sched), maximum(sched))

function window_ratios(cur, nxt_m, scale::Float32)
    worst = zeros(Float64, 3)   # x, y, z
    for p in 1:6
        m  = cur.m[p];  mn = nxt_m === nothing ? cur.m[p] : nxt_m[p]
        am = cur.am[p]; bm = cur.bm[p]; cm = cur.cm[p]
        Nc = size(m, 1); Nz = size(m, 3)
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mref = min(m[i, j, k], mn[i, j, k])
            mref > 0f0 || continue
            ox = max(0f0, -am[i, j, k]) + max(0f0, am[i + 1, j, k])
            oy = max(0f0, -bm[i, j, k]) + max(0f0, bm[i, j + 1, k])
            oz = max(0f0, -cm[i, j, k]) + max(0f0, cm[i, j, k + 1])
            r = 2f0 * scale / mref
            worst[1] = max(worst[1], Float64(ox * r))
            worst[2] = max(worst[2], Float64(oy * r))
            worst[3] = max(worst[3], Float64(oz * r))
        end
    end
    return worst
end

req(steps, ratio) = max(1, ceil(Int, steps * ratio / target))

rx = Int[]; ry = Int[]; rz = Int[]; rc = Int[]
cur = load_cs_window(reader, 1)
for w in 1:Nt
    nxt = w < Nt ? load_cs_window(reader, w + 1) : nothing
    scale = Float32(flux_storage_substep_scale(Float32, sched[w], fk))
    wr = window_ratios(cur, nxt === nothing ? nothing : nxt.m, scale)
    push!(rx, req(sched[w], wr[1]))
    push!(ry, req(sched[w], wr[2]))
    push!(rz, req(sched[w], wr[3]))
    # combined uses the sum of budgets (what the stored schedule enforces)
    push!(rc, req(sched[w], wr[1] + wr[2] + wr[3]))
    nxt === nothing || (global cur = nxt)
end

using Statistics
@printf("  %-9s %6s %6s %6s\n", "direction", "median", "p90", "max")
for (name, v) in (("x", rx), ("y", ry), ("z", rz), ("combined", rc), ("stored", collect(sched)))
    @printf("  %-9s %6d %6d %6d\n", name, round(Int, median(v)),
            round(Int, quantile(v, 0.9)), maximum(v))
end
println("  per-window  x=", rx)
println("  per-window  y=", ry)
println("  per-window  z=", rz)
