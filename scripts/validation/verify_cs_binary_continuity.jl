#!/usr/bin/env julia
#
# Test 4 — replay-continuity validation for cubed-sphere transport binaries.
#
# For windows 1..Nt-1, replay the stored fluxes against the next stored
# window's `m` field.  For the final window, use the explicit `dm` payload
# when present: m_next = m_cur + (2 * steps_per_window) * dm.

using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: CubedSphereBinaryReader, load_cs_window,
    load_flux_delta_window!, verify_window_continuity_cs

const USAGE = """
Usage: julia --project=. scripts/validation/verify_cs_binary_continuity.jl \\
           --binary <transport_cs.bin> [--threshold 1e-10] [--report <out.csv>]
"""

function parse_args(argv::Vector{String})
    args = Dict{String, String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a in ("--binary", "--threshold", "--report")
            i + 1 <= length(argv) || error("Missing value for $a")
            args[a] = argv[i + 1]
            i += 2
        elseif a == "-h" || a == "--help"
            println(USAGE)
            exit(0)
        else
            error("Unknown argument: $a\n$USAGE")
        end
    end
    haskey(args, "--binary") || error("Missing --binary\n$USAGE")
    return args
end

function derived_next_mass(cur_m, dm)
    return ntuple(length(cur_m)) do p
        cur_m[p] .+ dm[p]
    end
end

function dm_endpoint_consistency(cur_m, dm, next_m)
    max_abs = 0.0
    denom = eps(Float64)
    worst_idx = (0, 0, 0, 0)
    for p in eachindex(cur_m)
        @inbounds for idx in CartesianIndices(next_m[p])
            expected = Float64(cur_m[p][idx]) + Float64(dm[p][idx])
            actual = Float64(next_m[p][idx])
            denom = max(denom, abs(actual))
            err = abs(expected - actual)
            if err > max_abs
                max_abs = err
                worst_idx = (p, Tuple(idx)...)
            end
        end
    end
    return (max_abs_err = max_abs, max_rel_err = max_abs / denom,
            worst_idx = worst_idx)
end

function maybe_write_report(path::AbstractString, rows)
    isempty(path) && return nothing
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "window,max_abs_err_kg,max_rel_err,worst_panel,worst_i,worst_j,worst_k,target_source")
        for row in rows
            @printf(io, "%d,%.12e,%.12e,%d,%d,%d,%d,%s\n",
                    row.window, row.max_abs_err, row.max_rel_err,
                    row.worst_idx..., row.target_source)
        end
    end
    return nothing
end

function main(argv = ARGS)
    args = parse_args(Vector{String}(argv))
    path = args["--binary"]
    threshold = parse(Float64, get(args, "--threshold", "1e-10"))
    report = get(args, "--report", "")

    @info "Test 4: CS binary replay-continuity validation"
    @info "  binary: $path"
    @info "  threshold: $threshold"

    reader = CubedSphereBinaryReader(path; FT = Float64)
    try
        h = reader.header
        Nt = h.nwindow
        steps = h.steps_per_window
        has_dm = :dm in h.payload_sections
        @info @sprintf("  grid: C%d panels=%d levels=%d windows=%d steps/window=%d",
                       h.Nc, h.npanel, h.nlevel, Nt, steps)
        @info "  payload sections: $(h.payload_sections)"

        rows = NamedTuple[]
        worst_rel = -Inf
        worst_abs = 0.0
        worst_win = 0
        worst_idx = (0, 0, 0, 0)
        worst_source = ""

        dm_worst = nothing
        cur = load_cs_window(reader, 1)
        for win in 1:Nt
            target_source = ""
            m_next = nothing
            next = nothing

            if win < Nt
                next = load_cs_window(reader, win + 1)
                m_next = next.m
                target_source = "next_window_m"

                if has_dm
                    deltas = load_flux_delta_window!(reader, win)
                    d = dm_endpoint_consistency(cur.m, deltas.dm, next.m)
                    if dm_worst === nothing || d.max_rel_err > dm_worst.max_rel_err
                        dm_worst = (window = win, max_abs_err = d.max_abs_err,
                                    max_rel_err = d.max_rel_err, worst_idx = d.worst_idx)
                    end
                end
            elseif has_dm
                deltas = load_flux_delta_window!(reader, win)
                m_next = derived_next_mass(cur.m, deltas.dm)
                target_source = "dm_payload"
            else
                @warn "Final window has no following mass endpoint and no dm payload; skipping it."
                break
            end

            diag = verify_window_continuity_cs(cur.m, cur.am, cur.bm, cur.cm,
                                               m_next, steps)
            row = (window = win, max_abs_err = diag.max_abs_err,
                   max_rel_err = diag.max_rel_err, worst_idx = diag.worst_idx,
                   target_source = target_source)
            push!(rows, row)
            if diag.max_rel_err > worst_rel
                worst_rel = diag.max_rel_err
                worst_abs = diag.max_abs_err
                worst_win = win
                worst_idx = diag.worst_idx
                worst_source = target_source
            end

            @info @sprintf("  window %02d: rel=%.3e abs=%.3e kg worst=%s target=%s",
                           win, diag.max_rel_err, diag.max_abs_err,
                           string(diag.worst_idx), target_source)

            cur = next === nothing ? cur : next
            win % 4 == 0 && GC.gc(false)
        end

        maybe_write_report(report, rows)
        isempty(report) || @info "  report written: $report"

        @info @sprintf("Worst replay residual: rel=%.3e abs=%.3e kg window=%d cell=%s target=%s",
                       worst_rel, worst_abs, worst_win, string(worst_idx), worst_source)
        if dm_worst !== nothing
            @info @sprintf("Worst dm→next-window endpoint mismatch: rel=%.3e abs=%.3e kg window=%d cell=%s",
                           dm_worst.max_rel_err, dm_worst.max_abs_err,
                           dm_worst.window, string(dm_worst.worst_idx))
        end

        if worst_rel <= threshold
            @info "PASS — replay continuity is within threshold."
            return 0
        else
            @error @sprintf("FAIL — replay continuity %.3e exceeds threshold %.3e",
                            worst_rel, threshold)
            return 1
        end
    finally
        close(reader)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
