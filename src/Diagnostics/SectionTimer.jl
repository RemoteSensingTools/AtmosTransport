module SectionTimer

# Plan TM5-storage-redesign Commit 1.
#
# Hand-rolled host-side section timer. Off by default; enabled when
# `ENV["ATMOSTR_TIMERS"] == "1"` at run-start. Measures wall-clock at
# named host call boundaries; relies on the existing `synchronize(backend)`
# calls inside operator applys to make host time ≈ GPU time per section.
# Per-phase decomposition inside a single kernel launch (build / LU / solve
# inside `_tm5_solve_column!`) is not host-timeable; that lives in the
# Commit 1b CPU microbenchmark.

using Printf

const _ENABLED = Ref(false)
const _TIMINGS = Dict{Symbol, Vector{Float64}}()
const _WALL_T0 = Ref{Float64}(0.0)
const _WALL_TOTAL = Ref{Float64}(0.0)

"""
    enable!()
Reset and turn on. Called at run start when `ATMOSTR_TIMERS=1`.
"""
function enable!()
    empty!(_TIMINGS)
    _ENABLED[] = true
    _WALL_T0[] = time_ns() / 1e9
    _WALL_TOTAL[] = 0.0
    return nothing
end

"""
    disable!()
Stop accumulating; existing samples remain in `_TIMINGS` until `enable!`
is called again.
"""
function disable!()
    _WALL_TOTAL[] = (time_ns() / 1e9) - _WALL_T0[]
    _ENABLED[] = false
    return nothing
end

is_enabled() = _ENABLED[]

@inline function _record_ns!(section::Symbol, ns::Float64)
    samples = get!(() -> Float64[], _TIMINGS, section)
    push!(samples, ns)
    return nothing
end

"""
    @section name expr
Time `expr` and accumulate the elapsed nanoseconds under `name` (a
`Symbol`). When timers are off the macro just executes `expr` with
zero overhead beyond a single `Ref` load.
"""
macro section(name, expr)
    quote
        if _ENABLED[]
            local _t0 = time_ns()
            local _result = $(esc(expr))
            _record_ns!($(esc(name)), Float64(time_ns() - _t0))
            _result
        else
            $(esc(expr))
        end
    end
end

"""
    time_section(f, name::Symbol)
Function-form equivalent of `@section`. Use when the timed region is
a do-block or already a closure.
"""
@inline function time_section(f, name::Symbol)
    _ENABLED[] || return f()
    t0 = time_ns()
    result = f()
    _record_ns!(name, Float64(time_ns() - t0))
    return result
end

function _summary_row(samples::Vector{Float64})
    n = length(samples)
    n == 0 && return (0, 0.0, 0.0, 0.0)
    total_s = sum(samples) / 1e9
    mean_ms = sum(samples) / n / 1e6
    sorted = sort(samples)
    p95_ms = sorted[max(1, ceil(Int, 0.95 * n))] / 1e6
    return (n, total_s, mean_ms, p95_ms)
end

"""
    report(io = stderr)
Print a per-section summary table. Columns: section, n_calls, total_s,
mean_ms, p95_ms, fraction_of_total. Fraction is over the sum of section
totals (not over wall-clock — a section can overlap none, so coverage
is reported separately).
"""
function report(io::IO = stderr)
    isempty(_TIMINGS) && (println(io, "[SectionTimer] no samples"); return)
    section_total_ns = sum(sum(v) for v in values(_TIMINGS); init=0.0)
    wall_s = _WALL_TOTAL[] > 0 ? _WALL_TOTAL[] : (time_ns() / 1e9 - _WALL_T0[])
    @printf(io, "[SectionTimer] wall=%.2fs  covered=%.2fs (%.1f%%)\n",
            wall_s, section_total_ns / 1e9,
            wall_s > 0 ? 100 * (section_total_ns / 1e9) / wall_s : 0.0)
    @printf(io, "%-30s %8s %10s %10s %10s %8s\n",
            "section", "n_calls", "total_s", "mean_ms", "p95_ms", "frac%")
    for (sec, samples) in sort(collect(_TIMINGS); by = p -> -sum(p.second))
        n, total_s, mean_ms, p95_ms = _summary_row(samples)
        frac = 100 * (sum(samples) / max(section_total_ns, eps()))
        @printf(io, "%-30s %8d %10.3f %10.3f %10.3f %8.2f\n",
                String(sec), n, total_s, mean_ms, p95_ms, frac)
    end
    return nothing
end

"""
    write_csv(path)
Emit the same summary as `report` to a CSV at `path`. Header:
`section,n_calls,total_s,mean_ms,p95_ms,fraction_of_total`.
Returns the path on success, or `nothing` if there are no samples.
"""
function write_csv(path::AbstractString)
    isempty(_TIMINGS) && return nothing
    section_total_ns = sum(sum(v) for v in values(_TIMINGS); init=0.0)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "section,n_calls,total_s,mean_ms,p95_ms,fraction_of_total")
        for (sec, samples) in sort(collect(_TIMINGS); by = p -> -sum(p.second))
            n, total_s, mean_ms, p95_ms = _summary_row(samples)
            frac = sum(samples) / max(section_total_ns, eps())
            @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f\n",
                    String(sec), n, total_s, mean_ms, p95_ms, frac)
        end
    end
    return path
end

"""
    maybe_enable_from_env!()
Inspect `ENV["ATMOSTR_TIMERS"]` at call time. `"1"` / `"true"` / `"on"`
turn timers on; anything else (or unset) is a no-op. Idempotent.
"""
function maybe_enable_from_env!()
    raw = get(ENV, "ATMOSTR_TIMERS", "")
    if lowercase(raw) in ("1", "true", "on", "yes")
        enable!()
        return true
    end
    return false
end

export @section, time_section, enable!, disable!, is_enabled,
       report, write_csv, maybe_enable_from_env!

end # module SectionTimer
