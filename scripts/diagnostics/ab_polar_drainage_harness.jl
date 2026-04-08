#!/usr/bin/env julia
# ============================================================================
# A/B harness for polar drainage diagnostics (mass_fixer=false investigations)
#
# Runs two configurations, captures pass/fail behavior (including nloop aborts),
# and compares post-run tracer diagnostics from output NetCDF when available.
#
# Usage:
#   julia --project=. scripts/diagnostics/ab_polar_drainage_harness.jl \
#       configA.toml configB.toml [--max-windows N] [--workdir DIR]
#
# Example:
#   julia --project=. scripts/diagnostics/ab_polar_drainage_harness.jl \
#       config/runs/era5_f64_debug_moist_v4_nofix.toml \
#       config/runs/era5_f64_debug_moist_v4_nofix_candidate.toml \
#       --max-windows 48 --workdir /tmp/ab_polar
# ============================================================================

using Dates
using Printf
using Statistics
using TOML

using NCDatasets

if Sys.isapple()
    using Metal
else
    using CUDA
    CUDA.allowscalar(false)
end
using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!

struct RunSummary
    label::String
    config_path::String
    env_overrides::Dict{String,String}
    output_path::String
    success::Bool
    elapsed_s::Float64
    err_text::String
    has_nloop_abort::Bool
    co2_nan_count::Int
    co2_neg_count::Int
    sh_pole_std_ppm::Union{Nothing,Float64}
    sh_mid_std_ppm::Union{Nothing,Float64}
    global_mean_ppm::Union{Nothing,Float64}
end

function _parse_cli(argv::Vector{String})
    positional = String[]
    max_windows = nothing
    workdir = "/tmp/ab_polar_drainage"
    env_a = Dict{String,String}()
    env_b = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--max-windows"
            i == length(argv) && error("--max-windows requires an integer")
            max_windows = parse(Int, argv[i + 1])
            i += 2
        elseif a == "--workdir"
            i == length(argv) && error("--workdir requires a path")
            workdir = expanduser(argv[i + 1])
            i += 2
        elseif a == "--env-a" || a == "--env-b"
            i == length(argv) && error("$a requires KEY=VALUE")
            kv = argv[i + 1]
            eq = findfirst(==('='), kv)
            eq === nothing && error("$a requires KEY=VALUE, got: $kv")
            key = strip(kv[1:eq-1])
            val = kv[eq+1:end]
            isempty(key) && error("$a key must be non-empty")
            if a == "--env-a"
                env_a[key] = val
            else
                env_b[key] = val
            end
            i += 2
        else
            push!(positional, a)
            i += 1
        end
    end
    length(positional) == 2 || error("Need exactly two config paths. Got $(length(positional)).")
    return positional[1], positional[2], max_windows, workdir, env_a, env_b
end

function _set_output_path!(cfg::Dict{String,Any}, outpath::String)
    out_cfg = get!(cfg, "output", Dict{String,Any}())
    out_cfg["filename"] = outpath
    return cfg
end

function _set_max_windows!(cfg::Dict{String,Any}, max_windows::Int)
    md = get!(cfg, "met_data", Dict{String,Any}())
    md["max_windows"] = max_windows
    return cfg
end

function _safe_array(a)
    try
        return Array(a)
    catch
        return nothing
    end
end

function _co2_quality_metrics(model)
    if !(model.tracers isa NamedTuple) || !haskey(model.tracers, :co2)
        return 0, 0
    end
    co2 = _safe_array(model.tracers.co2)
    co2 === nothing && return 0, 0
    return sum(isnan, co2), sum(co2 .< 0)
end

function _surface_metrics_from_netcdf(path::String)
    if !isfile(path)
        return nothing, nothing, nothing
    end
    ds = NCDataset(path, "r")
    try
        vname = if haskey(ds, "co2_surface")
            "co2_surface"
        elseif haskey(ds, "co2")
            "co2"
        else
            return nothing, nothing, nothing
        end
        A = ds[vname][:]
        nd = ndims(A)
        sfc = if nd == 3
            A[:, :, end]
        elseif nd == 4
            A[:, :, end, end]
        else
            return nothing, nothing, nothing
        end
        Ny = size(sfc, 2)
        n_pole = max(1, round(Int, 0.03 * Ny))
        sh = 1:max(1, Ny ÷ 2)
        sh_pole = 1:n_pole
        sh_mid_start = min(length(sh), n_pole + 1)
        sh_mid = sh_mid_start:length(sh)
        sh_pole_std = std(Float64.(sfc[:, sh_pole])) * 1e6
        sh_mid_std = isempty(sh_mid) ? NaN : std(Float64.(sfc[:, sh_mid])) * 1e6
        global_mean = mean(Float64.(sfc)) * 1e6
        return sh_pole_std, sh_mid_std, global_mean
    finally
        close(ds)
    end
end

function _with_env_overrides(f::Function, env_overrides::Dict{String,String})
    old_vals = Dict{String,Union{Nothing,String}}()
    for (k, v) in env_overrides
        old_vals[k] = haskey(ENV, k) ? ENV[k] : nothing
        ENV[k] = v
    end
    try
        return f()
    finally
        for (k, oldv) in old_vals
            if oldv === nothing
                pop!(ENV, k, nothing)
            else
                ENV[k] = oldv
            end
        end
    end
end

function _run_case(label::String, cfg_path::String, outpath::String, max_windows,
                   env_overrides::Dict{String,String})
    ok = true
    err = ""
    elapsed = 0.0
    model = _with_env_overrides(env_overrides) do
        cfg = TOML.parsefile(cfg_path)
        _set_output_path!(cfg, outpath)
        if max_windows !== nothing
            _set_max_windows!(cfg, max_windows)
        end
        mdl = build_model_from_config(cfg)
        t0 = time()
        try
            run!(mdl)
        catch e
            ok = false
            err = sprint(showerror, e, catch_backtrace())
        end
        elapsed = time() - t0
        mdl
    end

    co2_nan, co2_neg = _co2_quality_metrics(model)
    sh_pole_std, sh_mid_std, global_mean = _surface_metrics_from_netcdf(outpath)
    has_nloop = occursin("nloop hit max_nloop", lowercase(err))

    return RunSummary(
        label, cfg_path, env_overrides, outpath, ok, elapsed, err, has_nloop,
        co2_nan, co2_neg, sh_pole_std, sh_mid_std, global_mean
    )
end

function _fmt(x::Union{Nothing,Float64}; digits::Int=3)
    x === nothing && return "n/a"
    if isnan(x)
        return "nan"
    end
    return string(round(x, digits=digits))
end

function _print_summary(a::RunSummary, b::RunSummary)
    println("\n=== A/B Polar Drainage Summary ===")
    for s in (a, b)
        println("\n--- $(s.label) ---")
        @printf("config:   %s\n", s.config_path)
        if isempty(s.env_overrides)
            @printf("env:      (none)\n")
        else
            env_str = join(sort(collect(["$k=$v" for (k, v) in s.env_overrides])), ", ")
            @printf("env:      %s\n", env_str)
        end
        @printf("output:   %s\n", s.output_path)
        @printf("success:  %s\n", string(s.success))
        @printf("elapsed:  %.1f s\n", s.elapsed_s)
        @printf("nloop abort: %s\n", string(s.has_nloop_abort))
        @printf("co2 NaN / neg: %d / %d\n", s.co2_nan_count, s.co2_neg_count)
        @printf("SH pole std (ppm): %s\n", _fmt(s.sh_pole_std_ppm))
        @printf("SH mid  std (ppm): %s\n", _fmt(s.sh_mid_std_ppm))
        @printf("Global mean (ppm): %s\n", _fmt(s.global_mean_ppm))
    end

    println("\n=== Verdict (drainage-focused) ===")
    if a.success != b.success
        winner = a.success ? a.label : b.label
        println("Primary: $winner wins (other run failed).")
    elseif a.has_nloop_abort != b.has_nloop_abort
        winner = a.has_nloop_abort ? b.label : a.label
        println("Primary: $winner wins (avoids nloop max abort).")
    elseif a.co2_nan_count != b.co2_nan_count
        winner = a.co2_nan_count < b.co2_nan_count ? a.label : b.label
        println("Primary: $winner wins (fewer NaNs).")
    else
        println("Primary: tie on hard-failure criteria.")
    end

    if a.sh_pole_std_ppm !== nothing && b.sh_pole_std_ppm !== nothing
        da = a.sh_pole_std_ppm
        db = b.sh_pole_std_ppm
        if !(isnan(da) || isnan(db))
            better = da < db ? a.label : (db < da ? b.label : "tie")
            @printf("Secondary (lower SH pole std): %s  (A=%.3f ppm, B=%.3f ppm)\n",
                    better, da, db)
        end
    end
end

function main()
    if length(ARGS) < 2
        println("""
Usage:
  julia --project=. scripts/diagnostics/ab_polar_drainage_harness.jl <config_A.toml> <config_B.toml> [--max-windows N] [--workdir DIR]
""")
        return
    end

    cfg_a, cfg_b, max_windows, workdir, env_a, env_b = _parse_cli(ARGS)
    mkpath(workdir)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_a = joinpath(workdir, "A_$(stamp).nc")
    out_b = joinpath(workdir, "B_$(stamp).nc")

    println("Running A/B harness")
    println("A: $cfg_a")
    println("B: $cfg_b")
    println("workdir: $workdir")
    max_windows !== nothing && println("max_windows override: $max_windows")
    !isempty(env_a) && println("env A overrides: " * join(["$k=$v" for (k, v) in sort(collect(env_a))], ", "))
    !isempty(env_b) && println("env B overrides: " * join(["$k=$v" for (k, v) in sort(collect(env_b))], ", "))

    A = _run_case("A", cfg_a, out_a, max_windows, env_a)
    B = _run_case("B", cfg_b, out_b, max_windows, env_b)
    _print_summary(A, B)

    if !A.success
        println("\nA error (truncated):")
        println(first(split(A.err_text, '\n'), 12) |> x -> join(x, "\n"))
    end
    if !B.success
        println("\nB error (truncated):")
        println(first(split(B.err_text, '\n'), 12) |> x -> join(x, "\n"))
    end
end

main()

