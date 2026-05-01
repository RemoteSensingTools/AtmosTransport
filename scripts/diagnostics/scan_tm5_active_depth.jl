#!/usr/bin/env julia
# TM5-storage Commit 5 — per-binary active-depth measurement.
#
# Reads the per-window CSV produced by
# `scripts/validation/diagnose_tm5_active_layers.jl`, aggregates the
# code-active and any-active extremes, looks up the binary's hybrid
# coefficients, and writes a TOML safety file next to the binary that
# Commits 7 / 8 read at sim-setup time.
#
# Usage:
#   julia --project=. scripts/diagnostics/scan_tm5_active_depth.jl \
#       --binary <path.bin> --csv <prefix>.summary.csv [--margin 1] [--ps_pa 100000]
#
# If `--csv` is omitted the script invokes `diagnose_tm5_active_layers.jl`
# automatically with threshold 1e-6.

using TOML
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: CubedSphereBinaryReader

# Tiny CSV parser tailored to diagnose_tm5_active_layers's fixed schema —
# avoids pulling CSV.jl into Project.toml for a one-shot diagnostic.
function _read_summary_csv(path::AbstractString)
    lines = strip.(readlines(path))
    isempty(lines) && error("empty CSV: $(path)")
    header = split(lines[1], ',')
    idx = Dict(name => i for (i, name) in enumerate(header))
    needed = ("min_top_code", "min_top_any", "median_top_code", "p95_top_code",
              "any_without_detu_columns", "max_abs_without_detu", "n_columns")
    for k in needed
        haskey(idx, k) || error("CSV missing column `$(k)`: $(path)")
    end
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        cells = split(line, ',')
        row = (
            min_top_code              = parse(Int,     cells[idx["min_top_code"]]),
            min_top_any               = parse(Int,     cells[idx["min_top_any"]]),
            median_top_code           = parse(Float64, cells[idx["median_top_code"]]),
            p95_top_code              = parse(Float64, cells[idx["p95_top_code"]]),
            any_without_detu_columns  = parse(Int,     cells[idx["any_without_detu_columns"]]),
            max_abs_without_detu      = parse(Float64, cells[idx["max_abs_without_detu"]]),
            n_columns                 = parse(Int,     cells[idx["n_columns"]]),
        )
        push!(rows, row)
    end
    return rows
end

const USAGE = """
Usage: julia --project=. scripts/diagnostics/scan_tm5_active_depth.jl \\
           --binary <path.bin> --csv <prefix>.summary.csv
           [--margin 1] [--ps_pa 100000] [--out <path.toml>]
"""

function _parse(argv)
    binary = ""; csv = ""; margin = 1; ps_pa = 100000.0; out = ""
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--binary"; binary = expanduser(argv[i+1]); i += 2
        elseif a == "--csv"; csv = expanduser(argv[i+1]); i += 2
        elseif a == "--margin"; margin = parse(Int, argv[i+1]); i += 2
        elseif a == "--ps_pa"; ps_pa = parse(Float64, argv[i+1]); i += 2
        elseif a == "--out"; out = expanduser(argv[i+1]); i += 2
        elseif a in ("-h", "--help"); println(USAGE); exit(0)
        else; error("unknown arg `$(a)`\n$(USAGE)")
        end
        i ≤ length(argv) || break
    end
    isempty(binary) && error("--binary required\n$(USAGE)")
    isfile(binary) || error("binary not found: $(binary)")
    isempty(csv) || isfile(csv) || error("csv not found: $(csv)")
    return (; binary, csv, margin, ps_pa, out)
end

function _interface_pressure_pa(A_ifc::Vector{Float64}, B_ifc::Vector{Float64},
                                 k_ifc::Int, ps_pa::Float64)
    1 <= k_ifc <= length(A_ifc) || error("interface $(k_ifc) out of range 1:$(length(A_ifc))")
    return A_ifc[k_ifc] + B_ifc[k_ifc] * ps_pa
end

function main()
    opts = _parse(ARGS)

    # Open header for vertical coordinates.
    rdr = CubedSphereBinaryReader(opts.binary; FT = Float32)
    h = rdr.header
    Nz = h.nlevel

    # If the CSV wasn't provided, generate it via the existing scanner.
    csv_path = if isempty(opts.csv)
        prefix = joinpath(dirname(opts.binary), "tm5_active_safety_scan")
        cmd = `julia --project=. scripts/validation/diagnose_tm5_active_layers.jl $(prefix) $(opts.binary) 1e-6`
        @info "running diagnose scanner: $cmd"
        run(cmd)
        prefix * ".summary.csv"
    else
        opts.csv
    end

    rows = _read_summary_csv(csv_path)
    isempty(rows) && error("CSV is empty: $(csv_path)")

    # Aggregate across windows.
    min_top_code = minimum(r.min_top_code for r in rows)
    min_top_any  = minimum(r.min_top_any  for r in rows)
    median_top_code = median(Float64.([r.median_top_code for r in rows]))
    p95_top_code    = median(Float64.([r.p95_top_code    for r in rows]))
    detu_zero_other = sum(r.any_without_detu_columns for r in rows)
    n_columns_total = sum(r.n_columns for r in rows)
    detu_zero_other_frac = detu_zero_other / max(n_columns_total, 1)
    detu_zero_other_max = maximum(r.max_abs_without_detu for r in rows)

    # Recommended cap: 1-level margin TOA-ward of the deepest code-active top
    # seen across any window of any column (smallest k = highest level).
    recommended_min_active_level = max(1, min_top_code - opts.margin)

    # Pressure at that *interface* (top of layer k) under ps = ps_pa.
    A = Float64.(h.A_ifc); B = Float64.(h.B_ifc)
    p_recommended_pa = _interface_pressure_pa(A, B, recommended_min_active_level, opts.ps_pa)
    p_min_top_code_pa = _interface_pressure_pa(A, B, min_top_code, opts.ps_pa)
    p_min_top_any_pa  = _interface_pressure_pa(A, B, min_top_any,  opts.ps_pa)

    out_path = isempty(opts.out) ?
        joinpath(dirname(opts.binary), "tm5_active_safety.toml") :
        opts.out

    payload = Dict(
        "tm5_safety" => Dict(
            "binary"                                => opts.binary,
            "scanned_windows"                       => length(rows),
            "n_levels"                              => Nz,
            "tolerance"                             => 1.0e-6,
            "ps_pa_for_pressure"                    => opts.ps_pa,
            "deepest_active_level_from_top_code"    => Int(min_top_code),
            "deepest_active_pressure_hpa_code"      => p_min_top_code_pa / 100,
            "deepest_active_level_from_top_any"     => Int(min_top_any),
            "deepest_active_pressure_hpa_any"       => p_min_top_any_pa / 100,
            "recommended_min_active_level"          => Int(recommended_min_active_level),
            "recommended_min_active_pressure_hpa"   => p_recommended_pa / 100,
            "level_margin"                          => opts.margin,
            "median_top_code"                       => median_top_code,
            "p95_top_code"                          => p95_top_code,
            "detu_zero_other_nonzero_fraction"      => detu_zero_other_frac,
            "detu_zero_other_nonzero_max"           => detu_zero_other_max,
        ),
    )
    open(out_path, "w") do io; TOML.print(io, payload); end
    @info "wrote $(out_path)"
    @info @sprintf("recommended_min_active_level = %d (%.1f hPa)  margin = %d  Nz = %d",
                   recommended_min_active_level, p_recommended_pa / 100, opts.margin, Nz)
    @info @sprintf("detu==0 with other forcings nonzero: %.2f%% of column-hours, max |.| = %.3e",
                   100 * detu_zero_other_frac, detu_zero_other_max)
    return 0
end

exit(main())
