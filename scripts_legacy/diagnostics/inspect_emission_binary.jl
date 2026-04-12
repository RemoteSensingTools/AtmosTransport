#!/usr/bin/env julia
# =====================================================================
# inspect_emission_binary.jl — Quick inspection of CATRINE emission binaries
#
# Usage:
#   julia scripts/diagnostics/inspect_emission_binary.jl <file.bin>
#   julia scripts/diagnostics/inspect_emission_binary.jl <file.bin> --snapshot 12
#   julia scripts/diagnostics/inspect_emission_binary.jl <file.bin> --all
#
# Prints header metadata, time axis summary, and per-snapshot statistics.
# Designed to be as easy to use as `ncdump -h` for NetCDF files.
# =====================================================================

using JSON3
using Printf
using Dates
using Statistics

# ── Header reader ──────────────────────────────────────────────────────

const DEFAULT_HEADER_SIZE = 4096

function read_header(bin_path::String)
    io = open(bin_path, "r")
    initial = read(io, DEFAULT_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), initial), DEFAULT_HEADER_SIZE + 1) - 1
    initial_str = String(initial[1:json_end])

    # Check for larger header
    header_size = DEFAULT_HEADER_SIZE
    m = match(r"\"header_bytes\"\s*:\s*(\d+)", initial_str)
    if m !== nothing
        header_size = parse(Int, m[1])
    end

    seek(io, 0)
    full = read(io, header_size)
    json_end2 = something(findfirst(==(0x00), full), header_size + 1) - 1
    hdr = JSON3.read(String(full[1:json_end2]))
    close(io)
    return hdr, header_size
end

# ── Data reader ────────────────────────────────────────────────────────

function read_snapshot(bin_path::String, header_size::Int, Nc::Int, snapshot::Int;
                       float_type::Type{FT}=Float32) where FT
    panel_bytes = Nc * Nc * sizeof(FT)
    snapshot_bytes = 6 * panel_bytes
    offset = header_size + (snapshot - 1) * snapshot_bytes

    panels = ntuple(6) do p
        arr = Array{FT}(undef, Nc, Nc)
        open(bin_path, "r") do io
            seek(io, offset + (p - 1) * panel_bytes)
            read!(io, arr)
        end
        arr
    end
    return panels
end

# ── Statistics ─────────────────────────────────────────────────────────

function panel_stats(panels::NTuple{6})
    all_vals = vcat([vec(p) for p in panels]...)
    nonzero = filter(!iszero, all_vals)
    total = sum(all_vals)
    return (;
        min     = minimum(all_vals),
        max     = maximum(all_vals),
        mean    = mean(all_vals),
        median  = median(all_vals),
        total   = total,
        nonzero_frac = length(nonzero) / length(all_vals),
        n_cells = length(all_vals),
    )
end

# ── Formatting helpers ─────────────────────────────────────────────────

function format_time_hours(time_hours::Vector{Float64})
    Nt = length(time_hours)
    if Nt == 0
        return "  (empty)"
    elseif Nt == 1
        return "  [$(time_hours[1])] h  (static)"
    end

    dt_vals = diff(time_hours)
    median_dt = median(dt_vals)

    lines = String[]
    push!(lines, "  Range: $(time_hours[1]) → $(time_hours[end]) h  ($Nt snapshots)")

    # Detect cadence
    if all(x -> abs(x - median_dt) < 0.01, dt_vals)
        if abs(median_dt - 720.0) < 50
            push!(lines, "  Cadence: monthly (~$(round(median_dt, digits=1)) h)")
        elseif abs(median_dt - 24.0) < 1
            push!(lines, "  Cadence: daily ($(round(median_dt, digits=1)) h)")
        elseif abs(median_dt - 3.0) < 0.1
            push!(lines, "  Cadence: 3-hourly")
        else
            push!(lines, "  Cadence: $(round(median_dt, digits=2)) h (uniform)")
        end
    else
        push!(lines, "  Cadence: variable (median $(round(median_dt, digits=1)) h, " *
              "min $(round(minimum(dt_vals), digits=1)) h, max $(round(maximum(dt_vals), digits=1)) h)")
    end

    # Warn about suspicious values
    if Nt > 1 && maximum(abs, dt_vals) < 2.0
        push!(lines, "  ⚠ WARNING: time_hours look like sequential indices, not real hours!")
        push!(lines, "    Loader should recompute from calendar dates at load time.")
    end

    # Show first/last few
    if Nt <= 8
        push!(lines, "  Values: $(join([@sprintf("%.1f", h) for h in time_hours], ", "))")
    else
        first3 = join([@sprintf("%.1f", h) for h in time_hours[1:3]], ", ")
        last3 = join([@sprintf("%.1f", h) for h in time_hours[end-2:end]], ", ")
        push!(lines, "  Values: $(first3), ..., $(last3)")
    end

    return join(lines, "\n")
end

function fmt_sci(x::Real)
    if x == 0
        return "0"
    elseif abs(x) < 1e-3 || abs(x) > 1e6
        return @sprintf("%.4e", x)
    else
        return @sprintf("%.6f", x)
    end
end

# ── Main ───────────────────────────────────────────────────────────────

function main()
    if isempty(ARGS)
        println("""
        Usage: julia inspect_emission_binary.jl <file.bin> [options]

        Options:
          --snapshot N    Show detailed stats for snapshot N (1-based)
          --all           Show stats for ALL snapshots
          (default)       Show header + time axis + stats for first & last snapshot
        """)
        return
    end

    bin_path = expanduser(ARGS[1])
    if !isfile(bin_path)
        println("ERROR: File not found: $bin_path")
        return
    end

    # Parse options
    show_snapshot = nothing
    show_all = false
    for i in 2:length(ARGS)
        if ARGS[i] == "--snapshot" && i < length(ARGS)
            show_snapshot = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--all"
            show_all = true
        end
    end

    hdr, header_size = read_header(bin_path)
    FT = get(hdr, :float_type, "Float32") == "Float64" ? Float64 : Float32
    Nc = Int(hdr.Nc)
    Nt = Int(get(hdr, :Nt, 1))
    time_hours = haskey(hdr, :time_hours) ? Float64.(hdr.time_hours) : Float64[]

    file_size = filesize(bin_path)
    data_size = file_size - header_size
    expected_data = Nt * 6 * Nc * Nc * sizeof(FT)

    # ── Print header ──
    println("═"^70)
    println("  Emission Binary: $(basename(bin_path))")
    println("─"^70)
    println("  Species:      $(get(hdr, :species, "unknown"))")
    println("  Source:        $(get(hdr, :source, "unknown"))")
    println("  Grid:          Cubed-sphere C$(Nc)  ($(Nc)×$(Nc) × 6 panels)")
    println("  Float type:    $(FT)  ($(sizeof(FT)) bytes)")
    println("  Snapshots:     $(Nt)")
    println("  Units:         $(get(hdr, :units, "unknown"))")
    println("  Version:       $(get(hdr, :version, "?"))")
    println("  Header size:   $(header_size) bytes")
    println("  File size:     $(@sprintf("%.2f", file_size / 1e6)) MB")
    if data_size != expected_data
        println("  ⚠ Data size mismatch: expected $(expected_data) bytes, got $(data_size)")
    end
    println()

    # ── Time axis ──
    println("  Time Axis:")
    println(format_time_hours(time_hours))
    println()

    # ── Snapshot stats ──
    snapshots_to_show = if show_all
        1:Nt
    elseif show_snapshot !== nothing
        [show_snapshot]
    elseif Nt == 1
        [1]
    elseif Nt <= 4
        1:Nt
    else
        [1, Nt]
    end

    println("  Snapshot Statistics:")
    println("  " * "-"^66)
    @printf("  %-5s  %12s  %12s  %12s  %8s\n", "Snap", "Min", "Max", "Mean", "Nonzero%")
    println("  " * "-"^66)

    for s in snapshots_to_show
        if s < 1 || s > Nt
            println("  Snapshot $s out of range (1–$Nt)")
            continue
        end
        panels = read_snapshot(bin_path, header_size, Nc, s; float_type=FT)
        st = panel_stats(panels)
        th = s <= length(time_hours) ? @sprintf("%.0fh", time_hours[s]) : ""
        @printf("  %-5s  %12s  %12s  %12s  %7.1f%%\n",
                "$s $th", fmt_sci(st.min), fmt_sci(st.max), fmt_sci(st.mean),
                st.nonzero_frac * 100)
    end

    if !show_all && Nt > 4 && show_snapshot === nothing
        println("  ... ($(Nt - 2) snapshots omitted, use --all to show all)")
    end

    println("  " * "-"^66)
    println("═"^70)
end

main()
