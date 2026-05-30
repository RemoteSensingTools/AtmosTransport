#!/usr/bin/env julia

using CairoMakie
using AtmosTransport
using AtmosTransport.Visualization

function _arg(flag, default)
    idx = findfirst(==(flag), ARGS)
    idx === nothing && return default
    idx == length(ARGS) && error("missing value after $flag")
    return ARGS[idx + 1]
end

function _arg_int(flag, default)
    return parse(Int, _arg(flag, string(default)))
end

function _arg_float(flag, default)
    return parse(Float64, _arg(flag, string(default)))
end

function _arg_lats(flag, default)
    raw = _arg(flag, default)
    return Tuple(parse.(Float64, split(raw, ",")))
end

function main()
    at = expanduser(_arg("--at",
        "~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d.nc"))
    gc = expanduser(_arg("--gc",
        "~/data/AtmosTransport/catrine-geoschem-runs"))
    out_dir = expanduser(_arg("--out-dir",
        "~/data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d_animation"))
    species = Symbol(_arg("--species", "co2_fossil"))
    fps = _arg_int("--fps", 3)
    max_frames = _arg_int("--max-frames", 0)
    map_vmax = _arg_float("--map-vmax", 8.0)
    map_vmin = _arg_float("--map-vmin", 0.0)
    curtain_vmax = _arg_float("--curtain-vmax", 40.0)
    curtain_vmin = _arg_float("--curtain-vmin", 0.0)
    scale = Symbol(_arg("--scale", "symlog"))
    # Pass "--auto-range-day1 2,98" to override the four vmin/vmax knobs with
    # the (low,high) percentile pair computed across day-1 AT+GC data —
    # column range derived from column means, curtain range from full
    # curtain values. Both AT and GC panels share the resulting range.
    auto_range_arg = _arg("--auto-range-day1", "")
    auto_range_day1 = if isempty(auto_range_arg)
        nothing
    else
        parts = split(auto_range_arg, ",")
        length(parts) == 2 ||
            error("--auto-range-day1 expects \"low,high\" percentile pair, got $auto_range_arg")
        (parse(Float64, parts[1]), parse(Float64, parts[2]))
    end
    latitudes = _arg_lats("--latitudes", "40,0,-40")
    dlon = _arg_float("--dlon", 2.0)
    dp = _arg_float("--dp", 10.0)
    at_log_arg = _arg("--at-log", "")
    at_log = isempty(at_log_arg) ? nothing : expanduser(at_log_arg)

    result = catrine_map_curtains(at, gc;
        species,
        out_dir,
        fps,
        max_frames,
        map_vmax,
        map_vmin,
        curtain_vmax,
        curtain_vmin,
        scale,
        auto_range_day1,
        latitudes,
        dlon,
        dp,
        at_log,
        write_animation=true)

    println("Saved first frame: ", result.png)
    println("Saved animation: ", result.gif)
    println("Frames: ", result.frames)
end

main()
