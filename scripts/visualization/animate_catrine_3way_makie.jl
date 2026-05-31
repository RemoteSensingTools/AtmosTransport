#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Three-row CATRINE comparison: GeosChem | AT (GEOS-IT met) | AT (ERA5 met).
# Same column-map + longitude-pressure-curtain style as
# animate_catrine_map_curtains_makie.jl, with a third data row so the two AT
# transport solutions can be compared against each other and the GeosChem
# reference in one figure.
#
# Usage:
#   julia --project=. scripts/visualization/animate_catrine_3way_makie.jl \
#       --at-geos <geos_run.nc> --at-era5 <era5_run.nc> --gc <geoschem_dir> \
#       --species co2_natural --scale linear --auto-range-day1 2,98 \
#       --out-dir <dir>
#
# Pick scale/range per species: fossil is a 0-anchored enhancement
# (--scale symlog), natural is absolute CO2 around ~410 ppm (--scale linear,
# --auto-range-day1 2,98 derives the window from the data).
# ---------------------------------------------------------------------------

using CairoMakie
using AtmosTransport
using AtmosTransport.Visualization: catrine_map_curtains_3way

function _arg(flag, default)
    idx = findfirst(==(flag), ARGS)
    idx === nothing && return default
    idx == length(ARGS) && error("missing value after $flag")
    return ARGS[idx + 1]
end
_arg_int(flag, default) = parse(Int, _arg(flag, string(default)))
_arg_float(flag, default) = parse(Float64, _arg(flag, string(default)))

function main()
    at_geos = expanduser(_arg("--at-geos", ""))
    at_era5 = expanduser(_arg("--at-era5", ""))
    gc      = expanduser(_arg("--gc", "~/data/AtmosTransport/catrine-geoschem-runs"))
    isempty(at_geos) && error("--at-geos <geos_run.nc> is required")
    isempty(at_era5) && error("--at-era5 <era5_run.nc> is required")
    out_dir = expanduser(_arg("--out-dir",
        "~/data/AtmosTransport/output/catrine_3way_animation"))
    species = Symbol(_arg("--species", "co2_fossil"))
    scale   = Symbol(_arg("--scale", "symlog"))
    fps     = _arg_int("--fps", 3)
    max_frames = _arg_int("--max-frames", 0)

    auto_arg = _arg("--auto-range-day1", "")
    auto_range_day1 = if isempty(auto_arg)
        nothing
    else
        parts = split(auto_arg, ",")
        length(parts) == 2 || error("--auto-range-day1 expects \"low,high\", got $auto_arg")
        (parse(Float64, parts[1]), parse(Float64, parts[2]))
    end

    result = catrine_map_curtains_3way(at_geos, at_era5, gc;
        species, out_dir, fps, max_frames, scale, auto_range_day1,
        map_vmax = _arg_float("--map-vmax", 8.0),
        map_vmin = _arg_float("--map-vmin", 0.0),
        curtain_vmax = _arg_float("--curtain-vmax", 40.0),
        curtain_vmin = _arg_float("--curtain-vmin", 0.0))

    @info "3-way animation written" gif=result.gif png=result.png frames=result.frames
    return nothing
end

main()
