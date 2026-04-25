#!/usr/bin/env julia
# ===========================================================================
# GEOS native NetCDF → AtmosTransport v4 transport binary preprocessor.
#
# Currently supports: GEOS-IT C180 native cubed-sphere → CS{Nc} passthrough.
# (GEOS-FP C720 native and GEOS → LL/RG cross-topology paths land in
# follow-up commits of plan indexed-baking-valiant.)
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/preprocess_geos_transport_binary.jl \
#       config/preprocessing/geosit_c180_to_cs180.toml \
#       --start 2021-12-01 --end 2021-12-03
#
# The config TOML declares: source descriptor (path to met_sources/*.toml +
# `root_dir` of on-disk NetCDF), target grid + vertical, output directory,
# and numerics. See docs/preprocessing/geos_pipeline.md for the format.
# ===========================================================================

using Logging
using TOML
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing

# ---------------------------------------------------------------------------

function _parse_date_range(args::Vector{String})
    start_str = nothing
    end_str   = nothing
    i = 2
    while i <= length(args)
        if args[i] == "--start" && i + 1 <= length(args)
            start_str = args[i + 1]; i += 2
        elseif args[i] == "--end"   && i + 1 <= length(args)
            end_str   = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    start_str !== nothing || error("Missing --start YYYY-MM-DD")
    end_str   !== nothing || error("Missing --end YYYY-MM-DD")
    return Date(start_str), Date(end_str)
end

function _build_settings(cfg::Dict)
    src_cfg  = cfg["source"]
    toml_rel = String(src_cfg["toml"])
    toml_abs = isabspath(toml_rel) ? toml_rel :
               joinpath(@__DIR__, "..", "..", toml_rel)
    return load_met_settings(toml_abs;
                             root_dir = expanduser(String(src_cfg["root_dir"])))
end

function _build_target_geometry_from_cfg(cfg::Dict, FT::Type)
    return build_target_geometry(cfg["grid"], FT)
end

function _build_vertical_setup(cfg::Dict)
    coeff_path = expanduser(String(cfg["vertical"]["coefficients"]))
    vc = load_hybrid_coefficients(coeff_path)
    Nz = length(vc.A) - 1
    return (merged_vc = vc, Nz_native = Nz, Nz = Nz)
end

function _output_path(cfg::Dict, date::Date, FT::Type)
    out_dir = expanduser(String(cfg["output"]["directory"]))
    mkpath(out_dir)
    datestr = Dates.format(date, "yyyymmdd")
    suffix  = FT === Float64 ? "float64" : "float32"
    return joinpath(out_dir, "geos_transport_$(datestr)_$(suffix).bin")
end

# ---------------------------------------------------------------------------

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
    global_logger(AtmosTransport.Preprocessing._FlushingLogger(base_logger))

    isempty(ARGS) && error(
        "Usage: julia --project=. scripts/preprocessing/preprocess_geos_transport_binary.jl " *
        "<config.toml> --start YYYY-MM-DD --end YYYY-MM-DD")

    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)

    start_date, end_date = _parse_date_range(ARGS)

    settings = _build_settings(cfg)
    FT = String(get(get(cfg, "numerics", Dict()), "float_type", "Float64")) == "Float32" ?
         Float32 : Float64
    grid     = _build_target_geometry_from_cfg(cfg, FT)
    vertical = _build_vertical_setup(cfg)

    out_cfg          = cfg["output"]
    panel_convention = String(get(out_cfg, "panel_convention", "geos_native"))
    mass_basis       = Symbol(get(out_cfg, "mass_basis", "dry"))
    dt_met_seconds   = Float64(get(get(cfg, "numerics", Dict()),
                                   "dt_met_seconds", 3600.0))

    @info "GEOS preprocessor: $(start_date) to $(end_date)"
    @info "  source = $(typeof(settings))  Nc=$(settings.Nc)  mass_flux_dt=$(settings.mass_flux_dt)s"
    @info "  target = $(typeof(grid))  Nc=$(grid.Nc)  Nz=$(vertical.Nz)  FT=$(FT)"

    t_total = time()
    n = 0
    for d in start_date:Day(1):end_date
        n += 1
        out_path = _output_path(cfg, d, FT)
        @info "[$n] $(d) → $(out_path)"
        process_day(d, grid, settings, vertical;
                    out_path = out_path,
                    dt_met_seconds = dt_met_seconds,
                    FT = FT,
                    mass_basis = mass_basis,
                    panel_convention = panel_convention)
    end
    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)",
                   n, elapsed, elapsed / n)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
