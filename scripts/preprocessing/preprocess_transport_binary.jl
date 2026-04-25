#!/usr/bin/env julia
# ===========================================================================
# Unified transport-binary preprocessor — THE entry point.
#
# Generates v4 transport binaries from any supported met source onto any
# supported target grid. Config-driven via TOML; the canonical entrypoint
# detects native sources via `[source].toml` and routes through the
# `AbstractMetSettings` factory, otherwise falls back to the ERA5 spectral
# path. New sources plug in through `AbstractMetSettings` + a
# `config/met_sources/<source>.toml` descriptor — never via a parallel CLI.
#
# Usage:
#
#   # Single day
#   julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
#       <config.toml> --day 2021-12-01
#
#   # Date range (native sources only)
#   julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
#       <config.toml> --start 2021-12-01 --end 2021-12-03
#
# Configs:
# - ERA5 spectral:   `[input].spectral_dir = "..."` — legacy NamedTuple path.
# - Native sources:  `[source].toml = "config/met_sources/<src>.toml"` plus
#                    `[source].root_dir = "..."` — typed dispatch.
# ===========================================================================

using Logging
using TOML
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing

function _parse_cli(args::Vector{String})
    isempty(args) && error("Usage: preprocess_transport_binary.jl <config.toml> " *
                            "[--day YYYY-MM-DD | --start YYYY-MM-DD --end YYYY-MM-DD]")
    cfg_path = expanduser(args[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")

    day_override = nothing
    start_date   = nothing
    end_date     = nothing
    i = 2
    while i <= length(args)
        if     args[i] == "--day"   && i + 1 <= length(args); day_override = args[i + 1]; i += 2
        elseif args[i] == "--start" && i + 1 <= length(args); start_date   = args[i + 1]; i += 2
        elseif args[i] == "--end"   && i + 1 <= length(args); end_date     = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    return cfg_path, day_override, start_date, end_date
end

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
    global_logger(AtmosTransport.Preprocessing._FlushingLogger(base_logger))

    cfg_path, day_override, start_date, end_date = _parse_cli(ARGS)
    cfg = TOML.parsefile(cfg_path)

    process_day(cfg; day_override = day_override,
                     start_date   = start_date,
                     end_date     = end_date)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
