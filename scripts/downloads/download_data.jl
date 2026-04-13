#!/usr/bin/env julia
# ===========================================================================
# Unified download entry point — THE download script.
#
# Downloads meteorological data from any supported source using TOML configs.
# Mirrors the preprocessing CLI pattern (preprocess_transport_binary.jl).
#
# Usage:
#   julia --project=. scripts/downloads/download_data.jl config.toml \
#       [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]
#
# The TOML config specifies source, protocol, output layout, and schedule.
# Output paths follow the canonical Data Layout hierarchy:
#   <data_root>/met/<source>/<grid>/<cadence>/<payload>/
#
# See docs/reference/DATA_LAYOUT.md for the layout convention.
# ===========================================================================

using Logging
using TOML
using Dates

# Load the Downloads module (standalone — does not need the full AtmosTransport)
include(joinpath(@__DIR__, "..", "..", "src", "Downloads", "Downloads.jl"))
using .Downloads

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)

    isempty(ARGS) && error("""
        Usage: julia --project=. scripts/downloads/download_data.jl config.toml \\
               [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]

        Download recipe TOMLs are in config/downloads/.
        Met source definitions are in config/met_sources/.
        """)

    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)

    # Parse CLI flags
    start_date = _parse_flag(ARGS, "--start", Date)
    end_date   = _parse_flag(ARGS, "--end", Date)
    dry_run    = "--dry-run" in ARGS
    verify     = "--verify" in ARGS

    download_data!(cfg; start_date, end_date, dry_run, verify_only=verify)
end

function _parse_flag(args, flag, T)
    idx = findfirst(==(flag), args)
    idx === nothing && return nothing
    idx == length(args) && error("Missing value after $flag")
    return T(args[idx + 1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
