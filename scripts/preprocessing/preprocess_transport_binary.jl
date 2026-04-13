#!/usr/bin/env julia
# ===========================================================================
# Unified transport binary preprocessor — THE entry point.
#
# Generates Poisson-balanced transport binaries from any supported met source
# onto any supported target grid. Config-driven via TOML.
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
#       config.toml --day 2021-12-01
#
# The TOML config specifies source, target grid, level selection, and output.
# See docs/reference/PREPROCESSING.md for the config format.
# ===========================================================================

using Logging
using TOML
using Dates

# Load the AtmosTransport module (includes Preprocessing)
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(base_logger)

    isempty(ARGS) && error("Usage: julia --project=. scripts/preprocessing/preprocess_transport_binary.jl config.toml [--day YYYY-MM-DD]")

    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)

    # Parse --day argument
    day_str = nothing
    for i in 2:length(ARGS)
        if ARGS[i] == "--day" && i + 1 <= length(ARGS)
            day_str = ARGS[i + 1]
        end
    end

    # Delegate to the Preprocessing module's process_day function
    # (which dispatches on source × target × level selection)
    process_day(cfg; day_override=day_str)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
