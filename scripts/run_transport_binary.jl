#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# LL / RG driven transport runner — CLI wrapper.
#
# Plan 40 Commit 6a hoisted the implementation into
# `src/Models/DrivenRunner.jl`. This script stays as the canonical LL/RG
# entry point until Commit 6c introduces the unified
# `scripts/run_transport.jl` and turns this into a deprecation shim.
#
# Usage:
#   julia --project=. scripts/run_transport_binary.jl <config.toml>
#
# The TOML `[input]` block accepts either shape:
#   [input]
#   binary_paths = [ "a.bin", "b.bin" ]        # explicit list
# OR
#   [input]
#   folder       = "~/data/.../"
#   start_date   = "YYYY-MM-DD"
#   end_date     = "YYYY-MM-DD"
#   file_pattern = "<prefix>{YYYYMMDD}<suffix>"   # optional
# ---------------------------------------------------------------------------

using Logging
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/run_transport_binary.jl <config.toml>")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    return run_driven_simulation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
