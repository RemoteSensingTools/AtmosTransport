#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# DEPRECATION SHIM (plan 40 Commit 6c).
#
# `scripts/run_cs_driven.jl` is the old CS-specific CLI name. The
# canonical entry point is now `scripts/run_transport.jl`, which
# dispatches on `inspect_binary(first_path).grid_type` and handles every
# topology via a single library function. This shim forwards for one
# migration cycle; please update your invocations:
#
#   julia --project=. scripts/run_transport.jl <config.toml>
#
# For the low-level, advection-only CS benchmark, `run_cs_transport.jl`
# is untouched and remains a separate entry point.
#
# The shim will be removed in a follow-up plan.
# ---------------------------------------------------------------------------

using Logging
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    @warn "scripts/run_cs_driven.jl is a deprecation shim; use " *
          "scripts/run_transport.jl (plan 40 Commit 6c). Forwarding."
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/run_cs_driven.jl <config.toml>")
    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)
    return run_driven_simulation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
