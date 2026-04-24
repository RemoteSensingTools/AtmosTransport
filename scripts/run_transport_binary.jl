#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# DEPRECATION SHIM (plan 40 Commit 6c).
#
# `scripts/run_transport_binary.jl` is the old LL/RG-specific CLI name.
# The canonical entry point is now `scripts/run_transport.jl`, which
# dispatches on `inspect_binary(first_path).grid_type` and handles every
# topology. This shim forwards to the unified script for one migration
# cycle; please update your invocations:
#
#   julia --project=. scripts/run_transport.jl <config.toml>
#
# The shim will be removed in a follow-up plan.
# ---------------------------------------------------------------------------

using Logging
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    @warn "scripts/run_transport_binary.jl is a deprecation shim; use " *
          "scripts/run_transport.jl (plan 40 Commit 6c). Forwarding."
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/run_transport_binary.jl <config.toml>")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    return run_driven_simulation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
