#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Cubed-sphere driven transport runner — CLI wrapper.
#
# Plan 40 Commit 6b hoisted the implementation into
# `src/Models/DrivenRunner.jl`. `run_driven_simulation(cfg)` dispatches
# on `inspect_binary(first_path).grid_type`, so this script stays as a
# thin CLI for backward compatibility with in-tree configs (the
# `catrine_c48_10d/*.toml` header comments still invoke it by name).
# Commit 6c introduces the unified `scripts/run_transport.jl`.
#
# For low-level advection-only CS benchmarks, see `run_cs_transport.jl`
# — that path is untouched.
#
# Usage:
#   julia --project=. scripts/run_cs_driven.jl <config.toml>
#
# The TOML `[input]` block accepts either explicit `binary_paths = [...]`
# or `folder + start_date + end_date (+ file_pattern)` (plan 40 Commit 4).
# ---------------------------------------------------------------------------

using Logging
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/run_cs_driven.jl <config.toml>")
    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)
    return run_driven_simulation(cfg)
end

# Guarded so the script can be `include`d without auto-running
# (test_cs_driven_builders.jl does this to get AtmosTransport in scope).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
