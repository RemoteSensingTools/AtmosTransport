#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# scripts/run_transport.jl — unified driven transport runner (plan 40 Commit 6c).
#
# ONE CLI. Reads a TOML config, opens the first transport binary,
# dispatches on its `grid_type` header field (`:latlon`,
# `:reduced_gaussian`, `:cubed_sphere`), and runs the loop. All
# topology-specific logic (IC pipeline, surface flux, snapshot output,
# GPU residency check, capability validation) lives in
# `src/Models/DrivenRunner.jl` and dispatches on mesh type via
# multiple dispatch.
#
# The TOML `[input]` block accepts either shape:
#   [input]
#   binary_paths = [ "a.bin", "b.bin" ]
# OR
#   [input]
#   folder       = "~/data/.../"
#   start_date   = "YYYY-MM-DD"
#   end_date     = "YYYY-MM-DD"
#   file_pattern = "<prefix>{YYYYMMDD}<suffix>"   # optional
#
# Usage:
#   julia --project=. scripts/run_transport.jl <config.toml>
# ---------------------------------------------------------------------------

using Logging
using TOML

# Preload the GPU backend BEFORE `AtmosTransport` gets included so the
# whole stack compiles in a single world age. Doing the load dynamically
# later (from `_ensure_gpu_runtime!`) means every CuArray method
# (`size`, `getindex`, `Adapt.adapt_storage(CuArray, …)`) arrives in a
# newer world than the function bodies that call it, and Julia refuses
# to dispatch — `method too new to be called from this world context`.
# Inspecting the config here is ~1 ms and avoids the whole problem.
if !isempty(ARGS)
    _cfg_path = expanduser(ARGS[1])
    if isfile(_cfg_path)
        _cfg = try TOML.parsefile(_cfg_path) catch; nothing end
        if _cfg !== nothing
            _use_gpu = Bool(get(get(_cfg, "architecture", Dict{String, Any}()),
                                "use_gpu", false))
            if _use_gpu
                if Sys.isapple()
                    @info "Preloading Metal (use_gpu = true on macOS)"
                    using Metal
                else
                    @info "Preloading CUDA (use_gpu = true)"
                    using CUDA
                end
            end
        end
    end
end

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/run_transport.jl <config.toml>")
    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)
    return run_driven_simulation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
