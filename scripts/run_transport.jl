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
            _arch = get(_cfg, "architecture", Dict{String, Any}())
            _use_gpu = Bool(get(_arch, "use_gpu", false))
            _backend = lowercase(String(get(_arch, "backend",
                                            _use_gpu ? "auto" : "cpu")))
            _backend = replace(_backend, '-' => '_', ' ' => '_')
            _gpu_requested = _use_gpu ||
                             _backend in ("auto", "gpu", "cuda", "nvidia",
                                          "metal", "apple", "apple_metal")
            if _gpu_requested
                if _backend in ("metal", "apple", "apple_metal") ||
                   (_backend in ("auto", "gpu") && Sys.isapple())
                    @info "Preloading Metal (GPU backend)"
                    using Metal
                elseif _backend in ("cuda", "nvidia") ||
                       _backend in ("auto", "gpu")
                    @info "Preloading CUDA (GPU backend)"
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
