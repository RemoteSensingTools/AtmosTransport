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
