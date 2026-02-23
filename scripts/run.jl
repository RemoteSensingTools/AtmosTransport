#!/usr/bin/env julia
# ===========================================================================
# Universal model runner — accepts any TOML configuration file.
#
# Usage:
#   julia --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml
#   julia --project=. scripts/run.jl config/runs/era5_edgar.toml
#   julia --project=. scripts/run.jl config/runs/era5_preprocessed.toml
#
# For double-buffered I/O overlap, start Julia with multiple threads:
#   julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml
#
# The TOML file fully specifies architecture, grid, met data, tracers,
# output, and buffering strategy. See config/runs/ for examples.
# ===========================================================================

if isempty(ARGS)
    println(stderr, "Usage: julia --project=. scripts/run.jl <config.toml>\n")
    println(stderr, "Available configurations:")
    config_dir = joinpath(@__DIR__, "..", "config", "runs")
    if isdir(config_dir)
        for f in sort(readdir(config_dir))
            endswith(f, ".toml") && println(stderr, "  config/runs/$f")
        end
    end
    exit(1)
end

import TOML
config = TOML.parsefile(ARGS[1])
@info "Configuration: $(ARGS[1])"

# CRITICAL: Load CUDA before AtmosTransportModel to trigger the weak-dependency
# extension AtmosTransportModelCUDAExt. This cannot be deferred.
if get(get(config, "architecture", Dict()), "use_gpu", false)
    using CUDA
    CUDA.allowscalar(false)
    @info "GPU mode enabled (CUDA loaded)"
else
    @info "CPU mode"
end

using AtmosTransportModel
using AtmosTransportModel.IO: build_model_from_config
import AtmosTransportModel.Models: run!

@info "Building model..."
model = build_model_from_config(config)

@info "Starting simulation..."
run!(model)
