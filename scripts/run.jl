#!/usr/bin/env julia
# ===========================================================================
# Universal model runner — accepts any TOML configuration file.
#
# Usage:
#   julia --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml
#   julia --project=. scripts/run.jl config/runs/era5_edgar.toml
#   julia --project=. scripts/run.jl config/runs/era5_preprocessed.toml
#
# Flags:
#   -v, --verbose   Enable debug-level logging (detailed per-window diagnostics)
#   -c, --check     Run preflight checks before simulation (verify data availability)
#
# For double-buffered I/O overlap, start Julia with multiple threads:
#   julia --threads=2 --project=. scripts/run.jl config/runs/geosfp_cs_edgar.toml
#
# The TOML file fully specifies architecture, grid, met data, tracers,
# output, and buffering strategy. See config/runs/ for examples.
# ===========================================================================

using Logging

# --- Parse flags and positional arguments ---
const FLAGS = filter(a -> startswith(a, "-"), ARGS)
const POSITIONAL = filter(a -> !startswith(a, "-"), ARGS)
const VERBOSE = any(a -> a in ("--verbose", "-v"), FLAGS)
const RUN_CHECKS = any(a -> a in ("--check", "-c"), FLAGS)

# Set verbose mode via environment variable (checked by AtmosTransport code).
# We keep the global logger at Info to avoid Julia internal debug spam.
if VERBOSE
    ENV["ATMOSTR_VERBOSE"] = "1"
    @info "Verbose mode enabled"
end

if isempty(POSITIONAL)
    println(stderr, "Usage: julia --project=. scripts/run.jl [flags] <config.toml>")
    println(stderr, "\nFlags:")
    println(stderr, "  -v, --verbose   Enable debug-level logging")
    println(stderr, "  -c, --check     Run preflight checks before simulation")
    println(stderr, "\nAvailable configurations:")
    config_dir = joinpath(@__DIR__, "..", "config", "runs")
    if isdir(config_dir)
        for f in sort(readdir(config_dir))
            endswith(f, ".toml") && println(stderr, "  config/runs/$f")
        end
    end
    exit(1)
end

import TOML
config = TOML.parsefile(POSITIONAL[1])
@info "Configuration: $(POSITIONAL[1])"
flush(stderr)

# CRITICAL: Load GPU package before AtmosTransport to trigger the weak-dependency
# extension (AtmosTransportCUDAExt or AtmosTransportMetalExt). This cannot be deferred.
if get(get(config, "architecture", Dict()), "use_gpu", false)
    if Sys.isapple()
        using Metal
        @info "GPU mode enabled (Metal loaded)"
    else
        using CUDA
        CUDA.allowscalar(false)
        @info "GPU mode enabled (CUDA loaded)"
    end
else
    @info "CPU mode"
end
flush(stderr)

using AtmosTransport
using AtmosTransport.IO: build_model_from_config
import AtmosTransport.Models: run!, preflight_check!
@info "Packages loaded"
flush(stderr)

@info "Building model..."
flush(stderr)
model = build_model_from_config(config);

if RUN_CHECKS
    @info "Running preflight checks..."
    preflight_check!(model; verbose=VERBOSE)
end

@info "Starting simulation..."
flush(stderr)
run!(model)
