#!/usr/bin/env julia
# Compatibility shim for the old low-level cubed-sphere advection runner.
#
# Production transport runs must use `scripts/run_transport.jl`, which
# dispatches from the transport-binary grid type. The panel-native CS runner is
# retained only as a benchmark/debug harness under `scripts/benchmarks/`.

@warn "scripts/run_cs_transport.jl is benchmark-only and deprecated from the " *
      "production runtime matrix; use scripts/run_transport.jl for runs, or " *
      "scripts/benchmarks/run_cs_transport.jl for the low-level CS benchmark."

include(joinpath(@__DIR__, "..", "benchmarks", "run_cs_transport.jl"))
