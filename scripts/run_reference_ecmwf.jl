#!/usr/bin/env julia
# ===========================================================================
# Reproducible reference run (ECMWF/ERA5) — same config as docs/REFERENCE_RUN.md
#
# Writes to data/era5/output/reference_era5_output.nc for validation and
# TM5 comparison. Delegates to run_forward_era5.jl with REFERENCE_RUN=1.
#
# Usage:
#   julia --project=. scripts/run_reference_ecmwf.jl
# ===========================================================================

ENV["REFERENCE_RUN"] = "true"
include(joinpath(@__DIR__, "run_forward_era5.jl"))
