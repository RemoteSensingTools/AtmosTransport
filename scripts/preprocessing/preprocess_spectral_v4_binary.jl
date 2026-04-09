#!/usr/bin/env julia
# ===========================================================================
# Fused ERA5 spectral GRIB -> daily v4 binary preprocessor
#
# Goes directly from spectral VO/D/LNSP GRIB files to merged-level flat
# binary files, skipping the intermediate NetCDF step.
#
# Pipeline per day:
#   1. Read all hourly GRIB spectral data (VO, D, LNSP)
#   2. For each hour t (0-23):
#      a. Spectral -> gridpoint: VO/D -> U/V via SHT, compute am/bm/cm/m
#         at native 137 levels
#      b. Merge to target levels (merge_thin_levels with configurable min_dp)
#      c. Recompute cm from merged am/bm via continuity with B-correction
#      d. Compute deltas: dam = am[t+1] - am[t], dbm = bm[t+1] - bm[t],
#         dm = m[t+1] - m[t]
#         For the last hour of the day: use next day's hour 0 if available,
#         else zeros.
#      e. Write to daily v4 binary: m|am|bm|cm|ps|dam|dbm|dm per window
#   3. All spectral transforms done in Float64, stored as Float32.
#
# Produces ONLY core transport fields (m, am, bm, cm, ps) plus v4 deltas
# (dam, dbm, dm). No convection, surface, temperature, or QV.
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
#       config/preprocessing/era5_spectral_v4.toml [--day 2021-12-01]
# ===========================================================================

using GRIB
using FFTW
using LinearAlgebra: mul!
using JSON3
using Dates
using Printf
using TOML
using NCDatasets
using Logging

const PREPROCESS_SPECTRAL_V4_SCRIPT = @__FILE__
const PREPROCESS_SPECTRAL_V4_DIR = joinpath(@__DIR__, "preprocess_spectral_v4_binary")
const PREPROCESS_SPECTRAL_V4_REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

include(joinpath(PREPROCESS_SPECTRAL_V4_REPO_ROOT, "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2: LatLonMesh, ReducedGaussianMesh, cell_areas_by_latitude,
    nrings, ring_longitudes, read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh

include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "logging.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "constants.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "vertical_coordinates.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "mass_support.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "target_geometry.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "spectral_io.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "configuration.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "spectral_synthesis.jl"))
include(joinpath(PREPROCESS_SPECTRAL_V4_DIR, "binary_pipeline.jl"))

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited=false)
    global_logger(_FlushingLogger(base_logger))
    println(stderr, "[v4] Logger installed, starting…")
    flush(stderr)
    atexit(() -> (try flush(stderr) catch; end; try flush(stdout) catch; end))

    if isempty(ARGS)
        println("""
        Fused ERA5 spectral GRIB -> v4 binary preprocessor

        Goes directly from spectral GRIB to merged-level daily binary,
        skipping the intermediate NetCDF step.

        Usage:
          julia -t8 --project=. $(PROGRAM_FILE) config.toml [--day 2021-12-01]

        Produces ONLY core transport fields (m, am, bm, cm, ps) plus v4
        flux deltas (dam, dbm, dm).
        """)
        return
    end

    config_path = expanduser(ARGS[1])
    isfile(config_path) || error("Config not found: $config_path")
    cfg = TOML.parsefile(config_path)

    day_filter = parse_day_filter(ARGS)
    grid = build_target_geometry(cfg["grid"], Float64)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range, settings.min_dp, cfg["grid"])

    log_preprocessor_configuration(settings, grid, vertical)
    ensure_supported_target(grid)

    dates = select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))

    mkpath(settings.out_dir)
    t_total = time()

    for (i, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", i, length(dates), date)

        next_day_h0 = next_day_hour0(date, dates, settings.spectral_dir, settings.T_target)
        next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")

        result = process_day(date, grid, settings, vertical; next_day_hour0=next_day_h0)
        result === nothing && continue
    end

    wall_total = round(time() - t_total, digits=1)
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), wall_total,
                   length(dates) > 0 ? wall_total / length(dates) : 0.0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
