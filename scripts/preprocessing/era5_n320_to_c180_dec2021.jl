#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Drive `process_era5_n320_to_cs_day` for a date range from the CLI.
#
# Usage:
#   julia --project=. scripts/preprocessing/era5_n320_to_c180_dec2021.jl \
#       --start 2021-12-01 --end 2021-12-03 \
#       --nc 180 --steps-per-window 8 \
#       --out-dir ~/data/AtmosTransport/met/era5/n320_to_c180/transport_binary_dec2021_catrine_f32
#
# Default flags target the Dec 2021 Catrine validation: C180, Float32,
# Nz=137 (no vertical merge yet), 8 substeps per met window, no convection
# section (UDMF/DDMF → CMFMC conversion is a follow-on).
# ---------------------------------------------------------------------------

using Dates

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Preprocessing: ERA5N320Settings,
                                      process_era5_n320_to_cs_day,
                                      build_target_geometry

function _parse_cli(args::Vector{String})
    start = Date(2021, 12, 1)
    stop  = Date(2021, 12, 3)
    Nc = 180
    steps_per_window = 8
    Nz = 137
    out_dir = expanduser("~/data/AtmosTransport/met/era5/n320_to_c180/transport_binary_dec2021_catrine_f32")
    era5_root = expanduser("~/data/AtmosTransport/met/era5/N320/hourly/raw")
    cache_dir = expanduser("~/.cache/AtmosTransport/cr_regridding")
    include_convection = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--start" && i + 1 <= length(args)
            start = Date(args[i + 1]); i += 2
        elseif a == "--end" && i + 1 <= length(args)
            stop = Date(args[i + 1]); i += 2
        elseif a == "--nc" && i + 1 <= length(args)
            Nc = parse(Int, args[i + 1]); i += 2
        elseif a == "--steps-per-window" && i + 1 <= length(args)
            steps_per_window = parse(Int, args[i + 1]); i += 2
        elseif a == "--out-dir" && i + 1 <= length(args)
            out_dir = expanduser(args[i + 1]); i += 2
        elseif a == "--era5-root" && i + 1 <= length(args)
            era5_root = expanduser(args[i + 1]); i += 2
        elseif a == "--cache-dir" && i + 1 <= length(args)
            cache_dir = expanduser(args[i + 1]); i += 2
        elseif a == "--include-convection"
            include_convection = true; i += 1
        else
            i += 1
        end
    end
    return (; start, stop, Nc, steps_per_window, Nz, out_dir, era5_root,
              cache_dir, include_convection)
end

function main(args::Vector{String} = ARGS)
    cli = _parse_cli(args)
    @info "ERA5 N320 → CS driver" cli.start cli.stop cli.Nc cli.steps_per_window cli.out_dir cli.include_convection

    settings = ERA5N320Settings(; root_dir = cli.era5_root,
                                  include_convection = cli.include_convection)
    cfg = Dict{String, Any}(
        "type"             => "cubed_sphere",
        "Nc"               => cli.Nc,
        "panel_convention" => "geos_native",
        "definition"       => "gmao",
    )
    target_grid = build_target_geometry(Val(:cubed_sphere), cfg, Float32)
    mkpath(cli.out_dir)

    for date in cli.start:Day(1):cli.stop
        out_path = joinpath(cli.out_dir,
            "era5_n320_to_c$(cli.Nc)_transport_$(Dates.format(date, "yyyymmdd"))_float32.bin")
        @info "==>" date out_path
        process_era5_n320_to_cs_day(date, settings, target_grid;
            out_path = out_path,
            Nz = cli.Nz,
            steps_per_window = cli.steps_per_window,
            cache_dir = cli.cache_dir,
            include_convection = cli.include_convection)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
