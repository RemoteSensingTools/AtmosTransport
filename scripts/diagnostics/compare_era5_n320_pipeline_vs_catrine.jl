#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Compare the ERA5 N320 → C180 preprocessing pipeline (breakpoints B-F)
# against the Catrine GEOS-Chem reference snapshot for one (date, hour).
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_era5_n320_pipeline_vs_catrine.jl \
#       --date 2021-12-01 --hour 3 --nc 180
#
# Optional flags:
#   --era5-root <path>   default: ~/data/AtmosTransport/met/era5/N320/hourly/raw
#   --catrine <path>     default: ~/data/AtmosTransport/catrine-geoschem-runs
#   --float32            run the pipeline in Float32 (default Float64)
#
# Reports per-panel global statistics for PS (Pa) and 850 hPa T (K), comparing
# the regridded ERA5 fields against the Catrine snapshot at the same C-grid
# resolution. Both meteorologies are derived from ERA5-family analyses
# (Catrine drives GEOS-Chem with GEOS-IT-like met; this pipeline reads native
# ERA5 GRIB and synthesises to C180 directly), so we expect O(1 hPa) PS
# agreement and O(1 K) temperature agreement on average. Larger residuals
# point at convention mismatches in the regrid or the vertical merge.
# ---------------------------------------------------------------------------

using Dates
using NCDatasets
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Preprocessing: ERA5N320Settings,
                                      allocate_era5_n320_to_c180_pipeline,
                                      process_era5_n320_window!,
                                      open_era5_day, close_era5_day!,
                                      build_target_geometry

function _parse_cli(args::Vector{String})
    date = Date(2021, 12, 1)
    # Default to hour 6 — first 3-hourly Catrine snapshot of the day covered
    # by today's 06 UTC convection forecast (hours 0..5 need the previous
    # day's file, which may not be on disk at archive start).
    hour = 6
    Nc = 180
    FT = Float64
    era5_root = expanduser("~/data/AtmosTransport/met/era5/N320/hourly/raw")
    catrine = expanduser("~/data/AtmosTransport/catrine-geoschem-runs")

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--date" && i + 1 <= length(args)
            date = Date(args[i + 1]); i += 2
        elseif a == "--hour" && i + 1 <= length(args)
            hour = parse(Int, args[i + 1]); i += 2
        elseif a == "--nc" && i + 1 <= length(args)
            Nc = parse(Int, args[i + 1]); i += 2
        elseif a == "--era5-root" && i + 1 <= length(args)
            era5_root = expanduser(args[i + 1]); i += 2
        elseif a == "--catrine" && i + 1 <= length(args)
            catrine = expanduser(args[i + 1]); i += 2
        elseif a == "--float32"
            FT = Float32; i += 1
        else
            i += 1
        end
    end
    return (; date, hour, Nc, FT, era5_root, catrine)
end

function _catrine_path(catrine_root::String, date::Date, hour::Int)
    stem = Dates.format(date, "yyyymmdd")
    hh   = lpad(hour, 2, '0')
    return joinpath(catrine_root, "GEOSChem.CATRINE_inst.$(stem)_$(hh)00z.nc4")
end

function _summarize_panel(label::String, pipeline_panel::AbstractMatrix,
                            catrine_panel::AbstractMatrix)
    diff = Float64.(pipeline_panel) .- Float64.(catrine_panel)
    @printf "  %s panel diff: mean=%9.3f rms=%9.3f min=%9.3f max=%9.3f\n" label mean(diff) sqrt(mean(diff.^2)) minimum(diff) maximum(diff)
end

function main(args::Vector{String} = ARGS)
    cli = _parse_cli(args)
    @info "Comparison" cli.date cli.hour cli.Nc cli.FT cli.era5_root cli.catrine

    cli.Nc == 180 ||
        error("Catrine snapshots are C180 — comparison requires `--nc 180`. " *
              "For a faster scalar smoke without Catrine, run " *
              "`test/test_era5_n320_to_c180_pipeline.jl` with smaller Nc.")

    catrine_file = _catrine_path(cli.catrine, cli.date, cli.hour)
    isfile(catrine_file) ||
        error("Catrine snapshot not found: $catrine_file")

    settings = ERA5N320Settings(; root_dir = cli.era5_root,
                                  include_convection = true)
    handles  = open_era5_day(settings, cli.date)
    try
        cfg = Dict{String, Any}(
            "type"             => "cubed_sphere",
            "Nc"               => cli.Nc,
            "panel_convention" => "geos_native",
            "definition"       => "gmao",
        )
        target_grid = build_target_geometry(Val(:cubed_sphere), cfg, cli.FT)

        pipeline = allocate_era5_n320_to_c180_pipeline(handles, target_grid;
                                                        Nz = 137,
                                                        include_convection = true)
        process_era5_n320_window!(pipeline, handles, cli.date, cli.hour)

        # Catrine: PS_moist (PSC2WET in hPa? check units), T (Met_T)
        ds = NCDataset(catrine_file)
        try
            ps_catrine = Array(ds["Met_PSC2WET"])   # (180, 180, 6, 1) — hPa
            ps_unit    = get(ds["Met_PSC2WET"].attrib, "units", "hPa")

            ps_factor  = ps_unit in ("hPa", "millibar") ? 100.0 : 1.0
            @info "Catrine PSC2WET shape" size(ps_catrine) units=ps_unit converted_to_Pa=ps_factor

            println()
            println("PS (Pa) comparison — pipeline vs Catrine, per panel:")
            for p in 1:6
                _summarize_panel("PS p$p",
                                  pipeline.c180_fields.ps[p],
                                  ps_factor .* @view(ps_catrine[:, :, p, 1]))
            end

            # Pipeline T is at L137 native; Catrine T is at L72 when available.
            # Some Catrine snapshots (e.g. the 06 UTC instantaneous bundle)
            # carry only the 2D met fields, so the temperature block is
            # gated on `Met_T` being present.
            if haskey(ds, "Met_T")
                println()
                println("T (K) global range — pipeline (L137) vs Catrine (L72):")
                t_catrine  = Array(ds["Met_T"])
                t_pipe_all = vcat([vec(pipeline.c180_fields.t[p]) for p in 1:6]...)
                @printf "  pipeline T: min=%.2f max=%.2f mean=%.2f\n" minimum(t_pipe_all) maximum(t_pipe_all) mean(t_pipe_all)
                @printf "  catrine T : min=%.2f max=%.2f mean=%.2f\n" minimum(t_catrine) maximum(t_catrine) mean(t_catrine)
            else
                println()
                println("Met_T not in snapshot — skipping temperature dynamic-range check.")
            end

            # Dry-mass closure on the C180 mesh (re-derived from regridded
            # PS + Q at runtime, not regridded directly).
            println()
            println("ERA5 source-mesh dry-mass check:")
            ps_total_source = sum(pipeline.window_fields.ps) / length(pipeline.window_fields.ps)
            ps_dry_source   = sum(pipeline.dry_fields.ps_dry) / length(pipeline.dry_fields.ps_dry)
            @printf "  source-mesh global mean PS_total = %.1f Pa\n" ps_total_source
            @printf "  source-mesh global mean PS_dry   = %.1f Pa\n" ps_dry_source
            @printf "  PS_dry / PS_total ratio          = %.5f\n" ps_dry_source / ps_total_source
            @printf "  total dry-atmosphere mass        = %.3e kg\n" sum(pipeline.dry_fields.m_dry)
        finally
            close(ds)
        end
    finally
        close_era5_day!(handles)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
