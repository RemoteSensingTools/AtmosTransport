# TOML-driven preprocessing entry point used by CLI scripts.

# =========================================================================
# TOML-driven entry point — called by the CLI script
# =========================================================================

"""
    process_day(cfg::Dict; day_override=nothing)

Top-level TOML-driven entry point for the unified preprocessor CLI.
Parses the config, constructs target geometry and vertical setup,
then dispatches to the grid-specific `process_day(date, grid, ...)` method.

Called by `scripts/preprocessing/preprocess_transport_binary.jl`.
"""
function process_day(cfg::Dict{String, Any};
                     day_override::Union{String, Nothing}=nothing)
    grid = build_target_geometry(cfg["grid"], Float64)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range, settings.min_dp, cfg["grid"])

    log_preprocessor_configuration(settings, grid, vertical)
    ensure_supported_target(grid)

    # Date selection
    dates = if day_override !== nothing
        [Date(day_override)]
    else
        day_filter = parse_day_filter(day_override === nothing ? String[] : ["--day", day_override])
        select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)
    end

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))
    t_total = time()

    for (idx, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", idx, length(dates), date)
        next_day_h0 = next_day_hour0(date, dates, settings.spectral_dir, settings.T_target;
                                     cache_dir=settings.spectral_cache_dir)
        next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")
        process_day(date, grid, settings, vertical; next_day_hour0=next_day_h0)
    end

    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), elapsed, elapsed / length(dates))
end
