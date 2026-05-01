# Spectral ERA5 to structured lat-lon transport-binary preprocessing path.

"""
    process_day(date, grid::LatLonTargetGeometry, settings, vertical; next_day_hour0=nothing)

Run the full one-day preprocessing workflow for the structured lat-lon target:
read spectral input, process all windows, apply Poisson balance, and write the
final binary.
"""
function process_day(date::Date,
                     grid::LatLonTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    FT = settings.output_float_type
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nx = nlon(grid)
    Ny = nlat(grid)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral(vo_d_path, lnsp_path;
                             T_target=settings.T_target,
                             cache_dir=settings.spectral_cache_dir)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    counts = window_element_counts(grid, Nz;
                                    include_qv=settings.include_qv,
                                    tm5_convection=settings.tm5_convection_enable,
                                    include_surface=settings.include_surface)
    byte_sizes = window_byte_sizes(counts, FT, Nt)
    counts = merge(counts, (bytes_per_window = byte_sizes.bytes_per_window,))

    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    expected_sections = expected_payload_sections(settings)
    skip, reason = existing_output_schema_matches(bin_path, byte_sizes.total_bytes, expected_sections)
    if skip
        @info "  SKIP (exists, size + schema match): $(basename(bin_path))"
        return bin_path
    elseif isfile(bin_path) && filesize(bin_path) == byte_sizes.total_bytes
        @info "  REGEN (size match, $(reason)): $(basename(bin_path))"
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), byte_sizes.total_bytes / 1e9, Nt)

    provenance = script_provenance()
    sizes = (Nx = Nx, Ny = Ny, Nz = Nz, Nz_native = Nz_native, Nt = Nt,
             steps_per_met = steps_per_met)
    header = build_v4_header(date, grid, vertical, settings, FT, counts, sizes, provenance)
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large: $(length(header_json)) >= $(HEADER_SIZE)")

    transform = allocate_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)
    storage = allocate_window_storage(Nt, FT;
                                       include_qv=settings.include_qv,
                                       tm5_convection=settings.tm5_convection_enable,
                                       include_surface=settings.include_surface)
    qv = allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)

    # Plan 24 Commit 4: TM5 convection setup (LL target == ERA5 native
    # 720×361 only — see NOTES.md for the scope narrowing).  When
    # enabled, open the day's physics BIN, shape-check against the
    # target, and allocate the per-day workspace + cleanup stats.
    physics_reader = nothing
    tm5_ws         = nothing
    tm5_stats      = nothing
    if settings.tm5_convection_enable
        physics_reader = open_era5_physics_binary(settings.tm5_physics_bin_dir, date)
        Nlon_src = physics_reader.header.Nlon
        Nlat_src = physics_reader.header.Nlat
        (Nlon_src == Nx && Nlat_src == Ny) || error(
            "Plan 24 Commit 4 requires LL target == physics BIN shape. " *
            "BIN is ($Nlon_src, $Nlat_src), target is ($Nx, $Ny). " *
            "Either (a) use a 720×361 LL target config, or (b) wait for " *
            "Commit 4b/4c (regrid + PS sourcing for coarser / non-LL targets).")
        tm5_ws    = allocate_tm5_workspace(Nlon_src, Nlat_src, Nz_native, Nz, FT;
                                            physics_eltype = Float32)
        tm5_stats = TM5CleanupStats()
    end

    surface_reader = settings.include_surface ?
        open_era5_surface_reader(settings.surface_dir, date, Nx, Ny) : nothing

    log_mass_fix_configuration(settings)
    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    try
        for (win_idx, hour) in enumerate(spec.hours)
            process_window!(win_idx, hour, spec, grid, vertical, settings,
                            transform, merged, qv, storage, ps_offsets;
                            physics_reader = physics_reader,
                            tm5_ws         = tm5_ws,
                            tm5_stats      = tm5_stats,
                            surface_reader = surface_reader)
        end

        if settings.mass_fix_enable
            @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                           minimum(ps_offsets[1:Nt]),
                           maximum(ps_offsets[1:Nt]),
                           sum(ps_offsets[1:Nt]) / Nt)
        end

        tm5_stats === nothing || log_tm5_cleanup_stats(tm5_stats, date_str)

        last_hour_next = next_day_merged_fields(next_day_hour0, date, grid, vertical,
                                                settings, transform, merged, qv, ps_offsets)

        apply_poisson_balance!(storage, last_hour_next, sizes.steps_per_met)
        fill_qv_endpoints!(storage, last_hour_next)

        header["ps_offsets_pa_per_window"] = ps_offsets[1:Nt]
        header["ps_offsets_next_day_hour0_pa"] = ps_offsets[Nt + 1]
        header_json = JSON3.write(header)
        length(header_json) < HEADER_SIZE ||
            error("Header JSON too large after offsets update: $(length(header_json)) >= $(HEADER_SIZE)")

        write_day_binary!(bin_path, header_json, storage, settings, merged, last_hour_next)
    finally
        physics_reader === nothing || close_era5_physics_binary(physics_reader)
        surface_reader === nothing || close_era5_surface_reader(surface_reader)
    end

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path), actual / 1e9, time() - t_day)
    actual == byte_sizes.total_bytes ||
        error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d", byte_sizes.total_bytes, actual))

    last_merged = (m = storage.all_m[Nt], am = storage.all_am[Nt], bm = storage.all_bm[Nt])
    return bin_path, last_merged
end
