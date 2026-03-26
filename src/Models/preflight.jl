# ---------------------------------------------------------------------------
# Pre-run validation (preflight checks)
#
# Run with `--check` or `-c` flag: validates data availability, output paths,
# and config consistency BEFORE starting the simulation.
# ---------------------------------------------------------------------------

using Dates

"""
    preflight_check!(model; verbose::Bool=false)

Validate that all required input data exists and is accessible before running.
Collects all errors/warnings and reports them together.

Called from `scripts/run.jl` when `--check` or `-c` flag is passed.
"""
function preflight_check!(model; verbose::Bool=false)
    errors   = String[]
    warnings = String[]
    config   = get(model.metadata, "config", Dict())
    met_cfg  = get(config, "met_data", Dict())
    adv_cfg  = get(config, "advection", Dict())
    out_cfg  = get(config, "output", Dict())
    ic_cfg   = get(config, "initial_conditions", Dict())
    tracer_cfg = get(config, "tracers", Dict())

    @info "Preflight checks starting..."

    # --- 1. Met data availability ---
    _check_met_data!(errors, warnings, met_cfg; verbose)

    # --- 2. Surface data availability ---
    _check_surface_data!(errors, warnings, met_cfg, config; verbose)

    # --- 3. Initial condition files ---
    _check_initial_conditions!(errors, warnings, ic_cfg; verbose)

    # --- 4. Emission files ---
    _check_emission_files!(errors, warnings, tracer_cfg; verbose)

    # --- 5. Output directory + disk space ---
    _check_output!(errors, warnings, out_cfg; verbose)

    # --- 6. Coordinate file (CS grids) ---
    _check_coord_file!(errors, warnings, met_cfg; verbose)

    # --- 7. Config consistency ---
    _check_config_consistency!(warnings, adv_cfg, config)

    # --- 8. Physics data consistency (DTRAIN for RAS, QV for dry transport) ---
    _check_physics_data_consistency!(errors, warnings, met_cfg, config; verbose)

    # --- Report ---
    for w in warnings
        @warn w
    end
    if !isempty(errors)
        for e in errors
            @error e
        end
        error("Preflight failed with $(length(errors)) error(s):\n" *
              join(["  - $e" for e in errors], "\n"))
    end

    @info "Preflight checks passed ($(length(warnings)) warning(s))"
    return nothing
end

# =====================================================================
# Individual check functions
# =====================================================================

function _check_met_data!(errors, warnings, met_cfg; verbose::Bool=false)
    preprocessed_dir = expanduser(get(met_cfg, "preprocessed_dir", ""))
    netcdf_dir = expanduser(get(met_cfg, "netcdf_dir", ""))
    start_date = _parse_date(get(met_cfg, "start_date", ""))
    end_date   = _parse_date(get(met_cfg, "end_date", ""))

    if !isempty(preprocessed_dir)
        if !isdir(preprocessed_dir)
            push!(errors, "preprocessed_dir not found: $preprocessed_dir")
            return
        end
        is_verbose() && @info "Checking preprocessed binary files in $preprocessed_dir"

        # Check for binary files covering the date range
        if start_date !== nothing && end_date !== nothing
            missing_dates = Date[]
            d = start_date
            while d < end_date
                # Pattern: geosfp_cs_YYYYMMDD_float32.bin
                datestr = Dates.format(d, "yyyymmdd")
                found = any(f -> contains(f, datestr) && endswith(f, ".bin"),
                            readdir(preprocessed_dir))
                if !found
                    push!(missing_dates, d)
                end
                d += Day(1)
            end
            if !isempty(missing_dates)
                n = length(missing_dates)
                first_missing = missing_dates[1]
                last_missing = missing_dates[end]
                push!(errors, "Missing $n met data file(s) in $preprocessed_dir: " *
                      "$first_missing to $last_missing")
                if verbose
                    for md in missing_dates
                        is_verbose() && @info "  Missing: $md"
                    end
                end
            else
                is_verbose() && @info "All met data files present for $start_date to $end_date"
            end
        end
    elseif !isempty(netcdf_dir)
        if !isdir(netcdf_dir)
            push!(errors, "netcdf_dir not found: $netcdf_dir")
        end
    else
        push!(warnings, "No preprocessed_dir or netcdf_dir specified in [met_data]")
    end
end

function _check_surface_data!(errors, warnings, met_cfg, config; verbose::Bool=false)
    conv_cfg = get(config, "convection", Dict())
    diff_cfg = get(config, "diffusion", Dict())
    needs_sfc = get(conv_cfg, "type", "none") != "none" ||
                get(diff_cfg, "type", "none") != "none"
    !needs_sfc && return

    sfc_dir = expanduser(get(met_cfg, "surface_data_bin_dir", ""))
    if isempty(sfc_dir)
        push!(warnings, "Convection/diffusion enabled but no surface_data_bin_dir specified")
        return
    end
    if !isdir(sfc_dir)
        push!(errors, "surface_data_bin_dir not found: $sfc_dir")
        return
    end

    # Check date coverage
    start_date = _parse_date(get(met_cfg, "start_date", ""))
    end_date   = _parse_date(get(met_cfg, "end_date", ""))
    if start_date !== nothing && end_date !== nothing
        missing = Date[]
        d = start_date
        while d < end_date
            datestr = Dates.format(d, "yyyymmdd")
            found = any(f -> contains(f, datestr), readdir(sfc_dir))
            !found && push!(missing, d)
            d += Day(1)
        end
        if !isempty(missing)
            push!(errors, "Missing $(length(missing)) surface data file(s): " *
                  "$(missing[1]) to $(missing[end])")
        end
    end
    is_verbose() && @info "Surface data check passed" dir=sfc_dir
end

function _check_initial_conditions!(errors, warnings, ic_cfg; verbose::Bool=false)
    for (tracer, cfg) in ic_cfg
        cfg isa Dict || continue
        file = expanduser(get(cfg, "file", ""))
        if isempty(file)
            push!(warnings, "No IC file for tracer '$tracer'")
        elseif !isfile(file)
            push!(errors, "IC file not found for '$tracer': $file")
        else
            is_verbose() && @info "IC file OK: $tracer → $file"
        end
    end
end

function _check_emission_files!(errors, warnings, tracer_cfg; verbose::Bool=false)
    for (tracer, cfg) in tracer_cfg
        cfg isa Dict || continue
        file = expanduser(get(cfg, "file", ""))
        if !isempty(file) && !isfile(file)
            push!(errors, "Emission file not found for '$tracer': $file")
        elseif !isempty(file)
            is_verbose() && @info "Emission file OK: $tracer → $file"
        end
    end
end

function _check_output!(errors, warnings, out_cfg; verbose::Bool=false)
    filename = get(out_cfg, "filename", "")
    isempty(filename) && return

    out_dir = dirname(expanduser(filename))
    if !isdir(out_dir)
        try
            mkpath(out_dir)
            @info "Created output directory: $out_dir"
        catch e
            push!(errors, "Cannot create output directory: $out_dir ($e)")
            return
        end
    end

    # Disk space check
    _check_disk_space!(warnings, out_dir)
end

function _check_disk_space!(warnings, dir)
    try
        stat = read(`df -BG --output=avail $dir`, String)
        lines = split(strip(stat), '\n')
        length(lines) >= 2 || return
        m = match(r"(\d+)", lines[2])
        m === nothing && return
        avail_gb = parse(Int, m.captures[1])
        if avail_gb < 10
            push!(warnings, "Low disk space: $(avail_gb) GB available in $dir")
        else
            is_verbose() && @info "Disk space OK: $(avail_gb) GB available in $dir"
        end
    catch
        # Non-fatal — skip disk check on non-Linux systems
    end
end

function _check_coord_file!(errors, warnings, met_cfg; verbose::Bool=false)
    grid_type = get(met_cfg, "driver", "")
    !contains(grid_type, "cs") && return

    coord_file = expanduser(get(met_cfg, "coord_file", ""))
    if isempty(coord_file)
        push!(warnings, "No coord_file specified for cubed-sphere grid")
    elseif !isfile(coord_file)
        push!(errors, "Coordinate file not found: $coord_file")
    else
        is_verbose() && @info "Coordinate file OK: $coord_file"
    end
end

function _check_config_consistency!(warnings, adv_cfg, config)
    if get(adv_cfg, "per_step_remap", false) && !get(adv_cfg, "vertical_remap", false)
        push!(warnings, "per_step_remap=true but vertical_remap=false — per_step_remap has no effect")
    end
    if get(adv_cfg, "linrood", false) && get(adv_cfg, "scheme", "slopes") != "ppm"
        push!(warnings, "linrood=true requires scheme=\"ppm\" — linrood will be ignored")
    end
end

function _check_physics_data_consistency!(errors, warnings, met_cfg, config; verbose::Bool=false)
    conv_cfg = get(config, "convection", Dict())
    adv_cfg  = get(config, "advection", Dict())
    conv_type = get(conv_cfg, "type", "none")
    needs_dtrain = conv_type == "ras"
    needs_qv = get(adv_cfg, "gchp", false) || get(adv_cfg, "dry_correction", false)

    sfc_dir = expanduser(get(met_cfg, "surface_data_bin_dir", ""))
    start_date = _parse_date(get(met_cfg, "start_date", ""))
    end_date   = _parse_date(get(met_cfg, "end_date", ""))
    start_date === nothing && return
    end_date === nothing && return

    # Check DTRAIN (A3dyn binary) availability when RAS is enabled
    if needs_dtrain && !isempty(sfc_dir) && isdir(sfc_dir)
        missing_dtrain = Date[]
        d = start_date
        while d < end_date
            datestr = Dates.format(d, "yyyymmdd")
            has_a3dyn = any(f -> contains(f, datestr) && contains(f, "A3dyn"),
                           readdir(sfc_dir))
            !has_a3dyn && push!(missing_dtrain, d)
            d += Day(1)
        end
        if !isempty(missing_dtrain)
            n = length(missing_dtrain)
            push!(errors, "RAS convection requires DTRAIN (A3dyn binary), but $n day(s) " *
                  "are missing in $sfc_dir: $(missing_dtrain[1]) to $(missing_dtrain[end]). " *
                  "Without DTRAIN, convection silently falls back to Tiedtke — " *
                  "run convert_surface_cs_to_binary.jl to generate A3dyn binaries.")
        else
            is_verbose() && @info "A3dyn (DTRAIN) data present for all dates"
        end
    end

    # Check QV availability when GCHP/dry transport is enabled
    if needs_qv && !isempty(sfc_dir) && isdir(sfc_dir)
        preproc_dir = expanduser(get(met_cfg, "preprocessed_dir", ""))
        if !isempty(preproc_dir) && isdir(preproc_dir)
            # Check if v4 binary has embedded QV — sample first file
            first_bin = ""
            for f in readdir(preproc_dir)
                if endswith(f, ".bin") && contains(f, "float32")
                    first_bin = joinpath(preproc_dir, f)
                    break
                end
            end
            if !isempty(first_bin)
                try
                    raw = open(io -> String(read(io, 65536)), first_bin)
                    idx = findfirst('\0', raw)
                    hdr_str = idx === nothing ? raw : raw[1:idx-1]
                    if contains(hdr_str, "\"include_qv\":false") || !contains(hdr_str, "include_qv")
                        # v4 binary doesn't have QV — check I3 binary fallback
                        missing_qv = Date[]
                        d = start_date
                        while d < end_date
                            datestr = Dates.format(d, "yyyymmdd")
                            has_i3 = any(f -> contains(f, datestr) && contains(f, "I3"),
                                         readdir(sfc_dir))
                            !has_i3 && push!(missing_qv, d)
                            d += Day(1)
                        end
                        if !isempty(missing_qv)
                            push!(warnings, "GCHP/dry transport needs QV but mass flux binary " *
                                  "lacks embedded QV and $(length(missing_qv)) I3 binary " *
                                  "file(s) are missing — QV will fall back to slow NetCDF. " *
                                  "Re-preprocess with include_qv=true or convert I3 binaries.")
                        end
                    else
                        is_verbose() && @info "QV embedded in mass flux binary (v4 format)"
                    end
                catch e
                    is_verbose() && @info "Could not inspect mass flux binary header: $e"
                end
            end
        end
    end
end

# =====================================================================
# Helpers
# =====================================================================

function _parse_date(s::AbstractString)
    isempty(s) && return nothing
    try
        return Date(s)
    catch
        return nothing
    end
end
