"""
    parse_day_filter(args) -> Union{Nothing, Date}

Extract the optional `--day YYYY-MM-DD` filter from the script argument list.
Returns `nothing` when no day filter is requested.
"""
function parse_day_filter(args)
    idx = findfirst(==("--day"), args)
    idx === nothing && return nothing
    idx == length(args) && error("Missing date after --day")
    return Date(args[idx + 1])
end

"""
    resolve_mass_basis(cfg) -> (thermo_dir, include_qv, mass_basis)

Parse humidity-related runtime choices from the TOML configuration.

- `thermo_dir` is the directory containing daily ERA5 thermo NetCDF files.
- `include_qv` controls whether merged `qv` is written into the output payload.
- `mass_basis` is `:moist` or `:dry` and determines whether native fields are
  converted to dry basis before vertical merging.

Dry-basis is the default (Invariant 14). All transport binaries should use dry
basis so that Poisson balance and cm diagnosis operate on dry mass. Moist basis
is available as an explicit opt-out for diagnostic purposes only.

Dry-basis preprocessing requires a humidity source (`input.thermo_dir`).
"""
function resolve_mass_basis(cfg)
    thermo_dir = expanduser(get(get(cfg, "input", Dict()), "thermo_dir", ""))
    include_qv = Bool(get(get(cfg, "output", Dict{String, Any}()), "include_qv", !isempty(thermo_dir)))
    basis_str = lowercase(String(get(get(cfg, "output", Dict{String, Any}()), "mass_basis", "dry")))
    mass_basis = basis_str == "dry" ? :dry : :moist

    if basis_str != "moist" && basis_str != "dry"
        error("Invalid output.mass_basis='$basis_str' (expected 'moist' or 'dry')")
    end
    if mass_basis == :dry && isempty(thermo_dir)
        error("output.mass_basis='dry' requires input.thermo_dir for native qv")
    end

    return thermo_dir, include_qv, mass_basis
end

"""
    resolve_output_float_type(cfg) -> Type

Return the on-disk floating-point type requested by the config.
"""
resolve_output_float_type(cfg) =
    String(get(cfg["numerics"], "float_type", "Float32")) == "Float64" ? Float64 : Float32

"""
    resolve_mass_fix_settings(cfg) -> NamedTuple

Parse the optional global dry-surface-pressure mass-fix settings.
"""
function resolve_mass_fix_settings(cfg)
    mass_fix_cfg = get(cfg, "mass_fix", Dict{String, Any}())
    return (
        mass_fix_enable = Bool(get(mass_fix_cfg, "enable", true)),
        target_ps_dry_pa = Float64(get(mass_fix_cfg, "target_ps_dry_pa", 98726.0)),
        qv_global_climatology = Float64(get(mass_fix_cfg, "qv_global_climatology", 0.00247)),
    )
end

"""
    resolve_runtime_settings(cfg) -> NamedTuple

Resolve the script configuration into a compact runtime settings bundle used by
the day-processing pipeline.
"""
function resolve_runtime_settings(cfg)
    spectral_dir = expanduser(cfg["input"]["spectral_dir"])
    coeff_path = expanduser(cfg["input"]["coefficients"])
    out_dir = expanduser(cfg["output"]["directory"])
    thermo_dir, include_qv, mass_basis = resolve_mass_basis(cfg)

    level_top = Int(get(cfg["grid"], "level_top", 1))
    level_bot = Int(get(cfg["grid"], "level_bot", 137))
    min_dp = Float64(cfg["grid"]["merge_min_thickness_Pa"])

    dt = Float64(cfg["numerics"]["dt"])
    met_interval = Float64(cfg["numerics"]["met_interval"])

    return merge((
        spectral_dir = spectral_dir,
        coeff_path = coeff_path,
        out_dir = out_dir,
        thermo_dir = thermo_dir,
        include_qv = include_qv,
        mass_basis = mass_basis,
        level_range = level_top:level_bot,
        min_dp = min_dp,
        dt = dt,
        met_interval = met_interval,
        half_dt = dt / 2,
        output_float_type = resolve_output_float_type(cfg),
    ), resolve_mass_fix_settings(cfg))
end

"""
    build_vertical_setup(coeff_path, level_range, min_dp, cfg_grid) -> NamedTuple

Construct the native and merged vertical-coordinate metadata for the current
run, including the native-to-merged `merge_map`.

When `grid.echlevs` is set, the named level selection is used. Otherwise thin
levels are merged by the configured minimum pressure thickness.
"""
function build_vertical_setup(coeff_path::String, level_range, min_dp::Float64, cfg_grid)
    ab = load_ab_coefficients(coeff_path, level_range)
    vc_native = load_era5_vertical_coordinate(coeff_path, first(level_range), last(level_range))
    echlevs_name = get(cfg_grid, "echlevs", "")

    merged_vc, merge_map =
        if !isempty(echlevs_name)
            echlevs_map = Dict(
                "ml137_tropo34" => ECHLEVS_ML137_TROPO34,
                "ml137_66L" => ECHLEVS_ML137_66L,
                "ml137_full" => collect(137:-1:0),
            )
            haskey(echlevs_map, echlevs_name) ||
                error("Unknown echlevs config: $echlevs_name. Available: $(join(keys(echlevs_map), ", "))")
            select_levels_echlevs(vc_native, echlevs_map[echlevs_name])
        else
            merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
        end

    return (
        ab = ab,
        level_range = level_range,
        vc_native = vc_native,
        merged_vc = merged_vc,
        merge_map = merge_map,
        Nz_native = n_levels(vc_native),
        Nz = n_levels(merged_vc),
    )
end

"""
    available_spectral_dates(spectral_dir) -> Vector{Date}

Discover all processing dates available in the spectral-input directory by
scanning for `era5_spectral_YYYYMMDD_lnsp.gb`.
"""
function available_spectral_dates(spectral_dir::String)
    dates = Date[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    isempty(dates) && error("No spectral GRIB files found in $spectral_dir")
    return dates
end

"""
    select_processing_dates(dates, day_filter) -> Vector{Date}

Apply the optional single-day filter to the discovered date list.
"""
function select_processing_dates(dates, day_filter)
    day_filter === nothing && return dates
    filtered = filter(==(day_filter), dates)
    isempty(filtered) && error("Date $day_filter not found in spectral data")
    return filtered
end

"""
    next_day_hour0(date, dates, spectral_dir, T_target)

Load the next day's hour-0 spectral fields when available so the final window of
the current day can form forward differences for `dam`, `dbm`, and `dm`.
Returns `nothing` when no next-day file is available.
"""
function next_day_hour0(date::Date, dates, spectral_dir::String, T_target::Int)
    next_day = date + Day(1)
    next_path = joinpath(spectral_dir, "era5_spectral_$(Dates.format(next_day, "yyyymmdd"))_lnsp.gb")
    if next_day in dates || isfile(next_path)
        return read_hour0_spectral(spectral_dir, next_day; T_target)
    end
    return nothing
end

"""
    log_preprocessor_configuration(settings, grid, vertical)

Emit the human-readable startup summary for the current preprocessing run.
"""
function log_preprocessor_configuration(settings, grid::AbstractTargetGeometry, vertical)
    mass_fix_summary =
        if !settings.mass_fix_enable
            "OFF"
        elseif settings.include_qv || settings.mass_basis == :dry
            "ON  (target_ps_dry=$(settings.target_ps_dry_pa) Pa, qv_source=native_hourly)"
        else
            "ON  (target_ps_dry=$(settings.target_ps_dry_pa) Pa, qv_global=$(settings.qv_global_climatology))"
        end

    @info """
    Fused Spectral -> v4 Binary Preprocessor
    ==========================================
    Spectral dir:  $(settings.spectral_dir)
    Output dir:    $(settings.out_dir)
    Target grid:   $(target_summary(grid))
    Native levels: $(vertical.Nz_native) ($(first(vertical.level_range))-$(last(vertical.level_range)))
    Merged levels: $(vertical.Nz) (min_dp=$(settings.min_dp) Pa)
    DT:            $(settings.dt) s (half_dt=$(settings.half_dt) s)
    Met interval:  $(settings.met_interval) s
    T_target:      $(settings.T_target)
    Threads:       $(Threads.nthreads())
    Float type:    $(settings.output_float_type)
    Mass basis:    $(settings.mass_basis)
    Include qv:    $(settings.include_qv)
    Mass fix:      $(mass_fix_summary)
    """
end
