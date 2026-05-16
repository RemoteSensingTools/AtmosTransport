# =========================================================================
# TOML-driven preprocessing entry point — THE canonical orchestrator.
#
# Every supported source (ERA5 spectral, GEOS-IT, GEOS-FP, MERRA-2 …) and
# every supported target topology (LatLon, ReducedGaussian, CubedSphere)
# routes through this single function. New sources plug in via
# `AbstractMetSettings` + `load_met_settings`, never via a parallel CLI or
# orchestrator (architectural invariant — see plan geos-followups).
# =========================================================================

# ---------------------------------------------------------------------------
# Source detection. A config that declares `[source].toml = "..."` is
# routed through the typed `AbstractMetSettings` factory; otherwise the
# legacy ERA5-spectral path resolves a NamedTuple settings via
# `resolve_runtime_settings`. Both paths converge on
# `process_day(date, grid, settings, vertical; ...)` for actual work.
# ---------------------------------------------------------------------------

@inline _is_native_source_cfg(cfg::AbstractDict) =
    haskey(cfg, "source") && haskey(cfg["source"], "toml")

# ---------------------------------------------------------------------------
# Date selection — supports `--day YYYY-MM-DD` (single) and
# `--start ... --end ...` (range, inclusive).
# ---------------------------------------------------------------------------

function _resolve_dates_native(cfg::AbstractDict;
                               day_override = nothing,
                               start_date   = nothing,
                               end_date     = nothing)
    if day_override !== nothing
        return [Date(String(day_override))]
    elseif start_date !== nothing && end_date !== nothing
        return collect(Date(String(start_date)):Day(1):Date(String(end_date)))
    elseif haskey(cfg, "input") && haskey(cfg["input"], "start_date") &&
           haskey(cfg["input"], "end_date")
        return collect(Date(String(cfg["input"]["start_date"])):Day(1):Date(String(cfg["input"]["end_date"])))
    end
    error("Native-source preprocessing needs `--day`, `--start/--end`, " *
          "or `[input].start_date`/`[input].end_date` in the TOML.")
end

# ---------------------------------------------------------------------------
# Output path resolution for the native-source path. The TOML's
# `[output].directory` is the destination; the date and float type pick the
# filename. Sources can override via `_native_output_filename(settings, date, FT)`.
# ---------------------------------------------------------------------------

"""
    _native_output_filename(settings, date, FT) -> String

Per-source output filename for native-source preprocessing. Concrete
sources override this in their own files (e.g. `sources/geos.jl`); the
default is a source-agnostic prefix.
"""
_native_output_filename(::AbstractMetSettings, date::Date, FT::Type) =
    "transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"

function _native_output_path(cfg::AbstractDict, settings::AbstractMetSettings,
                             date::Date, FT::Type)
    out_dir = expand_data_path(String(cfg["output"]["directory"]))
    mkpath(out_dir)
    return joinpath(out_dir, _native_output_filename(settings, date, FT))
end

# ---------------------------------------------------------------------------
# Float-type / numerics resolution (shared between native and spectral paths).
# ---------------------------------------------------------------------------

function _resolve_float_type(cfg::AbstractDict)
    s = String(get(get(cfg, "numerics", Dict()), "float_type", "Float64"))
    return s == "Float32" ? Float32 : Float64
end

_resolve_dt_met(cfg::AbstractDict) =
    Float64(get(get(cfg, "numerics", Dict()), "dt_met_seconds", 3600.0))

_resolve_mass_basis(cfg::AbstractDict) =
    Symbol(get(get(cfg, "output", Dict()), "mass_basis", "dry"))

_resolve_chain_mass(cfg::AbstractDict) =
    Bool(get(get(cfg, "numerics", Dict()), "chain_mass", true))

# Canonical CS substep-positivity contract knobs. Defaults match the gate the
# regrid path has enforced since plan 39 — every CS-producing preprocessor
# (spectral, regrid, GEOS-native) reads them from the same `[numerics]` block
# so a config can't silently bypass the contract on one path while honoring it
# on another.
#
# Validation: `positivity_cfl_limit` must be a finite positive value bounded
# by 1.0. A `0.0` or negative value would make
# `summarize_cs_positivity_status` divide by zero (or produce a negative
# recommendation) AFTER a contract violation has been recorded — by then the
# preprocessor has already paid the cost of the loop, so we'd rather refuse
# to start than throw an `InexactError` at the end. Values > 1.0 are also
# nonsensical (the runtime's `_cs_static_subcycle_count` only protects
# against outgoing < cell mass).
function _resolve_positivity_cfl_limit(cfg::AbstractDict)
    raw = get(get(cfg, "numerics", Dict()), "positivity_cfl_limit", 0.95)
    limit = Float64(raw)
    isfinite(limit) && 0 < limit <= 1 ||
        error("Invalid `[numerics].positivity_cfl_limit = $(raw)`: must be a " *
              "finite value in (0, 1]. The default `0.95` is what the regrid " *
              "path has enforced since plan 39.")
    return limit
end

_resolve_require_substep_positivity(cfg::AbstractDict) =
    Bool(get(get(cfg, "numerics", Dict()), "require_substep_positivity", true))

# ---------------------------------------------------------------------------
# Native-source preprocessor: typed `AbstractMetSettings` + cross-day state
# carry (e.g. GEOS pressure-fixer chained mass).
# ---------------------------------------------------------------------------

function _process_day_native(cfg::AbstractDict;
                             day_override = nothing,
                             start_date   = nothing,
                             end_date     = nothing)
    src_cfg = cfg["source"]
    toml_relpath = String(src_cfg["toml"])
    toml_path = isabspath(toml_relpath) ? toml_relpath :
                joinpath(@__DIR__, "..", "..", "..", toml_relpath)

    # Single source of truth for the float type: build the grid AND the
    # source settings against the same `FT` so cell areas and the reader's
    # mass buffers share an element type. (Codex 2026-04-25: P2 fix —
    # without this, Float32 configs trip a `MethodError` in
    # `_delp_pa_to_air_mass_kg!` because cell_areas was Matrix{Float64}.)
    FT   = _resolve_float_type(cfg)
    grid = build_target_geometry(cfg["grid"], FT)
    ensure_supported_target(grid)

    # Single source of truth for hybrid coefficients: the source descriptor
    # owns it. Overrides flow back into `settings.coefficients_file` so the
    # reader (`open_day` → `endpoint_dry_mass!`) and the writer's vertical
    # setup never desync. (Codex 2026-04-25: P2 fix.)
    settings_kwargs = (root_dir = expand_data_path(String(src_cfg["root_dir"])),)
    if haskey(src_cfg, "include_surface")
        settings_kwargs = (settings_kwargs..., include_surface = Bool(src_cfg["include_surface"]))
    end
    if haskey(src_cfg, "include_convection")
        settings_kwargs = (settings_kwargs..., include_convection = Bool(src_cfg["include_convection"]))
    end
    for key in ("physics_dir", "surface_dir")
        if haskey(src_cfg, key)
            settings_kwargs = (settings_kwargs..., physics_dir = expand_data_path(String(src_cfg[key])))
            break
        end
    end
    if haskey(src_cfg, "physics_layout")
        settings_kwargs = (settings_kwargs..., physics_layout = Symbol(src_cfg["physics_layout"]))
    end
    cfg_vertical = get(cfg, "vertical", Dict())
    if haskey(cfg_vertical, "coefficients")
        settings_kwargs = (settings_kwargs..., coefficients_file =
                           expand_data_path(String(cfg_vertical["coefficients"])))
    end
    settings = load_met_settings(toml_path; settings_kwargs...)

    vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))
    Nz = length(vc.A) - 1
    vertical = (merged_vc = vc, Nz = Nz, Nz_native = Nz)

    mass_basis     = _resolve_mass_basis(cfg)
    dt_met_seconds = _resolve_dt_met(cfg)
    chain_mass     = _resolve_chain_mass(cfg)
    positivity_cfl_limit       = _resolve_positivity_cfl_limit(cfg)
    require_substep_positivity = _resolve_require_substep_positivity(cfg)
    if !(grid isa CubedSphereTargetGeometry && settings isa AbstractGEOSSettings)
        error("Unified native-source preprocessing currently supports only " *
              "native GEOS → cubed_sphere preprocessing.")
    end

    dates = _resolve_dates_native(cfg; day_override, start_date, end_date)

    @info @sprintf("Preprocessor: %s  Nc=%d  → %s  Nz=%d  FT=%s  %d day(s)",
                   typeof(settings), settings.Nc, typeof(grid),
                   vertical.Nz, FT, length(dates))
    @info "Plan 41 unified driver enabled for native GEOS → cubed_sphere"

    t_total = time()
    seed_m = nothing                   # source-defined cross-day state (e.g. GEOS PF endpoint)
    for (idx, d) in enumerate(dates)
        out_path = _native_output_path(cfg, settings, d, FT)
        @info "[$idx/$(length(dates))] $(d) → $(out_path)"
        day_kwargs = (
            out_path        = out_path,
            dt_met_seconds  = dt_met_seconds,
            FT              = FT,
            mass_basis      = mass_basis,
            chain_mass      = chain_mass,
            positivity_cfl_limit       = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            seed_m          = seed_m,
        )
        result = process_day(d, grid, settings, vertical; day_kwargs...)
        seed_m = get(result, :final_m, nothing)
    end
    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)",
                   length(dates), elapsed, elapsed / max(length(dates), 1))
    return nothing
end

# ---------------------------------------------------------------------------
# Legacy ERA5-spectral preprocessor: NamedTuple settings via
# `resolve_runtime_settings`. Kept verbatim so existing ERA5 configs keep
# working unchanged. New met sources must use the native-source path.
# ---------------------------------------------------------------------------

function _process_day_spectral(cfg::AbstractDict, grid::AbstractTargetGeometry;
                               day_override = nothing)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range,
                                    settings.min_dp, cfg["grid"])

    log_preprocessor_configuration(settings, grid, vertical)

    dates = if day_override !== nothing
        [Date(String(day_override))]
    else
        day_filter = parse_day_filter(day_override === nothing ? String[] : ["--day", day_override])
        select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)
    end

    positivity_cfl_limit       = _resolve_positivity_cfl_limit(cfg)
    require_substep_positivity = _resolve_require_substep_positivity(cfg)
    spectral_unified_supported =
        grid isa ReducedGaussianTargetGeometry ||
        grid isa CubedSphereTargetGeometry ||
        grid isa LatLonTargetGeometry
    if !spectral_unified_supported
        error("Unified ERA5 spectral preprocessing currently " *
              "supports only latlon, reduced_gaussian, and cubed_sphere targets.")
    end

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))
    @info "Plan 41 unified driver enabled for ERA5 spectral → $(grid_kind(grid))"
    t_total = time()
    run_cache = PreprocessorRunCache(typeof(grid), settings.output_float_type)
    for (idx, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", idx, length(dates), date)
        next_day_h0 = next_day_hour0(date, dates, settings.spectral_dir, settings.T_target;
                                     cache_dir = settings.spectral_cache_dir)
        next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")
        day_kwargs = (
            positivity_cfl_limit       = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            next_day_hour0             = next_day_h0,
            run_cache                  = run_cache,
        )
        process_day(date, grid, settings, vertical; day_kwargs...)
    end
    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)",
                   length(dates), elapsed, elapsed / max(length(dates), 1))
    return nothing
end

# ---------------------------------------------------------------------------
# Public entry point — called by the unified CLI script.
# ---------------------------------------------------------------------------

"""
    process_day(cfg::Dict; day_override=nothing, start_date=nothing, end_date=nothing)

Top-level TOML-driven preprocessor entry. Detects source type from `cfg`:

* `[source].toml = "config/met_sources/<source>.toml"` → typed
  `AbstractMetSettings` path, supports cross-day state carry (e.g. GEOS
  pressure-fixer chained mass) and `--start/--end` date ranges.
* otherwise → legacy ERA5 spectral path (`[input].spectral_dir`).

Both paths converge on `process_day(date, grid::AbstractTargetGeometry,
settings, vertical; ...)` for the per-day work. There is no parallel
GEOS-only or per-source CLI — new sources plug in via
`AbstractMetSettings` + `load_met_settings`.
"""
function process_day(cfg::Dict{String, Any};
                     day_override::Union{String, Nothing} = nothing,
                     start_date::Union{String, Date, Nothing} = nothing,
                     end_date::Union{String, Date, Nothing} = nothing)
    if _is_native_source_cfg(cfg)
        # Native path resolves FT first, then builds grid internally so
        # mesh element type matches reader/state buffers.
        return _process_day_native(cfg; day_override, start_date, end_date)
    else
        # Spectral path: keep the historical Float64 mesh build at the entry
        # (the spectral preprocessor casts to settings.output_float_type later).
        grid = build_target_geometry(cfg["grid"], Float64)
        ensure_supported_target(grid)
        return _process_day_spectral(cfg, grid; day_override)
    end
end
