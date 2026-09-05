# ===========================================================================
# TOML parsing — tracer specs (hoisted from run_transport_binary.jl:57-100)
# ===========================================================================

struct TransportTracerSpec
    name             :: Symbol
    init_cfg         :: Dict{String, Any}
    surface_flux_cfg :: Dict{String, Any}
end

_copy_cfg_dict(cfg) = Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg))

function _tracer_init_cfg(tracer_cfg)
    if haskey(tracer_cfg, "init")
        return _copy_cfg_dict(tracer_cfg["init"])
    end
    cfg = Dict{String, Any}()
    for key in ("kind", "background", "lon0_deg", "lat0_deg", "sigma_lon_deg",
                "sigma_lat_deg", "amplitude", "south_value", "north_value",
                "south", "north", "split_lat_deg", "file", "variable",
                "time_index")
        haskey(tracer_cfg, key) && (cfg[key] = tracer_cfg[key])
    end
    isempty(cfg) && return Dict{String, Any}("kind" => "uniform", "background" => 0.0)
    return cfg
end

function _tracer_surface_flux_cfg(tracer_cfg)
    if haskey(tracer_cfg, "surface_flux")
        return _copy_cfg_dict(tracer_cfg["surface_flux"])
    end
    cfg = Dict{String, Any}()
    for (src_key, dst_key) in (("surface_flux_kind", "kind"),
                               ("surface_flux_file", "file"),
                               ("surface_flux_variable", "variable"),
                               ("surface_flux_time_index", "time_index"),
                               ("surface_flux_month", "month"),
                               ("surface_flux_scale", "scale"))
        haskey(tracer_cfg, src_key) && (cfg[dst_key] = tracer_cfg[src_key])
    end
    return cfg
end

function _parse_tracer_specs(cfg)
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing
    names = sort!(collect(keys(tracers_cfg)))
    isempty(names) && throw(ArgumentError("config has [tracers] but no tracer sections"))
    return Tuple(TransportTracerSpec(Symbol(name),
                                     _tracer_init_cfg(tracers_cfg[name]),
                                     _tracer_surface_flux_cfg(tracers_cfg[name])) for name in names)
end

# ===========================================================================
# GPU runtime helpers (hoisted from run_transport_binary.jl:101-138)
# ===========================================================================

@inline _cfg_architecture_section(cfg) = get(cfg, "architecture", Dict{String, Any}())
@inline _cfg_runtime_backend(cfg) = runtime_backend_from_config(_cfg_architecture_section(cfg))
@inline _cfg_use_gpu(cfg) = is_gpu_backend(_cfg_runtime_backend(cfg))

function _ensure_gpu_runtime!(cfg)
    backend = _cfg_runtime_backend(cfg)
    is_gpu_backend(backend) || return false
    ensure_backend_runtime!(backend)
    return true
end

function _backend_array_adapter(cfg)
    backend = _cfg_runtime_backend(cfg)
    is_gpu_backend(backend) && _ensure_gpu_runtime!(cfg)
    return backend_array_adapter(backend)
end

function _backend_label(cfg)
    backend = _cfg_runtime_backend(cfg)
    return backend_label(backend)
end

function _cfg_float_type(cfg)
    raw = get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")
    s = lowercase(String(raw))
    if s == "float32"
        return Float32
    elseif s == "float64"
        return Float64
    end
    throw(ArgumentError(
        "[numerics] float_type must be \"Float32\" or \"Float64\"; got $(repr(raw))."))
end

function _capture_config_error!(f, errors::Vector{String})
    try
        f()
    catch err
        push!(errors, sprint(showerror, err))
    end
    return errors
end

# A partial window range across files skips forcing between handoffs while
# carrying tracer state forward. Restrict these ranges to single-file debugging.
function _check_multifile_window_range(driver, start_window, stop_window, file_count)
    file_count <= 1 && return nothing
    last_window = total_windows(driver)
    if start_window != 1 || (stop_window !== nothing && Int(stop_window) < last_window)
        throw(ArgumentError(
            "Partial window ranges with multiple input files would skip forcing " *
            "between files. Use one input file for a partial-window run, or " *
            "start_window=1 and the full window range for every file."))
    end
    return nothing
end

function _check_run_window_bounds!(cfg, errors::Vector{String})
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_raw = get(run_cfg, "start_window", 1)
    stop_raw = get(run_cfg, "stop_window", nothing)
    _capture_config_error!(errors) do
        start_window = Int(start_raw)
        start_window >= 1 ||
            throw(ArgumentError("[run] start_window must be >= 1; got $(start_raw)."))
        if stop_raw !== nothing
            stop_window = Int(stop_raw)
            stop_window >= start_window ||
                throw(ArgumentError("[run] stop_window=$(stop_raw) must be >= start_window=$(start_window)."))
        end
    end
    return nothing
end

"""
    validate_config(cfg::AbstractDict) -> (ok::Bool, errors::Vector{String})

Run inexpensive pre-flight checks for a driven runtime config: input shape,
resolved binary paths, numeric type, backend/float compatibility, tracer table
shape, physics option names and values, output settings, and run-window bounds. It does not open binary readers or allocate
model state; topology and payload capability checks still run when
`run_driven_simulation` inspects the first binary.
"""
function validate_config(cfg::AbstractDict)
    errors = String[]
    _capture_config_error!(errors) do
        _check_spec_keys(cfg, ("input", "architecture", "numerics", "run", "advection",
                              "diffusion", "convection", "chemistry", "tracers", "output", "init"),
                         "top-level configuration")
    end
    for (section, allowed) in (
        ("architecture", ("use_gpu", "backend")),
        ("numerics", ("float_type",)),
        ("run", ("start_window", "stop_window", "air_mass_reset_mode", "halo_padding",
                 "Hp", "tracer_name", "scheme", "ppm_order", "reset_air_mass_each_window")))
        _capture_config_error!(errors) do
            _check_spec_keys(get(cfg, section, Dict{String,Any}()), allowed, "[$section]")
        end
    end

    input_cfg = get(cfg, "input", nothing)
    if !(input_cfg isa AbstractDict)
        push!(errors, "[input] must be a TOML table with `binary_paths` or `folder + start_date + end_date`.")
    else
        binary_paths_ref = Ref(String[])
        _capture_config_error!(errors) do
            paths = expand_binary_paths(input_cfg)
            isempty(paths) && throw(ArgumentError("[input] resolved to an empty binary list."))
            binary_paths_ref[] = paths
        end
        for path in binary_paths_ref[]
            isfile(path) || push!(errors, "[input] resolved path does not exist: $(path)")
        end
    end

    ft_ref = Ref{Union{Nothing, DataType}}(nothing)
    _capture_config_error!(errors) do
        ft_ref[] = _cfg_float_type(cfg)
    end
    _capture_config_error!(errors) do
        backend = _cfg_runtime_backend(cfg)
        ft_ref[] === nothing || assert_backend_float_type!(backend, ft_ref[])
    end

    tracers_cfg = get(cfg, "tracers", nothing)
    if tracers_cfg !== nothing
        if !(tracers_cfg isa AbstractDict)
            push!(errors, "[tracers] must be a TOML table of tracer subtables.")
        elseif isempty(tracers_cfg)
            push!(errors, "[tracers] was provided but contains no tracer subtables.")
        end
    end

    for (parser, section) in ((advection_spec, _advection_section),
                              (diffusion_spec, _diffusion_section),
                              (convection_spec, _convection_section),
                              (chemistry_spec, _chemistry_section))
        _capture_config_error!(errors) do
            parser(section(cfg))
        end
    end
    _capture_config_error!(errors) do
        runtime_output_spec(get(cfg, "output", Dict{String, Any}()),
                            something(ft_ref[], Float64))
    end
    _capture_config_error!(errors) do
        _parse_tracer_specs(cfg)
    end
    _check_run_window_bounds!(cfg, errors)
    return isempty(errors), errors
end
