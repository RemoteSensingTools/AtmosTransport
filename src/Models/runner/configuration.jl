# ===========================================================================
# TOML parsing — tracer specs (hoisted from run_transport_binary.jl:57-100)
# ===========================================================================

"""
    TransportTracerSpec

Validated runtime tracer configuration: tracer name, initial-condition
configuration, and optional surface-flux configuration.
"""
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
# Execution architecture
# ===========================================================================

@inline _cfg_architecture_section(cfg) = get(cfg, "architecture", Dict{String, Any}())
@inline _cfg_architecture(cfg) = architecture_from_config(_cfg_architecture_section(cfg))

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
shape, and basic run-window bounds. It does not open binary readers or allocate
model state; topology and payload capability checks still run when
`run_driven_simulation` inspects the first binary.
"""
function validate_config(cfg::AbstractDict)
    errors = String[]

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
        arch = _cfg_architecture(cfg)
        ft_ref[] === nothing || assert_float_type!(arch, ft_ref[])
    end

    tracers_cfg = get(cfg, "tracers", nothing)
    if tracers_cfg !== nothing
        if !(tracers_cfg isa AbstractDict)
            push!(errors, "[tracers] must be a TOML table of tracer subtables.")
        elseif isempty(tracers_cfg)
            push!(errors, "[tracers] was provided but contains no tracer subtables.")
        end
    end

    _check_run_window_bounds!(cfg, errors)
    return isempty(errors), errors
end

@inline _ansi_enabled() =
    get(ENV, "NO_COLOR", "") == "" && get(ENV, "TERM", "dumb") != "dumb"

@inline function _ansi_style(text::AbstractString, code::AbstractString)
    return _ansi_enabled() ? string("\e[", code, "m", text, "\e[0m") : String(text)
end

@inline _bold(text::AbstractString) = _ansi_style(text, "1")
@inline _cyan(text::AbstractString) = _ansi_style(text, "1;36")

_advection_label(scheme) = String(nameof(typeof(scheme)))
_advection_label(::LinRoodPPMScheme{ORD}) where ORD = "Lin-Rood PPM$(ORD)"
_advection_label(::PPMScheme) = "PPM"
_advection_label(::SlopesScheme) = "Slopes"
_advection_label(::UpwindScheme) = "Upwind"

_diffusion_label(op) = String(nameof(typeof(op)))
