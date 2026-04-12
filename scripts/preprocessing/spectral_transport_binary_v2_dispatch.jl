# Config-driven ERA5 spectral -> transport-binary v2 preprocessing interface.
#
# Audience:
# - users call `build_spectral_transport_binary_v2_target` and
#   `run_spectral_transport_binary_v2_preprocessor`
# - contributors add new config-driven targets by subtyping
#   `AbstractSpectralTransportBinaryV2Target`
#
# Stability: this is the stable in-repo API for the ERA5 spectral day-builder
# preprocessors. It is a separate layer from the binary-in / binary-out target
# API in `transport_binary_v2_dispatch.jl`.

using Dates
using Printf
using TOML

"""
    AbstractSpectralTransportBinaryV2Target

Stable repo-level target interface for config-driven ERA5 spectral
transport-binary v2 preprocessors.

Targets are constructed from a TOML config plus optional CLI-style arguments
such as `--day YYYY-MM-DD`.
"""
abstract type AbstractSpectralTransportBinaryV2Target end

"""
    target_dates(target) -> Vector{Date}

Return the processing dates scheduled for this spectral preprocessing target.
"""
target_dates(target::AbstractSpectralTransportBinaryV2Target) =
    error("target_dates not implemented for $(typeof(target))")

"""
    target_summary(target) -> String

Return a short user-facing summary for logs and tests.
"""
target_summary(target::AbstractSpectralTransportBinaryV2Target) = string(typeof(target))

"""
    log_spectral_transport_binary_v2_configuration(target)

Contributor hook for the startup configuration summary.
"""
log_spectral_transport_binary_v2_configuration(target::AbstractSpectralTransportBinaryV2Target) =
    error("log_spectral_transport_binary_v2_configuration not implemented for $(typeof(target))")

"""
    process_spectral_transport_binary_v2_day(target, date)

Contributor hook that processes one day and returns the target-specific result.
"""
process_spectral_transport_binary_v2_day(target::AbstractSpectralTransportBinaryV2Target,
                                         date::Date) =
    error("process_spectral_transport_binary_v2_day not implemented for $(typeof(target))")

"""
    build_spectral_transport_binary_v2_target(::Val{kind}, config_path, cfg, argv, grid, settings, vertical, dates; FT=Float64)

Internal extension hook for config-driven spectral preprocessing target
families.

Users should call `build_spectral_transport_binary_v2_target(config_path,
argv; FT=...)`. Contributors add new targets by specializing this method for
the relevant `grid_kind(grid)`.
"""
build_spectral_transport_binary_v2_target(::Val{kind},
                                          config_path::AbstractString,
                                          cfg,
                                          argv::AbstractVector{<:AbstractString},
                                          grid,
                                          settings,
                                          vertical,
                                          dates;
                                          FT::Type{T} = Float64) where {kind, T <: AbstractFloat} =
    error("Unsupported ERA5 spectral transport-binary v2 target kind: :$kind")

"""
    build_spectral_transport_binary_v2_target(config_path, argv=String[]; FT=Float64) -> target

Stable front door for config-driven ERA5 spectral -> transport-binary v2
preprocessors.

`config_path` points at the TOML file, and `argv` carries any remaining
CLI-style arguments such as `["--day", "2021-12-01"]`.
"""
function build_spectral_transport_binary_v2_target(config_path::AbstractString,
                                                   argv::AbstractVector{<:AbstractString}=String[];
                                                   FT::Type{T} = Float64) where T <: AbstractFloat
    expanded_path = expanduser(String(config_path))
    isfile(expanded_path) || error("Config not found: $expanded_path")

    cfg = TOML.parsefile(expanded_path)
    day_filter = parse_day_filter(argv)
    grid = build_target_geometry(cfg["grid"], FT)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range, settings.min_dp, cfg["grid"])
    dates = select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)

    return build_spectral_transport_binary_v2_target(
        Val(grid_kind(grid)),
        expanded_path,
        cfg,
        argv,
        grid,
        settings,
        vertical,
        dates;
        FT=FT,
    )
end

"""
    run_spectral_transport_binary_v2_preprocessor(target) -> Vector

Generic day-loop driver for the config-driven ERA5 spectral preprocessing
targets.
"""
function run_spectral_transport_binary_v2_preprocessor(target::AbstractSpectralTransportBinaryV2Target)
    log_spectral_transport_binary_v2_configuration(target)

    dates = target_dates(target)
    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))

    t_total = time()
    results = Vector{Any}(undef, length(dates))
    for (i, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", i, length(dates), date)
        results[i] = process_spectral_transport_binary_v2_day(target, date)
    end

    wall_total = round(time() - t_total, digits=1)
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), wall_total,
                   length(dates) > 0 ? wall_total / length(dates) : 0.0)
    return results
end
