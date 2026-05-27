# ===========================================================================
# Native-GRIB ERA5 reader for the preprocessor.
#
# Consumes the canonical ECMWF download layout produced by
# `config/downloads/era5_native_daily.toml`:
#
#   <root_dir>/ml_an_native_core/era5_core_YYYYMMDD.grib
#   <root_dir>/ml_fc_convection/era5_convection_YYYYMMDD.grib   # gated by include_convection
#   <root_dir>/sfc_an_native/era5_surface_YYYYMMDD.grib          # gated by include_surface
#
# The `core` stream is the model-level analysis bundle (T, Q, VO, LNSP, D).
# T/VO/D/LNSP are stored as spherical-harmonic coefficients (`gridType=sh`),
# while Q is already on the reduced linear-Gaussian mesh (`gridType=reduced_gg`).
# The reader emits source-grid fields on the N-th reduced linear-Gaussian mesh
# selected by the `flavor` type parameter.
#
# Breakpoint A in this file: settings type, file-path resolution, per-day
# handle, and `AbstractMetSettings` trait hooks. The window-level GRIB decoder
# (`read_window!`) and spectral-synthesis plumbing land in subsequent commits.
# ===========================================================================

"""
    AbstractERA5GRIBSettings <: AbstractMetSettings

Abstract supertype for ERA5 native-GRIB sources. Concrete subtypes pick the
source mesh via the `flavor` parameter on [`ERA5GRIBSettings`](@ref).
"""
abstract type AbstractERA5GRIBSettings <: AbstractMetSettings end

"""
    ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings

Typed settings for one ERA5 native-GRIB flavor:

- `flavor = :n320` — reduced linear-Gaussian N320, the default MARS native grid
  for ERA5 analyses (640 longitudes at the equator, 137 hybrid levels).

ERA5 archives store hybrid model levels top-down (k = 1 is the TOA-side cap).
`level_orientation = :top_down` reflects that, matching `era5.toml`. The reader
flips to the project's runtime convention downstream — same path used by the
GEOS-IT bottom-up source.
"""
Base.@kwdef struct ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings
    root_dir             :: String
    include_surface      :: Bool   = false
    include_convection   :: Bool   = false
    include_vdiff_fields :: Bool   = false
    coefficients_file    :: String = "config/era5_L137_coefficients.toml"
    level_orientation    :: Symbol = :top_down
end

const ERA5N320Settings = ERA5GRIBSettings{:n320}

# ---------------------------------------------------------------------------
# Stream layout.
#
# `ERA5_GRIB_STREAMS` keeps the on-disk subdirectory and filename stem for each
# GRIB stream in one place so adding a new flavor (e.g. `:o320`) does not need
# to touch any of the call sites.
# ---------------------------------------------------------------------------

const ERA5_GRIB_STREAMS = (
    core       = (subdir = "ml_an_native_core", stem = "era5_core"),
    convection = (subdir = "ml_fc_convection",  stem = "era5_convection"),
    surface    = (subdir = "sfc_an_native",     stem = "era5_surface"),
)

"""
    era5_grib_path(settings, date, stream) -> String

Resolve the on-disk GRIB path for `stream` on `date`. `stream` must be one of
`:core`, `:convection`, or `:surface`. Existence is *not* checked here — the
caller (typically [`open_era5_day`](@ref)) decides whether a missing file is
fatal or merely "no next-day endpoint available".
"""
function era5_grib_path(settings::AbstractERA5GRIBSettings, date::Date,
                        stream::Symbol)
    hasproperty(ERA5_GRIB_STREAMS, stream) ||
        throw(ArgumentError("unknown ERA5 GRIB stream $(stream); expected one of " *
                            string(propertynames(ERA5_GRIB_STREAMS))))
    layout   = getproperty(ERA5_GRIB_STREAMS, stream)
    datestr  = Dates.format(date, "yyyymmdd")
    filename = "$(layout.stem)_$(datestr).grib"
    return joinpath(settings.root_dir, layout.subdir, filename)
end

# ---------------------------------------------------------------------------
# Day-handle container.
#
# GRIB.jl iterators are forward-only and cheap to reopen, so the handle holds
# resolved paths rather than pinned `GribFile` objects. This makes
# `close_day!` a no-op (idempotent) and avoids leaking file descriptors when
# the reader is restarted mid-day.
# ---------------------------------------------------------------------------

"""
    ERA5GRIBDayHandles{S<:AbstractERA5GRIBSettings}

Per-day source-file context. Carries the resolved on-disk paths to the day's
GRIB streams plus the optional next-day `core` path that supplies the right
endpoint of the last window.

`convection_path` is `nothing` unless `settings.include_convection` is set;
likewise `surface_path` is `nothing` unless `settings.include_surface` is set.
`next_core_path` is `nothing` either when the caller passed
`next_day_handle=false` or when the next-day file is not on disk (last day of
the available archive).
"""
struct ERA5GRIBDayHandles{S <: AbstractERA5GRIBSettings}
    settings        :: S
    date            :: Date
    core_path       :: String
    convection_path :: Union{Nothing, String}
    surface_path    :: Union{Nothing, String}
    next_core_path  :: Union{Nothing, String}
end

const ERA5_VALID_LEVEL_ORIENTATIONS = (:top_down, :bottom_up)

"""
    open_era5_day(settings, date; next_day_handle=true) -> ERA5GRIBDayHandles

Resolve the GRIB stream paths for `date` and assert that today's required
files are on disk. When `next_day_handle=true` and `<date+1>` has a `core`
GRIB available, records its path so the last window's right endpoint can be
read from the next day's hour-0 fields.

Errors are explicit about which file is missing so a misconfigured `root_dir`
or an incomplete download is immediately visible.
"""
function open_era5_day(settings::AbstractERA5GRIBSettings, date::Date;
                       next_day_handle::Bool = true)
    settings.level_orientation in ERA5_VALID_LEVEL_ORIENTATIONS ||
        throw(ArgumentError("ERA5 level_orientation must be one of " *
                            join(ERA5_VALID_LEVEL_ORIENTATIONS, ", ") *
                            "; got :$(settings.level_orientation)"))

    core_path = era5_grib_path(settings, date, :core)
    isfile(core_path) ||
        error("ERA5 core GRIB not found: $core_path")

    convection_path = nothing
    if settings.include_convection
        candidate = era5_grib_path(settings, date, :convection)
        isfile(candidate) ||
            error("ERA5 convection GRIB not found: $candidate " *
                  "(include_convection=true)")
        convection_path = candidate
    end

    surface_path = nothing
    if settings.include_surface
        candidate = era5_grib_path(settings, date, :surface)
        isfile(candidate) ||
            error("ERA5 surface GRIB not found: $candidate " *
                  "(include_surface=true)")
        surface_path = candidate
    end

    next_core_path = nothing
    if next_day_handle
        candidate = era5_grib_path(settings, date + Day(1), :core)
        next_core_path = isfile(candidate) ? candidate : nothing
    end

    return ERA5GRIBDayHandles{typeof(settings)}(
        settings, date,
        core_path, convection_path, surface_path, next_core_path)
end

"""
    close_era5_day!(handles::ERA5GRIBDayHandles)

Release any resources held by the day handle. Idempotent — safe to call from
a `finally` block. This is a no-op today because the handle only stores paths,
but the symbol exists so future flavors that pin file descriptors don't change
the call surface.
"""
close_era5_day!(::ERA5GRIBDayHandles) = nothing

# ---------------------------------------------------------------------------
# AbstractMetSettings trait hooks.
# ---------------------------------------------------------------------------

open_day(settings::AbstractERA5GRIBSettings, date::Date;
         next_day_handle::Bool = true) =
    open_era5_day(settings, date; next_day_handle = next_day_handle)

close_day!(handles::ERA5GRIBDayHandles) = close_era5_day!(handles)

windows_per_day(::AbstractERA5GRIBSettings, ::Date) = 24

has_surface(s::AbstractERA5GRIBSettings)      = s.include_surface
has_convection(s::AbstractERA5GRIBSettings)   = s.include_convection
has_vdiff_fields(s::AbstractERA5GRIBSettings) = s.include_vdiff_fields
