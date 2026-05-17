abstract type AbstractOutputSchedule end
abstract type AbstractOutputPartition end

struct ExplicitSnapshotSchedule <: AbstractOutputSchedule
    hours::Vector{Float64}
end

struct IntervalSnapshotSchedule <: AbstractOutputSchedule
    start_hour::Float64
    interval_hours::Float64
    stop_hour::Float64
end

struct SingleOutputFile <: AbstractOutputPartition end
struct DailyOutputFiles <: AbstractOutputPartition end

struct RuntimeOutputSpec{S <: AbstractOutputSchedule,
                         P <: AbstractOutputPartition,
                         O <: SnapshotWriteOptions}
    path::String
    schedule::S
    partition::P
    options::O
    enabled::Bool
end

function ExplicitSnapshotSchedule(hours::AbstractVector{<:Real})
    out = sort!(Float64.(hours))
    any(!isfinite, out) &&
        throw(ArgumentError("output snapshot hours must be finite"))
    return ExplicitSnapshotSchedule(out)
end

function IntervalSnapshotSchedule(; start_hour::Real=0.0,
                                  interval_hours::Real,
                                  stop_hour::Real)
    start = Float64(start_hour)
    interval = Float64(interval_hours)
    stop = Float64(stop_hour)
    isfinite(start) || throw(ArgumentError("output start_hour must be finite, got $(start_hour)"))
    isfinite(interval) && interval > 0 ||
        throw(ArgumentError("output cadence/interval hours must be finite and positive, got $(interval_hours)"))
    isfinite(stop) && stop >= start ||
        throw(ArgumentError("output stop_hour must be finite and >= start_hour, got $(stop_hour)"))
    return IntervalSnapshotSchedule(start, interval, stop)
end

function snapshot_hours(schedule::ExplicitSnapshotSchedule)
    return copy(schedule.hours)
end

function snapshot_hours(schedule::IntervalSnapshotSchedule)
    return collect(schedule.start_hour:schedule.interval_hours:schedule.stop_hour)
end

snapshot_hours(spec::RuntimeOutputSpec) = snapshot_hours(spec.schedule)
_schedule_has_times(schedule::ExplicitSnapshotSchedule) = !isempty(schedule.hours)
_schedule_has_times(schedule::IntervalSnapshotSchedule) = schedule.stop_hour >= schedule.start_hour
output_enabled(spec::RuntimeOutputSpec) =
    spec.enabled && !isempty(spec.path) && _schedule_has_times(spec.schedule)
output_path(spec::RuntimeOutputSpec) = spec.path
output_split(::RuntimeOutputSpec{<:Any, SingleOutputFile}) = :single
output_split(::RuntimeOutputSpec{<:Any, DailyOutputFiles}) = :daily

function _output_options(output_cfg::AbstractDict, ::Type{FT}) where FT <: AbstractFloat
    return SnapshotWriteOptions(float_type = FT,
                                deflate_level = Int(get(output_cfg, "deflate_level", 0)),
                                shuffle = Bool(get(output_cfg, "shuffle", true)))
end

function _output_path(output_cfg::AbstractDict, default_path::AbstractString)
    raw = get(output_cfg, "path",
              get(output_cfg, "snapshot_file",
                  get(output_cfg, "filename", default_path)))
    return expand_data_path(String(raw))
end

function _output_partition(output_cfg::AbstractDict)
    split = lowercase(String(get(output_cfg, "split", "single")))
    if split in ("single", "one_file", "one-file")
        return SingleOutputFile()
    elseif split in ("daily", "per_day", "per-day", "day")
        return DailyOutputFiles()
    else
        throw(ArgumentError("[output].split must be \"single\" or \"daily\", got $(repr(split))"))
    end
end

function _output_schedule(output_cfg::AbstractDict;
                          default_cap_hours::Real,
                          fallback_hours::AbstractVector{<:Real})
    if haskey(output_cfg, "snapshot_hours")
        return ExplicitSnapshotSchedule(output_cfg["snapshot_hours"])
    elseif haskey(output_cfg, "hours")
        return ExplicitSnapshotSchedule(output_cfg["hours"])
    end

    cadence_hours = if haskey(output_cfg, "snapshot_interval_hours")
        Float64(output_cfg["snapshot_interval_hours"])
    elseif haskey(output_cfg, "cadence_hours")
        Float64(output_cfg["cadence_hours"])
    elseif haskey(output_cfg, "interval_hours")
        Float64(output_cfg["interval_hours"])
    elseif haskey(output_cfg, "cadence_seconds")
        Float64(output_cfg["cadence_seconds"]) / 3600.0
    elseif haskey(output_cfg, "interval_seconds")
        Float64(output_cfg["interval_seconds"]) / 3600.0
    else
        nothing
    end
    if cadence_hours !== nothing
        start_hour = Float64(get(output_cfg, "start_hour", 0.0))
        stop_hour = Float64(get(output_cfg, "stop_hour", default_cap_hours))
        return IntervalSnapshotSchedule(start_hour = start_hour,
                                        interval_hours = cadence_hours,
                                        stop_hour = stop_hour)
    end
    return ExplicitSnapshotSchedule(fallback_hours)
end

"""
    runtime_output_spec(output_cfg, FT; default_path="", default_cap_hours=8760, fallback_hours=[])

Parse the run-time output contract from TOML-compatible `[output]` settings.

Preferred keys are `path`, `cadence_hours` or `hours`, and
`split = "single" | "daily"`. Legacy `snapshot_file`,
`snapshot_hours`, and `snapshot_interval_hours` remain accepted.
"""
function runtime_output_spec(output_cfg::AbstractDict, ::Type{FT};
                             default_path::AbstractString = "",
                             default_cap_hours::Real = 8760.0,
                             fallback_hours::AbstractVector{<:Real} = Float64[]) where
        {FT <: AbstractFloat}
    schedule = _output_schedule(output_cfg;
                                default_cap_hours = default_cap_hours,
                                fallback_hours = fallback_hours)
    partition = _output_partition(output_cfg)
    options = _output_options(output_cfg, FT)
    enabled = Bool(get(output_cfg, "enabled", true))
    path = _output_path(output_cfg, default_path)
    return RuntimeOutputSpec(path, schedule, partition, options, enabled)
end

function _insert_suffix_before_extension(path::AbstractString, suffix::AbstractString)
    root, ext = splitext(String(path))
    return isempty(ext) ? string(root, suffix, ".nc") : string(root, suffix, ext)
end

function output_path_for_day(spec::RuntimeOutputSpec, date_label::AbstractString,
                             day_index::Integer)
    path = spec.path
    day = lpad(string(day_index), 3, '0')
    if occursin("{date}", path) || occursin("{YYYYMMDD}", path) || occursin("{day}", path)
        out = replace(path, "{date}" => date_label)
        out = replace(out, "{YYYYMMDD}" => date_label)
        return replace(out, "{day}" => day)
    end
    label = isempty(date_label) ? day : date_label
    return _insert_suffix_before_extension(path, "_" * label)
end
