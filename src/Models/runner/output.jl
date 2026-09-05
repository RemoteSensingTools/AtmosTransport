function _output_display_path(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? output_path(spec) : "(disabled)"
end

function _output_basename(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? basename(output_path(spec)) : "(disabled)"
end

function _binary_date_label(path::AbstractString)
    m = match(r"(\d{8})", basename(path))
    return m === nothing ? "" : String(m.captures[1])
end

function _output_default_cap_hours(driver, binary_count::Integer;
                                   start_window::Integer = 1,
                                   stop_window_override = nothing)
    stop_window = stop_window_override === nothing ?
                  total_windows(driver) :
                  min(Int(stop_window_override), total_windows(driver))
    nw = max(stop_window - start_window + 1, 0)
    return Float64(nw * Int(binary_count)) * Float64(window_dt(driver)) / 3600.0
end

_output_path_for_partition(spec::RuntimeOutputSpec, ::SingleOutputFile,
                           ::AbstractString, ::Integer) = output_path(spec)
_output_path_for_partition(spec::RuntimeOutputSpec, ::DailyOutputFiles,
                           date_label::AbstractString, day_index::Integer) =
    output_path_for_day(spec, date_label, day_index)

function _push_snapshot_frame!(::SingleOutputFile,
                               snapshots::Vector{SnapshotFrame},
                               ::Vector{SnapshotFrame},
                               frame::SnapshotFrame)
    push!(snapshots, frame)
    return nothing
end

function _push_snapshot_frame!(::DailyOutputFiles,
                               ::Vector{SnapshotFrame},
                               day_snapshots::Vector{SnapshotFrame},
                               frame::SnapshotFrame)
    push!(day_snapshots, frame)
    return nothing
end

function _write_output_frames!(timer::RunProgressTimer,
                               spec::RuntimeOutputSpec,
                               partition::AbstractOutputPartition,
                               frames::Vector{SnapshotFrame},
                               grid;
                               mass_basis::Symbol,
                               date_label::AbstractString = "",
                               day_index::Integer = 1)
    output_enabled(spec) || return nothing
    isempty(frames) && return nothing
    path = _output_path_for_partition(spec, partition, date_label, day_index)
    timed_io_write!(timer, () -> if spec.format === :binary_mmap
        write_snapshot_binary(path, frames, grid;
                              mass_basis = mass_basis,
                              options = spec.options)
    else
        write_snapshot_netcdf(path, frames, grid;
                              mass_basis = mass_basis,
                              options = spec.options,
                              fields = spec.fields)
    end)
    return path
end

# Write accumulated HOST-side snapshot frames to disk. Used by the async
# daily-flush path (Threads.@spawn): runs off the main loop so the next day's
# GPU transport overlaps the disk write. Deliberately does NOT touch the run
# timer (the overlapped write is not charged to wall io_write) and never touches
# GPU memory (frames are `Array(...)` copies captured at snapshot time).
function _write_frames_to_disk(spec::RuntimeOutputSpec, path::AbstractString,
                               frames::Vector{SnapshotFrame}, grid, mass_basis::Symbol)
    isempty(frames) && return path
    if spec.format === :binary_mmap
        write_snapshot_binary(path, frames, grid; mass_basis = mass_basis,
                              options = spec.options)
    else
        write_snapshot_netcdf(path, frames, grid; mass_basis = mass_basis,
                              options = spec.options, fields = spec.fields)
    end
    return path
end

_flush_daily_output!(::SingleOutputFile, timer, spec, frames, grid;
                     mass_basis, date_label, day_index) = nothing

function _flush_daily_output!(partition::DailyOutputFiles, timer, spec, frames, grid;
                              mass_basis, date_label, day_index)
    isempty(frames) && return nothing
    written = _write_output_frames!(timer, spec, partition, frames, grid;
                                    mass_basis = mass_basis,
                                    date_label = date_label,
                                    day_index = day_index)
    empty!(frames)
    return written
end

_flush_single_output!(::DailyOutputFiles, timer, spec, frames, grid;
                      mass_basis) = nothing

function _flush_single_output!(partition::SingleOutputFile, timer, spec, frames, grid;
                               mass_basis)
    return _write_output_frames!(timer, spec, partition, frames, grid;
                                 mass_basis = mass_basis)
end

"""
    _assert_gpu_residency!(state, arch)

See `feedback_verify_gpu_runs_on_gpu`. When a GPU backend is
selected, assert that `state.air_mass` lives on that backend. A silent CPU
fallback aborts with a precise error. Called once after model construction,
before the run loop.
"""
