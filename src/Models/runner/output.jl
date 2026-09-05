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
function _diffusion_label(op::ImplicitVerticalDiffusion)
    coupling = uses_diffusive_surface_flux_boundary(op) ? ", surface_flux=boundary" :
               ", surface_flux=split"
    return string(nameof(typeof(op)), coupling)
end

function _schedule_label(driver)
    schedule = steps_per_window_schedule(driver)
    if isempty(schedule)
        return "n/a"
    end
    lo, hi = extrema(schedule)
    if lo == hi
        return string(first(schedule))
    end
    return @sprintf("%d..%d, max=%d", lo, hi, steps_per_window(driver))
end

function _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                  backend, FT, recipe, driver, tracers,
                                  binary_count, snapshot_file)
    scheme = _cyan(_advection_label(recipe.advection))
    return (
        @sprintf("%s", _bold(String(topology))),
        @sprintf("|-- grid:      %s, levels=%d, Hp=%d",
                 mesh_label, levels, halo_width),
        @sprintf("|-- numerics:  scheme=%s, FT=%s, backend=%s",
                 scheme, FT, backend),
        @sprintf("|-- physics:   diffusion=%s, convection=%s",
                 _diffusion_label(recipe.diffusion),
                 nameof(typeof(recipe.convection))),
        @sprintf("|-- schedule:  window_dt=%.0fs, steps/window=%s, binaries=%d",
                 Float64(window_dt(driver)), _schedule_label(driver),
                 binary_count),
        @sprintf("|-- tracers:   %s", join(String.(tracers), ", ")),
        @sprintf("`-- output:    %s", snapshot_file),
    )
end

function _log_runtime_summary(; topology, mesh_label, levels, halo_width,
                                backend, FT, recipe, driver, tracers,
                                binary_count, snapshot_file)
    lines = _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                   backend, FT, recipe, driver, tracers,
                                   binary_count, snapshot_file)
    @info "Driven runtime\n" * join(lines, "\n")
end

function _output_display_path(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? output_path(spec) : "(disabled)"
end

function _output_basename(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? basename(output_path(spec)) : "(disabled)"
end

function _binary_date_label(path::AbstractString)
    m = match(r"(\d{8})", basename(path))
    return m === nothing ? "" : String(something(m.captures[1], ""))
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
                               snapshots::AbstractVector{<:AbstractSnapshotFrame},
                               ::AbstractVector{<:AbstractSnapshotFrame},
                               frame::AbstractSnapshotFrame)
    push!(snapshots, frame)
    return nothing
end

# The outer run owns this resource, including exceptional exits from either topology.
mutable struct RunSnapshotOutput
    stream::Union{Nothing,NetCDFSnapshotStream}
end
RunSnapshotOutput() = RunSnapshotOutput(nothing)
function Base.close(output::RunSnapshotOutput)
    stream = output.stream
    stream === nothing || close(stream)
    return nothing
end

function _single_netcdf_stream(output::RunSnapshotOutput, spec::RuntimeOutputSpec, grid; mass_basis)
    output_enabled(spec) && spec.format === :netcdf && spec.partition isa SingleOutputFile ||
        return nothing
    output.stream = NetCDFSnapshotStream(output_path(spec), grid;
                                         mass_basis, options=spec.options, fields=spec.fields)
    return output.stream
end

_record_snapshot!(::Nothing, partition, snapshots, day_snapshots, frame) =
    _push_snapshot_frame!(partition, snapshots, day_snapshots, frame)
_record_snapshot!(stream::NetCDFSnapshotStream, partition, snapshots, day_snapshots, frame) =
    append_snapshot!(stream, frame)

function _push_snapshot_frame!(::DailyOutputFiles,
                               ::AbstractVector{<:AbstractSnapshotFrame},
                               day_snapshots::AbstractVector{<:AbstractSnapshotFrame},
                               frame::AbstractSnapshotFrame)
    push!(day_snapshots, frame)
    return nothing
end

function _write_output_frames!(timer::RunProgressTimer,
                               spec::RuntimeOutputSpec,
                               partition::AbstractOutputPartition,
                               frames::AbstractVector{<:AbstractSnapshotFrame},
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
                               frames::AbstractVector{<:AbstractSnapshotFrame}, grid, mass_basis::Symbol)
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
