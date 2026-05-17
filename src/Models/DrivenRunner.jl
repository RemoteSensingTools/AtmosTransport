"""
    DrivenRunner

Library-level entry point for the driven transport runtime.

Plan 40 Commit 6a hoists the LL/RG runner here from
`scripts/run_transport_binary.jl` so both the old script (now a thin
shim) and the forthcoming unified `scripts/run_transport.jl` share
one implementation. Commit 6b will fold the CS-specific flow from
`scripts/run_cs_driven.jl` into the same top-level
`run_driven_simulation(cfg)` with dispatch driven by the first
binary's header (`inspect_binary(first_path).grid_type`).

## Ownership boundary

- **Binary header** (`grid_type`, `mass_basis`, `payload_sections`,
  panel convention) — authoritative for topology and capability.
  Accessed here via `binary_capabilities(driver.reader)` and
  `air_mass_basis(driver)`.
- **TOML `[input]`** — either an explicit `binary_paths = [...]`
  list (Shape A) or a `folder + start_date + end_date
  (+ file_pattern)` block (Shape B). Both are resolved to a sorted
  `Vector{String}` by `expand_binary_paths`.
- **TOML physics** (`[advection]` / `[diffusion]` / `[convection]`)
  — validated against binary capabilities by
  `validate_runtime_physics_recipe` / `build_runtime_physics_recipe`
  before the loop.
- **TOML `[tracers.*]`** — tracer specs consumed via
  `build_initial_mixing_ratio` + `pack_initial_tracer_mass`
  (basis-aware per `feedback_vmr_to_mass_basis_aware`) and
  `build_surface_flux_sources`.

## GPU residency (feedback_verify_gpu_runs_on_gpu)

When `[architecture].use_gpu = true` or `backend` selects a GPU, the runner
asserts that `state.air_mass` lives on the selected backend after model
construction and prints a `[gpu verified] …` line. A silent CPU fallback
aborts the run with a precise error message.
"""
module DrivenRunner

using Adapt
using Printf: @sprintf, @printf
using Logging
using ProgressMeter: Progress, next!, finish!, update!

import ...expand_data_path
using ...SectionTimer
using ..State: AbstractMassBasis, DryBasis, MoistBasis, CellState,
                CubedSphereState, total_air_mass, total_mass, tracer_names,
                tracer_index, get_tracer
using ..Grids: nlevels
using ..Operators: LinRoodPPMScheme, PPMScheme, SlopesScheme, UpwindScheme
using ..Architectures: CPU, GPU,
                       runtime_backend_from_config, is_gpu_backend,
                       ensure_backend_runtime!, backend_array_adapter,
                       backend_label, backend_device_name, backend_name,
                       synchronize_backend!, assert_backend_residency!,
                       assert_backend_float_type!
using ..MetDrivers: TransportBinaryDriver, CubedSphereTransportDriver,
                     load_transport_window, driver_grid, air_mass_basis,
                     total_windows, window_dt, binary_capabilities,
                     inspect_binary, steps_per_window,
                     steps_per_window_schedule
using ..InitialConditionIO: build_initial_mixing_ratio,
                             pack_initial_tracer_mass,
                             build_surface_flux_sources
using ..BinaryPathExpander: expand_binary_paths
using ..Output: SnapshotFrame,
                RuntimeOutputSpec, runtime_output_spec, snapshot_hours,
                output_enabled, output_path, output_split, output_path_for_day,
                capture_snapshot, write_snapshot_netcdf
# TransportModel + DrivenSimulation live alongside us in the Models module;
# reach up to the parent and pull them in.
using ..Models: TransportModel
import ..Models: DrivenSimulation, run_window!, run!, step!, allocate_face_fluxes
# Physics-recipe helpers: `build_runtime_physics_recipe` /
# `validate_runtime_physics_recipe` are defined in `CSPhysicsRecipe.jl`
# (loaded before us in Models). Pull them in so we don't have to stutter
# through `Main.AtmosTransport.*`.
using ..Models: build_runtime_physics_recipe, validate_runtime_physics_recipe,
                 configured_halo_width, build_cs_advection

export run_driven_simulation, TransportTracerSpec

# ===========================================================================
# Forward-run progress timer — Transport vs IO wall-clock breakdown.
#
# Mirrors the `main:src/Models/run_loop.jl:105-150` pattern at coarser
# granularity: three accumulators (driver-open / transport / snapshot
# capture+write) plus a `ProgressMeter.Progress` bar over windows. Always
# on — no env var gating, no SectionTimer dep. End-of-run summary lands
# via `@info` so it surfaces alongside the existing run-completion logs.
# ===========================================================================

mutable struct RunProgressTimer
    prog            :: Progress
    t_start         :: Float64
    t_io_read       :: Float64   # TransportBinaryDriver open + window loads
    t_transport     :: Float64   # advection + diffusion + convection + emissions
    t_io_write      :: Float64   # snapshot capture + final NetCDF write
    windows_total   :: Int
    status_line     :: String
    detail_line     :: String
end

RunProgressTimer(total_windows::Integer; label::AbstractString = "Forward run ") =
    RunProgressTimer(
        Progress(max(Int(total_windows), 1);
                  desc = label, showspeed = true, barlen = 40),
        time(), 0.0, 0.0, 0.0, Int(total_windows),
        "initializing", "transport 0.0s | io_read 0.0s | io_write 0.0s")

@inline function _timed!(field::Symbol, timer::RunProgressTimer, f)
    t0 = time()
    val = f()
    delta = time() - t0
    setproperty!(timer, field, getproperty(timer, field) + delta)
    return val
end

# Mark IO read (e.g. opening a daily binary driver).
@inline timed_io_read!(timer, f) = _timed!(:t_io_read, timer, f)

# Mark transport (a single `run_window!` / `step!` block).
@inline timed_transport!(timer, f) = _timed!(:t_transport, timer, f)

# Mark IO write (snapshot capture + final NetCDF write).
@inline timed_io_write!(timer, f) = _timed!(:t_io_write, timer, f)

function _progress_detail_line(timer::RunProgressTimer)
    wall = max(time() - timer.t_start, eps())
    return @sprintf("transport %.1fs (%4.1f%%) | io_read %.1fs | io_write %.1fs | wall %.1fs",
                    timer.t_transport, 100 * timer.t_transport / wall,
                    timer.t_io_read, timer.t_io_write, wall)
end

@inline function _progress_showvalues(timer::RunProgressTimer)
    detail = isempty(timer.detail_line) ?
             _progress_detail_line(timer) :
             string(_progress_detail_line(timer), " | ", timer.detail_line)
    return [(:status, timer.status_line), (:timing, detail)]
end

function set_progress_status!(timer::RunProgressTimer;
                              status::Union{Nothing, AbstractString} = nothing,
                              detail::Union{Nothing, AbstractString} = nothing,
                              redraw::Bool = false)
    status === nothing || (timer.status_line = String(status))
    detail === nothing || (timer.detail_line = String(detail))
    redraw && update!(timer.prog, timer.prog.counter;
                      showvalues = _progress_showvalues(timer))
    return timer
end

# Tick the progress bar after one window has advanced. Keep routine runtime
# status in the two redrawable lines below the bar so `@info` output does not
# interrupt ETA/progress rendering during long runs.
@inline function tick_window!(timer::RunProgressTimer;
                              status::Union{Nothing, AbstractString} = nothing,
                              detail::Union{Nothing, AbstractString} = nothing)
    status === nothing || (timer.status_line = String(status))
    detail === nothing || (timer.detail_line = String(detail))
    next!(timer.prog; showvalues = [
        (:status, timer.status_line),
        (:timing, string(_progress_detail_line(timer), " | ", timer.detail_line)),
    ])
end

function summarize_progress!(timer::RunProgressTimer)
    finish!(timer.prog)
    wall = time() - timer.t_start
    accounted = timer.t_io_read + timer.t_transport + timer.t_io_write
    other = max(wall - accounted, 0.0)
    w = max(wall, eps())
    msg = @sprintf("Forward run wall %.1fs   transport %.1fs (%.1f%%)   io_read %.1fs (%.1f%%)   io_write %.1fs (%.1f%%)   other %.1fs (%.1f%%)", wall, timer.t_transport, 100*timer.t_transport/w, timer.t_io_read, 100*timer.t_io_read/w, timer.t_io_write, 100*timer.t_io_write/w, other, 100*other/w)
    @info msg
    return timer
end

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
                "sigma_lat_deg", "amplitude", "file", "variable", "time_index")
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
                 nameof(typeof(recipe.diffusion)),
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

function _write_output_frames!(timer::RunProgressTimer,
                               spec::RuntimeOutputSpec,
                               frames::Vector{SnapshotFrame},
                               grid;
                               mass_basis::Symbol,
                               date_label::AbstractString = "",
                               day_index::Integer = 1)
    output_enabled(spec) || return nothing
    isempty(frames) && return nothing
    path = output_split(spec) === :daily ?
           output_path_for_day(spec, date_label, day_index) :
           output_path(spec)
    timed_io_write!(timer,
        () -> write_snapshot_netcdf(path, frames, grid;
                                    mass_basis = mass_basis,
                                    options = spec.options,
                                    fields = spec.fields))
    return path
end

function _synchronize_backend!(cfg)
    synchronize_backend!(_cfg_runtime_backend(cfg))
    return nothing
end

"""
    _assert_gpu_residency!(state, cfg)

Plan 40 Commit 6a / `feedback_verify_gpu_runs_on_gpu`. When a GPU backend is
selected, assert that `state.air_mass` lives on that backend. A silent CPU
fallback aborts with a precise error. Called once after model construction,
before the run loop.
"""
function _assert_gpu_residency!(state, cfg)
    backend = _cfg_runtime_backend(cfg)
    is_gpu_backend(backend) || return nothing
    backing = assert_backend_residency!(state.air_mass, backend; label = "state.air_mass")
    wrapper = Base.typename(typeof(backing)).wrapper
    @info @sprintf("[gpu verified] backend=%s backing=%s device=%s",
                   String(backend_name(backend)),
                   String(nameof(wrapper)),
                   backend_device_name(backend))
    return nothing
end

# ===========================================================================
# Model construction (hoisted from run_transport_binary.jl:153-188)
#
# Uses `pack_initial_tracer_mass` (C1b) rather than raw `.* air_mass`:
# bit-exact on DryBasis, errors loudly on MoistBasis without qv
# (correctness rule feedback_vmr_to_mass_basis_aware). No LL/RG config
# in-tree uses MoistBasis, so no behaviour change for shipped configs.
# ===========================================================================

function _make_structured_model(driver::TransportBinaryDriver;
                                FT::Type{<:AbstractFloat},
                                recipe,
                                tracer_specs,
                                cfg)
    grid = driver_grid(driver)
    window = load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)

    tracer_specs_tuple = Tuple(tracer_specs)
    isempty(tracer_specs_tuple) && throw(ArgumentError("at least one tracer must be configured"))

    basis_type = air_mass_basis(driver) == :dry ? DryBasis : MoistBasis
    tracer_names_tup = Tuple(spec.name for spec in tracer_specs_tuple)
    rm_arrays = map(tracer_specs_tuple) do spec
        vmr = build_initial_mixing_ratio(air_mass, grid, spec.init_cfg;
                                         surface_pressure = window.surface_pressure)
        # MoistBasis LL/RG runs would need qv threaded from window.qv —
        # none in-tree today; the packer errors with a precise message.
        return pack_initial_tracer_mass(grid, air_mass, vmr;
                                        mass_basis = basis_type())
    end

    tracer_tuple = NamedTuple{tracer_names_tup}(Tuple(rm_arrays))
    state = CellState(basis_type, air_mass; tracer_tuple...)
    fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid);
                                  FT = FT, basis = basis_type)
    model = TransportModel(state, fluxes, grid, recipe.advection;
                           diffusion = recipe.diffusion,
                           convection = recipe.convection,
                           chemistry = recipe.chemistry)
    adaptor = _backend_array_adapter(cfg)
    return adaptor === Array ? model : Base.invokelatest(Adapt.adapt, adaptor, model)
end

# Snapshot capture and NetCDF writing live in `AtmosTransport.Output`. The
# runner only decides when to sample; the output module owns topology-specific
# diagnostics and file layout.

# ===========================================================================
# Capability validation (plan 40 Commit 6a)
#
# Validate TOML physics against binary capabilities BEFORE constructing the
# model, so users get a precise error up front instead of silently failing
# partway through. Runs after `build_runtime_physics_recipe` (which already
# validates kind strings against recipe types) but before model construction
# (which discovers problems at the first load).
# ===========================================================================

function _validate_capability_match(driver, recipe, cfg)
    caps = binary_capabilities(driver.reader)

    # Convection kind vs binary sections
    conv_kind = Symbol(lowercase(String(get(get(cfg, "convection", Dict()), "kind", "none"))))
    if conv_kind === :tm5 && !caps.tm5_convection
        throw(ArgumentError(
            "[convection] kind = \"tm5\" requires the binary to carry " *
            "entu, detu, entd, detd; this binary's payload_sections are " *
            "$(caps.payload_sections). Regenerate with a TM5-enabled " *
            "preprocessor or set convection.kind = \"none\"."))
    end
    if conv_kind === :cmfmc && !caps.cmfmc_convection
        throw(ArgumentError(
            "[convection] kind = \"cmfmc\" requires the binary to carry " *
            "the cmfmc section; this binary's payload_sections are " *
            "$(caps.payload_sections)."))
    end
    return nothing
end

# ===========================================================================
# run_driven_simulation — top-level entry
# ===========================================================================

function _validate_input_binary_expectations(caps, input_cfg::AbstractDict,
                                             path::AbstractString)
    if haskey(input_cfg, "expected_nlevel")
        expected = Int(input_cfg["expected_nlevel"])
        caps.nlevel == expected || throw(ArgumentError(
            "[input].expected_nlevel=$expected but $(basename(path)) has " *
            "nlevel=$(caps.nlevel). This usually means the run config is " *
            "pointing at an older preprocessing product."))
    end
    if haskey(input_cfg, "required_preprocessor_contract")
        required = String(input_cfg["required_preprocessor_contract"])
        actual = caps.preprocessor_contract
        actual == required || throw(ArgumentError(
            "[input].required_preprocessor_contract=$(repr(required)) but " *
            "$(basename(path)) declares $(repr(actual))."))
    end
    if Bool(get(input_cfg, "require_adaptive_substeps", false))
        caps.adaptive_substeps === true || throw(ArgumentError(
            "[input].require_adaptive_substeps=true but $(basename(path)) " *
            "does not declare adaptive_substeps=true."))
    end
    return nothing
end

"""
    run_driven_simulation(cfg::AbstractDict) -> TransportModel

Run a driven transport simulation from a TOML config. Resolves
`[input]` to a sorted binary list via `expand_binary_paths`, picks
the right driver based on the binary's `grid_type` header field,
validates physics-vs-capability, verifies GPU residency when
requested, runs the loop, optionally captures topology-native diagnostic snapshots
to NetCDF, and returns the terminal `TransportModel`.

Plan 40 Commit 6a supports LL/RG only (structured and
reduced-Gaussian). CS dispatch is added in Commit 6b.
"""
function run_driven_simulation(cfg::AbstractDict)
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) &&
        throw(ArgumentError("[input] resolved to an empty binary list"))
    # TM5-storage Commit 1: section timing instrumentation, off unless
    # ATMOSTR_TIMERS=1. Enabled here so every section accumulator covers
    # the whole driven loop including snapshot capture / write.
    timers_on = SectionTimer.maybe_enable_from_env!()
    # Plan 40 Commit 6b: dispatch on the first binary's grid_type —
    # the ownership boundary (binary header owns topology, TOML owns
    # physics kinds). The capability probe also runs the load-time
    # gates (stale-binary, cm-continuity) as a side effect of opening
    # the reader in `inspect_binary`.
    caps = inspect_binary(first(binary_paths); io = devnull)
    _validate_input_binary_expectations(caps, input_cfg, first(binary_paths))
    result = if caps.grid_type === :cubed_sphere
        _run_driven_simulation_cs(binary_paths, cfg)
    else
        _run_driven_simulation_structured(binary_paths, cfg)
    end
    if timers_on
        SectionTimer.disable!()
        SectionTimer.report(stderr)
        output_cfg = get(cfg, "output", Dict{String, Any}())
        snapshot_file = expand_data_path(String(get(output_cfg, "path",
                                                    get(output_cfg, "snapshot_file",
                                                        get(output_cfg, "filename", "")))))
        if !isempty(snapshot_file)
            csv_path = replace(snapshot_file, r"\.nc$" => "") * ".timings.csv"
            written = SectionTimer.write_csv(csv_path)
            written !== nothing && @info "Section timings → $(written)"
        end
    end
    return result
end

function _run_driven_simulation_structured(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ?
         Float32 : Float64
    assert_backend_float_type!(_cfg_runtime_backend(cfg), FT)
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    reset_air_mass_each_window = Bool(get(run_cfg, "reset_air_mass_each_window", false))

    init_cfg = get(cfg, "init", Dict{String, Any}())
    tracer_specs = something(_parse_tracer_specs(cfg),
                             (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                                  _copy_cfg_dict(init_cfg),
                                                  Dict{String, Any}()),))

    _ensure_gpu_runtime!(cfg)

    # Open first driver, build recipe, validate capability, build model
    first_driver = TransportBinaryDriver(first(binary_paths);
                                          FT = FT,
                                          arch = CPU())
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT;
                                      default_cap_hours = _output_default_cap_hours(
                                          first_driver, length(binary_paths);
                                          start_window = start_window,
                                          stop_window_override = stop_window_override))
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    recipe = build_runtime_physics_recipe(cfg, first_driver, FT)
    _validate_capability_match(first_driver, recipe, cfg)

    model = _make_structured_model(first_driver;
                                    FT = FT, recipe = recipe,
                                    tracer_specs = tracer_specs, cfg = cfg)
    _assert_gpu_residency!(model.state, cfg)

    grid_of_first = driver_grid(first_driver)
    surface_sources = build_surface_flux_sources(grid_of_first, tracer_specs, FT)
    m0 = total_air_mass(model.state)
    tracer_masses0 = Dict(name => total_mass(model.state, name)
                          for name in tracer_names(model.state))
    source_tracers = Set(source.tracer_name for source in surface_sources)

    _log_runtime_summary(topology = :Structured,
                         mesh_label = summary(grid_of_first.horizontal),
                         levels = nlevels(grid_of_first),
                         halo_width = 0,
                         backend = _backend_label(cfg),
                         FT = FT,
                         recipe = recipe,
                         driver = first_driver,
                         tracers = tracer_names(model.state),
                         binary_count = length(binary_paths),
                         snapshot_file = _output_display_path(output_spec))
    for source in surface_sources
        @info @sprintf("Surface source %s total mass rate: %.12e kg/s",
                       String(source.tracer_name),
                       Float64(sum(source.cell_mass_rate)))
    end

    snapshots = SnapshotFrame[]
    day_snapshots = SnapshotFrame[]
    snapshot_count = 0
    snap_idx = 1
    total_elapsed_hours = 0.0

    # Estimate total windows for the progress bar. Each daily binary has
    # the same window count for a homogeneous run; multiplying gives a
    # close-enough total. Use min(stop_window_override, ...) when set.
    per_binary = stop_window_override === nothing ?
                 total_windows(first_driver) - start_window + 1 :
                 Int(stop_window_override) - start_window + 1
    timer = RunProgressTimer(per_binary * length(binary_paths))

    function capture_structured!(hour_total)
        timed_io_write!(timer, () -> begin
            frame = capture_snapshot(model; time_hours = hour_total)
            if output_split(output_spec) === :daily
                push!(day_snapshots, frame)
            else
                push!(snapshots, frame)
            end
        end)
        snapshot_count += 1
        return nothing
    end

    if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
       abs(snapshot_schedule_hours[snap_idx]) < 0.5
        capture_structured!(0.0)
        set_progress_status!(timer;
                             detail = @sprintf("snapshot %d at t=%.0fh",
                                               snap_idx, 0.0),
                             redraw = true)
        snap_idx += 1
    end

    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver :
                 timed_io_read!(timer,
                     () -> TransportBinaryDriver(path; FT = FT, arch = CPU()))
        validate_runtime_physics_recipe(recipe, driver)
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = timed_io_read!(timer,
            () -> DrivenSimulation(model, driver;
                                    start_window = start_window,
                                    stop_window = stop_window,
                                    initialize_air_mass = initialize_air_mass,
                                    reset_air_mass_each_window = reset_air_mass_each_window,
                                    surface_sources = surface_sources,
                                    chemistry = recipe.chemistry))
        model = sim.model
        if !initialize_air_mass
            boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) /
                           max(maximum(abs.(sim.window.air_mass)), eps(FT))
            set_progress_status!(timer;
                                 detail = @sprintf("boundary air-mass mismatch before %s: %.3e",
                                                   basename(path), boundary_rel),
                                 redraw = true)
        end
        window_hours = Float64(window_dt(driver)) / 3600.0
        n_windows = stop_window - start_window + 1
        set_progress_status!(timer;
                             status = @sprintf("running %s with %s on %s (%d windows)",
                                               basename(path),
                                               nameof(typeof(recipe.advection)),
                                               summary(driver_grid(driver).horizontal),
                                               n_windows),
                             detail = "loading first window",
                             redraw = true)
        _synchronize_backend!(cfg)
        t0 = time()

        if do_snapshots
            for _ in 1:n_windows
                timed_transport!(timer, () -> run_window!(sim))
                tick_window!(timer;
                             status = @sprintf("%s window %d/%d  steps/window=%d",
                                               basename(path),
                                               sim.current_window_index,
                                               stop_window,
                                               sim.steps_per_window),
                             detail = @sprintf("snapshots=%d  output=%s",
                                               snapshot_count,
                                               _output_basename(output_spec)))
                total_elapsed_hours += window_hours
                while snap_idx <= length(snapshot_schedule_hours) &&
                      abs(total_elapsed_hours - snapshot_schedule_hours[snap_idx]) < 0.5
                    capture_structured!(total_elapsed_hours)
                    set_progress_status!(timer;
                                         detail = @sprintf("snapshot %d at t=%.0fh  output=%s",
                                                           snap_idx,
                                                           total_elapsed_hours,
                                                           _output_basename(output_spec)),
                                         redraw = true)
                    snap_idx += 1
                end
            end
        else
            timed_transport!(timer, () -> run!(sim))
            total_elapsed_hours += n_windows * window_hours
            # `run!` doesn't tick per window; advance the bar to the
            # binary's window count in one shot.
            for local_win in start_window:stop_window
                tick_window!(timer;
                             status = @sprintf("%s window %d/%d  steps/window=%d",
                                               basename(path), local_win,
                                               stop_window,
                                               sim.steps_per_window_schedule[local_win]),
                             detail = "batch window accounting after run!()")
            end
        end

        _synchronize_backend!(cfg)
        set_progress_status!(timer;
                             status = @sprintf("finished %s", basename(path)),
                             detail = @sprintf("file wall %.2fs", time() - t0),
                             redraw = true)
        if do_snapshots && output_split(output_spec) === :daily && !isempty(day_snapshots)
            written = _write_output_frames!(timer, output_spec, day_snapshots,
                                            driver_grid(first_driver);
                                            mass_basis = air_mass_basis(first_driver),
                                            date_label = _binary_date_label(path),
                                            day_index = idx)
            set_progress_status!(timer;
                                 detail = @sprintf("wrote %s", basename(written)),
                                 redraw = true)
            empty!(day_snapshots)
        end
        close(driver)
    end

    if do_snapshots && output_split(output_spec) === :single && !isempty(snapshots)
        # `air_mass_basis(driver)` already returns the Symbol and has been
        # validated to match `model.state`'s basis by
        # `_check_basis_compatibility` before any step!.
        _write_output_frames!(timer, output_spec, snapshots, driver_grid(first_driver);
                              mass_basis = air_mass_basis(first_driver))
    end

    summarize_progress!(timer)

    m1 = total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = Float64(total_mass(model.state, name))
        if name in source_tracers
            @info @sprintf("Final tracer mass for %s (with source): %.12e kg",
                           String(name), rm1)
        elseif abs(rm0) > eps(Float64)
            @info @sprintf("Final tracer-mass drift for %s:         %.3e",
                           String(name), (rm1 - rm0) / rm0)
        else
            @info @sprintf("Final tracer mass for %s:               %.12e kg",
                           String(name), rm1)
        end
    end
    return model
end

# ===========================================================================
# CS runner (plan 40 Commit 6b, hoisted from scripts/run_cs_driven.jl)
# ===========================================================================

_cfg_float_type(cfg) = let s = get(get(cfg, "numerics", Dict()), "float_type", "Float64")
    s == "Float32" ? Float32 : Float64
end

function _cfg_architecture(cfg)
    if _cfg_use_gpu(cfg)
        _ensure_gpu_runtime!(cfg)
        return GPU()
    end
    return CPU()
end

function _run_driven_simulation_cs(binary_paths::Vector{String}, cfg)
    FT   = _cfg_float_type(cfg)
    assert_backend_float_type!(_cfg_runtime_backend(cfg), FT)
    arch = _cfg_architecture(cfg)

    run_cfg = get(cfg, "run", Dict{String, Any}())
    advection = build_cs_advection(cfg)
    Hp = configured_halo_width(cfg, advection)
    stop_window_override = get(run_cfg, "stop_window", nothing)
    reset_air_mass_each_window = Bool(get(run_cfg, "reset_air_mass_each_window", false))

    tracers_cfg = get(cfg, "tracers", Dict{String, Any}())
    isempty(tracers_cfg) && error("[tracers] must define at least one tracer")
    # Use the same tracer-spec parser as the LL/RG runner so
    # `[tracers.*.surface_flux]` blocks are picked up. Plain inline-Dict
    # parsing (the previous implementation) silently dropped surface_flux
    # configs and produced zero fossil emissions on CS.
    tracer_specs = _parse_tracer_specs(cfg)
    tracer_specs === nothing &&
        error("[tracers] section is malformed; expected per-tracer subsections")
    tracer_init = Dict(spec.name => spec.init_cfg for spec in tracer_specs)

    # First driver + model (reuses air_mass from window 1)
    driver1 = CubedSphereTransportDriver(first(binary_paths);
                                          FT = FT, arch = arch, Hp = Hp)
    output_cfg = get(cfg, "output", Dict{String, Any}())
    output_spec = runtime_output_spec(output_cfg, FT;
                                      default_cap_hours = _output_default_cap_hours(
                                          driver1, length(binary_paths);
                                          stop_window_override = stop_window_override))
    snapshot_schedule_hours = snapshot_hours(output_spec)
    do_snapshots = output_enabled(output_spec)
    if stop_window_override !== nothing && length(binary_paths) > 1 &&
       Int(stop_window_override) < total_windows(driver1)
        close(driver1)
        throw(ArgumentError(
            "[run] stop_window=$(stop_window_override) with multiple cubed-sphere " *
            "daily binaries would carry state from a partial day into the next " *
            "day's 00Z forcing. Use one binary for partial-window debugging, or " *
            "omit stop_window for a continuous multi-day run."))
    end
    recipe  = build_runtime_physics_recipe(cfg, driver1, FT; halo_width = Hp)
    _validate_capability_match(driver1, recipe, cfg)

    grid    = driver_grid(driver1)
    mesh    = grid.horizontal
    window1 = load_transport_window(driver1, 1)
    air_mass = window1.air_mass
    Nz = size(air_mass[1], 3)

    # Honor the binary's mass_basis — `DrivenSimulation._check_basis_compatibility`
    # compares `mass_basis(model.state/fluxes)` against `air_mass_basis(driver)`
    # and throws if they diverge. Hardcoding `DryBasis` would trip a
    # runtime ArgumentError on any moist-basis CS binary.
    basis_sym = air_mass_basis(driver1)
    BasisT    = basis_sym === :dry   ? DryBasis   :
                basis_sym === :moist ? MoistBasis :
                error("CS binary has unsupported mass_basis $(basis_sym); expected :dry or :moist")

    # Plan 40 Commit 2 + 1c: CS tracers flow through the unified IC
    # pipeline. DryBasis is the default per invariant 14; MoistBasis
    # requires qv from window1 (feedback_vmr_to_mass_basis_aware), which
    # CS windows do not carry today — so moist binaries error explicitly
    # here rather than producing silently wrong tracer mass.
    basis_sym === :moist &&
        error("CS driven runner does not yet support moist-basis binaries: " *
              "`pack_initial_tracer_mass` needs qv, which `CubedSphereTransportWindow` " *
              "does not expose. Regenerate the binary on dry basis " *
              "(`regrid_ll_transport_binary_to_cs.jl --mass-basis dry`), " *
              "or extend the CS window + this runner to thread qv.")

    tracer_kwargs = Dict{Symbol, NTuple{6, typeof(air_mass[1])}}()
    for (name, init_cfg) in tracer_init
        vmr = build_initial_mixing_ratio(air_mass, grid, init_cfg;
                                         surface_pressure = window1.surface_pressure)
        tracer_kwargs[name] = pack_initial_tracer_mass(grid, air_mass, vmr;
                                                       mass_basis = BasisT())
    end

    state  = CubedSphereState(BasisT, mesh, air_mass; tracer_kwargs...)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = BasisT)

    model = TransportModel(state, fluxes, grid, recipe.advection;
                            diffusion  = recipe.diffusion,
                            convection = recipe.convection)
    # Adapt state + fluxes to the selected backend. `invokelatest` is required
    # because GPU packages may be loaded dynamically and their Adapt methods can
    # arrive in a newer world age than this function's compiled body.
    adaptor = _backend_array_adapter(cfg)
    if adaptor !== Array
        model  = Base.invokelatest(Adapt.adapt, adaptor, model)
        state  = model.state                           # rebind post-adapt
        fluxes = model.fluxes
    end
    _assert_gpu_residency!(model.state, cfg)

    # Build surface-flux sources from the parsed tracer specs and log per-source
    # mass rates. Matches the LL/RG path; `DrivenSimulation`'s constructor
    # adapts these to the model backend (CPU Array or GPU array) via
    # `_adapt_sources_to_model_backend`, so no manual adapt step here.
    surface_sources = build_surface_flux_sources(grid, tracer_specs, FT)
    source_tracers = Set(source.tracer_name for source in surface_sources)
    for source in surface_sources
        # `cell_mass_rate` is topology-shaped: 2D Array on LL/RG, 6-tuple of
        # Matrices on CS. Reduce to a scalar for the log either way.
        total_rate = source.cell_mass_rate isa Tuple ?
                     Float64(sum(sum, source.cell_mass_rate)) :
                     Float64(sum(source.cell_mass_rate))
        @info @sprintf("Surface source %s total mass rate: %.12e kg/s",
                       String(source.tracer_name), total_rate)
    end

    _log_runtime_summary(topology = :CubedSphere,
                         mesh_label = @sprintf("C%d", mesh.Nc),
                         levels = Nz,
                         halo_width = Hp,
                         backend = _backend_label(cfg),
                         FT = FT,
                         recipe = recipe,
                         driver = driver1,
                         tracers = keys(tracer_init),
                         binary_count = length(binary_paths),
                         snapshot_file = _output_display_path(output_spec))

    # Snapshot storage is full-state and topology-native; Output handles halo
    # stripping and NetCDF diagnostics.
    snapshots = SnapshotFrame[]
    day_snapshots = SnapshotFrame[]
    snapshot_count = 0

    # Progress + IO/Transport timer. Estimate total windows from the
    # first driver; closes to the truth on homogeneous daily runs.
    per_binary_estimate = stop_window_override === nothing ?
                          total_windows(driver1) :
                          min(Int(stop_window_override), total_windows(driver1))
    timer = RunProgressTimer(per_binary_estimate * length(binary_paths))

    function capture_cs!(hour_total)
        timed_io_write!(timer, () -> begin
            frame = capture_snapshot(model; time_hours = hour_total,
                                     halo_width = Hp)
            if output_split(output_spec) === :daily
                push!(day_snapshots, frame)
            else
                push!(snapshots, frame)
            end
        end)
        snapshot_count += 1
        return nothing
    end

    snap_idx = 1
    total_hour = 0.0
    if do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
       abs(snapshot_schedule_hours[snap_idx]) < 0.5
        capture_cs!(0.0)
        set_progress_status!(timer;
                             detail = @sprintf("snapshot %d at t=%.1fh",
                                               snap_idx, 0.0),
                             redraw = true)
        snap_idx += 1
    end

    t0 = time()
    for (driver_idx, path) in enumerate(binary_paths)
        driver = driver_idx == 1 ? driver1 :
                 timed_io_read!(timer,
                     () -> CubedSphereTransportDriver(expanduser(path);
                                                       FT = FT, arch = arch, Hp = Hp))
        validate_runtime_physics_recipe(recipe, driver; halo_width = Hp)
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) :
                      min(Int(stop_window_override), total_windows(driver))
        window_hours = window_dt(driver) / 3600.0

        # Plan-39 Commit G removed the window-boundary air_mass reset, so
        # the cross-day handoff is continuity-consistent. We rebuild the
        # sim around each day's driver; state + physics carry over.
        if driver_idx != 1
            fluxes_d = allocate_face_fluxes(mesh, Nz; FT = FT, basis = BasisT)
            # Match the device of the already-adapted `state`: on GPU runs
            # the freshly-allocated fluxes start as CPU Arrays and would
            # mix types with GPU tracers otherwise. `invokelatest` guards
            # the same dynamic-load world-age issue as the initial adapt.
            adaptor !== Array &&
                (fluxes_d = Base.invokelatest(Adapt.adapt, adaptor, fluxes_d))
            model = TransportModel(state, fluxes_d, grid, recipe.advection;
                                    diffusion  = recipe.diffusion,
                                    convection = recipe.convection,
                                    chemistry  = recipe.chemistry)
            adaptor !== Array &&
                (model = Base.invokelatest(Adapt.adapt, adaptor, model))
        end
        initialize_air_mass = driver_idx == 1
        sim = timed_io_read!(timer,
            () -> DrivenSimulation(model, driver;
                                    start_window = 1, stop_window = stop_window,
                                    initialize_air_mass = initialize_air_mass,
                                    reset_air_mass_each_window = reset_air_mass_each_window,
                                    surface_sources = surface_sources,
                                    chemistry = recipe.chemistry))
        # `DrivenSimulation` may wrap `model` with a surface-flux operator;
        # keep snapshots and the return value aligned with the stepped model.
        model = sim.model

        day_t0 = time()
        set_progress_status!(timer;
                             status = @sprintf("running %s (%d windows)",
                                               basename(path), stop_window),
                             detail = @sprintf("schedule max=%d current=%d",
                                               maximum(sim.steps_per_window_schedule),
                                               sim.steps_per_window),
                             redraw = true)
        while sim.iteration < sim.final_iteration
            timed_transport!(timer, () -> step!(sim))
            if sim.iteration == sim.current_window_end_iteration
                tick_window!(timer;
                             status = @sprintf("%s window %d/%d  steps/window=%d",
                                               basename(path),
                                               sim.current_window_index,
                                               stop_window,
                                               sim.steps_per_window),
                             detail = @sprintf("snapshots=%d  output=%s",
                                               snapshot_count,
                                               _output_basename(output_spec)))
                total_hour += window_hours
                while do_snapshots && snap_idx <= length(snapshot_schedule_hours) &&
                      abs(total_hour - snapshot_schedule_hours[snap_idx]) < 0.5
                    capture_cs!(total_hour)
                    set_progress_status!(timer;
                                         detail = @sprintf("snapshot %d at t=%.1fh  output=%s",
                                                           snap_idx,
                                                           total_hour,
                                                           _output_basename(output_spec)),
                                         redraw = true)
                    snap_idx += 1
                end
            end
        end
        set_progress_status!(timer;
                             status = @sprintf("finished %s", basename(path)),
                             detail = @sprintf("file wall %.1fs", time() - day_t0),
                             redraw = true)
        if do_snapshots && output_split(output_spec) === :daily && !isempty(day_snapshots)
            written = _write_output_frames!(timer, output_spec, day_snapshots,
                                            grid;
                                            mass_basis = BasisT === DryBasis ? :dry : :moist,
                                            date_label = _binary_date_label(path),
                                            day_index = driver_idx)
            set_progress_status!(timer;
                                 detail = @sprintf("wrote %s", basename(written)),
                                 redraw = true)
            empty!(day_snapshots)
        end
        close(driver)
    end

    @info @sprintf("Done: %.1fs  (%d snapshots, final t=%.1fh)",
                   time() - t0, snapshot_count, total_hour)

    for name in keys(tracer_init)
        rm1 = Float64(total_mass(state, name))
        if name in source_tracers
            @info @sprintf("  %s total mass (with source): %.6e kg", name, rm1)
        else
            @info @sprintf("  %s total mass:               %.6e kg", name, rm1)
        end
    end

    if do_snapshots && output_split(output_spec) === :single && !isempty(snapshots)
        # BasisT was bound at model construction (dry by default on CS per
        # invariant 14); reuse it so the NetCDF records the same basis the
        # `air_mass` arrays were stored under.
        _write_output_frames!(timer, output_spec, snapshots, grid;
                              mass_basis = BasisT === DryBasis ? :dry : :moist)
    end
    summarize_progress!(timer)
    return model
end

end # module DrivenRunner
