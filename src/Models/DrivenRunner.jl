"""
    DrivenRunner

Library-level entry point for the driven transport runtime.

The canonical CLI, `scripts/run_transport.jl`, is a thin wrapper over
`run_driven_simulation(cfg)`. The library function handles LL/RG and CS
runtime flows with dispatch driven by the first binary's header
(`inspect_binary(first_path).grid_type`). Historical runner names live under
`scripts/deprecated/` only for reference.

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

## Where the physics actually runs

`DrivenRunner` is the orchestration layer, not the kernel layer. It opens
transport binaries, builds initial state, chooses output cadence, and installs
the TOML-selected operators on a `TransportModel`. The live physics call chain
is:

1. `build_runtime_physics_recipe` reads `[advection]`, `[diffusion]`,
   `[convection]`, `[chemistry]`, and tracer surface-flux settings and returns
   the operator objects.
2. `_make_structured_model` or the CS model constructor installs those
   operators on `TransportModel(state, fluxes, grid, recipe.advection; ...)`.
3. `DrivenSimulation(model, driver; ...)` loads the current met window and
   wraps chemistry and surface sources into the model with `with_chemistry`
   and `with_emissions`.
4. The runtime loop below calls `run_window!(sim)` for LL/RG or `step!(sim)`
   for CS. Those functions live in `DrivenSimulation.jl`.
5. `DrivenSimulation.step!` refreshes time-varying forcing from the driver,
   then calls `TransportModel.step!` or, for binary-scheduled substeps,
   `transport_step!` plus an end-of-window `convection_chemistry_step!`.
6. `TransportModel.jl` is where advection, surface emissions, diffusion,
   convection, and chemistry are applied to the state.

So, when following a run as a scientist: start here to understand data and
configuration flow, then jump to `DrivenSimulation.step!` and
`TransportModel.step!` to see the actual physics ordering.

## GPU residency (feedback_verify_gpu_runs_on_gpu)

When `[architecture].use_gpu = true` or `backend` selects a GPU, the runner
asserts that `state.air_mass` lives on the selected backend after model
construction and prints a `[gpu verified] …` line. A silent CPU fallback
aborts the run with a precise error message.
"""
module DrivenRunner

using Adapt
using Dates: Date, DateTime
using Printf: @sprintf, @printf
using Logging
using ProgressMeter: Progress, next!, finish!, update!

import ...expand_data_path
using ...SectionTimer
using ..State: AbstractMassBasis, DryBasis, MoistBasis, CellState,
                CubedSphereState, total_air_mass, total_mass_full,
                tracer_names, tracer_index, halo_width,
                get_tracer_raw, set_tracer_reference!, REF_GLOBAL_MEAN,
                tracer_reference_value, mass_weighted_global_mean_vmr
using ..Grids: nlevels
using ..Operators: LinRoodPPMScheme, PPMScheme, SlopesScheme, UpwindScheme,
                  NoAdvection,
                  ImplicitVerticalDiffusion, NoDiffusion,
                  uses_diffusive_surface_flux_boundary,
                  AbstractConvection,
                  NoConvection, TM5Convection, CMFMCConvection,
                  CMFMCMatrixConvection,
                  NoChemistry, ExponentialDecay
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
                AbstractOutputPartition, SingleOutputFile, DailyOutputFiles,
                RuntimeOutputSpec, runtime_output_spec, snapshot_hours,
                output_enabled, output_path, output_path_for_day,
                capture_snapshot, write_snapshot_netcdf, write_snapshot_binary
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

export run_driven_simulation, validate_config, TransportTracerSpec

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
    name              :: Symbol
    init_cfg          :: Dict{String, Any}
    surface_flux_cfg  :: Dict{String, Any}
    reference_kind    :: Symbol   # :none | :global_mean   ([tracers.X.transport] reference)
    reference_cadence :: Symbol   # :fixed | :daily | :per_window
end

# Back-compat convenience: specs built without a [tracers.X.transport] block
# (and the LL single-tracer fallback) default to the raw, unreferenced path.
TransportTracerSpec(name::Symbol, init_cfg::Dict{String, Any},
                    surface_flux_cfg::Dict{String, Any}) =
    TransportTracerSpec(name, init_cfg, surface_flux_cfg, :none, :fixed)

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

# Parse + validate the optional [tracers.X.transport] block (reference-state /
# anomaly transport; see docs/plans/45_ANOMALY_REFERENCE_TRANSPORT/PLAN.md).
# Strings become Symbols HERE and nowhere downstream; unknown keys and values
# are parse-time errors so a typo cannot silently run unreferenced.
function _tracer_transport_cfg(name, tracer_cfg)
    transport_cfg = get(tracer_cfg, "transport", nothing)
    transport_cfg === nothing && return (reference = :none, cadence = :fixed)
    transport_cfg isa AbstractDict || throw(ArgumentError(
        "[tracers.$(name).transport] must be a table"))
    known = ("reference", "reference_cadence")
    for key in keys(transport_cfg)
        key in known || throw(ArgumentError(
            "[tracers.$(name).transport] has unknown key \"$(key)\"; " *
            "supported keys: $(join(known, ", "))"))
    end
    reference_raw = get(transport_cfg, "reference", "none")
    reference_raw isa AbstractString || throw(ArgumentError(
        "[tracers.$(name).transport] reference must be a string, got " *
        "$(typeof(reference_raw))"))
    reference = String(reference_raw)
    reference in ("none", "global_mean") || throw(ArgumentError(
        "[tracers.$(name).transport] reference=\"$(reference)\" is not supported; " *
        "use \"none\" or \"global_mean\""))
    cadence_raw = get(transport_cfg, "reference_cadence", "fixed")
    cadence_raw isa AbstractString || throw(ArgumentError(
        "[tracers.$(name).transport] reference_cadence must be a string, got " *
        "$(typeof(cadence_raw))"))
    cadence = String(cadence_raw)
    cadence in ("fixed", "daily", "per_window") || throw(ArgumentError(
        "[tracers.$(name).transport] reference_cadence=\"$(cadence)\" is not " *
        "supported; use \"fixed\", \"daily\", or \"per_window\""))
    return (reference = Symbol(reference), cadence = Symbol(cadence))
end

function _parse_tracer_specs(cfg)
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing
    names = sort!(collect(keys(tracers_cfg)))
    isempty(names) && throw(ArgumentError("config has [tracers] but no tracer sections"))
    return Tuple(begin
        transport = _tracer_transport_cfg(name, tracers_cfg[name])
        TransportTracerSpec(Symbol(name),
                            _tracer_init_cfg(tracers_cfg[name]),
                            _tracer_surface_flux_cfg(tracers_cfg[name]),
                            transport.reference,
                            transport.cadence)
    end for name in names)
end

# ===========================================================================
# Reference-state (anomaly) transport — compatibility gates + IC seeding
# (plan 45 Stage 2). A tracer may opt into `reference = "global_mean"` only
# when every operator acting on it is offset-invariant: the analytic
# reference must follow the air mass exactly while operators see the anomaly.
# ===========================================================================

"""
    is_offset_invariant(op, tracer_name) -> Bool

`true` when applying `op` to a tracer stored as anomaly mass
`q_anom·m = (q_full - q_ref)·m` produces exactly the anomaly of applying it
to the full field — i.e. a uniform-VMR field is an eigenstate of `op` and
`op` is linear in the tracer. Granularity is per-(operator, tracer,
configuration): decay applies per-tracer rates, convection flips with
`clamp`. The default is `false` — a new operator must opt IN by proving the
property, never inherit it.
"""
is_offset_invariant(op, tracer_name::Symbol) = false

# Advection: only the LinRood palindrome qualifies — its horizontal
# monotonicity works on VMR differences and its vertical sweep is pure
# donor-cell. The split-sweep schemes (Upwind/Slopes/PPM) share a moment
# limiter that clamps against stored tracer mass (`_limited_moment`), which
# inverts for a signed anomaly store; they are also runtime-guarded in
# `strang_split!`. (fillz on the LinRood path is handled by the Stage-3
# negativity gate, not by this trait.)
is_offset_invariant(::LinRoodPPMScheme, ::Symbol) = true
is_offset_invariant(::NoAdvection, ::Symbol) = true   # apply! is a no-op

# Diffusion: the implicit vertical solve is linear in the tracer with
# row-sum-1 (uniform columns are exact eigenstates; it already carries its
# own per-column anomaly subtraction internally).
is_offset_invariant(::ImplicitVerticalDiffusion, ::Symbol) = true

# Convection: flux-divergence / LU forms are linear and uniform-preserving
# ("uniform mixing ratio preserved" is a pinned TM5 test invariant) — but ONLY
# on the exact path. The optional CMFMC clamp is a positivity fixer on stored
# mass, and the level-merge approximation (`n_merge > 1`) disaggregates
# supercells with `fine_old / super_old` RATIOS of stored tracer values —
# nonlinear, and meaningless on a signed anomaly store. `n_merge = 1` is the
# bit-exact path and the only one that qualifies (codex review finding).
is_offset_invariant(::NoConvection, ::Symbol) = true
is_offset_invariant(op::CMFMCConvection, ::Symbol) = !op.clamp
is_offset_invariant(op::TM5Convection, tracer_name::Symbol) = op.n_merge == 1
is_offset_invariant(op::CMFMCMatrixConvection, tracer_name::Symbol) =
    is_offset_invariant(op.inner, tracer_name)

# Chemistry: exponential decay is multiplicative (`rm *= e^{-kΔt}`) — it
# would decay the stored anomaly but not the analytic reference. Only a
# tracer the decay operator does not act on is safe.
is_offset_invariant(op::ExponentialDecay, tracer_name::Symbol) =
    !(tracer_name in op.tracer_names)
is_offset_invariant(::NoChemistry, ::Symbol) = true

# NoDiffusion/NoAdvection-style defaults: a no-op is trivially invariant.
is_offset_invariant(::NoDiffusion, ::Symbol) = true

"""
    _validate_tracer_reference_compat(tracer_specs, recipe;
                                      reset_air_mass_each_window)

Model-level compatibility check for referenced tracers (CS runner). Throws
an `ArgumentError` naming the offending operator for the first referenced
tracer that any non-offset-invariant operator acts on, and rejects the
preserve-VMR window reset (it rescales FULL VMR, which double-counts the
reference — see plan 45 Risk 5).
"""
function _validate_tracer_reference_compat(tracer_specs, recipe;
                                           reset_air_mass_each_window::Bool)
    referenced = [spec for spec in tracer_specs if spec.reference_kind !== :none]
    isempty(referenced) && return nothing
    names = join((String(s.name) for s in referenced), ", ")
    # REBASE NOTE (air_mass_reset_mode refactor): preserve_vmr stays rejected;
    # preserve_tracer_mass is NOT automatically safe for referenced tracers —
    # the q_ref·m part of the burden rides the air mass, so the reset must
    # absorb anom += q_ref·(m_old − m_new) or also be rejected. See
    # docs/plans/45_ANOMALY_REFERENCE_TRANSPORT/REBASE_NOTES_air_mass_reset_mode.md
    reset_air_mass_each_window && throw(ArgumentError(
        "reset_air_mass_each_window = true preserves VMR across the window " *
        "air-mass reset, which is not reference-aware (it would double-count " *
        "q_ref for: $(names)); disable the reset or use reference = \"none\""))
    for spec in referenced
        for (label, op) in (("[advection]", recipe.advection),
                            ("[diffusion]", recipe.diffusion),
                            ("[convection]", recipe.convection),
                            ("[chemistry]", recipe.chemistry))
            is_offset_invariant(op, spec.name) || throw(ArgumentError(
                "[tracers.$(spec.name).transport] reference=\"global_mean\" is " *
                "incompatible with $(label) operator $(typeof(op)): it is not " *
                "offset-invariant, so it cannot act on anomaly-mass storage. " *
                (op isa CMFMCConvection && op.clamp ?
                     "Disable the CMFMC clamp or use reference=\"none\"." :
                 label == "[advection]" ?
                     "Use [advection] scheme = \"linrood\" for referenced tracers." :
                     "Use reference=\"none\" for this tracer.")))
        end
    end
    return nothing
end

"""
    _apply_reference_cadence!(state, tracer_specs, boundary::Symbol)

Re-reference hook (plan 45 Stage 5): for each referenced tracer whose
`reference_cadence` matches `boundary` (`:daily` at day/binary boundaries,
`:per_window` at met-window ends), recompute the reference and shift the
anomaly store by the drift of its global mean:

    Δ = Σ_interior(q_anom·m) / Σ_interior(m)      (F64)
    q_anom·m ← q_anom·m − Δ·m                      (FT muladd)
    q_ref    ← q_ref + Δ                           (F64)

The full-field burden `Σ(q_anom·m) + q_ref·Σm` is invariant in exact
arithmetic; the FT store incurs BOUNDED per-cell roundoff at the anomaly
scale (Δ is the accumulated mean drift, normally ≪ the anomaly spread), not
an exact-F64 guarantee. Runs on the live (possibly GPU-adapted) state — the
broadcast shift and F64 reductions are backend-generic, and the carrier is
host-shared.
"""
function _apply_reference_cadence!(state, tracer_specs, boundary::Symbol)
    for spec in tracer_specs
        spec.reference_kind === :none && continue
        spec.reference_cadence === boundary || continue
        idx = tracer_index(state, spec.name)
        idx === nothing && continue
        raw = get_tracer_raw(state, idx)
        m = state.air_mass
        Δ = mass_weighted_global_mean_vmr(raw, m, halo_width(state))
        q_ref_old = tracer_reference_value(state, idx)
        q_ref_old === nothing && continue   # defensive: spec/carrier mismatch
        FT = eltype(m[1])
        dq = FT(Δ)
        @inbounds for p in 1:6
            raw[p] .= muladd.(-dq, m[p], raw[p])
        end
        set_tracer_reference!(state.tracer_refs, idx, REF_GLOBAL_MEAN,
                              q_ref_old + Δ)
        @debug "re-referenced tracer" spec.name boundary Δ new_q_ref = q_ref_old + Δ
    end
    return nothing
end

# End-of-window callback for `reference_cadence = "per_window"` tracers;
# returns an empty tuple when no tracer needs it (zero overhead default).
function _reference_cadence_callbacks(tracer_specs)
    any(spec -> spec.reference_kind !== :none &&
                spec.reference_cadence === :per_window, tracer_specs) ||
        return NamedTuple()
    callback = sim -> begin
        sim.iteration == sim.current_window_end_iteration &&
            _apply_reference_cadence!(sim.model.state, tracer_specs, :per_window)
        nothing
    end
    return (reference_cadence = callback,)
end

"""
    _seed_tracer_references!(state, tracer_specs)

Convert each `reference = "global_mean"` tracer from full mass to anomaly
mass: `q_ref = mass_weighted_global_mean_vmr` (F64, interior cells), then
`raw ← raw - q_ref·m` computed in F64 and stored back in `FT`. Must run on
the CPU-resident state BEFORE backend adaptation, immediately after IC
packing — the stored field and the binary's window-1 air mass are still
consistent there. Logs each seeded reference (probe-before-build).
"""
function _seed_tracer_references!(state, tracer_specs)
    for spec in tracer_specs
        spec.reference_kind === :none && continue
        spec.reference_kind === :global_mean || throw(ArgumentError(
            "unsupported reference kind $(spec.reference_kind) for tracer $(spec.name)"))
        idx = tracer_index(state, spec.name)
        idx === nothing && throw(KeyError(spec.name))
        raw = get_tracer_raw(state, idx)
        m = state.air_mass
        q_ref = mass_weighted_global_mean_vmr(raw, m, halo_width(state))
        FT = eltype(m[1])
        @inbounds for p in 1:6
            # F64 subtraction, FT store: the seed must not round in FT before
            # the subtraction or the anomaly inherits the background's F32
            # quantization (the thing this scheme removes).
            raw[p] .= FT.(Float64.(raw[p]) .- q_ref .* Float64.(m[p]))
        end
        set_tracer_reference!(state.tracer_refs, idx, REF_GLOBAL_MEAN, q_ref)
        @info @sprintf("Tracer %s: reference-state transport enabled, q_ref = %.9e (dry VMR, global mean)",
                       String(spec.name), q_ref)
    end
    return nothing
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
        backend = _cfg_runtime_backend(cfg)
        ft_ref[] === nothing || assert_backend_float_type!(backend, ft_ref[])
    end

    tracers_cfg = get(cfg, "tracers", nothing)
    if tracers_cfg !== nothing
        if !(tracers_cfg isa AbstractDict)
            push!(errors, "[tracers] must be a TOML table of tracer subtables.")
        elseif isempty(tracers_cfg)
            push!(errors, "[tracers] was provided but contains no tracer subtables.")
        else
            # Full spec parse so per-tracer table errors ([tracers.X.transport]
            # unknown keys/values, staged-feature guards) surface in preflight
            # instead of only at runner start.
            _capture_config_error!(errors) do
                _parse_tracer_specs(cfg)
            end
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

function _synchronize_backend!(cfg)
    synchronize_backend!(_cfg_runtime_backend(cfg))
    return nothing
end

"""
    _assert_gpu_residency!(state, cfg)

See `feedback_verify_gpu_runs_on_gpu`. When a GPU backend is
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
# Capability validation
#
# Validate TOML physics against binary capabilities BEFORE constructing the
# model, so users get a precise error up front instead of silently failing
# partway through. Runs after `build_runtime_physics_recipe` (which already
# validates kind strings against recipe types) but before model construction
# (which discovers problems at the first load).
# ===========================================================================

function _validate_capability_match(driver, recipe)
    _validate_convection_capability(recipe.convection,
                                     binary_capabilities(driver.reader))
    return nothing
end

# Dispatch on the concrete convection-operator type so a new operator is a
# new method (compile-time coverage), not a new branch in an if-chain. The
# raw `cfg` no longer participates — the recipe has already been built and
# its convection field is authoritative.
_validate_convection_capability(::NoConvection, _caps) = nothing

function _validate_convection_capability(::TM5Convection, caps)
    caps.tm5_convection || throw(ArgumentError(
        "[convection] kind = \"tm5\" requires the binary to carry " *
        "entu, detu, entd, detd; this binary's payload_sections are " *
        "$(caps.payload_sections). Regenerate with a TM5-enabled " *
        "preprocessor or set convection.kind = \"none\"."))
    return nothing
end

function _validate_convection_capability(::CMFMCConvection, caps)
    caps.cmfmc_convection || throw(ArgumentError(
        "[convection] kind = \"cmfmc\" requires the binary to carry " *
        "the cmfmc section; this binary's payload_sections are " *
        "$(caps.payload_sections)."))
    return nothing
end

# The matrix variant has NO Tiedtke fallback — `dtrain` is the explicit
# detrainment rate that closes the continuity derivation
# `entu - detu = cmfmc[k] - cmfmc[k+1]`. A binary with cmfmc but no dtrain is
# hard-rejected up front so the failure mode is actionable at recipe-validation
# time (not at the first window load several seconds later).
function _validate_convection_capability(::CMFMCMatrixConvection, caps)
    (caps.cmfmc_convection && :dtrain in caps.payload_sections) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc_matrix\" requires the binary " *
            "to carry both cmfmc AND dtrain payloads (no Tiedtke fallback); " *
            "this binary's payload_sections are $(caps.payload_sections). " *
            "Regenerate the binary with a preprocessor that emits :dtrain, " *
            "or fall back to kind=\"cmfmc\" which has a Tiedtke path."))
    return nothing
end

# Catch-all for any future convection operator. Forces a method to be added
# here when a new operator type appears, which is the whole point of the
# dispatch refactor.
function _validate_convection_capability(op::AbstractConvection, _caps)
    throw(ArgumentError(
        "no _validate_convection_capability method for $(typeof(op)); " *
        "add a dispatch in DrivenRunner.jl when introducing a new convection " *
        "operator type."))
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

function _log_binary_summary(path::AbstractString, caps)
    schedule = caps.variable_step_schedule ?
        "adaptive" : string(caps.steps_per_window)
    fields = "[" * join(String.(sort(collect(caps.payload_sections))), ",") * "]"
    @info "[binary] $(path) grid=$(caps.grid_type) levels=$(caps.nlevel) " *
          "basis=$(caps.mass_basis) steps/window=$(schedule) fields=$(fields)"
    return nothing
end

"""
    run_driven_simulation(cfg::AbstractDict) -> TransportModel

Canonical high-level entry point for scientists running an AtmosTransport TOML.
Use this unless you are writing a custom time loop. It resolves `[input]` to a
sorted binary list, prints a one-line summary for each binary, dispatches on
the first binary's `grid_type`, validates physics-vs-capability, runs the loop,
optionally writes topology-native snapshots, and returns the terminal
`TransportModel`.

This function does not call the advection/diffusion/convection kernels
directly. The handoff to physics happens inside the structured loop at
`run_window!(sim)` and inside the CS loop at `step!(sim)`. Both routes enter
`DrivenSimulation.step!`, which refreshes forcing and then calls
`TransportModel.step!` / `transport_step!` / `convection_chemistry_step!`.
"""
function run_driven_simulation(cfg::AbstractDict)
    ok, errors = validate_config(cfg)
    ok || throw(ArgumentError(
        "Invalid AtmosTransport run config:\n  - " * join(errors, "\n  - ")))
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) &&
        throw(ArgumentError("[input] resolved to an empty binary list"))
    # Section timing instrumentation, off unless ATMOSTR_TIMERS=1.
    # Enabled here so every section accumulator covers the whole driven
    # loop including snapshot capture / write.
    timers_on = SectionTimer.maybe_enable_from_env!()
    # Dispatch on the first binary's grid_type — the ownership boundary
    # (binary header owns topology, TOML owns physics kinds). The
    # capability probe also runs the load-time
    # gates (stale-binary, cm-continuity) as a side effect of opening
    # the reader in `inspect_binary`.
    binary_caps = [(path = path, caps = inspect_binary(path; io = devnull))
                   for path in binary_paths]
    for item in binary_caps
        _log_binary_summary(item.path, item.caps)
    end
    caps = first(binary_caps).caps
    _validate_input_binary_expectations(caps, input_cfg, first(binary_paths))
    result = _run_driven_simulation_for(Val(caps.grid_type), binary_paths, cfg)
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

_run_driven_simulation_for(::Val{:latlon}, binary_paths::Vector{String}, cfg) =
    _run_driven_simulation_structured(binary_paths, cfg)
_run_driven_simulation_for(::Val{:reduced_gaussian}, binary_paths::Vector{String}, cfg) =
    _run_driven_simulation_structured(binary_paths, cfg)
_run_driven_simulation_for(::Val{:cubed_sphere}, binary_paths::Vector{String}, cfg) =
    _run_driven_simulation_cs(binary_paths, cfg)
function _run_driven_simulation_for(::Val{grid_type}, _binary_paths::Vector{String}, _cfg) where grid_type
    throw(ArgumentError("Unsupported transport-binary grid_type=$(grid_type)."))
end

# Parse the run start as a `DateTime` (at 00:00) from `[input].start_date`,
# used as the `reference_time` for time-varying surface fluxes so their slice
# times align to the simulation clock. Returns `nothing` if no start_date is
# present (the time-varying loader then assumes the file origin == run start).
function _run_reference_time(cfg)
    input_cfg = get(cfg, "input", nothing)
    input_cfg isa AbstractDict || return nothing
    haskey(input_cfg, "start_date") || return nothing
    return DateTime(Date(String(input_cfg["start_date"])))
end

# Reduce a surface source's per-cell rate to a scalar total for logging,
# handling both static (`cell_mass_rate`) and time-varying
# (`cell_mass_rate_series`, summed over the first slice) sources.
function _surface_source_total_rate(source)
    if hasproperty(source, :cell_mass_rate)
        r = source.cell_mass_rate
        return r isa Tuple ? Float64(sum(sum, r)) : Float64(sum(r))
    else
        series = source.cell_mass_rate_series   # NTuple{6} of (Nc,Nc,ntime)
        return Float64(sum(p -> sum(@view p[:, :, 1]), series))
    end
end

function _run_driven_simulation_structured(binary_paths::Vector{String}, cfg)
    FT = _cfg_float_type(cfg)
    assert_backend_float_type!(_cfg_runtime_backend(cfg), FT)
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    haskey(run_cfg, "reset_air_mass_each_window") &&
        throw(ArgumentError("run.reset_air_mass_each_window was replaced by " *
                            "run.air_mass_reset_mode = \"none\", " *
                            "\"preserve_vmr\", or \"preserve_tracer_mass\""))
    air_mass_reset_mode = get(run_cfg, "air_mass_reset_mode", "preserve_tracer_mass")

    init_cfg = get(cfg, "init", Dict{String, Any}())
    tracer_specs = something(_parse_tracer_specs(cfg),
                             (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                                  _copy_cfg_dict(init_cfg),
                                                  Dict{String, Any}()),))
    # Reference-state (anomaly) transport is cubed-sphere-only: the LL/RG
    # state carries no reference metadata and its schemes are split-sweep
    # (moment limiter is not offset-invariant).
    for spec in tracer_specs
        spec.reference_kind === :none || throw(ArgumentError(
            "[tracers.$(spec.name).transport] reference=\"global_mean\" is only " *
            "supported on cubed-sphere runs (LinRood advection); this is a " *
            "lat-lon / reduced-Gaussian run"))
    end

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
    _validate_capability_match(first_driver, recipe)

    # Build the stateful physics object. The recipe selected from TOML is
    # installed on `TransportModel` here, but no physics is applied yet; that
    # starts when `run_window!(sim)` or `run!(sim)` calls
    # `DrivenSimulation.step!` below.
    model = _make_structured_model(first_driver;
                                    FT = FT, recipe = recipe,
                                    tracer_specs = tracer_specs, cfg = cfg)
    _assert_gpu_residency!(model.state, cfg)

    grid_of_first = driver_grid(first_driver)
    surface_sources = build_surface_flux_sources(grid_of_first, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg))
    m0 = total_air_mass(model.state)
    tracer_masses0 = Dict(name => total_mass_full(model.state, name)
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
        @info @sprintf("Surface source %s total model-storage rate: %.12e kg_air_equiv/s",
                       String(source.tracer_name),
                       _surface_source_total_rate(source))
    end

    snapshots = SnapshotFrame[]
    day_snapshots = SnapshotFrame[]
    snapshot_count = Ref(0)
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
            _push_snapshot_frame!(output_spec.partition, snapshots,
                                  day_snapshots, frame)
        end)
        snapshot_count[] += 1
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

    run_time_seconds = 0.0   # accumulated across binaries (sim clock origin)
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
                                    air_mass_reset_mode = air_mass_reset_mode,
                                    surface_sources = surface_sources,
                                    chemistry = recipe.chemistry,
                                    # seconds since RUN start — see the CS loop
                                    # note: per-binary clock restarts replay
                                    # day-1 time-varying fluxes
                                    start_time = run_time_seconds))
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
                # Physics handoff for LL/RG: `run_window!` advances through all
                # substeps in the current met window. Each substep refreshes
                # forcing in `DrivenSimulation.step!` and then calls the
                # operator order documented in `TransportModel.step!`.
                timed_transport!(timer, () -> run_window!(sim))
                tick_window!(timer;
                             status = @sprintf("%s window %d/%d  steps/window=%d",
                                               basename(path),
                                               sim.current_window_index,
                                               stop_window,
                                               sim.steps_per_window),
                             detail = @sprintf("snapshots=%d  output=%s",
                                               snapshot_count[],
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
            # Same physics path as above, but without per-window snapshot
            # interrupts. `run!` repeatedly calls `DrivenSimulation.step!`.
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

        run_time_seconds += (stop_window - start_window + 1) * Float64(window_dt(driver))
        _synchronize_backend!(cfg)
        set_progress_status!(timer;
                             status = @sprintf("finished %s", basename(path)),
                             detail = @sprintf("file wall %.2fs", time() - t0),
                             redraw = true)
        if do_snapshots
            written = _flush_daily_output!(output_spec.partition, timer,
                                           output_spec, day_snapshots,
                                           driver_grid(first_driver);
                                           mass_basis = air_mass_basis(first_driver),
                                           date_label = _binary_date_label(path),
                                           day_index = idx)
            written !== nothing &&
                set_progress_status!(timer;
                                     detail = @sprintf("wrote %s", basename(written)),
                                     redraw = true)
        end
        close(driver)
    end

    if do_snapshots
        # `air_mass_basis(driver)` already returns the Symbol and has been
        # validated to match `model.state`'s basis by
        # `_check_basis_compatibility` before any step!.
        _flush_single_output!(output_spec.partition, timer, output_spec,
                              snapshots, driver_grid(first_driver);
                              mass_basis = air_mass_basis(first_driver))
    end

    summarize_progress!(timer)

    m1 = total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = Float64(total_mass_full(model.state, name))
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
# CS runner
# ===========================================================================

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
    haskey(run_cfg, "reset_air_mass_each_window") &&
        throw(ArgumentError("run.reset_air_mass_each_window was replaced by " *
                            "run.air_mass_reset_mode = \"none\", " *
                            "\"preserve_vmr\", or \"preserve_tracer_mass\""))
    air_mass_reset_mode = get(run_cfg, "air_mass_reset_mode", "preserve_tracer_mass")

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
    _validate_capability_match(driver1, recipe)
    # Reference-state (anomaly) tracers: every operator acting on a referenced
    # tracer must be offset-invariant, and the preserve-VMR window reset is
    # incompatible. Fail at setup with the offending operator named, before
    # any state is allocated.
    _validate_tracer_reference_compat(tracer_specs, recipe;
                                      reset_air_mass_each_window =
                                          reset_air_mass_each_window)

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

    # CS tracers flow through the unified IC pipeline.
    # DryBasis is the default per invariant 14; MoistBasis
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
    # Seed reference-state tracers while the state is CPU-resident and still
    # exactly consistent with window-1 air mass: full mass -> anomaly mass
    # (F64 math, FT store) + carrier metadata. Must precede backend adaptation.
    _seed_tracer_references!(state, tracer_specs)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = BasisT)

    # Build the CS physics object. The recipe-selected operators are installed
    # on the model here; the kernels start running later in the `step!(sim)`
    # loop after `DrivenSimulation` has loaded/refreshed each forcing window.
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
    surface_sources = build_surface_flux_sources(grid, tracer_specs, FT;
                                                 reference_time = _run_reference_time(cfg))
    source_tracers = Set(source.tracer_name for source in surface_sources)
    for source in surface_sources
        # Reduce the topology-shaped per-cell rate to a scalar for the log;
        # `_surface_source_total_rate` handles static (2D / 6-tuple) and
        # time-varying (first-slice) sources alike.
        @info @sprintf("Surface source %s total model-storage rate: %.12e kg_air_equiv/s",
                       String(source.tracer_name), _surface_source_total_rate(source))
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
    snapshot_count = Ref(0)

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
            _push_snapshot_frame!(output_spec.partition, snapshots,
                                  day_snapshots, frame)
        end)
        snapshot_count[] += 1
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

        # There is no window-boundary air_mass reset, so the cross-day
        # handoff is continuity-consistent. We rebuild the
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
        # Re-reference `reference_cadence = "daily"` tracers at the day/binary
        # boundary, before the new day's forcing applies (plan 45 Stage 5).
        driver_idx == 1 || _apply_reference_cadence!(state, tracer_specs, :daily)
        sim = timed_io_read!(timer,
            () -> DrivenSimulation(model, driver;
                                    start_window = 1, stop_window = stop_window,
                                    initialize_air_mass = initialize_air_mass,
                                    air_mass_reset_mode = air_mass_reset_mode,
                                    surface_sources = surface_sources,
                                    chemistry = recipe.chemistry,
                                    callbacks = _reference_cadence_callbacks(tracer_specs),
                                    # accumulated run time: time-varying sources
                                    # index emission slices in seconds since RUN
                                    # start; restarting at 0 per day replays
                                    # day-1 fluxes (the +1 Pg/month co2_natural
                                    # surplus, root-caused by plan 45 Stage 4)
                                    start_time = total_hour * 3600.0))
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
            # Physics handoff for CS: one `step!(sim)` is one runtime substep.
            # `DrivenSimulation.step!` refreshes window forcing, then delegates
            # to `TransportModel.step!` or to the binary-scheduled
            # `transport_step!` + end-of-window `convection_chemistry_step!`
            # split. See those functions for the actual operator order.
            timed_transport!(timer, () -> step!(sim))
            if sim.iteration == sim.current_window_end_iteration
                tick_window!(timer;
                             status = @sprintf("%s window %d/%d  steps/window=%d",
                                               basename(path),
                                               sim.current_window_index,
                                               stop_window,
                                               sim.steps_per_window),
                             detail = @sprintf("snapshots=%d  output=%s",
                                               snapshot_count[],
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
        if do_snapshots
            written = _flush_daily_output!(output_spec.partition, timer,
                                           output_spec, day_snapshots, grid;
                                           mass_basis = BasisT === DryBasis ? :dry : :moist,
                                           date_label = _binary_date_label(path),
                                           day_index = driver_idx)
            written !== nothing &&
                set_progress_status!(timer;
                                     detail = @sprintf("wrote %s", basename(written)),
                                     redraw = true)
        end
        close(driver)
    end

    @info @sprintf("Done: %.1fs  (%d snapshots, final t=%.1fh)",
                   time() - t0, snapshot_count[], total_hour)

    for name in keys(tracer_init)
        rm1 = Float64(total_mass_full(state, name))
        if name in source_tracers
            @info @sprintf("  %s total mass (with source): %.6e kg", name, rm1)
        else
            @info @sprintf("  %s total mass:               %.6e kg", name, rm1)
        end
    end

    if do_snapshots
        # BasisT was bound at model construction (dry by default on CS per
        # invariant 14); reuse it so the NetCDF records the same basis the
        # `air_mass` arrays were stored under.
        _flush_single_output!(output_spec.partition, timer, output_spec,
                              snapshots, grid;
                              mass_basis = BasisT === DryBasis ? :dry : :moist)
    end
    summarize_progress!(timer)
    return model
end

end # module DrivenRunner
