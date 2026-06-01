# ---------------------------------------------------------------------------
# Strided checkpoint driver for `cs_surface_emission_footprint`.
#
# `_collect_surface_footprints_stride` is the entry point used when the
# `checkpoint` kwarg is a `StrideCheckpoint(K)`. It runs a forward
# checkpoint pass that saves `panels_m` at every K-step boundary,
# discarding ops as it goes, and then walks the reverse pass one
# window at a time — re-recording each window's tape from its
# saved left-edge checkpoint, walking those window ops backwards,
# and dropping them before moving to the previous window.
#
# Scope (this commit): linear-scheme mass tape only. The nonlinear
# PPM tracer tape and the LinRood horizontal tape will get their
# own stride drivers in follow-up commits. The public API rejects
# non-`FullCheckpoint` schedules for those schemes with a clear
# `ArgumentError` rather than silently producing a wrong adjoint.
# ---------------------------------------------------------------------------

# Per-step kwargs (`diffusion_op`, `diffusion_workspace`,
# `convection_forcing`) are either Vectors of length `nsteps` or a
# scalar broadcast value (handled by `_diffusion_sequence_at` /
# `_convection_forcing_at`). When recording one window, slice
# Vector inputs to the window's step range; pass scalar inputs
# through unchanged so the downstream `_*_sequence_at` calls keep
# their broadcast semantics.
_slice_step_kwarg(v::AbstractVector, range::AbstractUnitRange) = view(v, range)
_slice_step_kwarg(v, ::AbstractUnitRange) = v

# Per-window / per-base-case subdirectory under a user-supplied
# `tape_path`. Five zero-padded digits keep alphabetical sort agreeing
# with numeric order through nsteps = 99999, which dominates any
# physically-motivated CS run length. The leading tag distinguishes
# Stride (`window_*`) from Revolve (`step_*`) subdirs so a single
# `tape_path` tree can host both schedules without collisions.
_stride_window_subdir(w::Integer) = "window_" * lpad(w, 5, '0')
_revolve_step_subdir(step::Integer) = "step_" * lpad(step, 5, '0')

# Lambda construction at the start of the reverse pass.
# - `final_adjoint_seed = nothing` (default): zero-allocate panels
#   shaped like `final_m` and seed via the objective. Used by
#   `cs_surface_emission_footprint` (objective-driven path).
# - `final_adjoint_seed::NTuple{6}` (A.3d from-seed stride): caller
#   supplies the final-time adjoint directly; the stride driver
#   copies it instead of running `_seed_objective!`. The objective
#   field is still threaded through for `CSFootprintResult` metadata,
#   but `_seed_objective!` / `_validate_objective` are bypassed so
#   `CSSeedObjective()` works without tripping its own validation
#   reject (`ObjectiveSeeding.jl:75-79`).
function _build_stride_lambda_panels(final_m, ::Nothing,
                                     objective, mesh, ::Type{FT}) where {FT}
    lambda_panels = ntuple(p -> begin
        a = similar(final_m[p])
        fill!(a, zero(FT))
        a
    end, 6)
    _seed_objective!(lambda_panels, objective, final_m, mesh)
    return lambda_panels
end

function _build_stride_lambda_panels(_final_m, final_adjoint_seed::NTuple{6},
                                     _objective, _mesh, ::Type{FT}) where {FT}
    return _copy_panel_tuple(final_adjoint_seed)
end

"""
    _propagate_mass_checkpoints(panels_m0, panels_am_steps, panels_bm_steps,
                                panels_cm_steps, mesh, scheme, schedule;
                                cfl_limit, flux_scale, dt,
                                diffusion_op, diffusion_workspace,
                                convection_op, convection_forcing)

Forward pass that emits the per-window left-edge checkpoints of
`panels_m`. Returns `Vector{NTuple{6, A}}` of length
`checkpoint_window_count(schedule, nsteps) + 1`: index `w` is
`panels_m` at the start of window `w`, and the final index is the
post-run state.

Implementation note: the forward propagation is driven by
`_record_cs_mass_tape` with `record_ops = false` so the recorder
runs the same `_sweep_x/y/z_panel!` + `fill_panel_halos!` kernels as
the FullCheckpoint path but skips every `_record_sweep!` /
`_stage_panels` / `push!(ops, ...)` site. This bounds peak
propagation memory to the live `panels_m` tuple (plus the
`dummy_rm` workspace inside the recorder), independent of `nsteps`,
while keeping the forward kernel path bit-identical to the
recording path.
"""
function _propagate_mass_checkpoints(panels_m0,
                                     panels_am_steps,
                                     panels_bm_steps,
                                     panels_cm_steps,
                                     mesh::CubedSphereMesh,
                                     scheme::CSAdjointLinearScheme,
                                     schedule::StrideCheckpoint;
                                     cfl_limit,
                                     flux_scale,
                                     dt,
                                     diffusion_op = NoDiffusion(),
                                     diffusion_workspace = nothing,
                                     convection_op = NoConvection(),
                                     convection_forcing = nothing)
    nsteps = length(panels_am_steps)
    nw = checkpoint_window_count(schedule, nsteps)

    initial = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial, mesh; dir = 0)

    checkpoints = Vector{typeof(initial)}(undef, nw + 1)
    checkpoints[1] = initial

    current = initial
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current = _record_cs_mass_tape(
            current,
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            tape_storage = :device,
            step_offset = first(window_range) - 1,
            record_ops = false)
        checkpoints[w + 1] = current
    end
    return checkpoints
end

"""
    _collect_surface_footprints_stride(panels_m0, panels_am_steps, ...,
                                       mesh, scheme, schedule, objective, dt;
                                       cfl_limit, flux_scale,
                                       diffusion_op, diffusion_workspace,
                                       diffusion_meteo,
                                       convection_op, convection_forcing,
                                       convection_workspace,
                                       tape_storage)

Strided-checkpoint variant of `_collect_surface_footprints`. Returns a
`CSFootprintResult` whose `footprints[step]` array agrees with the
`FullCheckpoint` path to bit accuracy on the linear-scheme paths
(`UpwindScheme`, `SlopesScheme(NoLimiter())`, `PPMScheme(NoLimiter())`),
since both routes call the same forward kernels in the same order
and the reverse pass adjoint is exact (not stochastic).

Each window's tape is built with the requested `tape_storage` policy
and dropped after its reverse walk completes. For `tape_storage = :mmap`,
this means peak disk is one window's worth of `records.bin`, not the
whole run's.
"""
function _collect_surface_footprints_stride(panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointLinearScheme,
                                            schedule::StrideCheckpoint,
                                            objective::AbstractCSFootprintObjective,
                                            dt;
                                            cfl_limit,
                                            flux_scale,
                                            diffusion_op = NoDiffusion(),
                                            diffusion_workspace = nothing,
                                            diffusion_meteo = nothing,
                                            convection_op = NoConvection(),
                                            convection_forcing = nothing,
                                            convection_workspace = nothing,
                                            tape_storage = :device,
                                            tape_path::Union{Nothing, AbstractString} = nothing,
                                            final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    # Objective validation is bypassed in the from-seed path because
    # `_validate_objective(::CSSeedObjective)` throws unconditionally
    # (`ObjectiveSeeding.jl:75`); from-seed callers supply the lambda
    # directly and only need the objective field for `CSFootprintResult`
    # metadata.
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    # The stride driver constructs a fresh tape storage per window
    # via `_tape_storage(tape_storage)` and finalize_tape!s it after
    # each window's reverse walk. If the caller passes an
    # *already-constructed* `AbstractCSTapeStorage`, `_tape_storage`
    # is identity — every window would share the same storage, and
    # the second window would throw "MmapCSTapeStorage ... is
    # finalised; cannot allocate new slot" because we finalize after
    # the first reverse walk. Reject the misuse up front with a
    # tape-aware diagnostic.
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "StrideCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window. Pass " *
        "`tape_storage = :mmap` (or similar) instead."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    checkpoints = _propagate_mass_checkpoints(
        panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme, schedule;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        convection_op = convection_op,
        convection_forcing = convection_forcing)

    nw = checkpoint_window_count(schedule, nsteps)
    final_m = checkpoints[nw + 1]

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    @inbounds for w in nw:-1:1
        window_range = checkpoint_window_range(schedule, w, nsteps)
        # Construct the per-window tape storage explicitly so we can
        # finalize it deterministically after the reverse walk. For
        # `:mmap`, this is load-bearing: each window opens its own
        # temp dir + `records.bin`, and leaning on GC could leave
        # multiple per-window dirs live simultaneously on tmpfs.
        # `finalize_tape!` is a generic no-op for `:device` /
        # `:pinned_host` so the call site is uniform. When `tape_path`
        # is supplied, `_build_window_storage` roots each window's
        # `records.bin` under `joinpath(tape_path, "window_NNNNN")`
        # and leaves the directory in place past `finalize_tape!`
        # (user-owned).
        storage_w = _build_window_storage(tape_storage, tape_path,
                                          _stride_window_subdir(w))
        ops_window, _ = _record_cs_mass_tape(
            checkpoints[w],
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            tape_storage = storage_w,
            step_offset = first(window_range) - 1)
        try
            _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                  diffusion_meteo = diffusion_meteo,
                                  convection_workspace = convection_workspace)
        finally
            # Drop the window-scoped tape (mmap close + manifest emit +
            # temp-dir cleanup for `MmapCSTapeStorage`; no-op for
            # `:device` / `:pinned_host` whose state is plain arrays
            # released by GC).
            finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

# Schedule dispatcher used by `cs_surface_emission_footprint` and
# `cs_surface_emission_footprint_from_seed`. A.3a (linear-mass),
# A.3b (nonlinear-PPM tracer), and A.3c (LinRood horizontal) are all
# wired up. The `_from_seed` entry still rejects non-Full schedules
# pending its own follow-up commit (separate seed-driven driver).
function _require_checkpoint_supported(scheme, schedule::AbstractCheckpointSchedule)
    schedule isa FullCheckpoint && return nothing
    scheme isa CSAdjointLinearScheme && return nothing
    scheme isa CSAdjointNonlinearScheme && return nothing
    scheme isa CSAdjointLinRoodScheme && return nothing
    throw(ArgumentError(
        "checkpoint=$(schedule) is not yet supported for scheme " *
        "$(nameof(typeof(scheme)))."))
end

# Per-scheme `tape_path` compatibility. LinRood is pinned to
# `:device` storage (`_linrood_validate_tape_storage` inside
# `_record_cs_linrood_tape`), so a non-nothing `tape_path` is
# meaningless and would only produce empty `records.bin` /
# `manifest.toml` side effects under the user's tree before the
# recorder throws. Reject here, BEFORE `_resolve_tape_path` runs
# `mkpath`, so an invalid request leaves the filesystem untouched.
function _require_tape_path_supported(scheme,
                                       tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    scheme isa CSAdjointLinRoodScheme && throw(ArgumentError(
        "tape_path is not supported with LinRoodPPMScheme; the LinRood " *
        "reverse tape is :device-only. Omit tape_path or pick a non-" *
        "LinRood scheme. (Mmap eviction for LinRood is reserved for a " *
        "Plan 26 follow-up that refactors the LinRood tape.)"))
    return nothing
end

# ---------------------------------------------------------------------------
# Strided checkpoint driver for the nonlinear PPM
# tracer tape.
#
# Differs from the linear-scheme driver above in three ways:
#
# 1. **Two state checkpoints per window.** The tracer tape's reverse
#    walk needs the rm-state replayed alongside panels_m, so the
#    propagation pass saves `(panels_rm, panels_m)` snapshots — not
#    just `panels_m`. `_record_cs_tracer_tape` is called with
#    `record_ops = false`, which still runs the forward diffusion,
#    emission, and convection kernels (those genuinely mutate
#    panels_rm and panels_m forward in the tracer tape, unlike the
#    mass tape).
#
# 2. **`base_emission_rates` is load-bearing.** The tracer-tape
#    recorder accepts a per-step emission rate and adds it to
#    panels_rm at the midpoint. Stride needs to thread the sliced
#    `base_emission_rates` into each window.
#
# 3. **3-tuple from `_record_cs_tracer_tape`.** The recorder returns
#    `(ops, panels_rm, panels_m)`; we use all three.
#
# The reverse walk reuses `_walk_window_reverse!` — the per-record
# dispatch is identical between mass and tracer tapes.
# ---------------------------------------------------------------------------

function _propagate_tracer_checkpoints(panels_rm0, panels_m0,
                                       panels_am_steps,
                                       panels_bm_steps,
                                       panels_cm_steps,
                                       mesh::CubedSphereMesh,
                                       scheme::CSAdjointNonlinearScheme,
                                       schedule::StrideCheckpoint;
                                       cfl_limit,
                                       flux_scale,
                                       dt,
                                       base_emission_rates = nothing,
                                       diffusion_op = NoDiffusion(),
                                       diffusion_workspace = nothing,
                                       diffusion_meteo = nothing,
                                       convection_op = NoConvection(),
                                       convection_forcing = nothing,
                                       convection_workspace = nothing)
    nsteps = length(panels_am_steps)
    nw = checkpoint_window_count(schedule, nsteps)

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m, mesh; dir = 0)

    rm_checkpoints = Vector{typeof(initial_rm)}(undef, nw + 1)
    m_checkpoints  = Vector{typeof(initial_m)}(undef, nw + 1)
    rm_checkpoints[1] = initial_rm
    m_checkpoints[1]  = initial_m

    current_rm = initial_rm
    current_m  = initial_m
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current_rm, current_m = _record_cs_tracer_tape(
            current_rm, current_m,
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = :device,
            step_offset = first(window_range) - 1,
            record_ops = false)
        rm_checkpoints[w + 1] = current_rm
        m_checkpoints[w + 1]  = current_m
    end
    return rm_checkpoints, m_checkpoints
end

"""
    _collect_surface_footprints_stride(panels_rm0, panels_m0, ...,
                                       scheme::CSAdjointNonlinearScheme, ...)

Strided-checkpoint driver for the nonlinear-PPM (monotone-limited)
tracer tape. Returns a `CSFootprintResult` that matches the
`FullCheckpoint` path to bit / floating-point accuracy on the same
`(panels_rm0, panels_m0)` inputs; the same per-window mmap-lifetime
discipline as the linear driver applies (`try / finally` finalize).

`base_emission_rates` is sliced per window. `convection_forcing` and
`diffusion_workspace` are likewise sliced if they are
`AbstractVector`s.
"""
function _collect_surface_footprints_stride(panels_rm0, panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointNonlinearScheme,
                                            schedule::StrideCheckpoint,
                                            objective::AbstractCSFootprintObjective,
                                            dt;
                                            cfl_limit,
                                            flux_scale,
                                            base_emission_rates = nothing,
                                            diffusion_op = NoDiffusion(),
                                            diffusion_workspace = nothing,
                                            diffusion_meteo = nothing,
                                            convection_op = NoConvection(),
                                            convection_forcing = nothing,
                                            convection_workspace = nothing,
                                            tape_storage = :device,
                                            tape_path::Union{Nothing, AbstractString} = nothing,
                                            final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "StrideCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    rm_checkpoints, m_checkpoints = _propagate_tracer_checkpoints(
        panels_rm0, panels_m0,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme, schedule;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace)

    nw = checkpoint_window_count(schedule, nsteps)
    final_m = m_checkpoints[nw + 1]

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    @inbounds for w in nw:-1:1
        window_range = checkpoint_window_range(schedule, w, nsteps)
        storage_w = _build_window_storage(tape_storage, tape_path,
                                          _stride_window_subdir(w))
        ops_window, _, _ = _record_cs_tracer_tape(
            rm_checkpoints[w], m_checkpoints[w],
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = storage_w,
            step_offset = first(window_range) - 1)
        try
            _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                  diffusion_meteo = diffusion_meteo,
                                  convection_workspace = convection_workspace)
        finally
            finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

# ---------------------------------------------------------------------------
# Strided checkpoint driver for the LinRood horizontal
# tape (LinRoodPPMScheme).
#
# LinRood differs from the nonlinear PPM tracer tape in two key ways:
#
# 1. **Storage policy is fixed to `:device`.** The `_CSLinRoodHorizRecord`
#    struct holds raw `NTuple{6, P}` references rather than per-policy
#    slots, so `_record_cs_linrood_tape` rejects non-`:device` storage
#    via `_linrood_validate_tape_storage`. Per-window mmap finalization
#    is therefore a no-op for LinRood; storage construction returns a
#    plain `DeviceCSTapeStorage`.
#
# 2. **Horizontal substep is unsplit.** Each step runs one
#    `_record_linrood_horizontal_substep!` (which itself allocates
#    per-substep face / q_buf scratch — those go out of scope at the
#    substep's function exit, so peak scratch memory is bounded by
#    one substep, independent of `nsteps`). Setting `record_ops =
#    false` further elides the per-substep `panels_rm_tape` /
#    `panels_m_tape` / `panels_q_buf_phase{2,3}` snapshots that would
#    otherwise be retained for the reverse pass.
#
# Structurally identical to `_propagate_tracer_checkpoints` /
# `_collect_surface_footprints_stride(::CSAdjointNonlinearScheme)`
# above — same `(rm, m)`-pair propagation, same `_walk_window_reverse!`
# reverse loop, same sliced per-step kwargs.
# ---------------------------------------------------------------------------

function _propagate_linrood_checkpoints(panels_rm0, panels_m0,
                                        panels_am_steps,
                                        panels_bm_steps,
                                        panels_cm_steps,
                                        mesh::CubedSphereMesh,
                                        scheme::CSAdjointLinRoodScheme,
                                        schedule::StrideCheckpoint;
                                        cfl_limit,
                                        flux_scale,
                                        dt,
                                        base_emission_rates = nothing,
                                        diffusion_op = NoDiffusion(),
                                        diffusion_workspace = nothing,
                                        diffusion_meteo = nothing,
                                        convection_op = NoConvection(),
                                        convection_forcing = nothing,
                                        convection_workspace = nothing)
    nsteps = length(panels_am_steps)
    nw = checkpoint_window_count(schedule, nsteps)

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    rm_checkpoints = Vector{typeof(initial_rm)}(undef, nw + 1)
    m_checkpoints  = Vector{typeof(initial_m)}(undef, nw + 1)
    rm_checkpoints[1] = initial_rm
    m_checkpoints[1]  = initial_m

    current_rm = initial_rm
    current_m  = initial_m
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current_rm, current_m = _record_cs_linrood_tape(
            current_rm, current_m,
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = :device,
            step_offset = first(window_range) - 1,
            record_ops = false)
        rm_checkpoints[w + 1] = current_rm
        m_checkpoints[w + 1]  = current_m
    end
    return rm_checkpoints, m_checkpoints
end

"""
    _collect_surface_footprints_stride(panels_rm0, panels_m0, ...,
                                       scheme::CSAdjointLinRoodScheme, ...)

Strided-checkpoint driver for the LinRood horizontal tape. Storage is
fixed to `:device` (see `_linrood_validate_tape_storage`); passing
`tape_storage = :mmap` raises an `ArgumentError` deep in the recorder
on the first window's reverse-pass call. Reject explicitly up front
so the failure mode is loud and stride-aware.
"""
function _collect_surface_footprints_stride(panels_rm0, panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointLinRoodScheme,
                                            schedule::StrideCheckpoint,
                                            objective::AbstractCSFootprintObjective,
                                            dt;
                                            cfl_limit,
                                            flux_scale,
                                            base_emission_rates = nothing,
                                            diffusion_op = NoDiffusion(),
                                            diffusion_workspace = nothing,
                                            diffusion_meteo = nothing,
                                            convection_op = NoConvection(),
                                            convection_forcing = nothing,
                                            convection_workspace = nothing,
                                            tape_storage = :device,
                                            tape_path::Union{Nothing, AbstractString} = nothing,
                                            final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "StrideCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device for LinRood), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window."))
    # LinRood's reverse path only accepts `:device`; rather than wait
    # for the recorder to throw inside the first window, surface it
    # here with a stride-aware diagnostic.
    tape_storage === :device || throw(ArgumentError(
        "LinRoodPPMScheme + StrideCheckpoint requires tape_storage = :device " *
        "(got $(repr(tape_storage))). The `_CSLinRoodHorizRecord` struct " *
        "holds device-resident panel tuples directly; mmap eviction is " *
        "reserved for a Plan 26 follow-up that refactors the LinRood tape."))
    # `tape_path` only makes sense for `:mmap` storage. LinRood is
    # already pinned to `:device` above, so any non-nothing path is
    # incompatible — reject loudly here instead of silently ignoring
    # the kwarg.
    tape_path === nothing || throw(ArgumentError(
        "LinRoodPPMScheme does not support tape_path: storage is fixed " *
        "to :device. Use StrideCheckpoint with the nonlinear PPM scheme " *
        "(tape_storage = :mmap) if disk-backed tapes are required."))

    rm_checkpoints, m_checkpoints = _propagate_linrood_checkpoints(
        panels_rm0, panels_m0,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme, schedule;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace)

    nw = checkpoint_window_count(schedule, nsteps)
    final_m = m_checkpoints[nw + 1]

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    @inbounds for w in nw:-1:1
        window_range = checkpoint_window_range(schedule, w, nsteps)
        storage_w = _tape_storage(tape_storage)
        ops_window, _, _ = _record_cs_linrood_tape(
            rm_checkpoints[w], m_checkpoints[w],
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = storage_w,
            step_offset = first(window_range) - 1)
        try
            _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                  diffusion_meteo = diffusion_meteo,
                                  convection_workspace = convection_workspace)
        finally
            finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

# ---------------------------------------------------------------------------
# RevolveCheckpoint driver.
#
# Recursive-bisection variant of Griewank-Walther Revolve. The reverse
# pass walks the step range via depth-first bisection:
#
#   reverse_range!(state, lo, hi):
#       if hi == lo:   return
#       if hi == lo+1: record one step from state, walk reverse, drop
#       else:
#           mid = (lo + hi) ÷ 2
#           state_at_mid = propagate(state, lo+1:mid)  # record_ops=false
#           reverse_range!(state_at_mid, mid, hi)
#           reverse_range!(state,        lo, mid)
#
# Snapshot memory is the recursion depth — `ceil(log2(nsteps))` copies
# of the per-scheme state tuple (panels_m for linear; (panels_rm,
# panels_m) for nonlinear / LinRood). Recorder calls always copy
# their inputs internally, so the snapshot at a frame's `lo` lives in
# that frame's local `state` binding without an explicit save/restore
# step.
#
# Scope (this commit): bisection only. Optimal binomial splits
# (Griewank-Walther Algorithm 799) are a future promotion behind the
# same `RevolveCheckpoint` API.
# ---------------------------------------------------------------------------

function _collect_surface_footprints_revolve(panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointLinearScheme,
                                             ::RevolveCheckpoint,
                                             objective::AbstractCSFootprintObjective,
                                             dt;
                                             cfl_limit,
                                             flux_scale,
                                             diffusion_op = NoDiffusion(),
                                             diffusion_workspace = nothing,
                                             diffusion_meteo = nothing,
                                             convection_op = NoConvection(),
                                             convection_forcing = nothing,
                                             convection_workspace = nothing,
                                             tape_storage = :device,
                                             tape_path::Union{Nothing, AbstractString} = nothing,
                                             final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    # Initial state (halo-filled copy of input).
    initial_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_m, mesh; dir = 0)

    # Forward propagation to compute the final state (used for the
    # objective seed). The propagation pass runs the same kernels as
    # the per-step record but with `record_ops = false`, so the final
    # state matches the FullCheckpoint forward path bit-for-bit.
    _, final_m = _record_cs_mass_tape(
        initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    # Recursive bisection driver. `state_m` is the panels_m state at
    # step `lo`; the function reverses steps [lo+1, hi] in place
    # against `lambda_panels`, accumulating into `footprints[]`.
    function reverse_range!(state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _build_window_storage(tape_storage, tape_path,
                                              _revolve_step_subdir(hi))
            ops_window, _ = _record_cs_mass_tape(
                state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            # Propagate state from lo to mid (no record). The recorder
            # copies its input internally, so the original `state_m`
            # in this frame still points to the lo state — usable for
            # the second recursive call below.
            _, state_m_at_mid = _record_cs_mass_tape(
                state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_m_at_mid, mid, hi)
            reverse_range!(state_m,        lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

function _collect_surface_footprints_revolve(panels_rm0, panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointNonlinearScheme,
                                             ::RevolveCheckpoint,
                                             objective::AbstractCSFootprintObjective,
                                             dt;
                                             cfl_limit,
                                             flux_scale,
                                             base_emission_rates = nothing,
                                             diffusion_op = NoDiffusion(),
                                             diffusion_workspace = nothing,
                                             diffusion_meteo = nothing,
                                             convection_op = NoConvection(),
                                             convection_forcing = nothing,
                                             convection_workspace = nothing,
                                             tape_storage = :device,
                                             tape_path::Union{Nothing, AbstractString} = nothing,
                                             final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    # Forward propagation to compute the final state for seeding.
    _, _, final_m = _record_cs_tracer_tape(
        initial_rm, initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    function reverse_range!(state_rm, state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _build_window_storage(tape_storage, tape_path,
                                              _revolve_step_subdir(hi))
            ops_window, _, _ = _record_cs_tracer_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, hi:hi),
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                convection_workspace = convection_workspace,
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            _, state_rm_at_mid, state_m_at_mid = _record_cs_tracer_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, (lo + 1):mid),
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                convection_workspace = convection_workspace,
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_rm_at_mid, state_m_at_mid, mid, hi)
            reverse_range!(state_rm,         state_m,         lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_rm, initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

function _collect_surface_footprints_revolve(panels_rm0, panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointLinRoodScheme,
                                             ::RevolveCheckpoint,
                                             objective::AbstractCSFootprintObjective,
                                             dt;
                                             cfl_limit,
                                             flux_scale,
                                             base_emission_rates = nothing,
                                             diffusion_op = NoDiffusion(),
                                             diffusion_workspace = nothing,
                                             diffusion_meteo = nothing,
                                             convection_op = NoConvection(),
                                             convection_forcing = nothing,
                                             convection_workspace = nothing,
                                             tape_storage = :device,
                                             tape_path::Union{Nothing, AbstractString} = nothing,
                                             final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device for LinRood), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_storage === :device || throw(ArgumentError(
        "LinRoodPPMScheme + RevolveCheckpoint requires tape_storage = :device " *
        "(got $(repr(tape_storage))). The `_CSLinRoodHorizRecord` struct " *
        "holds device-resident panel tuples directly; mmap eviction is " *
        "reserved for a Plan 26 follow-up that refactors the LinRood tape."))
    tape_path === nothing || throw(ArgumentError(
        "LinRoodPPMScheme does not support tape_path: storage is fixed " *
        "to :device. Use RevolveCheckpoint with the nonlinear PPM scheme " *
        "(tape_storage = :mmap) if disk-backed tapes are required."))

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    _, _, final_m = _record_cs_linrood_tape(
        initial_rm, initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    function reverse_range!(state_rm, state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _tape_storage(tape_storage)
            ops_window, _, _ = _record_cs_linrood_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, hi:hi),
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                convection_workspace = convection_workspace,
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            _, state_rm_at_mid, state_m_at_mid = _record_cs_linrood_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, (lo + 1):mid),
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                convection_workspace = convection_workspace,
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_rm_at_mid, state_m_at_mid, mid, hi)
            reverse_range!(state_rm,         state_m,         lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_rm, initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end
