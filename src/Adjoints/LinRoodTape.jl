# ---------------------------------------------------------------------------
# LinRood adjoint tape integration (Plan 25 Commit 6).
#
# Wires the per-kernel LinRood adjoints shipped in Plan 25 Commits 1–5
# (in `src/Operators/Advection/linrood_adjoint_kernels.jl`) into the
# CS surface-emission-footprint reverse pass managed by Adjoints.jl.
# Provides:
#
#   * `_CSLinRoodHorizRecord` — per-substep tape record holding all 6
#     panels' input state and the two intermediate `q_buf` snapshots
#     needed by the reverse pass.
#   * `_record_cs_linrood_tape` — forward recording function that
#     replays `fv_tp_2d_cs!` phase-by-phase, captures snapshots, and
#     returns the operations list + final air-mass panels.
#   * `_apply_cs_linrood_horizontal_adjoint!` — reverse of one
#     `_CSLinRoodHorizRecord`: per-panel single-panel adjoint
#     composition (Plan 25 Commit 4) followed by
#     `_adjoint_fill_panel_halos!` for cross-panel halo
#     redistribution.
#
# Limitations carried forward (Plan 25 NOTES):
#   * `copy_corners!` reverse not implemented; the small contribution
#     from corner halos to the gradient is treated as zero. Real-data
#     impact is concentrated near panel corners and decays inward.
#
# Lifted limitations:
#   * ORD ∈ {5, 7} (LinRoodPPMScheme(5) and LinRoodPPMScheme(7) both
#     supported as of Plan-25 Commit 3b, 2026-05-15). `_CSLinRoodHorizRecord`
#     binds ORD as a type parameter; the reverse pass reads it via
#     dispatch and forwards `Val(ORD)` to the face-kernel adjoints.
# ---------------------------------------------------------------------------

# Forward kernels + adjoint wrappers from Operators.Advection. Imported at
# Adjoints.jl module scope; this file is `include`d inside that module.

# LinRood records hold raw panel tuples rather than per-policy tape
# slots, so any non-`:device` storage request is currently a footgun:
# the forward pass would silently keep the tape on the source backend
# while the user thought they had opted into mmap eviction. Reject
# explicitly until storage plumbing reaches `_CSLinRoodHorizRecord`.
_linrood_validate_tape_storage(::DeviceCSTapeStorage) = nothing
function _linrood_validate_tape_storage(storage::Symbol)
    storage === :device || _linrood_storage_unsupported(storage)
    return nothing
end
_linrood_validate_tape_storage(storage) = _linrood_storage_unsupported(storage)

function _linrood_storage_unsupported(storage)
    throw(ArgumentError(
        "LinRoodPPMScheme reverse tape currently only supports " *
        "tape_storage = :device / DeviceCSTapeStorage(); got " *
        repr(storage) * ". The LinRood per-substep record stores " *
        "panel tuples directly rather than per-policy slots, so " *
        "mmap / pinned-host eviction is not yet wired through " *
        "(Plan 26 follow-up)."))
end

# Per-substep LinRood horizontal tape record. The forward state is
# stored ONCE per substep for all six panels. `ORD` (5 or 7) binds the
# record to the LinRood scheme order that built it; the reverse pass
# (`_apply_cs_linrood_horizontal_adjoint!`) reads it from the type and
# dispatches the face-kernel adjoints to the matching ORD=5 or ORD=7
# kernel — guaranteeing the adjoint matches the forward path at the
# panel-edge boundary correction.
struct _CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD}
    panels_rm    :: NTuple{6, P}     # rm at substep start, post-halo-fill
    panels_m     :: NTuple{6, P}     # m  at substep start, post-halo-fill
    panels_q_buf_phase2 :: NTuple{6, P}   # state B (post-phase-1 pre_advect_y)
    panels_q_buf_phase3 :: NTuple{6, P}   # state C (post-phase-2 pre_advect_x)
    panels_fx_in  :: NTuple{6, A3x}
    panels_fx_out :: NTuple{6, A3x}
    panels_fy_in  :: NTuple{6, A3y}
    panels_am     :: NTuple{6, A3x}
    panels_bm     :: NTuple{6, A3y}
    flux_scale    :: FT
end

# Run one LinRood horizontal substep across all six panels, replicating
# the forward `fv_tp_2d_cs!` (LinRood.jl:695-779). Updates `panels_rm`,
# `panels_m` in place. With `record_ops = true` (default) captures
# per-phase snapshots and returns a `_CSLinRoodHorizRecord` for the
# reverse pass; with `record_ops = false` (used by the strided
# checkpoint propagation pass — Plan 26 A.3c) skips every state
# snapshot and returns `nothing`, leaving only the face / q_buf
# scratch buffers that are required to run the kernels themselves
# (they go out of scope at function exit, so peak memory is bounded
# by one substep's worth of scratch rather than the full tape).
function _record_linrood_horizontal_substep!(
    panels_rm, panels_m,
    panels_am, panels_bm,
    mesh::CubedSphereMesh{FT},
    flux_scale;
    record_ops::Bool = true,
    ord::Val{ORD} = Val(5),
) where {FT, ORD}
    Nc = mesh.Nc; Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    N = Nc + 2 * Hp
    backend = get_backend(panels_rm[1])

    init_k!    = _init_q_buf_kernel!(backend, 256)
    y_face_k!  = _ppm_y_face_kernel!(backend, 256)
    x_face_k!  = _ppm_x_face_kernel!(backend, 256)
    xq_face_k! = _ppm_x_face_from_q_kernel!(backend, 256)
    yq_face_k! = _ppm_y_face_from_q_kernel!(backend, 256)
    pre_y_k!   = _pre_advect_y_kernel!(backend, 256)
    pre_x_k!   = _pre_advect_x_kernel!(backend, 256)
    update_k!  = _linrood_update_kernel!(backend, 256)

    # ── Halo fills (pre-phase-1) ─────────────────────────────────────
    fill_panel_halos!(panels_rm, mesh)
    fill_panel_halos!(panels_m,  mesh)
    copy_corners!(panels_rm, mesh, 2)
    copy_corners!(panels_m,  mesh, 2)

    # Capture rm/m state at the start of the substep (post halo + corner).
    # Only allocated in recording mode — propagation mode (`record_ops =
    # false`) skips these snapshots since the reverse pass doesn't run.
    panels_rm_tape = record_ops ? ntuple(p -> copy(panels_rm[p]), Val(6)) : nothing
    panels_m_tape  = record_ops ? ntuple(p -> copy(panels_m[p]),  Val(6)) : nothing

    # Allocate per-panel face / q_buf buffers.
    panels_fy_in  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc, Nc + 1, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fy_out = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc, Nc + 1, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fx_in  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc + 1, Nc, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_fx_out = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (Nc + 1, Nc, Nz)); fill!(b, zero(FT)); b
    end, Val(6))
    panels_q_buf  = ntuple(p -> begin
        b = similar(panels_rm[p], FT, (N, N, Nz)); fill!(b, zero(FT)); b
    end, Val(6))

    # ── Phase 1: init q_buf, y_face, pre_y ───────────────────────────
    for p in 1:6
        init_k!(panels_q_buf[p], panels_rm[p], panels_m[p];
                ndrange=(N, N, Nz))
    end
    synchronize(backend)
    for p in 1:6
        y_face_k!(panels_fy_in[p], panels_rm[p], panels_m[p], panels_bm[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
        pre_y_k!(panels_q_buf[p], panels_rm[p], panels_m[p], panels_bm[p],
                 panels_fy_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # Snapshot q_buf state B (post-phase-1). Skipped in propagation mode.
    panels_q_buf_phase2 = record_ops ?
        ntuple(p -> copy(panels_q_buf[p]), Val(6)) : nothing

    # ── Phase 2: x-corners, xq_face / x_face, re-init, pre_x ─────────
    copy_corners!(panels_q_buf, mesh, 1)
    copy_corners!(panels_rm,    mesh, 1)
    copy_corners!(panels_m,     mesh, 1)

    for p in 1:6
        xq_face_k!(panels_fx_out[p], panels_q_buf[p], panels_am[p], panels_m[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
        x_face_k!(panels_fx_in[p], panels_rm[p], panels_m[p], panels_am[p],
                  Hp, Nc, Val(ORD); ndrange=(Nc + 1, Nc, Nz))
    end
    synchronize(backend)

    for p in 1:6
        init_k!(panels_q_buf[p], panels_rm[p], panels_m[p];
                ndrange=(N, N, Nz))
    end
    synchronize(backend)
    for p in 1:6
        pre_x_k!(panels_q_buf[p], panels_rm[p], panels_m[p], panels_am[p],
                 panels_fx_in[p], Hp; ndrange=(Nc, Nc, Nz))
    end
    synchronize(backend)

    # Snapshot q_buf state C (post-phase-2). Skipped in propagation mode.
    panels_q_buf_phase3 = record_ops ?
        ntuple(p -> copy(panels_q_buf[p]), Val(6)) : nothing

    # ── Phase 3: y-corners, yq_face, update ──────────────────────────
    copy_corners!(panels_q_buf, mesh, 2)

    for p in 1:6
        yq_face_k!(panels_fy_out[p], panels_q_buf[p], panels_bm[p], panels_m[p],
                   Hp, Nc, Val(ORD); ndrange=(Nc, Nc + 1, Nz))
    end
    synchronize(backend)

    # Apply the update in-place using a temporary destination.
    rm_buf = similar(panels_rm[1]); fill!(rm_buf, zero(FT))
    m_buf  = similar(panels_m[1]);  fill!(m_buf,  zero(FT))
    for p in 1:6
        update_k!(rm_buf, m_buf,
                  panels_rm[p], panels_m[p], panels_am[p], panels_bm[p],
                  panels_fx_in[p], panels_fx_out[p],
                  panels_fy_in[p], panels_fy_out[p], Hp;
                  ndrange=(Nc, Nc, Nz))
        synchronize(backend)
        @inbounds for k in 1:Nz, j in (Hp + 1):(Hp + Nc), i in (Hp + 1):(Hp + Nc)
            panels_rm[p][i, j, k] = rm_buf[i, j, k]
            panels_m[p][i, j, k]  = m_buf[i, j, k]
        end
    end

    record_ops || return nothing

    A3  = typeof(panels_rm_tape[1])
    A3x = typeof(panels_fx_in[1])
    A3y = typeof(panels_fy_in[1])
    P   = A3
    return _CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD}(
        panels_rm_tape, panels_m_tape,
        panels_q_buf_phase2, panels_q_buf_phase3,
        panels_fx_in, panels_fx_out, panels_fy_in,
        panels_am, panels_bm,
        FT(flux_scale),
    )
end

# Reverse of one LinRood horizontal substep. Mutates the `lambda_panels_rm`,
# `lambda_panels_m` adjoint accumulators IN PLACE. The record's `ORD` type
# parameter selects the face-kernel adjoint variant (Val(5) or Val(7))
# so the reverse pass matches the forward path's panel-edge boundary
# behaviour.
function _apply_cs_linrood_horizontal_adjoint!(
    lambda_panels_rm, lambda_panels_m,
    record::_CSLinRoodHorizRecord{FT, A3, A3x, A3y, P, ORD},
    mesh::CubedSphereMesh{FT},
) where {FT, A3, A3x, A3y, P, ORD}
    # Step 1: per-panel single-panel composition (Plan 25 Commit 4).
    # Each panel's lambda_rm / lambda_m accumulate contributions to
    # their own interior + halo from the panel's own kernel adjoints.
    for p in 1:6
        sub_lambda_rm = similar(lambda_panels_rm[p])
        sub_lambda_m  = similar(lambda_panels_m[p])
        fill!(sub_lambda_rm, zero(FT))
        fill!(sub_lambda_m,  zero(FT))
        # The substep adjoint maps lambda(rm_new, m_new) — which is the
        # CURRENT lambda_panels_rm[p], lambda_panels_m[p] — into
        # (sub_lambda_rm, sub_lambda_m), the adjoint w.r.t. the
        # substep INPUT state (rm0, m0 of the substep).
        apply_linrood_horizontal_adjoint_single_panel!(
            sub_lambda_rm, sub_lambda_m,
            lambda_panels_rm[p], lambda_panels_m[p],
            record.panels_rm[p], record.panels_m[p],
            record.panels_am[p], record.panels_bm[p],
            record.panels_q_buf_phase2[p], record.panels_q_buf_phase3[p],
            record.panels_fx_in[p], record.panels_fx_out[p],
            record.panels_fy_in[p],
            mesh, Val(ORD),
        )
        # Carry-over: substep output's halo lambda is NOT overwritten
        # by the substep update (which only touches interior cells).
        # Add the halo carry from the OUTPUT lambda back into the
        # substep-input adjoint.
        Nc = mesh.Nc; Hp = mesh.Hp
        @inbounds for k in axes(lambda_panels_rm[p], 3),
                      j in axes(lambda_panels_rm[p], 2),
                      i in axes(lambda_panels_rm[p], 1)
            is_interior = (Hp + 1 <= i <= Hp + Nc) && (Hp + 1 <= j <= Hp + Nc)
            if !is_interior
                sub_lambda_rm[i, j, k] += lambda_panels_rm[p][i, j, k]
                sub_lambda_m[i, j, k]  += lambda_panels_m[p][i, j, k]
            end
        end
        # Replace the running lambda with the substep-input adjoint.
        copyto!(lambda_panels_rm[p], sub_lambda_rm)
        copyto!(lambda_panels_m[p],  sub_lambda_m)
    end

    # Step 2: cross-panel halo adjoint. The forward path filled halos at
    # the start of the substep; the reverse aggregates each panel's
    # halo lambda contributions into the corresponding neighbour
    # panel's interior cells.
    _adjoint_fill_panel_halos!(lambda_panels_rm, mesh; dir=0)
    _adjoint_fill_panel_halos!(lambda_panels_m,  mesh; dir=0)
    return nothing
end

# ---------------------------------------------------------------------------
# Top-level tape recording for LinRoodPPMScheme. Records one LinRood
# horizontal record + Z sweeps per substep, plus optional diffusion +
# convection records (matching the existing CS tracer-tape contract).
# ---------------------------------------------------------------------------
function _record_cs_linrood_tape(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh{FT},
                                  scheme::LinRoodPPMScheme{ORD};
                                  flux_scale = one(FT),
                                  dt = one(FT),
                                  cfl_limit = 0.95,
                                  base_emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing,
                                  tape_storage = :device,
                                  step_offset::Int = 0,
                                  record_ops::Bool = true) where {FT, ORD}
    _ = cfl_limit  # LinRood doesn't subcycle horizontally — single substep per step

    # `step_offset` and `record_ops` are Plan 26 A.3c additions for
    # strided checkpointing — `step_offset` shifts `_CSMidpointRecord`
    # indices into absolute step numbers for window invocations;
    # `record_ops = false` is the propagation pass that runs every
    # forward kernel (horizontal substep, Z half-sweeps, diffusion,
    # emissions, convection) but elides each `_stage_panels_strict` /
    # `push!(ops, ...)` site. Default 0 / true keeps the FullCheckpoint
    # path bit-exact.

    # ORD ∈ {5, 7}: the adjoint kernels carry `Val(ORD)` end-to-end
    # (record type binds it; `_apply_cs_linrood_horizontal_adjoint!`
    # reads it from the record). The ORD=7 reverse-pass applies the
    # discontinuous-edge boundary correction at panel-edge faces
    # (`face_idx ∈ {1, Nc+1}`) so the tape and the FD reference match.
    (ORD == 5 || ORD == 7) || throw(ArgumentError(
        "LinRoodPPMScheme adjoint tape supports ORD ∈ {5, 7}; got " *
        "ORD=$(ORD)."))

    # LinRoodPPMScheme stages its forward state through
    # `_stage_panels_strict` (which hardcodes `DeviceCSTapeStorage()`)
    # — the `_CSLinRoodHorizRecord` struct holds raw `NTuple{6, P}`
    # references rather than per-policy slots. Until the LinRood tape
    # is refactored to plumb the storage policy through (Plan 26
    # follow-up), any non-`:device` storage request would be silently
    # ignored, leaving the mmap tape with `cursor=0, records=0` and
    # the LinRood tape entirely device-resident — a latent OOM trap
    # for large LinRood footprints. Reject explicitly so the failure
    # mode is loud. Skipped in propagation mode (no tape needed).
    record_ops && _linrood_validate_tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    dt_ft = FT(dt)

    # Mutable copies of the panel state — updated in place by the
    # forward replay.
    panels_rm = ntuple(p -> copy(panels_rm0[p]), Val(6))
    panels_m  = ntuple(p -> copy(panels_m0[p]),  Val(6))

    ops = Any[]
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_rm[1], 3)

    # PPM(MonotoneLimiter) is the Strang Z reverse scheme — LinRood
    # uses the same _sweep_z_panel! Z kernel as PPM(MonotoneLimiter).
    z_scheme = PPMScheme(MonotoneLimiter())

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]

        # Production LinRood Strang palindrome (LinRood.jl:921-935,
        # `_strang_split_linrood_ppm_cs!`):
        #     H → Z_half → midpoint/diffusion/emissions → Z_half → H
        # The tape mirrors this exactly so that the FD-reference forward
        # in `_linrood_run_forward_step!` and the recorded forward are
        # the same operator (up to numerical rounding).

        absolute_step = step + step_offset

        # LinRood horizontal substep (first half of the palindrome).
        record_a = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale);
            record_ops = record_ops, ord = Val(ORD))
        record_ops && push!(ops, record_a)

        # Z half-sweep.
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                            z_scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz;
                            flux_scale = FT(flux_scale))
        end
        if record_ops
            push!(ops, _CSSweepRecord(:z, z_scheme,
                                      _stage_panels_strict(panels_m),
                                      _stage_panels_strict(panels_rm),
                                      panels_cm, FT(flux_scale)))
        end

        # Diffusion + midpoint + emissions (between the two Z halves).
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                    "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            record_ops && push!(ops, _CSMidpointRecord(absolute_step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            half_dt = dt_ft / FT(2)
            if record_ops
                panels_m_midpoint = _stage_panels_strict(panels_m)
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
            end
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
            record_ops && push!(ops, _CSMidpointRecord(absolute_step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
            if record_ops
                push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                              panels_m_midpoint, half_dt))
            end
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
        end

        # Z half-sweep (second half).
        for p in 1:6
            _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                            z_scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz;
                            flux_scale = FT(flux_scale))
        end
        if record_ops
            push!(ops, _CSSweepRecord(:z, z_scheme,
                                      _stage_panels_strict(panels_m),
                                      _stage_panels_strict(panels_rm),
                                      panels_cm, FT(flux_scale)))
        end

        # LinRood horizontal substep (second half of the palindrome).
        record_b = _record_linrood_horizontal_substep!(
            panels_rm, panels_m, panels_am, panels_bm, mesh, FT(flux_scale);
            record_ops = record_ops, ord = Val(ORD))
        record_ops && push!(ops, record_b)

        # Convection (optional, post-transport).
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            if record_ops
                push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                               _stage_panels_strict(panels_m),
                                               dt_ft))
            end
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt_ft,
                                          convection_workspace, mesh)
        end
    end

    return ops, panels_rm, panels_m
end

# Internal helper: build a DeviceCSTapeStorage-staged version of
# panels for the tape. The existing `_stage_panels(storage, panels)`
# requires a `storage` argument; we don't have one in scope for the
# LinRood path because the standalone API doesn't expose it. Stage
# strictly: just copy in-place onto the same backend.
function _stage_panels_strict(panels::NTuple{6})
    return _stage_panels(DeviceCSTapeStorage(), panels)
end

# ---------------------------------------------------------------------------
# Forward driver for the FD-reference path inside `_run_cs_footprint_forward`
# / `_run_cs_observations_forward`. The standard `strang_split_cs!` doesn't
# know how to dispatch LinRoodPPMScheme (no per-direction face kernels);
# this helper bridges to `strang_split_linrood_ppm!` which IS the right
# forward for LinRood.
# ---------------------------------------------------------------------------
function _linrood_run_forward_step!(panels_rm, panels_m,
                                     panels_am, panels_bm, panels_cm,
                                     mesh::CubedSphereMesh{FT},
                                     scheme::LinRoodPPMScheme{ORD},
                                     ws::CSAdvectionWorkspace,
                                     midpoint!) where {FT, ORD}
    Nz = size(panels_rm[1], 3)
    # The user-facing `strang_split_linrood_ppm!` requires a
    # `LinRoodWorkspace`. The FD-reference path doesn't carry one;
    # allocate a fresh, backend-aware workspace per call. This is
    # only on the FD path (not on the production tape recording),
    # so the allocation overhead is acceptable.
    array_type = typeof(parent(panels_rm[1]))
    ws_lr = LinRoodWorkspace(mesh; FT = FT, Nz = Nz, array_type = array_type)
    # The Strang palindrome midpoint! callback is applied between the
    # two halves; emulate it by manually invoking Z + horiz + (midpoint)
    # + Z + horiz like `strang_split_linrood_ppm!` does, but with
    # midpoint! inserted in the centre. For the LinRood path the
    # midpoint goes between the two Z sweeps inside the function (see
    # `_strang_split_linrood_ppm_cs!`).
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm,
                                      mesh, Val(ORD), ws, ws_lr)
    _sweep_z!(panels_rm, panels_m, panels_cm, mesh, ws)
    if midpoint! !== nothing
        midpoint!()
    end
    _sweep_z!(panels_rm, panels_m, panels_cm, mesh, ws)
    fv_tp_2d_cs!(panels_rm, panels_m, panels_am, panels_bm,
                                      mesh, Val(ORD), ws, ws_lr)
    return nothing
end
