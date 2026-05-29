# ===========================================================================
# ERA5 N320 → C180 cubed-sphere transport-binary writer.
#
# Drives one UTC day end-to-end:
#
#   per window (24 hourly):
#     1. Run the ERA5 per-window pipeline (B/C/D/E from sources/era5.jl):
#        synthesise U/V/T/Q/PS on N320, derive dry-basis mass on the source
#        mesh, read UDMF/DDMF/UDRF/DDRF convection (optional), and
#        conservatively regrid PS / U / V / T / Q to the C180 target.
#     2. Re-derive dry-mass on C180 from the regridded moist PS + Q so the
#        target-side column closure Σ_k DELP_dry = PS_dry holds to roundoff.
#     3. Rotate cell-centre winds geographic → panel-local using the CS
#        tangent basis.
#     4. Reconstruct Arakawa-C face mass fluxes (am, bm) from rotated U/V
#        and panel DELP via the existing CS helper.
#
#   per window transition (windows 2..24):
#     5. Read the next window's pipeline output so we can close continuity
#        against the explicit endpoint-mass target.
#     6. Poisson-balance the current window's horizontal fluxes against
#        the next-window mass tendency (column or global balance, same
#        knob as the LL→CS path).
#     7. Diagnose cm from the balanced fluxes + endpoint mass tendency.
#     8. Verify the per-substep positivity gate and the write-time replay
#        gate. Update the worst-case accumulators.
#     9. Convert the next-window mass target into the forward `dm` payload
#        and stream-write the window to the staging binary.
#
# The final window writes with `dm = 0` (no next-day endpoint look-ahead
# from a separate file yet — the warning is emitted at write time, mirroring
# the `allow_terminal_zero_tendency` path in `regrid_ll_binary_to_cs`).
#
# Surface (PBL) payload sections are intentionally not written by this
# branch — the ERA5 N320 source doesn't expose them yet.
#
# TM5 convection (entu/detu/entd/detd) IS now written when
# `include_convection = true`. The N320 forecast (UDMF/DDMF/UDRF/DDRF)
# is converted to TM5 fields via `ec2tm_from_rates!` on each column,
# then regridded to C180 via the existing conservative path, then
# attached to the per-window writer payload as `window.tm5_fields`.
# CMFMC/DTRAIN is NOT written from this preprocessor; consumers that
# want CMFMC should read from a GEOS-IT binary or convert from TM5
# downstream.
# ===========================================================================

"""
    _fill_cs_mass_delta_payload!(dm_payload, m_cur, m_next)

Fill the on-disk forward endpoint-difference payload without mutating the
absolute next endpoint. The ERA5 writer's sliding-window loop swaps
`m_next` into `m_cur` after each write, so in-place conversion of the endpoint
would make the following window start from `m_next - m_cur`.
"""
function _fill_cs_mass_delta_payload!(dm_payload::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_next::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(dm_payload[p])
            dm_payload[p][idx] = m_next[p][idx] - m_cur[p][idx]
        end
    end
    return dm_payload
end

"""
    _next_day_core_only_handle(handles)

Build a minimal next-day `ERA5GRIBDayHandles` that points only at the recorded
`next_core_path`. Used for the final-window mass-endpoint look-ahead, which
needs PS + Q from hour 0 of `date + 1` but not the convection or surface
streams. Returns `nothing` when `handles.next_core_path` is absent (archive
boundary day).
"""
function _next_day_core_only_handle(handles::ERA5GRIBDayHandles)
    handles.next_core_path === nothing && return nothing
    return ERA5GRIBDayHandles{typeof(handles.settings)}(
        handles.settings,
        handles.date + Day(1),
        handles.next_core_path,
        nothing,  # convection_path — not needed for mass endpoint
        nothing,  # surface_path
        nothing,  # next_core_path (chain stops here)
        nothing,  # prev_convection_path
    )
end

"""
    process_era5_n320_to_cs_day(date, settings, target_grid;
                                 out_path,
                                 FT = Float32,
                                 mass_basis = :dry,
                                 Nz = ERA5_NATIVE_LEVEL_COUNT,
                                 dt_met_seconds = 3600.0,
                                 steps_per_window = 8,
                                 cs_balance_tol = 1e-14,
                                 cs_balance_project_every = 50,
                                 positivity_cfl_limit = 0.95,
                                 cache_dir = nothing,
                                 include_convection = false)

Generate a v4 cubed-sphere transport binary for one UTC `date` from the
ERA5 native-GRIB source described by `settings`, written to `out_path`.

`mass_basis` is fixed to `:dry` here — the writer pulls dry-basis layer
mass and dry surface pressure (re-derived on C180 from regridded PS + Q).
A `:moist` request would need the moist-basis runtime contract, which is
not the project's runtime default (`feedback_dry_basis_default.md`).

`steps_per_window` controls the number of Strang substeps per met window
written into the binary. Each `am` / `bm` per-face slot stores the
substep-mass amount; the runtime CFL is `cfl = am[i,j,k] / m[i-1,j,k]`
per substep so a larger value softens the per-substep CFL at the cost of
a larger binary.

The function stages writes to `out_path.tmp` and promotes to `out_path`
on success — a partial run leaves no usable file at the requested path.
"""
function process_era5_n320_to_cs_day(date::Date,
                                       settings::ERA5N320Settings,
                                       target_grid::CubedSphereTargetGeometry{FT};
                                       out_path::AbstractString,
                                       Nz::Integer = ERA5_NATIVE_LEVEL_COUNT,
                                       mass_basis::Symbol = :dry,
                                       dt_met_seconds::Real = 3600.0,
                                       steps_per_window::Integer = 8,
                                       cs_balance_tol::Real = 1e-14,
                                       cs_balance_project_every::Integer = 50,
                                       positivity_cfl_limit::Real = 0.95,
                                       cache_dir::Union{Nothing, AbstractString} = nothing,
                                       include_convection::Bool = false) where FT
    mass_basis === :dry ||
        throw(ArgumentError("ERA5 N320 → CS writer only supports mass_basis=:dry on this branch; got $(mass_basis)"))
    Nz_int = Int(Nz)
    steps_per_met = Int(steps_per_window)
    steps_per_met >= 1 || throw(ArgumentError("steps_per_window must be ≥ 1; got $(steps_per_met)"))

    t_start = time()
    Nc = target_grid.Nc

    @info @sprintf("Process ERA5 N320 → CS day: date=%s, Nc=%d, Nz=%d, FT=%s",
                   string(date), Nc, Nz_int, string(FT))

    handles = open_era5_day(settings, date; next_day_handle = true)
    try
        # --- Allocate two ERA5 pipelines: current + next sliding-window. ---
        @info "  Allocating ERA5 N320 → C180 pipelines (×2, sliding window)..."
        cur_pipe = allocate_era5_n320_to_c180_pipeline(
            handles, target_grid; Nz = Nz_int,
            cache_dir = cache_dir,
            include_convection = include_convection)
        nxt_pipe = allocate_era5_n320_to_c180_pipeline(
            handles, target_grid; Nz = Nz_int,
            cache_dir = cache_dir,
            include_convection = include_convection)

        vc = cur_pipe.vc
        mesh = target_grid.mesh

        # --- CS-side workspaces (dry mass on target, face flux, balance). ---
        cs_cell_areas = mesh.cell_areas   # (Nc, Nc), shared across panels
        cur_m_dry      = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_delp_dry   = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_ps_dry     = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        cur_ps_dry_acc = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
        cur_am         = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz_int), 6)
        cur_bm         = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz_int), 6)
        cur_cm         = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int + 1), 6)
        cur_dp_panels  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_u_local    = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_v_local    = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        cur_dm_dry     = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_m_dry      = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_delp_dry   = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
        nxt_ps_dry     = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        nxt_ps_dry_acc = ntuple(_ -> zeros(Float64, Nc, Nc), 6)

        Δx = mesh.Δx
        Δy = mesh.Δy
        gravity = FT(GRAV)
        out_dt_factor = FT(dt_met_seconds / (2 * steps_per_met))

        # --- Open the streaming writer. ---
        nwindow = 24
        mkpath(dirname(out_path))
        tmp_path = out_path * ".tmp"
        isfile(tmp_path) && rm(tmp_path)
        @info @sprintf("  Output: %s (Nc=%d, Nz=%d, FT=%s)",
                       basename(out_path), Nc, Nz_int, string(FT))
        inner_writer = open_streaming_cs_transport_binary(
            tmp_path, Nc, 6, Nz_int, nwindow, vc;
            FT = FT,
            dt_met_seconds = Float64(dt_met_seconds),
            half_dt_seconds = Float64(dt_met_seconds) / 2,
            steps_per_window = steps_per_met,
            include_flux_delta = true,
            include_tm5conv = include_convection,
            mass_basis = mass_basis,
            panel_convention = _cs_panel_convention_tag(target_grid),
            cs_definition = _cs_definition_tag(target_grid),
            cs_coordinate_law = _cs_coordinate_law_tag(target_grid),
            cs_center_law = _cs_center_law_tag(target_grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(mesh)),
            extra_header = Dict{String, Any}(
                "preprocessor" => "process_era5_n320_to_cs_day",
                "source_type"  => "era5_n320_native_grib",
                "source_root"  => settings.root_dir,
                "target_type"  => "cubed_sphere",
                "regrid_method" => "conservative",
                "poisson_balanced" => true,
                "tm5_convection_source" => include_convection ?
                    "ec2tm_from_rates(udmf,ddmf,udrf,ddrf)" : "none",
            ))
        writer = CubedSphereBinaryWriter(inner_writer,
                                          mass_basis_from_symbol(mass_basis);
                                          Nc = Nc,
                                          npanel = 6,
                                          final_path = String(out_path))

        write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
        replay_tol = replay_tolerance(FT)

        # Helper: drive the pipeline + derive dry mass + rotate + reconstruct
        # fluxes for one window, writing into the provided panel buffers.
        function _process_window_to_cs!(win::Int,
                                          pipe::ERA5N320ToC180Pipeline,
                                          m_dry, delp_dry, ps_dry, ps_dry_acc,
                                          am, bm)
            hour = win - 1
            process_era5_n320_window!(pipe, handles, date, hour)

            # Dry mass on C180 from regridded PS + Q.
            derive_c180_dry_mass!(m_dry, delp_dry, ps_dry, ps_dry_acc,
                                   pipe.c180_fields.ps, pipe.c180_fields.qv,
                                   vc, cs_cell_areas)

            # DELP for face flux reconstruction is the MOIST pressure
            # thickness (matches what `reconstruct_cs_fluxes!` expects
            # alongside the moist PS used by the LL → CS path). The
            # dry-mass output above is the binary's `m` payload; the
            # face fluxes are reconstructed from MOIST PS and DELP so they
            # are bit-comparable with the GEOS-IT path.
            @inbounds for p in 1:6
                for k in 1:Nz_int
                    dA = Float64(vc.A[k + 1]) - Float64(vc.A[k])
                    dB = Float64(vc.B[k + 1]) - Float64(vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        cur_dp_panels[p][i, j, k] = FT(abs(dA + dB * Float64(pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end

            # Rotate geographic → panel-local. Output buffers are separate
            # from the pipeline's `c180_fields.u / .v` so the pipeline
            # output stays in the geographic frame for downstream diagnostics.
            rotate_winds_to_panel_local!(cur_u_local, cur_v_local,
                                          pipe.c180_fields.u,
                                          pipe.c180_fields.v,
                                          mesh, Nz_int)

            # Arakawa-C face flux reconstruction.
            reconstruct_cs_fluxes!(am, bm, cur_u_local, cur_v_local,
                                    cur_dp_panels, pipe.c180_fields.ps,
                                    vc.A, vc.B, Δx, Δy,
                                    gravity, out_dt_factor, Nc, Nz_int)
            return nothing
        end

        worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
        worst_replay_rel = 0.0; worst_replay_abs = 0.0; worst_replay_win = 0
        worst_positivity = init_cs_positivity_accumulator()
        apply_horizontal_balance = horizontal_poisson_balance_enabled()

        # --- Window 1. ---
        t0 = time()
        _process_window_to_cs!(1, cur_pipe,
                                cur_m_dry, cur_delp_dry, cur_ps_dry, cur_ps_dry_acc,
                                cur_am, cur_bm)
        @info @sprintf("    Window  1/%d: pipeline+regrid+rotate+flux %.2fs",
                       nwindow, time() - t0)

        # --- Windows 2..24: read next, balance current, diagnose cm, write. ---
        for win in 2:nwindow
            t0 = time()
            _process_window_to_cs!(win, nxt_pipe,
                                    nxt_m_dry, nxt_delp_dry, nxt_ps_dry, nxt_ps_dry_acc,
                                    cur_am, cur_bm)   # nxt_am/bm not needed — overwritten next round
            t_read = time() - t0

            # Restore the current window's am/bm (the _process call above
            # wrote into the cur_am/bm — but we want to balance the CURRENT
            # window using the NEXT window's mass target. Re-run the flux
            # reconstruction for the current window from its preserved
            # cur_pipe outputs. Cheap: panel rotation + face flux.
            # We could split _process_window_to_cs! into "pipeline" and
            # "rotate+flux" but the rotate+flux step is fast and keeping
            # the call site uniform avoids duplicate code paths.
            rotate_winds_to_panel_local!(cur_u_local, cur_v_local,
                                          cur_pipe.c180_fields.u,
                                          cur_pipe.c180_fields.v,
                                          mesh, Nz_int)
            @inbounds for p in 1:6
                for k in 1:Nz_int
                    dA = Float64(vc.A[k + 1]) - Float64(vc.A[k])
                    dB = Float64(vc.B[k + 1]) - Float64(vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        cur_dp_panels[p][i, j, k] = FT(abs(dA + dB * Float64(cur_pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end
            reconstruct_cs_fluxes!(cur_am, cur_bm, cur_u_local, cur_v_local,
                                    cur_dp_panels, cur_pipe.c180_fields.ps,
                                    vc.A, vc.B, Δx, Δy,
                                    gravity, out_dt_factor, Nc, Nz_int)

            t_bal = time()
            bal_diag = if apply_horizontal_balance
                balance_cs_global_mass_fluxes!(
                    cur_am, cur_bm, cur_m_dry, nxt_m_dry,
                    target_grid.face_table, target_grid.cell_degree, steps_per_met,
                    target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                    max_iter = 20000,
                    project_every = Int(cs_balance_project_every))
            else
                balance_cs_column_mass_fluxes!(
                    cur_am, cur_bm, cur_m_dry, nxt_m_dry,
                    target_grid.face_table, target_grid.cell_degree, steps_per_met,
                    target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                    max_iter = 20000,
                    project_every = Int(cs_balance_project_every))
            end
            t_bal = time() - t_bal

            worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
            worst_post = max(worst_post, bal_diag.max_post_residual)
            worst_iter = max(worst_iter, bal_diag.max_cg_iter)

            sync_all_cs_boundary_mirrors!(cur_am, cur_bm, mesh.connectivity, Nc, Nz_int)

            fill_cs_window_mass_tendency!(cur_dm_dry, cur_m_dry, nxt_m_dry, steps_per_met)
            for p in 1:6; fill!(cur_cm[p], zero(FT)); end
            diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cur_dm_dry, cur_m_dry, Nc, Nz_int)

            pos_diag = if write_replay_on
                contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                       nxt_m_dry,
                                                       steps_per_met, win - 1;
                                                       replay_tol = replay_tol,
                                                       positivity_cfl_limit = positivity_cfl_limit)
                if worst_replay_win == 0 || contract.replay.max_rel_err > worst_replay_rel
                    worst_replay_rel = contract.replay.max_rel_err
                    worst_replay_abs = contract.replay.max_abs_err
                    worst_replay_win = win - 1
                end
                contract.positivity
            else
                verify_substep_positivity_cs!(cur_m_dry, cur_am, cur_bm, cur_cm;
                                              cfl_limit = positivity_cfl_limit)
            end
            worst_positivity = update_cs_positivity_accumulator(worst_positivity, pos_diag, win - 1)

            _fill_cs_mass_delta_payload!(cur_dm_dry, cur_m_dry, nxt_m_dry)

            base_payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                            ps = cur_ps_dry, dm = cur_dm_dry)
            payload = include_convection ?
                merge(base_payload, (; tm5_fields = (
                    entu = cur_pipe.tm5_c180_fields.entu,
                    detu = cur_pipe.tm5_c180_fields.detu,
                    entd = cur_pipe.tm5_c180_fields.entd,
                    detd = cur_pipe.tm5_c180_fields.detd))) :
                base_payload
            write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(win - 1, payload))

            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                            win - 1, nwindow, t_bal, bal_diag.max_pre_residual,
                            bal_diag.max_post_residual, bal_diag.max_cg_iter,
                            win, t_read)

            # Swap current ↔ next. The pipelines themselves swap so the
            # GRIB read state stays paired with the dry mass we cached.
            cur_pipe, nxt_pipe = nxt_pipe, cur_pipe
            cur_m_dry, nxt_m_dry         = nxt_m_dry, cur_m_dry
            cur_delp_dry, nxt_delp_dry   = nxt_delp_dry, cur_delp_dry
            cur_ps_dry, nxt_ps_dry       = nxt_ps_dry, cur_ps_dry
            cur_ps_dry_acc, nxt_ps_dry_acc = nxt_ps_dry_acc, cur_ps_dry_acc
        end

        # --- Final window: next-day hour-0 endpoint look-ahead. ---
        # The closed contract for window h is m(h+1) - m(h); the final
        # window's m_next must be the next day's hour-0 endpoint, not a
        # zero-tendency copy of m_cur. A zero-tendency fallback forces
        # Poisson balance to absorb the actual ERA5 wind divergence into
        # `cm`, which the positivity gate then (correctly) rejects.
        # Reuse `nxt_pipe`'s window/regrid buffers — those are convection-
        # agnostic so we skip the heavier `process_era5_n320_window!` path.
        next_handles = _next_day_core_only_handle(handles)
        if next_handles !== nothing
            read_era5_n320_window_fields!(nxt_pipe.window_fields,
                                           nxt_pipe.spectral_ws,
                                           next_handles,
                                           handles.date + Day(1), 0)
            regrid_n320_to_c180!(nxt_pipe.c180_fields,
                                   nxt_pipe.window_fields,
                                   nxt_pipe.regrid_ws,
                                   target_grid)
            derive_c180_dry_mass!(nxt_m_dry, nxt_delp_dry, nxt_ps_dry, nxt_ps_dry_acc,
                                   nxt_pipe.c180_fields.ps, nxt_pipe.c180_fields.qv,
                                   vc, cs_cell_areas)
        else
            # HACK: zero-tendency fallback for the final day of the archive
            # (no next_core_path on disk). The positivity gate will likely
            # warn/fail in this case — that's correct: a zero-tendency
            # closure for the last hour is a known artifact, not a passing
            # binary. Run with [numerics].require_substep_positivity = false
            # only when you understand and accept this for boundary days.
            @warn "process_era5_n320_to_cs_day: archive-boundary fallback — " *
                  "no next-day core file for $(handles.date + Day(1)), using " *
                  "zero-tendency m_next. Final window's continuity will not close."
            for p in 1:6
                copyto!(nxt_m_dry[p], cur_m_dry[p])
            end
        end

        rotate_winds_to_panel_local!(cur_u_local, cur_v_local,
                                      cur_pipe.c180_fields.u,
                                      cur_pipe.c180_fields.v,
                                      mesh, Nz_int)
        @inbounds for p in 1:6
            for k in 1:Nz_int
                dA = Float64(vc.A[k + 1]) - Float64(vc.A[k])
                dB = Float64(vc.B[k + 1]) - Float64(vc.B[k])
                for j in 1:Nc, i in 1:Nc
                    cur_dp_panels[p][i, j, k] = FT(abs(dA + dB * Float64(cur_pipe.c180_fields.ps[p][i, j])))
                end
            end
        end
        reconstruct_cs_fluxes!(cur_am, cur_bm, cur_u_local, cur_v_local,
                                cur_dp_panels, cur_pipe.c180_fields.ps,
                                vc.A, vc.B, Δx, Δy,
                                gravity, out_dt_factor, Nc, Nz_int)

        bal_diag = if apply_horizontal_balance
            balance_cs_global_mass_fluxes!(
                cur_am, cur_bm, cur_m_dry, nxt_m_dry,
                target_grid.face_table, target_grid.cell_degree, steps_per_met,
                target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                max_iter = 20000,
                project_every = Int(cs_balance_project_every))
        else
            balance_cs_column_mass_fluxes!(
                cur_am, cur_bm, cur_m_dry, nxt_m_dry,
                target_grid.face_table, target_grid.cell_degree, steps_per_met,
                target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                max_iter = 20000,
                project_every = Int(cs_balance_project_every))
        end
        sync_all_cs_boundary_mirrors!(cur_am, cur_bm, mesh.connectivity, Nc, Nz_int)
        fill_cs_window_mass_tendency!(cur_dm_dry, cur_m_dry, nxt_m_dry, steps_per_met)
        for p in 1:6; fill!(cur_cm[p], zero(FT)); end
        diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cur_dm_dry, cur_m_dry, Nc, Nz_int)
        final_pos_diag = if write_replay_on
            contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                   nxt_m_dry,
                                                   steps_per_met, nwindow;
                                                   replay_tol = replay_tol,
                                                   positivity_cfl_limit = positivity_cfl_limit)
            if worst_replay_win == 0 || contract.replay.max_rel_err > worst_replay_rel
                worst_replay_rel = contract.replay.max_rel_err
                worst_replay_abs = contract.replay.max_abs_err
                worst_replay_win = nwindow
            end
            contract.positivity
        else
            verify_substep_positivity_cs!(cur_m_dry, cur_am, cur_bm, cur_cm;
                                          cfl_limit = positivity_cfl_limit)
        end
        worst_positivity = update_cs_positivity_accumulator(worst_positivity, final_pos_diag, nwindow)
        _fill_cs_mass_delta_payload!(cur_dm_dry, cur_m_dry, nxt_m_dry)

        final_base_payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                              ps = cur_ps_dry, dm = cur_dm_dry)
        final_payload = include_convection ?
            merge(final_base_payload, (; tm5_fields = (
                entu = cur_pipe.tm5_c180_fields.entu,
                detu = cur_pipe.tm5_c180_fields.detu,
                entd = cur_pipe.tm5_c180_fields.entd,
                detd = cur_pipe.tm5_c180_fields.detd))) :
            final_base_payload
        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(nwindow, final_payload))

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        # Summarize positivity BEFORE promoting the .tmp so a failed-gate day
        # quarantines the staged file (matches cubed_sphere_regrid.jl:610-617).
        summarize_cs_positivity_status(worst_positivity;
                                       cfl_limit = positivity_cfl_limit,
                                       steps_per_window = steps_per_met,
                                       quarantine_path = writer_staging_path(writer))
        promote_streaming_binary!(writer)

        elapsed = time() - t_start
        @info @sprintf("ERA5 N320 → C180 day complete: %.1fs (%.2fs/window). Worst bal pre=%.2e post=%.2e iter=%d.",
                        elapsed, elapsed / nwindow, worst_pre, worst_post, worst_iter)
        worst_replay_win > 0 &&
            @info @sprintf("  Worst replay: rel=%.2e abs=%.2e at win=%d",
                            worst_replay_rel, worst_replay_abs, worst_replay_win)
        return nothing
    finally
        close_era5_day!(handles)
    end
end

# ===========================================================================
# Unified-CLI dispatch — wires the per-day driver into
# `preprocess_transport_binary.jl` via the standard
# `process_day(date, grid, settings, vertical; ...)` extension point.
# ===========================================================================

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings::AbstractERA5GRIBSettings,
                vertical; out_path, FT, mass_basis, dt_met_seconds, positivity_cfl_limit,
                kwargs...)

Adapter that the unified preprocessor CLI calls into. Forwards to
[`process_era5_n320_to_cs_day`](@ref) with the kwargs the underlying
function actually accepts; the rest of the unified-CLI day-kwargs (e.g.
`chain_mass`, `adaptive_substeps`, `min_steps_per_window`, `seed_m`) are
absorbed by the trailing `kwargs...` and silently ignored — ERA5 N320 has
no day-to-day mass-chain state, and the writer currently uses a fixed
substep count rather than the adaptive policy.

Returns `(; final_m = nothing)` so the unified CLI's `seed_m = get(result,
:final_m, nothing)` chain remains a no-op.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::AbstractERA5GRIBSettings,
                     vertical;
                     out_path::AbstractString,
                     mass_basis::Symbol = :dry,
                     dt_met_seconds::Real = 3600.0,
                     positivity_cfl_limit::Real = 0.95,
                     min_steps_per_window::Union{Integer, Nothing} = nothing,
                     kwargs...)
    # Honor the substep policy's min_steps_per_window if the CLI passed one;
    # otherwise fall back to the established N320 default. Adaptive substep
    # scheduling isn't yet supported on this path.
    steps_per_window = min_steps_per_window === nothing ? 8 : Int(min_steps_per_window)
    process_era5_n320_to_cs_day(date, settings, grid;
        out_path                  = out_path,
        Nz                        = vertical.Nz,
        mass_basis                = mass_basis,
        dt_met_seconds            = dt_met_seconds,
        steps_per_window          = steps_per_window,
        positivity_cfl_limit      = positivity_cfl_limit,
        cache_dir                 = grid.cache_dir,
        include_convection        = settings.include_convection)
    return (; final_m = nothing)
end

# Output filename matches the standalone CLI script's naming convention so
# downstream tools that already grep for `era5_n320_to_cNNN_transport_…`
# pick up the unified-CLI output without changes.
function _native_output_filename(::AbstractERA5GRIBSettings, date::Date, FT::Type)
    return "era5_n320_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"
end
