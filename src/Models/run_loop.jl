# ---------------------------------------------------------------------------
# Unified run loop — single _run_loop! via multiple dispatch
#
# Replaces 4 separate methods (LL×SB, LL×DB, CS×SB, CS×DB) with one.
# Grid differences → phase functions in physics_phases.jl
# Buffering differences → IOScheduler in io_scheduler.jl
# Allocation differences → factories in simulation_state.jl
# ---------------------------------------------------------------------------

"""
    run!(model::TransportModel)

Run the forward model using the met driver, sources, and buffering strategy
stored in the model. Dispatches on `(model.grid, model.buffering)`.
"""
function run!(model::TransportModel)
    return _run_loop!(model, model.grid, model.buffering)
end

"""
    _run_loop!(model, grid, buffering)

Unified run loop for all (grid type, buffering strategy) combinations.
Uses multiple dispatch on `grid` for physics phases, and `IOScheduler{B}`
for buffering. No if/else on types in the main loop body.
"""
function _run_loop!(model, grid::AbstractGrid{FT},
                    buffering::AbstractBufferingStrategy) where FT
    driver  = model.met_data
    arch    = model.architecture
    sources = model.sources
    writers = model.output_writers

    n_win    = total_windows(driver)
    n_sub    = steps_per_window(driver)
    dt_sub   = FT(driver.dt)
    dt_window = FT(dt_sub * n_sub)
    half_dt  = FT(dt_sub / 2)

    # === Allocation ===
    sched     = build_io_scheduler(grid, arch, buffering)
    phys      = allocate_physics_buffers(grid, arch, model)
    tracers   = allocate_tracers(model, grid)
    air       = allocate_air_mass(grid, arch)
    _use_lr   = _needs_linrood(model.advection_scheme)
    _use_vr   = _needs_vertical_remap(model.advection_scheme)
    gc_ws     = allocate_geometry_and_workspace(grid, arch; use_linrood=_use_lr,
                                                 use_vertical_remap=_use_vr)
    gc        = gc_ws.gc
    ws        = gc_ws.ws
    ws_lr     = gc_ws.ws_lr
    ws_vr     = gc_ws.ws_vr
    emi_state = prepare_emissions(sources, grid, driver, arch)
    dw        = setup_diffusion_phase(model, grid, dt_window, sched)
    diag      = MassDiagnostics()

    # === Initial load ===
    kw = physics_load_kwargs(phys, grid)
    initial_load!(sched, driver, grid, 1; kw...)

    # === Timers + progress ===
    step = Ref(0)
    wall_start = time()
    t_io  = 0.0
    t_gpu = 0.0
    t_out = 0.0

    log_simulation_start(model, grid, buffering, n_win, n_sub, dw)
    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)

    for w in 1:n_win
        has_next = w < n_win
        t0 = time()

        # ── IO Phase ─────────────────────────────────────────────────
        upload_met!(sched)
        load_and_upload_physics!(phys, sched, driver, grid, w; arch)

        # Start async load of next window (SB: sync; DB: Threads.@spawn)
        if has_next
            begin_load!(sched, driver, grid, w + 1; kw...)
        end

        # CS: compute PS from DELP (CPU, reads curr_cpu — safe after spawn)
        compute_ps_phase!(phys, sched, grid)

        t_io += time() - t0
        t0 = time()

        # ── GPU Compute Phase ────────────────────────────────────────
        process_met_after_upload!(sched, phys, grid, driver, half_dt)
        # Always compute air mass from current DELP to stay consistent with
        # met mass fluxes (am/bm). For the remap path, the previous window's
        # update_air_mass_from_target set m from scaled dp_tgt — this drifts
        # from the actual DELP over time due to fix_target_bottom_pe! scaling.
        # The QV mismatch between remap target and new DELP is < 0.1% per window
        # and much smaller than the PE fixer drift.
        compute_air_mass_phase!(sched, air, phys, grid, gc;
                                dry_correction=get(model.metadata, "dry_correction", true))

        # First window: IC finalization + initial output
        if w == 1
            finalize_ic_phase!(tracers, sched, air, phys, grid)
            save_reference_mass!(sched, air, grid)
            record_initial_mass!(diag, tracers, grid)
            for tname in sort(collect(keys(diag.initial_mass)))
                @info @sprintf("  IC mass %s: %.6e kg", tname, diag.initial_mass[tname])
            end
            write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                              grid, half_dt, dt_window)
        else
            save_reference_mass!(sched, air, grid)
        end

        update_cfl_diagnostic!(diag, sched, air, grid, ws, w)

        # Emissions
        sim_hours = Float64((w - 1) * dt_window / 3600)
        apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                grid, dt_window; sim_hours, arch)

        # Pre-advection mass snapshot (target for global mass fixer)
        snapshot_pre_advection!(diag, tracers, grid)

        # Deferred fetch: wait for next-window DELP (CS DoubleBuffer only)
        t_f0 = time()
        wait_and_upload_next_delp!(sched, grid)
        t_io += time() - t_f0

        # Skip cm computation for vertical remap path (no Z-advection needed)
        if ws_vr === nothing
            compute_cm_phase!(sched, air, grid, gc)
        end

        # Advection + convection sub-stepping
        advection_phase!(tracers, sched, air, phys, model,
                          grid, ws, n_sub, dt_sub, step;
                          ws_lr, ws_vr, gc, has_next)

        # Post-advection physics: BLD + PBL + chemistry
        post_advection_physics!(tracers, sched, air, phys, model,
                                 grid, dt_window, dw)

        # Global mass correction (CS only)
        apply_mass_correction!(tracers, grid, diag;
                                mass_fixer=get(model.metadata, "mass_fixer", true))

        # Mass diagnostics
        update_mass_diagnostics!(diag, tracers, grid)

        t_gpu += time() - t0
        t0 = time()

        # ── Output Phase ─────────────────────────────────────────────
        sim_time = Float64(step[] * dt_sub)
        out_mass = compute_output_mass(sched, air, phys, grid)
        met = build_met_fields(sched, phys, grid, half_dt, dt_window)
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=out_mass, tracers=tracers, met_fields=met)
        end

        t_out += time() - t0

        # ── Buffer management ────────────────────────────────────────
        wait_load!(sched)
        swap!(sched)

        update_progress!(prog, diag, grid, w, step[], dt_sub,
                          wall_start, t_io, t_gpu, t_out)
    end

    finish!(prog)
    finalize_simulation!(writers, diag, tracers, grid,
                          wall_start, n_win, step[], t_io, t_gpu, t_out)
    return model
end
