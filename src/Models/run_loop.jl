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

    # === Set active panel map for multi-GPU dispatch ===
    pm = get_panel_map(grid)
    set_panel_map!(pm)

    # === Allocation (panel_map from grid — all allocators read it) ===
    _use_gchp_sched = _needs_gchp(model.advection_scheme)
    sched     = build_io_scheduler(grid, arch, buffering; use_gchp=_use_gchp_sched,
                                    panel_map=pm)
    phys      = allocate_physics_buffers(grid, arch, model; panel_map=pm)
    tracers   = allocate_tracers(model, grid)
    air       = allocate_air_mass(grid, arch)
    _use_lr   = _needs_linrood(model.advection_scheme)
    _use_vr   = _needs_vertical_remap(model.advection_scheme)
    _use_gchp = _needs_gchp(model.advection_scheme)
    gc_ws     = allocate_geometry_and_workspace(grid, arch; use_linrood=_use_lr,
                                                 use_vertical_remap=_use_vr,
                                                 use_gchp=_use_gchp,
                                                 panel_map=pm)
    gc        = gc_ws.gc
    ws        = gc_ws.ws
    ws_lr     = gc_ws.ws_lr
    ws_vr     = gc_ws.ws_vr
    geom_gchp = gc_ws.geom_gchp
    ws_gchp   = gc_ws.ws_gchp
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
    # Fine-grained per-phase timing (accumulated over all windows)
    t_phases = Dict{String,Float64}(
        "io_load"      => 0.0,  # met data loading + upload
        "io_upload_phys" => 0.0,  # physics field upload (CMFMC, QV, PBL)
        "io_wait"      => 0.0,  # wait_load! blocking time
        "io_delp_fetch"=> 0.0,  # deferred DELP fetch
        "gpu_met_proc" => 0.0,  # process_met_after_upload! (flux scaling)
        "gpu_airmass"  => 0.0,  # compute_air_mass_phase!
        "gpu_emissions"=> 0.0,  # apply_emissions_phase!
        "gpu_cm"       => 0.0,  # compute_cm_phase!
        "gpu_advection"=> 0.0,  # advection_phase! (incl. convection inside)
        "gpu_diffusion"=> 0.0,  # post_advection_physics! (diffusion + chemistry)
        "gpu_massfixer" => 0.0, # apply_mass_correction!
        "gpu_diag"     => 0.0,  # mass diagnostics
        "output"       => 0.0,  # write_output!
    )

    log_simulation_start(model, grid, buffering, n_win, n_sub, dw)
    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)

    for w in 1:n_win
        has_next = w < n_win
        t0 = time()

        # ── IO Phase ─────────────────────────────────────────────────
        _t = time(); upload_met!(sched);                              t_phases["io_load"] += time() - _t
        _t = time(); load_and_upload_physics!(phys, sched, driver, grid, w; arch); t_phases["io_upload_phys"] += time() - _t

        # Start async load of next window (SB: sync; DB: Threads.@spawn)
        if has_next
            begin_load!(sched, driver, grid, w + 1; kw...)
        end

        # Write staging progress file (for NVMe staging daemon, if configured)
        _write_staging_progress(model, w)

        # CS: compute PS from DELP (CPU, reads curr_cpu — safe after spawn)
        compute_ps_phase!(phys, sched, grid)

        t_io += time() - t0
        t0 = time()

        # ── GPU Compute Phase ────────────────────────────────────────
        _t = time(); process_met_after_upload!(sched, phys, grid, driver, half_dt;
            use_gchp=_needs_gchp(model.advection_scheme)); t_phases["gpu_met_proc"] += time() - _t
        _t = time()
        compute_air_mass_phase!(sched, air, phys, grid, gc;
                                dry_correction=get(model.metadata, "dry_correction", true))
        t_phases["gpu_airmass"] += time() - _t

        # Compute dry mass for LL rm↔c_dry conversions (m_dry = m_ref × (1 - QV))
        compute_ll_dry_mass!(phys, sched, grid)

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

        # Pre-advection mass snapshot (target for global mass fixer)
        snapshot_pre_advection!(diag, tracers, grid)

        # Deferred fetch: wait for next-window DELP (CS DoubleBuffer only)
        t_f0 = time()
        wait_and_upload_next_delp!(sched, grid)
        t_phases["io_delp_fetch"] += time() - t_f0
        t_io += time() - t_f0

        # Skip cm computation for vertical remap path (no Z-advection needed)
        _t = time()
        if ws_vr === nothing
            compute_cm_phase!(sched, air, phys, grid, gc;
                               has_next, n_sub)
        end
        t_phases["gpu_cm"] += time() - _t

        # Advection + convection sub-stepping
        _t = time()
        advection_phase!(tracers, sched, air, phys, model,
                          grid, ws, n_sub, dt_sub, step;
                          ws_lr, ws_vr, gc, geom_gchp, ws_gchp, has_next)
        t_phases["gpu_advection"] += time() - _t

        # Track transport mass change
        _mass_after_adv = compute_mass_totals(tracers, grid)
        record_mass_change!(diag.cumulative_transport, diag.pre_adv_mass, _mass_after_adv)

        # Emissions (after advection: transport → emissions → mixing → chemistry)
        _t = time()
        sim_hours = Float64((w - 1) * dt_window / 3600)
        apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                grid, dt_window; sim_hours, arch)
        t_phases["gpu_emissions"] += time() - _t

        # Track emission mass change
        _mass_after_emi = compute_mass_totals(tracers, grid)
        record_mass_change!(diag.cumulative_emissions, _mass_after_adv, _mass_after_emi)

        # Post-advection physics: BLD + PBL + chemistry
        _t = time()
        post_advection_physics!(tracers, sched, air, phys, model,
                                 grid, dt_window, dw)
        t_phases["gpu_diffusion"] += time() - _t

        # Track physics mass change (diffusion + chemistry)
        _mass_after_phys = compute_mass_totals(tracers, grid)
        record_mass_change!(diag.cumulative_physics, _mass_after_emi, _mass_after_phys)

        # Global mass correction (CS only)
        _t = time()
        apply_mass_correction!(tracers, grid, diag;
                                mass_fixer=get(model.metadata, "mass_fixer", true),
                                mass_fixer_tracers=String.(get(model.metadata, "mass_fixer_tracers", String[])))
        t_phases["gpu_massfixer"] += time() - _t

        # Mass diagnostics
        _t = time()
        update_mass_diagnostics!(diag, tracers, grid)
        t_phases["gpu_diag"] += time() - _t

        t_gpu += time() - t0
        t0 = time()

        # ── Output Phase ─────────────────────────────────────────────
        sim_time = Float64(step[] * dt_sub)
        out_mass = compute_output_mass(sched, air, phys, grid)
        met = build_met_fields(sched, phys, grid, half_dt, dt_window)
        # Use m_ref for output until the rm/m_dev contract is fully resolved.
        # Transport with evolving m_dev is correct (verified by MCP tests), but
        # emissions/diffusion still use m_ref for rm↔c conversion, creating
        # inconsistency. Using m_ref for output matches the post-physics state.
        compute_ll_dry_mass!(phys, sched, grid)
        # Convert rm → dry VMR for output (LL only; CS is identity)
        c_tracers = rm_to_vmr(tracers, sched, phys, grid)
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=out_mass, tracers=c_tracers, met_fields=met,
                          rm_tracers=tracers)
        end
        snapshot_massfixer_interval!(diag)
        t_phases["output"] += time() - t0
        t_out += time() - t0

        # ── Buffer management ────────────────────────────────────────
        _t = time()
        wait_load!(sched)
        t_phases["io_wait"] += time() - _t
        swap!(sched)

        # Verbose: per-window phase timing (visible with --verbose / -v)
        if is_verbose()
            @info @sprintf("  [v] win %d/%d: adv=%.3f io_phys=%.3f out=%.3f delp=%.3f diff=%.3f em=%.3f diag=%.3f",
                w, n_win, t_phases["gpu_advection"]/w, t_phases["io_upload_phys"]/w,
                t_phases["output"]/w, t_phases["io_delp_fetch"]/w,
                t_phases["gpu_diffusion"]/w, t_phases["gpu_emissions"]/w, t_phases["gpu_diag"]/w)
        end

        update_progress!(prog, diag, grid, w, step[], dt_sub,
                          wall_start, t_io, t_gpu, t_out)
    end

    finish!(prog)
    finalize_simulation!(writers, diag, tracers, grid,
                          wall_start, n_win, step[], t_io, t_gpu, t_out;
                          t_phases)
    return model
end
