# ---------------------------------------------------------------------------
# Physics phase functions — grid-dispatched wrappers for the unified run loop
#
# Each phase dispatches on grid type (LatitudeLongitudeGrid vs CubedSphereGrid).
# The unified run loop calls these in sequence without any if/else on grid type.
# ---------------------------------------------------------------------------

# =====================================================================
# Setup helpers
# =====================================================================

"""Build diffusion workspace, dispatched on grid type for correct template array."""
function setup_diffusion_phase(model, grid::LatitudeLongitudeGrid, dt_window, sched)
    _setup_bld_workspace(model.diffusion, grid, dt_window, current_gpu(sched).m_ref)
end

function setup_diffusion_phase(model, grid::CubedSphereGrid{FT}, dt_window, sched) where FT
    AT = array_type(model.architecture)
    Hp = grid.Hp
    ref_panel = AT(zeros(FT, grid.Nc + 2Hp, grid.Nc + 2Hp, grid.Nz))
    _setup_bld_workspace(model.diffusion, grid, dt_window, ref_panel)
end

"""Return kwargs NamedTuple for IOScheduler begin_load! (physics buffers for CS)."""
function physics_load_kwargs(phys, grid::LatitudeLongitudeGrid)
    return (;)  # LL loads physics separately, not via IOScheduler
end

function physics_load_kwargs(phys, grid::CubedSphereGrid)
    # Surface field buffers: use PBL buffers if available, else dummy for tropopause
    sfc = if phys.has_pbl
        phys.pbl_sfc_cpu
    else
        (pblh=phys.troph_cpu, ustar=phys.troph_cpu,
         hflux=phys.troph_cpu, t2m=phys.troph_cpu)
    end
    return (;
        cmfmc_cpu=phys.cmfmc_cpu, dtrain_cpu=phys.dtrain_cpu,
        sfc_cpu=sfc, troph_cpu=phys.troph_cpu,
        needs_cmfmc=phys.has_conv, needs_dtrain=phys.has_dtrain,
        needs_sfc=true, needs_qv=true,
        qv_cpu=phys.qv_cpu, ps_panels=phys.ps_cpu)
end

"""Log simulation start message (unified format for all grid/buffering combos)."""
function log_simulation_start(model, grid, buffering, n_win, n_sub, dw)
    grid_str = if hasproperty(grid, :Nc)
        "C$(grid.Nc)"
    else
        "$(grid.Nx)×$(grid.Ny)×$(grid.Nz)"
    end
    buf_str = nameof(typeof(buffering))

    parts = ["Starting simulation: $n_win windows × $n_sub sub-steps ($buf_str, $grid_str)"]
    if dw !== nothing && hasproperty(model.diffusion, :Kz_max)
        push!(parts, " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]")
    end
    if _needs_pbl(model.diffusion)
        push!(parts, " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]")
    end
    if _needs_convection(model.convection)
        push!(parts, " [convection: $(nameof(typeof(model.convection)))]")
    end
    @info join(parts)
end

# =====================================================================
# Phase 1: Load + upload physics fields (CMFMC, DTRAIN, QV, surface)
# =====================================================================

"""
    load_and_upload_physics!(phys, sched, driver, grid, w; arch)

Load physics fields and upload to GPU. For LL, loads each field from driver
(IOScheduler only loads met data). For CS, physics was already loaded by
IOScheduler's `begin_load!` — just upload to GPU based on `sched.io_result`.
"""
function load_and_upload_physics!(phys, sched, driver,
                                   grid::LatitudeLongitudeGrid, w; arch=nothing)
    # CMFMC
    if phys.has_conv
        status = load_cmfmc_window!(phys.cmfmc_cpu, driver, grid, w)
        phys.cmfmc_loaded[] = status !== false
        if phys.cmfmc_loaded[] && status !== :cached
            copyto!(phys.cmfmc_gpu, phys.cmfmc_cpu)
        end
    end

    # DTRAIN
    if phys.has_dtrain && phys.cmfmc_loaded[]
        status = load_dtrain_window!(phys.dtrain_cpu, driver, grid, w)
        phys.dtrain_loaded[] = status !== false
        if phys.dtrain_loaded[] && status !== :cached
            copyto!(phys.dtrain_gpu, phys.dtrain_cpu)
        end
    end

    # Invalidate RAS CFL cache on fresh data
    if phys.cmfmc_loaded[] || phys.dtrain_loaded[]
        invalidate_ras_cfl_cache!()
    end

    # Surface fields
    if phys.has_pbl
        phys.sfc_loaded[] = load_surface_fields_window!(
            phys.pbl_sfc_cpu, driver, grid, w)
        if phys.sfc_loaded[]
            copyto!(phys.pbl_sfc_gpu.pblh,  phys.pbl_sfc_cpu.pblh)
            copyto!(phys.pbl_sfc_gpu.ustar, phys.pbl_sfc_cpu.ustar)
            copyto!(phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_cpu.hflux)
            copyto!(phys.pbl_sfc_gpu.t2m,   phys.pbl_sfc_cpu.t2m)
        end
    end

    # QV (specific humidity for dry-air output)
    qv_status = load_qv_window!(phys.qv_cpu, driver, grid, w)
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        copyto!(phys.qv_gpu, phys.qv_cpu)
    end
end

function load_and_upload_physics!(phys, sched, driver,
                                   grid::CubedSphereGrid, w; arch=nothing)
    io = sched.io_result
    io === nothing && return

    # CMFMC
    if phys.has_conv
        cmfmc_status = io.cmfmc
        phys.cmfmc_loaded[] = cmfmc_status !== false
        if phys.cmfmc_loaded[] && cmfmc_status !== :cached
            for p in 1:6
                copyto!(phys.cmfmc_gpu[p], phys.cmfmc_cpu[p])
            end
        end
    end

    # DTRAIN
    if phys.has_dtrain
        dtrain_status = io.dtrain
        phys.dtrain_loaded[] = dtrain_status !== false
        if phys.dtrain_loaded[] && dtrain_status !== :cached
            for p in 1:6
                copyto!(phys.dtrain_gpu[p], phys.dtrain_cpu[p])
            end
        end
    end

    # Invalidate RAS CFL cache on fresh data
    if (phys.cmfmc_loaded[] && io.cmfmc !== :cached) ||
       (phys.dtrain_loaded[] && io.dtrain !== :cached)
        invalidate_ras_cfl_cache!()
    end

    # QV
    qv_status = io.qv
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        for p in 1:6
            copyto!(phys.qv_gpu[p], phys.qv_cpu[p])
        end
        fill_panel_halos!(phys.qv_gpu, grid)
    end

    # Surface fields
    sfc_status = io.sfc
    phys.sfc_loaded[] = sfc_status !== false
    phys.troph_loaded[] = phys.sfc_loaded[]
    if phys.sfc_loaded[] && phys.has_pbl
        for p in 1:6
            copyto!(phys.pbl_sfc_gpu.pblh[p],  phys.pbl_sfc_cpu.pblh[p])
            copyto!(phys.pbl_sfc_gpu.ustar[p], phys.pbl_sfc_cpu.ustar[p])
            copyto!(phys.pbl_sfc_gpu.hflux[p], phys.pbl_sfc_cpu.hflux[p])
            copyto!(phys.pbl_sfc_gpu.t2m[p],   phys.pbl_sfc_cpu.t2m[p])
        end
    end

end

"""Compute PS from DELP (CS only). Called after begin_load! — reads curr_cpu (safe)."""
compute_ps_phase!(phys, sched, grid::LatitudeLongitudeGrid) = nothing
compute_ps_phase!(phys, sched, grid::CubedSphereGrid) = _compute_ps_from_delp!(phys, sched, grid)

"""Compute surface pressure from DELP for CS grids (no-op if PS loaded from binary)."""
function _compute_ps_from_delp!(phys, sched, grid::CubedSphereGrid{FT}) where FT
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    ps_cpu = phys.ps_cpu
    _ps_from_bin = phys.sfc_loaded[] && !iszero(ps_cpu[1][Hp + 1, Hp + 1])
    _ps_from_bin && return

    cpu = current_cpu(sched)
    for p in 1:6
        fill!(ps_cpu[p], zero(FT))
        delp_p = cpu.delp[p]
        @inbounds for k in 1:Nz
            for jj in 1:Nc, ii in 1:Nc
                ps_cpu[p][Hp + ii, Hp + jj] += delp_p[Hp + ii, Hp + jj, k]
            end
        end
    end
end

# =====================================================================
# Phase 2: Process met after upload (scale fluxes, compute DELP for LL)
# =====================================================================

"""Process met data after GPU upload. LL: handle raw met + compute DELP. CS: scale fluxes."""
function process_met_after_upload!(sched, phys, grid::LatitudeLongitudeGrid{FT},
                                    driver, half_dt) where FT
    gpu = current_gpu(sched)
    cpu = current_cpu(sched)

    if driver isa AbstractRawMetDriver
        copyto!(gpu.Δp, gpu.m_ref)
        compute_air_mass!(gpu.m_ref, gpu.Δp, grid)
        copyto!(gpu.m_dev, gpu.m_ref)
        copyto!(gpu.u, gpu.am)
        copyto!(gpu.v, gpu.bm)
        compute_mass_fluxes!(gpu.am, gpu.bm, gpu.cm,
                              gpu.u, gpu.v, grid, gpu.Δp, half_dt)
    elseif phys.needs_delp
        _compute_delp_ll!(gpu, cpu, phys, grid)
    end
end

function process_met_after_upload!(sched, phys, grid::CubedSphereGrid{FT},
                                    driver, half_dt) where FT
    gpu = current_gpu(sched)
    for p in 1:6
        gpu.am[p] .*= half_dt
        gpu.bm[p] .*= half_dt
    end
end

"""Compute DELP for LL from air mass: Δp = m × g / area."""
function _compute_delp_ll!(gpu, cpu, phys, grid::LatitudeLongitudeGrid{FT}) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    delp_cpu = phys.delp_cpu
    area_j = phys.area_j
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        delp_cpu[i, j, k] = cpu.m[i, j, k] * g / area_j[j]
    end
    copyto!(gpu.Δp, delp_cpu)
end

# =====================================================================
# Phase 3: Compute air mass from DELP
# =====================================================================

"""Compute air mass. LL: already in m_ref after upload. CS: from DELP per panel (dry when QV available and dry_correction enabled)."""
compute_air_mass_phase!(sched, air, phys, grid::LatitudeLongitudeGrid, gc; dry_correction::Bool=true) = nothing

function compute_air_mass_phase!(sched, air, phys, grid::CubedSphereGrid, gc; dry_correction::Bool=true)
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    if dry_correction && phys.qv_loaded[]
        for p in 1:6
            compute_air_mass_panel!(air.m[p], gpu.delp[p], phys.qv_gpu[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    else
        for p in 1:6
            compute_air_mass_panel!(air.m[p], gpu.delp[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    end
    # Always compute MOIST air mass for convection (m_wet = delp × area / g).
    # Convective mass fluxes (CMFMC, DTRAIN) are on MOIST basis, so rm↔q
    # conversion must use moist air mass for consistency (GCHP convention).
    for p in 1:6
        compute_air_mass_panel!(air.m_wet[p], gpu.delp[p],
                                gc.area[p], gc.gravity, Nc, Nz, Hp)
    end
end

# =====================================================================
# Phase 4: IC finalization + reference mass save
# =====================================================================

"""Finalize deferred initial conditions (first window only)."""
function finalize_ic_phase!(tracers, sched, air, phys, grid::LatitudeLongitudeGrid)
    has_deferred_ic_vinterp() || return
    finalize_ic_vertical_interp!(tracers, current_gpu(sched).m_ref, grid)
end

function finalize_ic_phase!(tracers, sched, air, phys, grid::CubedSphereGrid)
    has_deferred_ic_vinterp() || return
    gpu = current_gpu(sched)
    # IC values are dry VMR. air.m is already dry when QV available, moist
    # otherwise. Either way: rm = q × m gives correct tracer mass.
    finalize_ic_vertical_interp!(tracers, air.m, gpu.delp, grid; qv_panels=nothing)
end

"""Save reference air mass. LL: no-op (m_ref is the reference). CS: copy m → m_ref."""
save_reference_mass!(sched, ::Nothing, ::LatitudeLongitudeGrid) = nothing

function save_reference_mass!(sched, air, grid::CubedSphereGrid)
    for p in 1:6
        copyto!(air.m_ref[p], air.m[p])
    end
end

# =====================================================================
# Phase 5: Compute vertical mass flux (cm)
# =====================================================================

"""Compute cm. LL: already in met buffer. CS: from continuity of am/bm."""
compute_cm_phase!(sched, air, grid::LatitudeLongitudeGrid, gc) = nothing

function compute_cm_phase!(sched, air, grid::CubedSphereGrid, gc)
    gpu = current_gpu(sched)
    Nc, Nz = grid.Nc, grid.Nz

    # Pure mass-flux closure: cm from continuity of am/bm only.
    # No pressure tendency — air mass evolves freely from the fluxes.
    for p in 1:6
        compute_cm_panel!(gpu.cm[p], gpu.am[p], gpu.bm[p], gc.bt, Nc, Nz)
    end
end

"""Wait for next-window DELP and upload (CS DoubleBuffer pressure fixer)."""
wait_and_upload_next_delp!(::IOScheduler{SingleBuffer}, ::Any) = nothing
wait_and_upload_next_delp!(::IOScheduler, ::LatitudeLongitudeGrid) = nothing

function wait_and_upload_next_delp!(sched::IOScheduler{DoubleBuffer},
                                     grid::CubedSphereGrid)
    if sched.load_task !== nothing
        result = fetch(sched.load_task)
        sched.io_result = result isa NamedTuple ? result : nothing
        sched.load_task = nothing
    end
    nc = next_cpu(sched)
    ng = next_gpu(sched)
    for p in 1:6
        copyto!(ng.delp[p], nc.delp[p])
    end
end

# =====================================================================
# Phase 6: CFL diagnostic
# =====================================================================

"""Update CFL diagnostic string (CS only, periodic)."""
update_cfl_diagnostic!(diag, sched, air, grid::LatitudeLongitudeGrid, ws, w) = nothing

function update_cfl_diagnostic!(diag, sched, air, grid::CubedSphereGrid, ws, w)
    (w == 1 || w % 24 == 0) || return
    gpu = current_gpu(sched)
    Hp = grid.Hp
    cfl_x = maximum(max_cfl_x_cs(gpu.am[p], air.m_ref[p], ws.cfl_x, Hp) for p in 1:6)
    cfl_y = maximum(max_cfl_y_cs(gpu.bm[p], air.m_ref[p], ws.cfl_y, Hp) for p in 1:6)
    diag.cfl_value = @sprintf("x=%.3f y=%.3f", cfl_x, cfl_y)
end

# =====================================================================
# Phase 7: Apply emissions
# =====================================================================

"""Apply emissions, dispatched on grid type."""
function apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                 grid::LatitudeLongitudeGrid, dt_window;
                                 sim_hours::Float64=0.0, arch=nothing)
    emi_data, area_j_dev, A_coeff, B_coeff = emi_state
    gpu = current_gpu(sched)
    _apply_emissions_latlon!(tracers, emi_data, area_j_dev,
                              gpu.ps, A_coeff, B_coeff, grid, dt_window;
                              sim_hours, arch,
                              delp=(phys.sfc_loaded[] ? gpu.Δp : nothing),
                              pblh=(phys.sfc_loaded[] && phys.pbl_sfc_gpu !== nothing ?
                                    phys.pbl_sfc_gpu.pblh : nothing))
end

function apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                 grid::CubedSphereGrid, dt_window;
                                 sim_hours::Float64=0.0, arch=nothing)
    emi_data = emi_state[1]
    gpu = current_gpu(sched)
    Nc, Hp = grid.Nc, grid.Hp
    _apply_emissions_cs!(tracers, emi_data, gc.area, dt_window, Nc, Hp;
                          sim_hours, arch,
                          delp=gpu.delp,
                          pblh=(phys.pbl_sfc_gpu !== nothing ?
                                phys.pbl_sfc_gpu.pblh : nothing))
end

# =====================================================================
# Phase 8: Advection sub-stepping (+ convection per sub-step)
# =====================================================================

"""
    advection_phase!(tracers, sched, air, phys, model, grid, ws, n_sub, dt_sub, step)

Advection + convection sub-stepping. LL: NamedTuple dispatch (all tracers together).
CS: per-tracer advection with mass fixer + m_ref advance along pressure trajectory.
"""
function advection_phase!(tracers, sched, air, phys, model,
                           grid::LatitudeLongitudeGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           has_next::Bool=false) where FT
    gpu = current_gpu(sched)
    for _ in 1:n_sub
        step[] += 1
        copyto!(gpu.m_dev, gpu.m_ref)
        _apply_advection_latlon!(tracers, gpu.m_dev,
                                  gpu.am, gpu.bm, gpu.cm,
                                  grid, model.advection_scheme, gpu.ws;
                                  cfl_limit=FT(0.95))
    end
    # Convective transport ONCE per window (after all substeps).
    if phys.cmfmc_loaded[]
        dt_conv = FT(n_sub) * dt_sub
        convect!(tracers, phys.cmfmc_gpu, gpu.Δp,
                  model.convection, grid, dt_conv, phys.planet;
                  dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                  workspace=phys.ras_workspace)
    end
end

function advection_phase!(tracers, sched, air, phys, model,
                           grid::CubedSphereGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           has_next::Bool=false) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp

    if ws_vr !== nothing
        # ═══════════════════════════════════════════════════════════════════
        # REMAP PATH: Horizontal-only Lin-Rood + vertical remap
        #
        # fv_tp_2d_cs! evolves BOTH rm and m via _linrood_update_kernel!.
        # m must evolve naturally through both calls per substep so that
        # q = rm/m stays consistent. After both calls, we RESCALE rm to
        # match prescribed m: rm_new = (rm/m_evolved) × m_prescribed.
        # This preserves the true VMR while keeping rm on the prescribed-m
        # basis for the next substep. Without rescaling, restoring m alone
        # creates q errors (up to 1500 ppm at TOA in point-source tests).
        #
        # Two calls per substep (matching strang_split_linrood_ppm!):
        # total = 2 × n_sub × half_dt = dt_window.
        # ═══════════════════════════════════════════════════════════════════

        # Save prescribed m (from compute_air_mass_phase!)
        for p in 1:6; copyto!(ws_vr.m_save[p], air.m[p]); end

        for _ in 1:n_sub
            step[] += 1

            for (_, rm_t) in pairs(tracers)
                # Restore prescribed m before each tracer's transport
                for p in 1:6; copyto!(air.m[p], ws_vr.m_save[p]); end

                # Two half-substep Lin-Rood calls = one full substep.
                # m evolves naturally through both calls (no restore between).
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, Val(_ppm_order(model.advection_scheme)), ws, ws_lr;
                              damp_coeff=model.advection_scheme.damp_coeff)
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, Val(_ppm_order(model.advection_scheme)), ws, ws_lr;
                              damp_coeff=FT(0))

                # Rescale rm to prescribed-m basis while preserving VMR:
                # rm_new = (rm / m_evolved) × m_prescribed = q_true × m_prescribed
                for p in 1:6
                    rm_t[p] .*= ws_vr.m_save[p] ./ air.m[p]
                end
            end
        end

        # ── Convective transport ONCE per window (after all substeps) ─────
        # Both TM5 and GCHP apply convection as a separate operator outside
        # the advection substep loop. Interleaving convection with substeps
        # causes 12× over-mixing because RAS recomputes q_cloud from the
        # already-mixed environment at each call (nonlinear feedback).
        # RAS internal subcycling handles CFL stability for the larger dt.
        for p in 1:6; copyto!(air.m[p], ws_vr.m_save[p]); end
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end

        # GCHP-style PE computation (fv_tracer2d.F90:988-1035):
        # Source PE: direct cumsum from actual air mass (m_save = DELP×(1-QV)×area/g).
        #   Derived inside remap kernel from m_src (hybrid_pe=false).
        #   GCHP uses cumsum of post-horizontal dpA — our m_save is equivalent
        #   (prescribed dry air mass on current window's pressure structure).
        # Target PE: direct cumsum of next-window dry DELP (no hybrid reconstruction).
        #   The hybrid formula (PE=ak+bk×PS) deviates from actual met DELP by
        #   0.1-1% per level (up to 250 Pa), causing systematic vertical pumping.
        #   GCHP compensates with calcScalingFactor; direct cumsum avoids the issue.
        if has_next
            ng = next_gpu(sched)
            if phys.qv_loaded[]
                compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                    phys.qv_gpu, gc, grid)
            else
                compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
            end
        else
            compute_target_pressure_from_mass_direct!(ws_vr, air.m, gc, grid)
        end

        # Vertical remap: source PE from m_src inside kernel (hybrid_pe=false)
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid;
                               hybrid_pe=false)
        end

        # Update air.m to target state (m = dp_tgt * area / g)
        update_air_mass_from_target!(air.m, ws_vr, gc, grid)
        # Copy to m_ref for output/diagnostics
        for p in 1:6; copyto!(air.m_ref[p], air.m[p]); end

        # Recompute m_wet from updated air.m for post-advection physics.
        # After remap, air.m is on target pressure basis; m_wet must match
        # so that diffusion's q = rm/m_wet is consistent with remapped rm.
        # m_wet = m_dry / (1-qv). Uses current-window QV (<0.1% approx).
        if phys.qv_loaded[]
            for p in 1:6
                air.m_wet[p] .= air.m[p] ./ max.(1 .- phys.qv_gpu[p], eps(FT))
            end
        else
            for p in 1:6; copyto!(air.m_wet[p], air.m[p]); end
        end
    else
        # ═══════════════════════════════════════════════════════════════════
        # STRANG PATH: Existing cm-based Z-advection
        # ═══════════════════════════════════════════════════════════════════
        for _ in 1:n_sub
            step[] += 1

            # Advect each tracer independently (m reset per tracer)
            for (_, rm_t) in pairs(tracers)
                for p in 1:6
                    copyto!(air.m[p], air.m_ref[p])
                end
                _apply_advection_cs!(rm_t, air.m, gpu.am, gpu.bm, gpu.cm,
                                      grid, model.advection_scheme, ws; ws_lr)
            end
            # air.m now holds m_evolved (same for all tracers)

            # Per-cell mass fixer: rm = (rm / m_evolved) × m_ref
            if get(model.metadata, "mass_fixer", true)
                for (_, rm_t) in pairs(tracers)
                    for p in 1:6
                        apply_mass_fixer!(rm_t[p], air.m_ref[p], air.m[p], Nc, Nz, Hp)
                    end
                end
            end
        end

        # ── Convective transport ONCE per window (after all substeps) ─────
        # See remap path comment: TM5/GCHP apply convection outside substeps.
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end
    end
end

# =====================================================================
# Phase 9: Post-advection physics (BLD + PBL diffusion + chemistry)
# =====================================================================

"""Post-advection physics: BLD diffusion, PBL diffusion, chemistry."""
function post_advection_physics!(tracers, sched, air, phys, model,
                                  grid::LatitudeLongitudeGrid, dt_window, dw)
    gpu = current_gpu(sched)

    _apply_bld!(tracers, dw)

    if phys.sfc_loaded[]
        diffuse_pbl!(tracers, gpu.Δp,
                      phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                      phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                      phys.w_scratch,
                      model.diffusion, grid, dt_window, phys.planet)
    end

    apply_chemistry!(tracers, grid, model.chemistry, dt_window)
end

function post_advection_physics!(tracers, sched, air, phys, model,
                                  grid::CubedSphereGrid, dt_window, dw)
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp

    # Use DRY air mass for rm↔q conversion in diffusion. GeosChem's
    # pbl_mix_mod.F90 uses AD = State_Met%AD (dry air mass) for PBL mixing.
    # Using moist mass creates QV-dependent VMR artifacts: ~0.8 ppm
    # tropical-polar bias in dry VMR output from q_wet/(1-QV) conversion.
    # TODO: verify convection should also switch to dry (GCHP convection_mod
    # uses BMASS = DELP_DRY for RAS, but CMFMC flux is moist — needs care).
    for (_, rm_t) in pairs(tracers)
        _apply_bld_cs!(rm_t, air.m, dw, Nc, Nz, Hp)
    end

    if phys.sfc_loaded[] && phys.has_pbl
        for (_, rm_t) in pairs(tracers)
            diffuse_pbl!(rm_t, air.m, gpu.delp,
                          phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                          phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                          phys.w_scratch,
                          model.diffusion, grid, dt_window, phys.planet)
        end
    end

    apply_chemistry!(tracers, grid, model.chemistry, dt_window)
end

# =====================================================================
# Phase 10: Global mass correction (CS only)
# =====================================================================

"""Apply global mass fixer. No-op for LL; scales tracers for CS."""
apply_mass_correction!(tracers, grid::LatitudeLongitudeGrid, diag; mass_fixer::Bool=true) = nothing

function apply_mass_correction!(tracers, grid::CubedSphereGrid, diag; mass_fixer::Bool=true)
    mass_fixer || return
    isempty(diag.pre_adv_mass) && return
    diag.fix_value = apply_global_mass_fixer!(tracers, grid, diag.pre_adv_mass)
end

# =====================================================================
# Phase 11: Compute output air mass (dry correction)
# =====================================================================

"""Compute air mass for output. If QV loaded, returns dry mass."""
function compute_output_mass(sched, air, phys, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if phys.qv_loaded[]
        phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
        return phys.m_dry
    end
    return gpu.m_ref
end

function compute_output_mass(sched, air, phys, grid::CubedSphereGrid)
    # Use current air.m — after vertical remap, air.m is the target mass
    # and rm has been remapped to match. VMR = rm / air.m is correct.
    # (air.m_ref is the START-of-window mass, inconsistent with post-remap rm.)
    return air.m
end

# =====================================================================
# Phase 12: Build met_fields for output writers
# =====================================================================

"""Build met_fields NamedTuple for output writers."""
function build_met_fields(sched, phys, grid::LatitudeLongitudeGrid, half_dt, dt_window)
    return (;)  # LL output writers don't need met_fields
end

function build_met_fields(sched, phys, grid::CubedSphereGrid, half_dt, dt_window)
    gpu = current_gpu(sched)
    base = (; ps=phys.ps_cpu,
              mass_flux_x=gpu.am, mass_flux_y=gpu.bm,
              mf_scale=half_dt, dt_window=dt_window)

    if phys.sfc_loaded[] && phys.has_pbl && phys.troph_loaded[]
        base = merge(base, (; pblh=phys.pbl_sfc_cpu.pblh, troph=phys.troph_cpu))
    elseif phys.sfc_loaded[] && phys.has_pbl
        base = merge(base, (; pblh=phys.pbl_sfc_cpu.pblh))
    elseif phys.troph_loaded[]
        base = merge(base, (; troph=phys.troph_cpu))
    end

    if phys.qv_loaded[]
        base = merge(base, (; qv=phys.qv_cpu))
    end
    return base
end

# =====================================================================
# IC output (first window only)
# =====================================================================

"""Write IC output snapshot (no-op for LL, writes t=0 output for CS)."""
write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                  grid::LatitudeLongitudeGrid, half_dt, dt_window) = nothing

function write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                            grid::CubedSphereGrid, half_dt, dt_window)
    gpu = current_gpu(sched)

    # Build IC met_fields
    met_ic = (; ps=phys.ps_cpu,
                mass_flux_x=gpu.am, mass_flux_y=gpu.bm,
                mf_scale=half_dt, dt_window=dt_window)
    if phys.sfc_loaded[] && phys.has_pbl
        met_ic = merge(met_ic, (; pblh=phys.pbl_sfc_cpu.pblh))
    end
    if phys.troph_loaded[]
        met_ic = merge(met_ic, (; troph=phys.troph_cpu))
    end

    # IC mass (dry if QV available)
    ic_mass = compute_output_mass(sched, air, phys, grid)

    for writer in writers
        write_output!(writer, model, 0.0;
                      air_mass=ic_mass, tracers=tracers, met_fields=met_ic)
    end
end

# =====================================================================
# Progress bar update (unified format)
# =====================================================================

"""Update progress bar with timing + diagnostics."""
function update_progress!(prog, diag, grid::LatitudeLongitudeGrid,
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out)
    sim_time = Float64(step * dt_sub)
    sv = Pair{Symbol,Any}[
        :day  => @sprintf("%.1f", sim_time / 86400),
        :rate => @sprintf("%.2f s/win", w > 1 ? (time() - wall_start) / w : 0.0)]
    isempty(diag.showvalue) || push!(sv, :mass => diag.showvalue)
    next!(prog; showvalues=sv)
end

function update_progress!(prog, diag, grid::CubedSphereGrid,
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out)
    sv = Pair{Symbol,Any}[
        :day  => div(w - 1, 24) + 1,
        :IO   => @sprintf("%.2f s/win", t_io / w),
        :GPU  => @sprintf("%.2f s/win", t_gpu / w),
        :Out  => @sprintf("%.2f s/win", t_out / w)]
    isempty(diag.cfl_value)  || push!(sv, :CFL  => diag.cfl_value)
    isempty(diag.fix_value)  || push!(sv, :fix  => diag.fix_value)
    isempty(diag.showvalue)  || push!(sv, :mass => diag.showvalue)
    next!(prog; showvalues=sv)
end

# =====================================================================
# Finalize simulation (summary logging + output finalization)
# =====================================================================

"""Finalize simulation: log mass conservation summary + close output writers."""
function finalize_simulation!(writers, diag, tracers, grid,
                                wall_start, n_win, step, t_io, t_gpu, t_out)
    wall_total = time() - wall_start

    # Final mass summary (Δ = mass closure bias vs expected = initial + emissions)
    if !isempty(diag.expected_mass)
        final_mass = compute_mass_totals(tracers, grid)
        for tname in sort(collect(keys(final_mass)))
            total = final_mass[tname]
            if haskey(diag.expected_mass, tname) && diag.expected_mass[tname] != 0.0
                rel = (total - diag.expected_mass[tname]) /
                      abs(diag.expected_mass[tname]) * 100
                @info @sprintf("  Final mass %s: %.6e kg (Δ=%.4e%%)", tname, total, rel)
            else
                @info @sprintf("  Final mass %s: %.6e kg", tname, total)
            end
        end
    end

    # Timing summary
    if t_io > 0 || t_gpu > 0
        @info @sprintf(
            "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
            step, wall_total, t_io / n_win, t_gpu / n_win, t_out / n_win)
    else
        dt_per_win = n_win > 0 ? wall_total / n_win : 0.0
        @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                       step, wall_total, dt_per_win)
    end

    for writer in writers
        finalize_output!(writer)
    end
end
