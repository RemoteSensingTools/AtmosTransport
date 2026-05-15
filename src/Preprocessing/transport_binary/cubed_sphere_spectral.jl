# Spectral ERA5 to cubed-sphere transport-binary preprocessing path.

mutable struct CubedSphereSpectralWindowWorkspace{FT, SG, R, TW, MW, QW, CSW, DX, DY} <:
               AbstractWindowWorkspace{CubedSphereTargetGeometry, FT}
    staging_grid :: SG
    regridder    :: R
    transform    :: TW
    merged       :: MW
    qv           :: QW
    ps_offsets   :: Vector{Float64}
    cs_ws        :: CSW
    A_ifc        :: Vector{Float64}
    B_ifc        :: Vector{Float64}
    gravity      :: FT
    dt_factor    :: FT
    Δx           :: DX
    Δy           :: DY
    cur_m        :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cur_ps       :: NTuple{CS_PANEL_COUNT, Array{FT, 2}}
    cur_am       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cur_bm       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cur_cm       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    regridder_time :: Float64
    qv_time        :: Float64
end

function _cs_regridder_cache_key(grid::CubedSphereTargetGeometry)
    return Symbol("cs_ll_to_cs_regridder_C", grid.Nc, "_",
                  grid.staging_nlon, "x", grid.staging_nlat)
end

function _get_or_build_cs_regridder!(cache,
                                     grid::CubedSphereTargetGeometry,
                                     staging_grid::LatLonTargetGeometry)
    cache_key = _cs_regridder_cache_key(grid)
    cached = cache === nothing ? nothing : get(cache, cache_key, nothing)
    t_reg = time()
    if cached !== nothing
        regridder = cached
        n_src = length(regridder.src_areas)
        n_dst = length(regridder.dst_areas)
        @info @sprintf("  Regridder: reused %d×%d  nnz=%d (%.1fs)",
                       n_src, n_dst, length(regridder.intersections.nzval),
                       time() - t_reg)
        return regridder
    end

    regridder = build_regridder(staging_grid.mesh, grid.mesh;
                                normalize = false,
                                cache_dir = grid.cache_dir)
    cache === nothing || (cache[cache_key] = regridder)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    @info @sprintf("  Regridder: %d×%d  nnz=%d (%.1fs)",
                   n_src, n_dst, length(regridder.intersections.nzval),
                   time() - t_reg)
    return regridder
end

function allocate_window_workspace(grid::CubedSphereTargetGeometry,
                                   settings,
                                   vertical,
                                   spec,
                                   date::Date,
                                   ::Type{FT};
                                   cache = nothing) where FT
    Nc = grid.Nc
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz

    mesh_FT = eltype(grid.mesh)
    staging_grid = build_target_geometry(Val(:latlon),
        Dict{String, Any}("type" => "latlon",
                          "nlon" => grid.staging_nlon,
                          "nlat" => grid.staging_nlat), mesh_FT)
    Nx_stg = nlon(staging_grid)
    Ny_stg = nlat(staging_grid)
    @info @sprintf("  Staging grid: %d×%d LL → C%d CS (%d panels)",
                   Nx_stg, Ny_stg, Nc, CS_PANEL_COUNT)

    t_regridder = time()
    regridder = _get_or_build_cs_regridder!(cache, grid, staging_grid)
    t_regridder = time() - t_regridder
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)

    transform = allocate_transform_workspace(staging_grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(staging_grid, Nz_native, Nz, FT)
    t_qv = time()
    qv = allocate_qv_workspace(staging_grid, settings, date, Nz_native, Nz, FT)
    @debug @sprintf("  QV workspace allocated in %.2fs", time() - t_qv)
    ps_offsets = zeros(Float64, spec.n_times + 1)
    cs_ws = allocate_cs_preprocess_workspace(Nc, Nx_stg, Ny_stg, Nz,
                                             n_src, n_dst, FT)

    vc_merged = vertical.merged_vc
    A_ifc = Float64.(vc_merged.A)
    B_ifc = Float64.(vc_merged.B)
    gravity = FT(GRAV)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    dt_factor = FT(settings.met_interval / (2 * steps_per_met))
    Δx = grid.mesh.Δx
    Δy = grid.mesh.Δy

    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)

    return CubedSphereSpectralWindowWorkspace{
        FT, typeof(staging_grid), typeof(regridder), typeof(transform),
        typeof(merged), typeof(qv), typeof(cs_ws), typeof(Δx), typeof(Δy)}(
            staging_grid, regridder, transform, merged, qv, ps_offsets,
            cs_ws, A_ifc, B_ifc, gravity, dt_factor, Δx, Δy,
            cur_m, cur_ps, cur_am, cur_bm, cur_cm,
            t_regridder, t_qv)
end

function ingest_window!(workspace::CubedSphereSpectralWindowWorkspace{FT},
                        win_idx::Int,
                        hour::Int,
                        spec,
                        grid::CubedSphereTargetGeometry,
                        vertical,
                        settings,
                        m_out,
                        ps_out,
                        am_out,
                        bm_out) where FT
    t0 = time()
    transform = workspace.transform
    merged = workspace.merged
    staging_grid = workspace.staging_grid
    cs_ws = workspace.cs_ws
    Nx_stg = nlon(staging_grid)
    Ny_stg = nlat(staging_grid)
    Nz_native = vertical.Nz_native
    Nc = grid.Nc
    Nz = vertical.Nz

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, staging_grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_window_qv!(workspace.qv, win_idx, Nx_stg, Ny_stg, Nz_native)
    apply_mass_fix_if_needed!(workspace.qv, transform, staging_grid, vertical,
                              settings, workspace.ps_offsets, win_idx)
    apply_dry_basis_if_needed!(settings.mass_basis, transform, workspace.qv)
    merge_native_window!(merged, transform, workspace.qv, vertical, settings)

    regrid_3d_to_cs_panels!(m_out, workspace.regridder, merged.m_merged,
                            cs_ws, Nc, ExtensiveCellField())
    regrid_2d_to_cs_panels!(ps_out, workspace.regridder, transform.sp,
                            cs_ws, Nc, IntensiveCellField())

    stg_lats = staging_grid.lats
    Δy_ll = FT(staging_grid.mesh.radius * deg2rad(staging_grid.mesh.Δφ))
    Δlon_ll = FT(deg2rad(staging_grid.mesh.Δλ))

    recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
        merged.am_merged, merged.bm_merged, transform.sp,
        workspace.A_ifc, workspace.B_ifc, stg_lats, Δy_ll, Δlon_ll,
        FT(staging_grid.mesh.radius), workspace.gravity, workspace.dt_factor)

    regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, workspace.regridder,
                            cs_ws.u_cc, cs_ws, Nc)
    regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, workspace.regridder,
                            cs_ws.v_cc, cs_ws, Nc)
    rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                  cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                  grid.tangent_basis, Nc, Nz)

    reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                           cs_ws.dp_panels, ps_out,
                           workspace.A_ifc, workspace.B_ifc,
                           workspace.Δx, workspace.Δy,
                           workspace.gravity, workspace.dt_factor, Nc, Nz)
    return time() - t0
end

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings, vertical; ...)

Spectral→CS transport binary: spectral synthesis to an internal LL staging grid,
conservative regridding to CS panels, endpoint continuity closure, and streaming
binary write. No on-disk LL intermediate.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings,
                     vertical;
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     next_day_hour0=nothing,
                     run_cache = nothing)
    FT = settings.output_float_type
    Nc = grid.Nc
    Nz = vertical.Nz
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    cs_balance_tol = Float64(get(settings, :cs_balance_tol, 1e-14))
    cs_balance_project_every = Int(get(settings, :cs_balance_project_every, 50))
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral(vo_d_path, lnsp_path;
                             T_target=settings.T_target,
                             cache_dir=settings.spectral_cache_dir)
    t_spectral_read = time() - t_day
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, t_spectral_read)
    Nt = spec.n_times

    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    # Stage to `.tmp` so any mid-loop exception (replay failure, IO error,
    # quarantined positivity violation) never leaves a partial binary at
    # `bin_path`. Promote `tmp_path -> bin_path` only after every contract
    # gate passes (or after a warning-only positivity summary returns).
    tmp_path = bin_path * ".tmp"
    isfile(tmp_path) && rm(tmp_path; force = true)

    workspace = allocate_window_workspace(grid, settings, vertical, spec, date, FT;
                                          cache = run_cache)
    staging_grid = workspace.staging_grid
    Nx_stg = nlon(staging_grid)
    Ny_stg = nlat(staging_grid)
    ps_offsets = workspace.ps_offsets
    cs_ws = workspace.cs_ws
    vc_merged = vertical.merged_vc

    log_mass_fix_configuration(settings)
    @info "  Streaming: spectral → LL staging → CS regrid → balance → write..."
    write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
    write_replay_on || @info "  Write-time CS replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    replay_tol = replay_tolerance(FT)

    cur_m = workspace.cur_m
    cur_ps = workspace.cur_ps
    cur_am = workspace.cur_am
    cur_bm = workspace.cur_bm
    cur_cm = workspace.cur_cm

    @inline function _copy_panels!(dst, src)
        for p in 1:CS_PANEL_COUNT
            copyto!(dst[p], src[p])
        end
    end

    # Wrap the writer open, work loop, contract summary, file-size check
    # and `tmp -> bin` promotion in try/finally. The two inner function
    # definitions above are intentionally kept outside the try block —
    # closure-in-try parsing is brittle in Julia and offers no benefit
    # here (those defs cannot leave a `.tmp` behind on their own). The
    # writer open is moved INSIDE the try so a header-write failure (or
    # any other exception before the loop) still hits the cleanup path.
    writer        = nothing
    writer_closed = false
    mv_done       = false
    try
    # --- Open streaming CS binary writer ---
    writer = open_streaming_cs_transport_binary(
        tmp_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=settings.met_interval,
        half_dt_seconds=settings.half_dt,
        steps_per_window=steps_per_met,
        include_flux_delta=true,
        mass_basis=Symbol(settings.mass_basis),
        panel_convention=_cs_panel_convention_tag(grid),
        cs_definition=_cs_definition_tag(grid),
        cs_coordinate_law=_cs_coordinate_law_tag(grid),
        cs_center_law=_cs_center_law_tag(grid),
        longitude_offset_deg=longitude_offset_deg(cs_definition(grid.mesh)),
        extra_header=Dict{String, Any}(
            "preprocessor"     => "preprocess_transport_binary.jl",
            "source_type"      => "era5_spectral",
            "target_type"      => "cubed_sphere",
            "staging_nlon"     => Nx_stg,
            "staging_nlat"     => Ny_stg,
            "regrid_method"    => "conservative",
            "vertical_mapping_method" => String(vertical_mapping_method(vertical)),
            "target_vertical_name" => hasproperty(vertical, :target_vertical_name) ?
                vertical.target_vertical_name : "",
            "target_coefficients" => hasproperty(vertical, :target_coefficients) ?
                vertical.target_coefficients : "",
            "merge_map" => vertical.merge_map,
            "poisson_balanced" => true,
            "mass_fix_enabled" => settings.mass_fix_enable,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(FT)
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path),
                   expected_total / 1e9, Nt)

    worst_pre = 0.0
    worst_post = 0.0
    worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)
    window_contract = CubedSphereContract{FT}(
        replay_tol = replay_tol,
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        steps_per_window = steps_per_met,
    )
    total_synth_regrid = 0.0
    total_balance = 0.0
    total_replay = 0.0
    total_write = 0.0
    total_last_endpoint = 0.0
    apply_horizontal_balance = horizontal_poisson_balance_enabled()
    if apply_horizontal_balance
        @info "  Applying per-layer CS Poisson mass-flux balance (legacy opt-in)..."
    else
        @info "  Applying column CS mass-balance correction; diagnosing cm from endpoint mass tendency..."
    end

    # --- Process first window ---
    t_first_synth = ingest_window!(workspace, 1, spec.hours[1], spec,
                                   grid, vertical, settings,
                                   cs_ws.m_panels, cs_ws.ps_panels,
                                   cs_ws.am_panels, cs_ws.bm_panels)
    total_synth_regrid += t_first_synth
    @info @sprintf("    Window  1/%d (hour %02d): synth+regrid %.2fs  offset=%+.3f Pa",
                   Nt, spec.hours[1], t_first_synth, ps_offsets[1])

    _copy_panels!(cur_m,  cs_ws.m_panels)
    _copy_panels!(cur_ps, cs_ws.ps_panels)
    _copy_panels!(cur_am, cs_ws.am_panels)
    _copy_panels!(cur_bm, cs_ws.bm_panels)

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t_synth = ingest_window!(workspace, win, spec.hours[win], spec,
                                 grid, vertical, settings,
                                 cs_ws.m_panels, cs_ws.ps_panels,
                                 cs_ws.am_panels, cs_ws.bm_panels)
        total_synth_regrid += t_synth

        # Copy m_next for balance (unmodified — the Poisson CG internally
        # projects the RHS to mean-zero, handling the topological constraint
        # Σ div_h = 0 on a closed sphere without modifying the stored m)
        _copy_panels!(cs_ws.m_next_panels, cs_ws.m_panels)

        # Close the PREVIOUS window using (m_cur, m_next)
        t_bal = time()
        bal_diag = if apply_horizontal_balance
            balance_cs_global_mass_fluxes!(
                cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
                grid.face_table, grid.cell_degree, steps_per_met,
                grid.poisson_scratch; tol=cs_balance_tol, max_iter=20000,
                project_every=cs_balance_project_every)
        else
            balance_cs_column_mass_fluxes!(
                cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
                grid.face_table, grid.cell_degree, steps_per_met,
                grid.poisson_scratch; tol=cs_balance_tol, max_iter=20000,
                project_every=cs_balance_project_every)
        end
        t_bal = time() - t_bal
        total_balance += t_bal

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        # Sync ALL boundary mirrors (including de-duplicated faces) so that
        # per-panel flux divergence and the advection kernel telescope correctly.
        sync_all_cs_boundary_mirrors!(cur_am, cur_bm, grid.mesh.connectivity, Nc, Nz)

        # Diagnose cm from balanced am/bm and raw mass tendency.
        fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
        for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
        diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
        pos_diag = if write_replay_on
            t_replay = time()
            contract_diag = verify_window!((m_cur = cur_m,
                                             am = cur_am,
                                             bm = cur_bm,
                                             cm = cur_cm,
                                             m_next = cs_ws.m_next_panels),
                                            window_contract, win - 1)
            total_replay += time() - t_replay
            if worst_replay_win == 0 || contract_diag.replay.max_rel_err > worst_replay_rel
                worst_replay_rel = contract_diag.replay.max_rel_err
                worst_replay_abs = contract_diag.replay.max_abs_err
                worst_replay_win = win - 1
                worst_replay_idx = contract_diag.replay.worst_idx
            end
            contract_diag.positivity
        else
            verify_substep_positivity_cs!(cur_m, cur_am, cur_bm, cur_cm;
                                         cfl_limit = positivity_cfl_limit)
        end
        update_accumulator!(window_contract, pos_diag, win - 1)
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        # Write balanced previous window
        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        t_write = time()
        write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)
        total_write += time() - t_write

        if should_log_window(win - 1, Nt)
            if apply_horizontal_balance
                @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | synth %2d (%.2fs)",
                               win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                               bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_synth)
            else
                @info @sprintf("    Window %2d/%d: wrote (colbal %.2fs pre=%.2e post=%.2e iter=%d) | synth %2d (%.2fs)",
                               win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                               bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_synth)
            end
        end

        # Swap: copy just-synthesized panels into cur (no allocation)
        _copy_panels!(cur_m,  cs_ws.m_panels)
        _copy_panels!(cur_ps, cs_ws.ps_panels)
        _copy_panels!(cur_am, cs_ws.am_panels)
        _copy_panels!(cur_bm, cs_ws.bm_panels)
    end

    # --- Balance & write LAST window (next-day closure when available) ---
    t_last_endpoint = time()
    last_hour_next = next_day_merged_fields(next_day_hour0, date, staging_grid, vertical,
                                            settings, workspace.transform,
                                            workspace.merged, workspace.qv,
                                            ps_offsets)
    total_last_endpoint = time() - t_last_endpoint
    if last_hour_next !== nothing
        # `m` is extensive (kg/cell): density-convert via the dispatcher.
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, workspace.regridder,
                                last_hour_next.m, cs_ws, Nc,
                                ExtensiveCellField())
    else
        _copy_panels!(cs_ws.m_next_panels, cur_m)
    end
    t_bal = time()
    bal_diag = if apply_horizontal_balance
        balance_cs_global_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            grid.face_table, grid.cell_degree, steps_per_met,
            grid.poisson_scratch; tol=cs_balance_tol, max_iter=5000,
            project_every=cs_balance_project_every)
    else
        balance_cs_column_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            grid.face_table, grid.cell_degree, steps_per_met,
            grid.poisson_scratch; tol=cs_balance_tol, max_iter=5000,
            project_every=cs_balance_project_every)
    end
    t_bal = time() - t_bal
    total_balance += t_bal

    worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
    worst_post = max(worst_post, bal_diag.max_post_residual)
    worst_iter = max(worst_iter, bal_diag.max_cg_iter)

    # Sync ALL boundary mirrors (including de-duplicated faces) — same as main loop.
    sync_all_cs_boundary_mirrors!(cur_am, cur_bm, grid.mesh.connectivity, Nc, Nz)

    fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
    for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
    diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
    pos_diag = if write_replay_on
        t_replay = time()
        contract_diag = verify_window!((m_cur = cur_m,
                                         am = cur_am,
                                         bm = cur_bm,
                                         cm = cur_cm,
                                         m_next = cs_ws.m_next_panels),
                                        window_contract, Nt)
        total_replay += time() - t_replay
        if worst_replay_win == 0 || contract_diag.replay.max_rel_err > worst_replay_rel
            worst_replay_rel = contract_diag.replay.max_rel_err
            worst_replay_abs = contract_diag.replay.max_abs_err
            worst_replay_win = Nt
            worst_replay_idx = contract_diag.replay.worst_idx
        end
        contract_diag.positivity
    else
        verify_substep_positivity_cs!(cur_m, cur_am, cur_bm, cur_cm;
                                     cfl_limit = positivity_cfl_limit)
    end
    update_accumulator!(window_contract, pos_diag, Nt)
    convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

    window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                 dm=cs_ws.m_next_panels)
    t_write = time()
    write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)
    total_write += time() - t_write

    if apply_horizontal_balance
        @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre=%.2e post=%.2e iter=%d",
                       Nt, Nt, t_bal, bal_diag.max_pre_residual,
                       bal_diag.max_post_residual, bal_diag.max_cg_iter)
    else
        @info @sprintf("    Window %2d/%d (last): colbal %.2fs  pre=%.2e post=%.2e iter=%d",
                       Nt, Nt, t_bal, bal_diag.max_pre_residual,
                       bal_diag.max_post_residual, bal_diag.max_cg_iter)
    end

    # --- Finalize ---
    close_streaming_transport_binary!(writer)
    writer_closed = true

    # Per-substep positivity gate. Close the writer first so a quarantine
    # delete (require_substep_positivity=true + violation) can remove the
    # binary cleanly without an open file handle on it.
    summarize_status!(window_contract; quarantine_path = tmp_path)

    # Reached only when positivity passed, or when require_substep_positivity=false
    # turned a violation into a warning. Promote the staged file either way.
    mv(tmp_path, bin_path; force = true)
    mv_done = true

    if settings.mass_fix_enable
        ps_offsets_day = @view ps_offsets[1:Nt]
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(ps_offsets_day), maximum(ps_offsets_day), sum(ps_offsets_day) / Nt)
    end

    if apply_horizontal_balance
        @info @sprintf("  Poisson balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                       worst_pre, worst_post, worst_iter)
    else
        @info @sprintf("  Column balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                       worst_pre, worst_post, worst_iter)
    end
    if write_replay_on
        replay_msg = worst_replay_win > 0 ?
            @sprintf("max rel=%.3e abs=%.3e kg win=%d cell=%s",
                     worst_replay_rel, worst_replay_abs, worst_replay_win, worst_replay_idx) :
            "no windows checked"
        @info "  Write-time replay gate: $replay_msg"
    end

    total_timed = t_spectral_read + workspace.regridder_time + workspace.qv_time +
                  total_synth_regrid + total_balance + total_replay +
                  total_write + total_last_endpoint
    @info @sprintf("  Timing summary (s): spectral_read=%.1f  regridder=%.1f  qv=%.1f  synth+regrid=%.1f  balance=%.1f  replay=%.1f  write=%.1f  last_endpoint=%.1f",
                   t_spectral_read, workspace.regridder_time,
                   workspace.qv_time, total_synth_regrid,
                   total_balance, total_replay, total_write, total_last_endpoint)
    @info @sprintf("  Timing fractions: balance=%.1f%%  synth+regrid=%.1f%%  spectral_read=%.1f%%",
                   100 * total_balance / max(total_timed, eps()),
                   100 * total_synth_regrid / max(total_timed, eps()),
                   100 * t_spectral_read / max(total_timed, eps()))

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path),
                   actual / 1e9, time() - t_day)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return bin_path
    finally
        # Guard on `writer !== nothing` so a failure in `open_…` itself
        # (before assignment) does not try to close a `nothing`.
        if writer !== nothing && !writer_closed
            close_streaming_transport_binary!(writer)
        end
        # On any non-clean exit (loop exception, replay error, quarantined
        # positivity violation), remove the staged file so it cannot be
        # mistaken for a finished binary on retry. `summarize_…` already
        # deleted it in the explicit quarantine case; this guard catches the
        # other failure modes.
        if !mv_done && isfile(tmp_path)
            rm(tmp_path; force = true)
        end
    end
end
