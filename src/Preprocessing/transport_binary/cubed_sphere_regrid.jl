# Lat-lon transport-binary to cubed-sphere transport-binary regrid path.

"""
    regrid_ll_binary_to_cs(ll_binary_path, cs_grid, out_path; FT=Float64, mass_basis=nothing)

Regrid an existing LL transport binary to a cubed-sphere binary.

Reads each window from the LL binary, recovers cell-center winds from am/bm,
conservatively regrids m/ps/winds to CS panels, rotates winds to panel-local
coordinates, reconstructs CS face fluxes, applies global Poisson balance,
diagnoses cm from continuity, and stream-writes the CS binary.

This reuses the entire CS regrid/balance/write infrastructure from the
spectral→CS path — the only difference is the data source (binary reader
instead of spectral synthesis).

Timestep metadata (`dt_met_seconds`, `steps_per_window`) is read directly
from the source header. The output CS binary carries the same values, so
the per-substep flux semantics survive the regrid without rescaling.

## Keyword arguments
- `FT::Type = Float64` — on-disk float type for the output CS binary.
- `mass_basis::Union{Nothing, Symbol} = nothing` — output mass-basis label.
  `nothing` (default) = match the source. Setting this to a value that
  differs from the source's `mass_basis` currently errors: actual
  dry↔moist conversion requires loading the source's `qv` and applying
  `apply_dry_basis_native!`, which this function does not do. Invariant
  14 mandates `:dry` end-to-end; use a dry source.
- `allow_terminal_zero_tendency::Bool = false` — diagnostic-only escape hatch
  for legacy LL sources that do not carry `dm`. Production-safe regrids should
  leave this at `false` so the final CS window is closed against an explicit
  endpoint target instead of an inferred zero-tendency fallback.
"""
function regrid_ll_binary_to_cs(ll_binary_path::String,
                                cs_grid::CubedSphereTargetGeometry,
                                out_path::String;
                                FT::Type{<:AbstractFloat} = Float64,
                                mass_basis::Union{Nothing, Symbol} = nothing,
                                allow_terminal_zero_tendency::Bool = false)
    t_start = time()
    Nc = cs_grid.Nc

    # --- Open LL binary reader ---
    reader = TransportBinaryReader(ll_binary_path; FT=FT)
    h = reader.header
    Nx_ll = h.Nx
    Ny_ll = h.Ny
    Nz = h.nlevel
    Nt = h.nwindow
    A_ifc = Float64.(h.A_ifc)
    B_ifc = Float64.(h.B_ifc)

    # Refuse silent basis relabeling. The function reads raw `m/am/bm/ps`
    # from the source and never touches `qv`, so mismatched basis produces
    # a mislabeled binary (invariant 14 violation). Matching or unset is OK.
    source_basis = Symbol(h.mass_basis)
    output_basis = mass_basis === nothing ? source_basis : Symbol(mass_basis)
    output_basis === source_basis || throw(ArgumentError(
        "regrid_ll_binary_to_cs: requested mass_basis=$(output_basis) " *
        "differs from source header mass_basis=$(source_basis). This " *
        "function does not perform dry↔moist conversion (it would need " *
        "to load `qv` from the source and apply `apply_dry_basis_native!` " *
        "to m/am/bm). Regenerate the source on the desired basis, or omit " *
        "the `mass_basis` kwarg to match the source."))
    source_has_dm = :dm in h.payload_sections
    source_has_dm || allow_terminal_zero_tendency || throw(ArgumentError(
        "regrid_ll_binary_to_cs requires source `dm` payloads to close the final " *
        "CS window safely. Source $(basename(ll_binary_path)) lacks `dm`; " *
        "regenerate the LL binary with flux deltas or pass " *
        "`allow_terminal_zero_tendency=true` for diagnostic-only regrids."
    ))
    source_has_dm && delta_semantics(reader) === :forward_window_endpoint_difference ||
        !source_has_dm || throw(ArgumentError(
            "regrid_ll_binary_to_cs requires source delta_semantics = " *
            ":forward_window_endpoint_difference when `dm` is present, got " *
            "$(delta_semantics(reader))."
        ))

    # Timestep metadata comes from the source header. The stored `am/bm`
    # are per-substep mass (flux_kind = :substep_mass_amount) with the
    # source's own `steps_per_window`, and the CS writer reuses the same
    # substep count — so the per-substep semantics match end-to-end.
    met_interval = Float64(h.dt_met_seconds)
    steps_per_met = Int(h.steps_per_window)
    dt_factor = FT(met_interval / (2 * steps_per_met))
    gravity = FT(GRAV)

    @info @sprintf("  LL source: %s (%d×%d×%d, %d windows)",
                   basename(ll_binary_path), Nx_ll, Ny_ll, Nz, Nt)
    @info @sprintf("  CS target: C%d (%d panels, %d levels)", Nc, CS_PANEL_COUNT, Nz)

    # --- Build LL source mesh for regridder ---
    # Reconstruct the LL mesh from the binary header metadata
    ll_mesh = LatLonMesh(; FT=FT,
                          size=(Nx_ll, Ny_ll),
                          longitude=(-180, 180),
                          latitude=(-90, 90),
                          radius=FT(R_EARTH))
    ll_lats = FT.(ll_mesh.φᶜ)
    Δy_ll = FT(ll_mesh.radius * deg2rad(ll_mesh.Δφ))
    Δlon_ll = FT(deg2rad(ll_mesh.Δλ))

    # --- Build conservative regridder (LL → CS) ---
    t_reg = time()
    regridder = build_regridder(ll_mesh, cs_grid.mesh;
                                normalize=false,
                                cache_dir=cs_grid.cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    @info @sprintf("  Regridder: %d→%d  nnz=%d (%.1fs)",
                   n_src, n_dst, length(regridder.intersections.nzval), time() - t_reg)

    # --- Allocate workspaces ---
    cs_ws = allocate_cs_preprocess_workspace(Nc, Nx_ll, Ny_ll, Nz, n_src, n_dst, FT)
    Δx = cs_grid.mesh.Δx
    Δy = cs_grid.mesh.Δy

    # Pre-allocate LL read buffers
    m_ll  = Array{FT}(undef, Nx_ll, Ny_ll, Nz)
    am_ll = Array{FT}(undef, Nx_ll + 1, Ny_ll, Nz)
    bm_ll = Array{FT}(undef, Nx_ll, Ny_ll + 1, Nz)
    cm_ll = Array{FT}(undef, Nx_ll, Ny_ll, Nz + 1)
    ps_ll = Array{FT}(undef, Nx_ll, Ny_ll)
    dm_ll = source_has_dm ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing

    # --- Build vertical coordinate from binary header ---
    vc_merged = HybridSigmaPressure(A_ifc, B_ifc)

    # --- Open streaming CS binary writer ---
    mkpath(dirname(out_path))
    writer = open_streaming_cs_transport_binary(
        out_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=met_interval,
        half_dt_seconds=met_interval / 2,
        steps_per_window=steps_per_met,
        include_flux_delta=true,
        mass_basis=output_basis,
        panel_convention=_cs_panel_convention_tag(cs_grid),
        extra_header=Dict{String, Any}(
            "preprocessor"      => "regrid_ll_binary_to_cs",
            "source_type"       => "ll_transport_binary",
            "source_path"       => ll_binary_path,
            "target_type"       => "cubed_sphere",
            "regrid_method"     => "conservative",
            "poisson_balanced"  => true,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(FT)
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(out_path),
                   expected_total / 1e9, Nt)
    @info "  Streaming: LL binary → CS regrid → balance → write..."
    write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
    write_replay_on || @info "  Write-time CS replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    replay_tol = replay_tolerance(FT)

    # --- Helper: read one LL window and regrid to CS ---
    function _read_and_regrid_to_cs!(win_idx, m_out, ps_out, am_out, bm_out)
        load_window!(reader, win_idx; m=m_ll, ps=ps_ll, am=am_ll, bm=bm_ll, cm=cm_ll)

        # Conservative regrid scalars: m, ps → CS panels
        regrid_3d_to_cs_panels!(m_out, regridder, m_ll, cs_ws, Nc)
        regrid_2d_to_cs_panels!(ps_out, regridder, ps_ll, cs_ws, Nc)
        _enforce_perlevel_mass_consistency!(m_out, m_ll, Nc, Nz)

        # Recover LL cell-center winds from binary's am/bm
        recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
            am_ll, bm_ll, ps_ll,
            A_ifc, B_ifc, ll_lats, Δy_ll, Δlon_ll,
            FT(ll_mesh.radius), gravity, dt_factor)

        # Regrid geographic winds to CS + rotate to panel-local
        regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, regridder, cs_ws.v_cc, cs_ws, Nc)
        rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_grid.tangent_basis, Nc, Nz)

        # Reconstruct CS face fluxes
        reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                               cs_ws.dp_panels, ps_out,
                               A_ifc, B_ifc, Δx, Δy, gravity, dt_factor, Nc, Nz)
    end

    # --- Pre-allocate sliding buffer ---
    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)

    @inline function _copy_panels_regrid!(dst, src)
        for p in 1:CS_PANEL_COUNT; copyto!(dst[p], src[p]); end
    end

    worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)

    # --- Process first window ---
    t0 = time()
    _read_and_regrid_to_cs!(1, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
    @info @sprintf("    Window  1/%d: read+regrid %.2fs", Nt, time() - t0)

    _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
    _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
    _copy_panels_regrid!(cur_am, cs_ws.am_panels)
    _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t0 = time()
        _read_and_regrid_to_cs!(win, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
        t_read = time() - t0

        _copy_panels_regrid!(cs_ws.m_next_panels, cs_ws.m_panels)

        t_bal = time()
        bal_diag = balance_cs_global_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
            cs_grid.poisson_scratch; tol=1e-14, max_iter=20000)
        t_bal = time() - t_bal

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        sync_all_cs_boundary_mirrors!(cur_am, cur_bm, cs_grid.mesh.connectivity, Nc, Nz)

        fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
        for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
        diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
        if write_replay_on
            diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                                  cs_ws.m_next_panels,
                                                  steps_per_met, replay_tol, win - 1)
            if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
                worst_replay_rel = diag_replay.max_rel_err
                worst_replay_abs = diag_replay.max_abs_err
                worst_replay_win = win - 1
                worst_replay_idx = diag_replay.worst_idx
            end
        end
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

        should_log_window(win - 1, Nt) &&
            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                           win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_read)

        _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
        _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
        _copy_panels_regrid!(cur_am, cs_ws.am_panels)
        _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)
    end

    # --- Balance & write LAST window ---
    if source_has_dm
        deltas = load_flux_delta_window!(reader, Nt; dm=dm_ll)
        (deltas !== nothing && haskey(deltas, :dm)) || throw(ArgumentError(
            "regrid_ll_binary_to_cs: source header for $(basename(ll_binary_path)) " *
            "declares `dm`, but window $(Nt) could not be loaded."
        ))
        @. cs_ws.u_cc = m_ll + dm_ll
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        _enforce_perlevel_mass_consistency!(cs_ws.m_next_panels, cs_ws.u_cc, Nc, Nz)
    else
        @warn "regrid_ll_binary_to_cs: using zero-tendency fallback for the final CS window because source `dm` is unavailable."
        _copy_panels_regrid!(cs_ws.m_next_panels, cur_m)
    end
    t_bal = time()
    bal_diag = balance_cs_global_mass_fluxes!(
        cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
        cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
        cs_grid.poisson_scratch; tol=1e-14, max_iter=5000)
    t_bal = time() - t_bal

    worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
    worst_post = max(worst_post, bal_diag.max_post_residual)
    worst_iter = max(worst_iter, bal_diag.max_cg_iter)

    sync_all_cs_boundary_mirrors!(cur_am, cur_bm, cs_grid.mesh.connectivity, Nc, Nz)

    fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
    for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
    diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
    if write_replay_on
        diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                              cs_ws.m_next_panels,
                                              steps_per_met, replay_tol, Nt)
        if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
            worst_replay_rel = diag_replay.max_rel_err
            worst_replay_abs = diag_replay.max_abs_err
            worst_replay_win = Nt
            worst_replay_idx = diag_replay.worst_idx
        end
    end
    convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

    window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                 dm=cs_ws.m_next_panels)
    write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre=%.2e post=%.2e iter=%d",
                   Nt, Nt, t_bal, bal_diag.max_pre_residual,
                   bal_diag.max_post_residual, bal_diag.max_cg_iter)

    close_streaming_transport_binary!(writer)
    close(reader)

    @info @sprintf("  Poisson balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                   worst_pre, worst_post, worst_iter)
    if write_replay_on
        replay_msg = worst_replay_win > 0 ?
            @sprintf("max rel=%.3e abs=%.3e kg win=%d cell=%s",
                     worst_replay_rel, worst_replay_abs, worst_replay_win, worst_replay_idx) :
            "no windows checked"
        @info "  Write-time replay gate: $replay_msg"
    end

    actual = filesize(out_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(out_path),
                   actual / 1e9, time() - t_start)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return out_path
end
