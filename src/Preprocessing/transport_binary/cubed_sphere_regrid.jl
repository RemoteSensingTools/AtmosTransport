# Lat-lon transport-binary to cubed-sphere transport-binary regrid path.

"""
    _check_cs_substep_positivity(cur_m, cur_am, cur_bm, cur_cm, Nc, Hp, Nz;
                                  cfl_limit::Real = 0.95)

Verify the per-substep horizontal positivity contract that
`_cs_static_subcycle_count` uses at runtime: for every interior cell
on every panel and direction, the outgoing-mass-per-substep must not
exceed the cell's mass.  Returns `(direction, ratio, location)` for
the worst offender; `(nothing, 0.0, ())` when the binary is safe.

The replay gate (`verify_write_replay_cs!`) only checks endpoint
continuity (`m_n + Σ flux·dt = m_{n+1}`) and silently passes any
binary that nukes a cell mid-sweep before the negative tracer
re-fills from inflow.  This gate fires *before* the binary is
considered shippable.
"""
function _check_cs_substep_positivity(cur_m, cur_am, cur_bm, cur_cm,
                                       Nc::Int, Hp::Int, Nz::Int;
                                       cfl_limit::Real = 0.95)
    FT = eltype(cur_m[1])
    iL = Hp + 1
    iH = Hp + Nc
    worst_dir = nothing
    worst_ratio = 0.0
    worst_loc = (0, 0, 0, 0)
    for p in 1:6
        m_p = cur_m[p]
        for (dir, F_lo_view, F_hi_view) in (
            (:x, view(cur_am[p], iL    :iH,     iL:iH,     1:Nz),
                 view(cur_am[p], iL + 1:iH + 1, iL:iH,     1:Nz)),
            (:y, view(cur_bm[p], iL:iH,     iL    :iH,     1:Nz),
                 view(cur_bm[p], iL:iH,     iL + 1:iH + 1, 1:Nz)),
            (:z, view(cur_cm[p], iL:iH, iL:iH, 1    :Nz),
                 view(cur_cm[p], iL:iH, iL:iH, 2:Nz + 1)),
        )
            m_int = view(m_p, iL:iH, iL:iH, 1:Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                mi = m_int[i, j, k]
                mi > zero(FT) || continue
                fl = F_lo_view[i, j, k]
                fh = F_hi_view[i, j, k]
                outgoing = max(zero(FT), -fl) + max(zero(FT), fh)
                ratio = outgoing / mi
                if ratio > worst_ratio
                    worst_ratio = ratio
                    worst_dir = dir
                    worst_loc = (p, i, j, k)
                end
            end
        end
    end
    return (direction = worst_dir, ratio = Float64(worst_ratio),
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

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
from the source header by default. Passing `steps_per_window` overrides the
output substep count: LL winds are recovered with the source scaling, then CS
face fluxes are reconstructed with the output scaling.

## Keyword arguments
- `FT::Type = Float64` — on-disk float type for the output CS binary.
- `mass_basis::Union{Nothing, Symbol} = nothing` — output mass-basis label.
  `nothing` (default) = match the source. Setting this to a value that
  differs from the source's `mass_basis` currently errors: actual
  dry↔moist conversion requires loading the source's `qv` and applying
  `apply_dry_basis_native!`, which this function does not do. Invariant
  14 mandates `:dry` end-to-end; use a dry source.
- `steps_per_window::Union{Nothing, Integer} = nothing` — output substep
  count. `nothing` = match the source; larger values reduce stored
  per-substep fluxes while preserving the window-integrated transport.
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
                                allow_terminal_zero_tendency::Bool = false,
                                positivity_cfl_limit::Real = 0.95,
                                require_substep_positivity::Bool = true,
                                steps_per_window::Union{Nothing, Integer} = nothing,
                                cs_balance_tol::Real = 1e-14,
                                cs_balance_project_every::Integer = 50)
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
    source_has_tm5 = has_tm5_convection(reader)
    source_has_surface = all(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m))
    source_surface_partial = any(s in h.payload_sections for s in (:pblh, :ustar, :pbl_hflux, :t2m)) &&
                             !source_has_surface
    source_surface_partial && throw(ArgumentError(
        "regrid_ll_binary_to_cs: source has a partial PBL surface payload; " *
        "expected all of pblh, ustar, pbl_hflux, t2m."))
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
    #
    # `steps_per_window` keyword overrides the substep count for the CS
    # output. Recover winds with the source scaling, then reconstruct CS
    # face fluxes with the output scaling; using one factor for both would
    # cancel the override. The forward delta (dm) is rescaled by
    # `fill_cs_window_mass_tendency!`, so a larger value produces smaller
    # per-substep flux without changing the window-integrated transport.
    met_interval = Float64(h.dt_met_seconds)
    src_steps_per_met = Int(h.steps_per_window)
    steps_per_met = steps_per_window === nothing ? src_steps_per_met : Int(steps_per_window)
    steps_per_met >= 1 || throw(ArgumentError(
        "steps_per_window must be ≥ 1, got $(steps_per_met)"))
    if steps_per_met != src_steps_per_met
        @info @sprintf("  steps_per_window override: source=%d → output=%d (%.3fx flux per substep)",
                       src_steps_per_met, steps_per_met, src_steps_per_met / steps_per_met)
    end
    src_dt_factor = FT(met_interval / (2 * src_steps_per_met))
    out_dt_factor = FT(met_interval / (2 * steps_per_met))
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

    # Raw PBL surface fields are per-area/intensive diagnostics. They are
    # area-averaged to CS cell centers and carried verbatim in the output
    # binary; runtime derives Kz from them for the active met window.
    pblh_ll  = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    ustar_ll = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    hflux_ll = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing
    t2m_ll   = source_has_surface ? Array{FT}(undef, Nx_ll, Ny_ll) : nothing

    # TM5 convection sections are layer-center kg/m²/s.  Conservative
    # LL→CS regrid is the right path (per
    # `feedback_conservative_regrid_for_mass_fluxes.md`); no rotation,
    # no Poisson balance — they're purely vertical-flux carriers, the
    # horizontal advection sees zero divergence from them.
    entu_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    detu_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    entd_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing
    detd_ll = source_has_tm5 ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing

    # --- Build vertical coordinate from binary header ---
    vc_merged = HybridSigmaPressure(A_ifc, B_ifc)

    # --- Open streaming CS binary writer ---
    # Stage to `out_path.tmp` so the positivity gate (final step) can
    # error without leaving a bad binary at `out_path`. Renamed to the
    # final name on success.
    mkpath(dirname(out_path))
    tmp_path = out_path * ".tmp"
    isfile(tmp_path) && rm(tmp_path)
    writer = open_streaming_cs_transport_binary(
        tmp_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=met_interval,
        half_dt_seconds=met_interval / 2,
        steps_per_window=steps_per_met,
        include_flux_delta=true,
        include_surface=source_has_surface,
        include_tm5conv=source_has_tm5,
        mass_basis=output_basis,
        panel_convention=_cs_panel_convention_tag(cs_grid),
        cs_definition=_cs_definition_tag(cs_grid),
        cs_coordinate_law=_cs_coordinate_law_tag(cs_grid),
        cs_center_law=_cs_center_law_tag(cs_grid),
        longitude_offset_deg=longitude_offset_deg(cs_definition(cs_grid.mesh)),
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
    function _read_and_regrid_to_cs!(win_idx, m_out, ps_out, am_out, bm_out;
                                     pblh_out = nothing, ustar_out = nothing,
                                     hflux_out = nothing, t2m_out = nothing,
                                     entu_out = nothing, detu_out = nothing,
                                     entd_out = nothing, detd_out = nothing)
        load_window!(reader, win_idx; m=m_ll, ps=ps_ll, am=am_ll, bm=bm_ll, cm=cm_ll)

        # Conservative regrid:
        #   `m` (kg/cell, extensive) → density convert → regrid → ×dst_area
        #   `ps` (Pa, intensive) → straight area-averaged regrid
        # Without the `ExtensiveCellField()` tag, `m` would be treated as
        # intensive and the LL pole-row's small cells would distort the CS
        # polar mass by ~12×. See `Quantities.jl` for the full taxonomy.
        regrid_3d_to_cs_panels!(m_out, regridder, m_ll, cs_ws, Nc, ExtensiveCellField())
        regrid_2d_to_cs_panels!(ps_out, regridder, ps_ll, cs_ws, Nc, IntensiveCellField())

        # Recover LL cell-center winds from binary's am/bm
        recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
            am_ll, bm_ll, ps_ll,
            A_ifc, B_ifc, ll_lats, Δy_ll, Δlon_ll,
            FT(ll_mesh.radius), gravity, src_dt_factor)

        # Regrid geographic winds to CS + rotate to panel-local
        regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, regridder, cs_ws.v_cc, cs_ws, Nc)
        rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_grid.tangent_basis, Nc, Nz)

        # Reconstruct CS face fluxes
        reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                               cs_ws.dp_panels, ps_out,
                               A_ifc, B_ifc, Δx, Δy, gravity, out_dt_factor, Nc, Nz)

        if source_has_surface
            load_surface_window!(reader, win_idx;
                                 pblh = pblh_ll, ustar = ustar_ll,
                                 hflux = hflux_ll, t2m = t2m_ll)
            regrid_2d_to_cs_panels!(pblh_out, regridder, pblh_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(ustar_out, regridder, ustar_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(hflux_out, regridder, hflux_ll, cs_ws, Nc, IntensiveCellField())
            regrid_2d_to_cs_panels!(t2m_out, regridder, t2m_ll, cs_ws, Nc, IntensiveCellField())
        end

        # TM5 sections (layer-center kg/m²/s): conservative regrid to
        # CS panels.  No rotation — entu/detu/entd/detd are scalar
        # vertical-flux carriers, not horizontal vectors.
        if source_has_tm5
            load_tm5_convection_window!(reader, win_idx;
                                         entu = entu_ll, detu = detu_ll,
                                         entd = entd_ll, detd = detd_ll)
            regrid_3d_to_cs_panels!(entu_out, regridder, entu_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(detu_out, regridder, detu_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(entd_out, regridder, entd_ll, cs_ws, Nc)
            regrid_3d_to_cs_panels!(detd_out, regridder, detd_ll, cs_ws, Nc)
        end
    end

    # --- Pre-allocate sliding buffer ---
    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)
    cur_pblh  = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_ustar = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_hflux = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    cur_t2m   = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_pblh  = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_ustar = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_hflux = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    nxt_t2m   = source_has_surface ? ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT) : nothing
    # TM5 sliding buffers: 2 sets — `cur_*` holds the window we're
    # writing, `nxt_*` holds the next window's read so we can swap
    # without re-allocating.  Both sets only allocate when the source
    # actually carries TM5 sections.
    cur_entu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_detu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_entd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    cur_detd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_entu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_detu = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_entd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing
    nxt_detd = source_has_tm5 ? ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT) : nothing

    @inline function _copy_panels_regrid!(dst, src)
        for p in 1:CS_PANEL_COUNT; copyto!(dst[p], src[p]); end
    end

    worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)
    worst_pos_ratio = 0.0
    worst_pos_dir   = :none
    worst_pos_win   = 0
    worst_pos_loc   = (0, 0, 0, 0)

    # --- Process first window ---
    t0 = time()
    _read_and_regrid_to_cs!(1, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels;
                             pblh_out = cur_pblh, ustar_out = cur_ustar,
                             hflux_out = cur_hflux, t2m_out = cur_t2m,
                             entu_out = cur_entu, detu_out = cur_detu,
                             entd_out = cur_entd, detd_out = cur_detd)
    @info @sprintf("    Window  1/%d: read+regrid %.2fs", Nt, time() - t0)

    _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
    _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
    _copy_panels_regrid!(cur_am, cs_ws.am_panels)
    _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)
    # cur_entu/detu/entd/detd were filled directly by `_read_and_regrid_to_cs!`
    # (TM5 sections need no rebalancing — they're vertical-flux carriers).

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t0 = time()
        _read_and_regrid_to_cs!(win, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels;
                                 pblh_out = nxt_pblh, ustar_out = nxt_ustar,
                                 hflux_out = nxt_hflux, t2m_out = nxt_t2m,
                                 entu_out = nxt_entu, detu_out = nxt_detu,
                                 entd_out = nxt_entd, detd_out = nxt_detd)
        t_read = time() - t0

        _copy_panels_regrid!(cs_ws.m_next_panels, cs_ws.m_panels)

        t_bal = time()
        bal_diag = balance_cs_global_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
            cs_grid.poisson_scratch; tol=Float64(cs_balance_tol),
            max_iter=20000, project_every=Int(cs_balance_project_every))
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
        # Per-substep positivity gate. The replay test above only proves
        # endpoint continuity (m_n + Σ flux·dt = m_{n+1}); a binary that
        # nukes a cell mid-sweep can still pass replay because the cell
        # re-fills from inflow before the window ends. This gate is the
        # actual contract the runtime's `_cs_static_subcycle_count`
        # depends on.
        pos_diag = _check_cs_substep_positivity(cur_m, cur_am, cur_bm, cur_cm,
                                                 Nc, 0, Nz;
                                                 cfl_limit = positivity_cfl_limit)
        if pos_diag.ratio > worst_pos_ratio
            worst_pos_ratio = pos_diag.ratio
            worst_pos_dir   = pos_diag.direction === nothing ? :none : pos_diag.direction
            worst_pos_win   = win - 1
            worst_pos_loc   = pos_diag.location
        end
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        if source_has_surface
            window_nt = merge(window_nt,
                              (surface=(pblh=cur_pblh, ustar=cur_ustar,
                                        hflux=cur_hflux, t2m=cur_t2m),))
        end
        if source_has_tm5
            window_nt = merge(window_nt,
                              (tm5_fields=(entu=cur_entu, detu=cur_detu,
                                           entd=cur_entd, detd=cur_detd),))
        end
        write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

        should_log_window(win - 1, Nt) &&
            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                           win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_read)

        _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
        _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
        _copy_panels_regrid!(cur_am, cs_ws.am_panels)
        _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)
        if source_has_surface
            _copy_panels_regrid!(cur_pblh, nxt_pblh)
            _copy_panels_regrid!(cur_ustar, nxt_ustar)
            _copy_panels_regrid!(cur_hflux, nxt_hflux)
            _copy_panels_regrid!(cur_t2m, nxt_t2m)
        end
        if source_has_tm5
            _copy_panels_regrid!(cur_entu, nxt_entu)
            _copy_panels_regrid!(cur_detu, nxt_detu)
            _copy_panels_regrid!(cur_entd, nxt_entd)
            _copy_panels_regrid!(cur_detd, nxt_detd)
        end
    end

    # --- Balance & write LAST window ---
    if source_has_dm
        deltas = load_flux_delta_window!(reader, Nt; dm=dm_ll)
        (deltas !== nothing && haskey(deltas, :dm)) || throw(ArgumentError(
            "regrid_ll_binary_to_cs: source header for $(basename(ll_binary_path)) " *
            "declares `dm`, but window $(Nt) could not be loaded."
        ))
        @. cs_ws.u_cc = m_ll + dm_ll
        # `m + dm` is extensive (kg/cell): density-convert via the dispatcher.
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, regridder, cs_ws.u_cc,
                                cs_ws, Nc, ExtensiveCellField())
    else
        @warn "regrid_ll_binary_to_cs: using zero-tendency fallback for the final CS window because source `dm` is unavailable."
        _copy_panels_regrid!(cs_ws.m_next_panels, cur_m)
    end
    t_bal = time()
    bal_diag = balance_cs_global_mass_fluxes!(
        cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
        cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
        cs_grid.poisson_scratch; tol=Float64(cs_balance_tol),
        max_iter=5000, project_every=Int(cs_balance_project_every))
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
    pos_diag = _check_cs_substep_positivity(cur_m, cur_am, cur_bm, cur_cm,
                                             Nc, 0, Nz;
                                             cfl_limit = positivity_cfl_limit)
    if pos_diag.ratio > worst_pos_ratio
        worst_pos_ratio = pos_diag.ratio
        worst_pos_dir   = pos_diag.direction === nothing ? :none : pos_diag.direction
        worst_pos_win   = Nt
        worst_pos_loc   = pos_diag.location
    end
    convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

    window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                 dm=cs_ws.m_next_panels)
    if source_has_surface
        window_nt = merge(window_nt,
                          (surface=(pblh=cur_pblh, ustar=cur_ustar,
                                    hflux=cur_hflux, t2m=cur_t2m),))
    end
    if source_has_tm5
        window_nt = merge(window_nt,
                          (tm5_fields=(entu=cur_entu, detu=cur_detu,
                                       entd=cur_entd, detd=cur_detd),))
    end
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

    pos_msg = @sprintf("max outgoing/m=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                       worst_pos_ratio, worst_pos_dir, worst_pos_win,
                       worst_pos_loc, positivity_cfl_limit)
    if worst_pos_ratio > positivity_cfl_limit
        recommended = max(steps_per_met,
                          ceil(Int, worst_pos_ratio / positivity_cfl_limit) * steps_per_met)
        msg = "Per-substep positivity contract violated: $pos_msg. " *
              "The runtime's `_cs_static_subcycle_count` would still subcycle " *
              "(post `mapreduce(max, ...)` patch), but the binary itself stores " *
              "fluxes that violate positivity at the recorded `steps_per_window=$(steps_per_met)`. " *
              "Suggest re-running with `steps_per_window=$recommended` (set " *
              "via the `[numerics].steps_per_met` knob on the source LL " *
              "preprocessing config), or re-invoke `regrid_ll_binary_to_cs` " *
              "with `require_substep_positivity=false` to suppress."
        if require_substep_positivity
            # Quarantine the bad binary so a downstream consumer can't
            # accidentally pick it up — the gate is the contract, the
            # `.tmp` path is the staging area.
            isfile(tmp_path) && rm(tmp_path; force = true)
            error(msg)
        else
            @warn msg
            # Promote the tmp file even on warn-only so the caller gets
            # a binary at `out_path` for diagnostic inspection.
            mv(tmp_path, out_path; force = true)
        end
    else
        @info "  Per-substep positivity gate: $pos_msg"
        mv(tmp_path, out_path; force = true)
    end

    actual = filesize(out_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(out_path),
                   actual / 1e9, time() - t_start)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return out_path
end
