# ===========================================================================
# Native GEOS-IT/FP cubed-sphere → v4 transport binary preprocessing path.
#
# Source axis:  AbstractGEOSSettings (read native CTM_A1/CTM_I1 NetCDF)
# Target axis:  CubedSphereTargetGeometry, source mesh == target mesh
#               (passthrough — IdentityRegrid)
#
# Critical design choice (per the user's correction 2026-04-24): no Poisson
# balance is invoked on the passthrough path. FV3's native MFXC/MFYC are
# already discretely conservative; running a CG projection on top would only
# absorb floating-point noise at the cost of slightly distorting the
# physics-consistent fluxes. We DO run a write-time replay gate so any
# silent unit / sign / shape error fails loudly before the binary is shipped.
#
# ---------------------------------------------------------------------------
# Window-by-window loop:
#
#   read_window!(settings, handles, date, win)            # both endpoints + MFXC/MFYC
#   geos_native_to_face_flux!(am_v4, bm_v4, raw, conn,    # (Nc,Nc) → (Nc+1,Nc)/(Nc,Nc+1)
#                              Nc, Nz, dt_factor / g)     #     panel halos via mirror sync
#   fill_cs_window_mass_tendency!(dm, m, m_next, steps_per_met)
#   diagnose_cs_cm!(cm, am_v4, bm_v4, dm, m, Nc, Nz)
#   verify_write_replay_cs!(...)                          # < 1e-12 expected
#   convert_cs_mass_target_to_delta!(...)
#   write_streaming_cs_window!(writer, window_nt, Nc, 6)
# ===========================================================================

"""
    process_day(date, grid::CubedSphereTargetGeometry,
                settings::AbstractGEOSSettings, vertical;
                out_path,
                dt_met_seconds = 3600.0,
                FT = Float64,
                mass_basis = :dry,
                replay_tol = 1e-12,
                next_day_hour0 = nothing) -> NamedTuple

Build a v4 cubed-sphere transport binary at `out_path` from one UTC day of
native GEOS data. Source mesh and target mesh are required to match (CS
passthrough); for cross-resolution or cross-topology output, use the
LL/RG path or the LL→CS regrid (separate orchestrators).

Returns a NamedTuple with diagnostic statistics: worst replay-gate
relative error, worst cm residual, and elapsed time.

`next_day_hour0` is part of the inherited topology-dispatch contract but
unused here — the GEOS reader handles next-day endpoints internally via
`next_ctm_i1`.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::AbstractGEOSSettings,
                     vertical;
                     out_path::AbstractString,
                     dt_met_seconds::Real = 3600.0,
                     FT::Type{<:AbstractFloat} = Float64,
                     mass_basis::Symbol = :dry,
                     replay_tol::Real = 1e-12,
                     panel_convention::AbstractString = "geos_native",
                     next_day_hour0 = nothing)
    Nc     = grid.Nc
    npanel = CS_PANEL_COUNT
    Nz     = vertical.Nz
    vc     = vertical.merged_vc
    g      = FT(GRAV)

    # GEOS dynamics: each met window is an integral of `mass_flux_dt`-second
    # FV3 substeps. The v4 binary's per-substep flux scaling is:
    #   am_v4 = u × dp × Δy / g × dt_factor   with dt_factor = dt_met / (2·steps)
    # For native MFXC (already u·dp·Δy·mass_flux_dt summed-over-substeps in Pa·m²),
    # the equivalent conversion is:
    #   am_v4 = MFXC / mass_flux_dt × dt_factor / g
    # which the reader has already done for the / mass_flux_dt half. The
    # remaining factor is dt_factor / g.
    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    dt_factor = FT(dt_met_seconds / (2 * steps_per_met))
    flux_scale = dt_factor / g

    nw = windows_per_day(settings, date)

    @info "GEOS → CS: $(date), source=$(settings) → $(out_path)"
    @info "  Nc=$Nc  Nz=$Nz  windows=$nw  steps_per_met=$steps_per_met  flux_scale=$flux_scale"

    # ---- Open NCDataset handles (with next-day endpoint for last window) ----
    handles = open_geos_day(settings, date)
    @info "  Level orientation: $(handles.orientation)  (next-day endpoint: $(handles.next_ctm_i1 !== nothing))"

    mkpath(dirname(out_path))

    writer = open_streaming_cs_transport_binary(
        out_path, Nc, npanel, Nz, nw, vc;
        FT = FT,
        dt_met_seconds = dt_met_seconds,
        steps_per_window = steps_per_met,
        mass_basis = mass_basis,
        include_flux_delta = true,                # write-time replay needs `dm`
        panel_convention = panel_convention,
    )

    try
        # ---- v4-shape buffers (reused across windows) ----
        am_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz),     npanel)
        bm_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz),     npanel)
        cm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1),     npanel)
        dm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz),         npanel)
        m_target = ntuple(_ -> zeros(FT, Nc, Nc, Nz),      npanel)

        worst_replay_rel = 0.0
        worst_replay_abs = 0.0
        worst_replay_win = 0

        t_start = time()

        @inbounds for win in 1:nw
            raw = read_window!(settings, handles, date, win; FT = FT)

            # 1. Native MFXC/MFYC (Pa·m²/s on cell-centered indexing) → v4 face-staggered
            #    (kg per substep on (Nc+1,Nc,Nz) / (Nc,Nc+1,Nz)) with panel-halo mirror sync.
            geos_native_to_face_flux!(am_v4, bm_v4, raw.am, raw.bm,
                                      grid.mesh.connectivity, Nc, Nz, flux_scale)

            # 2. Mass tendency `dm = (m_next - m) / steps_per_met` (per-substep view).
            fill_cs_window_mass_tendency!(dm_v4, raw.m, raw.m_next, steps_per_met)

            # 3. Diagnose `cm` from continuity: cm[k+1] = cm[k] - (am_div + bm_div + dm)
            #    so the column closes mass by construction at every cell.
            for p in 1:npanel; fill!(cm_v4[p], zero(FT)); end
            diagnose_cs_cm!(cm_v4, am_v4, bm_v4, dm_v4, raw.m, Nc, Nz)

            # 4. Replay gate: integrate fluxes forward and check m_evolved vs m_next.
            replay = verify_write_replay_cs!(raw.m, am_v4, bm_v4, cm_v4,
                                             raw.m_next, steps_per_met, replay_tol, win)
            if worst_replay_win == 0 || replay.max_rel_err > worst_replay_rel
                worst_replay_rel = replay.max_rel_err
                worst_replay_abs = replay.max_abs_err
                worst_replay_win = win
            end

            # 5. Pack m_target = m_next as a delta and write the window.
            for p in 1:npanel; copyto!(m_target[p], raw.m_next[p]); end
            convert_cs_mass_target_to_delta!(m_target, raw.m)

            window_nt = (m  = raw.m,    am = am_v4, bm = bm_v4,
                         cm = cm_v4,    ps = raw.ps,
                         dm = m_target)
            write_streaming_cs_window!(writer, window_nt, Nc, npanel)
        end

        elapsed = time() - t_start
        @info @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                       elapsed, elapsed / nw, worst_replay_rel, worst_replay_abs, worst_replay_win)

        return (
            elapsed = elapsed,
            worst_replay_rel = worst_replay_rel,
            worst_replay_abs = worst_replay_abs,
            worst_replay_win = worst_replay_win,
            out_path = out_path,
        )
    finally
        close_streaming_transport_binary!(writer)
        close_geos_day!(handles)
    end
end
