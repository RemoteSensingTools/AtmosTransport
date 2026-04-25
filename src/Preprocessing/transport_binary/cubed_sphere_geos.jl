# ===========================================================================
# Native GEOS-IT/FP cubed-sphere → v4 transport binary preprocessing path.
#
# Source axis:  AbstractGEOSSettings (read native CTM_A1/CTM_I1 NetCDF)
# Target axis:  CubedSphereTargetGeometry, source mesh == target mesh
#               (passthrough — IdentityRegrid)
#
# Critical design choices:
#
#  1. **No Poisson balance.** FV3's native MFXC/MFYC are already discretely
#     conservative; running a CG projection on top would only absorb
#     floating-point noise at the cost of distorting the physics-consistent
#     fluxes. (User correction 2026-04-24.)
#
#  2. **Pressure-fixer cm + chained endpoint mass** (codex Option C, validated
#     2026-04-25). FV3 conserves moist mass; per-column dry mass changes via
#     both horizontal MFXC divergence AND vertical moisture transport. The
#     raw GEOS DELP_dry endpoints don't satisfy strict per-level
#     `(m_next-m)/(2·steps) = -(div_h+div_v)` for any local `cm` choice.
#     The historical GEOS-FP runner (commit `76fa489::compute_cm_panel_cpu!`)
#     instead used FV3's pressure-fixer rule
#       `cm[k+1]-cm[k] = C_k - ΔB[k]·pit`,  pit = Σ_k C_k
#     which closes `cm[Nz+1] = 0` exactly without any per-cell residual
#     redistribution. Substituted into the v4 replay equation, the per-level
#     mass evolution is `Δm[k] = +2·steps · ΔB[k]·pit`. We CHAIN the stored
#     `m`: window 1 starts from raw GEOS DELP_dry; subsequent windows take
#     `m_cur = m_next_pf` from the previous window. The stored m drifts
#     from raw GEOS DELP over the day by exactly the column moisture-source
#     term (small for dry CO2 transport), but the binary is internally
#     self-consistent: replay closes to roundoff and the runtime tracer
#     mass evolves with the same fluxes that produced m_evolved.
#
#  3. **Window-by-window loop**:
#
#       read_window!(settings, handles, date, win)         # raw GEOS endpoints
#       geos_native_to_face_flux!(am_v4, bm_v4, ...)       # face-stagger + panel halos
#       compute_cs_cm_pressure_fixer!(cm_v4, am_v4, bm_v4, ΔB, ...)
#       evolve m_next_pf = m_cur + 2·steps · ΔB·pit         # closes replay exactly
#       fill dm = m_next_pf - m_cur, write window, m_cur ← m_next_pf
# ===========================================================================

"""
    _delp_pa_to_air_mass_kg!(m_kg, m_pa, cell_areas, inv_g) -> m_kg

In-place: convert pressure thickness in Pa to cell air mass in kg per
`m_kg[i, j, k] = m_pa[i, j, k] × cell_areas[i, j] × inv_g`. Cell areas are
in m² and apply identically to every CS panel by symmetry.
"""
function _delp_pa_to_air_mass_kg!(m_kg::AbstractArray{FT, 3},
                                  m_pa::AbstractArray{FT, 3},
                                  cell_areas::AbstractMatrix{FT},
                                  inv_g::FT) where {FT}
    Nx, Ny, Nz = size(m_kg)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_kg[i, j, k] = m_pa[i, j, k] * cell_areas[i, j] * inv_g
    end
    return m_kg
end

"""
    _evolve_mass_pressure_fixer!(m_next, m_cur, am_v4, bm_v4, ΔB,
                                 two_steps, Nc, Nz)

Per-cell column evolution under the FV3 pressure-fixer rule:

    pit       = Σ_k (am_inflow_k + bm_inflow_k)
    m_next[k] = m_cur[k] + two_steps · ΔB[k] · pit

This is the unique mass evolution that makes the replay equation close
exactly when the stored `cm` is the pressure-fixer's
`cm[k+1]-cm[k] = C_k - ΔB[k]·pit`. See module-header rationale for why
this differs from the raw GEOS DELP_dry endpoint tendency.
"""
function _evolve_mass_pressure_fixer!(
        m_next::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_cur::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        ΔB::AbstractVector,
        two_steps::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for p in 1:CS_PANEL_COUNT
        am = am_v4[p]; bm = bm_v4[p]
        m  = m_cur[p]; mn = m_next[p]
        for j in 1:Nc, i in 1:Nc
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i, j, k] - am[i + 1, j, k]) +
                       (bm[i, j, k] - bm[i, j + 1, k])
            end
            for k in 1:Nz
                mn[i, j, k] = m[i, j, k] + two_steps * FT(ΔB[k]) * pit
            end
        end
    end
    return nothing
end

"""
    _ps_from_air_mass!(ps, m, area, g, Nc, Nz)

Set `ps[i,j] = (Σ_k m[i,j,k]) · g / area[i,j]` (Pa). Used to keep the
binary's stored `ps` consistent with the chained pressure-fixer mass.
"""
function _ps_from_air_mass!(ps::AbstractMatrix{FT},
                            m::AbstractArray{FT, 3},
                            cell_areas::AbstractMatrix{FT},
                            g::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += m[i, j, k]
        end
        ps[i, j] = s * g / cell_areas[i, j]
    end
    return ps
end

"""
    process_day(date, grid::CubedSphereTargetGeometry,
                settings::AbstractGEOSSettings, vertical;
                out_path,
                dt_met_seconds = 3600.0,
                FT = Float64,
                mass_basis = :dry,
                replay_tol = replay_tolerance(FT),
                seed_m = nothing,
                next_day_hour0 = nothing) -> NamedTuple

Build a v4 cubed-sphere transport binary at `out_path` from one UTC day of
native GEOS data. Source mesh and target mesh must match (CS passthrough).

Stored mass is the pressure-fixer's chained evolution (see module header),
not the raw GEOS DELP_dry. The replay gate closes to roundoff by
construction; the maximum absolute residual goes down to floating-point
noise instead of the ~1% column-residual the naive
`m=DELP_dry, cm=diagnose_cs_cm` path produced.

For multi-day preprocessing, `seed_m` carries the pressure-fixer endpoint
from the previous day so adjacent daily binaries share a boundary mass:
pass `nothing` (default) on day 1 to seed from raw GEOS DELP_dry, and on
day N+1 pass the `final_m` returned by day N's `process_day`. Without
this threading, each day reinitializes from raw GEOS and the chained
mass discontinuously jumps at every daily boundary.

The returned NamedTuple includes `final_m::NTuple{6, Array{FT, 3}}`, the
pressure-fixer state at the END of the last window — i.e., the seed for
the next day's `process_day`.

`next_day_hour0` is part of the inherited topology-dispatch contract but
unused — the GEOS reader handles next-day endpoints internally via
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
                     replay_tol::Real = replay_tolerance(FT),
                     seed_m::Union{Nothing, NTuple{6, <:AbstractArray}} = nothing,
                     next_day_hour0 = nothing)
    # Reject configurations the path cannot honor:
    mass_basis === :dry ||
        error("GEOS-CS passthrough only supports mass_basis=:dry; got $(mass_basis). " *
              "GEOS MFXC/MFYC are already dry; the chained pressure-fixer is dry-basis.")
    grid.mesh.convention isa GEOSNativePanelConvention ||
        error("GEOS-CS passthrough requires panel_convention=`geos_native` on " *
              "the target geometry; got $(typeof(grid.mesh.convention)).")
    # Single source of truth: the binary's panel-convention attrib comes from
    # the target mesh's convention, not from a duplicated config key.
    panel_convention = "geos_native"

    Nc     = grid.Nc
    npanel = CS_PANEL_COUNT
    Nz     = vertical.Nz
    vc     = vertical.merged_vc
    g      = FT(GRAV)
    inv_g  = inv(g)
    cell_areas = grid.mesh.cell_areas       # (Nc, Nc) — same metric for all panels

    # ΔB[k] = B[k+1] − B[k] (top-to-bottom; Σ ΔB = 1 by construction).
    ΔB = FT[FT(vc.B[k + 1] - vc.B[k]) for k in 1:Nz]

    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    dt_factor = FT(dt_met_seconds / (2 * steps_per_met))
    flux_scale = dt_factor / g
    two_steps = FT(2 * steps_per_met)

    nw = windows_per_day(settings, date)

    @info "GEOS → CS: $(date), source=$(settings) → $(out_path)"
    @info "  Nc=$Nc  Nz=$Nz  windows=$nw  steps_per_met=$steps_per_met  flux_scale=$flux_scale"

    handles = open_day(settings, date)
    @info "  Level orientation: $(handles.orientation)  (next-day endpoint: $(handles.next_ctm_i1 !== nothing))"

    mkpath(dirname(out_path))

    writer = open_streaming_cs_transport_binary(
        out_path, Nc, npanel, Nz, nw, vc;
        FT = FT,
        dt_met_seconds = dt_met_seconds,
        steps_per_window = steps_per_met,
        mass_basis = mass_basis,
        include_flux_delta = true,
        panel_convention = panel_convention,
    )

    try
        # ---- v4-shape buffers (reused across windows) ----
        am_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz),     npanel)
        bm_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz),     npanel)
        cm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1),     npanel)
        dm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz),         npanel)
        # Chained-state buffers: m_cur (this window's start) and m_next_pf
        # (the pressure-fixer-evolved end, which becomes next window's m_cur).
        m_cur     = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     npanel)
        m_next_pf = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     npanel)
        ps_cur    = ntuple(_ -> zeros(FT, Nc, Nc),         npanel)
        # Source-grid raw window: filled in place by `read_window!` once per window.
        raw = allocate_raw_window(settings; FT = FT, Nz = Nz)

        worst_replay_rel = 0.0
        worst_replay_abs = 0.0
        worst_replay_win = 0

        t_start = time()

        @inbounds for win in 1:nw
            read_window!(raw, settings, handles, date, win)

            # 1. Native MFXC/MFYC (Pa·m²/s on cell-centered indexing) → v4
            #    face-staggered (kg per substep) with panel-halo one-way prop.
            geos_native_to_face_flux!(am_v4, bm_v4, raw.am, raw.bm,
                                      grid.mesh.connectivity, Nc, Nz, flux_scale)

            # 2. First window: seed m_cur from `seed_m` (previous day's PF
            #    endpoint) when supplied, otherwise from raw GEOS DELP_dry.
            #    Subsequent windows: m_cur was set by the previous window's
            #    pressure-fixer evolution.
            if win == 1
                if seed_m === nothing
                    for p in 1:npanel
                        _delp_pa_to_air_mass_kg!(m_cur[p], raw.m[p], cell_areas, inv_g)
                    end
                else
                    for p in 1:npanel
                        size(seed_m[p]) == (Nc, Nc, Nz) ||
                            error("seed_m[$p] shape $(size(seed_m[p])) ≠ ($Nc, $Nc, $Nz)")
                        copyto!(m_cur[p], seed_m[p])
                    end
                end
                for p in 1:npanel
                    _ps_from_air_mass!(ps_cur[p], m_cur[p], cell_areas, g, Nc, Nz)
                end
            end

            # 3. Pressure-fixer cm (closes cm[Nz+1]=0 by construction).
            compute_cs_cm_pressure_fixer!(cm_v4, am_v4, bm_v4, ΔB, Nc, Nz)

            # 4. Pressure-fixer-evolved next-window mass.
            _evolve_mass_pressure_fixer!(m_next_pf, m_cur, am_v4, bm_v4, ΔB,
                                         two_steps, Nc, Nz)

            # 5. Per-substep mass tendency consistent with the stored cm:
            #    dm[k] = (m_next_pf[k] − m_cur[k]) / (2·steps_per_met).
            fill_cs_window_mass_tendency!(dm_v4, m_cur, m_next_pf, steps_per_met)

            # 6. Replay gate: under the chained PF state this closes by
            #    construction at every cell at every level.
            replay = verify_write_replay_cs!(m_cur, am_v4, bm_v4, cm_v4,
                                             m_next_pf, steps_per_met, replay_tol, win)
            if worst_replay_win == 0 || replay.max_rel_err > worst_replay_rel
                worst_replay_rel = replay.max_rel_err
                worst_replay_abs = replay.max_abs_err
                worst_replay_win = win
            end

            # 7. Pack the on-disk endpoint delta `dm = m_next_pf − m_cur` and write.
            #    `convert_cs_mass_target_to_delta!` mutates m_next_pf into the
            #    delta in place; we use a fresh copy so the chained state survives.
            m_target = ntuple(p -> copy(m_next_pf[p]), npanel)
            convert_cs_mass_target_to_delta!(m_target, m_cur)

            window_nt = (m  = m_cur,    am = am_v4, bm = bm_v4,
                         cm = cm_v4,    ps = ps_cur,
                         dm = m_target)
            write_streaming_cs_window!(writer, window_nt, Nc, npanel)

            # 8. Chain: next window's m_cur is this window's m_next_pf, and
            #    ps follows from m via Σ.
            for p in 1:npanel
                copyto!(m_cur[p], m_next_pf[p])
                _ps_from_air_mass!(ps_cur[p], m_cur[p], cell_areas, g, Nc, Nz)
            end
        end

        elapsed = time() - t_start
        @info @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                       elapsed, elapsed / nw, worst_replay_rel, worst_replay_abs, worst_replay_win)

        # Capture the pressure-fixer endpoint from the last window so the
        # caller can seed the next day's `process_day` and preserve cross-day
        # continuity (codex 2026-04-25 P2).
        final_m = ntuple(p -> copy(m_cur[p]), npanel)

        return (
            elapsed = elapsed,
            worst_replay_rel = worst_replay_rel,
            worst_replay_abs = worst_replay_abs,
            worst_replay_win = worst_replay_win,
            out_path = out_path,
            final_m = final_m,
        )
    finally
        close_streaming_transport_binary!(writer)
        close_day!(handles)
    end
end
