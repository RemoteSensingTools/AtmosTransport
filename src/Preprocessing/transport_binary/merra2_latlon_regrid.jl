# ===========================================================================
# MERRA-2 wind-derived → C180 cubed-sphere transport-binary writer.
#
# Reproduces the validated GEOS-Chem CO₂ transport input path: derive the
# horizontal mass fluxes from MERRA-2 WINDS (U/V) + a Cameron-Smith column
# pressure-fix (the Poisson balance), instead of GEOS native cubed-sphere
# MFXC. Purely additive — the GEOS-native and ERA5 paths are untouched.
#
# This is a near-clone of `process_era5_n320_to_cs_day`
# (transport_binary/era5_n320_regrid.jl): identical mass-derivation, global
# dry-mass pin, wind rotation, flux reconstruction, Poisson balance
# (= the pressure-fixer), cm diagnosis, adaptive substep policy, contract
# verification, and streaming writer. The ONLY substantive change is replacing
# the ERA5 spectral pipeline with a direct MERRA-2 NetCDF read + conservative
# regrid to C180, and `nwindow = 8` instead of 24.
#
# Drives one UTC day end-to-end:
#
#   per window (8 × 3-hourly):
#     1. Read native MERRA-2 LL fields (PS/QV from inst3 slice `win`, U/V from
#        tavg3 slice `win` = the 3-hr time-average advecting winds) and
#        conservatively regrid PS / U / V / QV to the C180 target.
#     2. Re-derive dry-mass on C180 from the regridded moist PS + QV so the
#        target-side column closure Σ_k DELP_dry = PS_dry holds to roundoff.
#     3. Rotate cell-centre winds geographic → panel-local using the CS
#        tangent basis.
#     4. Reconstruct Arakawa-C face mass fluxes (am, bm) from rotated U/V
#        and panel DELP via the existing CS helper.
#
#   per window transition (windows 2..8):
#     5. Read the next window's fields so we can close continuity against the
#        explicit endpoint-mass target.
#     6. Poisson-balance the current window's horizontal fluxes against the
#        next-window mass tendency (the Cameron-Smith column pressure-fix).
#     7. Diagnose cm from the balanced fluxes + endpoint mass tendency.
#     8. Verify the per-substep positivity gate and the write-time replay gate.
#     9. Convert the next-window mass target into the forward `dm` payload and
#        stream-write the window to the staging binary.
#
# The final window's right endpoint is the next day's inst3 slice-1 PS/QV; on
# the archive boundary a zero-tendency fallback is used with a warning,
# mirroring the ERA5 N320 writer.
#
# Surface / VDIFF / TM5-convection payloads are intentionally not written on
# this branch — only the core `m, am, bm, cm, ps, dm` sections.
# ===========================================================================

# Cap on adaptive substep refinements per window (mirror of the N320 path).
const _MERRA2_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8

"""
    MERRA2ToC180Pipeline{FT, R}

Per-day MERRA-2 → C180 preprocessing workspace. Owns the conservative LL→CS
regridder, the CS preprocess scratch, and the per-window regridded C180
scalar fields (`c180_fields.{ps, qv, u, v}`), laid out as `NTuple{6, …}`
panels so the shared CS helpers (`derive_c180_dry_mass!`,
`rotate_winds_to_panel_local!`, `reconstruct_cs_fluxes!`) work unchanged.

One pipeline allocated per day, reused across the 8 windows.
"""
struct MERRA2ToC180Pipeline{FT <: AbstractFloat, R}
    regridder   :: R
    ws          :: CubedSpherePreprocessWorkspace{FT}
    Nz          :: Int
    c180_fields :: NamedTuple{(:ps, :qv, :u, :v),
                              Tuple{NTuple{6, Matrix{FT}},
                                    NTuple{6, Array{FT, 3}},
                                    NTuple{6, Array{FT, 3}},
                                    NTuple{6, Array{FT, 3}}}}
end

"""
    _merra2_source_latlon_mesh(FT, radius) -> LatLonMesh

Build the MERRA-2 native source mesh with cell CENTERS coincident with the
archive coordinates — lon centers at `-180:0.625:179.375` (periodic; faces at
±0.3125° so each data point IS its cell center, not the west edge) and lat
points at `-90:0.5:90` with the two POLAR cells as half-width caps clamped at
±90 (the GEOS-5 point-registered grid includes the poles as points). This makes
the conservative regridder use the correct source-cell geometry. A plain
`LatLonMesh(longitude=(-180,180), latitude=(-90,90))` offsets every center by a
half-cell (centers at -179.6875°, lat spacing 180/361) → a ~0.3° spatial shift
and non-archive areas (Codex P1, 2026-06-04). Faces are passed explicitly to the
inner constructor; `ConservativeRegridding` builds cell polygons from `λᶠ`/`φᶠ`.
"""
function _merra2_source_latlon_mesh(::Type{FT}, radius) where FT
    Δλ = FT(360) / MERRA2_NX                        # 0.625°
    Δφ = FT(180) / (MERRA2_NY - 1)                  # 0.5°
    λᶜ = FT[FT(-180) + (i - 1) * Δλ for i in 1:MERRA2_NX]           # archive lon centers
    λᶠ = FT[FT(-180) - Δλ / 2 + (i - 1) * Δλ for i in 1:MERRA2_NX + 1]
    φᶜ = FT[FT(-90) + (j - 1) * Δφ for j in 1:MERRA2_NY]            # -90:0.5:90 (poles incl.)
    φᶠ = Vector{FT}(undef, MERRA2_NY + 1)
    φᶠ[1] = FT(-90)
    @inbounds for j in 2:MERRA2_NY
        φᶠ[j] = FT(-90) + (FT(j) - FT(1.5)) * Δφ    # midpoint(φᶜ[j-1], φᶜ[j])
    end
    φᶠ[MERRA2_NY + 1] = FT(90)
    return LatLonMesh{FT}(MERRA2_NX, MERRA2_NY, Δλ, Δφ, λᶜ, λᶠ, φᶜ, φᶠ, FT(radius))
end

"""
    allocate_merra2_to_c180_pipeline(target_grid; Nz, cache_dir) -> MERRA2ToC180Pipeline

Build (or JLD2-load from `cache_dir`) the MERRA-2 LL → C180 conservative
regridder and allocate every per-window buffer. The source LL mesh is built
with the TARGET mesh radius so the two manifolds match (`build_regridder`
rejects a radius mismatch).
"""
function allocate_merra2_to_c180_pipeline(target_grid::CubedSphereTargetGeometry{FT};
                                          Nz::Integer,
                                          cache_dir::Union{Nothing, AbstractString} = nothing) where FT
    Nz_int = Int(Nz)
    Nz_int >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nc = target_grid.mesh.Nc

    source_ll_mesh = _merra2_source_latlon_mesh(FT, target_grid.mesh.radius)
    regridder = build_regridder(source_ll_mesh, target_grid.mesh;
                                normalize = false, cache_dir = cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    n_src == MERRA2_NX * MERRA2_NY ||
        throw(DimensionMismatch("regridder src_areas length $n_src ≠ MERRA-2 cells $(MERRA2_NX * MERRA2_NY)"))
    n_dst == ncells(target_grid.mesh) ||
        throw(DimensionMismatch("regridder dst_areas length $n_dst ≠ C180 cells $(ncells(target_grid.mesh))"))

    ws = allocate_cs_preprocess_workspace(Nc, MERRA2_NX, MERRA2_NY, Nz_int,
                                          n_src, n_dst, FT)
    c180_fields = (
        ps = ntuple(_ -> zeros(FT, Nc, Nc), 6),
        qv = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        u  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        v  = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
    )
    return MERRA2ToC180Pipeline{FT, typeof(regridder)}(regridder, ws, Nz_int, c180_fields)
end

"""
    process_merra2_window!(pipe, handles, win; FT) -> pipe

Read native MERRA-2 LL fields for window `win` (PS/QV from inst3 slice `win`,
U/V from tavg3 slice `win`) and conservatively regrid PS (2D intensive) and
QV/U/V (3D intensive) onto the C180 panels. U/V/QV are intensive → default
field type, as in the ERA5 path. NO level flip (top-down already).
"""
function process_merra2_window!(pipe::MERRA2ToC180Pipeline{FT},
                                handles::MERRA2DayHandles, win::Integer) where FT
    Nc = size(pipe.c180_fields.ps[1], 1)
    fields = read_merra2_window_fields(handles, win, pipe.Nz; FT = FT)
    regrid_2d_to_cs_panels!(pipe.c180_fields.ps, pipe.regridder, fields.ps,
                            pipe.ws, Nc, IntensiveCellField())
    regrid_3d_to_cs_panels!(pipe.c180_fields.qv, pipe.regridder, fields.qv, pipe.ws, Nc)
    regrid_3d_to_cs_panels!(pipe.c180_fields.u,  pipe.regridder, fields.u,  pipe.ws, Nc)
    regrid_3d_to_cs_panels!(pipe.c180_fields.v,  pipe.regridder, fields.v,  pipe.ws, Nc)
    return pipe
end

"""
    process_merra2_to_cs_day(date, settings, target_grid; out_path, …)

Generate a v4 cubed-sphere transport binary for one UTC `date` from MERRA-2
winds, written to `out_path`. The horizontal mass fluxes are reconstructed
from regridded MERRA-2 U/V and Poisson-balanced (the Cameron-Smith column
pressure-fix) against the inst3 dry-mass endpoints. `mass_basis` is fixed to
`:dry` (the runtime default). Stages to `out_path.tmp`, promotes on success.
"""
function process_merra2_to_cs_day(date::Date,
                                  settings::MERRA2Settings,
                                  target_grid::CubedSphereTargetGeometry{FT};
                                  out_path::AbstractString,
                                  Nz::Integer = MERRA2_NATIVE_LEVEL_COUNT,
                                  mass_basis::Symbol = :dry,
                                  dt_met_seconds::Real = 10800.0,
                                  steps_per_window::Integer = 1,
                                  adaptive_substeps::Bool = true,
                                  substep_cfl_target::Real = 0.95,
                                  max_steps_per_window::Integer = typemax(Int),
                                  cs_balance_tol::Real = 1e-14,
                                  cs_balance_project_every::Integer = 50,
                                  positivity_cfl_limit::Real = 0.95,
                                  require_substep_positivity::Bool = true,
                                  cache_dir::Union{Nothing, AbstractString} = nothing,
                                  global_mass_pin::Bool = false,
                                  global_mass_target_kg::Real = NaN) where FT
    mass_basis === :dry ||
        throw(ArgumentError("MERRA-2 → CS writer only supports mass_basis=:dry; got $(mass_basis)"))
    Nz_int = Int(Nz)
    steps_per_met = Int(steps_per_window)
    steps_per_met >= 1 || throw(ArgumentError("steps_per_window must be ≥ 1; got $(steps_per_met)"))

    substep_policy = SubstepSchedulePolicy(;
        adaptive_substeps = adaptive_substeps,
        substep_cfl_target = Float64(substep_cfl_target),
        min_steps_per_window = steps_per_met,
        max_steps_per_window = Int(max_steps_per_window))

    t_start = time()
    Nc = target_grid.Nc

    @info @sprintf("Process MERRA-2 → CS day: date=%s, Nc=%d, Nz=%d, FT=%s, winds=%s",
                   string(date), Nc, Nz_int, string(FT), String(settings.winds_collection))

    handles = open_merra2_day(settings, date; next_day_handle = true)
    try
        # --- Allocate two pipelines: current + next sliding-window. ---
        @info "  Allocating MERRA-2 → C180 pipelines (×2, sliding window)..."
        cur_pipe = allocate_merra2_to_c180_pipeline(target_grid; Nz = Nz_int, cache_dir = cache_dir)
        nxt_pipe = allocate_merra2_to_c180_pipeline(target_grid; Nz = Nz_int, cache_dir = cache_dir)

        vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))
        length(vc.A) == length(vc.B) == Nz_int + 1 ||
            throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz_int + 1); " *
                                    "check `settings.coefficients_file` vs the requested Nz"))
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
        out_dt_factor_for(steps) = FT(dt_met_seconds / (2 * steps))
        out_dt_factor = out_dt_factor_for(steps_per_met)

        # Build the MOIST DELP for flux reconstruction from the regridded PS.
        _fill_moist_dp! = function (pipe)
            @inbounds for p in 1:6
                for k in 1:Nz_int
                    dA = Float64(vc.A[k + 1]) - Float64(vc.A[k])
                    dB = Float64(vc.B[k + 1]) - Float64(vc.B[k])
                    for j in 1:Nc, i in 1:Nc
                        cur_dp_panels[p][i, j, k] =
                            FT(abs(dA + dB * Float64(pipe.c180_fields.ps[p][i, j])))
                    end
                end
            end
            return nothing
        end

        # Global dry-air mass pin (mirror of the GEOS-CS / N320 path). Removes
        # the residual global-mean dry-mass drift so the binary's absolute dry
        # mass matches a fixed target shared across driver families. `ps` is
        # recomputed from the pinned mass so the stored surface pressure stays
        # consistent. The fixed target is start-day/window independent (required
        # for a coherent multi-day archive under day-threading).
        do_mass_pin = global_mass_pin && isfinite(global_mass_target_kg)
        do_mass_pin && @info @sprintf("  Global dry-mass pin ON: target=%.9e kg (%.3f Pa dry ⟨ps⟩)",
                                      Float64(global_mass_target_kg),
                                      Float64(global_mass_target_kg) * Float64(gravity) /
                                      (6 * sum(Float64, cs_cell_areas)))
        pin_endpoint_mass! = function (m_dry, ps_dry)
            do_mass_pin || return nothing
            _pin_cs_global_air_mass!(m_dry, cs_cell_areas, gravity, global_mass_target_kg)
            for p in 1:6
                _ps_from_air_mass!(ps_dry[p], m_dry[p], cs_cell_areas, gravity, Nc, Nz_int)
            end
            return nothing
        end

        # --- Open the streaming writer. ---
        nwindow = 8
        steps_schedule = fill(steps_per_met, nwindow)
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
            include_tm5conv = false,
            include_surface = false,
            include_gchp_vdiff = false,
            mass_basis = mass_basis,
            panel_convention = _cs_panel_convention_tag(target_grid),
            cs_definition = _cs_definition_tag(target_grid),
            cs_coordinate_law = _cs_coordinate_law_tag(target_grid),
            cs_center_law = _cs_center_law_tag(target_grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(mesh)),
            extra_header = Dict{String, Any}(
                "preprocessor" => "process_merra2_to_cs_day",
                # Declare the per-window advection substep contract so the
                # runtime applies advection at the baked substep cadence but
                # runs convection + chemistry ONCE per met window (not per
                # substep). The substep schedule itself is baked per window.
                "runtime_substep_contract" => "binary_schedule",
                "preprocessor_contract" => "plan41_variable_substeps",
                "adaptive_substeps" => substep_policy.adaptive_substeps,
                "source_type"  => "merra2_native_latlon",
                "source_root"  => settings.root_dir,
                "target_type"  => "cubed_sphere",
                "regrid_method" => "conservative",
                "poisson_balanced" => true,
                # The Poisson column balance IS the Cameron-Smith pressure-fix:
                # it forces the column horizontal flux convergence to match the
                # analyzed (inst3) dry-mass tendency, exactly as
                # `pjc_pfix_mod.F90` does for the GEOS-Chem wind-derived path.
                "wind_flux_pressure_fix" => "cameron_smith_column_balance",
                "winds_collection" => String(settings.winds_collection),
                "global_mass_pin_enabled" => do_mass_pin,
                "global_mass_pin_target_kg" => do_mass_pin ?
                    Float64(global_mass_target_kg) : nothing,
            ))
        writer = CubedSphereBinaryWriter(inner_writer,
                                         mass_basis_from_symbol(mass_basis);
                                         Nc = Nc,
                                         npanel = 6,
                                         final_path = String(out_path))

        write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
        replay_tol = replay_tolerance(FT)

        # Drive read + regrid + derive dry mass + rotate + reconstruct fluxes
        # for one window into the provided panel buffers.
        function _process_window_to_cs!(win::Int,
                                        pipe::MERRA2ToC180Pipeline,
                                        m_dry, delp_dry, ps_dry, ps_dry_acc,
                                        am, bm)
            process_merra2_window!(pipe, handles, win)

            derive_c180_dry_mass!(m_dry, delp_dry, ps_dry, ps_dry_acc,
                                  pipe.c180_fields.ps, pipe.c180_fields.qv,
                                  vc, cs_cell_areas)
            pin_endpoint_mass!(m_dry, ps_dry)

            _fill_moist_dp!(pipe)
            rotate_winds_to_panel_local!(cur_u_local, cur_v_local,
                                         pipe.c180_fields.u, pipe.c180_fields.v,
                                         mesh, Nz_int)
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

        # Reconstruct + Poisson-balance + diagnose `cm` for the CURRENT window
        # at a given substep count. `pipe` rotation + Δp are substep-independent,
        # so they are prepared in `_adapt_window!` before the loop; only the flux
        # scaling, balance, mass tendency, and `cm` depend on `steps`.
        _balance_window_at_steps! = function (pipe, m_dry, m_next, am, bm, cm, dm, steps)
            reconstruct_cs_fluxes!(am, bm, cur_u_local, cur_v_local,
                                   cur_dp_panels, pipe.c180_fields.ps,
                                   vc.A, vc.B, Δx, Δy,
                                   gravity, out_dt_factor_for(steps), Nc, Nz_int)
            bal_diag = if apply_horizontal_balance
                balance_cs_global_mass_fluxes!(
                    am, bm, m_dry, m_next,
                    target_grid.face_table, target_grid.cell_degree, steps,
                    target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                    max_iter = 20000, project_every = Int(cs_balance_project_every))
            else
                balance_cs_column_mass_fluxes!(
                    am, bm, m_dry, m_next,
                    target_grid.face_table, target_grid.cell_degree, steps,
                    target_grid.poisson_scratch; tol = Float64(cs_balance_tol),
                    max_iter = 20000, project_every = Int(cs_balance_project_every))
            end
            sync_all_cs_boundary_mirrors!(am, bm, mesh.connectivity, Nc, Nz_int)
            fill_cs_window_mass_tendency!(dm, m_dry, m_next, steps)
            for p in 1:6; fill!(cm[p], zero(FT)); end
            diagnose_cs_cm!(cm, am, bm, dm, m_dry, Nc, Nz_int)
            return bal_diag
        end

        # Rotate `pipe` winds + build Δp (substep-independent), then adaptively
        # raise the substep count until the per-substep CFL drops under the
        # target (or the schedule converges). Returns the chosen `steps` and the
        # final balance diagnostics.
        _adapt_window! = function (pipe, m_dry, m_next, am, bm, cm, dm)
            rotate_winds_to_panel_local!(cur_u_local, cur_v_local,
                                         pipe.c180_fields.u, pipe.c180_fields.v,
                                         mesh, Nz_int)
            _fill_moist_dp!(pipe)
            steps = steps_per_met
            bal_diag = _balance_window_at_steps!(pipe, m_dry, m_next, am, bm, cm, dm, steps)
            if substep_policy.adaptive_substeps
                for _ in 1:_MERRA2_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
                    pos = verify_substep_positivity_cs!(m_dry, am, bm, cm;
                                                        cfl_limit = substep_cfl_target,
                                                        m_next = m_next)
                    next = next_substeps(substep_policy, steps, pos.ratio)
                    next == steps && break
                    steps = next
                    bal_diag = _balance_window_at_steps!(pipe, m_dry, m_next, am, bm, cm, dm, steps)
                end
            end
            return steps, bal_diag
        end

        # --- Window 1. ---
        t0 = time()
        _process_window_to_cs!(1, cur_pipe,
                               cur_m_dry, cur_delp_dry, cur_ps_dry, cur_ps_dry_acc,
                               cur_am, cur_bm)
        @info @sprintf("    Window  1/%d: read+regrid+rotate+flux %.2fs",
                       nwindow, time() - t0)

        # --- Windows 2..8: read next, balance current, diagnose cm, write. ---
        for win in 2:nwindow
            t0 = time()
            _process_window_to_cs!(win, nxt_pipe,
                                   nxt_m_dry, nxt_delp_dry, nxt_ps_dry, nxt_ps_dry_acc,
                                   cur_am, cur_bm)   # cur_am/bm overwritten next round
            t_read = time() - t0

            t_bal = time()
            win_steps, bal_diag = _adapt_window!(cur_pipe, cur_m_dry, nxt_m_dry,
                                                 cur_am, cur_bm, cur_cm, cur_dm_dry)
            t_bal = time() - t_bal
            steps_schedule[win - 1] = win_steps

            worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
            worst_post = max(worst_post, bal_diag.max_post_residual)
            worst_iter = max(worst_iter, bal_diag.max_cg_iter)

            pos_diag = if write_replay_on
                contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                      nxt_m_dry,
                                                      win_steps, win - 1;
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

            payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                       ps = cur_ps_dry, dm = cur_dm_dry)
            write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(win - 1, payload))

            @info @sprintf("    Window %2d/%d: wrote (steps=%d bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                           win - 1, nwindow, win_steps, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter,
                           win, t_read)

            # Swap current ↔ next so the regrid state stays paired with the
            # dry mass we cached.
            cur_pipe, nxt_pipe = nxt_pipe, cur_pipe
            cur_m_dry, nxt_m_dry           = nxt_m_dry, cur_m_dry
            cur_delp_dry, nxt_delp_dry     = nxt_delp_dry, cur_delp_dry
            cur_ps_dry, nxt_ps_dry         = nxt_ps_dry, cur_ps_dry
            cur_ps_dry_acc, nxt_ps_dry_acc = nxt_ps_dry_acc, cur_ps_dry_acc
        end

        # --- Final window: next-day inst3 slice-1 endpoint look-ahead. ---
        # The closed contract for window h is m(h+1) - m(h); the final window's
        # m_next must be the next day's first endpoint, not a zero-tendency copy.
        next_ep = read_merra2_next_day_endpoint(handles, Nz_int; FT = FT)
        if next_ep !== nothing
            # Regrid next-day hour-0 PS/QV onto the (reused) nxt_pipe panels,
            # then re-derive dry mass on C180. nxt_pipe's winds are stale but
            # the final window's fluxes come from cur_pipe winds, so only PS/QV
            # matter here.
            regrid_2d_to_cs_panels!(nxt_pipe.c180_fields.ps, nxt_pipe.regridder,
                                    next_ep.ps, nxt_pipe.ws, Nc, IntensiveCellField())
            regrid_3d_to_cs_panels!(nxt_pipe.c180_fields.qv, nxt_pipe.regridder,
                                    next_ep.qv, nxt_pipe.ws, Nc)
            derive_c180_dry_mass!(nxt_m_dry, nxt_delp_dry, nxt_ps_dry, nxt_ps_dry_acc,
                                  nxt_pipe.c180_fields.ps, nxt_pipe.c180_fields.qv,
                                  vc, cs_cell_areas)
            pin_endpoint_mass!(nxt_m_dry, nxt_ps_dry)
        else
            # HACK: zero-tendency fallback for the final day of the archive (no
            # next-day inst3 file on disk). The positivity gate will likely
            # warn/fail — that's correct: a zero-tendency closure for the last
            # window is a known artifact, not a passing binary. Run with
            # [numerics].require_substep_positivity = false only when you
            # understand and accept this for boundary days.
            @warn "process_merra2_to_cs_day: archive-boundary fallback — " *
                  "no next-day inst3 file for $(date + Day(1)), using " *
                  "zero-tendency m_next. Final window's continuity will not close."
            for p in 1:6
                copyto!(nxt_m_dry[p], cur_m_dry[p])
            end
        end

        final_steps, bal_diag = _adapt_window!(cur_pipe, cur_m_dry, nxt_m_dry,
                                               cur_am, cur_bm, cur_cm, cur_dm_dry)
        steps_schedule[nwindow] = final_steps
        final_pos_diag = if write_replay_on
            contract = verify_cs_window_contract!(cur_m_dry, cur_am, cur_bm, cur_cm,
                                                  nxt_m_dry,
                                                  final_steps, nwindow;
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

        final_payload = (m = cur_m_dry, am = cur_am, bm = cur_bm, cm = cur_cm,
                         ps = cur_ps_dry, dm = cur_dm_dry)
        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(nwindow, final_payload))

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        # Stamp the adaptive per-window substep schedule onto the header.
        set_streaming_steps_per_window_schedule!(writer.inner, steps_schedule)

        # Summarize positivity BEFORE promoting the .tmp so a failed-gate day
        # quarantines the staged file.
        summarize_cs_positivity_status(worst_positivity;
                                       cfl_limit = positivity_cfl_limit,
                                       require_substep_positivity = require_substep_positivity,
                                       steps_per_window = maximum(steps_schedule),
                                       quarantine_path = writer_staging_path(writer))
        promote_streaming_binary!(writer)

        elapsed = time() - t_start
        @info @sprintf("MERRA-2 → C180 day complete: %.1fs (%.2fs/window). substeps=[%d..%d]. Worst bal pre=%.2e post=%.2e iter=%d.",
                       elapsed, elapsed / nwindow,
                       minimum(steps_schedule), maximum(steps_schedule),
                       worst_pre, worst_post, worst_iter)
        worst_replay_win > 0 &&
            @info @sprintf("  Worst replay: rel=%.2e abs=%.2e at win=%d",
                           worst_replay_rel, worst_replay_abs, worst_replay_win)
        return nothing
    finally
        close_merra2_day!(handles)
    end
end

# ===========================================================================
# Unified-CLI dispatch — wires the per-day driver into
# `preprocess_transport_binary.jl` via the standard
# `process_day(date, grid, settings, vertical; ...)` extension point.
# ===========================================================================

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings::MERRA2Settings,
                vertical; out_path, mass_basis, dt_met_seconds, …)

Adapter that the unified preprocessor CLI calls into. Forwards to
[`process_merra2_to_cs_day`](@ref) with the kwargs the underlying function
accepts; the rest of the unified-CLI day-kwargs (e.g. `chain_mass`,
`seed_m`, `balance_mode`, `cm_closure`) are absorbed by the trailing
`kwargs...` and ignored — MERRA-2 has no day-to-day mass-chain state and the
flux balance is the fixed Cameron-Smith column pressure-fix.

Returns `(; final_m = nothing, global_mass_target_kg)` so the unified CLI's
`seed_m`/`global_mass_target_kg` chain remains a no-op.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::MERRA2Settings,
                     vertical;
                     out_path::AbstractString,
                     mass_basis::Symbol = :dry,
                     dt_met_seconds::Real = 10800.0,
                     positivity_cfl_limit::Real = 0.95,
                     min_steps_per_window::Union{Integer, Nothing} = nothing,
                     adaptive_substeps::Bool = true,
                     substep_cfl_target::Real = 0.95,
                     max_steps_per_window::Integer = typemax(Int),
                     require_substep_positivity::Bool = true,
                     global_mass_pin::Bool = false,
                     global_mass_target_kg::Real = NaN,
                     kwargs...)
    steps_floor = min_steps_per_window === nothing ? 1 : Int(min_steps_per_window)
    process_merra2_to_cs_day(date, settings, grid;
        out_path              = out_path,
        Nz                    = vertical.Nz,
        mass_basis            = mass_basis,
        dt_met_seconds        = dt_met_seconds,
        steps_per_window      = steps_floor,
        adaptive_substeps     = adaptive_substeps,
        substep_cfl_target    = substep_cfl_target,
        max_steps_per_window  = max_steps_per_window,
        positivity_cfl_limit  = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        cache_dir             = grid.cache_dir,
        global_mass_pin       = global_mass_pin,
        global_mass_target_kg = global_mass_target_kg)
    return (; final_m = nothing,
            global_mass_target_kg = global_mass_target_kg)
end

# Source/target support matrix entry for the TOML entrypoint.
preprocessor_pair_supported(::CubedSphereTargetGeometry, ::MERRA2Settings) = true

# Output filename: clear MERRA-2 prefix, matching the other native writers.
_native_output_filename(::MERRA2Settings, date::Date, FT::Type) =
    "merra2_transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"
