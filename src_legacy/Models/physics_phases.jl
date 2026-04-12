# ---------------------------------------------------------------------------
# Physics phase functions — grid-dispatched wrappers for the unified run loop
#
# Each phase dispatches on grid type (LatitudeLongitudeGrid vs CubedSphereGrid).
# The unified run loop calls these in sequence without any if/else on grid type.
# ---------------------------------------------------------------------------

# =====================================================================
# Sub-step diagnostics for moist advection debugging
# =====================================================================

"""
Diagnostic container for capturing intermediate moist advection state.
Set `MOIST_DIAG[] = MoistSubStepDiag(...)` before `run!` to enable.
"""
mutable struct MoistSubStepDiag{FT}
    q_wet_post_hadv::Vector{Array{FT,3}}   # after gchp_tracer_2d!, substep 1
    rm_post_vremap::Vector{Array{FT,3}}    # after vertical_remap_cs!, substep 1
    qv_start::Vector{Array{FT,3}}          # QV at window start
    qv_back::Vector{Array{FT,3}}           # QV used for back-conversion
    delp_start::Vector{Array{FT,3}}        # moist DELP at start
    delp_end::Vector{Array{FT,3}}          # moist DELP at end
    q_dry_init::Vector{Array{FT,3}}        # dry MR before dry→wet conversion
    q_dry_final::Vector{Array{FT,3}}       # dry MR after wet→dry conversion
    captured::Bool                          # true once first window captured
end

function MoistSubStepDiag(FT::Type, Nc::Int, Nz::Int, Hp::Int)
    N = Nc + 2Hp
    alloc() = [zeros(FT, N, N, Nz) for _ in 1:6]
    MoistSubStepDiag{FT}(alloc(), alloc(), alloc(), alloc(),
                          alloc(), alloc(), alloc(), alloc(), false)
end

"""Global ref for moist sub-step diagnostics. Set to a `MoistSubStepDiag` to enable capture."""
const MOIST_DIAG = Ref{Any}(nothing)

_ll_mass_basis(::Nothing) = Val(:moist)
_ll_mass_basis(driver) = _IO.mass_basis(driver) === :dry ? Val(:dry) : Val(:moist)

_compute_ll_dry_mass!(dest, src, qv, ::Val{:dry}) = copyto!(dest, src)
_compute_ll_dry_mass!(dest, src, qv, ::Val{:moist}) = (dest .= src .* (1 .- qv))

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
        qv_cpu=phys.qv_cpu, ps_panels=phys.ps_cpu,
        qv_next_cpu=phys.qv_next_cpu, ps_next_panels=nothing)
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
# Dry mass helpers for LL rm↔c_dry conversions
#
# c_dry = rm / m_dry, where m_dry = m_moist × (1 - QV).
# Computed once per window; used by IC, emissions, diffusion, convection, output.
# =====================================================================

    # Mass-CFL pilot is now in src/Advection/mass_cfl_pilot.jl
    # Called from advection_phase! via Advection.find_mass_cfl_refinement()

"""Compute LL dry air mass for rm↔VMR conversions.

For moist-basis transport, `m_dry = m × (1 - QV)`.
For dry-basis binaries, `m` is already dry and is copied directly.
Skips QV correction if vertical levels don't match (e.g. merged-grid output)."""
compute_ll_dry_mass!(phys, sched, grid::CubedSphereGrid, driver=nothing) = nothing
function compute_ll_dry_mass!(phys, sched, grid::LatitudeLongitudeGrid, driver=nothing)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_ref)
        _compute_ll_dry_mass!(phys.m_dry, gpu.m_ref, phys.qv_gpu, _ll_mass_basis(driver))
    else
        copyto!(phys.m_dry, gpu.m_ref)
    end
end

"""Return dry mass for LL rm↔c_dry conversions (pre-computed by `compute_ll_dry_mass!`)."""
ll_dry_mass(phys) = phys.m_dry


"""Recompute m_dry from evolved m_dev (post-advection) for LL output.
TM5 always uses m_evolved for mixing ratios — never m_prescribed.
The Strang-split advection evolves m_dev away from m_ref; using m_ref
for output creates noise in c = rm/m because rm was evolved with m_dev."""
compute_ll_dry_mass_evolved!(phys, sched, grid::CubedSphereGrid, driver=nothing) = nothing
function compute_ll_dry_mass_evolved!(phys, sched, grid::LatitudeLongitudeGrid, driver=nothing)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_dev)
        _compute_ll_dry_mass!(phys.m_dry, gpu.m_dev, phys.qv_gpu, _ll_mass_basis(driver))
    else
        copyto!(phys.m_dry, gpu.m_dev)
    end
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
    # TM5 convection fields (entu, detu, entd, detd)
    if phys.has_tm5conv
        status = load_tm5conv_window!(phys.tm5conv_cpu, driver, grid, w)
        phys.tm5conv_loaded[] = status !== false
        if phys.tm5conv_loaded[]
            for name in (:entu, :detu, :entd, :detd)
                copyto!(phys.tm5conv_gpu[name], phys.tm5conv_cpu[name])
            end
        end
    end

    # CMFMC (for Tiedtke/RAS, not used with TM5 matrix convection)
    if phys.has_conv && !phys.has_tm5conv
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

    # QV (specific humidity for dry-air output conversion)
    qv_status = load_qv_window!(phys.qv_cpu, driver, grid, w)
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        copyto!(phys.qv_gpu, phys.qv_cpu)
    end
end

function load_and_upload_physics!(phys, sched, driver,
                                   grid::CubedSphereGrid, w; arch=nothing)
    # Wait for async physics load (split from met load for IO overlap)
    wait_phys!(sched)
    io = sched.io_result
    io === nothing && return

    # CMFMC
    if phys.has_conv
        cmfmc_status = io.cmfmc
        phys.cmfmc_loaded[] = cmfmc_status !== false
        if phys.cmfmc_loaded[] && cmfmc_status !== :cached
            for_panels_nosync() do p
                copyto!(phys.cmfmc_gpu[p], phys.cmfmc_cpu[p])
            end
        end
    end

    # DTRAIN
    if phys.has_dtrain
        dtrain_status = io.dtrain
        phys.dtrain_loaded[] = dtrain_status !== false
        if phys.dtrain_loaded[] && dtrain_status !== :cached
            for_panels_nosync() do p
                copyto!(phys.dtrain_gpu[p], phys.dtrain_cpu[p])
            end
        end
    end

    # Invalidate RAS CFL cache on fresh data
    if (phys.cmfmc_loaded[] && io.cmfmc !== :cached) ||
       (phys.dtrain_loaded[] && io.dtrain !== :cached)
        invalidate_ras_cfl_cache!()
    end

    # QV (current window = SPHU1 in GCHP terminology)
    qv_status = io.qv
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        for_panels_nosync() do p
            copyto!(phys.qv_gpu[p], phys.qv_cpu[p])
        end
        fill_panel_halos!(phys.qv_gpu, grid)
    end

    # QV next window (= SPHU2 in GCHP terminology)
    # Used for target PE in dry-basis remap: dp_dry_target = DELP_next × (1-QV_next)
    if hasproperty(io, :qv_next_from_v4) && io.qv_next_from_v4
        # v4 binary: QV_end already loaded atomically into qv_next_cpu by load_all_window!
        phys.qv_next_loaded[] = true
        for_panels_nosync() do p
            copyto!(phys.qv_next_gpu[p], phys.qv_next_cpu[p])
        end
        fill_panel_halos!(phys.qv_next_gpu, grid)
    else
        # Fallback: load from separate CTM_I1/I3 file
        n_win = total_windows(driver)
        if w < n_win
            qv_next_status = load_qv_window!(phys.qv_next_cpu, driver, grid, w + 1)
            phys.qv_next_loaded[] = qv_next_status !== false
            if phys.qv_next_loaded[] && qv_next_status !== :cached
                for_panels_nosync() do p
                    copyto!(phys.qv_next_gpu[p], phys.qv_next_cpu[p])
                end
                fill_panel_halos!(phys.qv_next_gpu, grid)
            end
        else
            phys.qv_next_loaded[] = false
        end
    end

    # Surface fields
    sfc_status = io.sfc
    phys.sfc_loaded[] = sfc_status !== false
    phys.troph_loaded[] = phys.sfc_loaded[]
    if phys.sfc_loaded[] && phys.has_pbl
        for_panels_nosync() do p
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
                                    driver, half_dt; use_gchp::Bool=false) where FT
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
    # Preprocessed binary: am/bm/cm are per spectral half_dt and used directly.
    # NO runtime scaling — the preprocessor stores fluxes per half-step (450s),
    # and n_sub Strang cycles accumulate the correct total:
    #   n_sub × 2 × am = steps_per_met × 2 × half_dt = window_dt
    # This matches the CS convention (no /n_sub, no scaling) and the old LL code.

    # === Pole-row am cleanup (NOT a TM5 deviation) ===
    #
    # TM5 convention: j=1, Ny are real cells at ±89.75°.  TM5's dynam0
    # (advect_tools.F90:767-773) computes am = dtu*pu at j=1..jmr, where
    # pu is a gridded mass flux.  At the geographic pole, pu is physically
    # zero (zonal velocity is undefined there) so TM5 gets am≈0 naturally.
    #
    # Our spectral preprocessor (preprocess_spectral_v4_binary.jl:748)
    # computes am = u_stag/cos(lat) * dp * R * dlat * half_dt.
    # At j=1 and j=Ny=361, cos(lat) ≈ 0.00436 (cell center at ±89.75°).
    # Any residual u from the spectral interpolation (~1e-10) gets
    # multiplied by 1/cos_lat → huge garbage values (observed up to 4e17).
    # These are NOT physical mass flux; they are numerical noise.
    #
    # Zeroing am at the pole rows restores TM5-equivalent behavior by
    # setting am to its physical value of ~0.  This is a PREPROCESSOR
    # ARTIFACT CLEANUP, not a TM5 algorithmic deviation.
    #
    # bm at pole FACES (j=1 = south pole face, j=Ny+1 = north pole face)
    # is already zero from the preprocessor (no meridional flux through pole).
    Ny = grid.Ny
    @views gpu.am[:, 1, :]  .= zero(FT)
    @views gpu.am[:, Ny, :] .= zero(FT)
end

function process_met_after_upload!(sched, phys, grid::CubedSphereGrid{FT},
                                    driver, half_dt; use_gchp::Bool=false) where FT
    gpu = current_gpu(sched)
    if use_gchp
        # GCHP path: leave AM/BM/CX/CY/XFX/YFX UNSCALED.
        # AM/BM are in kg/s; conversion to Pa·m² happens in gchp_offline_advection_phase!.
        # CX/CY/XFX/YFX stay at full accumulated values; subcycling divides them.
        return
    end
    for_panels_nosync() do p
        gpu.am[p] .*= half_dt
        gpu.bm[p] .*= half_dt
    end
    if gpu.cx !== nothing
        for_panels_nosync() do p
            gpu.cx[p]  .*= FT(0.5)
            gpu.cy[p]  .*= FT(0.5)
        end
    end
    if gpu.xfx !== nothing
        for_panels_nosync() do p
            gpu.xfx[p] .*= FT(0.5)
            gpu.yfx[p] .*= FT(0.5)
        end
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
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], gpu.delp[p], phys.qv_gpu[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], gpu.delp[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    end
    # Always compute MOIST air mass for convection (m_wet = delp × area / g).
    # Convective mass fluxes (CMFMC, DTRAIN) are on MOIST basis, so rm↔q
    # conversion must use moist air mass for consistency (GCHP convention).
    for_panels_nosync() do p
        compute_air_mass_panel!(air.m_wet[p], gpu.delp[p],
                                gc.area[p], gc.gravity, Nc, Nz, Hp)
    end
end

# =====================================================================
# Phase 4: IC finalization + reference mass save
# =====================================================================

"""Finalize deferred initial conditions (first window only).
LL: after vertical interp sets VMR c, convert to tracer mass rm = c × m_ref.
TM5-faithful: transport on moist basis, dry correction at output only."""
function finalize_ic_phase!(tracers, sched, air, phys, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if has_deferred_ic_vinterp()
        finalize_ic_vertical_interp!(tracers, gpu.m_ref, grid)
    end
    # Convert VMR → tracer mass: rm = c × m_ref (moist basis, like TM5)
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
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
    for_panels_nosync() do p
        copyto!(air.m_ref[p], air.m[p])
    end
end

# =====================================================================
# Phase 5: Compute vertical mass flux (cm)
# =====================================================================

"""Compute cm. LL: already in met buffer. CS: from continuity of am/bm."""
compute_cm_phase!(sched, air, phys, grid::LatitudeLongitudeGrid, gc;
                   has_next::Bool=false, n_sub::Int=1) = nothing

function compute_cm_phase!(sched, air, phys, grid::CubedSphereGrid, gc;
                            has_next::Bool=false, n_sub::Int=1)
    gpu = current_gpu(sched)
    Nc, Nz = grid.Nc, grid.Nz

    # Pure mass-flux closure: cm from continuity of dry am/bm.
    # Dry air mass is conserved (no sources/sinks), so the dry horizontal
    # flux divergence exactly determines the dry vertical flux.
    for_panels_nosync() do p
        compute_cm_panel!(gpu.cm[p], gpu.am[p], gpu.bm[p], gc.bt, Nc, Nz)
    end
end

"""Wait for next-window DELP and upload (CS DoubleBuffer pressure fixer)."""
wait_and_upload_next_delp!(::IOScheduler{SingleBuffer}, ::Any) = nothing
wait_and_upload_next_delp!(::IOScheduler{SingleBuffer}, ::LatitudeLongitudeGrid) = nothing
wait_and_upload_next_delp!(::IOScheduler, ::LatitudeLongitudeGrid) = nothing

function wait_and_upload_next_delp!(sched::IOScheduler{DoubleBuffer},
                                     grid::CubedSphereGrid)
    # Wait for met-only task (fast: DELP + fluxes). Physics task runs independently.
    if sched.load_task !== nothing
        fetch(sched.load_task)
        sched.load_task = nothing
    end
    nc = next_cpu(sched)
    ng = next_gpu(sched)
    for_panels_nosync() do p
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

"""Apply emissions, dispatched on grid type.
LL tracers are rm — convert to VMR for emission kernels, then back (same m_ref → exact).
TM5-faithful: moist basis for boundary conversions, dry correction at output only."""
function apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                 grid::LatitudeLongitudeGrid, dt_window;
                                 sim_hours::Float64=0.0, arch=nothing)
    emi_data, area_j_dev, A_coeff, B_coeff = emi_state
    gpu = current_gpu(sched)
    # rm → c for emission kernels (moist basis, like TM5)
    for (_, rm) in pairs(tracers)
        rm ./= gpu.m_ref
    end
    _apply_emissions_latlon!(tracers, emi_data, area_j_dev,
                              gpu.ps, A_coeff, B_coeff, grid, dt_window;
                              sim_hours, arch,
                              delp=(phys.sfc_loaded[] ? gpu.Δp : nothing),
                              pblh=(phys.sfc_loaded[] && phys.pbl_sfc_gpu !== nothing ?
                                    phys.pbl_sfc_gpu.pblh : nothing))
    # c → rm
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
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

Recompute cm from horizontal divergence of am/bm on CPU.
Simple top-down accumulation per column. Called once per substep when interpolating (v4).
cm[1] = 0 (TOA), cm[Nz+1] = residual → enforced to zero at boundaries.
"""
function _compute_cm_from_divergence_gpu!(cm_gpu, am_gpu, bm_gpu, m_gpu, grid)
    cm = Array(cm_gpu)
    am = Array(am_gpu)
    bm = Array(bm_gpu)
    FT = eltype(cm)
    Nx, Ny = grid.Nx, grid.Ny
    Nz = size(cm, 3) - 1
    fill!(cm, zero(FT))

    # Use B-coefficient correction (TM5 dynam0 formula) if available
    B_ifc = hasproperty(grid.vertical, :B) ? Float64.(grid.vertical.B) : Float64[]

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        @inbounds for j in 1:Ny, i in 1:Nx
            pit = 0.0
            for k in 1:Nz
                pit += (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                       (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
            end
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h + (B_ifc[k+1] - B_ifc[k]) * pit
                cm[i, j, k+1] = FT(acc)
            end
        end
    else
        @inbounds for j in 1:Ny, i in 1:Nx
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc -= div_h
                cm[i, j, k+1] = FT(acc)
            end
        end
    end
    @views cm[:, :, 1]   .= zero(FT)
    @views cm[:, :, end] .= zero(FT)
    copyto!(cm_gpu, cm)
end

"""Enforce cm[:,:,1]=0 and cm[:,:,end]=0 on GPU arrays (TM5 advectz.F90:320)."""
function _enforce_cm_boundaries_gpu!(cm)
    FT = eltype(cm)
    @views cm[:, :, 1]   .= zero(FT)
    @views cm[:, :, end] .= zero(FT)
end

"""
    recompute_cm_runtime!(cm_gpu, am_gpu, bm_gpu, dB_gpu)

Recompute cm on GPU from horizontal divergence of am/bm + Segers B-correction.
TM5's dynam0 formula (advect.F90:215-282, r1112):

    div_h(k) = am[i+1,j,k] - am[i,j,k] + bm[i,j+1,k] - bm[i,j,k]
    pit = Σ div_h(k)  over all k
    cm[k+1] = cm[k] - div_h(k) + dB[k] × pit

Uses Neumaier compensated summation for pit and cm accumulation to maintain
accuracy in Float32 (equivalent to Float64 accumulation on CPU).
`dB_gpu` is a device vector of length Nz with dB[k] = B_ifc[k+1] - B_ifc[k].

Verified: matches preprocessor's cm to F32 precision (relative error 5e-8).
"""
function recompute_cm_runtime!(cm_gpu, am_gpu, bm_gpu, dB_gpu)
    backend = get_backend(cm_gpu)
    Nx = size(am_gpu, 1) - 1
    Ny = size(bm_gpu, 2) - 1
    k! = _dynam0_kernel!(backend, 256)
    k!(cm_gpu, am_gpu, bm_gpu, dB_gpu, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
    return nothing
end

@kernel function _dynam0_kernel!(cm, @Const(am), @Const(bm), @Const(dB), Nx, Ny)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)
    Nz = size(am, 3)
    @inbounds begin
        # Column-integrated horizontal divergence (Neumaier sum for F32 accuracy)
        pit = zero(FT)
        pit_c = zero(FT)  # Neumaier compensation
        for k in 1:Nz
            div_h = (am[i+1, j, k] - am[i, j, k]) +
                    (bm[i, j+1, k] - bm[i, j, k])
            t = pit + div_h
            if abs(pit) >= abs(div_h)
                pit_c += (pit - t) + div_h
            else
                pit_c += (div_h - t) + pit
            end
            pit = t
        end
        pit += pit_c  # corrected sum

        # Build cm from TOA downward (Neumaier accumulation)
        cm[i, j, 1] = zero(FT)
        acc = zero(FT)
        acc_c = zero(FT)
        for k in 1:Nz
            div_h = (am[i+1, j, k] - am[i, j, k]) +
                    (bm[i, j+1, k] - bm[i, j, k])
            term = -div_h + dB[k] * pit
            t = acc + term
            if abs(acc) >= abs(term)
                acc_c += (acc - t) + term
            else
                acc_c += (term - t) + acc
            end
            acc = t
            cm[i, j, k + 1] = acc + acc_c
        end
        cm[i, j, Nz + 1] = zero(FT)  # surface boundary
    end
end

"""
Clamp cm so |cm[k]| / m[k] <= cfl_limit and |cm[k]| / m[k-1] <= cfl_limit at every cell.
ERA5 spectral data has ~0.02% of cells with extreme Z-CFL from deep convection.
Operates on CPU buffers before GPU upload (called once per window).
"""
function _clamp_cm_cfl!(cm_gpu, m_gpu, cfl_limit)
    cm = Array(cm_gpu)
    m = Array(m_gpu)
    Nx, Ny, Nz = size(m)
    FT = eltype(cm)
    n_clamped = 0
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        # cm[i,j,k] is flux at interface between level k-1 (above) and k (below)
        m_above = m[i, j, k-1]
        m_below = m[i, j, k]
        cm_val = cm[i, j, k]
        if cm_val > zero(FT)
            # Downward into k: donor is level k-1 (above)
            max_flux = cfl_limit * m_above
        else
            # Upward from k: donor is level k (below)
            max_flux = cfl_limit * m_below
        end
        if abs(cm_val) > max_flux
            cm[i, j, k] = sign(cm_val) * max_flux
            n_clamped += 1
        end
    end
    if n_clamped > 0
        copyto!(cm_gpu, cm)
        @info "Clamped $n_clamped cm faces ($(round(n_clamped/(Nx*Ny*Nz)*100, digits=4))%) to CFL<$cfl_limit" maxlog=5
    end
end

"""
Advection + convection sub-stepping. LL: NamedTuple dispatch (all tracers together).
CS: per-tracer advection with mass fixer + m_ref advance along pressure trajectory.
"""
function advection_phase!(tracers, sched, air, phys, model,
                           grid::LatitudeLongitudeGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           geom_gchp=nothing, ws_gchp=nothing,
                           has_next::Bool=false,
                           dB_gpu=nothing) where FT
    gpu = current_gpu(sched)
    # Build advection workspace — for Prather, augment with per-tracer slope storage
    adv_ws = _build_advection_workspace(gpu.ws, model.advection_scheme, tracers, gpu.m_ref)

    # Per-substep flux interpolation (v4) + runtime cm recomputation (dynam0).
    has_deltas = gpu.dam !== nothing
    _use_dynam0 = dB_gpu !== nothing
    _use_mass_fixer = get(model.metadata, "mass_fixer", true)
    Ny = grid.Ny

    if _use_dynam0
        recompute_cm_runtime!(gpu.cm, gpu.am, gpu.bm, dB_gpu)
    end
    # NO cm clamping.  The raw cm from the preprocessor satisfies cell-level
    # mass continuity (m + cm[k] - cm[k+1] > 0 for every cell).  The previous
    # per-face clamp was unphysical: it independently capped cm[k] and cm[k+1]
    # at the per-face CFL=0.95, which creates two-sided drain at polar cells
    # (e.g. cm[k]=-0.95*m AND cm[k+1]=+0.95*m → m_new=-0.9*m).
    # The evolving-mass pilot in advect_z_massflux_subcycled! handles per-pass
    # CFL by subcycling.  Verified: raw cm has max Z-CFL ≈ 9.1, all cells
    # maintain m_new > 0 at every timestep.

    copyto!(gpu.m_dev, gpu.m_ref)

    # === TM5 global Check_CFL pre-pass (advectm_cfl.F90:217-336) ===
    # Tests the full Strang sequence on a pilot mass; if any cell would go
    # negative or any face exceeds CFL, halves am/bm/cm globally and doubles
    # the substep count.  Complements the per-(j,l)/per-level local nloop
    # refinement inside X/Y by handling cumulative drainage.
    #
    # 2026-04-07: also runs for the has_deltas (v4) path. After halving the
    # base am/bm/cm in place, we ALSO halve dam/dbm/dcm so the substep
    # interpolation `am(t) = am0 + t·dam` still gives per-substep fluxes
    # at 1/n_extra of original. dm is NOT halved (whole-window mass change
    # is invariant under refinement). Total transport over the window is
    # preserved.
    n_sub_eff = n_sub
    if adv_ws isa Advection.MassFluxWorkspace
        # When mass_fixer is on, the pilot must reset m at every substep to
        # mimic the runtime fixer (otherwise the pilot accumulates drainage
        # that the actual run wouldn't see).
        n_extra = Advection.check_global_cfl_and_scale!(
            gpu.am, gpu.bm, gpu.cm, gpu.m_dev, grid, adv_ws;
            n_sub=n_sub, cfl_limit=FT(1.0),
            reset_per_substep=_use_mass_fixer)
        if has_deltas && n_extra > 1
            inv_n = FT(1) / FT(n_extra)
            gpu.dam .*= inv_n
            gpu.dbm .*= inv_n
            gpu.dcm !== nothing && (gpu.dcm .*= inv_n)
            # NOT dm — it's the whole-window mass change, sampled at
            # t_end = s/n_sub_eff in the mass_fixer. Halving dm would
            # under-prescribe the mass trajectory.
        end
        n_sub_eff = n_sub * n_extra
    end

    # IMPORTANT: am0/bm0 must be captured AFTER the pre-pass, so they
    # reflect the (possibly halved) base values. Otherwise the substep
    # interpolation would re-apply the original (unhalved) am0.
    am0 = has_deltas ? copy(gpu.am) : gpu.am
    bm0 = has_deltas ? copy(gpu.bm) : gpu.bm

    for s in 1:n_sub_eff
        step[] += 1

        if has_deltas
            # Interpolate am/bm to substep midpoint (TM5 TimeInterpolation)
            t = FT(s - FT(0.5)) / FT(n_sub_eff)
            gpu.am .= am0 .+ t .* gpu.dam
            gpu.bm .= bm0 .+ t .* gpu.dbm
            # Re-apply pole zeroing
            @views gpu.am[:, 1, :]    .= zero(FT)
            @views gpu.am[:, Ny, :]   .= zero(FT)
            @views gpu.bm[:, 1, :]    .= zero(FT)
            @views gpu.bm[:, Ny+1, :] .= zero(FT)
        end

        if _use_dynam0
            # Runtime dynam0: recompute cm from current am/bm + B-correction
            # (TM5 advect.F90:215-282, r1112). Neumaier GPU accumulation.
            recompute_cm_runtime!(gpu.cm, gpu.am, gpu.bm, dB_gpu)
            # No cm clamping — see comment above the substep loop.
        end

        _apply_advection_latlon!(tracers, gpu.m_dev,
                                  gpu.am, gpu.bm, gpu.cm,
                                  grid, model.advection_scheme, adv_ws;
                                  cfl_limit=FT(1.0))  # TM5 advecty/x__slopes.F90 default

        # Per-substep mass correction.
        # mass_fixer=true (default): prescribe m_dev along the met-data mass
        #   trajectory and scale rm to preserve VMR. Needed for F32 stability
        #   but amplifies gradients at fronts.
        # mass_fixer=false (TM5-faithful): let m evolve freely with the fluxes.
        #   No rm scaling, no m_dev reset. Requires F64 to avoid B-correction
        #   accumulation errors driving cells negative.
        if _use_mass_fixer && gpu.dm !== nothing
            # Use n_sub_eff so the prescribed trajectory is sampled at the
            # right substep when global Check_CFL has refined dt.
            t_end = FT(s) / FT(n_sub_eff)
            m_prescribed = gpu.m_ref .+ t_end .* gpu.dm
            scale = m_prescribed ./ max.(gpu.m_dev, FT(1))
            for (_, rm_t) in pairs(tracers)
                rm_t .*= scale
            end
            copyto!(gpu.m_dev, m_prescribed)
        end
    end

    # Set m_ref to end-of-window mass.
    # mass_fixer=true: use prescribed trajectory (m_ref += dm).
    # mass_fixer=false: use evolved mass from advection (TM5-faithful).
    if _use_mass_fixer && gpu.dm !== nothing
        gpu.m_ref .+= gpu.dm
    else
        copyto!(gpu.m_ref, gpu.m_dev)
    end

    # Convective transport ONCE per window (after all substeps).
    # rm↔c roundtrip uses m_ref (now = m_dev, consistent with rm).
    dt_conv = FT(n_sub) * dt_sub
    _has_conv = (phys.has_tm5conv && phys.tm5conv_loaded[]) || phys.cmfmc_loaded[]
    if _has_conv
        for (_, rm) in pairs(tracers)
            rm ./= gpu.m_ref
        end
        if phys.has_tm5conv && phys.tm5conv_loaded[]
            # TM5 matrix convection — GPU path via KA kernels.
            # Uses pre-allocated workspace for the per-column transfer matrix.
            convect!(tracers, phys.tm5conv_gpu, gpu.Δp,
                      model.convection, grid, dt_conv, phys.planet;
                      workspace=phys.tm5conv_ws)
        else
            convect!(tracers, phys.cmfmc_gpu, gpu.Δp,
                      model.convection, grid, dt_conv, phys.planet;
                      dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                      workspace=phys.ras_workspace)
        end
        for (_, c) in pairs(tracers)
            c .*= gpu.m_ref
        end
    end
end

# Lazy-allocated workspace caches (avoids re-allocation per window)
const _PRATHER_WS_CACHE = Ref{Any}(nothing)
const _CS_PRATHER_WS_CACHE = Ref{Any}(nothing)
const _PROG_SLOPES_CACHE = Ref{Any}(nothing)

function _build_advection_workspace(ws, scheme, tracers, m)
    # Check if prognostic slopes are enabled via model metadata
    # (set by configuration when advection.prognostic_slopes = true)
    if scheme isa SlopesAdvection && scheme.prognostic_slopes
        if _PROG_SLOPES_CACHE[] === nothing
            _PROG_SLOPES_CACHE[] = allocate_prognostic_slope_workspaces(tracers, m)
            @info "Allocated prognostic slope workspaces for $(length(tracers)) tracers"
        end
        return (; base=ws, prog_slopes=_PROG_SLOPES_CACHE[])
    end
    return ws  # default: return MassFluxWorkspace as-is
end
function _build_advection_workspace(ws, scheme::PratherAdvection, tracers, m)
    if _PRATHER_WS_CACHE[] === nothing
        _PRATHER_WS_CACHE[] = allocate_prather_workspaces(tracers, m)
        @info "Allocated Prather workspaces for $(length(tracers)) tracers"
    end
    return (; base=ws, prather=_PRATHER_WS_CACHE[])
end

"""Get per-tracer CS Prather workspace (lazy-allocated)."""
function _get_cs_prather_ws(tracers, grid, arch)
    if _CS_PRATHER_WS_CACHE[] === nothing
        _CS_PRATHER_WS_CACHE[] = allocate_cs_prather_workspaces(tracers, grid, arch)
        @info "Allocated CS Prather workspaces for $(length(tracers)) tracers"
    end
    return _CS_PRATHER_WS_CACHE[]
end

# ---------------------------------------------------------------------------
# GCHP advection: dry-basis implementation
# ---------------------------------------------------------------------------
function _gchp_advection_dry!(tracers, sched, air, phys, model,
        grid::CubedSphereGrid{FT}, ws, ws_lr, ws_vr, gc,
        n_sub, dt_sub, step, _ORD, has_next) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    N = Nc + 2Hp
    cx_gpu, cy_gpu = gpu.cx, gpu.cy
    g_val = FT(grid.gravity)

    step[] += n_sub

    # Step 1: rm → q (dry VMR)
    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, air.m, grid)
    end

    # Pre-compute constants
    mfx_dt = FT(model.met_data.mass_flux_dt)
    am_to_mfx = g_val * mfx_dt
    inv_am_to_mfx = FT(1) / am_to_mfx
    rarea = ntuple(p -> FT(1) ./ gc.area[p], 6)

    n_loop = has_next ? n_sub : 1
    per_step = model.advection_scheme.per_step_remap

    # QV_next for target PE / scaling (used in both paths)
    qv_tgt = phys.qv_next_loaded[] ? phys.qv_next_gpu :
             phys.qv_loaded[] ? phys.qv_gpu : nothing

    if has_next && phys.qv_loaded[] && per_step
        # ── Per-substep remap: matches GCHP offline_tracer_advection ────────
        # Each 450s step: horizontal transport → source PE → target PE
        # (hybrid from evolved PS) → remap → scale → rm→q for next step.
        ng = next_gpu(sched)

        for _isub in 1:n_loop
            frac_start = FT(_isub - 1) / FT(n_loop)
            frac_end   = FT(_isub)     / FT(n_loop)

            # Reset dp_work to interpolated dry DELP at start of substep
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                          frac_start; ndrange=(N, N, Nz))
            end

            # Horizontal advection: dp_work → evolved dpA
            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end

            # Source PE from evolved dpA (dp_work after horizontal step)
            for_panels_nosync() do p
                compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
            compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

            # Target PE: direct cumsum from prescribed dp, with surface PE locked
            # to source (evolved) surface PE. This prevents column mass change
            # through the remap while keeping interior levels at prescribed positions.
            # GCHP uses hybrid PE + surface locking, but hybrid needs moist PS.
            # For dry basis: direct cumsum + surface lock + bottom-layer absorption.
            if _isub < n_loop
                # Target PE: direct cumsum from prescribed dp, with column mass
                # scaled to match source (evolved) surface pressure.
                # This distributes the mass adjustment proportionally across ALL
                # levels (not just the bottom layer, which caused q distortion).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                              phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                              frac_end; ndrange=(N, N, Nz))
                end
                compute_target_pressure_from_delp_direct!(ws_vr, ws_vr.dp_work, gc, grid)
                # Scale dp_tgt so column sum = source column sum (ps_src from evolved mass).
                # dp_tgt[k] *= ps_src / ps_tgt (proportional distribution).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.pe_tgt[p])
                    _scale_dp_tgt_to_source_ps_kernel!(be, 256)(
                        ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                        ws_vr.pe_src[p], ws_vr.ps_src[p],
                        Nc, Nz; ndrange=(Nc, Nc))
                end
            else
                # Last: target = ng.delp × (1-qv_tgt) — identical to single-remap
                if qv_tgt !== nothing
                    compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                        qv_tgt, gc, grid)
                else
                    compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
                end
            end

            # q → rm, remap, fillz
            for (_, rm_t) in pairs(tracers)
                q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
            end
            for (_, rm_t) in pairs(tracers)
                vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
            end
            for (_, rm_t) in pairs(tracers)
                fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
            end

            # Scale and prepare for next substep
            if _isub < n_loop
                # No intermediate scaling — proportional dp_tgt scaling + surface-locked
                # PE ensures column mass conservation. Float32 PPM drift is ~0.002%/2d.

                # Convert rm → q using TARGET air mass (surface-locked dp_tgt).
                # Using prescribed dp_work would cause mass loss because the
                # prescribed column mass ≠ surface-locked column mass.
                # Write dp_tgt back to dp_work interior for compute_air_mass_panel!.
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _copy_dp_tgt_to_dp_work_kernel!(be, 256)(
                        ws_vr.dp_work[p], ws_vr.dp_tgt[p], Hp, Nc, Nz;
                        ndrange=(Nc, Nc, Nz))
                end
                for_panels_nosync() do p
                    compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                            gc.area[p], g_val, Nc, Nz, Hp)
                end
                for (_, rm_t) in pairs(tracers)
                    rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
                end
            else
                # Last substep: scale against ng.delp × (1-qv_tgt)
                for (tname, rm_t) in pairs(tracers)
                    scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp,
                                  gc, grid; qv_panels=qv_tgt)
                    @info "calcScaling: $tname = $scaling" maxlog=200
                    apply_scaling_factor!(rm_t, scaling, grid)
                end
                # tracers remain in rm form after last substep
            end
        end

    elseif has_next && phys.qv_loaded[]
        # ── n_sub loop, single remap at end (original behavior) ─────────────
        ng = next_gpu(sched)
        for _isub in 1:n_loop
            frac = FT(_isub - 1) / FT(n_loop)
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                          frac; ndrange=(N, N, Nz))
            end

            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end
        end

    else
        # Single step: use current dry dp (no next window available)
        if phys.qv_loaded[]
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _compute_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], phys.qv_gpu[p];
                          ndrange=(N, N, Nz))
            end
        else
            for_panels_nosync() do p; copyto!(ws_vr.dp_work[p], gpu.delp[p]); end
        end

        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                          cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                          gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end
    end

    # ── Vertical remap (single, at end) — skipped when per_step_remap did it ──
    if !(has_next && phys.qv_loaded[] && per_step)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
        compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

        if has_next
            ng = next_gpu(sched)
            if qv_tgt !== nothing
                compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                    qv_tgt, gc, grid)
            else
                compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
            end
        else
            compute_target_pressure_from_mass_direct!(ws_vr, ws_vr.m_save, gc, grid)
        end
        for (_, rm_t) in pairs(tracers)
            q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
        end
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
        end
        for (_, rm_t) in pairs(tracers)
            fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
        end

        if has_next
            ng = next_gpu(sched)
            for (tname, rm_t) in pairs(tracers)
                scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid;
                    qv_panels=qv_tgt)
                @info "calcScaling: $tname = $scaling" maxlog=200
                apply_scaling_factor!(rm_t, scaling, grid)
            end
        end
    end

    # air.m = dry mass from prescribed endpoint
    if has_next
        ng = next_gpu(sched)
        if qv_tgt !== nothing
            for_panels_nosync() do p
                compute_air_mass_panel!(air.m[p], ng.delp[p], qv_tgt[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
        else
            for_panels_nosync() do p
                compute_air_mass_panel!(air.m[p], ng.delp[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end
    for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end
    if phys.qv_loaded[]
        for_panels_nosync() do p
            air.m_wet[p] .= air.m[p] ./ max.(1 .- phys.qv_gpu[p], eps(FT))
        end
    else
        for_panels_nosync() do p; copyto!(air.m_wet[p], air.m[p]); end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# GCHP advection: moist-basis with interpolated QV
#
# Key differences from dry:
# 1. dp = moist DELP (no QV correction)
# 2. MFX = dry × /(1-QV) (moist mass flux)
# 3. Target PE = hybrid from evolved moist PS (column-preserving!)
# 4. Back-conversion uses QV_next from met data (NOT prognostic QV)
#    → no PPM/remap noise in QV, smooth met-data QV at end of window
# 5. air.m = dry mass from DELP_next × (1-QV_next)
#
# Since sum(DELP) = PS exactly in GEOS-IT, the moist basis is perfectly
# consistent with the hybrid ak/bk coefficients.
# ---------------------------------------------------------------------------
function _gchp_advection_moist!(tracers, sched, air, phys, model,
        grid::CubedSphereGrid{FT}, ws, ws_lr, ws_vr, gc,
        n_sub, dt_sub, step, _ORD, has_next) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    N = Nc + 2Hp
    cx_gpu, cy_gpu = gpu.cx, gpu.cy
    g_val = FT(grid.gravity)

    step[] += n_sub

    # ── Moist sub-step diagnostics: capture initial state (first window only) ──
    _do_diag = MOIST_DIAG[] !== nothing && !(MOIST_DIAG[]::MoistSubStepDiag{FT}).captured
    if _do_diag
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6
            _diag.qv_start[p]   .= Array(phys.qv_gpu[p])
            _diag.delp_start[p] .= Array(gpu.delp[p])
        end
        # q_dry_init: tracers are in rm form at this point (before rm_to_q)
        if haskey(tracers, :co2)
            for p in 1:6; _diag.q_dry_init[p] .= Array(tracers.co2[p]); end
        end
    end

    # Step 1: rm → q_wet (divide by moist air mass)
    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, air.m_wet, grid)
    end

    # Pre-compute constants
    mfx_dt = FT(model.met_data.mass_flux_dt)
    am_to_mfx = g_val * mfx_dt
    inv_am_to_mfx = FT(1) / am_to_mfx
    rarea = ntuple(p -> FT(1) ./ gc.area[p], 6)

    # ── n_sub loop with dp-reset (interpolated DELP, moist) ─────────────
    n_loop = has_next ? n_sub : 1
    per_step = model.advection_scheme.per_step_remap

    if has_next && per_step
        # ── Per-substep remap: matches GCHP offline_tracer_advection ────────
        ng = next_gpu(sched)

        # GCHP applies humidity correction ONCE before all substeps (GCHPctmEnv:1029).
        # Scale am/bm to Pa·m² and correct for humidity — keep corrected for all substeps.
        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end

        # QV advection as tracer NQ+1 (GCHP: AdvCore:1068) was tested but provides
        # only ~3% correction to the moist path artifact (see MoistSubStepDiag analysis).
        # The fundamental issue is that PPM homogenizes q_total, erasing the QV imprint,
        # while QV retains its structure — so back-conversion reintroduces QV patterns.
        # Disabled to avoid GPU allocation overhead (6 CuArrays per window).
        tracers_with_qv = tracers

        for _isub in 1:n_loop
            frac_start = FT(_isub - 1) / FT(n_loop)
            frac_end   = FT(_isub)     / FT(n_loop)

            # Reset dp_work to interpolated moist DELP at start of substep
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          frac_start; ndrange=(N, N, Nz))
            end

            # Horizontal advection (fluxes already corrected for humidity)
            gchp_tracer_2d!(tracers_with_qv, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)

            # ── Diag: capture q_wet after horizontal advection (substep 1, first window) ──
            if _do_diag && _isub == 1 && haskey(tracers, :co2)
                for p in 1:6
                    (MOIST_DIAG[]::MoistSubStepDiag{FT}).q_wet_post_hadv[p] .= Array(tracers.co2[p])
                end
            end

            # Source PE from evolved dpA (moist)
            for_panels_nosync() do p
                compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
            compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

            # Target PE: direct cumsum from prescribed moist DELP, with column
            # mass scaled proportionally to match evolved surface pressure.
            # (Same approach as dry fix7 — hybrid PE accumulates drift over 384 substeps.)
            if _isub < n_loop
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                              frac_end; ndrange=(N, N, Nz))
                end
                compute_target_pressure_from_delp_direct!(ws_vr, ws_vr.dp_work, gc, grid)
                for_panels_nosync() do p
                    be = get_backend(ws_vr.pe_tgt[p])
                    _scale_dp_tgt_to_source_ps_kernel!(be, 256)(
                        ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                        ws_vr.pe_src[p], ws_vr.ps_src[p],
                        Nc, Nz; ndrange=(Nc, Nc))
                end
            else
                compute_target_pe_from_evolved_ps!(ws_vr, gc, grid)
            end

            # q → rm (moist), remap, fillz — including QV tracer
            for (_, rm_t) in pairs(tracers_with_qv)
                q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
            end
            for (_, rm_t) in pairs(tracers_with_qv)
                vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
            end
            for (_, rm_t) in pairs(tracers_with_qv)
                fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
            end

            # ── Diag: capture rm after vertical remap + fillz (substep 1, first window) ──
            if _do_diag && _isub == 1 && haskey(tracers, :co2)
                for p in 1:6
                    (MOIST_DIAG[]::MoistSubStepDiag{FT}).rm_post_vremap[p] .= Array(tracers.co2[p])
                end
            end

            if _isub < n_loop
                # No intermediate scaling — proportional dp_tgt scaling + surface-locked
                # PE ensures column mass conservation.

                # Convert rm → q using TARGET mass (dp_tgt, surface-locked).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _copy_dp_tgt_to_dp_work_kernel!(be, 256)(
                        ws_vr.dp_work[p], ws_vr.dp_tgt[p], Hp, Nc, Nz;
                        ndrange=(Nc, Nc, Nz))
                end
                for_panels_nosync() do p
                    compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                            gc.area[p], g_val, Nc, Nz, Hp)
                end
                for (_, rm_t) in pairs(tracers_with_qv)
                    rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
                end
            else
                # Last substep: scale against actual moist ng.delp (all tracers + QV)
                for (tname, rm_t) in pairs(tracers_with_qv)
                    scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid)
                    @info "calcScaling: $tname = $scaling" maxlog=200
                    apply_scaling_factor!(rm_t, scaling, grid)
                end
                # tracers remain in rm form for back-conversion below
            end
        end

        # Restore am/bm: reverse humidity correction + unscale
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end

    elseif has_next
        ng = next_gpu(sched)
        for _isub in 1:n_loop
            frac = FT(_isub - 1) / FT(n_loop)
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          frac; ndrange=(N, N, Nz))
            end

            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            for_panels_nosync() do p
                be = get_backend(gpu.am[p])
                _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
                _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                be = get_backend(gpu.am[p])
                _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
                _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
            end
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end
        end
    else
        for_panels_nosync() do p; copyto!(ws_vr.dp_work[p], gpu.delp[p]); end
        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                          cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                          gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end
    end

    # ── Vertical remap (single, at end) — skipped when per_step_remap did it ──
    if !(has_next && per_step)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
        compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)
        compute_target_pe_from_evolved_ps!(ws_vr, gc, grid)

        for (_, rm_t) in pairs(tracers)
            q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
        end
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
        end
        for (_, rm_t) in pairs(tracers)
            fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
        end

        if has_next
            ng = next_gpu(sched)
            for (tname, rm_t) in pairs(tracers)
                scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid)
                @info "calcScaling: $tname = $scaling" maxlog=200
                apply_scaling_factor!(rm_t, scaling, grid)
            end
        end
    end

    # Back-conversion: wet→dry using met-data QV.
    # Note: GCHP uses advected QV (tracer NQ+1) but testing showed only ~3% improvement
    # due to PPM homogenizing q_total while QV retains its structure.
    qv_back = phys.qv_next_loaded[] ? phys.qv_next_gpu : phys.qv_gpu

    # ── Diag: capture delp_end (first window only) ──
    if _do_diag && has_next
        _ng = next_gpu(sched)
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6; _diag.delp_end[p] .= Array(_ng.delp[p]); end
    end

    if has_next
        ng = next_gpu(sched)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ng.delp[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end

    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
    end
    # ── Diag: capture qv_back AFTER rm_to_q conversion ──
    if _do_diag
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6
            _diag.qv_back[p] .= Array(qv_back[p])
        end
    end

    for (_, q_t) in pairs(tracers)
        for_panels_nosync() do p
            be = get_backend(q_t[p])
            _divide_by_1_minus_qv_kernel!(be, 256)(q_t[p], qv_back[p], Hp; ndrange=(Nc, Nc, Nz))
        end
    end

    # ── Diag: capture q_dry_final after wet→dry back-conversion (first window) ──
    if _do_diag && haskey(tracers, :co2)
        for p in 1:6
            (MOIST_DIAG[]::MoistSubStepDiag{FT}).q_dry_final[p] .= Array(tracers.co2[p])
        end
        (MOIST_DIAG[]::MoistSubStepDiag{FT}).captured = true
        @info "MoistSubStepDiag: first window captured"
    end

    if has_next
        ng = next_gpu(sched)
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ng.delp[p], qv_back[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end

    for (_, q_t) in pairs(tracers)
        q_to_rm_panels!(q_t, air.m, grid)
    end

    for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end
    for_panels_nosync() do p
        air.m_wet[p] .= air.m[p] ./ max.(1 .- qv_back[p], eps(FT))
    end
    return nothing
end

# ---------------------------------------------------------------------------

function advection_phase!(tracers, sched, air, phys, model,
                           grid::CubedSphereGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           geom_gchp=nothing, ws_gchp=nothing,
                           has_next::Bool=false,
                           dB_gpu=nothing) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    # GCHP-faithful: get CX/CY from GPU met buffer
    cx_gpu = gpu.cx
    cy_gpu = gpu.cy

    _use_gchp = _needs_gchp(model.advection_scheme)
    _ORD = model.advection_scheme isa PPMAdvection ?
        Val(_ppm_order(model.advection_scheme)) : Val(4)

    if _use_gchp && ws_vr !== nothing && cx_gpu !== nothing && gpu.xfx !== nothing
        # ═══════════════════════════════════════════════════════════════════
        # GCHP PATH: dispatch on transport basis (dry or moist)
        # ═══════════════════════════════════════════════════════════════════
        _basis = get(model.metadata, "pressure_basis", "dry")
        if _basis == "moist" && phys.qv_loaded[]
            _gchp_advection_moist!(tracers, sched, air, phys, model, grid,
                ws, ws_lr, ws_vr, gc, n_sub, dt_sub, step, _ORD, has_next)
        else
            _gchp_advection_dry!(tracers, sched, air, phys, model, grid,
                ws, ws_lr, ws_vr, gc, n_sub, dt_sub, step, _ORD, has_next)
        end

        # ── Convection AFTER full advection ─────────────────────────
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end

    elseif ws_vr !== nothing
        # ═══════════════════════════════════════════════════════════════════
        # REMAP PATH: Standard Lin-Rood + vertical remap (prescale/rescale)
        # ═══════════════════════════════════════════════════════════════════

        # Save prescribed m (from compute_air_mass_phase!)
        for_panels_nosync() do p; copyto!(ws_vr.m_save[p], air.m[p]); end

        for _ in 1:n_sub
            step[] += 1

            for (_, rm_t) in pairs(tracers)
                for_panels_nosync() do p; copyto!(air.m[p], ws_vr.m_save[p]); end

                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, _ORD, ws, ws_lr;
                              damp_coeff=model.advection_scheme.damp_coeff)
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, _ORD, ws, ws_lr;
                              damp_coeff=FT(0))

                for_panels_nosync() do p
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
        for_panels_nosync() do p; copyto!(air.m[p], ws_vr.m_save[p]); end
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
        for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end

        # Recompute m_wet from updated air.m for post-advection physics.
        # After remap, air.m is on target pressure basis; m_wet must match
        # so that diffusion's q = rm/m_wet is consistent with remapped rm.
        # m_wet = m_dry / (1-qv). Uses current-window QV (<0.1% approx).
        if phys.qv_loaded[]
            for_panels_nosync() do p
                air.m_wet[p] .= air.m[p] ./ max.(1 .- phys.qv_gpu[p], eps(FT))
            end
        else
            for_panels_nosync() do p; copyto!(air.m_wet[p], air.m[p]); end
        end
    else
        # ═══════════════════════════════════════════════════════════════════
        # STRANG PATH: Existing cm-based Z-advection
        # ═══════════════════════════════════════════════════════════════════
        for _ in 1:n_sub
            step[] += 1

            # Advect each tracer independently (m reset per tracer)
            # For Prather, get per-tracer CS workspace (lazy-allocated)
            _cs_pw_dict = model.advection_scheme isa PratherAdvection ?
                _get_cs_prather_ws(tracers, grid, model.architecture) : nothing
            for (tname, rm_t) in pairs(tracers)
                for_panels_nosync() do p
                    copyto!(air.m[p], air.m_ref[p])
                end
                _pw_cs = _cs_pw_dict !== nothing ? _cs_pw_dict[tname] : nothing
                _apply_advection_cs!(rm_t, air.m, gpu.am, gpu.bm, gpu.cm,
                                      grid, model.advection_scheme, ws;
                                      ws_lr, cx=cx_gpu, cy=cy_gpu,
                                      geom_gchp, ws_gchp, pw_cs=_pw_cs)
            end
            # air.m now holds m_evolved (same for all tracers)

            # Per-cell mass fixer: rm = (rm / m_evolved) × m_ref
            if get(model.metadata, "mass_fixer", true)
                for (_, rm_t) in pairs(tracers)
                    for_panels_nosync() do p
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

"""Post-advection physics: BLD diffusion, PBL diffusion, chemistry.
LL tracers are rm — convert to VMR for operators, then back (same m_ref → exact).
TM5-faithful: moist basis for boundary conversions."""
function post_advection_physics!(tracers, sched, air, phys, model,
                                  grid::LatitudeLongitudeGrid, dt_window, dw)
    gpu = current_gpu(sched)

    # rm → c (moist VMR) for diffusion and chemistry
    for (_, rm) in pairs(tracers)
        rm ./= gpu.m_ref
    end

    _apply_bld!(tracers, dw)

    if phys.sfc_loaded[]
        diffuse_pbl!(tracers, gpu.Δp,
                      phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                      phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                      phys.w_scratch,
                      model.diffusion, grid, dt_window, phys.planet)
    end

    apply_chemistry!(tracers, grid, model.chemistry, dt_window)

    # c → rm
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
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
apply_mass_correction!(tracers, grid::LatitudeLongitudeGrid, diag;
                        mass_fixer::Bool=true, mass_fixer_tracers::Vector{String}=String[]) = nothing

function apply_mass_correction!(tracers, grid::CubedSphereGrid, diag;
                                 mass_fixer::Bool=true,
                                 mass_fixer_tracers::Vector{String}=String[])
    mass_fixer || return
    isempty(diag.pre_adv_mass) && return
    diag.fix_value = apply_global_mass_fixer!(tracers, grid, diag.pre_adv_mass;
                                               fix_interval_scale=diag.fix_interval_scale,
                                               fix_total_scale=diag.fix_total_scale,
                                               allowed_tracers=mass_fixer_tracers)
end

# =====================================================================
# Phase 11: Compute output air mass (dry correction)
# =====================================================================

"""Compute LL output air mass in the transport basis expected by diagnostics.

Moist-basis binaries are converted to dry mass when QV is available.
Dry-basis binaries return their stored mass directly."""
function compute_output_mass(sched, air, phys, grid::LatitudeLongitudeGrid, driver=nothing)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_ref)
        _compute_ll_dry_mass!(phys.m_dry, gpu.m_ref, phys.qv_gpu, _ll_mass_basis(driver))
        return phys.m_dry
    end
    return gpu.m_ref
end

"""Convert rm tracers to dry VMR for output. LL: c_dry = rm / m_dry.
Creates GPU temporaries (once per output interval). CS: tracers are already rm."""
rm_to_vmr(tracers, sched, phys, grid::LatitudeLongitudeGrid) =
    map(rm -> rm ./ ll_dry_mass(phys), tracers)
rm_to_vmr(tracers, sched, phys, grid::CubedSphereGrid) = tracers

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
    gpu = current_gpu(sched)
    base = (; ps=Array(gpu.ps))
    if phys.sfc_loaded[] && phys.has_pbl
        base = merge(base, (; pblh=Array(phys.pbl_sfc_gpu.pblh)))
    end
    return base
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

"""Write IC output snapshot (t=0) for all grid types."""
function write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                          grid::LatitudeLongitudeGrid, half_dt, dt_window)
    met_ic = build_met_fields(sched, phys, grid, half_dt, dt_window)
    ic_mass = compute_output_mass(sched, air, phys, grid, model.met_data)
    compute_ll_dry_mass!(phys, sched, grid, model.met_data)
    c_tracers = rm_to_vmr(tracers, sched, phys, grid)
    for writer in writers
        write_output!(writer, model, 0.0;
                      air_mass=ic_mass, tracers=c_tracers, met_fields=met_ic,
                      rm_tracers=tracers)
    end
end

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
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out;
                           sim_time=nothing)
    # Prefer caller-provided sim_time; fall back to step*dt_sub for callers
    # that don't yet pass it. The fallback is wrong when the global
    # Check_CFL pre-pass has inflated step[] (n_extra > 1) — pass sim_time
    # explicitly to get correct day display.
    actual_sim_time = sim_time === nothing ? Float64(step * dt_sub) : Float64(sim_time)
    sv = Pair{Symbol,Any}[
        :day  => @sprintf("%.1f", actual_sim_time / 86400),
        :rate => @sprintf("%.2f s/win", w > 1 ? (time() - wall_start) / w : 0.0)]
    isempty(diag.showvalue) || push!(sv, :mass => diag.showvalue)
    next!(prog; showvalues=sv)
end

function update_progress!(prog, diag, grid::CubedSphereGrid,
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out;
                           sim_time=nothing)
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
                                wall_start, n_win, step, t_io, t_gpu, t_out;
                                t_phases=nothing)
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
