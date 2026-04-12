# ---------------------------------------------------------------------------
# Run-loop helper functions: physics setup, advection dispatch, emission prep
#
# Extracted from run_implementations.jl for modularity.
# All functions here are used by the unified run loop and phase functions.
# ---------------------------------------------------------------------------

using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid, cell_area
using ..Grids: fill_panel_halos!, allocate_cubed_sphere_field
using ..Architectures: array_type,
                       AbstractPanelMap, SingleGPUMap, PanelGPUMap,
                       PerGPUWorkspace, workspace_for,
                       allocate_ntuple_panels, sync_all_gpus
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ..Advection: MassFluxWorkspace, allocate_massflux_workspace,
                   allocate_cs_massflux_workspace,
                   PrognosticSlopeWorkspace, allocate_prognostic_slope_workspaces,
                   strang_split_massflux!, strang_split_massflux_ppm!,
                   strang_split_prognostic!,
                   PratherAdvection, PratherWorkspace,
                   allocate_prather_workspace, allocate_prather_workspaces,
                   strang_split_prather!,
                   CSPratherWorkspace, allocate_cs_prather_workspace,
                   allocate_cs_prather_workspaces, strang_split_prather_cs!,
                   LinRoodWorkspace, fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!,
                   GCHPGridGeometry, GCHPTransportWorkspace,
                   fv_tp_2d_gchp!, fv_tp_2d_gchp_fluxes!, gchp_tracer_2d!,
                   _correct_mfx_humidity_kernel!, _correct_mfy_humidity_kernel!,
                   _reverse_mfx_humidity_kernel!, _reverse_mfy_humidity_kernel!,
                   _multiply_by_1_minus_qv_kernel!, _divide_by_1_minus_qv_kernel!,
                   _compute_dry_dp_kernel!, _copy_nohalo_to_halo_kernel!,
                   _interpolate_dry_dp_kernel!, _interpolate_dp_kernel!,
                   _column_dp_correction_kernel!, _column_dp_correction_moist_kernel!,
                   strang_split_gchp_ppm!, compute_area_fluxes!,
                   rm_to_q_panels!, q_to_rm_panels!,
                   compute_dp_from_m_panels!, set_m_from_dp_panels!,
                   VerticalRemapWorkspace, vertical_remap_cs!,
                   fix_target_bottom_pe!,
                   compute_target_pe_from_hybrid_coords!,
                   compute_source_pe_from_hybrid!,
                   compute_target_pe_from_ps_hybrid!,
                   compute_source_pe_from_evolved_mass!,
                   compute_target_pe_from_evolved_ps!,
                   compute_target_pressure_from_next_delp!,
                   compute_target_pressure_from_delp_direct!,
                   compute_target_pressure_from_dry_delp_direct!,
                   compute_target_pressure_from_mass_direct!,
                   compute_target_pressure_from_mass!,
                   _lock_surface_pe_kernel!, _copy_dp_tgt_to_dp_work_kernel!,
                   _scale_dp_tgt_to_source_ps_kernel!,
                   _column_sum_rm_kernel!, _column_mass_correct_kernel!,
                   update_air_mass_from_target!,
                   calc_scaling_factor, apply_scaling_factor!, gchp_calc_scaling_factor,
                   fillz_panels!,
                   compute_dry_ple!,
                   compute_air_mass!,
                   compute_mass_fluxes!, compute_air_mass_panel!, compute_air_mass_panels!, compute_cm_panel!,
                   compute_cm_pressure_fixer_panel!,
                   compute_cm_mass_weighted_panel!,
                   apply_mass_fixer!,
                   apply_dry_delp_panel!, apply_dry_am_panel!, apply_dry_bm_panel!,
                   apply_dry_cmfmc_panel!, apply_dry_dtrain_panel!,
                   build_geometry_cache, max_cfl_x_cs, max_cfl_y_cs,
                   SlopesAdvection, PPMAdvection
using ..Diffusion: DiffusionWorkspace, diffuse_gpu!, diffuse_cs_panels!,
                   BoundaryLayerDiffusion, PBLDiffusion, NonLocalPBLDiffusion,
                   diffuse_pbl!, diffuse_nonlocal_pbl!,
                   build_diffusion_coefficients
using ..Parameters: PlanetParameters, load_parameters
using ..Convection: AbstractConvection, TiedtkeConvection, RASConvection,
                    TM5MatrixConvection, convect!, invalidate_ras_cfl_cache!,
                    TM5ConvWorkspace, allocate_tm5conv_workspace
using ..Sources: AbstractSurfaceFlux, apply_surface_flux!, apply_surface_flux_pbl!,
                 SurfaceFlux, TimeVaryingSurfaceFlux,
                 LatLonLayout, CubedSphereLayout,
                 apply_emissions_window!, apply_emissions_window_pbl!,
                 update_time_index!, flux_data,
                 M_AIR, M_CO2, M_SF6, M_RN222
using ..Chemistry: apply_chemistry!
using ..IO: AbstractMetDriver, AbstractRawMetDriver, AbstractMassFluxMetDriver,
            total_windows, window_dt, steps_per_window, load_met_window!,
            load_cmfmc_window!, load_dtrain_window!, load_tm5conv_window!,
            load_qv_window!, load_surface_fields_window!,
            load_qv_and_ps_pair!, load_ps_from_ctm_i1!, load_cx_cy_window!,
            load_all_window!, load_physics_window!,
            LatLonMetBuffer, LatLonCPUBuffer, CubedSphereMetBuffer, CubedSphereCPUBuffer,
            upload!, AbstractOutputWriter, write_output!, finalize_output!,
            get_pending_ic, apply_pending_ic!,
            finalize_ic_vertical_interp!, has_deferred_ic_vinterp
using ..Diagnostics: column_mean!, surface_slice!
using Printf
using ProgressMeter

# =====================================================================
# Physics setup/apply helpers — dispatch instead of isa checks
# =====================================================================

# --- BoundaryLayerDiffusion workspace (pre-computed tridiagonal factors) ---
_setup_bld_workspace(d::BoundaryLayerDiffusion, grid, dt, template) =
    DiffusionWorkspace(grid, eltype(template)(d.Kz_max),
                       eltype(template)(d.H_scale), dt, template)
_setup_bld_workspace(::AbstractDiffusion, grid, dt, template) = nothing
_setup_bld_workspace(::Nothing, grid, dt, template) = nothing

# --- Apply BLD diffusion (static Kz) ---
_apply_bld!(tracers::NamedTuple, ::Nothing) = nothing
_apply_bld!(tracers::NamedTuple, dw::DiffusionWorkspace) = diffuse_gpu!(tracers, dw)
_apply_bld_cs!(rm_panels, m_panels, ::Nothing, Nc, Nz, Hp) = nothing
_apply_bld_cs!(rm_panels, m_panels, dw::DiffusionWorkspace, Nc, Nz, Hp) =
    diffuse_cs_panels!(rm_panels, m_panels, dw, Nc, Nz, Hp)

# --- Query whether physics buffers are needed ---
_needs_convection(::TiedtkeConvection) = true
_needs_convection(::RASConvection) = true
_needs_convection(::TM5MatrixConvection) = true
_needs_convection(::AbstractConvection) = false
_needs_convection(::Nothing) = false
_needs_dtrain(::RASConvection) = true
_needs_dtrain(::AbstractConvection) = false
_needs_dtrain(::Nothing) = false
_needs_tm5conv(::TM5MatrixConvection) = true
_needs_tm5conv(::AbstractConvection) = false
_needs_tm5conv(::Nothing) = false
_needs_pbl(::PBLDiffusion) = true
_needs_pbl(::NonLocalPBLDiffusion) = true
_needs_pbl(::AbstractDiffusion) = false
_needs_pbl(::Nothing) = false
_diff_label(::PBLDiffusion) = "pbl"
_diff_label(::NonLocalPBLDiffusion) = "nonlocal_pbl"
_diff_label(d::AbstractDiffusion) = string(typeof(d))
_diff_label(::Nothing) = "none"

# Extract Int32 cluster_sizes for reduced-grid x-advection, or nothing for uniform.
# Cap at MAX_CLUSTER to keep GPU kernel memory reads O(1) per thread;
# larger clusters cause O(r) reads which dominates runtime.
const _MAX_GPU_CLUSTER = Int32(720)

function _get_cluster_sizes_cpu(grid::LatitudeLongitudeGrid)
    rg = grid.reduced_grid
    rg === nothing && return nothing
    # Round cluster sizes down to the nearest power of 2, capped
    cs = Int32.(rg.cluster_sizes)
    @inbounds for j in eachindex(cs)
        r = cs[j]
        if r > _MAX_GPU_CLUSTER
            cs[j] = _MAX_GPU_CLUSTER
        end
    end
    # Verify all cluster sizes divide Nx
    Nx = grid.Nx
    @inbounds for j in eachindex(cs)
        r = cs[j]
        if r > 1 && Nx % r != 0
            cs[j] = Int32(1)
        end
    end
    return cs
end

# --- Apply advection with dispatch on scheme type ---

"""Query whether an advection scheme needs the LinRoodWorkspace."""
_needs_linrood(::SlopesAdvection) = false
_needs_linrood(::PratherAdvection) = false
_needs_linrood(s::PPMAdvection) = s.use_linrood || s.use_gchp

"""Query whether an advection scheme uses vertical remapping instead of Z-advection."""
_needs_vertical_remap(::SlopesAdvection) = false
_needs_vertical_remap(::PratherAdvection) = false
_needs_vertical_remap(s::PPMAdvection) = s.use_vertical_remap

"""Query whether an advection scheme uses GCHP-faithful transport (needs CX/CY)."""
_needs_gchp(::SlopesAdvection) = false
_needs_gchp(::PratherAdvection) = false
_needs_gchp(s::PPMAdvection) = s.use_gchp

"""Extract PPM order as Int for Val dispatch."""
_ppm_order(::PPMAdvection{ORD}) where ORD = ORD

"""Apply cubed-sphere mass-flux advection, dispatching on advection scheme."""
function _apply_advection_cs!(rm_panels, m_panels, am, bm, cm, grid,
                               scheme::SlopesAdvection, ws; ws_lr=nothing, kwargs...)
    strang_split_massflux!(rm_panels, m_panels, am, bm, cm, grid, true, ws)
end

function _apply_advection_cs!(rm_panels, m_panels, am, bm, cm, grid,
                               scheme::PPMAdvection{ORD}, ws;
                               ws_lr=nothing,
                               cx=nothing, cy=nothing,
                               geom_gchp=nothing, ws_gchp=nothing,
                               kwargs...) where ORD
    if scheme.use_gchp && cx !== nothing && geom_gchp !== nothing && ws_lr !== nothing
        # GCHP-faithful transport: area-based pre-advection + Courant PPM
        # Convert rm → q, run GCHP horizontal + Z-sweep, convert q → rm
        compute_area_fluxes!(ws_gchp, cx, cy, geom_gchp, grid)
        rm_to_q_panels!(rm_panels, m_panels, grid)
        strang_split_gchp_ppm!(rm_panels, m_panels, am, bm, cm,
                                 cx, cy, geom_gchp, ws_gchp,
                                 grid, Val(ORD), ws, ws_lr)
        q_to_rm_panels!(rm_panels, m_panels, grid)
    elseif scheme.use_linrood && ws_lr !== nothing
        strang_split_linrood_ppm!(rm_panels, m_panels, am, bm, cm, grid,
                                   Val(ORD), ws, ws_lr; damp_coeff=scheme.damp_coeff)
    else
        strang_split_massflux_ppm!(rm_panels, m_panels, am, bm, cm, grid, Val(ORD), ws;
                                   damp_coeff=scheme.damp_coeff)
    end
end

function _apply_advection_cs!(rm_panels, m_panels, am, bm, cm, grid,
                               scheme::PratherAdvection, ws;
                               ws_lr=nothing, pw_cs=nothing, kwargs...)
    strang_split_prather_cs!(rm_panels, m_panels, am, bm, cm,
                              grid, pw_cs, scheme.use_limiter;
                              cfl_ws_x=ws.cfl_x, cfl_ws_y=ws.cfl_y)
end

"""Apply lat-lon mass-flux advection, dispatching on advection scheme.
Tracers are tracer mass `rm` (TM5-style prognostic variable)."""
function _apply_advection_latlon!(tracers, m, am, bm, cm, grid,
                                   scheme::SlopesAdvection, ws;
                                   cfl_limit)
    # NOTE: prognostic slopes path is disabled pending TM5-faithful
    # per-row evolving-mass loop refinement (see ToClaude.md plan).
    actual_ws = ws isa NamedTuple && haskey(ws, :base) ? ws.base : ws
    strang_split_massflux!(tracers, m, am, bm, cm, grid, true, actual_ws; cfl_limit)
end

function _apply_advection_latlon!(tracers, m, am, bm, cm, grid,
                                   scheme::PPMAdvection{ORD}, ws;
                                   cfl_limit) where ORD
    strang_split_massflux_ppm!(tracers, m, am, bm, cm, grid, Val(ORD), ws;
                                cfl_limit)
end

function _apply_advection_latlon!(tracers, m, am, bm, cm, grid,
                                   scheme::PratherAdvection, ws;
                                   cfl_limit)
    # ws is a NamedTuple with .base (MassFluxWorkspace) and .prather (per-tracer PratherWorkspace)
    strang_split_prather!(tracers, m, am, bm, cm, grid, ws.prather, scheme.use_limiter)
end

# =====================================================================
# Emission preparation and application helpers
# =====================================================================

"""
Prepare emission source data on the device for a lat-lon grid.
Returns (flux_dev, area_j_dev, A_coeff, B_coeff) or nothing if no sources.
"""
function _prepare_latlon_emissions(sources, grid::LatitudeLongitudeGrid{FT},
                                    driver, arch) where FT
    AT = array_type(arch)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    area_j_dev = AT(area_j_cpu)

    A_coeff = hasproperty(driver, :A_coeff) ? driver.A_coeff : nothing
    B_coeff = hasproperty(driver, :B_coeff) ? driver.B_coeff : nothing

    emission_data = []
    for src in sources
        if src isa TimeVaryingSurfaceFlux{LatLonLayout}
            flux_dev = AT(flux_data(src))
            push!(emission_data, (src, Ref(flux_dev), Ref(src.current_idx)))
        elseif src isa SurfaceFlux{LatLonLayout}
            flux_dev = AT(src.flux)
            push!(emission_data, (src, flux_dev, nothing))
        end
    end

    return emission_data, area_j_dev, A_coeff, B_coeff
end

"""
Apply emissions for lat-lon grids (GPU-compatible via KA kernels).
Handles both static `SurfaceFlux` and `TimeVaryingSurfaceFlux` sources.
"""
function _apply_emissions_latlon!(tracers, emission_data, area_j_dev, ps_dev,
                                    A_coeff, B_coeff, grid, dt_window;
                                    sim_hours::Float64=0.0, arch=nothing,
                                    delp=nothing, pblh=nothing)
    Nz = grid.Nz
    g = grid.gravity
    use_pbl = delp !== nothing && pblh !== nothing
    for (src, flux_ref, idx_ref) in emission_data
        if src isa TimeVaryingSurfaceFlux{LatLonLayout}
            update_time_index!(src, sim_hours)
            if src.current_idx != idx_ref[]
                AT = array_type(arch)
                flux_ref[] = AT(flux_data(src))
                idx_ref[] = src.current_idx
            end
            flux_dev = flux_ref[]
        else
            flux_dev = flux_ref
        end

        name = src.species
        haskey(tracers, name) || continue
        c = tracers[name]
        mm = src.molar_mass
        if use_pbl
            apply_emissions_window_pbl!(c, flux_dev, delp, pblh,
                                         g, dt_window; molar_mass=mm)
        elseif A_coeff !== nothing && B_coeff !== nothing
            apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev,
                                     A_coeff, B_coeff, Nz, g, dt_window;
                                     molar_mass=mm)
        else
            FT = eltype(c)
            Δp_approx = FT(grid.reference_pressure / Nz)
            mol = FT(M_AIR / mm)
            @. c[:, :, Nz] += flux_dev * dt_window * mol * grid.gravity / Δp_approx
        end
    end
end

"""
Prepare cubed-sphere emission data on the device.
Uploads CPU flux panels to GPU once; returns vector of (source, flux_dev) tuples.
"""
function _prepare_cs_emissions(sources, grid::CubedSphereGrid{FT}, arch) where FT
    AT = array_type(arch)
    emission_data = []
    for src in sources
        if src isa TimeVaryingSurfaceFlux{CubedSphereLayout}
            panels = flux_data(src)
            flux_dev = ntuple(p -> AT(panels[p]), 6)
            push!(emission_data, (src, Ref(flux_dev), Ref(src.current_idx)))
        elseif src isa SurfaceFlux{CubedSphereLayout}
            flux_dev = ntuple(p -> AT(src.flux[p]), 6)
            push!(emission_data, (src, flux_dev, nothing))
        end
    end
    return emission_data
end

"""
Apply cubed-sphere emissions using pre-uploaded device flux panels.
"""
function _apply_emissions_cs!(cs_tracers::NamedTuple, emission_data,
                                area_panels::NTuple{6}, dt_window, Nc::Int, Hp::Int;
                                sim_hours::Float64=0.0, arch=nothing,
                                delp::Union{NTuple{6}, Nothing}=nothing,
                                pblh::Union{NTuple{6}, Nothing}=nothing)
    use_pbl = delp !== nothing && pblh !== nothing
    FT = eltype(area_panels[1])

    for (src, flux_ref, idx_ref) in emission_data
        species = src.species
        haskey(cs_tracers, species) || continue
        rm_t = cs_tracers[species]

        if src isa TimeVaryingSurfaceFlux{CubedSphereLayout}
            update_time_index!(src, sim_hours)
            if src.current_idx != idx_ref[]
                AT = array_type(arch)
                panels = flux_data(src)
                flux_ref[] = ntuple(p -> AT(panels[p]), 6)
                idx_ref[] = src.current_idx
            end
            flux_dev = flux_ref[]
        else
            flux_dev = flux_ref
        end

        if use_pbl
            mol_ratio = FT(M_AIR / src.molar_mass)
            apply_surface_flux_pbl!(rm_t, flux_dev, area_panels, delp, pblh,
                                     FT(dt_window), mol_ratio, Nc, Hp)
        else
            apply_surface_flux!(rm_t,
                SurfaceFlux(flux_dev, src.species, src.label;
                            molar_mass=src.molar_mass),
                area_panels, dt_window, Nc, Hp)
        end
    end
end

# =====================================================================
# Staging progress file — for NVMe staging daemon
# =====================================================================

"""Write current window date to staging progress file (if configured)."""
function _write_staging_progress(model, w::Int)
    cfg = get(model.metadata, "config", Dict())
    stg = get(cfg, "staging", nothing)
    stg === nothing && return
    pf = get(stg, "progress_file", "")
    isempty(pf) && return
    # Compute date from window index
    driver = model.met_data
    dt_window = driver.dt * steps_per_window(driver)
    sim_seconds = (w - 1) * dt_window
    date = driver._start_date + Dates.Second(round(Int, sim_seconds))
    try
        open(expanduser(pf), "w") do io
            println(io, Dates.format(Date(date), "yyyy-mm-dd"))
        end
    catch
        # Non-fatal — staging daemon is optional
    end
end

