# ---------------------------------------------------------------------------
# run!() implementations — dispatch on (grid type, buffering strategy)
#
# Each _run_loop! method implements the full forward simulation loop for a
# specific (grid, buffering) combination:
#
#   _run_loop!(model, ::LatitudeLongitudeGrid, ::SingleBuffer)
#   _run_loop!(model, ::LatitudeLongitudeGrid, ::DoubleBuffer)
#   _run_loop!(model, ::CubedSphereGrid, ::SingleBuffer)
#
# The run!() entry point dispatches to the appropriate loop.
# ---------------------------------------------------------------------------

using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid, cell_area
using ..Grids: fill_panel_halos!, allocate_cubed_sphere_field
using ..Architectures: array_type
using KernelAbstractions: get_backend, synchronize
using ..Advection: MassFluxWorkspace, allocate_massflux_workspace,
                   allocate_cs_massflux_workspace,
                   strang_split_massflux!, strang_split_massflux_ppm!,
                   compute_air_mass!,
                   compute_mass_fluxes!, compute_air_mass_panel!, compute_cm_panel!,
                   compute_cm_pressure_fixer_panel!, compute_dm_per_sub_panel!,
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
using ..Convection: AbstractConvection, TiedtkeConvection, RASConvection, convect!,
                    invalidate_ras_cfl_cache!
using ..Sources: AbstractSurfaceFlux, apply_surface_flux!, apply_surface_flux_pbl!,
                 SurfaceFlux, TimeVaryingSurfaceFlux,
                 LatLonLayout, CubedSphereLayout,
                 apply_emissions_window!, apply_emissions_window_pbl!,
                 update_time_index!, flux_data,
                 M_AIR, M_CO2, M_SF6, M_RN222
using ..Chemistry: apply_chemistry!
using ..IO: AbstractMetDriver, AbstractRawMetDriver, AbstractMassFluxMetDriver,
            total_windows, window_dt, steps_per_window, load_met_window!,
            load_cmfmc_window!, load_dtrain_window!, load_qv_window!, load_surface_fields_window!,
            load_all_window!,
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
_needs_convection(::AbstractConvection) = false
_needs_convection(::Nothing) = false
_needs_dtrain(::RASConvection) = true
_needs_dtrain(::AbstractConvection) = false
_needs_dtrain(::Nothing) = false
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
const _MAX_GPU_CLUSTER = Int32(4)

function _get_cluster_sizes_cpu(grid::LatitudeLongitudeGrid)
    rg = grid.reduced_grid
    rg === nothing && return nothing
    # Round cluster sizes down to the nearest power of 2, capped
    cs = Int32.(rg.cluster_sizes)
    @inbounds for j in eachindex(cs)
        r = cs[j]
        if r > _MAX_GPU_CLUSTER
            # Find largest power-of-2 divisor of Nx that is ≤ _MAX_GPU_CLUSTER
            # (must divide Nx evenly for the kernel to work)
            cs[j] = _MAX_GPU_CLUSTER
        end
    end
    # Verify all cluster sizes divide Nx
    Nx = grid.Nx
    @inbounds for j in eachindex(cs)
        r = cs[j]
        if r > 1 && Nx % r != 0
            cs[j] = Int32(1)  # fallback to uniform if it doesn't divide evenly
        end
    end
    return cs
end

# --- Mass conservation diagnostic ---
"""
    _compute_mass_totals(cs_tracers, grid)

Compute total tracer mass (kg) for each tracer. Returns `Dict{Symbol,Float64}`.
"""
function _compute_mass_totals(cs_tracers, grid)
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    result = Dict{Symbol,Float64}()
    for (tname, rm_t) in pairs(cs_tracers)
        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end
        result[tname] = total
    end
    return result
end

"""
    _compute_mass_totals_ll(tracers, grid)

Compute total tracer mass (kg) for lat-lon tracers (simple 3D arrays, no panels/halos).
"""
function _compute_mass_totals_ll(tracers, grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    result = Dict{Symbol,Float64}()
    for (tname, rm) in pairs(tracers)
        rm_cpu = Array(rm)
        total = 0.0
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            total += Float64(rm_cpu[i, j, k])
        end
        result[tname] = total
    end
    return result
end

"""
    _compute_mass_totals_subset(cs_tracers, grid, subset)

Like `_compute_mass_totals` but only for tracers in `subset` with nonzero values.
Used to avoid GPU→CPU copies for tracers that don't need mass fixing.
"""
function _compute_mass_totals_subset(cs_tracers, grid, subset::Dict{Symbol,Float64})
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    result = Dict{Symbol,Float64}()
    for (tname, rm_t) in pairs(cs_tracers)
        haskey(subset, tname) || continue
        subset[tname] == 0.0 && continue
        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end
        result[tname] = total
    end
    return result
end

"""
    _mass_showvalue(mass_totals, initial_mass)

Format mass diagnostics as a compact string for progress bar showvalues.
"""
function _mass_showvalue(mass_totals::Dict{Symbol,Float64},
                          initial_mass::Dict{Symbol,Float64})
    parts = String[]
    for tname in sort(collect(keys(mass_totals)))
        total = mass_totals[tname]
        if haskey(initial_mass, tname) && initial_mass[tname] != 0.0
            rel = (total - initial_mass[tname]) / abs(initial_mass[tname]) * 100
            push!(parts, @sprintf("%s:Δ=%.4e%%", tname, rel))
        end
    end
    return join(parts, "  ")
end

"""
    _apply_global_mass_fixer!(cs_tracers, grid, target_mass) → String

Scale rm globally so total tracer mass matches `target_mass` for each tracer.
Returns a compact string describing the correction magnitude (in ppm).
Typically called with pre-advection mass to correct Strang splitting drift
while preserving legitimate emission gains.
"""
function _apply_global_mass_fixer!(cs_tracers, grid, target_mass::Dict{Symbol,Float64})
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    parts = String[]
    for (tname, rm_t) in pairs(cs_tracers)
        haskey(target_mass, tname) || continue
        m0 = target_mass[tname]
        m0 == 0.0 && continue

        # Compute current total (Float64 accumulation)
        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end

        total == 0.0 && continue
        scale = m0 / total
        correction_ppm = (scale - 1.0) * 1e6

        # Apply scaling on GPU
        FT = eltype(rm_t[1])
        for p in 1:6
            rm_t[p] .*= FT(scale)
        end
        push!(parts, @sprintf("%s:%.1fppm", tname, correction_ppm))
    end
    return join(parts, " ")
end

"""
    _mass_total_f64(rm_panels, Nc, Hp, Nz) → Float64

Sum total tracer mass across all 6 panels in Float64 precision.
Used for per-stage mass balance diagnostics.
"""
function _mass_total_f64(rm_panels, Nc, Hp, Nz)
    total = 0.0
    for p in 1:6
        rm_cpu = Array(rm_panels[p])
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            total += Float64(rm_cpu[Hp+i, Hp+j, k])
        end
    end
    return total
end

# Global storage for mass balance diagnostics (populated during first N windows)
const MASS_DIAG = Dict{String, Vector{Float64}}()
const MASS_DIAG_WINDOWS = Ref(3)  # how many windows to diagnose

"""Legacy wrapper: print mass conservation via @info (used by single-buffer path)."""
function _log_mass_conservation(cs_tracers, grid, window::Int, label::String;
                                 initial_mass::Union{Nothing, Dict{Symbol,Float64}}=nothing)
    totals = _compute_mass_totals(cs_tracers, grid)
    for tname in sort(collect(keys(totals)))
        total = totals[tname]
        if initial_mass !== nothing && haskey(initial_mass, tname)
            m0 = initial_mass[tname]
            rel = m0 == 0.0 ? 0.0 : (total - m0) / abs(m0) * 100
            @info @sprintf("  MASS %s win=%d [%s]: %.6e kg (Δ=%.4e%%)", tname, window, label, total, rel)
        else
            @info @sprintf("  MASS %s win=%d [%s]: %.6e kg", tname, window, label, total)
        end
    end
end

# --- Apply advection with dispatch on scheme type ---
"""Apply cubed-sphere mass-flux advection, dispatching on advection scheme."""
function _apply_advection_cs!(rm_panels, m_panels, am, bm, cm, grid,
                               scheme::SlopesAdvection, ws)
    strang_split_massflux!(rm_panels, m_panels, am, bm, cm, grid, true, ws)
end

function _apply_advection_cs!(rm_panels, m_panels, am, bm, cm, grid,
                               scheme::PPMAdvection{ORD}, ws) where ORD
    strang_split_massflux_ppm!(rm_panels, m_panels, am, bm, cm, grid, Val(ORD), ws)
end

"""Apply lat-lon mass-flux advection, dispatching on advection scheme."""
function _apply_advection_latlon!(tracers, m, am, bm, cm, grid,
                                   scheme::SlopesAdvection, ws; cfl_limit)
    strang_split_massflux!(tracers, m, am, bm, cm, grid, true, ws; cfl_limit)
end

function _apply_advection_latlon!(tracers, m, am, bm, cm, grid,
                                   scheme::PPMAdvection{ORD}, ws; cfl_limit) where ORD
    strang_split_massflux_ppm!(tracers, m, am, bm, cm, grid, Val(ORD), ws; cfl_limit)
end

# =====================================================================
# Helper: prepare emission data for GPU-compatible injection
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

    # Get A/B coefficients from driver if available
    A_coeff = hasproperty(driver, :A_coeff) ? driver.A_coeff : nothing
    B_coeff = hasproperty(driver, :B_coeff) ? driver.B_coeff : nothing

    # Upload emission fluxes to device (static + time-varying)
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
For time-varying sources, updates the time index and re-uploads the flux
to GPU when the active snapshot changes.
"""
function _apply_emissions_latlon!(tracers, emission_data, area_j_dev, ps_dev,
                                    A_coeff, B_coeff, grid, dt_window;
                                    sim_hours::Float64=0.0, arch=nothing,
                                    delp=nothing, pblh=nothing)
    Nz = grid.Nz
    g = grid.gravity
    use_pbl = delp !== nothing && pblh !== nothing
    for (src, flux_ref, idx_ref) in emission_data
        # Handle time-varying flux updates
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
            # Fallback: uniform Δp approximation
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
For TimeVaryingSurfaceFlux{CubedSphereLayout}, uploads snapshot 1 initially.
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
For TimeVaryingSurfaceFlux{CubedSphereLayout}, updates the time index and re-uploads
panels to GPU when the active snapshot changes.

When `delp` and `pblh` are provided, distributes emissions across boundary layer levels
proportionally to air mass. Otherwise falls back to bottom-cell injection.
"""
function _apply_emissions_cs!(cs_tracers::NamedTuple, emission_data,
                                area_panels::NTuple{6}, dt_window, Nc::Int, Hp::Int;
                                sim_hours::Float64=0.0, arch=nothing,
                                delp::Union{NTuple{6}, Nothing}=nothing,
                                pblh::Union{NTuple{6}, Nothing}=nothing)
    use_pbl = delp !== nothing && pblh !== nothing
    FT = eltype(area_panels[1])

    for (src, flux_ref, idx_ref) in emission_data
        # Route emission to the correct tracer by species name
        species = src.species
        haskey(cs_tracers, species) || continue
        rm_t = cs_tracers[species]

        if src isa TimeVaryingSurfaceFlux{CubedSphereLayout}
            update_time_index!(src, sim_hours)
            if src.current_idx != idx_ref[]
                # Re-upload new snapshot to GPU
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
# Main entry point — dispatches on (grid, buffering)
# =====================================================================

"""
    run!(model::TransportModel)

Run the forward model using the met driver, sources, and buffering strategy
stored in the model. Dispatches on `(model.grid, model.buffering)` to select
the appropriate run loop.
"""
function run!(model::TransportModel)
    grid = model.grid
    buf  = model.buffering
    return _run_loop!(model, grid, buf)
end

# =====================================================================
# Lat-lon + SingleBuffer
# =====================================================================

function _run_loop!(model, grid::LatitudeLongitudeGrid{FT},
                    ::SingleBuffer) where FT
    driver  = model.met_data
    arch    = model.architecture
    sources = model.sources
    writers = model.output_writers
    AT      = array_type(arch)

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_win     = total_windows(driver)
    n_sub     = steps_per_window(driver)
    dt_sub    = FT(driver.dt)

    # Reduced-grid cluster sizes for x-advection kernel
    cs_cpu = _get_cluster_sizes_cpu(grid)

    # Allocate GPU met buffer + CPU staging
    gpu_buf = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu)
    cpu_buf = LatLonCPUBuffer(FT, Nx, Ny, Nz)

    # Prepare emission data on device
    emi_data, area_j_dev, A_coeff, B_coeff =
        _prepare_latlon_emissions(sources, grid, driver, arch)

    # Diffusion timestep = full window duration
    dt_window = FT(dt_sub * n_sub)

    # Build diffusion workspace (pre-computed tridiagonal factors on device)
    dw = _setup_bld_workspace(model.diffusion, grid, dt_window, gpu_buf.m_ref)

    # Physics: convective mass flux buffers
    has_convection = _needs_convection(model.convection)
    cmfmc_cpu = has_convection ? Array{FT}(undef, Nx, Ny, Nz + 1) : nothing
    cmfmc_gpu = has_convection ? AT(zeros(FT, Nx, Ny, Nz + 1)) : nothing

    # DTRAIN buffers for RAS convection (layer centers)
    needs_dtrain_alloc = _needs_dtrain(model.convection)
    dtrain_cpu = needs_dtrain_alloc ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    dtrain_gpu = needs_dtrain_alloc ? AT(zeros(FT, Nx, Ny, Nz)) : nothing
    ras_workspace = model.convection isa RASConvection ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    # Physics: PBL surface field buffers
    has_pbl_diff = _needs_pbl(model.diffusion)
    pbl_sfc_cpu = has_pbl_diff ? (
        pblh  = Array{FT}(undef, Nx, Ny),
        ustar = Array{FT}(undef, Nx, Ny),
        hflux = Array{FT}(undef, Nx, Ny),
        t2m   = Array{FT}(undef, Nx, Ny)) : nothing
    pbl_sfc_gpu = has_pbl_diff ? (
        pblh  = AT(zeros(FT, Nx, Ny)),
        ustar = AT(zeros(FT, Nx, Ny)),
        hflux = AT(zeros(FT, Nx, Ny)),
        t2m   = AT(zeros(FT, Nx, Ny))) : nothing
    w_scratch = has_pbl_diff ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    # Physics: pressure thickness CPU buffer (m → Δp for convection/PBL kernels)
    needs_delp = has_convection || has_pbl_diff
    delp_cpu = needs_delp ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    area_j = needs_delp ? FT[cell_area(1, j, grid) for j in 1:Ny] : nothing

    # Planet parameters for physics kernels
    planet = (has_convection || has_pbl_diff) ? load_parameters(FT).planet : nothing

    step = 0
    wall_start = time()

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (SingleBuffer, LatLon)" *
          (dw !== nothing ? " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]" : "") *
          (has_pbl_diff ? " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]" : "") *
          (has_convection ? " [convection: $(nameof(typeof(model.convection)))]" : "")

    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)

    for w in 1:n_win
        # Load met data to CPU
        load_met_window!(cpu_buf, driver, grid, w)

        # Upload to GPU
        upload!(gpu_buf, cpu_buf)

        # For raw met drivers: m_ref has Δp after upload; compute mass fluxes
        if driver isa AbstractRawMetDriver
            copyto!(gpu_buf.Δp, gpu_buf.m_ref)
            compute_air_mass!(gpu_buf.m_ref, gpu_buf.Δp, grid)
            copyto!(gpu_buf.m_dev, gpu_buf.m_ref)
            copyto!(gpu_buf.u, gpu_buf.am)
            copyto!(gpu_buf.v, gpu_buf.bm)
            compute_mass_fluxes!(gpu_buf.am, gpu_buf.bm, gpu_buf.cm,
                                  gpu_buf.u, gpu_buf.v, grid, gpu_buf.Δp, dt_sub / 2)
        elseif needs_delp
            # Preprocessed driver: compute Δp = m * g / area from air mass
            g = FT(grid.gravity)
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                delp_cpu[i, j, k] = cpu_buf.m[i, j, k] * g / area_j[j]
            end
            copyto!(gpu_buf.Δp, delp_cpu)
        end

        # First window: finalize deferred IC (uniform mixing ratio → tracer mass)
        if w == 1 && has_deferred_ic_vinterp()
            finalize_ic_vertical_interp!(model.tracers, gpu_buf.m_ref, grid)
        end

        # Load convective mass flux (optional: returns false if not in file)
        cmfmc_loaded = false
        if cmfmc_cpu !== nothing
            cmfmc_loaded = load_cmfmc_window!(cmfmc_cpu, driver, grid, w)
            if cmfmc_loaded
                copyto!(cmfmc_gpu, cmfmc_cpu)
            end
        end

        # Load DTRAIN from A3dyn if RAS convection enabled and CMFMC loaded
        dtrain_loaded = false
        if dtrain_cpu !== nothing && cmfmc_loaded
            dtrain_loaded = load_dtrain_window!(dtrain_cpu, driver, grid, w)
            if dtrain_loaded
                copyto!(dtrain_gpu, dtrain_cpu)
            end
        end

        # Load PBL surface fields (optional: returns false if not in file)
        sfc_loaded = false
        if pbl_sfc_cpu !== nothing
            sfc_loaded = load_surface_fields_window!(pbl_sfc_cpu, driver, grid, w)
            if sfc_loaded
                copyto!(pbl_sfc_gpu.pblh,  pbl_sfc_cpu.pblh)
                copyto!(pbl_sfc_gpu.ustar, pbl_sfc_cpu.ustar)
                copyto!(pbl_sfc_gpu.hflux, pbl_sfc_cpu.hflux)
                copyto!(pbl_sfc_gpu.t2m,   pbl_sfc_cpu.t2m)
            end
        end

        # Apply emissions (GPU-compatible via KA kernels)
        # Use PBL-distributed injection when surface fields available
        sim_hours = Float64((w - 1) * dt_window / 3600)
        _apply_emissions_latlon!(model.tracers, emi_data, area_j_dev,
                                  gpu_buf.ps, A_coeff, B_coeff, grid, dt_window;
                                  sim_hours, arch,
                                  delp=(sfc_loaded ? gpu_buf.Δp : nothing),
                                  pblh=(sfc_loaded && pbl_sfc_gpu !== nothing ?
                                        pbl_sfc_gpu.pblh : nothing))

        # Advection + convection sub-steps
        for sub in 1:n_sub
            step += 1
            copyto!(gpu_buf.m_dev, gpu_buf.m_ref)
            _apply_advection_latlon!(model.tracers, gpu_buf.m_dev,
                                     gpu_buf.am, gpu_buf.bm, gpu_buf.cm,
                                     grid, model.advection_scheme, gpu_buf.ws;
                                     cfl_limit=FT(0.95))
            # Convective transport (per substep for CFL stability)
            if cmfmc_loaded
                convect!(model.tracers, cmfmc_gpu, gpu_buf.Δp,
                          model.convection, grid, dt_sub, planet;
                          dtrain_panels=dtrain_loaded ? dtrain_gpu : nothing,
                          workspace=ras_workspace)
            end
        end

        # Boundary-layer vertical diffusion (implicit solve, once per window)
        _apply_bld!(model.tracers, dw)

        # Met-driven PBL diffusion (variable Kz from surface fields)
        if sfc_loaded
            diffuse_pbl!(model.tracers, gpu_buf.Δp,
                          pbl_sfc_gpu.pblh, pbl_sfc_gpu.ustar,
                          pbl_sfc_gpu.hflux, pbl_sfc_gpu.t2m,
                          w_scratch,
                          model.diffusion, grid, dt_window, planet)
        end

        # Chemistry (e.g. radioactive decay for ²²²Rn)
        apply_chemistry!(model.tracers, grid, model.chemistry, dt_window)

        # Output
        sim_time = Float64(step * dt_sub)
        _current_air_mass = gpu_buf.m_ref
        for writer in writers
            write_output!(writer, model, sim_time; air_mass=_current_air_mass)
        end

        # Progress bar
        next!(prog; showvalues=[(:day, @sprintf("%.1f", sim_time / 86400)),
                                (:rate, @sprintf("%.2f s/win", w > 1 ? (time() - wall_start) / w : 0.0))])
    end
    finish!(prog)

    wall_total = time() - wall_start
    @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                   step, wall_total, wall_total / n_win)

    for writer in writers
        finalize_output!(writer)
    end
    return model
end

# =====================================================================
# Lat-lon + DoubleBuffer
# =====================================================================

function _run_loop!(model, grid::LatitudeLongitudeGrid{FT},
                    ::DoubleBuffer) where FT
    driver  = model.met_data
    arch    = model.architecture
    sources = model.sources
    writers = model.output_writers
    AT      = array_type(arch)

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_win     = total_windows(driver)
    n_sub     = steps_per_window(driver)
    dt_sub    = FT(driver.dt)
    half_dt   = dt_sub / 2

    # Reduced-grid cluster sizes for x-advection kernel
    cs_cpu = _get_cluster_sizes_cpu(grid)

    # Allocate TWO GPU met buffers + TWO CPU staging buffers
    buf_A = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu)
    buf_B = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu)
    cpu_A = LatLonCPUBuffer(FT, Nx, Ny, Nz)
    cpu_B = LatLonCPUBuffer(FT, Nx, Ny, Nz)

    # Prepare emission data on device
    emi_data, area_j_dev, A_coeff, B_coeff =
        _prepare_latlon_emissions(sources, grid, driver, arch)

    # Diffusion timestep = full window duration
    dt_window_const = FT(dt_sub * n_sub)

    # Build diffusion workspace (pre-computed tridiagonal factors on device)
    dw = _setup_bld_workspace(model.diffusion, grid, dt_window_const, buf_A.m_ref)

    # Physics: convective mass flux buffers
    has_convection = _needs_convection(model.convection)
    cmfmc_cpu = has_convection ? Array{FT}(undef, Nx, Ny, Nz + 1) : nothing
    cmfmc_gpu = has_convection ? AT(zeros(FT, Nx, Ny, Nz + 1)) : nothing

    # DTRAIN buffers for RAS convection (layer centers)
    needs_dtrain_alloc = _needs_dtrain(model.convection)
    dtrain_cpu = needs_dtrain_alloc ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    dtrain_gpu = needs_dtrain_alloc ? AT(zeros(FT, Nx, Ny, Nz)) : nothing
    ras_workspace = model.convection isa RASConvection ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    # Physics: PBL surface field buffers
    has_pbl_diff = _needs_pbl(model.diffusion)
    pbl_sfc_cpu = has_pbl_diff ? (
        pblh  = Array{FT}(undef, Nx, Ny),
        ustar = Array{FT}(undef, Nx, Ny),
        hflux = Array{FT}(undef, Nx, Ny),
        t2m   = Array{FT}(undef, Nx, Ny)) : nothing
    pbl_sfc_gpu = has_pbl_diff ? (
        pblh  = AT(zeros(FT, Nx, Ny)),
        ustar = AT(zeros(FT, Nx, Ny)),
        hflux = AT(zeros(FT, Nx, Ny)),
        t2m   = AT(zeros(FT, Nx, Ny))) : nothing
    w_scratch = has_pbl_diff ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    # Physics: pressure thickness CPU buffer (m → Δp for convection/PBL kernels)
    needs_delp = has_convection || has_pbl_diff
    delp_cpu = needs_delp ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    area_j = needs_delp ? FT[cell_area(1, j, grid) for j in 1:Ny] : nothing

    # Planet parameters for physics kernels
    planet = (has_convection || has_pbl_diff) ? load_parameters(FT).planet : nothing

    # QV (specific humidity) for dry-air output conversion
    qv_cpu = Array{FT}(undef, Nx, Ny, Nz)
    qv_gpu = AT(zeros(FT, Nx, Ny, Nz))
    m_dry_ll = AT(zeros(FT, Nx, Ny, Nz))  # scratch for dry mass at output time

    # Helper: compute Δp from air mass on CPU and upload to gpu_buf.Δp
    function _compute_delp!(gpu_buf, cpu_buf)
        g = FT(grid.gravity)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            delp_cpu[i, j, k] = cpu_buf.m[i, j, k] * g / area_j[j]
        end
        copyto!(gpu_buf.Δp, delp_cpu)
    end

    step = 0
    wall_start = time()

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (DoubleBuffer, LatLon)" *
          (dw !== nothing ? " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]" : "") *
          (has_pbl_diff ? " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]" : "") *
          (has_convection ? " [convection: $(nameof(typeof(model.convection)))]" : "")

    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)

    # Preload first window into buffer A
    load_met_window!(cpu_A, driver, grid, 1)
    upload!(buf_A, cpu_A)
    if driver isa AbstractRawMetDriver
        copyto!(buf_A.Δp, buf_A.m_ref)
        compute_air_mass!(buf_A.m_ref, buf_A.Δp, grid)
        copyto!(buf_A.m_dev, buf_A.m_ref)
        copyto!(buf_A.u, buf_A.am)
        copyto!(buf_A.v, buf_A.bm)
        compute_mass_fluxes!(buf_A.am, buf_A.bm, buf_A.cm,
                              buf_A.u, buf_A.v, grid, buf_A.Δp, half_dt)
    elseif needs_delp
        _compute_delp!(buf_A, cpu_A)
    end

    curr, next_buf = buf_A, buf_B
    cpu_curr, cpu_next = cpu_A, cpu_B

    # First window: finalize deferred IC (uniform mixing ratio → tracer mass)
    if has_deferred_ic_vinterp()
        finalize_ic_vertical_interp!(model.tracers, buf_A.m_ref, grid)
    end

    # Mass conservation tracking
    _initial_mass_ll = _compute_mass_totals_ll(model.tracers, grid)
    for tname in sort(collect(keys(_initial_mass_ll)))
        @info @sprintf("  IC mass %s: %.6e kg", tname, _initial_mass_ll[tname])
    end
    _last_mass_ll = ""

    for w in 1:n_win
        has_next = w < n_win
        dt_window = FT(dt_sub * n_sub)

        # Start CPU preload of next window (overlaps with GPU compute)
        if has_next
            load_met_window!(cpu_next, driver, grid, w + 1)
        end

        # Load convective mass flux (optional)
        cmfmc_loaded = false
        if cmfmc_cpu !== nothing
            cmfmc_loaded = load_cmfmc_window!(cmfmc_cpu, driver, grid, w)
            if cmfmc_loaded
                copyto!(cmfmc_gpu, cmfmc_cpu)
            end
        end

        # Load DTRAIN from A3dyn if RAS convection enabled and CMFMC loaded
        dtrain_loaded = false
        if dtrain_cpu !== nothing && cmfmc_loaded
            dtrain_loaded = load_dtrain_window!(dtrain_cpu, driver, grid, w)
            if dtrain_loaded
                copyto!(dtrain_gpu, dtrain_cpu)
            end
        end

        # Load PBL surface fields (optional)
        sfc_loaded = false
        if pbl_sfc_cpu !== nothing
            sfc_loaded = load_surface_fields_window!(pbl_sfc_cpu, driver, grid, w)
            if sfc_loaded
                copyto!(pbl_sfc_gpu.pblh,  pbl_sfc_cpu.pblh)
                copyto!(pbl_sfc_gpu.ustar, pbl_sfc_cpu.ustar)
                copyto!(pbl_sfc_gpu.hflux, pbl_sfc_cpu.hflux)
                copyto!(pbl_sfc_gpu.t2m,   pbl_sfc_cpu.t2m)
            end
        end

        # Load specific humidity for dry-air correction (graceful: false if unavailable)
        _qv_status = load_qv_window!(qv_cpu, driver, grid, w)
        _qv_loaded = _qv_status !== false
        if _qv_loaded && _qv_status !== :cached
            copyto!(qv_gpu, qv_cpu)
        end

        # NOTE: No dry-air correction of DELP/am/bm for advection — wet transport
        # is used. See double-buffer CS path for rationale. Dry VMRs in output via
        # m_dry = m_wet × (1-qv) at output time only.

        # Apply emissions (GPU-compatible via KA kernels)
        # Use PBL-distributed injection when surface fields available
        sim_hours = Float64((w - 1) * dt_window / 3600)
        _apply_emissions_latlon!(model.tracers, emi_data, area_j_dev,
                                  curr.ps, A_coeff, B_coeff, grid, dt_window;
                                  sim_hours, arch,
                                  delp=(sfc_loaded ? curr.Δp : nothing),
                                  pblh=(sfc_loaded && pbl_sfc_gpu !== nothing ?
                                        pbl_sfc_gpu.pblh : nothing))

        # Advection + convection sub-steps on current buffer
        for sub in 1:n_sub
            step += 1
            copyto!(curr.m_dev, curr.m_ref)
            _apply_advection_latlon!(model.tracers, curr.m_dev,
                                     curr.am, curr.bm, curr.cm,
                                     grid, model.advection_scheme, curr.ws;
                                     cfl_limit=FT(0.95))
            # Convective transport (per substep for CFL stability)
            if cmfmc_loaded
                convect!(model.tracers, cmfmc_gpu, curr.Δp,
                          model.convection, grid, dt_sub, planet;
                          dtrain_panels=dtrain_loaded ? dtrain_gpu : nothing,
                          workspace=ras_workspace)
            end
        end

        # Boundary-layer vertical diffusion (implicit solve, once per window)
        _apply_bld!(model.tracers, dw)

        # Met-driven PBL diffusion (variable Kz from surface fields)
        if sfc_loaded
            diffuse_pbl!(model.tracers, curr.Δp,
                          pbl_sfc_gpu.pblh, pbl_sfc_gpu.ustar,
                          pbl_sfc_gpu.hflux, pbl_sfc_gpu.t2m,
                          w_scratch,
                          model.diffusion, grid, dt_window, planet)
        end

        # Chemistry (e.g. radioactive decay for ²²²Rn)
        apply_chemistry!(model.tracers, grid, model.chemistry, dt_window)

        # Mass conservation tracking
        _mass_totals = _compute_mass_totals_ll(model.tracers, grid)
        _last_mass_ll = _mass_showvalue(_mass_totals, _initial_mass_ll)

        # Upload next window (after GPU compute)
        if has_next
            upload!(next_buf, cpu_next)
            if driver isa AbstractRawMetDriver
                copyto!(next_buf.Δp, next_buf.m_ref)
                compute_air_mass!(next_buf.m_ref, next_buf.Δp, grid)
                copyto!(next_buf.m_dev, next_buf.m_ref)
                copyto!(next_buf.u, next_buf.am)
                copyto!(next_buf.v, next_buf.bm)
                compute_mass_fluxes!(next_buf.am, next_buf.bm, next_buf.cm,
                                      next_buf.u, next_buf.v, grid, next_buf.Δp, half_dt)
            elseif needs_delp
                _compute_delp!(next_buf, cpu_next)
            end
            # Swap
            curr, next_buf = next_buf, curr
            cpu_curr, cpu_next = cpu_next, cpu_curr
        end

        # Output (note: curr may have been swapped, use the buffer that was just computed)
        sim_time = Float64(step * dt_sub)
        _wet_mass = has_next ? next_buf.m_ref : curr.m_ref  # pre-swap buffer
        _current_air_mass = _wet_mass
        if _qv_loaded
            m_dry_ll .= _wet_mass .* (1 .- qv_gpu)
            _current_air_mass = m_dry_ll
        end
        for writer in writers
            write_output!(writer, model, sim_time; air_mass=_current_air_mass)
        end

        # Progress bar
        _sv_ll = Pair{Symbol,Any}[
            :day  => @sprintf("%.1f", sim_time / 86400),
            :rate => @sprintf("%.2f s/win", w > 1 ? (time() - wall_start) / w : 0.0)]
        isempty(_last_mass_ll) || push!(_sv_ll, :mass => _last_mass_ll)
        next!(prog; showvalues=_sv_ll)
    end
    finish!(prog)

    wall_total = time() - wall_start
    @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                   step, wall_total, wall_total / n_win)

    # Final mass conservation summary
    _mass_final = _compute_mass_totals_ll(model.tracers, grid)
    for tname in sort(collect(keys(_mass_final)))
        total = _mass_final[tname]
        if haskey(_initial_mass_ll, tname) && _initial_mass_ll[tname] != 0.0
            rel = (total - _initial_mass_ll[tname]) / abs(_initial_mass_ll[tname]) * 100
            @info @sprintf("  Final mass %s: %.6e kg (Δ=%.4e%%)", tname, total, rel)
        else
            @info @sprintf("  Final mass %s: %.6e kg", tname, total)
        end
    end

    for writer in writers
        finalize_output!(writer)
    end
    return model
end

# =====================================================================
# CubedSphere + SingleBuffer
# =====================================================================

function _run_loop!(model, grid::CubedSphereGrid{FT},
                    ::SingleBuffer) where FT
    driver  = model.met_data
    arch    = model.architecture
    AT      = array_type(arch)
    sources = model.sources
    writers = model.output_writers

    Nc = grid.Nc
    Nz = hasproperty(grid, :Nz) ? grid.Nz : driver.Nz
    Hp = hasproperty(driver, :Hp) ? driver.Hp : 3
    n_win     = total_windows(driver)
    n_sub     = steps_per_window(driver)
    dt_sub    = FT(driver.dt)

    # Build geometry cache + workspace
    ref_panel = AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz))
    gc = build_geometry_cache(grid, ref_panel)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)

    # Allocate per-tracer panel arrays (each tracer gets its own rm_panels)
    tracer_names = keys(model.tracers)
    n_tracers = length(tracer_names)
    cs_tracers = NamedTuple{tracer_names}(
        ntuple(_ -> allocate_cubed_sphere_field(grid, Nz), n_tracers)
    )

    # Apply deferred initial conditions (CS tracers are now properly allocated)
    pending_ic = get_pending_ic()
    if !isempty(pending_ic.entries)
        apply_pending_ic!(cs_tracers, pending_ic, grid)
    end

    # Shared air mass (same meteorology for all tracers)
    m_panels      = allocate_cubed_sphere_field(grid, Nz)
    m_ref_panels  = allocate_cubed_sphere_field(grid, Nz)   # reference air mass for sub-step reset

    # GPU met buffers
    delp_gpu = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    am_gpu   = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    bm_gpu   = ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6)
    cm_gpu   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz + 1)), 6)

    # CPU staging
    cpu_buf = CubedSphereCPUBuffer(FT, Nc, Nz, Hp)

    # GPU met buffer wrapper for upload!
    cs_gpu_buf = CubedSphereMetBuffer(delp_gpu, am_gpu, bm_gpu, cm_gpu)

    # Pre-upload emission flux panels to device
    emi_data = _prepare_cs_emissions(sources, grid, arch)

    # Diffusion timestep = full window duration
    dt_window = FT(dt_sub * n_sub)

    # Build diffusion workspace (pre-computed tridiagonal factors on device)
    dw = _setup_bld_workspace(model.diffusion, grid, dt_window, ref_panel)

    # Allocate convective mass flux panels (filled by met driver when CMFMC data is available)
    has_convection = _needs_convection(model.convection)
    cmfmc_gpu = if has_convection
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1)), 6)
    else
        nothing
    end
    cmfmc_cpu = if has_convection
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), 6)
    else
        nothing
    end

    # Allocate DTRAIN panels for RAS convection (layer centers, Nz levels)
    needs_dtrain_alloc = _needs_dtrain(model.convection)
    dtrain_gpu = if needs_dtrain_alloc
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end
    dtrain_cpu = if needs_dtrain_alloc
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    else
        nothing
    end
    # QV (specific humidity) for dry-air transport and output
    qv_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    qv_gpu = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    qv_loaded = false
    # RAS workspace for updraft concentration tracking (q_cloud)
    ras_workspace = if model.convection isa RASConvection
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end

    # PBL diffusion: allocate surface field panels + workspace
    has_pbl_diff = _needs_pbl(model.diffusion)
    pbl_sfc_cpu = if has_pbl_diff
        (pblh  = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         ustar = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         hflux = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         t2m   = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6))
    else
        nothing
    end
    pbl_sfc_gpu = if has_pbl_diff
        (pblh  = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         ustar = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         hflux = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         t2m   = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6))
    else
        nothing
    end
    w_scratch_panels = if has_pbl_diff
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end

    # Met 2D output buffers (tropopause pressure, surface pressure for output writer)
    # Allocated unconditionally — cheap (6 × Nc² × 4 bytes each)
    troph_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)
    ps_cpu    = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)
    troph_loaded = false

    # Planet parameters for physics kernels (PBL diffusion, convection)
    planet = load_parameters(FT).planet

    step = 0
    wall_start = time()
    t_io      = 0.0   # met load + H2D upload
    t_compute = 0.0   # air mass, emissions, advection (GPU)
    t_output  = 0.0   # diagnostics + NetCDF write

    # Strang splitting half-step: scale mass fluxes from kg/s → kg per half-sub-step
    half_dt = FT(dt_sub / 2)

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (SingleBuffer, C$Nc)" *
          (dw !== nothing ? " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]" : "") *
          (has_pbl_diff ? " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]" : "") *
          (has_convection ? " [convection: $(nameof(typeof(model.convection)))]" : "")

    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)
    _initial_mass = Dict{Symbol,Float64}()

    for w in 1:n_win
        # ── I/O: load met from disk → CPU → GPU ───────────────────────
        t0 = time()
        load_met_window!(cpu_buf, driver, grid, w)
        upload!(cs_gpu_buf, cpu_buf)

        # Load CMFMC from A3mstE if convection enabled and data available
        # Returns :cached when data unchanged (skip GPU upload), true/false otherwise
        cmfmc_loaded = false
        if cmfmc_cpu !== nothing
            cmfmc_status = load_cmfmc_window!(cmfmc_cpu, driver, grid, w)
            cmfmc_loaded = cmfmc_status !== false
            if cmfmc_loaded && cmfmc_status !== :cached
                for p in 1:6
                    copyto!(cmfmc_gpu[p], cmfmc_cpu[p])
                end
            end
        end

        # Load DTRAIN from A3dyn if RAS convection enabled and CMFMC loaded
        dtrain_loaded = false
        if dtrain_cpu !== nothing && cmfmc_loaded
            dtrain_status = load_dtrain_window!(dtrain_cpu, driver, grid, w)
            dtrain_loaded = dtrain_status !== false
            if dtrain_loaded && dtrain_status !== :cached
                for p in 1:6
                    copyto!(dtrain_gpu[p], dtrain_cpu[p])
                end
            end
        end

        # Load QV (specific humidity) for dry-air transport
        qv_status = load_qv_window!(qv_cpu, driver, grid, w)
        qv_loaded = qv_status !== false
        if qv_loaded && qv_status !== :cached
            for p in 1:6
                copyto!(qv_gpu[p], qv_cpu[p])
            end
            fill_panel_halos!(qv_gpu, grid)
        end

        # Load surface fields from A1 if PBL diffusion enabled
        sfc_loaded = false
        if pbl_sfc_cpu !== nothing
            sfc_loaded = load_surface_fields_window!(pbl_sfc_cpu, driver, grid, w;
                                                      troph_panels=troph_cpu,
                                                      ps_panels=ps_cpu)
            troph_loaded = sfc_loaded
            if sfc_loaded
                for p in 1:6
                    copyto!(pbl_sfc_gpu.pblh[p],  pbl_sfc_cpu.pblh[p])
                    copyto!(pbl_sfc_gpu.ustar[p], pbl_sfc_cpu.ustar[p])
                    copyto!(pbl_sfc_gpu.hflux[p], pbl_sfc_cpu.hflux[p])
                    copyto!(pbl_sfc_gpu.t2m[p],   pbl_sfc_cpu.t2m[p])
                end
            end
        else
            # Even without PBL diffusion, try to load tropopause from A1
            _dummy_sfc = (pblh=troph_cpu, ustar=troph_cpu, hflux=troph_cpu, t2m=troph_cpu)
            troph_loaded = load_surface_fields_window!(
                _dummy_sfc, driver, grid, w; troph_panels=troph_cpu,
                ps_panels=ps_cpu)
        end

        # Compute surface pressure from DELP if PS was not loaded from binary
        _ps_from_bin = sfc_loaded && !iszero(ps_cpu[1][Hp + 1, Hp + 1])
        if !_ps_from_bin
            for p in 1:6
                fill!(ps_cpu[p], zero(FT))
                delp_p = cpu_buf.delp[p]
                @inbounds for k in 1:Nz
                    for jj in 1:Nc, ii in 1:Nc
                        ps_cpu[p][Hp + ii, Hp + jj] += delp_p[Hp + ii, Hp + jj, k]
                    end
                end
            end
        end
        t_io += time() - t0

        # ── GPU compute: air mass + emissions + advection ─────────────
        t0 = time()

        # Scale mass fluxes from kg/s → kg per half-sub-step
        for p in 1:6
            am_gpu[p] .*= half_dt
            bm_gpu[p] .*= half_dt
        end

        # NOTE: No dry-air correction of DELP/am/bm for advection — wet transport
        # is used. See double-buffer path for rationale. Dry VMRs in output via
        # m_dry = m_wet × (1-qv) at output time only.

        for p in 1:6
            compute_air_mass_panel!(m_panels[p], delp_gpu[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # First window: finalize deferred IC vertical interpolation + mass conversion
        if w == 1 && has_deferred_ic_vinterp()
            finalize_ic_vertical_interp!(cs_tracers, m_panels, delp_gpu, grid)
        end

        # Save reference air mass from DELP for this window.
        for p in 1:6
            copyto!(m_ref_panels[p], m_panels[p])
        end

        # Time-0 IC snapshot: log mass conservation + write output before any physics
        if w == 1
            _initial_mass = Dict{Symbol,Float64}()
            for (tname, rm_t) in pairs(cs_tracers)
                total = 0.0
                for p in 1:6
                    rm_cpu = Array(rm_t[p])
                    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                        total += Float64(rm_cpu[Hp+i, Hp+j, k])
                    end
                end
                _initial_mass[tname] = total
            end
            _log_mass_conservation(cs_tracers, grid, 0, "IC")
            _met_ic = (; ps=ps_cpu,
                         mass_flux_x=am_gpu, mass_flux_y=bm_gpu,
                         mf_scale=half_dt, dt_window=dt_window)
            if sfc_loaded && has_pbl_diff
                _met_ic = merge(_met_ic, (; pblh=pbl_sfc_cpu.pblh))
            end
            if troph_loaded
                _met_ic = merge(_met_ic, (; troph=troph_cpu))
            end
            _ic_mass = m_ref_panels
            if qv_loaded
                for p in 1:6
                    copyto!(m_panels[p], m_ref_panels[p])
                    apply_dry_delp_panel!(m_panels[p], qv_gpu[p], Nc, Nz, Hp)
                end
                synchronize(get_backend(m_panels[1]))
                _ic_mass = m_panels
            end
            for writer in writers
                write_output!(writer, model, 0.0;
                              air_mass=_ic_mass, tracers=cs_tracers,
                              met_fields=_met_ic)
            end
        end

        for p in 1:6
            compute_cm_panel!(cm_gpu[p], am_gpu[p], bm_gpu[p], gc.bt, Nc, Nz)
        end

        # CFL diagnostic (first window + every 24th)
        if w == 1 || w % 24 == 0
            cfl_x = maximum(max_cfl_x_cs(am_gpu[p], m_ref_panels[p], ws.cfl_x, Hp) for p in 1:6)
            cfl_y = maximum(max_cfl_y_cs(bm_gpu[p], m_ref_panels[p], ws.cfl_y, Hp) for p in 1:6)
            @info @sprintf("  CFL diag (win %d): max_x=%.3f max_y=%.3f", w, cfl_x, cfl_y)
        end

        _sim_hrs = Float64((w - 1) * dt_window) / 3600.0
        _apply_emissions_cs!(cs_tracers, emi_data, gc.area, dt_window, Nc, Hp;
                              sim_hours=_sim_hrs, arch=model.architecture,
                              delp=delp_gpu,
                              pblh=(pbl_sfc_gpu !== nothing ? pbl_sfc_gpu.pblh : nothing))

        for _ in 1:n_sub
            step += 1
            # Advect each tracer independently (air mass reset per tracer)
            for (_, rm_t) in pairs(cs_tracers)
                for p in 1:6
                    copyto!(m_panels[p], m_ref_panels[p])
                end
                _apply_advection_cs!(rm_t, m_panels,
                                     am_gpu, bm_gpu, cm_gpu,
                                     grid, model.advection_scheme, ws)
                # Mass fixer: preserve q = rm/m across air mass reset
                for p in 1:6
                    apply_mass_fixer!(rm_t[p], m_ref_panels[p], m_panels[p], Nc, Nz, Hp)
                end
            end
            # Convective transport per tracer (per substep for CFL stability)
            if cmfmc_loaded
                for (_, rm_t) in pairs(cs_tracers)
                    convect!(rm_t, m_ref_panels, cmfmc_gpu, delp_gpu,
                              model.convection, grid, dt_sub, planet;
                              dtrain_panels=dtrain_loaded ? dtrain_gpu : nothing,
                              workspace=ras_workspace)
                end
            end
        end
        # Boundary-layer vertical diffusion per tracer (implicit solve, once per window)
        for (_, rm_t) in pairs(cs_tracers)
            _apply_bld_cs!(rm_t, m_ref_panels, dw, Nc, Nz, Hp)
        end
        # Met-driven PBL diffusion per tracer (variable Kz from surface fields)
        if sfc_loaded
            for (_, rm_t) in pairs(cs_tracers)
                diffuse_pbl!(rm_t, m_ref_panels, delp_gpu,
                              pbl_sfc_gpu.pblh, pbl_sfc_gpu.ustar,
                              pbl_sfc_gpu.hflux, pbl_sfc_gpu.t2m,
                              w_scratch_panels,
                              model.diffusion, grid, dt_window, planet)
            end
        end
        # Chemistry (e.g. radioactive decay for ²²²Rn)
        apply_chemistry!(cs_tracers, grid, model.chemistry, dt_window)
        t_compute += time() - t0

        # Mass conservation tracker (first 48 windows = 2 days, then every 24th)
        if w <= 48 || w % 24 == 0
            _log_mass_conservation(cs_tracers, grid, w, "post-physics";
                                    initial_mass=_initial_mass)
        end

        # ── Output: diagnostics + regrid + NetCDF ─────────────────────
        t0 = time()
        sim_time = Float64(step * dt_sub)
        _current_air_mass = m_ref_panels
        if qv_loaded
            for p in 1:6
                copyto!(m_panels[p], m_ref_panels[p])
                apply_dry_delp_panel!(m_panels[p], qv_gpu[p], Nc, Nz, Hp)
            end
            synchronize(get_backend(m_panels[1]))
            _current_air_mass = m_panels
        end
        # Build met_fields: 2D met fields + mass fluxes for flux diagnostics
        # Mass fluxes am_gpu are scaled by half_dt; include scale factor for unscaling
        _met_base = (; ps=ps_cpu,
                       mass_flux_x=am_gpu, mass_flux_y=bm_gpu,
                       mf_scale=half_dt, dt_window=dt_window)
        _met_2d = if sfc_loaded && troph_loaded
            merge(_met_base, (; pblh=pbl_sfc_cpu.pblh, troph=troph_cpu))
        elseif sfc_loaded
            merge(_met_base, (; pblh=pbl_sfc_cpu.pblh))
        elseif troph_loaded
            merge(_met_base, (; troph=troph_cpu))
        else
            _met_base
        end
        if qv_loaded
            _met_2d = merge(_met_2d, (; qv=qv_cpu))
        end
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=_current_air_mass, tracers=cs_tracers,
                          met_fields=_met_2d)
        end
        t_output += time() - t0

        # ── Progress ───────────────────────────────────────────────────
        # Progress bar with timing breakdown
        next!(prog; showvalues=[
            (:day, div(w - 1, 24) + 1),
            (:IO,  @sprintf("%.2f s/win", t_io / w)),
            (:GPU, @sprintf("%.2f s/win", t_compute / w)),
            (:Out, @sprintf("%.2f s/win", t_output / w))])
    end
    finish!(prog)

    wall_total = time() - wall_start
    @info @sprintf(
        "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
        step, wall_total, t_io / n_win, t_compute / n_win, t_output / n_win)

    for writer in writers
        finalize_output!(writer)
    end
    return model
end

# =====================================================================
# CubedSphere + DoubleBuffer
# Overlaps disk I/O (Threads.@spawn) with PCIe upload + GPU compute.
# Requires ≥ 2 Julia threads: julia --threads=2 ...
# =====================================================================

function _run_loop!(model, grid::CubedSphereGrid{FT},
                    ::DoubleBuffer) where FT
    driver  = model.met_data
    arch    = model.architecture
    sources = model.sources
    writers = model.output_writers

    Nc = grid.Nc
    Nz = hasproperty(grid, :Nz) ? grid.Nz : driver.Nz
    Hp = hasproperty(driver, :Hp) ? driver.Hp : 3
    n_win  = total_windows(driver)
    n_sub  = steps_per_window(driver)
    dt_sub = FT(driver.dt)

    if Threads.nthreads() < 2
        @warn "CubedSphere DoubleBuffer needs ≥ 2 Julia threads for disk/GPU overlap. " *
              "Relaunch with: julia --threads=2 --project=. <script>"
    end

    # Build geometry cache + workspace
    AT = array_type(arch)
    ref_panel = AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz))
    gc = build_geometry_cache(grid, ref_panel)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)

    # Per-tracer panel arrays (each tracer gets its own rm_panels)
    tracer_names = keys(model.tracers)
    n_tracers = length(tracer_names)
    cs_tracers = NamedTuple{tracer_names}(
        ntuple(_ -> allocate_cubed_sphere_field(grid, Nz), n_tracers)
    )

    # Apply deferred initial conditions (CS tracers are now properly allocated)
    pending_ic = get_pending_ic()
    if !isempty(pending_ic.entries)
        apply_pending_ic!(cs_tracers, pending_ic, grid)
    end

    # Shared air mass (same meteorology for all tracers — not double-buffered)
    m_panels          = allocate_cubed_sphere_field(grid, Nz)
    m_ref_panels      = allocate_cubed_sphere_field(grid, Nz)   # reference air mass for sub-step reset
    dm_per_sub_panels = allocate_cubed_sphere_field(grid, Nz)   # pressure-fixer: Δm per sub-step

    # TWO GPU met buffers (A = current, B = next)
    gpu_A = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp)
    gpu_B = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp)

    # TWO CPU staging buffers
    cpu_A = CubedSphereCPUBuffer(FT, Nc, Nz, Hp)
    cpu_B = CubedSphereCPUBuffer(FT, Nc, Nz, Hp)

    # Pre-upload emission flux panels to device
    emi_data = _prepare_cs_emissions(sources, grid, arch)

    # Diffusion timestep = full window duration
    dt_window = FT(dt_sub * n_sub)

    # Build diffusion workspace (pre-computed tridiagonal factors on device)
    dw = _setup_bld_workspace(model.diffusion, grid, dt_window, ref_panel)

    # Allocate convective mass flux panels (filled by met driver when CMFMC data is available)
    has_convection = _needs_convection(model.convection)
    cmfmc_gpu = if has_convection
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1)), 6)
    else
        nothing
    end
    cmfmc_cpu = if has_convection
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), 6)
    else
        nothing
    end

    # Allocate DTRAIN panels for RAS convection (layer centers, Nz levels)
    needs_dtrain_alloc = _needs_dtrain(model.convection)
    dtrain_gpu = if needs_dtrain_alloc
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end
    dtrain_cpu = if needs_dtrain_alloc
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    else
        nothing
    end
    # QV (specific humidity) for dry-air transport and output
    qv_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    qv_gpu = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    qv_loaded = false
    # RAS workspace for updraft concentration tracking (q_cloud)
    ras_workspace = if model.convection isa RASConvection
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end

    # PBL diffusion: allocate surface field panels + workspace
    has_pbl_diff = _needs_pbl(model.diffusion)
    pbl_sfc_cpu = if has_pbl_diff
        (pblh  = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         ustar = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         hflux = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
         t2m   = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6))
    else
        nothing
    end
    pbl_sfc_gpu = if has_pbl_diff
        (pblh  = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         ustar = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         hflux = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
         t2m   = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6))
    else
        nothing
    end
    w_scratch_panels = if has_pbl_diff
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    else
        nothing
    end

    # Met 2D output buffers (tropopause pressure, surface pressure for output writer)
    troph_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)
    ps_cpu    = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)
    troph_loaded = false

    # Planet parameters for physics kernels (PBL diffusion, convection)
    planet = load_parameters(FT).planet

    # Pre-load window 1 synchronously (mass fluxes + CMFMC + surface + DTRAIN)
    _sfc_for_load = has_pbl_diff ? pbl_sfc_cpu :
        (pblh=troph_cpu, ustar=troph_cpu, hflux=troph_cpu, t2m=troph_cpu)
    io_status_1 = load_all_window!(cpu_A, cmfmc_cpu, dtrain_cpu, _sfc_for_load, troph_cpu,
                                    driver, grid, 1;
                                    needs_cmfmc=has_convection,
                                    needs_dtrain=needs_dtrain_alloc,
                                    needs_sfc=true,
                                    needs_qv=true, qv_cpu=qv_cpu,
                                    ps_panels=ps_cpu)

    curr_cpu, next_cpu = cpu_A, cpu_B
    curr_gpu, next_gpu = gpu_A, gpu_B

    step = 0
    wall_start = time()
    t_io      = 0.0   # upload + disk-wait time
    t_compute = 0.0   # GPU compute
    t_output  = 0.0   # diagnostics + NetCDF
    load_task = nothing
    io_status = io_status_1

    # Strang splitting half-step: scale mass fluxes from kg/s → kg per half-sub-step
    half_dt = FT(dt_sub / 2)

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (DoubleBuffer, C$Nc)" *
          (dw !== nothing ? " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]" : "") *
          (has_pbl_diff ? " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]" : "") *
          (has_convection ? " [convection: $(nameof(typeof(model.convection)))]" : "")

    prog = Progress(n_win; desc="Simulation ", showspeed=true, barlen=40)
    _initial_mass = Dict{Symbol,Float64}()
    _last_cfl  = ""
    _last_mass = ""
    _last_fix  = ""

    for w in 1:n_win
        has_next = w < n_win

        t0 = time()

        # Upload current CPU buffer → GPU (main thread, PCIe DMA)
        # All shared-buffer reads (cmfmc_cpu, dtrain_cpu, qv_cpu) happen BEFORE
        # the async spawn below to prevent race conditions.
        upload!(curr_gpu, curr_cpu)

        # Upload CMFMC if freshly loaded (not :cached)
        cmfmc_loaded = io_status.cmfmc !== false
        if cmfmc_loaded && io_status.cmfmc !== :cached && cmfmc_gpu !== nothing
            for p in 1:6
                copyto!(cmfmc_gpu[p], cmfmc_cpu[p])
            end
        end

        # Upload DTRAIN if freshly loaded
        dtrain_loaded = io_status.dtrain !== false
        if dtrain_loaded && io_status.dtrain !== :cached && dtrain_gpu !== nothing
            for p in 1:6
                copyto!(dtrain_gpu[p], dtrain_cpu[p])
            end
        end

        # Invalidate RAS CFL cache when CMFMC or DTRAIN data changed
        if (cmfmc_loaded && io_status.cmfmc !== :cached) ||
           (dtrain_loaded && io_status.dtrain !== :cached)
            invalidate_ras_cfl_cache!()
        end

        # QV: upload to GPU + halo exchange for dry-air transport
        qv_loaded = io_status.qv !== false
        if qv_loaded && io_status.qv !== :cached
            for p in 1:6
                copyto!(qv_gpu[p], qv_cpu[p])
            end
            fill_panel_halos!(qv_gpu, grid)
        end

        # ── Spawn async disk read for next window (ALL data) ──────────
        # Spawned AFTER all shared-buffer reads above to prevent race conditions:
        # the async task writes to qv_cpu, cmfmc_cpu, dtrain_cpu, ps_cpu which
        # must not be overwritten while the main thread is still reading them.
        if has_next
            load_task = Threads.@spawn load_all_window!(
                next_cpu, cmfmc_cpu, dtrain_cpu, _sfc_for_load, troph_cpu,
                driver, grid, w + 1;
                needs_cmfmc=has_convection,
                needs_dtrain=needs_dtrain_alloc,
                needs_sfc=true,
                needs_qv=true, qv_cpu=qv_cpu,
                ps_panels=ps_cpu)
        end

        # Upload surface fields if loaded
        sfc_loaded = io_status.sfc !== false
        troph_loaded = sfc_loaded
        if sfc_loaded && has_pbl_diff
            for p in 1:6
                copyto!(pbl_sfc_gpu.pblh[p],  pbl_sfc_cpu.pblh[p])
                copyto!(pbl_sfc_gpu.ustar[p], pbl_sfc_cpu.ustar[p])
                copyto!(pbl_sfc_gpu.hflux[p], pbl_sfc_cpu.hflux[p])
                copyto!(pbl_sfc_gpu.t2m[p],   pbl_sfc_cpu.t2m[p])
            end
        end

        # Compute surface pressure from DELP if PS was not loaded from binary.
        # v2+ A1 binaries include PS directly; v1 binaries and NetCDF fallbacks don't.
        if !(sfc_loaded && !iszero(ps_cpu[1][Hp + 1, Hp + 1]))
            for p in 1:6
                fill!(ps_cpu[p], zero(FT))
                delp_p = curr_cpu.delp[p]
                @inbounds for k in 1:Nz
                    for jj in 1:Nc, ii in 1:Nc
                        ps_cpu[p][Hp + ii, Hp + jj] += delp_p[Hp + ii, Hp + jj, k]
                    end
                end
            end
        end
        # ── Fetch next-window data for pressure fixer (before GPU compute) ──
        # Wait for async load of window w+1 so we can use DELP_next.
        # Binary IO is ~0.06 s/win — typically already complete by now.
        fetched_early = false
        if has_next && load_task !== nothing
            io_status = fetch(load_task)
            fetched_early = true
            for p in 1:6
                copyto!(next_gpu.delp[p], next_cpu.delp[p])
            end
        end

        t_io += time() - t0

        # ── GPU compute ───────────────────────────────────────────────
        t0 = time()

        # Scale mass fluxes from kg/s → kg per half-sub-step
        for p in 1:6
            curr_gpu.am[p] .*= half_dt
            curr_gpu.bm[p] .*= half_dt
        end

        # NOTE: No dry-air correction of DELP/am/bm for advection.
        # Wet (total moist air) transport is used because:
        #   (1) The tracer flux q×F is identical in wet and dry bases
        #       (the (1-qv) factors cancel: q_dry × F_dry / m_dry = q_dry × F_wet / m_wet)
        #   (2) Wet mass continuity holds (∂DELP/∂t + ∇·F_wet = 0), so
        #       m_evolved ≈ m_ref → mass fixer applies small corrections only
        #   (3) Naive dry correction (multiply by 1-qv) violates dry continuity
        #       by the moisture flux divergence term, creating ~1% per-cell errors
        #       that the mass fixer amplifies into oscillating tracer gradients
        # Dry VMR output is achieved by dividing rm by m_dry at output time only.
        # Ref: GCHP (Eastham+ 2018), TM5 (Huijnen+ 2010)

        for p in 1:6
            compute_air_mass_panel!(m_panels[p], curr_gpu.delp[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # First window: finalize deferred IC vertical interpolation + mass conversion
        if w == 1 && has_deferred_ic_vinterp()
            finalize_ic_vertical_interp!(cs_tracers, m_panels, curr_gpu.delp, grid)
        end

        # Save reference air mass from DELP for this window.
        for p in 1:6
            copyto!(m_ref_panels[p], m_panels[p])
        end

        # Time-0 IC snapshot: log initial mass + write output before any physics
        if w == 1
            _initial_mass = _compute_mass_totals(cs_tracers, grid)
            for tname in sort(collect(keys(_initial_mass)))
                @info @sprintf("  IC mass %s: %.6e kg", tname, _initial_mass[tname])
            end
            _met_ic = (; ps=ps_cpu,
                         mass_flux_x=curr_gpu.am, mass_flux_y=curr_gpu.bm,
                         mf_scale=half_dt, dt_window=dt_window)
            if sfc_loaded && has_pbl_diff
                _met_ic = merge(_met_ic, (; pblh=pbl_sfc_cpu.pblh))
            end
            if troph_loaded
                _met_ic = merge(_met_ic, (; troph=troph_cpu))
            end
            # Dry mass for IC output (same pattern as end-of-window output)
            _ic_mass = m_ref_panels
            if qv_loaded
                for p in 1:6
                    copyto!(m_panels[p], m_ref_panels[p])
                    apply_dry_delp_panel!(m_panels[p], qv_gpu[p], Nc, Nz, Hp)
                end
                synchronize(get_backend(m_panels[1]))
                _ic_mass = m_panels
            end
            for writer in writers
                write_output!(writer, model, 0.0;
                              air_mass=_ic_mass, tracers=cs_tracers,
                              met_fields=_met_ic)
            end
        end

        # Compute cm: pressure-fixer (with DELP_next) or bt-only (last window).
        # The pressure fixer incorporates the DELP tendency directly into cm,
        # aligning vertical mass flux with dm_per_sub for minimal mass fixer corrections.
        for p in 1:6
            if has_next
                compute_cm_pressure_fixer_panel!(curr_gpu.cm[p], curr_gpu.am[p], curr_gpu.bm[p],
                    gc.bt, curr_gpu.delp[p], next_gpu.delp[p], gc.area[p], gc.gravity,
                    n_sub, Nc, Nz, Hp)
                compute_dm_per_sub_panel!(dm_per_sub_panels[p], curr_gpu.delp[p], next_gpu.delp[p],
                    gc.area[p], gc.gravity, n_sub, Nc, Nz, Hp)
            else
                compute_cm_panel!(curr_gpu.cm[p], curr_gpu.am[p], curr_gpu.bm[p], gc.bt, Nc, Nz)
                fill!(dm_per_sub_panels[p], zero(FT))
            end
        end

        # CFL diagnostic (first window + every 24th) — shown in progress bar
        if w == 1 || w % 24 == 0
            cfl_x = maximum(max_cfl_x_cs(curr_gpu.am[p], m_ref_panels[p], ws.cfl_x, Hp) for p in 1:6)
            cfl_y = maximum(max_cfl_y_cs(curr_gpu.bm[p], m_ref_panels[p], ws.cfl_y, Hp) for p in 1:6)
            _last_cfl = @sprintf("x=%.3f y=%.3f", cfl_x, cfl_y)
        end

        _sim_hrs = Float64((w - 1) * dt_window) / 3600.0

        # ── MASS BALANCE DIAGNOSTICS ──────────────────────────────────
        _do_diag = w <= MASS_DIAG_WINDOWS[]
        _diag_tname = first(keys(cs_tracers))
        _diag_rm() = _do_diag ? _mass_total_f64(cs_tracers[_diag_tname], Nc, Hp, Nz) : 0.0

        _m_start = _diag_rm()

        _apply_emissions_cs!(cs_tracers, emi_data, gc.area, dt_window, Nc, Hp;
                              sim_hours=_sim_hrs, arch=model.architecture,
                              delp=curr_gpu.delp,
                              pblh=(pbl_sfc_gpu !== nothing ? pbl_sfc_gpu.pblh : nothing))

        _m_after_emit = _diag_rm()

        # Snapshot mass after emissions / before advection — target for mass fixer.
        # Only computes IC tracers (those with nonzero initial_mass).
        _pre_adv_mass = if !isempty(_initial_mass)
            _compute_mass_totals_subset(cs_tracers, grid, _initial_mass)
        else
            _initial_mass
        end

        _m_after_adv_total = _m_after_emit  # will be updated after all sub-steps
        for sub_idx in 1:n_sub
            step += 1
            _m_before_sub = _do_diag && sub_idx <= 2 ? _diag_rm() : 0.0

            # Advect each tracer independently (m reset per tracer; all produce same m_evolved)
            for (_, rm_t) in pairs(cs_tracers)
                for p in 1:6
                    copyto!(m_panels[p], m_ref_panels[p])
                end
                _apply_advection_cs!(rm_t, m_panels,
                                     curr_gpu.am, curr_gpu.bm, curr_gpu.cm,
                                     grid, model.advection_scheme, ws)
            end
            # m_panels now = m_evolved (identical for all tracers: same am/bm/cm, same m_ref start)

            _m_after_adv = _do_diag && sub_idx <= 2 ? _diag_rm() : 0.0

            # Per-cell mass fixer: rm *= m_ref / m_evolved
            # This is the CS equivalent of the LL path's c = rm/m_evolved → rm = m_ref*c
            # conversion. Without it, rm drifts from the correct mixing ratio at each
            # sub-step, compounding into exponential tracer gradient growth.
            for (_, rm_t) in pairs(cs_tracers)
                for p in 1:6
                    apply_mass_fixer!(rm_t[p], m_ref_panels[p], m_panels[p],
                                      Nc, Nz, Hp)
                end
            end

            # Advance m_ref along prescribed pressure trajectory
            for p in 1:6
                m_ref_panels[p] .+= dm_per_sub_panels[p]
            end

            # Convective transport per tracer (per substep for CFL stability)
            if cmfmc_loaded
                for (_, rm_t) in pairs(cs_tracers)
                    convect!(rm_t, m_ref_panels, cmfmc_gpu, curr_gpu.delp,
                              model.convection, grid, dt_sub, planet;
                              dtrain_panels=dtrain_loaded ? dtrain_gpu : nothing,
                              workspace=ras_workspace)
                end
            end

            _m_after_conv = _do_diag && sub_idx <= 2 ? _diag_rm() : 0.0

            if _do_diag && sub_idx <= 2
                @info @sprintf("  [W%d S%d] %s: adv Δ=%+.6e (%.2f ppm)  conv Δ=%+.6e (%.2f ppm)",
                    w, sub_idx, _diag_tname,
                    _m_after_adv - _m_before_sub,
                    (_m_after_adv - _m_before_sub) / max(abs(_m_before_sub), 1e-30) * 1e6,
                    _m_after_conv - _m_after_adv,
                    (_m_after_conv - _m_after_adv) / max(abs(_m_after_adv), 1e-30) * 1e6)
            end
        end
        _m_after_adv_total = _diag_rm()

        # Boundary-layer vertical diffusion per tracer (implicit solve, once per window)
        for (_, rm_t) in pairs(cs_tracers)
            _apply_bld_cs!(rm_t, m_ref_panels, dw, Nc, Nz, Hp)
        end
        _m_after_bld = _diag_rm()

        # Met-driven PBL diffusion per tracer (variable Kz from surface fields)
        if sfc_loaded && has_pbl_diff
            for (_, rm_t) in pairs(cs_tracers)
                diffuse_pbl!(rm_t, m_ref_panels, curr_gpu.delp,
                              pbl_sfc_gpu.pblh, pbl_sfc_gpu.ustar,
                              pbl_sfc_gpu.hflux, pbl_sfc_gpu.t2m,
                              w_scratch_panels,
                              model.diffusion, grid, dt_window, planet)
            end
        end
        _m_after_pbl = _diag_rm()

        # Chemistry (e.g. radioactive decay for ²²²Rn)
        apply_chemistry!(cs_tracers, grid, model.chemistry, dt_window)
        _m_after_chem = _diag_rm()

        t_compute += time() - t0

        if _do_diag
            _emit_delta  = _m_after_emit - _m_start
            _adv_delta   = _m_after_adv_total - _m_after_emit
            _bld_delta   = _m_after_bld - _m_after_adv_total
            _pbl_delta   = _m_after_pbl - _m_after_bld
            _chem_delta  = _m_after_chem - _m_after_pbl
            _total_delta = _m_after_chem - _m_start
            _ppm(d) = d / max(abs(_m_start), 1e-30) * 1e6
            @info string(
                @sprintf("[W%d] %s MASS BUDGET (kg):", w, _diag_tname),
                @sprintf("\n  start      = %.10e", _m_start),
                @sprintf("\n  emit       = %+.6e (%+.4f ppm)", _emit_delta, _ppm(_emit_delta)),
                @sprintf("\n  advection  = %+.6e (%+.4f ppm)", _adv_delta, _ppm(_adv_delta)),
                @sprintf("\n  BL diff    = %+.6e (%+.4f ppm)", _bld_delta, _ppm(_bld_delta)),
                @sprintf("\n  PBL diff   = %+.6e (%+.4f ppm)", _pbl_delta, _ppm(_pbl_delta)),
                @sprintf("\n  chemistry  = %+.6e (%+.4f ppm)", _chem_delta, _ppm(_chem_delta)),
                @sprintf("\n  TOTAL      = %+.6e (%+.4f ppm)", _total_delta, _ppm(_total_delta)))
        end
        # ── END MASS BALANCE DIAGNOSTICS ──────────────────────────────

        _last_fix = _apply_global_mass_fixer!(cs_tracers, grid, _pre_adv_mass)

        # Mass conservation — update showvalue every window
        _mass_totals = _compute_mass_totals(cs_tracers, grid)
        _last_mass = _mass_showvalue(_mass_totals, _initial_mass)
        # ── Output ────────────────────────────────────────────────────
        t0 = time()
        sim_time = Float64(step * dt_sub)
        # Build met_fields: 2D met fields + mass fluxes for flux diagnostics
        _met_base = (; ps=ps_cpu,
                       mass_flux_x=curr_gpu.am, mass_flux_y=curr_gpu.bm,
                       mf_scale=half_dt, dt_window=dt_window)
        _met_2d = if sfc_loaded && has_pbl_diff && troph_loaded
            merge(_met_base, (; pblh=pbl_sfc_cpu.pblh, troph=troph_cpu))
        elseif sfc_loaded && has_pbl_diff
            merge(_met_base, (; pblh=pbl_sfc_cpu.pblh))
        elseif troph_loaded
            merge(_met_base, (; troph=troph_cpu))
        else
            _met_base
        end
        if qv_loaded
            _met_2d = merge(_met_2d, (; qv=qv_cpu))
        end
        # For output: convert wet air mass to dry so that q = rm/m reports dry VMRs.
        # Uses m_panels as scratch (not needed between windows).
        _out_mass = m_ref_panels
        if qv_loaded
            for p in 1:6
                copyto!(m_panels[p], m_ref_panels[p])
            end
            for p in 1:6
                apply_dry_delp_panel!(m_panels[p], qv_gpu[p], Nc, Nz, Hp)
            end
            synchronize(get_backend(m_panels[1]))
            _out_mass = m_panels
        end
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=_out_mass, tracers=cs_tracers,
                          met_fields=_met_2d)
        end
        t_output += time() - t0

        # ── Wait for disk read (if not already fetched for pressure fixer) ──
        t0 = time()
        if has_next && !fetched_early
            io_status = fetch(load_task)
        end
        t_io += time() - t0

        curr_cpu, next_cpu = next_cpu, curr_cpu
        curr_gpu, next_gpu = next_gpu, curr_gpu

        # Progress bar with timing + diagnostics below the bar
        _sv = Pair{Symbol,Any}[
            :day  => div(w - 1, 24) + 1,
            :IO   => @sprintf("%.2f s/win", t_io / w),
            :GPU  => @sprintf("%.2f s/win", t_compute / w),
            :Out  => @sprintf("%.2f s/win", t_output / w)]
        isempty(_last_cfl)  || push!(_sv, :CFL  => _last_cfl)
        isempty(_last_fix)  || push!(_sv, :fix  => _last_fix)
        isempty(_last_mass) || push!(_sv, :mass => _last_mass)
        next!(prog; showvalues=_sv)
    end
    finish!(prog)

    # Final mass conservation summary
    _mass_final = _compute_mass_totals(cs_tracers, grid)
    for tname in sort(collect(keys(_mass_final)))
        total = _mass_final[tname]
        if haskey(_initial_mass, tname) && _initial_mass[tname] != 0.0
            rel = (total - _initial_mass[tname]) / abs(_initial_mass[tname]) * 100
            @info @sprintf("  Final mass %s: %.6e kg (Δ=%.4e%%)", tname, total, rel)
        else
            @info @sprintf("  Final mass %s: %.6e kg", tname, total)
        end
    end

    wall_total = time() - wall_start
    @info @sprintf(
        "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
        step, wall_total, t_io / n_win, t_compute / n_win, t_output / n_win)
    for writer in writers
        finalize_output!(writer)
    end
    return model
end
