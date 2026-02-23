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
using ..Advection: MassFluxWorkspace, allocate_massflux_workspace,
                   allocate_cs_massflux_workspace,
                   strang_split_massflux!, compute_air_mass!,
                   compute_mass_fluxes!, compute_air_mass_panel!, compute_cm_panel!,
                   build_geometry_cache
using ..Diffusion: DiffusionWorkspace, diffuse_gpu!
using ..Sources: AbstractSource, apply_surface_flux!, AbstractGriddedEmission,
                 CubedSphereEmission, GriddedEmission, apply_emissions_window!,
                 M_AIR, M_CO2
using ..IO: AbstractMetDriver, AbstractRawMetDriver, AbstractMassFluxMetDriver,
            total_windows, window_dt, steps_per_window, load_met_window!,
            LatLonMetBuffer, LatLonCPUBuffer, CubedSphereMetBuffer, CubedSphereCPUBuffer,
            upload!, AbstractOutputWriter, write_output!
using ..Diagnostics: column_mean!, surface_slice!
using Printf

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

    # Upload emission fluxes to device
    emission_data = []
    for src in sources
        if src isa GriddedEmission
            flux_dev = AT(src.flux)
            push!(emission_data, (src, flux_dev))
        end
    end

    return emission_data, area_j_dev, A_coeff, B_coeff
end

"""
Apply emissions for lat-lon grids (GPU-compatible via KA kernels).
"""
function _apply_emissions_latlon!(tracers, emission_data, area_j_dev, ps_dev,
                                    A_coeff, B_coeff, grid, dt_window)
    Nz = grid.Nz
    g = grid.gravity
    for (src, flux_dev) in emission_data
        name = src.species
        haskey(tracers, name) || continue
        c = tracers[name]
        if A_coeff !== nothing && B_coeff !== nothing
            apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev,
                                     A_coeff, B_coeff, Nz, g, dt_window)
        else
            # Fallback: uniform Δp approximation
            FT = eltype(c)
            Δp_approx = FT(grid.reference_pressure / Nz)
            mol = FT(1e6 * M_AIR / M_CO2)
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
        if src isa CubedSphereEmission
            flux_dev = ntuple(p -> AT(src.flux_panels[p]), 6)
            push!(emission_data, (src, flux_dev))
        end
    end
    return emission_data
end

"""
Apply cubed-sphere emissions using pre-uploaded device flux panels.
"""
function _apply_emissions_cs!(rm_panels::NTuple{6}, emission_data,
                                area_panels::NTuple{6}, dt_window, Nc::Int, Hp::Int)
    for (src, flux_dev) in emission_data
        apply_surface_flux!(rm_panels,
            CubedSphereEmission(flux_dev, src.species, src.label),
            area_panels, dt_window, Nc, Hp)
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

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_win     = total_windows(driver)
    n_sub     = steps_per_window(driver)
    dt_sub    = FT(driver.dt)

    # Allocate GPU met buffer + CPU staging
    gpu_buf = LatLonMetBuffer(arch, FT, Nx, Ny, Nz)
    cpu_buf = LatLonCPUBuffer(FT, Nx, Ny, Nz)

    # Prepare emission data on device
    emi_data, area_j_dev, A_coeff, B_coeff =
        _prepare_latlon_emissions(sources, grid, driver, arch)

    step = 0
    wall_start = time()

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (SingleBuffer, LatLon)"

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
        end

        # Apply emissions (GPU-compatible via KA kernels)
        dt_window = FT(dt_sub * n_sub)
        _apply_emissions_latlon!(model.tracers, emi_data, area_j_dev,
                                  gpu_buf.ps, A_coeff, B_coeff, grid, dt_window)

        # Advection sub-steps
        for sub in 1:n_sub
            step += 1
            copyto!(gpu_buf.m_dev, gpu_buf.m_ref)
            strang_split_massflux!(model.tracers, gpu_buf.m_dev,
                                    gpu_buf.am, gpu_buf.bm, gpu_buf.cm,
                                    grid, true, gpu_buf.ws; cfl_limit=FT(0.95))
        end

        # Output
        sim_time = Float64(step * dt_sub)
        _current_air_mass = gpu_buf.m_ref
        for writer in writers
            write_output!(writer, model, sim_time; air_mass=_current_air_mass)
        end

        # Progress
        if w % max(1, n_win ÷ 20) == 0 || w == n_win
            elapsed = round(time() - wall_start, digits=1)
            rate = w > 1 ? round((time() - wall_start) / w, digits=2) : 0.0
            @info @sprintf("  Window %d/%d (day %.1f): wall=%.0fs (%.2fs/win)",
                           w, n_win, sim_time / 86400, elapsed, rate)
        end
    end

    wall_total = time() - wall_start
    @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                   step, wall_total, wall_total / n_win)
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

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_win     = total_windows(driver)
    n_sub     = steps_per_window(driver)
    dt_sub    = FT(driver.dt)
    half_dt   = dt_sub / 2

    # Allocate TWO GPU met buffers + TWO CPU staging buffers
    buf_A = LatLonMetBuffer(arch, FT, Nx, Ny, Nz)
    buf_B = LatLonMetBuffer(arch, FT, Nx, Ny, Nz)
    cpu_A = LatLonCPUBuffer(FT, Nx, Ny, Nz)
    cpu_B = LatLonCPUBuffer(FT, Nx, Ny, Nz)

    # Prepare emission data on device
    emi_data, area_j_dev, A_coeff, B_coeff =
        _prepare_latlon_emissions(sources, grid, driver, arch)

    step = 0
    wall_start = time()

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (DoubleBuffer, LatLon)"

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
    end

    curr, next_buf = buf_A, buf_B
    cpu_curr, cpu_next = cpu_A, cpu_B

    for w in 1:n_win
        has_next = w < n_win
        dt_window = FT(dt_sub * n_sub)

        # Start CPU preload of next window (overlaps with GPU compute)
        if has_next
            load_met_window!(cpu_next, driver, grid, w + 1)
        end

        # Apply emissions (GPU-compatible via KA kernels)
        _apply_emissions_latlon!(model.tracers, emi_data, area_j_dev,
                                  curr.ps, A_coeff, B_coeff, grid, dt_window)

        # Advection sub-steps on current buffer
        for sub in 1:n_sub
            step += 1
            copyto!(curr.m_dev, curr.m_ref)
            strang_split_massflux!(model.tracers, curr.m_dev,
                                    curr.am, curr.bm, curr.cm,
                                    grid, true, curr.ws; cfl_limit=FT(0.95))
        end

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
            end
            # Swap
            curr, next_buf = next_buf, curr
            cpu_curr, cpu_next = cpu_next, cpu_curr
        end

        # Output (note: curr may have been swapped, use the buffer that was just computed)
        sim_time = Float64(step * dt_sub)
        _current_air_mass = has_next ? next_buf.m_ref : curr.m_ref  # pre-swap buffer
        for writer in writers
            write_output!(writer, model, sim_time; air_mass=_current_air_mass)
        end

        # Progress
        if w % max(1, n_win ÷ 20) == 0 || w == n_win
            elapsed = round(time() - wall_start, digits=1)
            rate = w > 1 ? round((time() - wall_start) / w, digits=2) : 0.0
            @info @sprintf("  Window %d/%d (day %.1f): wall=%.0fs (%.2fs/win)",
                           w, n_win, sim_time / 86400, elapsed, rate)
        end
    end

    wall_total = time() - wall_start
    @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                   step, wall_total, wall_total / n_win)
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
    ws = allocate_cs_massflux_workspace(ref_panel)

    # Allocate panel arrays
    rm_panels = allocate_cubed_sphere_field(grid, Nz)
    m_panels  = allocate_cubed_sphere_field(grid, Nz)

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

    step = 0
    wall_start = time()
    t_io      = 0.0   # met load + H2D upload
    t_compute = 0.0   # air mass, emissions, advection (GPU)
    t_output  = 0.0   # diagnostics + NetCDF write

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (SingleBuffer, C$Nc)"

    for w in 1:n_win
        # ── I/O: load met from disk → CPU → GPU ───────────────────────
        t0 = time()
        load_met_window!(cpu_buf, driver, grid, w)
        upload!(cs_gpu_buf, cpu_buf)
        t_io += time() - t0

        # ── GPU compute: air mass + emissions + advection ─────────────
        t0 = time()
        for p in 1:6
            compute_air_mass_panel!(m_panels[p], delp_gpu[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
        for p in 1:6
            compute_cm_panel!(cm_gpu[p], am_gpu[p], bm_gpu[p], gc.bt, Nc, Nz)
        end

        dt_window = FT(dt_sub * n_sub)
        _apply_emissions_cs!(rm_panels, emi_data, gc.area, dt_window, Nc, Hp)

        for _ in 1:n_sub
            step += 1
            strang_split_massflux!(rm_panels, m_panels,
                                    am_gpu, bm_gpu, cm_gpu,
                                    grid, true, ws)
        end
        t_compute += time() - t0

        # ── Output: diagnostics + regrid + NetCDF ─────────────────────
        t0 = time()
        sim_time = Float64(step * dt_sub)
        _current_air_mass = m_panels
        tracer_names = keys(model.tracers)
        cs_tracers = NamedTuple{tracer_names}(ntuple(_ -> rm_panels, length(tracer_names)))
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=_current_air_mass, tracers=cs_tracers)
        end
        t_output += time() - t0

        # ── Progress ───────────────────────────────────────────────────
        if w % max(1, n_win ÷ 20) == 0 || w == 1 || w == n_win
            elapsed = time() - wall_start
            day = div(w - 1, 24) + 1
            @info @sprintf(
                "  Day %d, win %3d/%d: wall=%5.0fs | IO=%5.2f  GPU=%5.2f  Out=%5.2f  s/win",
                day, w, n_win, elapsed,
                t_io / w, t_compute / w, t_output / w)
        end
    end

    wall_total = time() - wall_start
    @info @sprintf(
        "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
        step, wall_total, t_io / n_win, t_compute / n_win, t_output / n_win)
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
    ws = allocate_cs_massflux_workspace(ref_panel)

    # Tracers + air-mass (single set — not double-buffered)
    rm_panels = allocate_cubed_sphere_field(grid, Nz)
    m_panels  = allocate_cubed_sphere_field(grid, Nz)

    # TWO GPU met buffers (A = current, B = next)
    gpu_A = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp)
    gpu_B = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp)

    # TWO CPU staging buffers
    cpu_A = CubedSphereCPUBuffer(FT, Nc, Nz, Hp)
    cpu_B = CubedSphereCPUBuffer(FT, Nc, Nz, Hp)

    # Pre-upload emission flux panels to device
    emi_data = _prepare_cs_emissions(sources, grid, arch)

    # Pre-load window 1 synchronously before the loop
    load_met_window!(cpu_A, driver, grid, 1)

    curr_cpu, next_cpu = cpu_A, cpu_B
    curr_gpu, next_gpu = gpu_A, gpu_B

    step = 0
    wall_start = time()
    t_io      = 0.0   # upload + disk-wait time
    t_compute = 0.0   # GPU compute
    t_output  = 0.0   # diagnostics + NetCDF
    load_task = nothing

    @info "Starting simulation: $n_win windows × $n_sub sub-steps (DoubleBuffer, C$Nc)"

    for w in 1:n_win
        has_next = w < n_win

        # ── Spawn async disk read for next window ──────────────────────
        # Runs on a worker thread while main thread does upload + GPU.
        t0 = time()
        if has_next
            load_task = Threads.@spawn load_met_window!(next_cpu, driver, grid, w + 1)
        end

        # Upload current CPU buffer → GPU (main thread, PCIe DMA)
        upload!(curr_gpu, curr_cpu)
        t_io += time() - t0

        # ── GPU compute ───────────────────────────────────────────────
        t0 = time()
        for p in 1:6
            compute_air_mass_panel!(m_panels[p], curr_gpu.delp[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
        for p in 1:6
            compute_cm_panel!(curr_gpu.cm[p], curr_gpu.am[p], curr_gpu.bm[p], gc.bt, Nc, Nz)
        end

        dt_window = FT(dt_sub * n_sub)
        _apply_emissions_cs!(rm_panels, emi_data, gc.area, dt_window, Nc, Hp)

        for _ in 1:n_sub
            step += 1
            strang_split_massflux!(rm_panels, m_panels,
                                    curr_gpu.am, curr_gpu.bm, curr_gpu.cm,
                                    grid, true, ws)
        end
        t_compute += time() - t0

        # ── Output ────────────────────────────────────────────────────
        t0 = time()
        sim_time = Float64(step * dt_sub)
        tracer_names = keys(model.tracers)
        cs_tracers = NamedTuple{tracer_names}(ntuple(_ -> rm_panels, length(tracer_names)))
        for writer in writers
            write_output!(writer, model, sim_time;
                          air_mass=m_panels, tracers=cs_tracers)
        end
        t_output += time() - t0

        # ── Wait for disk read, then swap ─────────────────────────────
        # Any remaining disk-read time that wasn't hidden by GPU+output.
        t0 = time()
        has_next && wait(load_task)
        t_io += time() - t0

        curr_cpu, next_cpu = next_cpu, curr_cpu
        curr_gpu, next_gpu = next_gpu, curr_gpu

        # ── Progress ──────────────────────────────────────────────────
        if w % max(1, n_win ÷ 20) == 0 || w == 1 || w == n_win
            elapsed = time() - wall_start
            day = div(w - 1, 24) + 1
            @info @sprintf(
                "  Day %d, win %3d/%d: wall=%5.0fs | IO=%5.2f  GPU=%5.2f  Out=%5.2f  s/win",
                day, w, n_win, elapsed,
                t_io / w, t_compute / w, t_output / w)
        end
    end

    wall_total = time() - wall_start
    @info @sprintf(
        "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
        step, wall_total, t_io / n_win, t_compute / n_win, t_output / n_win)
    return model
end
