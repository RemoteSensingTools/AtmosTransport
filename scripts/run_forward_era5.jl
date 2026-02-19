#!/usr/bin/env julia
# ===========================================================================
# Forward model test using downloaded ERA5 data
#
# Loads ERA5 pressure-level and single-level data downloaded by
# scripts/download_era5_week.jl, sets up a LatitudeLongitudeGrid at 2.5°
# with 37 pressure levels, and runs a 2-day forward tracer simulation.
#
# Usage:
#   julia --project=. scripts/run_forward_era5.jl
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Parameters
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Load configuration from TOML (with optional override)
# ---------------------------------------------------------------------------
const OVERRIDE_TOML = get(ENV, "CONFIG", nothing)
const FT_STR = get(ENV, "USE_FLOAT32", "false") == "true" ? Float32 : Float64
const PARAMS = load_parameters(FT_STR; override=OVERRIDE_TOML)
const FT     = FT_STR
const Δt     = PARAMS.simulation.dt
const N_HOURS = PARAMS.simulation.n_hours

const USE_GPU = get(ENV, "USE_GPU", "false") == "true"
if USE_GPU
    using CUDA
end

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const DATA_DIR     = joinpath(PROJECT_ROOT, "data", "era5")
const PL_FILE      = joinpath(DATA_DIR, "era5_pressure_levels_20250201_20250207.nc")
const SL_FILE      = joinpath(DATA_DIR, "era5_single_levels_20250201_20250207.nc")
const OUTDIR       = joinpath(DATA_DIR, "output")
const REFERENCE_RUN = get(ENV, "REFERENCE_RUN", "false") == "true"
const OUTFILE      = joinpath(OUTDIR, REFERENCE_RUN ? "reference_era5_output.nc" : "forward_era5_output.nc")

# Standard 37 ERA5 pressure levels [hPa]
const ERA5_PRESSURE_LEVELS = FT.([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000,
])

# ---------------------------------------------------------------------------
# Step 1: Load ERA5 data from NetCDF files
# ---------------------------------------------------------------------------
function load_era5_data(pl_file::String, sl_file::String, ::Type{FT};
                        time_idx::Int=1) where {FT}
    @info "Loading ERA5 pressure-level data from $pl_file"
    @info "Loading ERA5 single-level data from $sl_file"

    # Pre-declare variables used across try/catch boundary
    local lons_raw, lats_raw, levels
    local Nx, Ny, Nz
    local u_raw, v_raw, w_raw, t_raw, q_raw

    # Read pressure-level data
    ds_pl = NCDataset(pl_file)
    try
        # Inspect dimensions and variables
        @info "Pressure-level file dimensions: $(keys(ds_pl.dim))"
        @info "Pressure-level file variables: $(keys(ds_pl))"
        for (k, v) in ds_pl.dim
            @info "  $k: $v"
        end

        # Read coordinates — ERA5 NetCDF may use "longitude"/"latitude" or "lon"/"lat"
        lon_key = haskey(ds_pl, "longitude") ? "longitude" : "lon"
        lat_key = haskey(ds_pl, "latitude") ? "latitude" : "lat"
        lev_key = haskey(ds_pl, "pressure_level") ? "pressure_level" :
                  haskey(ds_pl, "level") ? "level" : "lev"

        lons_raw = FT.(ds_pl[lon_key][:])
        lats_raw = FT.(ds_pl[lat_key][:])
        levels   = ds_pl[lev_key][:]

        Nx = length(lons_raw)
        Ny = length(lats_raw)
        Nz = length(levels)

        @info "  Raw grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"
        @info "  Lon range: $(lons_raw[1]) to $(lons_raw[end])"
        @info "  Lat range: $(lats_raw[1]) to $(lats_raw[end])"
        @info "  Levels: $(levels[1]) to $(levels[end]) ($(length(levels)) levels)"

        # Determine variable names (CDS short names)
        u_key = haskey(ds_pl, "u") ? "u" : "U"
        v_key = haskey(ds_pl, "v") ? "v" : "V"
        w_key = haskey(ds_pl, "w") ? "w" : "W"
        t_key = haskey(ds_pl, "t") ? "t" : "T"
        q_key = haskey(ds_pl, "q") ? "q" : "Q"

        # Read one time step of 3D fields: dims are (lon, lat, level, time) or similar
        @info "  Reading wind and thermodynamic fields (time index=$time_idx)..."
        ndims_u = ndims(ds_pl[u_key])
        if ndims_u == 4
            u_raw = FT.(ds_pl[u_key][:, :, :, time_idx])
            v_raw = FT.(ds_pl[v_key][:, :, :, time_idx])
            w_raw = FT.(ds_pl[w_key][:, :, :, time_idx])
            t_raw = FT.(ds_pl[t_key][:, :, :, time_idx])
            q_raw = FT.(ds_pl[q_key][:, :, :, time_idx])
        elseif ndims_u == 3
            u_raw = FT.(ds_pl[u_key][:, :, :])
            v_raw = FT.(ds_pl[v_key][:, :, :])
            w_raw = FT.(ds_pl[w_key][:, :, :])
            t_raw = FT.(ds_pl[t_key][:, :, :])
            q_raw = FT.(ds_pl[q_key][:, :, :])
        else
            error("Unexpected dimensions for u: $ndims_u")
        end
    catch e
        close(ds_pl)
        rethrow(e)
    end
    close(ds_pl)

    # Read single-level data
    sp_raw = nothing
    if isfile(sl_file)
        ds_sl = NCDataset(sl_file)
        try
            sp_key = haskey(ds_sl, "sp") ? "sp" :
                     haskey(ds_sl, "SP") ? "SP" : "surface_pressure"

            ndims_sp = ndims(ds_sl[sp_key])
            if ndims_sp == 3
                sp_raw = FT.(ds_sl[sp_key][:, :, time_idx])
            elseif ndims_sp == 2
                sp_raw = FT.(ds_sl[sp_key][:, :])
            end
            @info "  Surface pressure range: $(minimum(sp_raw)/100) — $(maximum(sp_raw)/100) hPa"
        catch e
            @warn "Could not read surface pressure: $e"
        finally
            close(ds_sl)
        end
    else
        @warn "Single-level file not found: $sl_file"
    end

    # --- Transform coordinates ---

    # ERA5 longitude: 0..357.5 → -180..177.5
    shift_idx = 0
    if lons_raw[1] >= 0 && lons_raw[end] > 180
        shift_idx = findfirst(l -> l >= 180, lons_raw)
        if shift_idx !== nothing
            shift_n = Nx - shift_idx + 1
            lons = circshift(lons_raw, shift_n)
            lons[1:shift_n] .-= FT(360)

            u_raw = circshift(u_raw, (shift_n, 0, 0))
            v_raw = circshift(v_raw, (shift_n, 0, 0))
            w_raw = circshift(w_raw, (shift_n, 0, 0))
            t_raw = circshift(t_raw, (shift_n, 0, 0))
            q_raw = circshift(q_raw, (shift_n, 0, 0))
            if sp_raw !== nothing
                sp_raw = circshift(sp_raw, (shift_n, 0))
            end
            @info "  Shifted longitude by $shift_n to range [$(lons[1]), $(lons[end])]"
        else
            lons = lons_raw
        end
    else
        lons = lons_raw
    end

    # ERA5 latitude: 90..-90 (N→S) → -90..90 (S→N)
    if lats_raw[1] > lats_raw[end]
        lats = reverse(lats_raw)
        u_raw = u_raw[:, end:-1:1, :]
        v_raw = v_raw[:, end:-1:1, :]
        w_raw = w_raw[:, end:-1:1, :]
        t_raw = t_raw[:, end:-1:1, :]
        q_raw = q_raw[:, end:-1:1, :]
        if sp_raw !== nothing
            sp_raw = sp_raw[:, end:-1:1]
        end
        @info "  Reversed latitude to S→N: [$(lats[1]), $(lats[end])]"
    else
        lats = lats_raw
    end

    # ERA5 pressure levels: ensure k=1 is top of atmosphere (lowest pressure)
    # ERA5 CDS often stores surface-first (1000→10 hPa); reverse if needed.
    if levels[1] > levels[end]
        levels = reverse(levels)
        u_raw = u_raw[:, :, end:-1:1]
        v_raw = v_raw[:, :, end:-1:1]
        w_raw = w_raw[:, :, end:-1:1]
        t_raw = t_raw[:, :, end:-1:1]
        q_raw = q_raw[:, :, end:-1:1]
        @info "  Reversed pressure levels to top→surface: [$(levels[1]), $(levels[end])] hPa"
    end

    @info "  Transformed grid: lon=[$(lons[1]),$(lons[end])], lat=[$(lats[1]),$(lats[end])]"
    @info "  Wind ranges: u=[$(round(minimum(u_raw),digits=1)),$(round(maximum(u_raw),digits=1))] m/s"
    @info "               v=[$(round(minimum(v_raw),digits=1)),$(round(maximum(v_raw),digits=1))] m/s"
    @info "               w=[$(round(minimum(w_raw),digits=2)),$(round(maximum(w_raw),digits=2))] Pa/s"

    return (; u=u_raw, v=v_raw, w=w_raw, t=t_raw, q=q_raw,
              sp=sp_raw, lons, lats, levels=FT.(levels), Nx, Ny, Nz)
end

# ---------------------------------------------------------------------------
# Step 2: Build grid and stagger winds
# ---------------------------------------------------------------------------
function build_era5_grid(data, ::Type{FT}) where {FT}
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    lons, lats = data.lons, data.lats

    # Build a pure-pressure vertical coordinate from the 37 standard levels
    # Pressure at interfaces: midpoints between level centers, plus boundaries
    p_levels_Pa = data.levels .* FT(100)  # hPa → Pa

    p_edges = zeros(FT, Nz + 1)
    p_edges[1] = FT(0)  # model top (above 1 hPa)
    for k in 1:Nz-1
        p_edges[k+1] = (p_levels_Pa[k] + p_levels_Pa[k+1]) / 2
    end
    p_edges[Nz+1] = PARAMS.planet.reference_surface_pressure

    @info "  Vertical: $(Nz) levels, p_top=$(p_edges[1]) Pa, p_surface=$(p_edges[end]) Pa"
    @info "  Level centers (hPa): $(round.(p_levels_Pa ./ 100, digits=1)[1:min(5,Nz)])... $(round.(p_levels_Pa ./ 100, digits=1)[max(1,Nz-2):Nz])"
    @info "  p_edges monotonic? $(all(diff(p_edges) .> 0))"
    dp_min = minimum(diff(p_edges))
    dp_max = maximum(diff(p_edges))
    @info "  Layer thickness range: $(round(dp_min, digits=1)) — $(round(dp_max, digits=1)) Pa"

    # Pure sigma: a=0, b=p/p_surface
    Ps = p_edges[end]
    b_values = p_edges ./ Ps
    a_values = zeros(FT, Nz + 1)

    vc = HybridSigmaPressure(a_values, b_values)

    Δlon = lons[2] - lons[1]
    arch = USE_GPU ? GPU() : CPU()
    pp = PARAMS.planet
    grid = LatitudeLongitudeGrid(arch;
        FT,
        size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]) - Δlon/2, FT(lons[end]) + Δlon/2),
        latitude  = (FT(-90), FT(90)),
        vertical  = vc,
        radius             = pp.radius,
        gravity             = pp.gravity,
        reference_pressure  = pp.reference_surface_pressure)

    @info "Grid constructed: Nx=$Nx, Ny=$Ny, Nz=$Nz"
    @info "  Longitude: [$(lons[1]), $(lons[end])]"
    @info "  Latitude: [$(lats[1]), $(lats[end])]"

    return grid, p_levels_Pa, p_edges
end

function stagger_winds(data, grid, p_levels_Pa, p_edges, ::Type{FT}) where {FT}
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    u_cc = data.u
    v_cc = data.v
    omega = data.w  # vertical velocity in Pa/s

    # CPU copies of grid coords for host-side staggering
    φᶠ_cpu = Array(grid.φᶠ)
    φᶜ_cpu = Array(grid.φᶜ)

    cfl_limit = FT(0.4)

    # Stagger u to x-faces: (Nx+1, Ny, Nz), periodic
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    # Stagger v to y-faces: (Nx, Ny+1, Nz), zero at poles
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j-1, k] + v_cc[i, j, k]) / 2
    end

    # Diagnose vertical omega from horizontal divergence (TM5 approach):
    # 1. Compute horizontal divergence at each level
    # 2. Remove barotropic component so ω=0 at BOTH top and surface
    # 3. Integrate from top to get omega profile
    # Combined with unsplit advection, this guarantees exact
    # physical mass conservation (all boundary fluxes = 0).
    R_earth = grid.radius
    div_h = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        dx = Δx(i, j, grid)
        div_u = (u[ip, j, k] - u[i, j, k]) / dx

        cos_N = cosd(φᶠ_cpu[j + 1])
        cos_S = cosd(φᶠ_cpu[j])
        sin_N = sind(φᶠ_cpu[j + 1])
        sin_S = sind(φᶠ_cpu[j])
        dsinphi = max(abs(sin_N - sin_S), FT(1e-30))
        div_v = (v[i, j+1, k] * cos_N - v[i, j, k] * cos_S) / (R_earth * dsinphi)

        div_h[i, j, k] = div_u + div_v
    end

    P_total = p_edges[Nz+1] - p_edges[1]
    @inbounds for j in 1:Ny, i in 1:Nx
        pit = FT(0)
        for k in 1:Nz
            pit += div_h[i, j, k] * (p_edges[k+1] - p_edges[k])
        end
        for k in 1:Nz
            div_h[i, j, k] -= pit / P_total
        end
    end

    w = zeros(FT, Nx, Ny, Nz + 1)
    @inbounds for j in 1:Ny, i in 1:Nx
        for k in 1:Nz
            w[i, j, k+1] = w[i, j, k] - div_h[i, j, k] * (p_edges[k+1] - p_edges[k])
        end
    end

    max_cfl_z = FT(0)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        dp = min(p_edges[k+1] - p_edges[k], p_edges[k] - p_edges[k-1])
        cfl_z = abs(w[i, j, k]) * FT(Δt) / dp
        max_cfl_z = max(max_cfl_z, cfl_z)
    end
    @info "  Max vertical CFL: $(round(max_cfl_z, digits=2))"
    @info "  w(surface) max|w|: $(round(maximum(abs.(w[:,:,Nz+1])), sigdigits=3))"

    @info "Staggered wind ranges:"
    @info "  u: [$(round(minimum(u),digits=1)), $(round(maximum(u),digits=1))] m/s"
    @info "  v: [$(round(minimum(v),digits=1)), $(round(maximum(v),digits=1))] m/s"
    @info "  w: [$(round(minimum(w),digits=2)), $(round(maximum(w),digits=2))] Pa/s"

    ArrayType = array_type(grid.architecture)
    return (; u=ArrayType(u), v=ArrayType(v), w=ArrayType(w))
end

# ---------------------------------------------------------------------------
# Step 3: Initialize tracer
# ---------------------------------------------------------------------------
function initialize_tracer(data, ::Type{FT}) where {FT}
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    lons, lats = data.lons, data.lats

    # CO2-like background: 420 ppm everywhere
    c = fill(FT(420.0), Nx, Ny, Nz)

    # Add Gaussian blobs over major source regions

    # Europe: centered at 10°E, 50°N
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Δlon = lons[i] - FT(10)
        Δlat = lats[j] - FT(50)
        # Concentrate in lower troposphere (levels near surface, high k)
        Δk = FT(k - (Nz - 3))
        blob = FT(30.0) * exp(-(Δlon^2 / FT(450) + Δlat^2 / FT(200) + Δk^2 / FT(8)))
        c[i, j, k] += blob
    end

    # East Asia: centered at 115°E, 35°N
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Δlon = lons[i] - FT(115)
        Δlat = lats[j] - FT(35)
        Δk = FT(k - (Nz - 3))
        blob = FT(25.0) * exp(-(Δlon^2 / FT(450) + Δlat^2 / FT(200) + Δk^2 / FT(8)))
        c[i, j, k] += blob
    end

    # Eastern US: centered at -85°E, 40°N
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Δlon = lons[i] - FT(-85)
        Δlat = lats[j] - FT(40)
        Δk = FT(k - (Nz - 3))
        blob = FT(20.0) * exp(-(Δlon^2 / FT(450) + Δlat^2 / FT(200) + Δk^2 / FT(8)))
        c[i, j, k] += blob
    end

    return (; c = c)
end

function to_device(tracers, arch)
    ArrayType = array_type(arch)
    return (; c = ArrayType(tracers.c))
end

# ---------------------------------------------------------------------------
# Step 4: Output
# ---------------------------------------------------------------------------
function save_output(outfile, tracers, lons, lats, Nz, step, time_hours)
    NCDataset(outfile, isfile(outfile) ? "a" : "c") do ds
        if !haskey(ds.dim, "lon")
            defDim(ds, "lon", length(lons))
            defDim(ds, "lat", length(lats))
            defDim(ds, "lev", Nz)
            defDim(ds, "time", Inf)

            defVar(ds, "lon", Float64, ("lon",))[:] = lons
            defVar(ds, "lat", Float64, ("lat",))[:] = lats
            defVar(ds, "time_hours", Float64, ("time",))

            ds.attrib["description"] = "AtmosTransportModel ERA5 forward run"
            ds.attrib["created"] = string(now())
            ds.attrib["met_source"] = "ERA5 2.5° 37-level"
            ds.attrib["time_step_seconds"] = Δt
        end

        tidx = haskey(ds, "time_hours") ? length(ds["time_hours"]) + 1 : 1

        if !haskey(ds, "tracer_c")
            defVar(ds, "tracer_c", Float32, ("lon", "lat", "lev", "time"))
        end

        ds["time_hours"][tidx] = time_hours
        ds["tracer_c"][:, :, :, tidx] = Float32.(tracers.c)
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    @info "=" ^ 60
    @info "AtmosTransportModel — ERA5 Forward Run"
    @info "=" ^ 60

    # Check data files exist
    for f in [PL_FILE]
        if !isfile(f)
            error("Missing data file: $f\nRun scripts/download_era5_week.jl first.")
        end
    end

    # Step 1: Load ERA5 data
    @info "\n--- Step 1: Loading ERA5 data ---"
    data = load_era5_data(PL_FILE, SL_FILE, FT; time_idx=1)

    # Step 2: Build grid and stagger winds
    @info "\n--- Step 2: Building grid and staggering winds ---"
    grid, p_levels_Pa, p_edges = build_era5_grid(data, FT)
    vel = stagger_winds(data, grid, p_levels_Pa, p_edges, FT)

    # Step 3: Initialize tracers
    @info "\n--- Step 3: Initializing tracers ---"
    tracers_cpu = initialize_tracer(data, FT)
    @info "  Initial tracer: min=$(round(minimum(tracers_cpu.c), digits=2)), " *
          "max=$(round(maximum(tracers_cpu.c), digits=2)), " *
          "mean=$(round(sum(tracers_cpu.c)/length(tracers_cpu.c), digits=2))"
    tracers = to_device(tracers_cpu, grid.architecture)
    @info "  Backend: $(USE_GPU ? "GPU (CUDA)" : "CPU (KernelAbstractions)")"

    # Step 4: Run simulation
    @info "\n--- Step 4: Running forward simulation ---"
    @info "  Δt = $(Δt)s, N_hours = $N_HOURS ($(N_HOURS÷24) days)"

    scheme = SlopesAdvection(use_limiter=true)
    N_steps = round(Int, N_HOURS * 3600 / Δt)

    mkpath(OUTDIR)
    isfile(OUTFILE) && rm(OUTFILE)

    # Save initial state (always save CPU data)
    save_output(OUTFILE, (; c=Array(tracers.c)), data.lons, data.lats, data.Nz, 0, 0.0)

    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz

    # Volume weights for physical mass: V[i,j,k] ∝ Δ(sinφ) × Δp
    φᶠ_cpu = Array(grid.φᶠ)
    vol_cpu = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dsinphi = abs(sind(φᶠ_cpu[j+1]) - sind(φᶠ_cpu[j]))
        dp_k = p_edges[k+1] - p_edges[k]
        vol_cpu[i, j, k] = dsinphi * dp_k
    end
    vol = array_type(grid.architecture)(vol_cpu)

    physical_mass(c) = sum(Array(c) .* vol_cpu)
    total_mass_initial = physical_mass(tracers.c)
    simple_mass_initial = sum(Array(tracers.c))
    t0 = time()
    last_report = t0

    # Pre-extract staggered velocities
    u_vel = vel.u
    v_vel = vel.v
    w_vel = vel.w

    for step in 1:N_steps
        half = FT(Δt / 2)

        # Strang splitting: XYZ / ZYX
        advect_x!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_z!(tracers, vel, grid, scheme, half)

        advect_z!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_x!(tracers, vel, grid, scheme, half)

        time_hours = step * Δt / 3600.0

        # Report every 6 simulated hours
        if step % round(Int, 6 * 3600 / Δt) == 0
            c_cpu = Array(tracers.c)
            phys_mass = physical_mass(tracers.c)
            simp_mass = sum(c_cpu)
            phys_change = (phys_mass - total_mass_initial) / abs(total_mass_initial)
            simp_change = (simp_mass - simple_mass_initial) / abs(simple_mass_initial)
            wall = round(time() - t0, digits=1)
            @info "  t = $(round(time_hours, digits=1))h: " *
                  "min=$(round(minimum(c_cpu), digits=2)), " *
                  "max=$(round(maximum(c_cpu), digits=2)), " *
                  "Σc_change=$(round(simp_change * 100, sigdigits=3))%, " *
                  "phys_mass_change=$(round(phys_change * 100, sigdigits=3))%, " *
                  "wall=$(wall)s"

            save_output(OUTFILE, (; c=c_cpu), data.lons, data.lats, data.Nz, step, time_hours)
        end
    end

    # Save final state
    c_final = Array(tracers.c)
    save_output(OUTFILE, (; c=c_final), data.lons, data.lats, data.Nz, N_steps, FT(N_HOURS))

    phys_mass_final = physical_mass(tracers.c)
    simp_mass_final = sum(c_final)
    phys_conservation = abs(phys_mass_final - total_mass_initial) / abs(total_mass_initial)
    simp_conservation = abs(simp_mass_final - simple_mass_initial) / abs(simple_mass_initial)
    wall_total = round(time() - t0, digits=1)

    @info "\n--- Results ---"
    @info "  Total steps: $N_steps"
    @info "  Wall time: $(wall_total)s"
    @info "  Final tracer: min=$(round(minimum(c_final), digits=2)), " *
          "max=$(round(maximum(c_final), digits=2))"
    @info "  Σc conservation: $(round(simp_conservation * 100, sigdigits=3))% change"
    @info "  Physical mass conservation: $(round(phys_conservation * 100, sigdigits=3))% change"
    @info "  Output saved to: $OUTFILE"
    @info "=" ^ 60
end

main()
