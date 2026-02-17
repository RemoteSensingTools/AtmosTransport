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
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const FT        = Float64
const Δt        = 600.0                # 10-minute time step [s]
const N_HOURS   = 48                   # simulation length [hours]

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const DATA_DIR     = joinpath(PROJECT_ROOT, "data", "era5")
const PL_FILE      = joinpath(DATA_DIR, "era5_pressure_levels_20250201_20250207.nc")
const SL_FILE      = joinpath(DATA_DIR, "era5_single_levels_20250201_20250207.nc")
const OUTDIR       = joinpath(DATA_DIR, "output")
const OUTFILE      = joinpath(OUTDIR, "forward_era5_output.nc")

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

    # ERA5 pressure levels: 1..1000 hPa (top → surface)
    # Our convention: k=1 is top of atmosphere, k=Nz is near surface
    # ERA5 already stores data this way, so no reorder needed

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
    p_edges[Nz+1] = FT(101325)  # approximate surface pressure

    @info "  Vertical: $(Nz) levels, p_top=$(p_edges[1]) Pa, p_surface=$(p_edges[end]) Pa"

    # Pure sigma: a=0, b=p/p_surface
    Ps = p_edges[end]
    b_values = p_edges ./ Ps
    a_values = zeros(FT, Nz + 1)

    vc = HybridSigmaPressure(a_values, b_values)

    Δlon = lons[2] - lons[1]
    grid = LatitudeLongitudeGrid(CPU();
        size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]) - Δlon/2, FT(lons[end]) + Δlon/2),
        latitude  = (FT(lats[1]) - Δlon/2, FT(lats[end]) + Δlon/2),
        vertical  = vc)

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

    # CFL limiter parameters
    cfl_limit = FT(0.4)
    Δlon_m = FT(2.5 * 111000)  # rough estimate of grid spacing at equator
    Δlat_m = FT(2.5 * 111000)
    u_max = cfl_limit * Δlon_m / FT(Δt)
    v_max = cfl_limit * Δlat_m / FT(Δt)

    @info "  CFL limits: u_max=$(round(u_max, digits=1)) m/s, v_max=$(round(v_max, digits=1)) m/s"

    # Stagger u to x-faces: (Nx+1, Ny, Nz), periodic
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u_val = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
        u[i, j, k] = clamp(u_val, -u_max, u_max)
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    # Stagger v to y-faces: (Nx, Ny+1, Nz), zero at poles
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v_val = (v_cc[i, j-1, k] + v_cc[i, j, k]) / 2
        v[i, j, k] = clamp(v_val, -v_max, v_max)
    end

    # w at z-interfaces: (Nx, Ny, Nz+1)
    # omega (Pa/s), positive downward → negate for our convention (w>0 upward)
    w = zeros(FT, Nx, Ny, Nz + 1)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        w_raw = -(omega[i, j, k-1] + omega[i, j, k]) / 2
        # CFL limit based on pressure thickness
        dp = p_edges[k+1] - p_edges[k]
        dp_prev = p_edges[k] - p_edges[k-1]
        dp_min = min(abs(dp), abs(dp_prev))
        w_lim = cfl_limit * dp_min / FT(Δt)
        w[i, j, k] = clamp(w_raw, -w_lim, w_lim)
    end
    # w[:,:,1] = 0 (top), w[:,:,Nz+1] = 0 (surface)

    @info "Staggered wind ranges:"
    @info "  u: [$(round(minimum(u),digits=1)), $(round(maximum(u),digits=1))] m/s"
    @info "  v: [$(round(minimum(v),digits=1)), $(round(maximum(v),digits=1))] m/s"
    @info "  w: [$(round(minimum(w),digits=2)), $(round(maximum(w),digits=2))] Pa/s"

    return (; u, v, w)
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
    tracers = initialize_tracer(data, FT)
    @info "  Initial tracer: min=$(round(minimum(tracers.c), digits=2)), " *
          "max=$(round(maximum(tracers.c), digits=2)), " *
          "mean=$(round(sum(tracers.c)/length(tracers.c), digits=2))"

    # Step 4: Run simulation
    @info "\n--- Step 4: Running forward simulation ---"
    @info "  Δt = $(Δt)s, N_hours = $N_HOURS ($(N_HOURS÷24) days)"

    scheme = SlopesAdvection(use_limiter=true)
    N_steps = round(Int, N_HOURS * 3600 / Δt)

    mkpath(OUTDIR)
    isfile(OUTFILE) && rm(OUTFILE)

    # Save initial state
    save_output(OUTFILE, tracers, data.lons, data.lats, data.Nz, 0, 0.0)

    total_mass_initial = sum(tracers.c)
    t0 = time()
    last_report = t0

    for step in 1:N_steps
        half = Δt / 2

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
            total_mass = sum(tracers.c)
            mass_change = abs(total_mass - total_mass_initial) / abs(total_mass_initial)
            wall = round(time() - t0, digits=1)
            @info "  t = $(round(time_hours, digits=1))h: " *
                  "min=$(round(minimum(tracers.c), digits=2)), " *
                  "max=$(round(maximum(tracers.c), digits=2)), " *
                  "mass_change=$(round(mass_change * 100, sigdigits=3))%, " *
                  "wall=$(wall)s"

            save_output(OUTFILE, tracers, data.lons, data.lats, data.Nz, step, time_hours)
        end
    end

    # Save final state
    save_output(OUTFILE, tracers, data.lons, data.lats, data.Nz, N_steps, FT(N_HOURS))

    total_mass_final = sum(tracers.c)
    mass_conservation = abs(total_mass_final - total_mass_initial) / abs(total_mass_initial)
    wall_total = round(time() - t0, digits=1)

    @info "\n--- Results ---"
    @info "  Total steps: $N_steps"
    @info "  Wall time: $(wall_total)s"
    @info "  Final tracer: min=$(round(minimum(tracers.c), digits=2)), " *
          "max=$(round(maximum(tracers.c), digits=2))"
    @info "  Mass conservation: $(round(mass_conservation * 100, sigdigits=3))% change"
    @info "  Output saved to: $OUTFILE"
    @info "=" ^ 60
end

main()
