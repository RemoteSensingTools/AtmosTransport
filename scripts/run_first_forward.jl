#!/usr/bin/env julia
# ===========================================================================
# First end-to-end forward run of AtmosTransport
#
# Downloads a small GEOS-FP test dataset via OPeNDAP (no authentication),
# initializes a tracer blob, runs the model for 24 hours, and writes
# NetCDF output.
#
# Usage:
#   julia --project=. scripts/run_first_forward.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Diffusion
using AtmosTransport.IO: prepare_met_for_physics
using AtmosTransport.TimeSteppers
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const FT        = Float64
const TEST_DATE = Date(2024, 12, 1)   # recent date for GEOS-FP
const STRIDE    = 10                   # spatial coarsening (10 → ~3° resolution)
const Δt        = 300.0                # 5-minute time step [s]
const N_HOURS   = 24                   # simulation length [hours]
const OUTDIR    = joinpath(homedir(), "data", "metDrivers", "geosfp", "test")
const OUTFILE   = joinpath(OUTDIR, "forward_run_output.nc")

# ---------------------------------------------------------------------------
# Step 1: Download GEOS-FP test data via OPeNDAP
# ---------------------------------------------------------------------------
function download_geosfp_test(date::Date, outdir::String; stride::Int=STRIDE)
    mkpath(outdir)
    outfile = joinpath(outdir, "geosfp_asm_$(Dates.format(date, "yyyymmdd")).nc")

    if isfile(outfile) && filesize(outfile) > 1000
        @info "Using cached: $outfile ($(filesize(outfile) ÷ 1024) KB)"
        return outfile
    end

    @info "Downloading GEOS-FP via OPeNDAP for $date (stride=$stride)..."
    url = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv"
    @info "Opening: $url"

    ds = NCDataset(url)
    try
        lons  = ds["lon"][:]
        lats  = ds["lat"][:]
        levs  = ds["lev"][:]
        times = ds["time"][:]

        # Find closest time to 12Z on the requested date
        target_dt = DateTime(date) + Hour(12)
        if eltype(times) <: DateTime
            diffs = [abs(Dates.value(t - target_dt)) for t in times]
        else
            # Numeric time (days since 1-1-1)
            target_day = Dates.value(date - Date(1, 1, 1)) + 0.5
            diffs = abs.(times .- target_day)
        end
        tidx = argmin(diffs)
        @info "Time index: $tidx (time=$(times[tidx]))"

        # Subsample spatially
        lon_idx = 1:stride:length(lons)
        lat_idx = 1:stride:length(lats)
        Nx_out  = length(lon_idx)
        Ny_out  = length(lat_idx)
        Nz_out  = length(levs)
        @info "Grid: $(Nx_out)×$(Ny_out)×$(Nz_out)"

        vars_3d = ["u", "v", "omega", "t", "delp"]
        vars_2d = ["ps"]

        NCDataset(outfile, "c") do out
            defDim(out, "lon", Nx_out)
            defDim(out, "lat", Ny_out)
            defDim(out, "lev", Nz_out)

            defVar(out, "lon", Float64, ("lon",))[:] = lons[lon_idx]
            defVar(out, "lat", Float64, ("lat",))[:] = lats[lat_idx]
            defVar(out, "lev", Float64, ("lev",))[:] = levs

            out.attrib["source"] = "GEOS-FP inst3_3d_asm_Nv"
            out.attrib["date"] = Dates.format(date, "yyyy-mm-dd")
            out.attrib["stride"] = stride

            for varname in vars_3d
                @info "  Reading $varname..."
                data = ds[varname][lon_idx, lat_idx, :, tidx]
                defVar(out, varname, Float32, ("lon", "lat", "lev"))[:] = Float32.(data)
            end

            for varname in vars_2d
                @info "  Reading $varname..."
                data = ds[varname][lon_idx, lat_idx, tidx]
                defVar(out, varname, Float32, ("lon", "lat"))[:] = Float32.(data)
            end
        end
    finally
        close(ds)
    end

    @info "Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)"
    return outfile
end

# ---------------------------------------------------------------------------
# Step 2: Load met data from the downloaded NetCDF file
# ---------------------------------------------------------------------------
function load_met_from_netcdf(filepath::String, ::Type{FT}) where {FT}
    @info "Loading met data from $filepath"
    ds = NCDataset(filepath)
    try
        lons = ds["lon"][:]
        lats = ds["lat"][:]
        Nx = length(lons)
        Ny = length(lats)
        Nz = length(ds["lev"][:])

        u_cc  = FT.(ds["u"][:, :, :])
        v_cc  = FT.(ds["v"][:, :, :])
        omega = FT.(ds["omega"][:, :, :])
        delp  = FT.(ds["delp"][:, :, :])

        # Compute column-mean pressure thickness per level
        mean_delp = zeros(FT, Nz)
        for k in 1:Nz
            mean_delp[k] = sum(delp[:, :, k]) / (Nx * Ny)
        end
        @info "  Pressure thickness range: $(round(minimum(mean_delp), sigdigits=3)) - $(round(maximum(mean_delp), sigdigits=3)) Pa"

        # CFL limiter factor
        cfl_limit = FT(0.4)

        # Grid spacing estimates for CFL limiting
        Δx_min = FT(STRIDE * 0.3125 * 111000.0)
        Δy_min = FT(STRIDE * 0.25 * 111000.0)
        u_max = cfl_limit * Δx_min / FT(Δt)
        v_max = cfl_limit * Δy_min / FT(Δt)
        @info "  CFL limit: u_max=$(round(u_max, digits=1)) m/s, v_max=$(round(v_max, digits=1)) m/s"

        # Stagger u to x-faces: (Nx+1, Ny, Nz), periodic
        u = zeros(FT, Nx + 1, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            ip = i == Nx ? 1 : i + 1
            u_val = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
            u[i, j, k] = clamp(u_val, -u_max, u_max)
        end
        u[Nx + 1, :, :] .= u[1, :, :]

        # Stagger v to y-faces: (Nx, Ny+1, Nz), zero at poles
        v = zeros(FT, Nx, Ny + 1, Nz)
        for k in 1:Nz, j in 2:Ny, i in 1:Nx
            v_val = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
            v[i, j, k] = clamp(v_val, -v_max, v_max)
        end

        # w at z-interfaces: (Nx, Ny, Nz+1), convert from omega
        # omega > 0 = downward; advect_z! w > 0 = downward. No sign flip.
        # Apply CFL limiter: |w * Δt / Δz| < 0.5
        w = zeros(FT, Nx, Ny, Nz + 1)
        for k in 2:Nz, j in 1:Ny, i in 1:Nx
            w_raw = (omega[i, j, k - 1] + omega[i, j, k]) / 2
            # Use the thinner of the two adjacent layers for CFL check
            dp_min = min(mean_delp[k - 1], mean_delp[k])
            w_max = cfl_limit * dp_min / FT(Δt)
            w[i, j, k] = clamp(w_raw, -w_max, w_max)
        end

        met_fields = (; u, v, w, pressure_thickness = delp, mean_delp)

        return met_fields, lons, lats, Nx, Ny, Nz
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Step 3: Initialize model and tracers
# ---------------------------------------------------------------------------
function build_model(Nx, Ny, Nz, lons, lats, met_fields, ::Type{FT}) where {FT}
    # Build vertical coordinate from actual GEOS-FP pressure thicknesses
    mean_delp = met_fields.mean_delp
    p_edges = zeros(FT, Nz + 1)
    p_edges[1] = FT(0.0)  # model top
    for k in 1:Nz
        p_edges[k + 1] = p_edges[k] + mean_delp[k]
    end
    @info "  Vertical: p_top=$(p_edges[1]) Pa, p_surface=$(round(p_edges[end], sigdigits=6)) Pa"

    # Pure sigma coordinates: a = 0, b = p/p_surface
    Ps = p_edges[end]
    b_values = p_edges ./ Ps
    a_values = zeros(FT, Nz + 1)

    vc = HybridSigmaPressure(a_values, b_values)

    grid = LatitudeLongitudeGrid(CPU();
        size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(lons[2] - lons[1])),
        latitude  = (FT(lats[1]), FT(lats[end])),
        vertical  = vc)

    @info "Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"
    @info "Longitude: [$(lons[1]), $(lons[end])]"
    @info "Latitude: [$(lats[1]), $(lats[end])]"

    return grid
end

function initialize_tracer_blob(Nx, Ny, Nz, lons, lats, ::Type{FT}) where {FT}
    # CO2-like background: 400 ppm everywhere
    c = fill(FT(400.0), Nx, Ny, Nz)

    # Add a localized Gaussian blob over Europe
    lon_center = FT(10.0)
    lat_center = FT(50.0)
    σ_lon = FT(15.0)
    σ_lat = FT(10.0)
    k_center = max(1, Nz - 5)  # near surface

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Δlon = lons[i] - lon_center
        Δlat = lats[j] - lat_center
        Δk = FT(k - k_center)
        blob = FT(50.0) * exp(-(Δlon^2 / (2 * σ_lon^2) +
                                Δlat^2 / (2 * σ_lat^2) +
                                Δk^2 / FT(8.0)))
        c[i, j, k] += blob
    end

    return (; c = c)
end

# ---------------------------------------------------------------------------
# Step 4: Run the forward model
# ---------------------------------------------------------------------------
function save_output(outfile, tracers, lons, lats, Nz, step, time_hours)
    NCDataset(outfile, isfile(outfile) ? "a" : "c") do ds
        if !haskey(ds.dim, "lon")
            defDim(ds, "lon", length(lons))
            defDim(ds, "lat", length(lats))
            defDim(ds, "lev", Nz)
            defDim(ds, "time", Inf)  # unlimited

            defVar(ds, "lon", Float64, ("lon",))[:] = lons
            defVar(ds, "lat", Float64, ("lat",))[:] = lats
            defVar(ds, "time_hours", Float64, ("time",))

            ds.attrib["description"] = "AtmosTransport forward run output"
            ds.attrib["created"] = string(now())
        end

        tidx = haskey(ds, "time_hours") ? length(ds["time_hours"]) + 1 : 1

        if !haskey(ds, "tracer_c")
            defVar(ds, "tracer_c", Float32, ("lon", "lat", "lev", "time"))
        end

        ds["time_hours"][tidx] = time_hours
        ds["tracer_c"][:, :, :, tidx] = Float32.(tracers.c)
    end
end

function main()
    @info "=" ^ 60
    @info "AtmosTransport — First Forward Run"
    @info "=" ^ 60

    # Step 1: Download test data
    @info "\n--- Step 1: Downloading GEOS-FP test data ---"
    metfile = download_geosfp_test(TEST_DATE, OUTDIR; stride=STRIDE)

    # Step 2: Load met data
    @info "\n--- Step 2: Loading meteorological fields ---"
    met_fields, lons, lats, Nx, Ny, Nz = load_met_from_netcdf(metfile, FT)

    @info "Wind ranges:"
    @info "  u: [$(minimum(met_fields.u)), $(maximum(met_fields.u))] m/s"
    @info "  v: [$(minimum(met_fields.v)), $(maximum(met_fields.v))] m/s"
    @info "  w: [$(minimum(met_fields.w)), $(maximum(met_fields.w))] Pa/s"

    # Step 3: Build model grid
    @info "\n--- Step 3: Building model grid and tracers ---"
    grid = build_model(Nx, Ny, Nz, lons, lats, met_fields, FT)

    tracers = initialize_tracer_blob(Nx, Ny, Nz, lons, lats, FT)
    @info "Initial tracer: min=$(minimum(tracers.c)), max=$(maximum(tracers.c)), " *
          "mean=$(sum(tracers.c) / length(tracers.c))"

    # Step 4: Run the simulation
    @info "\n--- Step 4: Running forward simulation ---"
    @info "  Δt = $(Δt)s, N_hours = $N_HOURS"

    scheme = SlopesAdvection(use_limiter=true)
    N_steps = round(Int, N_HOURS * 3600 / Δt)

    # Remove old output
    isfile(OUTFILE) && rm(OUTFILE)

    # Save initial state
    save_output(OUTFILE, tracers, lons, lats, Nz, 0, 0.0)

    vel = (; u = met_fields.u, v = met_fields.v, w = met_fields.w)

    total_mass_initial = sum(tracers.c)

    for step in 1:N_steps
        # Advection in all three directions (Strang splitting)
        half = Δt / 2
        advect_x!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_z!(tracers, vel, grid, scheme, half)

        advect_z!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_x!(tracers, vel, grid, scheme, half)

        time_hours = step * Δt / 3600.0

        # Save every 3 hours
        if step % round(Int, 3 * 3600 / Δt) == 0
            save_output(OUTFILE, tracers, lons, lats, Nz, step, time_hours)
            total_mass = sum(tracers.c)
            mass_change = abs(total_mass - total_mass_initial) / abs(total_mass_initial)
            @info "  t = $(round(time_hours, digits=1))h: " *
                  "min=$(round(minimum(tracers.c), digits=2)), " *
                  "max=$(round(maximum(tracers.c), digits=2)), " *
                  "mass_change=$(round(mass_change * 100, sigdigits=3))%"
        end
    end

    # Save final state
    save_output(OUTFILE, tracers, lons, lats, Nz, N_steps, N_HOURS)

    total_mass_final = sum(tracers.c)
    mass_conservation = abs(total_mass_final - total_mass_initial) / abs(total_mass_initial)

    @info "\n--- Results ---"
    @info "  Total steps: $N_steps"
    @info "  Final tracer: min=$(round(minimum(tracers.c), digits=2)), " *
          "max=$(round(maximum(tracers.c), digits=2))"
    @info "  Mass conservation: $(round(mass_conservation * 100, sigdigits=3))% change"
    @info "  Output saved to: $OUTFILE"
    @info "=" ^ 60
end

main()
