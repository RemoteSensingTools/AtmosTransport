#!/usr/bin/env julia
# ===========================================================================
# Forward model integration test with downloaded GEOS-FP data
#
# Loads the week of GEOS-FP met data downloaded by download_geosfp_week.jl,
# builds a LatitudeLongitudeGrid, initializes a Gaussian tracer pulse,
# and runs the model forward for 56 time steps (one week at 3-hour intervals)
# using Strang-split advection + diffusion.
#
# Usage:
#   julia --project=. scripts/run_forward_geosfp.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Diffusion
using AtmosTransport.Chemistry
using AtmosTransport.TimeSteppers
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const FT       = Float64
const DATADIR  = joinpath(@__DIR__, "..", "data", "geosfp")
const DATAFILE = joinpath(DATADIR, "geosfp_4x5_20250201_20250207.nc")
const Δt       = 10800.0  # 3-hour outer time step [seconds]

# ---------------------------------------------------------------------------
# Step 1: Load grid and met data from the downloaded NetCDF
# ---------------------------------------------------------------------------
function load_data(filepath::String, ::Type{FT}) where {FT}
    @info "Loading met data from $filepath"
    @info "  File size: $(round(filesize(filepath) / 1024 / 1024, digits=1)) MB"

    ds = NCDataset(filepath)
    lons = ds["lon"][:]
    lats = ds["lat"][:]
    levs = ds["lev"][:]
    time_hours = ds["time"][:]

    Nx = length(lons)
    Ny = length(lats)
    Nz = length(levs)
    Nt = length(time_hours)

    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt"
    @info "  Lon: [$(lons[1]), $(lons[end])], Δlon=$(round(lons[2]-lons[1], digits=4))°"
    @info "  Lat: [$(lats[1]), $(lats[end])], Δlat=$(round(lats[2]-lats[1], digits=4))°"
    @info "  Levels: $Nz"
    @info "  Time steps: $Nt ($(time_hours[1])h to $(time_hours[end])h)"

    close(ds)
    return lons, lats, levs, Nx, Ny, Nz, Nt
end

function load_timestep(filepath::String, tidx::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u_cc    = FT.(ds["u"][:, :, :, tidx])
        v_cc    = FT.(ds["v"][:, :, :, tidx])
        omega   = FT.(ds["omega"][:, :, :, tidx])
        delp    = FT.(ds["delp"][:, :, :, tidx])
        ps      = FT.(ds["ps"][:, :, tidx])

        return u_cc, v_cc, omega, delp, ps
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Step 2: Build staggered velocity fields from cell-center met data
# ---------------------------------------------------------------------------
function stagger_velocities(u_cc, v_cc, omega, delp, Nx, Ny, Nz, grid, ::Type{FT}) where {FT}
    # Compute column-mean pressure thickness for CFL limiting
    mean_delp = zeros(FT, Nz)
    for k in 1:Nz
        mean_delp[k] = sum(delp[:, :, k]) / (Nx * Ny)
    end

    cfl_limit = FT(0.4)

    # Grid spacing estimates for CFL limiting
    Δlon = grid.Δλ
    Δlat = grid.Δφ
    Δx_min = FT(Δlon * 111000.0 * 0.5)  # conservative (mid-latitude cos factor)
    Δy_min = FT(Δlat * 111000.0)
    u_max = cfl_limit * Δx_min / FT(Δt)
    v_max = cfl_limit * Δy_min / FT(Δt)

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
    w = zeros(FT, Nx, Ny, Nz + 1)
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        w_raw = -(omega[i, j, k - 1] + omega[i, j, k]) / 2
        dp_min = min(mean_delp[k - 1], mean_delp[k])
        w_max = cfl_limit * dp_min / FT(Δt)
        w[i, j, k] = clamp(w_raw, -w_max, w_max)
    end

    return (; u, v, w, pressure_thickness = delp, mean_delp)
end

# ---------------------------------------------------------------------------
# Step 3: Build grid
# ---------------------------------------------------------------------------
function build_grid(Nx, Ny, Nz, lons, lats, mean_delp, ::Type{FT}) where {FT}
    # Build vertical coordinate from mean pressure thicknesses
    p_edges = zeros(FT, Nz + 1)
    p_edges[1] = FT(0.0)  # model top
    for k in 1:Nz
        p_edges[k + 1] = p_edges[k] + mean_delp[k]
    end
    @info "  Vertical: p_top=$(p_edges[1]) Pa, p_surface=$(round(p_edges[end], sigdigits=6)) Pa"

    Ps = p_edges[end]
    b_values = p_edges ./ Ps
    a_values = zeros(FT, Nz + 1)
    vc = HybridSigmaPressure(a_values, b_values)

    Δlon = lons[2] - lons[1]
    grid = LatitudeLongitudeGrid(CPU();
        size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
        latitude  = (FT(lats[1]), FT(lats[end])),
        vertical  = vc)

    return grid
end

# ---------------------------------------------------------------------------
# Step 4: Initialize tracer field with Gaussian pulse
# ---------------------------------------------------------------------------
function initialize_tracer(Nx, Ny, Nz, lons, lats, ::Type{FT}) where {FT}
    # Background CO2-like field: 400 ppm everywhere
    c = fill(FT(400.0), Nx, Ny, Nz)

    # Add a localized Gaussian pulse (representing a surface source)
    lon_center = FT(10.0)   # Europe
    lat_center = FT(50.0)
    σ_lon = FT(15.0)
    σ_lat = FT(10.0)
    k_center = max(1, Nz - 5)  # near surface (level 67 of 72)

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
# Step 5: Run the forward model
# ---------------------------------------------------------------------------
function run_forward()
    @info "=" ^ 70
    @info "AtmosTransport — Forward Integration Test (1 week GEOS-FP)"
    @info "=" ^ 70

    if !isfile(DATAFILE)
        error("Met data file not found: $DATAFILE\nRun scripts/download_geosfp_week.jl first.")
    end

    # Load grid info
    lons, lats, levs, Nx, Ny, Nz, Nt = load_data(DATAFILE, FT)

    # Load first timestep to build grid (need mean_delp)
    @info "\n--- Building grid from first timestep ---"
    u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, 1, Nx, Ny, Nz, FT)

    mean_delp = zeros(FT, Nz)
    for k in 1:Nz
        mean_delp[k] = sum(delp[:, :, k]) / (Nx * Ny)
    end
    @info "  Pressure thickness range: $(round(minimum(mean_delp), sigdigits=3)) - $(round(maximum(mean_delp), sigdigits=3)) Pa"

    grid = build_grid(Nx, Ny, Nz, lons, lats, mean_delp, FT)
    @info "Grid: Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz)"
    @info "  Δλ=$(grid.Δλ)°, Δφ=$(grid.Δφ)°"

    # Initialize tracers
    @info "\n--- Initializing tracers ---"
    tracers = initialize_tracer(Nx, Ny, Nz, lons, lats, FT)
    initial_mass = sum(tracers.c)
    @info "  Initial: min=$(round(minimum(tracers.c), digits=2)), " *
          "max=$(round(maximum(tracers.c), digits=2)), " *
          "mean=$(round(sum(tracers.c)/length(tracers.c), digits=2)), " *
          "total_mass=$(round(initial_mass, sigdigits=8))"

    # Set up physics operators
    scheme    = SlopesAdvection(use_limiter=true)
    conv      = TiedtkeConvection()
    diff      = BoundaryLayerDiffusion(Kz_max=100.0)
    chem      = NoChemistry()

    @info "\n--- Physics operators ---"
    @info "  Advection:  SlopesAdvection(use_limiter=true)"
    @info "  Convection: TiedtkeConvection (no-op without conv_mass_flux)"
    @info "  Diffusion:  BoundaryLayerDiffusion(Kz_max=100.0)"
    @info "  Chemistry:  NoChemistry"
    @info "  Δt = $(Δt)s ($(Δt/3600)h)"

    # Time-stepping loop
    @info "\n--- Running forward simulation ($Nt steps) ---"
    wall_start = time()
    half = Δt / 2

    for step in 1:Nt
        t_step_start = time()

        # Load met data for this timestep
        u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, step, Nx, Ny, Nz, FT)
        met_fields = stagger_velocities(u_cc, v_cc, omega, delp, Nx, Ny, Nz, grid, FT)

        vel = (; u = met_fields.u, v = met_fields.v, w = met_fields.w)

        # Symmetric Strang splitting (TM5-style)
        # Forward half-steps
        advect_x!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_z!(tracers, vel, grid, scheme, half)

        # Full steps for convection, diffusion, chemistry
        convect!(tracers, met_fields, grid, conv, Δt)
        diffuse!(tracers, met_fields, grid, diff, Δt)
        apply_chemistry!(tracers, grid, chem, Δt)

        # Backward half-steps
        advect_z!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_x!(tracers, vel, grid, scheme, half)

        # Diagnostics
        c = tracers.c
        c_min = minimum(c)
        c_max = maximum(c)
        c_mean = sum(c) / length(c)
        total_mass = sum(c)
        mass_change = abs(total_mass - initial_mass) / abs(initial_mass) * 100

        elapsed_step = round(time() - t_step_start, digits=2)
        elapsed_total = round(time() - wall_start, digits=1)

        sim_hours = step * Δt / 3600.0
        sim_days = sim_hours / 24.0

        @info "Step $step/$Nt (day $(round(sim_days, digits=2))): " *
              "min=$(round(c_min, digits=2)), " *
              "max=$(round(c_max, digits=2)), " *
              "mean=$(round(c_mean, digits=2)), " *
              "mass_change=$(round(mass_change, sigdigits=3))%, " *
              "wall=$(elapsed_step)s (total $(elapsed_total)s)"
    end

    wall_total = round(time() - wall_start, digits=1)

    # Final diagnostics
    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "=" ^ 70

    c = tracers.c
    final_mass = sum(c)
    mass_conservation = abs(final_mass - initial_mass) / abs(initial_mass) * 100
    n_negative = count(x -> x < 0, c)
    min_val = minimum(c)

    @info "  Total steps: $Nt"
    @info "  Simulation time: $(Nt * Δt / 3600 / 24) days"
    @info "  Wall time: $(wall_total)s"
    @info ""
    @info "  Tracer statistics:"
    @info "    min  = $(round(min_val, digits=4))"
    @info "    max  = $(round(maximum(c), digits=4))"
    @info "    mean = $(round(sum(c)/length(c), digits=4))"
    @info ""
    @info "  Mass conservation:"
    @info "    Initial mass: $(round(initial_mass, sigdigits=8))"
    @info "    Final mass:   $(round(final_mass, sigdigits=8))"
    @info "    Change:       $(round(mass_conservation, sigdigits=3))%"
    @info ""
    @info "  Positivity:"
    @info "    Negative cells: $n_negative / $(length(c))"
    @info "    Min value: $(round(min_val, digits=6))"
    @info ""

    # Verification checks
    passed = true
    if mass_conservation > 1.0
        @warn "Mass conservation exceeds 1%: $(round(mass_conservation, digits=3))%"
        passed = false
    else
        @info "  ✓ Mass conservation within 1%"
    end

    if n_negative > 0 && min_val < -1.0
        @warn "Significant negative values detected: min=$(round(min_val, digits=4))"
        passed = false
    else
        @info "  ✓ Tracer near-positive (min=$(round(min_val, digits=4)))"
    end

    if passed
        @info "\n  ★ Integration test PASSED ★"
    else
        @warn "\n  Integration test FAILED — see warnings above"
    end

    @info "=" ^ 70
end

run_forward()
