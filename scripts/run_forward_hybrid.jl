#!/usr/bin/env julia
# ===========================================================================
# Forward transport simulation using universal met data infrastructure
#
# End-to-end forward run with:
# - build_vertical_coordinate() from met config (L72 hybrid sigma-pressure)
# - Manual NetCDF read (single-file GEOS-FP, no MetDataSource collection)
# - CFL-adaptive subcycled advection (SlopesAdvection)
# - Strang-split X→Y→Z→Z→Y→X
# - Pressure-weighted mass conservation diagnostics
# - NetCDF output
#
# Usage:
#   julia --project=. scripts/run_forward_hybrid.jl
#
# Environment variables:
#   USE_GPU     — "true" for GPU (default: false)
#   USE_FLOAT32 — "true" for Float32 (default: false)
#   DT          — outer time step [s] (default: 10800 to match 3-hourly met)
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
const USE_GPU     = get(ENV, "USE_GPU", "false") == "true"
const USE_FLOAT32 = get(ENV, "USE_FLOAT32", "false") == "true"
const DT          = parse(Float64, get(ENV, "DT", "10800"))

const FT = USE_FLOAT32 ? Float32 : Float64

if USE_GPU
    using CUDA
end

# Data paths: GEOS-FP 4x5 test file
const DATAFILE = expanduser("~/data/metDrivers/geosfp/geosfp_4x5_20250201_20250207.nc")
const OUTDIR   = expanduser("~/data/output/geosfp_hybrid_test")

# ---------------------------------------------------------------------------
# Step 1: Load grid metadata from NetCDF
# ---------------------------------------------------------------------------
"""
    load_grid_info(filepath, FT)

Read lon, lat, lev arrays and dimensions from the GEOS-FP NetCDF file.
Returns (lons, lats, levs, Nx, Ny, Nz, Nt).
"""
function load_grid_info(filepath::String, ::Type{FT}) where {FT}
    @info "Loading grid info from $filepath"
    ds = NCDataset(filepath)
    lons = FT.(ds["lon"][:])
    lats = FT.(ds["lat"][:])
    levs = ds["lev"][:]
    time_hours = ds["time"][:]
    close(ds)

    Nx = length(lons)
    Ny = length(lats)
    Nz = length(levs)
    Nt = length(time_hours)

    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt"
    @info "  Lon: [$(lons[1]), $(lons[end])], Δlon=$(round(lons[2] - lons[1], digits=4))°"
    @info "  Lat: [$(lats[1]), $(lats[end])], Δlat=$(round(lats[2] - lats[1], digits=4))°"
    @info "  Time steps: $Nt (3-hourly, 7 days)"

    return lons, lats, levs, Nx, Ny, Nz, Nt
end

"""
    load_timestep(filepath, tidx, Nx, Ny, Nz, FT)

Read u, v, omega, delp, ps for a single time step.
"""
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
# Step 2: Stagger velocities to faces
#
# Physics:
# - u at x-faces (Nx+1, Ny, Nz): periodic in longitude
# - v at y-faces (Nx, Ny+1, Nz): zero at poles (bounded latitude)
# - w at z-interfaces (Nx, Ny, Nz+1): omega>0 downward, w>0 upward → negate
#   Average omega from level centers to interfaces for w
# ---------------------------------------------------------------------------
function stagger_velocities(u_cc, v_cc, omega, Nx, Ny, Nz, ::Type{FT}) where {FT}
    # u → x-faces (Nx+1, Ny, Nz), periodic
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    # v → y-faces (Nx, Ny+1, Nz), zero at poles
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end

    # omega → w at z-interfaces (Nx, Ny, Nz+1)
    # omega > 0 = downward (toward surface, increasing k).
    # advect_z! uses w > 0 = downward (increasing k), so no sign flip.
    w = zeros(FT, Nx, Ny, Nz + 1)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        w[i, j, k] = (omega[i, j, k - 1] + omega[i, j, k]) / 2
    end

    return (; u, v, w)
end

# ---------------------------------------------------------------------------
# Step 3: Pressure-weighted mass (conservation metric)
#
# Mass = sum over cells of c[i,j,k] * cell_volume(i,j,k)
#       = sum of c * (area * Δp / g)
# Uses grid.reference_pressure for Δp (hybrid coordinate). For column-varying
# ps, a more accurate metric would use actual ps(i,j) per column.
# ---------------------------------------------------------------------------
function pressure_weighted_mass(c, grid; p_surface=nothing)
    FT = eltype(c)
    p_surf = p_surface === nothing ? grid.reference_pressure : FT(p_surface)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    mass = zero(FT)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        vol = cell_volume(i, j, k, grid; p_surface = p_surf)
        mass += c[i, j, k] * vol
    end
    return mass
end

# ---------------------------------------------------------------------------
# Step 4: Main simulation
# ---------------------------------------------------------------------------
function run_forward()
    @info "=" ^ 70
    @info "AtmosTransport — Forward Hybrid (universal met infrastructure)"
    @info "=" ^ 70
    @info "  USE_GPU=$(USE_GPU), USE_FLOAT32=$(USE_FLOAT32), DT=$(DT)s"

    if !isfile(DATAFILE)
        error("Met data file not found: $DATAFILE")
    end

    # Load parameters for planet constants
    params = load_parameters(FT)
    pp = params.planet

    # Load met config and build vertical coordinate from L72 coefficients
    @info "\n--- Building vertical coordinate from met config ---"
    config = default_met_config("geosfp")
    vc = build_vertical_coordinate(config; FT = FT)
    @info "  Vertical: $(n_levels(vc)) levels (HybridSigmaPressure)"

    # Load grid info from NetCDF
    lons, lats, levs, Nx, Ny, Nz, Nt = load_grid_info(DATAFILE, FT)

    # Architecture and array type
    arch = USE_GPU ? GPU() : CPU()
    ArrayType = array_type(arch)

    # Build grid with universal vertical coordinate
    @info "\n--- Building grid ---"
    Δlon = lons[2] - lons[1]
    lon_west = FT(lons[1])
    lon_east = FT(lons[end]) + FT(Δlon)
    grid = LatitudeLongitudeGrid(arch;
        FT,
        size = (Nx, Ny, Nz),
        longitude = (lon_west, lon_east),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius,
        gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)
    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"

    # Initialize tracers: uniform 420 ppm CO2
    @info "\n--- Initializing tracers ---"
    c = ArrayType(fill(FT(420.0), Nx, Ny, Nz))
    tracers = (; c)

    # Pressure-weighted initial mass
    c_cpu = Array(c)
    initial_mass = pressure_weighted_mass(c_cpu, grid)
    @info "  Initial: min=$(round(minimum(c_cpu), digits=2)), " *
          "max=$(round(maximum(c_cpu), digits=2)), " *
          "mass=$(round(initial_mass, sigdigits=8))"

    half_dt = FT(DT / 2)

    # ---------------------------------------------------------------
    # TM5-style precomputation: build geometry cache + allocate buffers
    # ---------------------------------------------------------------
    @info "\n--- Precomputing grid geometry cache (TM5 dynam0 style) ---"
    Δp_cpu  = Array{FT}(undef, Nx, Ny, Nz)
    Δp_dev  = ArrayType(zeros(FT, Nx, Ny, Nz))
    u_dev   = ArrayType(zeros(FT, Nx + 1, Ny, Nz))
    v_dev   = ArrayType(zeros(FT, Nx, Ny + 1, Nz))
    m_dev   = ArrayType(zeros(FT, Nx, Ny, Nz))
    am_dev  = ArrayType(zeros(FT, Nx + 1, Ny, Nz))
    bm_dev  = ArrayType(zeros(FT, Nx, Ny + 1, Nz))
    cm_dev  = ArrayType(zeros(FT, Nx, Ny, Nz + 1))

    gc = build_geometry_cache(grid, Δp_dev)
    ws = allocate_massflux_workspace(m_dev, am_dev, bm_dev, cm_dev)

    @info "  Geometry cache + workspace allocated (zero per-step allocation)"

    @info "\n--- Running forward simulation ($Nt steps, DT=$(DT)s) ---"
    @info "  Using TM5-faithful mass-flux advection"
    wall_start = time()

    for step in 1:Nt
        t_step_start = time()

        # Load met data for this timestep
        u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, step, Nx, Ny, Nz, FT)
        met = stagger_velocities(u_cc, v_cc, omega, Nx, Ny, Nz, FT)

        # Fill pre-allocated CPU → device arrays
        Advection._build_Δz_3d!(Δp_cpu, grid, ps)
        copyto!(Δp_dev, Δp_cpu)
        copyto!(u_dev, met.u)
        copyto!(v_dev, met.v)

        # Precompute air mass and mass fluxes using cached geometry
        compute_air_mass!(m_dev, Δp_dev, gc)
        compute_mass_fluxes!(am_dev, bm_dev, cm_dev, u_dev, v_dev,
                              gc, Δp_dev, half_dt)

        # CFL diagnostics (mass-based)
        cfl_x = max_cfl_massflux_x(am_dev, m_dev, ws.cfl_x)

        # TM5-faithful mass-flux Strang split (X→Y→Z→Z→Y→X)
        strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                               grid, true, ws; cfl_limit = FT(0.95))

        # Diagnostics
        c_now = Array(tracers.c)
        c_min = minimum(c_now)
        c_max = maximum(c_now)
        c_mean = sum(c_now) / length(c_now)
        mass_now = pressure_weighted_mass(c_now, grid)
        mass_change_pct = abs(mass_now - initial_mass) / abs(initial_mass) * 100

        elapsed_step = round(time() - t_step_start, digits = 2)
        elapsed_total = round(time() - wall_start, digits = 1)
        sim_hours = step * DT / 3600.0
        sim_days = sim_hours / 24.0

        @info "Step $step/$Nt (day $(round(sim_days, digits=2))): " *
              "min=$(round(c_min, digits=2)), " *
              "max=$(round(c_max, digits=2)), " *
              "mean=$(round(c_mean, digits=2)), " *
              "mass_change=$(round(mass_change_pct, sigdigits=3))%, " *
              "CFL_x=$(round(cfl_x, digits=2)), " *
              "wall=$(elapsed_step)s (total $(elapsed_total)s)"
    end

    wall_total = round(time() - wall_start, digits = 1)

    # Final diagnostics
    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "=" ^ 70
    c_final = Array(tracers.c)
    final_mass = pressure_weighted_mass(c_final, grid)
    mass_conservation_pct = abs(final_mass - initial_mass) / abs(initial_mass) * 100
    n_negative = count(x -> x < 0, c_final)
    min_val = minimum(c_final)

    @info "  Total steps: $Nt"
    @info "  Simulation time: $(Nt * DT / 3600 / 24) days"
    @info "  Wall time: $(wall_total)s"
    @info ""
    @info "  Tracer statistics:"
    @info "    min  = $(round(min_val, digits=4))"
    @info "    max  = $(round(maximum(c_final), digits=4))"
    @info "    mean = $(round(sum(c_final) / length(c_final), digits=4))"
    @info ""
    @info "  Mass conservation (pressure-weighted):"
    @info "    Initial mass: $(round(initial_mass, sigdigits=8))"
    @info "    Final mass:   $(round(final_mass, sigdigits=8))"
    @info "    Change:       $(round(mass_conservation_pct, sigdigits=3))%"
    @info ""
    @info "  Positivity:"
    @info "    Negative cells: $n_negative / $(length(c_final))"
    @info ""

    # Save output to NetCDF
    @info "\n--- Saving output ---"
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "geosfp_hybrid_forward.nc")

    NCDataset(outfile, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defDim(ds, "time", 1)  # final state only

        defVar(ds, "lon", FT, ("lon",))
        defVar(ds, "lat", FT, ("lat",))
        defVar(ds, "lev", FT, ("lev",))
        defVar(ds, "time", FT, ("time",))
        defVar(ds, "co2", FT, ("lon", "lat", "lev", "time"))

        ds["lon"][:] = Array(grid.λᶜ)
        ds["lat"][:] = Array(grid.φᶜ)
        ds["lev"][:] = [znode(k, grid, Center()) for k in 1:Nz]
        ds["time"][1] = FT(Nt * DT / 3600)  # simulation end time [hours]
        ds["co2"][:, :, :, 1] = c_final
    end

    @info "  Output: $outfile"
    @info "=" ^ 70
end

run_forward()
