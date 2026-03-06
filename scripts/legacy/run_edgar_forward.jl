#!/usr/bin/env julia
# ===========================================================================
# Forward transport of EDGAR CO2 emissions using GEOS-FP meteorology
#
# Starts from zero CO2 everywhere, injects EDGAR v8.0 emissions at every
# time step, and advects with CFL-adaptive subcycled SlopesAdvection.
# Saves surface-layer CO2 at every step for animation.
#
# Usage:
#   julia --project=. scripts/run_edgar_forward.jl
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
using AtmosTransport.Sources: load_edgar_co2, apply_surface_flux!, GriddedEmission
using AtmosTransport.IO: default_met_config, build_vertical_coordinate
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const USE_GPU     = get(ENV, "USE_GPU", "false") == "true"
const USE_FLOAT32 = get(ENV, "USE_FLOAT32", "false") == "true"
const DT          = parse(Float64, get(ENV, "DT", "10800"))

const FT = USE_FLOAT32 ? Float32 : Float64

if USE_GPU
    using CUDA
end

const METFILE   = expanduser("~/data/metDrivers/geosfp/geosfp_4x5_20250201_20250207.nc")
const EDGARFILE = expanduser("~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc")
const OUTDIR    = expanduser("~/data/output/edgar_forward")

# ---------------------------------------------------------------------------
# Load grid metadata from NetCDF
# ---------------------------------------------------------------------------
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
    return lons, lats, levs, Nx, Ny, Nz, Nt
end

# ---------------------------------------------------------------------------
# Load a single timestep of met fields
# ---------------------------------------------------------------------------
function load_timestep(filepath::String, tidx::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u_cc  = FT.(ds["u"][:, :, :, tidx])
        v_cc  = FT.(ds["v"][:, :, :, tidx])
        omega = FT.(ds["omega"][:, :, :, tidx])
        delp  = FT.(ds["delp"][:, :, :, tidx])
        ps    = FT.(ds["ps"][:, :, tidx])
        return u_cc, v_cc, omega, delp, ps
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Stagger velocities to faces (no clamping)
# ---------------------------------------------------------------------------
function stagger_velocities(u_cc, v_cc, omega, Nx, Ny, Nz, ::Type{FT}) where {FT}
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end

    # omega > 0 = downward; advect_z! w > 0 = downward. No sign flip.
    w = zeros(FT, Nx, Ny, Nz + 1)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        w[i, j, k] = (omega[i, j, k - 1] + omega[i, j, k]) / 2
    end

    return (; u, v, w)
end

# ---------------------------------------------------------------------------
# Pressure-weighted mass diagnostic
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
# Compute expected cumulative emission in ppm·(pressure-volume) units
# for mass-budget comparison.
#
# Total emitted mass rate [kg/s] from the EDGAR source, converted to
# the same pressure-volume-weighted integral as pressure_weighted_mass.
# ---------------------------------------------------------------------------
function total_emission_rate_kg(source::GriddedEmission{FT}, grid) where {FT}
    Nx, Ny = grid.Nx, grid.Ny
    rate = zero(FT)
    @inbounds for j in 1:Ny, i in 1:Nx
        rate += source.flux[i, j] * cell_area(i, j, grid)
    end
    return rate
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
function run_edgar_forward()
    @info "=" ^ 70
    @info "AtmosTransport — EDGAR CO2 Forward Transport"
    @info "=" ^ 70
    @info "  USE_GPU=$USE_GPU, USE_FLOAT32=$USE_FLOAT32, DT=$(DT)s"

    for f in (METFILE, EDGARFILE)
        isfile(f) || error("File not found: $f")
    end

    params = load_parameters(FT)
    pp = params.planet

    # Vertical coordinate from GEOS-FP L72 coefficients
    @info "\n--- Building vertical coordinate ---"
    config = default_met_config("geosfp")
    vc = build_vertical_coordinate(config; FT = FT)
    @info "  Vertical: $(n_levels(vc)) levels (HybridSigmaPressure)"

    # Grid metadata
    lons, lats, levs, Nx, Ny, Nz, Nt = load_grid_info(METFILE, FT)

    arch = USE_GPU ? GPU() : CPU()
    ArrayType = array_type(arch)

    # Build model grid
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

    # Load EDGAR emissions and regrid to model grid
    @info "\n--- Loading EDGAR CO2 emissions ---"
    edgar_source = load_edgar_co2(EDGARFILE, grid; year=2022)
    emission_rate = total_emission_rate_kg(edgar_source, grid)
    @info "  Total emission rate: $(round(emission_rate, digits=2)) kg/s"

    # Initialize tracers: zero CO2 everywhere
    @info "\n--- Initializing tracers (zero CO2) ---"
    tracers = (; co2 = ArrayType(zeros(FT, Nx, Ny, Nz)))

    scheme = SlopesAdvection(use_limiter = true)
    half_dt = DT / 2

    # Prepare output
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "edgar_forward_snapshots.nc")
    @info "\n--- Output file: $outfile ---"

    NCDataset(outfile, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "time", Inf)  # unlimited for per-step snapshots

        defVar(ds, "lon", FT, ("lon",);
               attrib = Dict("units" => "degrees_east", "long_name" => "Longitude"))
        defVar(ds, "lat", FT, ("lat",);
               attrib = Dict("units" => "degrees_north", "long_name" => "Latitude"))
        defVar(ds, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since 2025-02-01 00:00:00",
                             "long_name" => "Simulation time"))
        defVar(ds, "co2_surface", FT, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm",
                             "long_name" => "Surface-layer CO2 from EDGAR"))
        defVar(ds, "mass_total", Float64, ("time",);
               attrib = Dict("units" => "ppm·Pa·m³",
                             "long_name" => "Total pressure-weighted CO2 mass"))
        defVar(ds, "mass_expected", Float64, ("time",);
               attrib = Dict("long_name" => "Expected cumulative emitted mass"))

        ds["lon"][:] = Array(grid.λᶜ)
        ds["lat"][:] = Array(grid.φᶜ)

        # Write initial state (step 0)
        ds["time"][1] = 0.0
        ds["co2_surface"][:, :, 1] = zeros(FT, Nx, Ny)
        ds["mass_total"][1] = 0.0
        ds["mass_expected"][1] = 0.0

        # Cumulative emitted mass for budget tracking (pressure-volume units)
        cumulative_emitted = zero(Float64)
        mass_before_emission = zero(Float64)

        @info "\n--- Running forward simulation ($Nt steps, DT=$(DT)s) ---"
        wall_start = time()

        for step in 1:Nt
            t_step_start = time()

            # Load met data
            u_cc, v_cc, omega, delp, ps = load_timestep(METFILE, step, Nx, Ny, Nz, FT)
            met = stagger_velocities(u_cc, v_cc, omega, Nx, Ny, Nz, FT)
            vel = (;
                u = ArrayType(met.u),
                v = ArrayType(met.v),
                w = ArrayType(met.w),
            )

            # Inject EDGAR emissions into surface layer
            apply_surface_flux!(tracers, edgar_source, grid, DT)

            # Track emitted mass: difference before/after injection
            c_after_emission = Array(tracers.co2)
            mass_after_emission = pressure_weighted_mass(c_after_emission, grid)
            cumulative_emitted += (mass_after_emission - mass_before_emission)

            # Strang-split advection: X→Y→Z→Z→Y→X
            advect_x_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)
            advect_y_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)
            advect_z_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)
            advect_z_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)
            advect_y_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)
            advect_x_subcycled!(tracers, vel, grid, scheme, half_dt; cfl_limit = 0.95)

            # Diagnostics
            c_now = Array(tracers.co2)
            c_sfc = c_now[:, :, end]
            mass_now = pressure_weighted_mass(c_now, grid)
            mass_before_emission = mass_now

            # Save snapshot
            tidx_out = step + 1  # offset by 1 for initial state
            sim_hours = step * DT / 3600.0
            ds["time"][tidx_out] = sim_hours
            ds["co2_surface"][:, :, tidx_out] = c_sfc
            ds["mass_total"][tidx_out] = mass_now
            ds["mass_expected"][tidx_out] = cumulative_emitted

            n_negative = count(x -> x < 0, c_now)
            elapsed_step = round(time() - t_step_start, digits = 2)
            elapsed_total = round(time() - wall_start, digits = 1)
            sim_days = sim_hours / 24.0

            mass_err_pct = cumulative_emitted > 0 ?
                abs(mass_now - cumulative_emitted) / cumulative_emitted * 100 : 0.0

            @info "Step $step/$Nt (day $(round(sim_days, digits=2))): " *
                  "sfc_max=$(round(maximum(c_sfc), digits=4)) ppm, " *
                  "mean=$(round(sum(c_now)/length(c_now), digits=6)) ppm, " *
                  "mass_err=$(round(mass_err_pct, sigdigits=3))%, " *
                  "neg=$n_negative, " *
                  "wall=$(elapsed_step)s (total $(elapsed_total)s)"
        end

        wall_total = round(time() - wall_start, digits = 1)

        # Final summary
        c_final = Array(tracers.co2)
        final_mass = pressure_weighted_mass(c_final, grid)
        n_negative = count(x -> x < 0, c_final)

        @info "\n" * "=" ^ 70
        @info "Simulation complete!"
        @info "=" ^ 70
        @info "  Total steps: $Nt"
        @info "  Simulation time: $(Nt * DT / 3600 / 24) days"
        @info "  Wall time: $(wall_total)s"
        @info ""
        @info "  Final tracer statistics:"
        @info "    min  = $(round(minimum(c_final), digits=6)) ppm"
        @info "    max  = $(round(maximum(c_final), digits=6)) ppm"
        @info "    mean = $(round(sum(c_final)/length(c_final), digits=6)) ppm"
        @info ""
        @info "  Mass conservation:"
        @info "    Final mass:    $(round(final_mass, sigdigits=8))"
        @info "    Expected mass: $(round(cumulative_emitted, sigdigits=8))"
        mass_err = cumulative_emitted > 0 ?
            abs(final_mass - cumulative_emitted) / cumulative_emitted * 100 : 0.0
        @info "    Error:         $(round(mass_err, sigdigits=3))%"
        @info ""
        @info "  Positivity: $n_negative / $(length(c_final)) negative cells"
        @info ""
        @info "  Output: $outfile"
        @info "=" ^ 70
    end
end

run_edgar_forward()
