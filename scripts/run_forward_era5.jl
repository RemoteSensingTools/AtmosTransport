#!/usr/bin/env julia
# ===========================================================================
# Forward transport simulation using ERA5 model-level data
#
# Uses ERA5 L88 (model levels 50-137) with hybrid sigma-pressure coordinates.
# Surface pressure derived from LNSP (log surface pressure).
#
# Features:
# - Universal met infrastructure: build_vertical_coordinate() with level_range
# - Continuity-consistent omega from horizontal wind divergence (TM5 approach)
# - Per-column surface pressure for accurate layer thickness and mass tracking
# - CFL-adaptive subcycled advection (SlopesAdvection)
# - Strang-split X→Y→Z→Z→Y→X
# - NetCDF output with snapshots
#
# Usage:
#   julia --project=. scripts/run_forward_era5.jl
#
# Environment variables:
#   DT          — outer time step [s] (default: 21600 = 6h, matches data)
#   LEVEL_TOP   — topmost model level (default: 50)
#   LEVEL_BOT   — bottommost model level (default: 137)
#   USE_GPU     — "true" for GPU (default: false)
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients, compute_continuity_omega
using NCDatasets
using Dates
using Printf

const USE_GPU     = get(ENV, "USE_GPU", "false") == "true"
const FT          = Float64
const DT          = parse(Float64, get(ENV, "DT", "900"))
const LEVEL_TOP   = parse(Int, get(ENV, "LEVEL_TOP", "50"))
const LEVEL_BOT   = parse(Int, get(ENV, "LEVEL_BOT", "137"))
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT

const DATADIR = expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607")
const OUTDIR  = expanduser("~/data/output/era5_ml_test")

# ---------------------------------------------------------------------------
# ERA5 data reader
#
# ERA5 conventions:
#   - Latitude: 90°N → 90°S (N→S) — we flip to S→N
#   - Longitude: 0° → 359°
#   - Level: 50 (top) → 137 (surface) — already top-to-bottom
#   - Omega (w): Pa/s, positive downward
#   - NCDatasets reads as (lon, lat, level) when time is indexed
# ---------------------------------------------------------------------------
function load_era5_timestep(filepath::String, tidx::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u_raw     = FT.(ds["u"][:, :, :, tidx])      # (lon, lat, lev)
        v_raw     = FT.(ds["v"][:, :, :, tidx])
        omega_raw = FT.(ds["w"][:, :, :, tidx])       # omega, Pa/s
        lnsp_raw  = FT.(ds["lnsp"][:, :, tidx])       # (lon, lat)

        # Flip latitude: ERA5 is N→S, we need S→N
        u     = u_raw[:, end:-1:1, :]
        v     = v_raw[:, end:-1:1, :]
        omega = omega_raw[:, end:-1:1, :]
        lnsp  = lnsp_raw[:, end:-1:1]

        ps = exp.(lnsp)  # surface pressure [Pa]
        return u, v, omega, ps
    finally
        close(ds)
    end
end

function get_era5_grid_info(filepath::String, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        lons = FT.(ds["longitude"][:])
        lats = FT.(ds["latitude"][:])
        levs = ds["model_level"][:]
        Nt   = length(ds["valid_time"][:])
        return lons, reverse(lats), levs, length(lons), length(lats), length(levs), Nt
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Stagger horizontal velocities to faces, then compute continuity-consistent
# vertical velocity from the divergence of the staggered horizontal winds.
# ---------------------------------------------------------------------------
function stagger_and_compute_omega(u_cc, v_cc, ps_2d, grid, ::Type{FT}) where {FT}
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

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

    w = compute_continuity_omega(u, v, grid; p_surface = ps_2d)

    return (; u, v, w, p_surface = ps_2d)
end

# ---------------------------------------------------------------------------
# Mass computation using actual per-column surface pressure
# ---------------------------------------------------------------------------
function total_mass(c, grid, ps_2d, A_coeff, B_coeff)
    FT = eltype(c)
    Nx, Ny, Nz = size(c)
    mass = zero(FT)
    g = grid.gravity

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        area = cell_area(i, j, grid)
        Δp = (A_coeff[k + 1] - A_coeff[k]) + (B_coeff[k + 1] - B_coeff[k]) * ps_2d[i, j]
        mass += c[i, j, k] * area * Δp / g
    end
    return mass
end

# ---------------------------------------------------------------------------
# Collect daily file paths
# ---------------------------------------------------------------------------
function find_era5_files(datadir::String)
    files = String[]
    for f in sort(readdir(datadir))
        if startswith(f, "era5_ml_") && endswith(f, ".nc") && !contains(f, "_tmp")
            push!(files, joinpath(datadir, f))
        end
    end
    return files
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
function run_forward()
    @info "=" ^ 70
    @info "AtmosTransport — ERA5 Model-Level Forward Run"
    @info "=" ^ 70

    files = find_era5_files(DATADIR)
    isempty(files) && error("No ERA5 files found in $DATADIR")
    @info "  Found $(length(files)) daily files"
    @info "  Level range: $(LEVEL_TOP)-$(LEVEL_BOT) ($(length(LEVEL_RANGE)) levels)"
    @info "  DT = $(DT)s ($(DT/3600)h)"

    # Build vertical coordinate for level subset
    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=LEVEL_RANGE)
    A_full, B_full = load_vertical_coefficients(config; FT)
    A_coeff = A_full[LEVEL_TOP:LEVEL_BOT+1]
    B_coeff = B_full[LEVEL_TOP:LEVEL_BOT+1]
    Nz_vc = length(LEVEL_RANGE)
    @info "  Vertical: $(Nz_vc) levels (HybridSigmaPressure)"
    @info "  A coeff range: [$(minimum(A_coeff)), $(maximum(A_coeff))] Pa"
    @info "  B coeff range: [$(minimum(B_coeff)), $(maximum(B_coeff))]"

    # Load grid info from first file
    lons, lats, levs, Nx, Ny, Nz, Nt_per_file = get_era5_grid_info(files[1], FT)
    @assert Nz == Nz_vc "File has $Nz levels but expected $Nz_vc"
    Δlon = lons[2] - lons[1]

    @info "\n--- Grid Info ---"
    @info "  Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt/file=$Nt_per_file"
    @info "  Lon: [$(lons[1])°, $(lons[end])°], Δ=$(Δlon)°"
    @info "  Lat: [$(lats[1])°, $(lats[end])°]"

    params = load_parameters(FT)
    pp = params.planet

    arch = USE_GPU ? GPU() : CPU()

    grid = LatitudeLongitudeGrid(arch;
        FT,
        size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius,
        gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)
    @info "  Grid constructed: $(typeof(grid))"

    # Initialize tracer: uniform 420 ppm CO₂
    c = fill(FT(420.0), Nx, Ny, Nz)
    tracers = (; c)

    # Compute initial mass using first-timestep surface pressure
    u0, v0, omega0, ps0 = load_era5_timestep(files[1], 1, FT)
    initial_mass = total_mass(c, grid, ps0, A_coeff, B_coeff)
    @info "\n--- Initial State ---"
    @info "  c: min=$(minimum(c)), max=$(maximum(c)), mean=$(sum(c)/length(c))"
    @info "  ps: min=$(round(minimum(ps0), sigdigits=5)) Pa, max=$(round(maximum(ps0), sigdigits=5)) Pa"
    @info "  Initial mass = $(round(initial_mass, sigdigits=10))"

    half_dt = DT / 2

    # Prepare output
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "era5_ml_forward.nc")

    # Each met timestep covers 6 hours; with DT < 6h, we sub-step within each
    met_interval = 21600.0  # 6 hours between met snapshots
    steps_per_met = max(1, round(Int, met_interval / DT))
    total_met_steps = Nt_per_file * length(files)
    total_steps = total_met_steps * steps_per_met
    @info "\n--- Running $total_steps steps ($(length(files)) days, DT=$(DT)s, $steps_per_met steps/met) ---"
    @info "  Using TM5-faithful mass-flux advection"

    wall_start = time()
    step = 0

    for (day_idx, filepath) in enumerate(files)
        day_str = basename(filepath)
        @info "\nDay $day_idx: $day_str"

        for tidx in 1:Nt_per_file
            # Load met once per 6-hour window
            u_cc, v_cc, omega, ps = load_era5_timestep(filepath, tidx, FT)
            vel = stagger_and_compute_omega(u_cc, v_cc, ps, grid, FT)

            # Compute pressure thickness and air mass (once per met window)
            Δp = Advection._build_Δz_3d(grid, ps)
            m = compute_air_mass(Δp, grid)

            # Compute mass fluxes for one half-timestep (TM5 dynam0)
            mf = compute_mass_fluxes(vel.u, vel.v, grid, Δp, half_dt)

            for sub in 1:steps_per_met
                step += 1
                t_step_start = time()

                # CFL diagnostics (mass-based)
                cfl_x = max_cfl_massflux_x(mf.am, m)
                cfl_z = max_cfl_massflux_z(mf.cm, m)

                # TM5-faithful mass-flux Strang split (X→Y→Z→Z→Y→X)
                # m tracks continuously through the split — NOT reset.
                strang_split_massflux!(tracers, m, mf.am, mf.bm, mf.cm,
                                       grid, true; cfl_limit = FT(0.95))

                # Diagnostics
                if sub == steps_per_met || step == 1
                    c_now = tracers.c
                    mass_now = total_mass(c_now, grid, ps, A_coeff, B_coeff)
                    Δmass_pct = (mass_now - initial_mass) / abs(initial_mass) * 100

                    elapsed = round(time() - t_step_start, digits = 2)
                    sim_hours = step * DT / 3600.0

                    @info @sprintf("  Step %d/%d (%.1fh): min=%.4f max=%.4f mean=%.4f Δmass=%.6f%% CFL_x=%.1f CFL_z=%.1f [%.1fs]",
                        step, total_steps, sim_hours,
                        minimum(c_now), maximum(c_now), sum(c_now)/length(c_now),
                        Δmass_pct, cfl_x, cfl_z, elapsed)
                end
            end
        end
    end

    wall_total = round(time() - wall_start, digits = 1)

    # Final diagnostics
    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "=" ^ 70
    c_final = tracers.c
    ps_last = load_era5_timestep(files[end], Nt_per_file, FT)[4]
    final_mass = total_mass(c_final, grid, ps_last, A_coeff, B_coeff)
    Δmass_pct = (final_mass - initial_mass) / abs(initial_mass) * 100
    n_negative = count(x -> x < 0, c_final)

    @info "  Total steps: $step"
    @info "  Simulation time: $(step * DT / 3600 / 24) days"
    @info "  Wall time: $(wall_total)s"
    @info ""
    @info "  Tracer statistics:"
    @info "    min  = $(round(minimum(c_final), digits=4))"
    @info "    max  = $(round(maximum(c_final), digits=4))"
    @info "    mean = $(round(sum(c_final)/length(c_final), digits=4))"
    @info ""
    @info "  Mass conservation:"
    @info "    Initial: $(round(initial_mass, sigdigits=10))"
    @info "    Final:   $(round(final_mass, sigdigits=10))"
    @info "    Change:  $(@sprintf("%.6f%%", Δmass_pct))"
    @info ""
    @info "  Positivity: $n_negative / $(length(c_final)) negative cells"

    # Save final state
    NCDataset(outfile, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)

        defVar(ds, "lon", FT, ("lon",))[:] = lons
        defVar(ds, "lat", FT, ("lat",))[:] = lats
        defVar(ds, "lev", FT, ("lev",))[:] = Float64.(levs)
        v_co2 = defVar(ds, "co2", FT, ("lon", "lat", "lev"))
        v_co2[:, :, :] = c_final
    end

    @info "  Output: $outfile"
    @info "=" ^ 70
end

run_forward()
