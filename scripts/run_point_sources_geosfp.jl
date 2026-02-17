#!/usr/bin/env julia
# ===========================================================================
# Point-source CO2 experiment: 3 locations (Amazon, Sahara, Alaska)
#
# - Background CO2 = 400 ppm everywhere.
# - Inject CO2 at 3 point sources for the first 24 hours (8 steps × 3 h).
# - Then run transport only for the rest of the week (56 steps total).
# - Writes NetCDF every step for easy video (56 frames).
#
# Usage:
#   julia --project=. scripts/run_point_sources_geosfp.jl
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Convection
using AtmosTransportModel.Diffusion
using AtmosTransportModel.Chemistry
using NCDatasets
using Dates

const FT = Float64
const DATAFILE = expanduser("~/data/metDrivers/geosfp/geosfp_4x5_20250201_20250207.nc")
const OUTFILE  = expanduser("~/data/metDrivers/geosfp/point_sources_co2.nc")
const Δt       = 10800.0   # 3-hour step [s]
const Nt       = 56        # 7 days
const STEPS_INJECT = 8     # inject for first 24 h
const PPM_PER_STEP = 18.0  # ppm added per step per source (lowest 3 levels)

# Point sources (lon °E, lat °N): Amazon, Sahara, Alaska
const SOURCES_LONLAT = [
    (-60.0,  -5.0),   # Amazon
    ( 20.0,  25.0),   # Sahara
    (-150.0, 65.0),   # Alaska
]

# ---------------------------------------------------------------------------
# Load / grid / stagger (same logic as run_forward_geosfp.jl)
# ---------------------------------------------------------------------------
function load_data(filepath::String, ::Type{FT}) where FT
    ds = NCDataset(filepath)
    lons = ds["lon"][:]
    lats = ds["lat"][:]
    levs = ds["lev"][:]
    Nx, Ny, Nz = length(lons), length(lats), length(levs)
    Nt = length(ds["time"])
    close(ds)
    return lons, lats, levs, Nx, Ny, Nz, Nt
end

function load_timestep(filepath::String, tidx::Int, Nx, Ny, Nz, ::Type{FT}) where FT
    ds = NCDataset(filepath)
    u_cc  = FT.(ds["u"][:, :, :, tidx])
    v_cc  = FT.(ds["v"][:, :, :, tidx])
    omega = FT.(ds["omega"][:, :, :, tidx])
    delp  = FT.(ds["delp"][:, :, :, tidx])
    ps    = FT.(ds["ps"][:, :, tidx])
    close(ds)
    return u_cc, v_cc, omega, delp, ps
end

function stagger_velocities(u_cc, v_cc, omega, delp, Nx, Ny, Nz, grid, ::Type{FT}) where FT
    mean_delp = [sum(delp[:, :, k]) / (Nx * Ny) for k in 1:Nz]
    cfl_limit = FT(0.4)
    Δlon, Δlat = grid.Δλ, grid.Δφ
    Δx_min = FT(Δlon * 111000.0 * 0.5)
    Δy_min = FT(Δlat * 111000.0)
    u_max = cfl_limit * Δx_min / FT(Δt)
    v_max = cfl_limit * Δy_min / FT(Δt)

    u = zeros(FT, Nx + 1, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = clamp((u_cc[i, j, k] + u_cc[ip, j, k]) / 2, -u_max, u_max)
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    v = zeros(FT, Nx, Ny + 1, Nz)
    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = clamp((v_cc[i, j-1, k] + v_cc[i, j, k]) / 2, -v_max, v_max)
    end

    w = zeros(FT, Nx, Ny, Nz + 1)
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        w_raw = -(omega[i, j, k-1] + omega[i, j, k]) / 2
        dp_min = min(mean_delp[k-1], mean_delp[k])
        w_max = cfl_limit * dp_min / FT(Δt)
        w[i, j, k] = clamp(w_raw, -w_max, w_max)
    end
    return (; u, v, w, pressure_thickness = delp, mean_delp)
end

function build_grid(Nx, Ny, Nz, lons, lats, mean_delp, ::Type{FT}) where FT
    p_edges = zeros(FT, Nz + 1)
    for k in 1:Nz
        p_edges[k + 1] = p_edges[k] + mean_delp[k]
    end
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

function find_nearest_ij(lons, lats, lon_deg, lat_deg)
    i = argmin(abs.(lons .- lon_deg))
    j = argmin(abs.(lats .- lat_deg))
    return i, j
end

function initialize_tracer_point_sources(Nx, Ny, Nz, lons, lats, ::Type{FT}) where FT
    c = fill(FT(400.0), Nx, Ny, Nz)  # background 400 ppm
    return (; c = c)
end

function inject_sources!(tracers, source_ijk, Nz; n_levels = 3, ppm_per_step = PPM_PER_STEP)
    k_lo = max(1, Nz - n_levels + 1)
    for (i, j) in source_ijk
        for k in k_lo:Nz
            tracers.c[i, j, k] += FT(ppm_per_step)
        end
    end
    return nothing
end

function save_snapshot(outfile, tracers, lons, lats, levs, step, time_hours)
    Nx, Ny, Nz = size(tracers.c)
    create = !isfile(outfile)
    NCDataset(outfile, create ? "c" : "a") do ds
        if create
            defDim(ds, "lon", Nx)
            defDim(ds, "lat", Ny)
            defDim(ds, "lev", Nz)
            defDim(ds, "time", Inf)
            defVar(ds, "lon", FT, ("lon",))[:] = lons
            defVar(ds, "lat", FT, ("lat",))[:] = lats
            defVar(ds, "lev", FT, ("lev",))[:] = levs
            defVar(ds, "time_hours", FT, ("time",))
            defVar(ds, "co2_ppm", FT, ("lon", "lat", "lev", "time"))
            ds.attrib["description"] = "Point-source CO2 experiment (Amazon, Sahara, Alaska)"
            ds.attrib["created"] = string(now())
        end
        n_time = size(ds["time_hours"], 1)
        ds["time_hours"][n_time + 1] = time_hours
        ds["co2_ppm"][:, :, :, n_time + 1] = tracers.c
    end
end

function main()
    @info "=" ^ 70
    @info "Point-source CO2 experiment (Amazon, Sahara, Alaska)"
    @info "=" ^ 70

    if !isfile(DATAFILE)
        error("Met data not found: $DATAFILE\nRun scripts/download_geosfp_week.jl first.")
    end

    lons, lats, levs, Nx, Ny, Nz, Nt_file = load_data(DATAFILE, FT)
    @info "Grid: $Nx × $Ny × $Nz, $Nt_file timesteps"

    # Build grid from first timestep
    u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, 1, Nx, Ny, Nz, FT)
    mean_delp = [sum(delp[:, :, k]) / (Nx * Ny) for k in 1:Nz]
    grid = build_grid(Nx, Ny, Nz, lons, lats, mean_delp, FT)

    # Source grid indices (nearest to each lon/lat)
    source_ijk = [find_nearest_ij(lons, lats, lon, lat) for (lon, lat) in SOURCES_LONLAT]
    for (idx, (lon, lat)) in enumerate(SOURCES_LONLAT)
        i, j = source_ijk[idx]
        @info "  Source $idx ($lon°E, $lat°N) → grid (i=$i, j=$j) lon=$(lons[i]) lat=$(lats[j])"
    end

    tracers = initialize_tracer_point_sources(Nx, Ny, Nz, lons, lats, FT)
    scheme = SlopesAdvection(use_limiter = true)
    conv   = TiedtkeConvection()
    diff   = BoundaryLayerDiffusion(Kz_max = 100.0)
    chem   = NoChemistry()

    isfile(OUTFILE) && rm(OUTFILE)
    save_snapshot(OUTFILE, tracers, lons, lats, collect(FT, levs), 0, 0.0)

    half = Δt / 2
    wall_start = time()

    for step in 1:Nt
        u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, step, Nx, Ny, Nz, FT)
        met_fields = stagger_velocities(u_cc, v_cc, omega, delp, Nx, Ny, Nz, grid, FT)
        vel = (; u = met_fields.u, v = met_fields.v, w = met_fields.w)

        # Inject at point sources for first 24 h
        if step <= STEPS_INJECT
            inject_sources!(tracers, source_ijk, Nz)
        end

        # Strang splitting
        advect_x!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_z!(tracers, vel, grid, scheme, half)
        convect!(tracers, met_fields, grid, conv, Δt)
        diffuse!(tracers, met_fields, grid, diff, Δt)
        apply_chemistry!(tracers, grid, chem, Δt)
        advect_z!(tracers, vel, grid, scheme, half)
        advect_y!(tracers, vel, grid, scheme, half)
        advect_x!(tracers, vel, grid, scheme, half)

        time_hours = step * Δt / 3600.0
        save_snapshot(OUTFILE, tracers, lons, lats, collect(FT, levs), step, time_hours)

        if step % 10 == 0 || step == Nt
            @info "Step $step/$Nt (day $(round(time_hours/24, digits=2))): " *
                  "min=$(round(minimum(tracers.c), digits=1)), " *
                  "max=$(round(maximum(tracers.c), digits=1))"
        end
    end

    @info "Done. Wall time: $(round(time() - wall_start, digits=1)) s"
    @info "Output: $OUTFILE ($(round(filesize(OUTFILE)/1e6, digits=2)) MB)"
    @info "Make video: python scripts/make_point_sources_video.py"
    @info "=" ^ 70
end

main()
