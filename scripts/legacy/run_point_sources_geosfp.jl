#!/usr/bin/env julia
# ===========================================================================
# Point-source CO2 experiment: 3 locations (Amazon, Sahara, Alaska)
#
# GEOS-FP met data on a regular latitude-longitude grid. All coordinates
# use cell-center convention (λᶜ, φᶜ): tracers and diagnostics are at cell
# centers; spacing Δx uses φᶜ so CFL is evaluated at the same points (TM5-style).
# See docs/TM5_GRID_REFERENCES.md for TM5 repo and KNMI TR-294 (reduced grid).
#
# - No background: tracer starts at 0 ppm.
# - Inject 200 kg CO2/s at each of 3 point sources for the first 24 h (8 steps).
# - Unit conversion: mass flux → ppm in surface layer using cell air mass
#   (delp * area / g). Tracer c = mole fraction × 1e6 (ppm).
# - Then run transport only for the rest of the week (56 steps total).
# - Writes NetCDF every step for video (lon/lat in file = grid cell centers).
#
# Usage:
#   julia --project=. scripts/run_point_sources_geosfp.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Diffusion
using AtmosTransport.Chemistry
using NCDatasets
using Dates

const FT = Float64
const DATAFILE = expanduser("~/data/metDrivers/geosfp/geosfp_4x5_20250201_20250207.nc")
const OUTFILE  = expanduser("~/data/metDrivers/geosfp/point_sources_co2.nc")
const Δt       = 10800.0   # 3-hour step [s]
const Nt       = 56        # 7 days
const STEPS_INJECT = 8     # inject for first 24 h

# Emission: 200 kg CO2/s per source
const EMISSION_KG_S = FT(200.0)
# Unit conversion: c in ppm (μmol/mol). Δc_ppm = ΔM_CO2 * 1e6 * (M_air/M_CO2) / m_air_kg
const G_MS2 = FT(9.81)           # m/s²
const M_AIR_KG = FT(28.97e-3)     # kg/mol
const M_CO2_KG = FT(44.01e-3)    # kg/mol
const MOLAR_RATIO = M_AIR_KG / M_CO2_KG  # M_air/M_CO2

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
    # CFL uses cell-center spacing: Δx(i,j) = R*cos(φᶜ[j])*Δλ (so at poles φᶜ=±90°, Δx=0).
    # TM5-style: limit zonal wind per column so u*Δt/Δx ≤ cfl; at poles set u_max=0.
    Δx_floor = FT(1.0)  # avoid division by zero at poles
    u_max_j = [cfl_limit * max(Δx(1, j, grid), Δx_floor) / FT(Δt) for j in 1:Ny]
    Δy_unif = Δy(1, 1, grid)  # Δy uniform in latitude
    v_max = cfl_limit * Δy_unif / FT(Δt)

    u = zeros(FT, Nx + 1, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = clamp((u_cc[i, j, k] + u_cc[ip, j, k]) / 2, -u_max_j[j], u_max_j[j])
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

"""Find (i,j) of the cell whose center (λᶜ[i], φᶜ[j]) is nearest to (lon_deg, lat_deg)."""
function find_nearest_ij(grid, lon_deg, lat_deg)
    λᶜ = grid.λᶜ
    φᶜ = grid.φᶜ
    i = argmin(abs.(λᶜ .- lon_deg))
    j = argmin(abs.(φᶜ .- lat_deg))
    return i, j
end

function initialize_tracer_point_sources(Nx, Ny, Nz, ::Type{FT}) where FT
    c = zeros(FT, Nx, Ny, Nz)  # no background
    return (; c = c)
end

"""
Inject 200 kg CO2/s into the surface layer at each source for this step.
ΔM = emission_kg_s * Δt [kg]. Air mass in cell (i,j,Nz): m_air = delp[i,j,Nz]*A/g [kg].
Δc_ppm = ΔM * 1e6 * (M_air/M_CO2) / m_air so that added mass = (Δc/1e6)*(M_CO2/M_air)*m_air = ΔM.
"""
function inject_sources!(tracers, source_ijk, grid, delp, step_Δt)
    Nz = size(tracers.c, 3)
    ΔM_kg = EMISSION_KG_S * step_Δt
    for (i, j) in source_ijk
        area_m2 = cell_area(i, j, grid)
        delp_ij = delp[i, j, Nz]  # Pa
        m_air_kg = delp_ij * area_m2 / G_MS2
        m_air_kg <= 0 && continue
        Δc_ppm = ΔM_kg * FT(1e6) * MOLAR_RATIO / m_air_kg
        tracers.c[i, j, Nz] += Δc_ppm
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
            ds.attrib["description"] = "Point-source CO2: 200 kg/s at Amazon, Sahara, Alaska; no background; c in ppm"
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

    # Source grid indices: use cell-center coordinates (λᶜ, φᶜ)
    source_ijk = [find_nearest_ij(grid, lon, lat) for (lon, lat) in SOURCES_LONLAT]
    for (idx, (lon, lat)) in enumerate(SOURCES_LONLAT)
        i, j = source_ijk[idx]
        @info "  Source $idx ($(lon)°E, $(lat)°N) → grid (i=$i, j=$j) λᶜ=$(grid.λᶜ[i]) φᶜ=$(grid.φᶜ[j])"
    end

    tracers = initialize_tracer_point_sources(Nx, Ny, Nz, FT)
    scheme = SlopesAdvection(use_limiter = true)
    conv   = TiedtkeConvection()
    diff   = BoundaryLayerDiffusion(Kz_max = 100.0)
    chem   = NoChemistry()

    isfile(OUTFILE) && rm(OUTFILE)
    # NetCDF coordinates = grid cell centers (λᶜ, φᶜ)
    save_snapshot(OUTFILE, tracers, grid.λᶜ, grid.φᶜ, collect(FT, levs), 0, 0.0)

    half = Δt / 2
    wall_start = time()
    mass_ref = nothing       # sum(c) after injection (not conserved when Δx varies with lat)
    mass_vol_ref = nothing   # volume-weighted sum (proper conserved quantity for slopes scheme)

    for step in 1:Nt
        u_cc, v_cc, omega, delp, ps = load_timestep(DATAFILE, step, Nx, Ny, Nz, FT)
        met_fields = stagger_velocities(u_cc, v_cc, omega, delp, Nx, Ny, Nz, grid, FT)
        vel = (; u = met_fields.u, v = met_fields.v, w = met_fields.w)

        # Inject 200 kg CO2/s at each source for first 24 h (no clamping)
        if step <= STEPS_INJECT
            inject_sources!(tracers, source_ijk, grid, delp, Δt)
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

        # Mass diagnostics: slopes scheme conserves volume-weighted sum (c*Δx*Δy*Δz), not sum(c)
        # when Δx varies with latitude. Ref = after injection.
        if step == STEPS_INJECT
            mass_ref = sum(tracers.c)
            mass_vol_ref = sum(tracers.c[i, j, k] * cell_volume(i, j, k, grid)
                            for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        end
        total_mass = sum(tracers.c)
        total_mass_vol = sum(tracers.c[i, j, k] * cell_volume(i, j, k, grid)
                             for i in 1:Nx, j in 1:Ny, k in 1:Nz)

        time_hours = step * Δt / 3600.0
        save_snapshot(OUTFILE, tracers, grid.λᶜ, grid.φᶜ, collect(FT, levs), step, time_hours)

        if step % 10 == 0 || step == Nt
            msg = "Step $step/$Nt (day $(round(time_hours/24, digits=2))): " *
                  "min=$(round(minimum(tracers.c), digits=1)), max=$(round(maximum(tracers.c), digits=1)), " *
                  "sum(c)=$(round(total_mass, sigdigits=6))"
            if mass_vol_ref !== nothing && mass_vol_ref > 0
                vol_change_pct = abs(total_mass_vol - mass_vol_ref) / mass_vol_ref * 100
                msg *= ", vol_weighted_change=$(round(vol_change_pct, sigdigits=3))%"
            end
            @info msg
        end
    end

    # Mass: slopes scheme conserves sum(c*Δx), sum(c*Δy), sum(c*Δz) per 1D step; with splitting
    # and variable Δx(lat), sum(c) and volume-weighted sum can drift (grid uses reference mean_delp).
    final_mass = sum(tracers.c)
    final_mass_vol = sum(tracers.c[i, j, k] * cell_volume(i, j, k, grid)
                         for i in 1:Nx, j in 1:Ny, k in 1:Nz)
    if mass_vol_ref !== nothing && mass_vol_ref > 0
        vol_conservation_pct = abs(final_mass_vol - mass_vol_ref) / mass_vol_ref * 100
        @info "Mass (ref = after step $STEPS_INJECT):"
        @info "  Volume-weighted (ref grid): change=$(round(vol_conservation_pct, sigdigits=3))%"
        @info "  sum(c): ref=$(round(mass_ref, sigdigits=6)), final=$(round(final_mass, sigdigits=6))"
    end

    @info "Done. Wall time: $(round(time() - wall_start, digits=1)) s"
    @info "Output: $OUTFILE ($(round(filesize(OUTFILE)/1e6, digits=2)) MB)"
    @info "Make video: python scripts/make_point_sources_video.py"
    @info "=" ^ 70
end

main()
