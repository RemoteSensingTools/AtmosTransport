#!/usr/bin/env julia
# ===========================================================================
# Production forward run: ERA5 JJA 2024 with EDGAR CO2 emissions
#
# Configurable via environment variables:
#   ERA5_DIR    — directory with ERA5 monthly NetCDF files
#   EDGAR_FILE  — EDGAR emission NetCDF file
#   OUTPUT_DIR  — output directory
#   RESOLUTION  — grid spacing in degrees (default: auto-detect from data)
#   DT          — time step in seconds (default: 900)
#   USE_FLOAT32 — "true" for Float32 (saves memory)
#   USE_GPU     — "true" for GPU execution
#   CHECKPOINT  — checkpoint file to resume from (optional)
#
# Usage:
#   julia --project=. scripts/run_production_era5_jja2024.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Sources
using AtmosTransport.Parameters
using NCDatasets
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const FT = get(ENV, "USE_FLOAT32", "false") == "true" ? Float32 : Float64
const PARAMS = load_parameters(FT)

const ERA5_DIR   = get(ENV, "ERA5_DIR",
    expanduser("~/data/metDrivers/era5/era5_025deg_20240601_20240831"))
const EDGAR_FILE = get(ENV, "EDGAR_FILE",
    expanduser("~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc"))
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR",
    expanduser("~/data/output/production_jja2024"))
const Δt         = parse(Float64, get(ENV, "DT", "900"))
const USE_GPU    = get(ENV, "USE_GPU", "false") == "true"
const CHECKPOINT = get(ENV, "CHECKPOINT", "")

const START_DATE = DateTime(get(ENV, "START_DATE", "2024-06-01T00:00:00"))
const END_DATE   = DateTime(get(ENV, "END_DATE",   "2024-08-31T18:00:00"))
const MET_INTERVAL = Hour(parse(Int, get(ENV, "MET_INTERVAL_HOURS", "6")))
const SAVE_INTERVAL = Hour(parse(Int, get(ENV, "SAVE_INTERVAL_HOURS", "6")))
const CHECKPOINT_INTERVAL = Day(parse(Int, get(ENV, "CHECKPOINT_INTERVAL_DAYS", "7")))

if USE_GPU; using CUDA; end

# ---------------------------------------------------------------------------
# ERA5 file discovery and data loading
# ---------------------------------------------------------------------------
function find_era5_files(dir::String)
    pl_files = Dict{String, String}()
    sfc_files = Dict{String, String}()

    if !isdir(dir)
        @warn "ERA5 directory not found: $dir"
        return pl_files, sfc_files
    end

    for f in readdir(dir; join=true)
        endswith(f, ".nc") || continue
        bn = basename(f)

        # Monthly format: era5_pl_YYYYMM.nc
        m = match(r"era5_pl_(\d{6})\.nc", bn)
        if m !== nothing
            pl_files[m.captures[1]] = f
            continue
        end
        m = match(r"era5_sfc_(\d{6})\.nc", bn)
        if m !== nothing
            sfc_files[m.captures[1]] = f
            continue
        end

        # Generic pressure-level files (combined)
        if occursin("pressure_level", bn) || occursin("pressure-level", bn)
            pl_files["combined"] = f
            continue
        end
        # Generic single-level / surface files
        if occursin("single_level", bn) || occursin("surface", bn)
            sfc_files["combined"] = f
            continue
        end
        # Catch-all for files starting with era5_pl / era5_pressure
        if startswith(bn, "era5_pressure") || startswith(bn, "era5_pl")
            if !haskey(pl_files, "combined")
                pl_files["combined"] = f
            end
        end
    end

    return pl_files, sfc_files
end

"""Load ERA5 fields for one time step from a NetCDF file."""
function load_era5_timestep(pl_file::String, sfc_file::Union{String,Nothing},
                            time_idx::Int, ::Type{FT}) where FT
    ds_pl = NCDataset(pl_file)

    lon_key = haskey(ds_pl, "longitude") ? "longitude" : "lon"
    lat_key = haskey(ds_pl, "latitude") ? "latitude" : "lat"
    lev_key = haskey(ds_pl, "pressure_level") ? "pressure_level" :
              haskey(ds_pl, "level") ? "level" : "lev"

    lons_raw = FT.(ds_pl[lon_key][:])
    lats_raw = FT.(ds_pl[lat_key][:])
    levels   = ds_pl[lev_key][:]

    Nx, Ny, Nz = length(lons_raw), length(lats_raw), length(levels)

    u_key = haskey(ds_pl, "u") ? "u" : "U"
    v_key = haskey(ds_pl, "v") ? "v" : "V"
    w_key = haskey(ds_pl, "w") ? "w" : "W"

    ndims_u = ndims(ds_pl[u_key])
    if ndims_u == 4
        u_raw = FT.(ds_pl[u_key][:, :, :, time_idx])
        v_raw = FT.(ds_pl[v_key][:, :, :, time_idx])
        w_raw = FT.(ds_pl[w_key][:, :, :, time_idx])
    elseif ndims_u == 3
        u_raw = FT.(ds_pl[u_key][:, :, :])
        v_raw = FT.(ds_pl[v_key][:, :, :])
        w_raw = FT.(ds_pl[w_key][:, :, :])
    else
        error("Unexpected u dimensions: $ndims_u")
    end
    close(ds_pl)

    sp_raw = nothing
    if sfc_file !== nothing && isfile(sfc_file)
        ds_sl = NCDataset(sfc_file)
        sp_key = haskey(ds_sl, "sp") ? "sp" :
                 haskey(ds_sl, "SP") ? "SP" : "surface_pressure"
        if haskey(ds_sl, sp_key)
            ndims_sp = ndims(ds_sl[sp_key])
            sp_raw = ndims_sp >= 3 ? FT.(ds_sl[sp_key][:, :, time_idx]) :
                                     FT.(ds_sl[sp_key][:, :])
        end
        close(ds_sl)
    end

    # Coordinate transforms: longitude 0..360 → -180..180
    if lons_raw[1] >= 0 && lons_raw[end] > 180
        shift_idx = findfirst(l -> l >= 180, lons_raw)
        if shift_idx !== nothing
            shift_n = Nx - shift_idx + 1
            lons_raw = circshift(lons_raw, shift_n)
            lons_raw[1:shift_n] .-= FT(360)
            u_raw = circshift(u_raw, (shift_n, 0, 0))
            v_raw = circshift(v_raw, (shift_n, 0, 0))
            w_raw = circshift(w_raw, (shift_n, 0, 0))
            sp_raw !== nothing && (sp_raw = circshift(sp_raw, (shift_n, 0)))
        end
    end

    # Latitude N→S → S→N
    if lats_raw[1] > lats_raw[end]
        lats_raw = reverse(lats_raw)
        u_raw = u_raw[:, end:-1:1, :]
        v_raw = v_raw[:, end:-1:1, :]
        w_raw = w_raw[:, end:-1:1, :]
        sp_raw !== nothing && (sp_raw = sp_raw[:, end:-1:1])
    end

    # Levels: surface-first → top-first
    if levels[1] > levels[end]
        levels = reverse(levels)
        u_raw = u_raw[:, :, end:-1:1]
        v_raw = v_raw[:, :, end:-1:1]
        w_raw = w_raw[:, :, end:-1:1]
    end

    # Handle below-surface levels: extrapolate the lowest above-surface wind
    # downward to avoid sharp velocity discontinuities at terrain boundaries.
    if sp_raw !== nothing
        levels_Pa = FT.(levels) .* 100
        @inbounds for j in 1:Ny, i in 1:Nx
            sp_ij = sp_raw[i, j]
            lowest_above = 0
            for k in Nz:-1:1
                if levels_Pa[k] <= sp_ij
                    lowest_above = k
                    break
                end
            end
            lowest_above == 0 && continue

            for k in (lowest_above + 1):Nz
                u_raw[i, j, k] = u_raw[i, j, lowest_above]
                v_raw[i, j, k] = v_raw[i, j, lowest_above]
                w_raw[i, j, k] = FT(0)
            end
        end
    end

    return (; u=u_raw, v=v_raw, w=w_raw, sp=sp_raw,
              lons=lons_raw, lats=lats_raw, levels=FT.(levels), Nx, Ny, Nz)
end

# ---------------------------------------------------------------------------
# Grid construction (only once)
# ---------------------------------------------------------------------------
function build_grid(data, ::Type{FT}) where FT
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    p_levels_Pa = data.levels .* FT(100)

    p_edges = zeros(FT, Nz + 1)
    p_edges[1] = FT(0)
    for k in 1:Nz-1
        p_edges[k+1] = (p_levels_Pa[k] + p_levels_Pa[k+1]) / 2
    end
    p_edges[Nz+1] = PARAMS.planet.reference_surface_pressure

    Ps = p_edges[end]
    b_values = p_edges ./ Ps
    a_values = zeros(FT, Nz + 1)
    vc = HybridSigmaPressure(a_values, b_values)

    Δlon = data.lons[2] - data.lons[1]
    arch = USE_GPU ? GPU() : CPU()
    pp = PARAMS.planet

    grid = LatitudeLongitudeGrid(arch;
        FT, size=(Nx, Ny, Nz),
        longitude=(FT(data.lons[1]) - Δlon/2, FT(data.lons[end]) + Δlon/2),
        latitude=(FT(-90), FT(90)),
        vertical=vc,
        radius=pp.radius, gravity=pp.gravity,
        reference_pressure=pp.reference_surface_pressure)

    @info "Grid: $(Nx)×$(Ny)×$(Nz), Δlon=$(round(Δlon, digits=4))°"
    if grid.reduced_grid !== nothing
        n_red = count(>(1), grid.reduced_grid.cluster_sizes)
        @info "  Reduced grid: $n_red latitudes with clustering"
    end

    return grid, p_edges
end

# ---------------------------------------------------------------------------
# Wind staggering (called every 6h when met data updates)
# ---------------------------------------------------------------------------
function stagger_winds!(data, grid, p_edges, ::Type{FT}) where FT
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    u_cc, v_cc, omega_cc = data.u, data.v, data.w

    # Stagger u to x-faces (average of neighbors, periodic in x)
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u[Nx + 1, :, :] .= u[1, :, :]

    # Stagger v to y-faces (average of neighbors, zero at poles)
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j-1, k] + v_cc[i, j, k]) / 2
    end

    # Stagger omega (Pa/s) to z-faces (average of neighbors, zero at boundaries)
    w = zeros(FT, Nx, Ny, Nz + 1)
    @inbounds for j in 1:Ny, i in 1:Nx
        for k in 2:Nz
            w[i, j, k] = (omega_cc[i, j, k-1] + omega_cc[i, j, k]) / 2
        end
    end

    ArrayType = array_type(grid.architecture)
    return (; u=ArrayType(u), v=ArrayType(v), w=ArrayType(w))
end

# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------
function init_output(outfile, grid, p_edges)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    lons = Array(grid.λᶜ)
    lats = Array(grid.φᶜ)

    NCDataset(outfile, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defDim(ds, "time", Inf)

        defVar(ds, "lon", Float64, ("lon",))[:] = lons
        defVar(ds, "lat", Float64, ("lat",))[:] = lats
        defVar(ds, "lev", Float64, ("lev",))[:] = p_edges[1:end-1] .+ diff(p_edges) ./ 2
        defVar(ds, "time_hours", Float64, ("time",))
        defVar(ds, "co2", Float32, ("lon", "lat", "lev", "time");
               attrib=Dict("units" => "ppm", "long_name" => "CO2 mixing ratio"))
        defVar(ds, "xco2", Float32, ("lon", "lat", "time");
               attrib=Dict("units" => "ppm", "long_name" => "column-mean CO2"))
        defVar(ds, "total_mass_ppm", Float64, ("time",);
               attrib=Dict("long_name" => "global physical mass (sum c*volume_weight)"))

        ds.attrib["description"] = "AtmosTransport Production Run: JJA 2024"
        ds.attrib["created"] = string(now())
        ds.attrib["time_step_seconds"] = Δt
        ds.attrib["advection_scheme"] = "SlopesAdvection(use_limiter=true)"
        ds.attrib["emissions"] = "EDGAR v8.0 CO2 2022"
    end
end

function save_snapshot!(outfile, tracers, vol_cpu, p_edges, time_hours)
    c_cpu = Array(tracers.co2)
    Nx, Ny, Nz = size(c_cpu)

    dp = diff(p_edges)
    xco2 = zeros(Float32, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        total_p = zero(Float64)
        weighted = zero(Float64)
        for k in 1:Nz
            weighted += c_cpu[i, j, k] * dp[k]
            total_p += dp[k]
        end
        xco2[i, j] = Float32(weighted / total_p)
    end

    phys_mass = sum(Float64.(c_cpu) .* vol_cpu)

    NCDataset(outfile, "a") do ds
        tidx = length(ds["time_hours"]) + 1
        ds["time_hours"][tidx] = time_hours
        ds["co2"][:, :, :, tidx] = Float32.(c_cpu)
        ds["xco2"][:, :, tidx] = xco2
        ds["total_mass_ppm"][tidx] = phys_mass
    end
end

function save_checkpoint(checkpoint_dir, tracers, sim_time, step)
    mkpath(checkpoint_dir)
    fname = joinpath(checkpoint_dir, "checkpoint_step$(step).nc")
    c_cpu = Array(tracers.co2)
    NCDataset(fname, "c") do ds
        Nx, Ny, Nz = size(c_cpu)
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defVar(ds, "co2", Float64, ("lon", "lat", "lev"))[:, :, :] = c_cpu
        ds.attrib["sim_time"] = string(sim_time)
        ds.attrib["step"] = step
        ds.attrib["dt"] = Δt
    end
    @info "  Checkpoint saved: $fname"
end

function load_checkpoint(checkpoint_file, arch)
    ds = NCDataset(checkpoint_file)
    c = Float64.(ds["co2"][:, :, :])
    step = ds.attrib["step"]
    sim_time = DateTime(ds.attrib["sim_time"])
    close(ds)
    ArrayType = array_type(arch)
    return (co2 = ArrayType(FT.(c)),), sim_time, step
end

# ---------------------------------------------------------------------------
# Met data time index mapping
# ---------------------------------------------------------------------------
function datetime_to_met_file_and_idx(dt::DateTime, era5_dir::String,
                                      pl_files, sfc_files)
    ym = Dates.format(dt, "yyyymm")
    if haskey(pl_files, ym)
        pl_f = pl_files[ym]
    elseif haskey(pl_files, "combined")
        pl_f = pl_files["combined"]
    else
        error("No ERA5 PL file for $ym in $era5_dir")
    end

    sfc_f = get(sfc_files, ym, get(sfc_files, "combined", nothing))

    ds = NCDataset(pl_f)
    time_key = "valid_time" in keys(ds.dim) ? "valid_time" :
               "time" in keys(ds.dim) ? "time" : nothing

    time_idx = 1
    if time_key !== nothing && haskey(ds, time_key)
        times = ds[time_key][:]
        target_unix = Dates.datetime2unix(dt)
        diffs = abs.(Dates.datetime2unix.(times) .- target_unix)
        time_idx = argmin(diffs)
        min_diff_hours = minimum(diffs) / 3600
        if min_diff_hours > 12
            @warn "Closest met time is $(round(min_diff_hours, digits=1))h away — recycling available data"
        end
    end
    close(ds)

    return pl_f, sfc_f, time_idx
end

# ---------------------------------------------------------------------------
# Terrain mask for pressure-level grids
# ---------------------------------------------------------------------------
"""
Precompute per-column index of the lowest above-surface level.
Returns an (Nx, Ny) array where value k means levels k:Nz are below surface
(k = Nz+1 means all levels are above surface).
"""
function compute_terrain_mask(sp::AbstractMatrix, levels_Pa, Nx, Ny, Nz)
    lowest_above = fill(Nz, Nx, Ny)  # index of lowest level above surface
    @inbounds for j in 1:Ny, i in 1:Nx
        sp_ij = sp[i, j]
        for k in Nz:-1:1
            if levels_Pa[k] <= sp_ij
                lowest_above[i, j] = k
                break
            end
        end
    end
    return lowest_above
end

"""
Reset below-surface cells to the nearest above-surface value.
Prevents oscillation growth at topographic boundaries.
"""
function apply_terrain_mask!(c::AbstractArray{T,3}, lowest_above, Nx, Ny, Nz) where T
    @inbounds for j in 1:Ny, i in 1:Nx
        la = lowest_above[i, j]
        la >= Nz && continue
        val = c[i, j, la]
        for k in (la + 1):Nz
            c[i, j, k] = val
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Volume weights for mass diagnostics
# ---------------------------------------------------------------------------
function compute_volume_weights(grid, p_edges, ::Type{FT}) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    φᶠ_cpu = Array(grid.φᶠ)
    vol = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dsinphi = abs(sind(φᶠ_cpu[j+1]) - sind(φᶠ_cpu[j]))
        dp_k = p_edges[k+1] - p_edges[k]
        vol[i, j, k] = dsinphi * dp_k
    end
    return vol
end

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
function main()
    @info "=" ^ 70
    @info "AtmosTransport — Production ERA5 Forward Run (JJA 2024)"
    @info "=" ^ 70

    # Discover ERA5 files
    pl_files, sfc_files = find_era5_files(ERA5_DIR)
    @info "ERA5 files found: $(length(pl_files)) PL, $(length(sfc_files)) SFC"

    # If no monthly files, try loading existing 2° weekly data as fallback
    if isempty(pl_files)
        fallback_pl = joinpath(dirname(dirname(ERA5_DIR)),
            "era5_pressure_levels_20250201_20250207.nc")
        project_pl = joinpath(@__DIR__, "..", "data", "era5",
            "era5_pressure_levels_20250201_20250207.nc")

        if isfile(fallback_pl)
            pl_files["combined"] = fallback_pl
            @info "Using fallback ERA5 file: $fallback_pl"
        elseif isfile(project_pl)
            pl_files["combined"] = project_pl
            @info "Using project ERA5 file: $project_pl"
        else
            error("No ERA5 data found. Run scripts/download_era5.jl first.")
        end
    end

    # Load first timestep to determine grid geometry
    first_key = first(sort(collect(keys(pl_files))))
    first_pl = pl_files[first_key]
    first_sfc = get(sfc_files, first_key, nothing)
    @info "Loading initial ERA5 from: $first_pl"
    data = load_era5_timestep(first_pl, first_sfc, 1, FT)

    # Build grid (once)
    grid, p_edges = build_grid(data, FT)
    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz

    # Stagger initial winds
    vel = stagger_winds!(data, grid, p_edges, FT)

    # Compute terrain mask from surface pressure
    levels_Pa = data.levels .* FT(100)
    terrain_mask = if data.sp !== nothing
        mask = compute_terrain_mask(data.sp, levels_Pa, Nx, Ny, Nz)
        n_below = count(x -> x < Nz, mask)
        @info "Terrain mask: $n_below columns have below-surface levels"
        mask
    else
        nothing
    end

    # Load EDGAR emissions
    edgar = if isfile(EDGAR_FILE)
        @info "Loading EDGAR CO2 emissions from: $EDGAR_FILE"
        load_edgar_co2(EDGAR_FILE, grid)
    else
        @warn "EDGAR file not found: $EDGAR_FILE — running without emissions"
        NoEmission()
    end

    # Initialize tracers: uniform 420 ppm CO2
    arch = grid.architecture
    ArrayType = array_type(arch)

    sim_time = START_DATE
    step_start = 0

    tracers = if !isempty(CHECKPOINT) && isfile(CHECKPOINT)
        @info "Resuming from checkpoint: $CHECKPOINT"
        t, sim_time, step_start = load_checkpoint(CHECKPOINT, arch)
        t
    else
        (co2 = ArrayType(fill(FT(420.0), Nx, Ny, Nz)),)
    end

    @info "Initial CO2: $(round(FT(420.0), digits=1)) ppm uniform"

    # Volume weights for mass conservation tracking
    vol_cpu = compute_volume_weights(grid, p_edges, FT)
    physical_mass(c) = sum(Float64.(Array(c)) .* vol_cpu)
    total_mass_initial = physical_mass(tracers.co2)

    # Output setup
    mkpath(OUTPUT_DIR)
    outfile = joinpath(OUTPUT_DIR, "production_jja2024_output.nc")
    checkpoint_dir = joinpath(OUTPUT_DIR, "checkpoints")

    if step_start == 0
        isfile(outfile) && rm(outfile)
        init_output(outfile, grid, p_edges)
        save_snapshot!(outfile, tracers, vol_cpu, p_edges, 0.0)
    end

    # Advection scheme
    scheme = SlopesAdvection(use_limiter=true)

    # Time stepping
    total_seconds = Dates.value(Millisecond(END_DATE - sim_time)) / 1000.0
    N_steps = round(Int, total_seconds / Δt)
    met_seconds = Dates.value(Millisecond(MET_INTERVAL)) / 1000.0
    save_seconds = Dates.value(Millisecond(SAVE_INTERVAL)) / 1000.0
    checkpoint_seconds = Dates.value(Millisecond(CHECKPOINT_INTERVAL)) / 1000.0
    steps_per_met_update = round(Int, met_seconds / Δt)
    steps_per_save = round(Int, save_seconds / Δt)
    steps_per_checkpoint = round(Int, checkpoint_seconds / Δt)

    @info "Simulation: $(sim_time) → $(END_DATE)"
    @info "  Total steps: $N_steps, Δt=$(Δt)s"
    @info "  Met update every $steps_per_met_update steps ($(met_seconds/3600)h)"
    @info "  Save every $steps_per_save steps ($(save_seconds/3600)h)"
    @info "  Checkpoint every $steps_per_checkpoint steps"

    current_met_time = sim_time
    t_wall = time()

    # Compute initial subcycling counts from CFL analysis
    sc = subcycling_counts(vel, grid, FT(Δt / 2))
    @info "  CFL (half-step): x=$(round(sc.cfl_x, digits=2)), y=$(round(sc.cfl_y, digits=2)), z=$(round(sc.cfl_z, digits=2))"
    @info "  Subcycling: nx=$(sc.nx), ny=$(sc.ny), nz=$(sc.nz)"

    for step in (step_start + 1):(step_start + N_steps)
        step_sim_time = sim_time + Millisecond(round(Int, (step - step_start) * Δt * 1000))
        half = FT(Δt / 2)

        # Update met data every 6h
        met_time_for_step = floor(step_sim_time, MET_INTERVAL)
        if met_time_for_step != current_met_time && step > step_start + 1
            @info "  Updating met data for $(met_time_for_step)..."
            try
                pl_f, sfc_f, tidx = datetime_to_met_file_and_idx(
                    met_time_for_step, ERA5_DIR, pl_files, sfc_files)
                data = load_era5_timestep(pl_f, sfc_f, tidx, FT)
                vel = stagger_winds!(data, grid, p_edges, FT)
                current_met_time = met_time_for_step
                sc = subcycling_counts(vel, grid, half)
                @info "  CFL: x=$(round(sc.cfl_x, digits=2)), y=$(round(sc.cfl_y, digits=2)), z=$(round(sc.cfl_z, digits=2)) → sub: $(sc.nx)/$(sc.ny)/$(sc.nz)"
            catch e
                @warn "Failed to update met data: $e (reusing previous)"
            end
        end

        # Strang splitting: XYZ / ZYX with CFL-adaptive subcycling
        advect_x_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.nx)
        advect_y_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.ny)
        advect_z_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.nz)

        # Inject emissions at full timestep (middle of Strang split)
        apply_surface_flux!(tracers, edgar, grid, Δt)

        advect_z_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.nz)
        advect_y_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.ny)
        advect_x_subcycled!(tracers, vel, grid, scheme, half; n_sub=sc.nx)

        # Reset below-surface cells to prevent oscillation growth at terrain
        if terrain_mask !== nothing
            for (name, c) in pairs(tracers)
                c_cpu = Array(c)
                apply_terrain_mask!(c_cpu, terrain_mask, Nx, Ny, Nz)
                copyto!(c, c_cpu)
            end
        end

        # Save snapshot every 6h
        if (step - step_start) % steps_per_save == 0
            time_hours = (step - step_start) * Δt / 3600.0
            c_cpu = Array(tracers.co2)
            phys_mass = physical_mass(tracers.co2)
            mass_change = (phys_mass - total_mass_initial) / abs(total_mass_initial)
            wall = round(time() - t_wall, digits=1)
            rate = round(time_hours * 3600 / (time() - t_wall), digits=1)

            @info "  $(step_sim_time): " *
                  "min=$(round(minimum(c_cpu), digits=2)), " *
                  "max=$(round(maximum(c_cpu), digits=2)), " *
                  "mean=$(round(sum(c_cpu)/length(c_cpu), digits=2)), " *
                  "mass_Δ=$(round(mass_change*100, sigdigits=3))%, " *
                  "wall=$(wall)s, $(rate)x realtime"

            save_snapshot!(outfile, tracers, vol_cpu, p_edges, time_hours)
        end

        # Checkpoint weekly
        if (step - step_start) % steps_per_checkpoint == 0
            save_checkpoint(checkpoint_dir, tracers, step_sim_time, step)
        end
    end

    # Final output
    total_time_hours = N_steps * Δt / 3600.0
    save_snapshot!(outfile, tracers, vol_cpu, p_edges, total_time_hours)
    save_checkpoint(checkpoint_dir, tracers, END_DATE, step_start + N_steps)

    phys_mass_final = physical_mass(tracers.co2)
    wall_total = round(time() - t_wall, digits=1)

    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "  Total steps: $N_steps"
    @info "  Wall time: $(wall_total)s ($(round(wall_total/3600, digits=1))h)"
    @info "  Mass change: $(round((phys_mass_final - total_mass_initial) / abs(total_mass_initial) * 100, sigdigits=3))%"
    @info "  Output: $outfile"
    @info "=" ^ 70
end

main()
