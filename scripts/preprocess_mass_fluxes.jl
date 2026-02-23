#!/usr/bin/env julia
# ===========================================================================
# Offline preprocessing: ERA5 model-level winds → mass fluxes (am, bm, cm, m)
#
# This is analogous to TM5's dynam0 preprocessing step. Mass fluxes depend
# only on (u, v, ps) and are static per met window, so they can be:
#   1. Pre-computed once (this script)
#   2. Saved to NetCDF for reuse across forward/adjoint/sensitivity runs
#   3. Loaded directly onto GPU without any wind staggering at runtime
#
# Input:  ERA5 model-level monthly files (u, v, lnsp on hybrid sigma-pressure)
#         These MUST be model-level data (levtype: ml), NOT pressure-level data.
#         See docs/METEO_PREPROCESSING.md for why pressure levels cannot be used.
#
# Output: One NetCDF mass-flux file PER MONTH (monthly sharding for scalability)
#         e.g., massflux_era5_202401_float32.nc, massflux_era5_202402_float32.nc, ...
#
# Usage:
#   julia --project=. scripts/preprocess_mass_fluxes.jl
#
# Environment variables:
#   FT_PRECISION — "Float32" or "Float64" (default: Float32)
#   DT           — advection sub-step in seconds (default: 900)
#   LEVEL_TOP    — topmost model level (default: 50)
#   LEVEL_BOT    — bottommost model level (default: 137)
#   ERA5_DIRS    — colon-separated list of directories with era5_ml_*.nc files
#   OUTDIR       — output directory for monthly shard files (default: same as ERA5)
#   OUTFILE      — (legacy) single output path; if set, disables monthly sharding
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients
using NCDatasets
using Dates
using Printf

const FT_STR     = get(ENV, "FT_PRECISION", "Float32")
const FT         = FT_STR == "Float32" ? Float32 : Float64
const DT         = parse(FT, get(ENV, "DT", "900"))
const LEVEL_TOP  = parse(Int, get(ENV, "LEVEL_TOP", "50"))
const LEVEL_BOT  = parse(Int, get(ENV, "LEVEL_BOT", "137"))
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT

const ERA5_DIRS_STR = get(ENV, "ERA5_DIRS",
    join([expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607"),
          expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240608_20240630")], ":"))
const ERA5_DIRS = split(ERA5_DIRS_STR, ":")

const LEGACY_OUTFILE = get(ENV, "OUTFILE", "")
const OUTDIR = expanduser(get(ENV, "OUTDIR",
    length(ERA5_DIRS) > 0 ? string(ERA5_DIRS[1]) :
    "~/data/metDrivers/era5"))

# ---------------------------------------------------------------------------
# ERA5 model-level reader (N→S latitudes flipped to S→N)
# ---------------------------------------------------------------------------
function load_era5_timestep(filepath::String, tidx::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u    = FT.(ds["u"][:, :, :, tidx])[:, end:-1:1, :]
        v    = FT.(ds["v"][:, :, :, tidx])[:, end:-1:1, :]
        lnsp = FT.(ds["lnsp"][:, :, tidx])[:, end:-1:1]
        return u, v, exp.(lnsp)
    finally
        close(ds)
    end
end

function get_era5_info(filepath::String, ::Type{FT}) where {FT}
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
# Stagger cell-center winds to faces (CPU)
# ---------------------------------------------------------------------------
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u_stag[Nx + 1, :, :] .= u_stag[1, :, :]

    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0
    v_stag[:, Ny + 1, :] .= 0
    return nothing
end

# ---------------------------------------------------------------------------
# Collect daily file paths
# ---------------------------------------------------------------------------
function find_era5_files(datadirs)
    files = String[]
    for datadir in datadirs
        isdir(datadir) || continue
        for f in readdir(datadir)
            if startswith(f, "era5_ml_") && endswith(f, ".nc") && !contains(f, "_tmp")
                push!(files, joinpath(datadir, f))
            end
        end
    end
    sort!(files; by = basename)
    return files
end

# ===========================================================================
# Group ERA5 files by YYYYMM for monthly sharding.
# Extracts month from filename: era5_ml_YYYYMM.nc or era5_ml_YYYYMMDD.nc
# ===========================================================================
function _extract_month_key(filepath::String)
    bn = basename(filepath)
    # Match era5_ml_YYYYMM.nc or era5_ml_YYYYMMDD.nc
    m = match(r"era5_ml_(\d{4})(\d{2})", bn)
    m === nothing && return "unknown"
    return m[1] * m[2]  # "YYYYMM"
end

function group_files_by_month(files::Vector{String})
    groups = Dict{String, Vector{String}}()
    for f in files
        key = _extract_month_key(f)
        push!(get!(groups, key, String[]), f)
    end
    for v in values(groups)
        sort!(v; by=basename)
    end
    return groups
end

# ===========================================================================
# Write one monthly mass-flux shard to NetCDF
# ===========================================================================
function write_monthly_shard(outpath::String, files::Vector{String},
                             grid, gc, A_coeff, B_coeff,
                             Nx, Ny, Nz, half_dt, met_interval,
                             steps_per_met, month_key::String)
    # Pre-allocate CPU work arrays
    u_stag = Array{FT}(undef, Nx + 1, Ny, Nz)
    v_stag = Array{FT}(undef, Nx, Ny + 1, Nz)
    Δp     = Array{FT}(undef, Nx, Ny, Nz)
    m      = Array{FT}(undef, Nx, Ny, Nz)
    am     = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm     = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm     = Array{FT}(undef, Nx, Ny, Nz + 1)

    lons = FT.(grid.λᶜ_cpu)
    lats = FT.(grid.φᶜ_cpu)

    # Count total windows in this month
    total_windows = 0
    for filepath in files
        ds_in = NCDataset(filepath)
        total_windows += length(ds_in["valid_time"][:])
        close(ds_in)
    end

    # Derive a time origin from the month key
    year_str  = month_key[1:4]
    month_str = month_key[5:6]
    time_origin = "hours since $(year_str)-$(month_str)-01 00:00:00"

    rm(outpath; force=true)

    NCDataset(outpath, "c") do ds
        ds.attrib["title"] = "Pre-computed mass fluxes for AtmosTransport"
        ds.attrib["source"] = "ERA5 model levels (levtype=ml), NOT pressure levels"
        ds.attrib["float_type"] = string(FT)
        ds.attrib["dt_seconds"] = Float64(DT)
        ds.attrib["half_dt_seconds"] = Float64(half_dt)
        ds.attrib["level_top"] = LEVEL_TOP
        ds.attrib["level_bot"] = LEVEL_BOT
        ds.attrib["steps_per_met_window"] = steps_per_met
        ds.attrib["month"] = month_key
        ds.attrib["history"] = "Created $(Dates.now()) by preprocess_mass_fluxes.jl"
        ds.attrib["WARNING"] = "Model-level data only. Pressure-level data CANNOT be used."

        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defDim(ds, "lon_u", Nx + 1)
        defDim(ds, "lat_v", Ny + 1)
        defDim(ds, "lev_w", Nz + 1)
        defDim(ds, "time", Inf)

        defVar(ds, "lon", Float32, ("lon",);
               attrib = Dict("units" => "degrees_east"))[:] = Float32.(lons)
        defVar(ds, "lat", Float32, ("lat",);
               attrib = Dict("units" => "degrees_north"))[:] = Float32.(lats)
        defVar(ds, "time", Float64, ("time",);
               attrib = Dict("units" => time_origin))

        defVar(ds, "A_coeff", Float64, ("lev",);
               attrib = Dict("long_name" => "Hybrid A coefficient at level centers"))
        defVar(ds, "B_coeff", Float64, ("lev",);
               attrib = Dict("long_name" => "Hybrid B coefficient at level centers"))
        ds["A_coeff"][:] = Float64.(A_coeff[1:Nz])
        ds["B_coeff"][:] = Float64.(B_coeff[1:Nz])

        defVar(ds, "m", FT, ("lon", "lat", "lev", "time");
               attrib = Dict("units" => "kg", "long_name" => "Air mass per cell"),
               deflatelevel = 1)
        defVar(ds, "am", FT, ("lon_u", "lat", "lev", "time");
               attrib = Dict("units" => "kg", "long_name" => "Eastward mass flux at x-faces per half-timestep"),
               deflatelevel = 1)
        defVar(ds, "bm", FT, ("lon", "lat_v", "lev", "time");
               attrib = Dict("units" => "kg", "long_name" => "Northward mass flux at y-faces per half-timestep"),
               deflatelevel = 1)
        defVar(ds, "cm", FT, ("lon", "lat", "lev_w", "time");
               attrib = Dict("units" => "kg", "long_name" => "Downward mass flux at z-interfaces per half-timestep"),
               deflatelevel = 1)
        defVar(ds, "ps", FT, ("lon", "lat", "time");
               attrib = Dict("units" => "Pa", "long_name" => "Surface pressure"),
               deflatelevel = 1)

        tidx_out = 0
        t_month_start = time()

        for filepath in files
            ds_in = NCDataset(filepath)
            Nt_local = length(ds_in["valid_time"][:])
            close(ds_in)

            for tidx in 1:Nt_local
                tidx_out += 1
                t0 = time()

                u_cc, v_cc, ps_data = load_era5_timestep(filepath, tidx, FT)
                stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
                Advection._build_Δz_3d!(Δp, grid, ps_data)

                compute_air_mass!(m, Δp, gc)
                compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, gc, Δp, half_dt)

                sim_hours = (tidx_out - 1) * Float64(met_interval) / 3600.0

                ds["time"][tidx_out] = sim_hours
                ds["m"][:, :, :, tidx_out]  = m
                ds["am"][:, :, :, tidx_out] = am
                ds["bm"][:, :, :, tidx_out] = bm
                ds["cm"][:, :, :, tidx_out] = cm
                ds["ps"][:, :, tidx_out]    = ps_data

                elapsed = round(time() - t0, digits=2)
                if tidx_out <= 2 || tidx_out == total_windows || tidx_out % 10 == 0
                    @info @sprintf("  [%s] Window %d/%d (hour %.0f): %.2fs",
                                   month_key, tidx_out, total_windows, sim_hours, elapsed)
                end
            end
        end

        wall_month = round(time() - t_month_start, digits=1)
        sz_mb = round(filesize(outpath) / 1e6, digits=1)
        @info @sprintf("  [%s] Done: %d windows in %.1fs (%.2fs/win), %.1f MB",
                       month_key, tidx_out, wall_month,
                       tidx_out > 0 ? wall_month / tidx_out : 0.0, sz_mb)
    end
end

# ===========================================================================
# Main preprocessing — monthly sharding or legacy single-file mode
# ===========================================================================
function preprocess()
    @info "=" ^ 70
    @info "AtmosTransport — Mass-Flux Preprocessing (TM5 dynam0 style)"
    @info "=" ^ 70

    files = find_era5_files(ERA5_DIRS)
    isempty(files) && error("No ERA5 model-level files found in $(ERA5_DIRS)")

    @info "  ERA5 files: $(length(files)) files"
    @info "  Level range: $(LEVEL_TOP)-$(LEVEL_BOT) ($(length(LEVEL_RANGE)) levels)"
    @info "  FT = $FT, DT = $(DT)s"

    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=LEVEL_RANGE)
    A_full, B_full = load_vertical_coefficients(config; FT)
    A_coeff = A_full[LEVEL_TOP:LEVEL_BOT+1]
    B_coeff = B_full[LEVEL_TOP:LEVEL_BOT+1]

    lons, lats, levs, Nx, Ny, Nz, Nt_per_file = get_era5_info(files[1], FT)
    Nz_vc = length(LEVEL_RANGE)
    @assert Nz == Nz_vc "File has $Nz levels but expected $Nz_vc"

    params = load_parameters(FT)
    pp = params.planet

    arch = CPU()
    grid = LatitudeLongitudeGrid(arch;
        FT, size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(lons[2] - lons[1])),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    half_dt = DT / 2
    met_interval = FT(21600)
    steps_per_met = max(1, round(Int, met_interval / DT))
    m_dummy = Array{FT}(undef, Nx, Ny, Nz)
    gc = build_geometry_cache(grid, m_dummy)

    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"
    @info "  Steps per met window: $steps_per_met"
    @info "  half_dt = $(half_dt)s"

    # --- Monthly sharding (default) or legacy single-file mode ---
    if !isempty(LEGACY_OUTFILE)
        @info "  Legacy mode: single output file $LEGACY_OUTFILE"
        write_monthly_shard(LEGACY_OUTFILE, files, grid, gc, A_coeff, B_coeff,
                            Nx, Ny, Nz, half_dt, met_interval, steps_per_met, "all")
    else
        month_groups = group_files_by_month(files)
        months_sorted = sort(collect(keys(month_groups)))
        @info "  Monthly sharding: $(length(months_sorted)) months → $OUTDIR"

        mkpath(OUTDIR)
        wall_start = time()

        for month_key in months_sorted
            month_files = month_groups[month_key]
            ft_tag = lowercase(FT_STR)
            outpath = joinpath(OUTDIR, "massflux_era5_$(month_key)_$(ft_tag).nc")

            if isfile(outpath)
                @info "  [$month_key] Already exists: $outpath — skipping"
                continue
            end

            @info "  [$month_key] Processing $(length(month_files)) files..."
            write_monthly_shard(outpath, month_files, grid, gc, A_coeff, B_coeff,
                                Nx, Ny, Nz, half_dt, met_interval, steps_per_met, month_key)
        end

        wall_total = round(time() - wall_start, digits=1)
        @info "\n" * "=" ^ 70
        @info "Preprocessing complete!"
        @info "=" ^ 70
        @info "  Months processed: $(length(months_sorted))"
        @info "  Wall time: $(wall_total)s"
        @info "  Output directory: $OUTDIR"
        for month_key in months_sorted
            ft_tag = lowercase(FT_STR)
            fp = joinpath(OUTDIR, "massflux_era5_$(month_key)_$(ft_tag).nc")
            if isfile(fp)
                sz_mb = round(filesize(fp) / 1e6, digits=1)
                @info "    $month_key: $(sz_mb) MB"
            end
        end
        @info "=" ^ 70
    end
end

preprocess()
