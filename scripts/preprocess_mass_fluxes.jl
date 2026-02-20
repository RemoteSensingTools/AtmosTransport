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
# Input:  ERA5 model-level daily files (u, v, lnsp on hybrid sigma-pressure)
#         These MUST be model-level data (levtype: ml), NOT pressure-level data.
#         See docs/METEO_PREPROCESSING.md for why pressure levels cannot be used.
#
# Output: Single NetCDF file with am, bm, cm, m for every met window
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
#   OUTFILE      — output path (default: auto-generated)
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Parameters
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate,
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

const OUTFILE = get(ENV, "OUTFILE",
    expanduser("~/data/metDrivers/era5/massflux_era5_202406_$(lowercase(FT_STR)).nc"))

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
# Main preprocessing
# ===========================================================================
function preprocess()
    @info "=" ^ 70
    @info "AtmosTransportModel — Mass-Flux Preprocessing (TM5 dynam0 style)"
    @info "=" ^ 70

    files = find_era5_files(ERA5_DIRS)
    isempty(files) && error("No ERA5 model-level files found in $(ERA5_DIRS)")

    @info "  ERA5 files: $(length(files)) daily files"
    @info "  Level range: $(LEVEL_TOP)-$(LEVEL_BOT) ($(length(LEVEL_RANGE)) levels)"
    @info "  FT = $FT, DT = $(DT)s"
    @info "  Output: $OUTFILE"

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

    total_windows = Nt_per_file * length(files)
    half_dt = DT / 2
    met_interval = FT(21600)
    steps_per_met = max(1, round(Int, met_interval / DT))

    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"
    @info "  Total met windows: $total_windows"
    @info "  Steps per met window: $steps_per_met"
    @info "  half_dt = $(half_dt)s"

    # Pre-allocate CPU work arrays
    u_stag = Array{FT}(undef, Nx + 1, Ny, Nz)
    v_stag = Array{FT}(undef, Nx, Ny + 1, Nz)
    Δp     = Array{FT}(undef, Nx, Ny, Nz)
    m      = Array{FT}(undef, Nx, Ny, Nz)
    am     = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm     = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm     = Array{FT}(undef, Nx, Ny, Nz + 1)

    gc = build_geometry_cache(grid, m)

    # Create output NetCDF
    mkpath(dirname(OUTFILE))
    rm(OUTFILE; force=true)

    NCDataset(OUTFILE, "c") do ds
        ds.attrib["title"] = "Pre-computed mass fluxes for AtmosTransportModel"
        ds.attrib["source"] = "ERA5 model levels (levtype=ml), NOT pressure levels"
        ds.attrib["float_type"] = string(FT)
        ds.attrib["dt_seconds"] = Float64(DT)
        ds.attrib["half_dt_seconds"] = Float64(half_dt)
        ds.attrib["level_top"] = LEVEL_TOP
        ds.attrib["level_bot"] = LEVEL_BOT
        ds.attrib["steps_per_met_window"] = steps_per_met
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
               attrib = Dict("units" => "hours since 2024-06-01 00:00:00"))

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
        wall_start = time()

        for (day_idx, filepath) in enumerate(files)
            @info "Day $day_idx/$(length(files)): $(basename(filepath))"

            ds_in = NCDataset(filepath)
            Nt_local = length(ds_in["valid_time"][:])
            close(ds_in)

            for tidx in 1:Nt_local
                tidx_out += 1
                t0 = time()

                u_cc, v_cc, ps = load_era5_timestep(filepath, tidx, FT)
                stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
                Advection._build_Δz_3d!(Δp, grid, ps)

                compute_air_mass!(m, Δp, gc)
                compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, gc, Δp, half_dt)

                sim_hours = (tidx_out - 1) * Float64(met_interval) / 3600.0

                ds["time"][tidx_out] = sim_hours
                ds["m"][:, :, :, tidx_out]  = m
                ds["am"][:, :, :, tidx_out] = am
                ds["bm"][:, :, :, tidx_out] = bm
                ds["cm"][:, :, :, tidx_out] = cm
                ds["ps"][:, :, tidx_out]    = ps

                elapsed = round(time() - t0, digits=2)
                @info @sprintf("  Window %d/%d (hour %.0f): %.2fs (load+stagger+flux+write)",
                               tidx_out, total_windows, sim_hours, elapsed)
            end
        end

        wall_total = round(time() - wall_start, digits=1)
        @info "\n" * "=" ^ 70
        @info "Preprocessing complete!"
        @info "=" ^ 70
        @info "  Windows processed: $tidx_out"
        @info "  Wall time: $(wall_total)s ($(round(wall_total/tidx_out, digits=2))s/window)"
        @info "  Output: $OUTFILE"
        sz_mb = round(filesize(OUTFILE) / 1e6, digits=1)
        @info "  File size: $(sz_mb) MB"
        @info "=" ^ 70
    end
end

preprocess()
