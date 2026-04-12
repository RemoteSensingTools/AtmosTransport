#!/usr/bin/env julia
# ===========================================================================
# Post-process ERA5 surface fields + convective mass flux for AtmosTransport
#
# Reads per-day downloads from:
#   surface:    ~/data/metDrivers/era5/surface_fields_june2023/
#                 era5_surface_YYYYMMDD.nc  (ZIP containing two inner NetCDF:
#                   stepType-instant: blh, t2m, u10, v10  (hourly analysis)
#                   stepType-accum:   sshf                (hourly accumulated J/m²))
#   convective: ~/data/metDrivers/era5/cmfmc_june2023/
#                 era5_cmfmc_YYYYMMDD.grib  (param 78.162, 3-hourly forecast)
#
# Derived fields appended to the spectral preprocessed NetCDF:
#   pblh   [m]      — boundary layer height (blh)
#   ustar  [m/s]    — friction velocity (bulk from 10m wind)
#   hflux  [W/m²]   — upward sensible heat flux (from SSHF, deaccumulated)
#   t2m    [K]      — 2m temperature
#   conv_mass_flux [kg/(m²·s)] — net upward convective mass flux at interfaces
#
# Usage:
#   julia --project=. scripts/postprocess_era5_surface_physics.jl \
#       config/preprocessing/spectral_june2023.toml
#
# Config additions (optional, uses defaults from main config):
#   [input]
#     surface_dir   = "~/data/metDrivers/era5/surface_fields_june2023"
#     cmfmc_dir     = "~/data/metDrivers/era5/cmfmc_june2023"
# ===========================================================================

using NCDatasets
using GRIB
using Dates
using Printf
using TOML
using Statistics: mean

length(ARGS) >= 1 || error(
    "Usage: julia --project=. scripts/postprocess_era5_surface_physics.jl <config.toml>")

const config = TOML.parsefile(ARGS[1])

# --- Directories ---
const OUTDIR = expanduser(config["output"]["directory"])
const SURFACE_DIR = expanduser(
    get(get(config, "input", Dict()), "surface_dir",
        "~/data/metDrivers/era5/surface_fields_june2023"))
const CMFMC_DIR = expanduser(
    get(get(config, "input", Dict()), "cmfmc_dir",
        "~/data/metDrivers/era5/cmfmc_june2023"))

const FT_STR = get(get(config, "numerics", Dict()), "float_type", "Float32")
const FT = FT_STR == "Float32" ? Float32 : Float64
const MET_INTERVAL = Float64(get(get(config, "numerics", Dict()), "met_interval", 21600.0))
const Nz = config["grid"]["level_bot"] - config["grid"]["level_top"] + 1

# Drag coefficient for ustar estimation (neutral, roughness-independent)
# ustar ≈ sqrt(Cd) * |U10|  with Cd ≈ 1.2e-3 for typical marine/land conditions
const Cd_NEUTRAL = 1.2e-3

# ===========================================================================
# Read surface fields from the CDS ZIP archive for one day
#
# Returns NamedTuple: (pblh, t2m, u10, v10, sshf) each as Array{FT}(lon, lat, 24)
# Times are hourly at 00,01,...,23 UTC.
# Lon dimension may be 0..360 (CDS) rather than 0..359.5; we keep as-is.
# ===========================================================================
function read_surface_day(date_str::String)
    path = joinpath(SURFACE_DIR, "era5_surface_$(replace(date_str, "-" => "")).nc")
    isfile(path) || error("Surface file not found: $path")

    # Detect format: ZIP (new CDS API) vs plain NetCDF
    magic = open(path, "r") do f; read(f, 2); end

    if magic == UInt8[0x50, 0x4b]  # PK = ZIP magic
        return _read_surface_zip(path)
    else
        return _read_surface_nc(path)
    end
end

function _read_surface_zip(path::String)
    # Use ZipFile.jl-free approach: shell out to unzip into tempdir
    tmpdir = mktempdir()
    run(Cmd(["unzip", "-q", "-o", path, "-d", tmpdir]))

    instant_file = joinpath(tmpdir, "data_stream-oper_stepType-instant.nc")
    accum_file   = joinpath(tmpdir, "data_stream-oper_stepType-accum.nc")

    result = NCDataset(instant_file, "r") do ds_i
        NCDataset(accum_file, "r") do ds_a
            # ERA5 lat from +90 to -90; flip to -90..+90
            # Use nomissing to handle missing values (fill with sensible defaults)
            _nm(v, default) = FT.(nomissing(v[:, :, :], default))
            blh  = permutedims(reverse(_nm(ds_i["blh"],  FT(1000)), dims=2), (1,2,3))
            t2m  = permutedims(reverse(_nm(ds_i["t2m"],  FT(285)),  dims=2), (1,2,3))
            u10  = permutedims(reverse(_nm(ds_i["u10"],  FT(0)),    dims=2), (1,2,3))
            v10  = permutedims(reverse(_nm(ds_i["v10"],  FT(0)),    dims=2), (1,2,3))
            sshf = permutedims(reverse(_nm(ds_a["sshf"], FT(0)),    dims=2), (1,2,3))
            (; blh, t2m, u10, v10, sshf)
        end
    end
    rm(tmpdir; recursive=true)
    return result
end

function _read_surface_nc(path::String)
    NCDataset(path, "r") do ds
        flip_lat = ds["latitude"][1] > ds["latitude"][end]  # ERA5: N→S

        function _read_flip(v, default=FT(0))
            arr = FT.(nomissing(ds[v][:, :, :], default))
            flip_lat ? reverse(arr, dims=2) : arr
        end

        blh  = _read_flip("blh", FT(1000))
        t2m  = _read_flip("t2m", FT(285))
        u10  = _read_flip("u10")
        v10  = _read_flip("v10")
        sshf = _read_flip("sshf")
        (; blh, t2m, u10, v10, sshf)
    end
end

# ===========================================================================
# Derive 6-hourly physics fields from hourly surface data
#
# ERA5 hourly SSHF is accumulated from the previous hour:
#   sshf[t] = integral(sfc_sensible_heat_flux, t-1h..t) [J/m²]
# Sign convention: positive SSHF = downward into surface.
# Our hflux = upward heat flux = -SSHF / 3600 [W/m²]
#
# Returns (pblh, ustar, hflux, t2m) each (Nlon, Nlat) at the given window hour.
# ===========================================================================
function derive_6h_fields(sfc, window_hour::Int)
    # window_hour ∈ {0, 6, 12, 18}
    # CDS hourly times: hour 1 = 00 UTC (t=1), hour 7 = 06 UTC (t=7), etc.
    t_idx = window_hour + 1   # 1-based: t=1→00Z, t=7→06Z, t=13→12Z, t=19→18Z

    pblh  = sfc.blh[:, :, t_idx]
    t2m   = sfc.t2m[:, :, t_idx]

    # ustar from neutral drag: ustar = sqrt(Cd) * |U10|
    wind10 = @. sqrt(sfc.u10[:, :, t_idx]^2 + sfc.v10[:, :, t_idx]^2)
    ustar  = @. FT(sqrt(Cd_NEUTRAL)) * wind10

    # hflux: deaccumulate SSHF (J/m²/hr) → upward W/m²
    # For t_idx=1 (00 UTC), take the last hour of previous day (sshf at t=1
    # is accumulated over 23→00 UTC). For simplicity, use rolling 6-hour mean.
    t_range = max(1, t_idx - 5):t_idx   # last 6 hours
    sshf_mean = mean(sfc.sshf[:, :, t] for t in t_range)  # mean accumulated J/m²/hr
    hflux = @. FT(-sshf_mean / 3600.0f0)   # J/m²/hr → W/m² (positive = upward)

    return (; pblh, ustar, hflux, t2m)
end

# ===========================================================================
# Read convective mass flux GRIB for one day
#
# param 78.162 = convective updraft mass flux [kg/(m²·s)], at half-levels
# Forecast from 06/18 UTC, steps 3/6/9/12.
# Valid times: 09/12/21/00(+1d)  (from 06 UTC) and 09/12/21/00 shifted by 12h
#
# We interpolate to 6-hourly valid times 00/06/12/18 UTC for this day.
# ===========================================================================
function read_cmfmc_day(date_str::String, Nlon::Int, Nlat::Int)
    path = joinpath(CMFMC_DIR, "era5_cmfmc_$(replace(date_str, "-" => "")).grib")
    isfile(path) || return nothing

    # cmfmc[lon, lat, lev_half, valid_hour_3h] where valid_hour_3h ∈ {3,6,9,12,15,18,21,24}
    # We have steps 3/6/9/12 from base times 06 and 18:
    # base 06 + step 3  → valid 09 UTC
    # base 06 + step 6  → valid 12 UTC
    # base 06 + step 9  → valid 15 UTC
    # base 06 + step 12 → valid 18 UTC
    # base 18 + step 3  → valid 21 UTC
    # base 18 + step 6  → valid 00 UTC (+1 day)
    # base 18 + step 9  → valid 03 UTC (+1 day)
    # base 18 + step 12 → valid 06 UTC (+1 day)
    #
    # Valid times we need: 00/06/12/18 UTC
    # - 00 UTC: from base 18 (-1 day) + step 6, or base 18 (this day) + step 6 (next day)
    #           → use base 18 + step 6 (next day's 00 UTC)
    # - 06 UTC: base 18 (this day) + step 12
    # - 12 UTC: base 06 (this day) + step 6
    # - 18 UTC: base 06 (this day) + step 12
    #
    # We collect all messages and index by valid_time.

    Nz_half = Nz + 1
    # Map valid_time_utc_hour → array[lon, lat, lev]
    fields = Dict{Int, Array{FT,3}}()

    f = GribFile(path)
    for msg in f
        step     = msg["stepRange"]     # e.g. "3"
        dataTime = msg["dataTime"]      # e.g. 600 (= 06:00)
        level    = msg["level"]

        base_h   = div(dataTime, 100)   # 6 or 18
        step_h   = parse(Int, string(step))
        valid_h  = mod(base_h + step_h, 24)   # valid UTC hour (0-23)

        if !haskey(fields, valid_h)
            fields[valid_h] = zeros(FT, Nlon, Nlat, Nz_half)
        end

        vals = msg["values"]
        # GRIB values are on a lat-lon grid: reshape and flip lat (ERA5 N→S)
        data2d = reshape(FT.(vals), Nlon, Nlat)
        data2d = reverse(data2d, dims=2)  # flip to S→N

        # level 1..137 = model level; half-level convention: top of level k = interface k
        # Our model uses interface k ∈ 1..Nz+1 (1=TOA, Nz+1=surface)
        if level >= 1 && level <= Nz_half
            fields[valid_h][:, :, level] .= data2d
        end
    end
    destroy(f)

    return fields
end

# Interpolate/select cmfmc at a target valid hour from the day's fields dict.
# Falls back to nearest available hour.
function cmfmc_at_hour(fields::Dict{Int,Array{FT,3}}, target_h::Int, Nlon::Int, Nlat::Int)
    isempty(fields) && return zeros(FT, Nlon, Nlat, Nz + 1)
    haskey(fields, target_h) && return fields[target_h]

    # Find closest available hour
    hours  = sort(collect(keys(fields)))
    diffs  = [mod(abs(h - target_h), 24) for h in hours]
    _, idx = findmin(diffs)
    @warn "CMFMC at $(target_h) UTC not found; using nearest $(hours[idx]) UTC"
    return fields[hours[idx]]
end

# ===========================================================================
# Append physics variables to an existing preprocessed NetCDF file
# ===========================================================================
function append_physics_to_nc!(ncpath::String, pblh_all, ustar_all, hflux_all,
                                t2m_all, cmfmc_all)
    NCDataset(ncpath, "a") do ds
        Nt = length(ds["time"][:])

        # Create variables if not present
        if !haskey(ds, "pblh")
            defVar(ds, "pblh", FT, ("lon", "lat", "time");
                   attrib=Dict("units"=>"m","long_name"=>"Boundary layer height"),
                   deflatelevel=1)
        end
        if !haskey(ds, "ustar")
            defVar(ds, "ustar", FT, ("lon", "lat", "time");
                   attrib=Dict("units"=>"m s-1","long_name"=>"Friction velocity (bulk estimate)"),
                   deflatelevel=1)
        end
        if !haskey(ds, "hflux")
            defVar(ds, "hflux", FT, ("lon", "lat", "time");
                   attrib=Dict("units"=>"W m-2","long_name"=>"Upward sensible heat flux"),
                   deflatelevel=1)
        end
        if !haskey(ds, "t2m")
            defVar(ds, "t2m", FT, ("lon", "lat", "time");
                   attrib=Dict("units"=>"K","long_name"=>"2m temperature"),
                   deflatelevel=1)
        end
        if !haskey(ds, "conv_mass_flux")
            defVar(ds, "conv_mass_flux", FT, ("lon", "lat", "lev_w", "time");
                   attrib=Dict("units"=>"kg m-2 s-1",
                               "long_name"=>"Convective updraft mass flux at interfaces"),
                   deflatelevel=1)
        end

        # Write in time chunks
        n_write = min(Nt, size(pblh_all, 3))
        ds["pblh"][:, :, 1:n_write]   = pblh_all[:, :, 1:n_write]
        ds["ustar"][:, :, 1:n_write]  = ustar_all[:, :, 1:n_write]
        ds["hflux"][:, :, 1:n_write]  = hflux_all[:, :, 1:n_write]
        ds["t2m"][:, :, 1:n_write]    = t2m_all[:, :, 1:n_write]
        if cmfmc_all !== nothing
            ds["conv_mass_flux"][:, :, :, 1:n_write] = cmfmc_all[:, :, :, 1:n_write]
        end
    end
    return nothing
end

# ===========================================================================
# Main
# ===========================================================================
function postprocess()
    # Find the preprocessed NetCDF file
    nc_files = filter(f -> endswith(f, ".nc"), readdir(OUTDIR))
    isempty(nc_files) && error("No preprocessed NetCDF found in $OUTDIR")
    ncpath = joinpath(OUTDIR, nc_files[1])
    @info "Appending physics to: $ncpath"

    # Get dimensions from existing file
    Nlon, Nlat, Nt = NCDataset(ncpath, "r") do ds
        length(ds["lon"][:]), length(ds["lat"][:]), length(ds["time"][:])
    end
    @info "Grid: $(Nlon) × $(Nlat), $(Nt) windows"

    # Determine dates from the spectral GRIB directory
    grib_dir = expanduser(config["input"]["spectral_dir"])
    dates = Date[]
    for f in readdir(grib_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    n_per_day = round(Int, 86400.0 / MET_INTERVAL)   # e.g. 4 for 6-hourly
    window_hours = [h for h in 0:div(86400, n_per_day):23]  # [0, 6, 12, 18]

    @info "Processing $(length(dates)) days × $(n_per_day) windows/day = $(length(dates)*n_per_day) total"

    # Pre-allocate output arrays
    pblh_all  = Array{FT}(undef, Nlon, Nlat, Nt)
    ustar_all = Array{FT}(undef, Nlon, Nlat, Nt)
    hflux_all = Array{FT}(undef, Nlon, Nlat, Nt)
    t2m_all   = Array{FT}(undef, Nlon, Nlat, Nt)
    has_cmfmc = isdir(CMFMC_DIR) && !isempty(readdir(CMFMC_DIR))
    cmfmc_all = has_cmfmc ? Array{FT}(undef, Nlon, Nlat, Nz+1, Nt) : nothing

    tidx = 0
    for (di, date) in enumerate(dates)
        date_str = Dates.format(date, "yyyy-mm-dd")
        t0 = time()

        # --- Surface fields ---
        sfc = try
            read_surface_day(date_str)
        catch e
            @warn "  Surface fields missing for $date_str: $e"
            nothing
        end

        # --- Convective mass flux ---
        cmfmc_day = if has_cmfmc
            try
                read_cmfmc_day(date_str, Nlon, Nlat)
            catch e
                @warn "  CMFMC missing for $date_str: $e"
                nothing
            end
        else
            nothing
        end

        for wh in window_hours
            tidx += 1
            tidx > Nt && break

            if sfc !== nothing
                f = derive_6h_fields(sfc, wh)
                pblh_all[:, :, tidx]  = f.pblh
                ustar_all[:, :, tidx] = f.ustar
                hflux_all[:, :, tidx] = f.hflux
                t2m_all[:, :, tidx]   = f.t2m
            else
                pblh_all[:, :, tidx]  .= FT(1000)   # fallback: 1 km
                ustar_all[:, :, tidx] .= FT(0.3)    # fallback: 0.3 m/s
                hflux_all[:, :, tidx] .= FT(0)
                t2m_all[:, :, tidx]   .= FT(285)
            end

            if cmfmc_all !== nothing
                fields = cmfmc_day !== nothing ? cmfmc_day : Dict{Int,Array{FT,3}}()
                cmfmc_all[:, :, :, tidx] = cmfmc_at_hour(fields, wh, Nlon, Nlat)
            end
        end

        elapsed = round(time() - t0, digits=1)
        @info @sprintf("  [%d/%d] %s (%.1fs)", di, length(dates), date_str, elapsed)
    end

    @info "Writing to NetCDF..."
    append_physics_to_nc!(ncpath, pblh_all, ustar_all, hflux_all, t2m_all, cmfmc_all)

    sz = round(filesize(ncpath) / 1e9, digits=2)
    @info @sprintf("Done. File size: %.2f GB (%s)", sz, ncpath)
end

postprocess()
