#!/usr/bin/env julia
# ===========================================================================
# Download meteorological test data for AtmosTransportModel
#
# Three sources:
#   1. GEOS-FP via OPeNDAP (NO authentication required) — primary test data
#   2. ERA5 via CDSAPI (requires ~/.cdsapirc)
#   3. MERRA-2 via HTTP (requires ~/.netrc for NASA Earthdata)
#
# Usage:
#   julia --project=scripts scripts/download_test_data.jl
# ===========================================================================

using CDSAPI
using Dates
using HTTP
using NCDatasets
using Base64

const DATA_ROOT  = joinpath(homedir(), "data", "metDrivers")
const GEOSFP_DIR = joinpath(DATA_ROOT, "geosfp", "test")
const ERA5_DIR   = joinpath(DATA_ROOT, "era5", "test")
const MERRA2_DIR = joinpath(DATA_ROOT, "merra2", "test")

# ===========================================================================
# Part A: GEOS-FP download via OPeNDAP (NO authentication needed!)
#
# This is the recommended primary test data source since it works without
# any credentials. We download a small subset (single time step, coarsened)
# from the NCCS OPeNDAP server.
# ===========================================================================

const GEOSFP_OPENDAP_BASE = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim"

"""
    download_geosfp_opendap(date::Date, outdir::String; stride=8)

Download a coarsened GEOS-FP snapshot via OPeNDAP for a recent date.
`stride` controls spatial coarsening (stride=8 → ~2.5°×2° output).

Downloads from three collections:
  - inst3_3d_asm_Nv: winds, temperature, humidity, pressure
  - tavg3_3d_trb_Ne: diffusivity (kh), edge pressure
  - tavg1_2d_slv_Nx: single-level diagnostics
"""
function download_geosfp_opendap(date::Date, outdir::String; stride::Int=8)
    mkpath(outdir)
    outfile = joinpath(outdir, "geosfp_asm_Nv_$(Dates.format(date, "yyyymmdd")).nc")

    if isfile(outfile) && filesize(outfile) > 1000
        println("  Already exists: $outfile ($(filesize(outfile) ÷ 1024) KB)")
        return outfile
    end

    println("  Downloading GEOS-FP via OPeNDAP for $date (stride=$stride)...")

    # Open remote OPeNDAP dataset
    url = "$(GEOSFP_OPENDAP_BASE)/inst3_3d_asm_Nv"
    println("    Opening: $url")

    ds = NCDataset(url)

    # Read coordinate arrays
    lons = ds["lon"][:]
    lats = ds["lat"][:]
    levs = ds["lev"][:]
    times = ds["time"][:]

    # Find the time index closest to 00Z on the requested date
    # GEOS-FP time is in "days since 1-1-1 00:00:0.0"
    target_day = Dates.value(date - Date(1, 1, 1))
    tidx = argmin(abs.(times .- target_day))

    println("    Time index: $tidx (time=$(times[tidx]))")

    # Subsample spatially
    lon_idx = 1:stride:length(lons)
    lat_idx = 1:stride:length(lats)

    println("    Grid: $(length(lon_idx))×$(length(lat_idx))×$(length(levs)) (from $(length(lons))×$(length(lats))×$(length(levs)))")

    # Read key 3D variables (one time step, subsampled horizontally, all levels)
    vars_3d = ["u", "v", "omega", "t", "qv", "delp"]
    vars_2d = ["ps", "phis"]

    # Write to local NetCDF
    NCDataset(outfile, "c") do out
        # Define dimensions
        defDim(out, "lon", length(lon_idx))
        defDim(out, "lat", length(lat_idx))
        defDim(out, "lev", length(levs))

        # Write coordinates
        lon_var = defVar(out, "lon", Float64, ("lon",))
        lon_var[:] = lons[lon_idx]

        lat_var = defVar(out, "lat", Float64, ("lat",))
        lat_var[:] = lats[lat_idx]

        lev_var = defVar(out, "lev", Float64, ("lev",))
        lev_var[:] = levs

        # Global attributes
        out.attrib["source"] = "GEOS-FP inst3_3d_asm_Nv"
        out.attrib["date"] = Dates.format(date, "yyyy-mm-dd")
        out.attrib["time_index"] = tidx
        out.attrib["stride"] = stride
        out.attrib["created_by"] = "AtmosTransportModel/scripts/download_test_data.jl"

        # Read and write 3D variables
        for varname in vars_3d
            println("    Reading $varname...")
            data = ds[varname][lon_idx, lat_idx, :, tidx]
            v = defVar(out, varname, Float32, ("lon", "lat", "lev"))
            v[:] = Float32.(data)
        end

        # Read and write 2D variables
        for varname in vars_2d
            println("    Reading $varname...")
            data = ds[varname][lon_idx, lat_idx, tidx]
            v = defVar(out, varname, Float32, ("lon", "lat"))
            v[:] = Float32.(data)
        end
    end

    close(ds)

    println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
    return outfile
end

"""
    download_geosfp_turbulence(date::Date, outdir::String; stride=8)

Download GEOS-FP turbulence data (diffusivity kh, edge pressure) via OPeNDAP.
"""
function download_geosfp_turbulence(date::Date, outdir::String; stride::Int=8)
    mkpath(outdir)
    outfile = joinpath(outdir, "geosfp_trb_Ne_$(Dates.format(date, "yyyymmdd")).nc")

    if isfile(outfile) && filesize(outfile) > 1000
        println("  Already exists: $outfile ($(filesize(outfile) ÷ 1024) KB)")
        return outfile
    end

    println("  Downloading GEOS-FP turbulence via OPeNDAP for $date...")

    url = "$(GEOSFP_OPENDAP_BASE)/tavg3_3d_trb_Ne"
    println("    Opening: $url")

    ds = NCDataset(url)
    lons = ds["lon"][:]
    lats = ds["lat"][:]
    levs = ds["lev"][:]
    times = ds["time"][:]

    target_day = Dates.value(date - Date(1, 1, 1)) + 0.0625  # tavg3 offset: 01:30Z
    tidx = argmin(abs.(times .- target_day))

    lon_idx = 1:stride:length(lons)
    lat_idx = 1:stride:length(lats)

    NCDataset(outfile, "c") do out
        defDim(out, "lon", length(lon_idx))
        defDim(out, "lat", length(lat_idx))
        defDim(out, "lev", length(levs))

        defVar(out, "lon", Float64, ("lon",))[:] = lons[lon_idx]
        defVar(out, "lat", Float64, ("lat",))[:] = lats[lat_idx]
        defVar(out, "lev", Float64, ("lev",))[:] = levs

        out.attrib["source"] = "GEOS-FP tavg3_3d_trb_Ne"
        out.attrib["date"] = Dates.format(date, "yyyy-mm-dd")

        for varname in ["kh", "ple"]
            println("    Reading $varname...")
            data = ds[varname][lon_idx, lat_idx, :, tidx]
            defVar(out, varname, Float32, ("lon", "lat", "lev"))[:] = Float32.(data)
        end
    end

    close(ds)
    println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
    return outfile
end

# ===========================================================================
# Part B: ERA5 download via CDSAPI.jl
# ===========================================================================
function download_era5_pressure_levels(date::Date, outdir::String)
    mkpath(outdir)
    outfile = joinpath(outdir, "era5_pressure_levels_$(Dates.format(date, "yyyymmdd")).nc")

    if isfile(outfile)
        println("  Already exists: $outfile")
        return outfile
    end

    println("  Downloading ERA5 pressure levels for $date...")

    request = Dict(
        "product_type" => ["reanalysis"],
        "variable" => [
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "temperature",
            "specific_humidity",
        ],
        "pressure_level" => ["1000", "925", "850", "700", "500", "300", "200", "100"],
        "year" => [string(Dates.year(date))],
        "month" => [lpad(Dates.month(date), 2, '0')],
        "day" => [lpad(Dates.day(date), 2, '0')],
        "time" => ["00:00", "06:00", "12:00", "18:00"],
        "format" => "netcdf",
        "grid" => "2.0/2.0",
    )

    CDSAPI.retrieve("reanalysis-era5-pressure-levels", request, outfile)

    println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
    return outfile
end

function download_era5_single_level(date::Date, outdir::String)
    mkpath(outdir)
    outfile = joinpath(outdir, "era5_single_level_$(Dates.format(date, "yyyymmdd")).nc")

    if isfile(outfile)
        println("  Already exists: $outfile")
        return outfile
    end

    println("  Downloading ERA5 single level for $date...")

    request = Dict(
        "product_type" => ["reanalysis"],
        "variable" => ["surface_pressure", "2m_temperature"],
        "year" => [string(Dates.year(date))],
        "month" => [lpad(Dates.month(date), 2, '0')],
        "day" => [lpad(Dates.day(date), 2, '0')],
        "time" => ["00:00", "06:00", "12:00", "18:00"],
        "format" => "netcdf",
        "grid" => "2.0/2.0",
    )

    CDSAPI.retrieve("reanalysis-era5-single-levels", request, outfile)

    println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
    return outfile
end

# ===========================================================================
# Part C: MERRA-2 download via HTTP.jl
#
# Uses native-level collections (Nv/Ne) instead of pressure levels (Np).
# Stream code is determined by year.
# ===========================================================================

function read_netrc(machine = "urs.earthdata.nasa.gov")
    netrc_path = joinpath(homedir(), ".netrc")
    isfile(netrc_path) || error("No .netrc file found at $netrc_path")
    for line in eachline(netrc_path)
        parts = split(strip(line))
        if length(parts) >= 6 && parts[1] == "machine" && parts[2] == machine
            return parts[4], parts[6]
        end
    end
    error("No entry for $machine in $netrc_path")
end

function merra2_stream_code(year::Int)
    year ≤ 1991 && return 100
    year ≤ 2000 && return 200
    year ≤ 2010 && return 300
    return 400
end

function download_merra2(url::String, outfile::String)
    if isfile(outfile)
        println("  Already exists: $outfile")
        return outfile
    end

    mkpath(dirname(outfile))
    login, password = read_netrc()

    println("  Downloading: $(basename(url))...")

    auth_header = "Basic " * base64encode("$login:$password")
    resp = HTTP.get(
        url;
        headers = ["Authorization" => auth_header],
        redirect = true,
        status_exception = false,
    )

    if resp.status == 200
        open(outfile, "w") do io
            write(io, resp.body)
        end
        println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
    else
        println("  ERROR: HTTP $(resp.status) for $url")
        println("  Trying curl with .netrc...")
        try
            run(pipeline(`curl -n -L -f -o $outfile $url`, stdout=devnull, stderr=devnull))
            if isfile(outfile) && filesize(outfile) > 0
                println("  Saved: $outfile ($(filesize(outfile) ÷ 1024) KB)")
            else
                println("  Failed: curl produced empty or missing file.")
            end
        catch e
            println("  Failed: $e")
            println("  Check ~/.netrc has: machine urs.earthdata.nasa.gov login <user> password <pass>")
        end
    end

    return outfile
end

function download_merra2_asm_Nv(date::Date, outdir::String)
    datestr = Dates.format(date, "yyyymmdd")
    stream = merra2_stream_code(Dates.year(date))
    filename = "MERRA2_$(stream).inst3_3d_asm_Nv.$(datestr).nc4"
    url = "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/$(Dates.year(date))/$(lpad(Dates.month(date), 2, '0'))/$filename"
    return download_merra2(url, joinpath(outdir, filename))
end

function download_merra2_trb_Ne(date::Date, outdir::String)
    datestr = Dates.format(date, "yyyymmdd")
    stream = merra2_stream_code(Dates.year(date))
    filename = "MERRA2_$(stream).tavg3_3d_trb_Ne.$(datestr).nc4"
    url = "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T3NETRB.5.12.4/$(Dates.year(date))/$(lpad(Dates.month(date), 2, '0'))/$filename"
    return download_merra2(url, joinpath(outdir, filename))
end

# ===========================================================================
# Part D: Verification
# ===========================================================================
function verify_file(path::String)
    if !isfile(path)
        println("  MISSING: $path")
        return false
    end

    ds = NCDataset(path)
    println("  File: $(basename(path))")
    println("    Size: $(filesize(path) ÷ 1024) KB")
    println("    Dims: $(keys(ds.dim))")
    println("    Vars: $(keys(ds))")
    close(ds)
    return true
end

# ===========================================================================
# Part E: Main function
# ===========================================================================
function main()
    date = Date(2024, 3, 1)

    println("=" ^ 70)
    println("Downloading test meteorological data for $date")
    println("=" ^ 70)

    # ---- GEOS-FP (primary — no auth needed) ----
    println("\n--- GEOS-FP (OPeNDAP, no auth) ---")
    try
        download_geosfp_opendap(date, GEOSFP_DIR; stride=8)
    catch e
        println("  GEOS-FP asm_Nv failed: $e")
    end

    try
        download_geosfp_turbulence(date, GEOSFP_DIR; stride=8)
    catch e
        println("  GEOS-FP trb_Ne failed: $e")
    end

    # ---- ERA5 ----
    println("\n--- ERA5 (CDS API) ---")
    try
        download_era5_pressure_levels(date, ERA5_DIR)
    catch e
        println("  ERA5 pressure levels failed: $e")
    end

    try
        download_era5_single_level(date, ERA5_DIR)
    catch e
        println("  ERA5 single level failed: $e")
    end

    # ---- MERRA-2 ----
    println("\n--- MERRA-2 (Earthdata) ---")
    try
        download_merra2_asm_Nv(date, MERRA2_DIR)
    catch e
        println("  MERRA-2 asm_Nv failed: $e")
    end

    try
        download_merra2_trb_Ne(date, MERRA2_DIR)
    catch e
        println("  MERRA-2 trb_Ne failed: $e")
    end

    # ---- Verification ----
    println("\n--- Verification ---")
    for dir in [GEOSFP_DIR, ERA5_DIR, MERRA2_DIR]
        isdir(dir) || continue
        for f in readdir(dir; join=true)
            (endswith(f, ".nc") || endswith(f, ".nc4")) || continue
            verify_file(f)
        end
    end

    println("\nDone!")
end

main()
