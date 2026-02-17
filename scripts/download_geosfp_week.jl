#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download one week of GEOS-FP data via OPeNDAP at ~3° resolution
#
# Target: 2025-02-01 to 2025-02-07, 3-hourly → 56 timesteps
# Subsampled: every 10th lon/lat → ~116 lon × ~73 lat × 72 lev
# Output: ~/data/metDrivers/geosfp/geosfp_4x5_20250201_20250207.nc
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

const DATA_DIR = expanduser("~/data/metDrivers/geosfp")
const OUT_FILE = joinpath(DATA_DIR, "geosfp_4x5_20250201_20250207.nc")
const OPENDAP_URL = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv"

const DATE_START = DateTime(2025, 2, 1, 0, 0, 0)
const DATE_END   = DateTime(2025, 2, 7, 21, 0, 0)
const LON_STRIDE = 10
const LAT_STRIDE = 10

const VARS_3D = ["u", "v", "omega", "t", "delp"]
const VARS_2D = ["ps"]

function download_geosfp_week()
    mkpath(DATA_DIR)

    if isfile(OUT_FILE)
        println("Output file already exists: $OUT_FILE")
        println("Delete it to re-download.")
        return
    end

    println("Connecting to GEOS-FP OPeNDAP...")
    ds = NCDataset(OPENDAP_URL)

    times_all = ds["time"][:]
    lon_full  = ds["lon"][:]
    lat_full  = ds["lat"][:]
    lev_full  = ds["lev"][:]

    # Subsample lon/lat
    lon_idx = 1:LON_STRIDE:length(lon_full)
    lat_idx = 1:LAT_STRIDE:length(lat_full)
    lon_sub = lon_full[lon_idx]
    lat_sub = lat_full[lat_idx]
    Nx = length(lon_sub)
    Ny = length(lat_sub)
    Nz = length(lev_full)

    println("Subsampled grid: $Nx lon × $Ny lat × $Nz lev")

    # Find time indices for our date range
    tidx_start = findfirst(t -> t >= DATE_START, times_all)
    tidx_end   = findlast(t -> t <= DATE_END, times_all)
    if tidx_start === nothing || tidx_end === nothing
        error("Could not find time range $DATE_START - $DATE_END in dataset")
    end
    time_indices = tidx_start:tidx_end
    times_sub = times_all[time_indices]
    Nt = length(times_sub)
    println("Time range: $(times_sub[1]) to $(times_sub[end]) ($Nt steps)")

    # Create output file
    println("Creating output file: $OUT_FILE")
    NCDataset(OUT_FILE, "c") do out
        # Dimensions
        defDim(out, "lon", Nx)
        defDim(out, "lat", Ny)
        defDim(out, "lev", Nz)
        defDim(out, "time", Nt)

        # Coordinate variables
        v_lon = defVar(out, "lon", Float64, ("lon",))
        v_lon.attrib["units"] = "degrees_east"
        v_lon[:] = lon_sub

        v_lat = defVar(out, "lat", Float64, ("lat",))
        v_lat.attrib["units"] = "degrees_north"
        v_lat[:] = lat_sub

        v_lev = defVar(out, "lev", Float64, ("lev",))
        v_lev.attrib["units"] = "layer"
        v_lev[:] = lev_full

        v_time = defVar(out, "time", Float64, ("time",))
        v_time.attrib["units"] = "hours since 2000-01-01 00:00:00"
        v_time[:] = [Dates.value(t - DateTime(2000, 1, 1)) / 3600000.0 for t in times_sub]

        # Data variables
        for vname in VARS_3D
            defVar(out, vname, Float32, ("lon", "lat", "lev", "time");
                   fillvalue = Float32(-9999))
        end
        for vname in VARS_2D
            defVar(out, vname, Float32, ("lon", "lat", "time");
                   fillvalue = Float32(-9999))
        end

        # Global attributes
        out.attrib["source"] = "NASA GEOS-FP via OPeNDAP"
        out.attrib["url"] = OPENDAP_URL
        out.attrib["date_range"] = "2025-02-01 to 2025-02-07"
        out.attrib["lon_stride"] = LON_STRIDE
        out.attrib["lat_stride"] = LAT_STRIDE
        out.attrib["created"] = string(now())

        # Download timestep by timestep
        for (local_t, global_t) in enumerate(time_indices)
            t_str = string(times_all[global_t])
            print("  [$local_t/$Nt] $t_str ... ")
            t0 = time()

            for attempt in 1:3
                try
                    for vname in VARS_3D
                        data = Float32.(ds[vname][lon_idx, lat_idx, :, global_t])
                        out[vname][:, :, :, local_t] = data
                    end
                    for vname in VARS_2D
                        data = Float32.(ds[vname][lon_idx, lat_idx, global_t])
                        out[vname][:, :, local_t] = data
                    end
                    break
                catch e
                    if attempt < 3
                        println("retry $attempt...")
                        sleep(5)
                    else
                        println("FAILED after 3 attempts: $e")
                        rethrow()
                    end
                end
            end

            elapsed = round(time() - t0, digits=1)
            println("done ($(elapsed)s)")
        end
    end

    close(ds)

    fsize = round(filesize(OUT_FILE) / 1e6, digits=1)
    println("\nDownload complete: $OUT_FILE ($fsize MB)")
    println("Grid: $Nx lon × $Ny lat × $Nz lev × $Nt times")
end

download_geosfp_week()
