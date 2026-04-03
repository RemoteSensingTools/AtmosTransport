#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Add specific humidity (QV) from ERA5 GRIB to preprocessed mass flux NetCDF.
#
# The preprocessed file has dimensions (lon, lat, level, time) with 6-hourly
# windows. This script reads QV from a separate GRIB file and adds it as a
# new variable "qv" to the existing NetCDF.
#
# Usage:
#   julia --project=. scripts/preprocessing/add_qv_to_preprocessed.jl \
#       ~/data/AtmosTransport/met/era5/qv/era5_qv_202112.nc \
#       ~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine/massflux_era5_spectral_202112_float32.nc
# ---------------------------------------------------------------------------

using GRIB
using NCDatasets
using Dates

function main()
    if length(ARGS) < 2
        println("Usage: julia add_qv_to_preprocessed.jl <qv_grib> <preprocessed_nc>")
        return
    end
    qv_grib_path = expanduser(ARGS[1])
    nc_path = expanduser(ARGS[2])

    # Read target NetCDF dimensions
    ds = NCDataset(nc_path, "r")
    Nx = length(ds["lon"])
    Ny = length(ds["lat"])
    Nz = length(ds["m"][1, 1, :, 1])  # levels from air mass variable
    Nt = length(ds["time"])
    lats_nc = Float64.(ds["lat"][:])
    lons_nc = Float64.(ds["lon"][:])
    close(ds)
    println("Target NetCDF: $(Nx)×$(Ny)×$(Nz)×$(Nt)")

    # Read QV from GRIB — organize by time step and level
    println("Reading QV from GRIB: $qv_grib_path")
    qv_data = zeros(Float32, Nx, Ny, Nz, Nt)

    gf = GribFile(qv_grib_path)
    msg_count = 0
    for msg in gf
        paramId = msg["paramId"]
        paramId == 133 || continue  # QV only

        level = msg["level"]
        # Get the time step index from date/time
        date_int = msg["date"]
        time_int = msg["time"]
        hour = time_int ÷ 100

        # Map to window index: 00→1, 06→2, 12→3, 18→4 per day
        year = date_int ÷ 10000
        month = (date_int ÷ 100) % 100
        day = date_int % 100
        day_of_month = day
        win_in_day = hour ÷ 6 + 1
        win_idx = (day_of_month - 1) * 4 + win_in_day

        if win_idx > Nt || level > Nz
            continue
        end

        # Read data — GRIB data is (lon, lat) with lat from N→S
        vals = msg["values"]
        Ni = msg["Ni"]
        Nj = msg["Nj"]
        data_2d = reshape(Float32.(vals), Ni, Nj)

        # Flip lat if needed (GRIB is N→S, our NetCDF is S→N)
        lat_first = msg["latitudeOfFirstGridPointInDegrees"]
        if lat_first > 0  # N→S in GRIB
            data_2d = data_2d[:, end:-1:1]
        end

        # Handle longitude wrapping (GRIB: 0→360, NetCDF: -180→180)
        lon_first = msg["longitudeOfFirstGridPointInDegrees"]
        if lon_first >= 0 && lons_nc[1] < 0
            # Shift: [0,360) → [-180,180)
            half = Ni ÷ 2
            data_2d = vcat(data_2d[half+1:end, :], data_2d[1:half, :])
        end

        if size(data_2d) == (Nx, Ny)
            qv_data[:, :, level, win_idx] .= data_2d
            msg_count += 1
        end
    end
    destroy(gf)
    println("Read $msg_count GRIB messages")

    # Verify we got data
    n_filled = count(qv_data .> 0)
    n_total = length(qv_data)
    println("QV coverage: $(n_filled)/$(n_total) ($(round(n_filled/n_total*100, digits=1))%)")

    # Add to NetCDF
    println("Adding QV to $nc_path ...")
    ds = NCDataset(nc_path, "a")
    if haskey(ds, "qv")
        println("  Overwriting existing qv variable")
        delete!(ds, "qv")
    end
    dim_names = ("lon", "lat", "level", "time")
    defVar(ds, "qv", qv_data, dim_names;
           attrib=Dict("long_name" => "specific_humidity",
                       "units" => "kg kg-1",
                       "source" => "ERA5 reanalysis-era5-complete param 133"))
    close(ds)

    # Verify
    ds = NCDataset(nc_path, "r")
    qv_check = ds["qv"][:, :, :, 1]
    println("QV sample (surface, win 1): min=$(minimum(qv_check[:,:,Nz])), max=$(maximum(qv_check[:,:,Nz]))")
    close(ds)
    println("Done.")
end

main()
