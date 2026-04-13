#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download one week of ERA5 data via CDS API at 2° resolution
#
# Target: 2025-02-01 to 2025-02-07, 6-hourly
# Resolution: 2° × 2° (90 lon × 46 lat) × 37 pressure levels
# Strategy: Download one day at a time to stay within CDS cost limits,
#           then merge into a single NetCDF file.
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

const DATA_DIR = expanduser("~/data/metDrivers/era5")
const PL_FILE = joinpath(DATA_DIR, "era5_2deg_20250201_20250207.nc")
const SFC_FILE = joinpath(DATA_DIR, "era5_2deg_sfc_20250201_20250207.nc")
const PYTHON = "/opt/miniconda/bin/python3"

const START_DATE = Date(2025, 2, 1)
const END_DATE   = Date(2025, 2, 7)
const DAYS = [START_DATE + Day(d) for d in 0:6]

const PRESSURE_LEVELS = [
    "10", "20", "30", "50", "70", "100", "150", "200", "250",
    "300", "400", "500", "600", "700", "800", "850", "900", "925", "950", "1000",
]

function download_era5_week()
    mkpath(DATA_DIR)

    # Check credentials
    cdsrc = expanduser("~/.cdsapirc")
    if !isfile(cdsrc)
        println("ERROR: ~/.cdsapirc not found.")
        return false
    end
    println("CDS API credentials found.")

    # --- Pressure levels (day-by-day) ---
    if !isfile(PL_FILE)
        println("\n=== Downloading ERA5 pressure levels (day by day) ===")
        daily_pl_files = String[]
        for day in DAYS
            daystr = Dates.format(day, "yyyymmdd")
            dayfile = joinpath(DATA_DIR, "era5_pl_$(daystr).nc")
            push!(daily_pl_files, dayfile)
            if isfile(dayfile)
                println("  $daystr: already downloaded")
                continue
            end
            println("  $daystr: requesting...")
            _download_era5_pl_day(day, dayfile)
        end
        println("Merging daily pressure-level files...")
        _merge_nc_files(daily_pl_files, PL_FILE)
        # Clean up daily files
        for f in daily_pl_files
            isfile(f) && rm(f)
        end
        println("Merged: $PL_FILE ($(round(filesize(PL_FILE)/1e6, digits=1)) MB)")
    else
        println("Pressure level file already exists: $PL_FILE")
    end

    # --- Single levels (day-by-day) ---
    if !isfile(SFC_FILE)
        println("\n=== Downloading ERA5 single levels (day by day) ===")
        daily_sfc_files = String[]
        for day in DAYS
            daystr = Dates.format(day, "yyyymmdd")
            dayfile = joinpath(DATA_DIR, "era5_sfc_$(daystr).nc")
            push!(daily_sfc_files, dayfile)
            if isfile(dayfile)
                println("  $daystr: already downloaded")
                continue
            end
            println("  $daystr: requesting...")
            _download_era5_sfc_day(day, dayfile)
        end
        println("Merging daily single-level files...")
        _merge_nc_files(daily_sfc_files, SFC_FILE)
        for f in daily_sfc_files
            isfile(f) && rm(f)
        end
        println("Merged: $SFC_FILE ($(round(filesize(SFC_FILE)/1e6, digits=1)) MB)")
    else
        println("Single level file already exists: $SFC_FILE")
    end

    return true
end

function _download_era5_pl_day(day::Date, outfile::String)
    daystr = Dates.format(day, "dd")
    month  = Dates.format(day, "mm")
    year   = Dates.format(day, "yyyy")

    plevels_py = "[" * join(["'$p'" for p in PRESSURE_LEVELS], ", ") * "]"

    python_script = """
import cdsapi
import os

output = "$outfile"
os.makedirs(os.path.dirname(output), exist_ok=True)

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind',
            'vertical_velocity', 'temperature', 'specific_humidity',
        ],
        'pressure_level': $plevels_py,
        'year': '$year',
        'month': '$month',
        'day': ['$daystr'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'grid': [2.0, 2.0],
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""

    tmp = tempname() * ".py"
    write(tmp, python_script)
    t0 = time()
    run(`$PYTHON $tmp`)
    elapsed = round((time() - t0) / 60, digits=1)
    println("    done in $(elapsed) min")
    rm(tmp)
end

function _download_era5_sfc_day(day::Date, outfile::String)
    daystr = Dates.format(day, "dd")
    month  = Dates.format(day, "mm")
    year   = Dates.format(day, "yyyy")

    python_script = """
import cdsapi
import os

output = "$outfile"
os.makedirs(os.path.dirname(output), exist_ok=True)

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['surface_pressure'],
        'year': '$year',
        'month': '$month',
        'day': ['$daystr'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'grid': [2.0, 2.0],
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""

    tmp = tempname() * ".py"
    write(tmp, python_script)
    t0 = time()
    run(`$PYTHON $tmp`)
    elapsed = round((time() - t0) / 60, digits=1)
    println("    done in $(elapsed) min")
    rm(tmp)
end

function _base_type(T)
    # Extract the non-Missing type from Union{Missing, T}
    if T isa Union
        types = Base.uniontypes(T)
        non_missing = filter(t -> t !== Missing, types)
        return length(non_missing) == 1 ? non_missing[1] : T
    end
    return T
end

function _merge_nc_files(files::Vector{String}, outfile::String)
    all_exist = all(isfile, files)
    if !all_exist
        mf = filter(!isfile, files)
        error("Missing files: $mf")
    end

    ds1 = NCDataset(files[1])
    dimnames_list = collect(keys(ds1.dim))
    time_dimname = "valid_time" in dimnames_list ? "valid_time" :
                   "time" in dimnames_list ? "time" :
                   error("No time dimension found in $dimnames_list")

    # Only keep numeric data variables (skip "expver", "number", etc.)
    varnames = String[]
    for k in keys(ds1)
        k == time_dimname && continue
        dims = NCDatasets.dimnames(ds1[k])
        time_dimname in dims || continue
        bt = _base_type(eltype(NCDatasets.variable(ds1, k)))
        bt <: Number || continue
        push!(varnames, k)
    end

    coord_vars = String[]
    for k in keys(ds1)
        k == time_dimname && continue
        dims = NCDatasets.dimnames(ds1[k])
        time_dimname in dims && continue
        bt = _base_type(eltype(NCDatasets.variable(ds1, k)))
        bt <: Number || continue
        push!(coord_vars, k)
    end

    println("  Time dim: $time_dimname, data vars: $varnames, coord vars: $coord_vars")

    NCDataset(outfile, "c") do out
        # Define dimensions
        for (dname, dsize) in ds1.dim
            defDim(out, dname, dname == time_dimname ? Inf : dsize)
        end

        # Define coordinate variables (non-time, numeric)
        for vname in coord_vars
            vraw = NCDatasets.variable(ds1, vname)
            bt = _base_type(eltype(vraw))
            dnames = NCDatasets.dimnames(ds1[vname])
            outv = defVar(out, vname, bt, dnames)
            for aname in keys(ds1[vname].attrib)
                outv.attrib[aname] = ds1[vname].attrib[aname]
            end
        end

        # Define time variable (raw numeric type)
        tv_raw = NCDatasets.variable(ds1, time_dimname)
        bt_time = _base_type(eltype(tv_raw))
        out_tv = defVar(out, time_dimname, bt_time, (time_dimname,))
        for aname in keys(ds1[time_dimname].attrib)
            out_tv.attrib[aname] = ds1[time_dimname].attrib[aname]
        end

        # Define data variables
        for vname in varnames
            vraw = NCDatasets.variable(ds1, vname)
            bt = _base_type(eltype(vraw))
            dnames = NCDatasets.dimnames(ds1[vname])
            defVar(out, vname, bt, dnames)
        end

        # Write coordinate data
        for vname in coord_vars
            vraw = NCDatasets.variable(ds1, vname)
            out[vname][:] = collect(vraw[:])
        end

        # Concatenate time-dependent data
        t_offset = 0
        for fname in files
            ds = NCDataset(fname)
            nt_local = ds.dim[time_dimname]

            tv_r = NCDatasets.variable(ds, time_dimname)
            out[time_dimname][t_offset+1:t_offset+nt_local] = collect(tv_r[:])

            for vname in varnames
                vraw = NCDatasets.variable(ds, vname)
                data = collect(vraw[:])
                dnames = NCDatasets.dimnames(ds[vname])
                tidx = findfirst(==(time_dimname), collect(dnames))
                tidx === nothing && continue
                ndims_v = length(dnames)
                idx = [i == tidx ? (t_offset+1:t_offset+nt_local) : Colon() for i in 1:ndims_v]
                out[vname][idx...] = data
            end

            t_offset += nt_local
            close(ds)
        end
    end

    close(ds1)
end

download_era5_week()
