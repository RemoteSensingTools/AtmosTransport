#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download ERA5 pressure-level and single-level data via CDS API.
#
# Configurable via environment variables or command-line arguments:
#   RESOLUTION  — grid spacing in degrees (default: 0.25)
#   START_DATE  — first date to download (default: 2024-06-01)
#   END_DATE    — last date to download (default: 2024-08-31)
#   DATA_DIR    — output directory (default: ~/data/metDrivers/era5/)
#
# Strategy: download one day at a time per variable class, then merge
# into monthly NetCDF files. Designed for the CDS API v2 (beta).
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

const RESOLUTION = parse(Float64, get(ENV, "RESOLUTION", "0.25"))
const START_DATE = Date(get(ENV, "START_DATE", "2024-06-01"))
const END_DATE   = Date(get(ENV, "END_DATE",   "2024-08-31"))
const DATA_DIR   = expanduser(get(ENV, "DATA_DIR",
    joinpath("~/data/metDrivers/era5",
             "era5_$(replace(string(RESOLUTION), "." => ""))deg_" *
             Dates.format(START_DATE, "yyyymmdd") * "_" *
             Dates.format(END_DATE,   "yyyymmdd"))))
const PYTHON     = get(ENV, "PYTHON", "/opt/miniconda/bin/python3")

const PRESSURE_LEVELS_37 = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600", "650", "700",
    "750", "775", "800", "825", "850", "875", "900", "925",
    "950", "975", "1000",
]

function main()
    mkpath(DATA_DIR)
    println("ERA5 download configuration:")
    println("  Resolution: $(RESOLUTION)°")
    println("  Period: $START_DATE to $END_DATE")
    println("  Output: $DATA_DIR")
    println("  Pressure levels: $(length(PRESSURE_LEVELS_37))")

    cdsrc = expanduser("~/.cdsapirc")
    if !isfile(cdsrc)
        error("~/.cdsapirc not found. Configure CDS API credentials first.")
    end

    days = collect(START_DATE:Day(1):END_DATE)
    months = unique(Dates.format.(days, "yyyy-mm"))
    println("  Total days: $(length(days)), months: $months")

    for month_str in months
        y, m = split(month_str, "-")
        month_days = filter(d -> Dates.format(d, "yyyy-mm") == month_str, days)

        pl_merged = joinpath(DATA_DIR, "era5_pl_$(y)$(m).nc")
        sfc_merged = joinpath(DATA_DIR, "era5_sfc_$(y)$(m).nc")

        _download_and_merge_month(month_days, pl_merged, :pressure_levels)
        _download_and_merge_month(month_days, sfc_merged, :single_levels)
    end

    println("\nAll downloads complete.")
    println("Files in $DATA_DIR:")
    for f in sort(readdir(DATA_DIR))
        sz = round(filesize(joinpath(DATA_DIR, f)) / 1e6, digits=1)
        println("  $f  ($(sz) MB)")
    end
end

function _download_and_merge_month(days, merged_file, var_type)
    if isfile(merged_file)
        println("  Already exists: $merged_file")
        return
    end

    daily_files = String[]
    for day in days
        daystr = Dates.format(day, "yyyymmdd")
        tag = var_type == :pressure_levels ? "pl" : "sfc"
        dayfile = joinpath(DATA_DIR, "era5_$(tag)_$(daystr).nc")
        push!(daily_files, dayfile)

        if isfile(dayfile)
            println("  $daystr $tag: already downloaded")
            continue
        end

        println("  $daystr $tag: requesting from CDS...")
        if var_type == :pressure_levels
            _download_pl_day(day, dayfile)
        else
            _download_sfc_day(day, dayfile)
        end
    end

    existing = filter(isfile, daily_files)
    if isempty(existing)
        println("  WARNING: No daily files for $(var_type), skipping merge")
        return
    end

    println("  Merging $(length(existing)) daily files → $merged_file")
    _merge_nc_files(existing, merged_file)

    for f in existing
        rm(f)
    end
    sz = round(filesize(merged_file) / 1e6, digits=1)
    println("  Merged: $merged_file ($(sz) MB)")
end

function _download_pl_day(day::Date, outfile::String)
    daystr = Dates.format(day, "dd")
    month  = Dates.format(day, "mm")
    year   = Dates.format(day, "yyyy")
    plevels_py = "[" * join(["'$p'" for p in PRESSURE_LEVELS_37], ", ") * "]"

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
        'grid': [$RESOLUTION, $RESOLUTION],
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""
    _run_python(python_script)
end

function _download_sfc_day(day::Date, outfile::String)
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
        'grid': [$RESOLUTION, $RESOLUTION],
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""
    _run_python(python_script)
end

function _run_python(script::String)
    tmp = tempname() * ".py"
    write(tmp, script)
    t0 = time()
    try
        run(`$PYTHON $tmp`)
    catch e
        @warn "Download failed: $e"
    finally
        rm(tmp; force=true)
    end
    elapsed = round((time() - t0) / 60, digits=1)
    println("    done in $(elapsed) min")
end

function _base_type(T)
    if T isa Union
        types = Base.uniontypes(T)
        non_missing = filter(t -> t !== Missing, types)
        return length(non_missing) == 1 ? non_missing[1] : T
    end
    return T
end

function _merge_nc_files(files::Vector{String}, outfile::String)
    ds1 = NCDataset(files[1])
    dimnames_list = collect(keys(ds1.dim))
    time_dimname = "valid_time" in dimnames_list ? "valid_time" :
                   "time" in dimnames_list ? "time" :
                   error("No time dimension found in $dimnames_list")

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

    NCDataset(outfile, "c") do out
        for (dname, dsize) in ds1.dim
            defDim(out, dname, dname == time_dimname ? Inf : dsize)
        end
        for vname in coord_vars
            vraw = NCDatasets.variable(ds1, vname)
            bt = _base_type(eltype(vraw))
            dnames = NCDatasets.dimnames(ds1[vname])
            outv = defVar(out, vname, bt, dnames)
            for aname in keys(ds1[vname].attrib)
                outv.attrib[aname] = ds1[vname].attrib[aname]
            end
        end
        tv_raw = NCDatasets.variable(ds1, time_dimname)
        bt_time = _base_type(eltype(tv_raw))
        out_tv = defVar(out, time_dimname, bt_time, (time_dimname,))
        for aname in keys(ds1[time_dimname].attrib)
            out_tv.attrib[aname] = ds1[time_dimname].attrib[aname]
        end
        for vname in varnames
            vraw = NCDatasets.variable(ds1, vname)
            bt = _base_type(eltype(vraw))
            dnames = NCDatasets.dimnames(ds1[vname])
            defVar(out, vname, bt, dnames)
        end
        for vname in coord_vars
            vraw = NCDatasets.variable(ds1, vname)
            out[vname][:] = collect(vraw[:])
        end

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

main()
