#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Download ERA5 model-level data via CDS API (reanalysis-era5-complete).
#
# Variables: U (131), V (132), omega (135), LNSP (152)
# Levels: 50-137 (troposphere + lower stratosphere, ~55 hPa to surface)
#
# Configurable via environment variables:
#   RESOLUTION    — grid spacing in degrees (default: 1.0)
#   START_DATE    — first date (default: 2024-01-01)
#   END_DATE      — last date (default: 2024-12-31)
#   DATA_DIR      — output directory
#   LEVEL_TOP     — topmost model level to download (default: 50)
#   LEVEL_BOT     — bottommost model level (default: 137)
#   PYTHON        — path to python3 with cdsapi installed
#   MAX_PARALLEL  — concurrent CDS requests per month (default: 4)
#   MAX_RETRIES   — retries per failed download (default: 3)
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

const RESOLUTION    = parse(Float64, get(ENV, "RESOLUTION", "1.0"))
const START_DATE    = Date(get(ENV, "START_DATE", "2024-01-01"))
const END_DATE      = Date(get(ENV, "END_DATE",   "2024-12-31"))
const LEVEL_TOP     = parse(Int, get(ENV, "LEVEL_TOP", "50"))
const LEVEL_BOT     = parse(Int, get(ENV, "LEVEL_BOT", "137"))
const MAX_PARALLEL  = parse(Int, get(ENV, "MAX_PARALLEL", "4"))
const MAX_RETRIES   = parse(Int, get(ENV, "MAX_RETRIES", "3"))
const DATA_DIR      = expanduser(get(ENV, "DATA_DIR",
    joinpath("~/data/metDrivers/era5",
             "era5_ml_$(replace(string(RESOLUTION), "." => ""))deg_" *
             Dates.format(START_DATE, "yyyymmdd") * "_" *
             Dates.format(END_DATE,   "yyyymmdd"))))
const PYTHON        = get(ENV, "PYTHON", "/usr/bin/python3")

function main()
    mkpath(DATA_DIR)
    levelist = join(LEVEL_TOP:LEVEL_BOT, "/")
    n_levels = LEVEL_BOT - LEVEL_TOP + 1

    println("ERA5 model-level download configuration:")
    println("  Resolution: $(RESOLUTION)°")
    println("  Period: $START_DATE to $END_DATE")
    println("  Levels: $LEVEL_TOP-$LEVEL_BOT ($n_levels levels)")
    println("  Parallel requests: $MAX_PARALLEL")
    println("  Max retries: $MAX_RETRIES")
    println("  Output: $DATA_DIR")

    cdsrc = expanduser("~/.cdsapirc")
    if !isfile(cdsrc)
        error("~/.cdsapirc not found. Configure CDS API credentials first.")
    end

    days = collect(START_DATE:Day(1):END_DATE)
    months = unique(Dates.format.(days, "yyyy-mm"))
    n_days = length(days)
    est_gb = n_days * 0.15  # ~150 MB/day at 1-deg L88
    println("  Total days: $n_days, months: $(length(months))")
    println("  Estimated download: ~$(round(est_gb, digits=1)) GB")

    for month_str in months
        y, m = split(month_str, "-")
        month_days = filter(d -> Dates.format(d, "yyyy-mm") == month_str, days)

        ml_merged = joinpath(DATA_DIR, "era5_ml_$(y)$(m).nc")

        if isfile(ml_merged)
            println("  [$month_str] Already exists: $ml_merged")
            continue
        end

        # Identify which days still need downloading
        daily_files = String[]
        days_to_download = Tuple{Date, String}[]
        for day in month_days
            daystr = Dates.format(day, "yyyymmdd")
            dayfile = joinpath(DATA_DIR, "era5_ml_$(daystr).nc")
            push!(daily_files, dayfile)
            if isfile(dayfile)
                println("  $daystr: already downloaded")
            else
                push!(days_to_download, (day, dayfile))
            end
        end

        # Download missing days in parallel batches
        if !isempty(days_to_download)
            println("  [$month_str] Downloading $(length(days_to_download)) days " *
                    "($(MAX_PARALLEL) parallel)...")
            t_month = time()
            _download_parallel(days_to_download, levelist, MAX_PARALLEL, MAX_RETRIES)
            elapsed = round((time() - t_month) / 60, digits=1)
            println("  [$month_str] Downloads finished in $(elapsed) min")
        end

        existing = filter(isfile, daily_files)
        if isempty(existing)
            println("  WARNING: [$month_str] No daily files, skipping merge")
            continue
        end

        if length(existing) < length(month_days)
            n_missing = length(month_days) - length(existing)
            @warn "[$month_str] $n_missing daily files missing — merging available $(length(existing))"
        end

        println("  [$month_str] Merging $(length(existing)) daily files → $ml_merged")
        _merge_nc_files(existing, ml_merged)
        for f in existing; rm(f); end
        sz = round(filesize(ml_merged) / 1e6, digits=1)
        println("  [$month_str] Merged: $ml_merged ($(sz) MB)")
    end

    println("\nAll downloads complete.")
    println("Files in $DATA_DIR:")
    total_gb = 0.0
    for f in sort(readdir(DATA_DIR))
        sz_bytes = filesize(joinpath(DATA_DIR, f))
        sz_mb = round(sz_bytes / 1e6, digits=1)
        total_gb += sz_bytes / 1e9
        println("  $f  ($(sz_mb) MB)")
    end
    println("  Total: $(round(total_gb, digits=2)) GB")
end

"""
Download days in parallel using asyncmap with bounded concurrency.
Failed downloads are retried up to `max_retries` times with exponential backoff.
"""
function _download_parallel(days_and_files::Vector{Tuple{Date, String}},
                            levelist::String, max_parallel::Int, max_retries::Int)
    asyncmap(days_and_files; ntasks=max_parallel) do (day, dayfile)
        daystr = Dates.format(day, "yyyymmdd")
        for attempt in 1:max_retries
            try
                if attempt > 1
                    wait_sec = 30 * 2^(attempt - 2)
                    println("    $daystr: retry $attempt/$max_retries (waiting $(wait_sec)s)...")
                    sleep(wait_sec)
                end
                _download_ml_day(day, dayfile, levelist)
                if isfile(dayfile)
                    sz = round(filesize(dayfile) / 1e6, digits=1)
                    println("    $daystr: OK ($(sz) MB)")
                    return
                end
            catch e
                @warn "    $daystr: attempt $attempt failed" exception=e
            end
        end
        @error "$daystr: all $max_retries attempts failed — skipping"
    end
end

function _download_ml_day(day::Date, outfile::String, levelist::String)
    daystr = Dates.format(day, "dd")
    month  = Dates.format(day, "mm")
    year   = Dates.format(day, "yyyy")
    datestr = Dates.format(day, "yyyy-mm-dd")

    # LNSP is a surface-level field on model levels (level 1 only).
    # U, V, omega are on the requested model levels.
    # We download them in separate requests and merge.

    ml_file = outfile * ".ml_tmp"
    lnsp_file = outfile * ".lnsp_tmp"

    python_ml = """
import cdsapi
import os

output = "$ml_file"
os.makedirs(os.path.dirname(output), exist_ok=True)

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-complete',
    {
        'class': 'ea',
        'expver': '1',
        'stream': 'oper',
        'type': 'an',
        'levtype': 'ml',
        'levelist': '$levelist',
        'param': '131/132/135',
        'date': '$datestr',
        'time': '00:00:00/06:00:00/12:00:00/18:00:00',
        'grid': '$RESOLUTION/$RESOLUTION',
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded ML: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""

    python_lnsp = """
import cdsapi
import os

output = "$lnsp_file"
os.makedirs(os.path.dirname(output), exist_ok=True)

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-complete',
    {
        'class': 'ea',
        'expver': '1',
        'stream': 'oper',
        'type': 'an',
        'levtype': 'ml',
        'levelist': '1',
        'param': '152',
        'date': '$datestr',
        'time': '00:00:00/06:00:00/12:00:00/18:00:00',
        'grid': '$RESOLUTION/$RESOLUTION',
        'format': 'netcdf',
    },
    output
)
print(f"Downloaded LNSP: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
"""

    _run_python(python_ml)
    _run_python(python_lnsp)

    # Merge ML fields and LNSP into a single file
    _combine_ml_lnsp(ml_file, lnsp_file, outfile)
    rm(ml_file; force=true)
    rm(lnsp_file; force=true)
end

function _combine_ml_lnsp(ml_file::String, lnsp_file::String, outfile::String)
    if !isfile(ml_file) || !isfile(lnsp_file)
        @warn "Missing files for combination, skipping"
        return
    end

    ds_ml = NCDataset(ml_file)
    ds_lnsp = NCDataset(lnsp_file)

    NCDataset(outfile, "c") do out
        # Copy dimensions from ML file
        for (dname, dsize) in ds_ml.dim
            defDim(out, dname, dname in ("time", "valid_time") ? Inf : dsize)
        end

        # Copy all variables from ML file
        for vname in keys(ds_ml)
            v = ds_ml[vname]
            vraw = NCDatasets.variable(ds_ml, vname)
            bt = _base_type(eltype(vraw))
            dnames = NCDatasets.dimnames(v)
            outv = defVar(out, vname, bt, dnames)
            for aname in keys(v.attrib)
                outv.attrib[aname] = v.attrib[aname]
            end
            data = Array(vraw)
            idxs = ntuple(i -> 1:size(data, i), ndims(data))
            outv[idxs...] = data
        end

        # Add LNSP from the surface file (it has no level dimension, or level=1)
        lnsp_key = nothing
        for k in keys(ds_lnsp)
            if lowercase(k) == "lnsp" || k == "152"
                lnsp_key = k
                break
            end
        end

        if lnsp_key !== nothing
            lnsp_var = ds_lnsp[lnsp_key]
            lnsp_raw = NCDatasets.variable(ds_lnsp, lnsp_key)
            bt = _base_type(eltype(lnsp_raw))

            # LNSP dims: (lon, lat, time) or (lon, lat, level, time)
            lnsp_dims = NCDatasets.dimnames(lnsp_var)
            lnsp_data = collect(lnsp_raw[:])

            # Squeeze out any level dimension (may be named "level" or "model_level")
            level_dim = findfirst(d -> d in ("level", "model_level"), collect(lnsp_dims))
            if level_dim !== nothing
                level_name = collect(lnsp_dims)[level_dim]
                lnsp_data = dropdims(lnsp_data; dims=level_dim)
                lnsp_dims_out = filter(!=(level_name), collect(lnsp_dims))
            else
                lnsp_dims_out = collect(lnsp_dims)
            end

            # Ensure lon/lat dims exist
            for d in lnsp_dims_out
                if !haskey(out.dim, d)
                    defDim(out, d, size(lnsp_data, findfirst(==(d), lnsp_dims_out)))
                end
            end

            outv = defVar(out, "lnsp", bt, Tuple(lnsp_dims_out))
            for aname in keys(lnsp_var.attrib)
                outv.attrib[aname] = lnsp_var.attrib[aname]
            end
            idxs = ntuple(i -> 1:size(lnsp_data, i), ndims(lnsp_data))
            outv[idxs...] = lnsp_data
        else
            @warn "LNSP variable not found in $lnsp_file"
        end
    end

    close(ds_ml)
    close(ds_lnsp)
end

function _run_python(script::String)
    tmp = tempname() * ".py"
    write(tmp, script)
    t0 = time()
    try
        run(`$PYTHON $tmp`)
        elapsed = round((time() - t0) / 60, digits=1)
        println("    done in $(elapsed) min")
        return true
    catch e
        @warn "Download failed: $e"
        return false
    finally
        rm(tmp; force=true)
    end
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
