# ---------------------------------------------------------------------------
# File discovery utilities for preprocessed met data and emissions
# ---------------------------------------------------------------------------

using Dates

"""
    find_massflux_shards(dir, ft_tag) → Vector{String}

Discover monthly mass-flux shard files in `dir`.
Prefers `.bin` over `.nc` for each month. Returns sorted list of file paths.

`ft_tag` is `"float32"` or `"float64"`.
"""
function find_massflux_shards(dir::String, ft_tag::String)
    all_files = readdir(dir; join=true)
    months = Dict{String, String}()
    for f in all_files
        bn = basename(f)
        m = match(Regex("massflux_era5_(?:\\w+_)?(\\d{6})_" * ft_tag), bn)
        m === nothing && continue
        month_key = m[1]
        if endswith(bn, ".bin")
            months[month_key] = f          # .bin always wins
        elseif endswith(bn, ".nc") && !haskey(months, month_key)
            months[month_key] = f
        end
    end
    return [months[k] for k in sort(collect(keys(months)))]
end

"""
    find_preprocessed_cs_files(dir, start_date, end_date, ft_tag) → Vector{String}

Find preprocessed cubed-sphere binary files in `dir` for the given date range.
Expected filename pattern: `geosfp_cs_YYYYMMDD_<ft_tag>.bin`.
"""
function find_preprocessed_cs_files(dir::String, start_date::Date, end_date::Date,
                                     ft_tag::String)
    files = String[]
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        fp = joinpath(dir, "geosfp_cs_$(datestr)_$(ft_tag).bin")
        isfile(fp) && push!(files, fp)
    end
    return files
end

"""
    find_geosfp_cs_files(datadir, start_date, end_date; product="geosfp_c720") → Vector{String}

Find raw GEOS cubed-sphere NetCDF files in `datadir`.

For `:hourly` products (GEOS-FP): expects `datadir/YYYYMMDD/*.nc4` (24 files/day).
For `:daily` products (GEOS-IT): expects `datadir/YYYYMMDD/GEOSIT.*.nc` (1 file/day).
"""
function find_geosfp_cs_files(datadir::String, start_date::Date, end_date::Date;
                              product::String = "geosfp_c720")
    info = GEOS_CS_PRODUCTS[product]
    files = String[]
    if info.layout === :hourly
        for date in start_date:Day(1):end_date
            daydir = joinpath(datadir, Dates.format(date, "yyyymmdd"))
            isdir(daydir) || continue
            for f in sort(readdir(daydir))
                if contains(f, "tavg_1hr_ctm_c0720_v72") && endswith(f, ".nc4")
                    push!(files, joinpath(daydir, f))
                end
            end
        end
    else  # :daily
        tag = "CTM_A1.C$(info.Nc)"
        for date in start_date:Day(1):end_date
            daydir = joinpath(datadir, Dates.format(date, "yyyymmdd"))
            isdir(daydir) || continue
            for f in sort(readdir(daydir))
                if contains(f, tag) && endswith(f, ".nc")
                    push!(files, joinpath(daydir, f))
                end
            end
        end
    end
    return files
end

"""
    find_era5_files(datadirs) → Vector{String}

Find ERA5 model-level NetCDF files across multiple data directories.
Matches files starting with `era5_ml_` and ending with `.nc`, excluding `_tmp` files.
"""
function find_era5_files(datadirs::Vector{String})
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

"""
    ensure_local_cache(src_path) → String

Copy binary file to local fast storage (NVMe) if not already cached.
Returns the path to use. Set `LOCAL_CACHE_DIR` environment variable to control
the cache location (default: `/var/tmp/massflux_cache`).
"""
function ensure_local_cache(src_path::String)
    cache_dir = get(ENV, "LOCAL_CACHE_DIR", "/var/tmp/massflux_cache")
    try
        mkpath(cache_dir)
    catch
        @warn "Cannot create local cache dir $cache_dir; using original path"
        return src_path
    end
    dst = joinpath(cache_dir, basename(src_path))
    if isfile(dst) && filesize(dst) == filesize(src_path) && mtime(dst) >= mtime(src_path)
        @info "  Using local cache: $dst"
        return dst
    end
    @info "  Copying to local NVMe cache: $dst ($(round(filesize(src_path)/1e9, digits=2)) GB)..."
    t0 = time()
    try
        cp(src_path, dst; force=true)
        chmod(dst, 0o644)  # ensure cached file is writable (mmap requires it)
        @info "  Cache copy done in $(round(time() - t0, digits=1))s"
        return dst
    catch e
        @warn "Cache copy failed ($e); using original path"
        rm(dst; force=true)  # clean up partial copy
        return src_path
    end
end
