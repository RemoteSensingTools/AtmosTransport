"""
Read spectral coefficients from a GRIB message into a complex matrix.
Returns spec[n+1, m+1] for m=0..T, n=m..T (upper triangular).
"""
function read_spectral_coeffs!(spec::Matrix{ComplexF64}, msg)
    return read_spectral_coeffs!(spec, msg, Float64[])
end

function read_spectral_coeffs!(spec::Matrix{ComplexF64}, msg, vals::Vector{Float64})
    handle = msg.ptr
    sz = Ref{Csize_t}(0)
    ccall((:codes_get_size, GRIB.eccodes), Cint,
          (Ptr{Cvoid}, Cstring, Ref{Csize_t}), handle, "values", sz)

    resize!(vals, sz[])
    ccall((:codes_get_double_array, GRIB.eccodes), Cint,
          (Ptr{Cvoid}, Cstring, Ptr{Float64}, Ref{Csize_t}),
          handle, "values", vals, sz)

    T = msg["J"]
    fill!(spec, zero(ComplexF64))

    idx = 1
    for m in 0:T
        for n in m:T
            spec[n + 1, m + 1] = complex(vals[idx], vals[idx + 1])
            idx += 2
        end
    end
    return T
end

const SPECTRAL_DAY_CACHE_VERSION = 1

"""
    spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target=0)

Return the deterministic on-disk cache path for one decoded spectral day.

The key includes absolute input paths plus file sizes and mtimes, so replacing
either GRIB file automatically invalidates stale coefficient caches. The cache
stores only the truncated coefficient tensors consumed by the transport
preprocessor, not the original GRIB payload.
"""
function spectral_day_cache_path(cache_dir::AbstractString,
                                 vo_d_path::AbstractString,
                                 lnsp_path::AbstractString;
                                 T_target::Int=0)
    vo_stat = stat(vo_d_path)
    lnsp_stat = stat(lnsp_path)
    key = join((
        "spectral-day-v$SPECTRAL_DAY_CACHE_VERSION",
        abspath(vo_d_path), string(vo_stat.size), string(vo_stat.mtime),
        abspath(lnsp_path), string(lnsp_stat.size), string(lnsp_stat.mtime),
        "T_target=$T_target",
    ), "\0")
    return joinpath(cache_dir, "era5_spectral_" * bytes2hex(sha1(key)) * ".jld2")
end

function _load_spectral_day_cache(path::AbstractString)
    data = JLD2.load(path)
    Int(data["format_version"]) == SPECTRAL_DAY_CACHE_VERSION ||
        error("unsupported spectral cache version in $path")
    hours = Vector{Int}(data["hours"])
    lnsp_all = Dict{Int, Matrix{ComplexF64}}(data["lnsp_all"])
    vo_by_hour = Dict{Int, Array{ComplexF64, 3}}(data["vo_by_hour"])
    d_by_hour = Dict{Int, Array{ComplexF64, 3}}(data["d_by_hour"])
    T = Int(data["T"])
    n_times = Int(data["n_times"])
    return (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times)
end

function _write_spectral_day_cache(path::AbstractString, spec)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())"
    isfile(tmp) && rm(tmp; force=true)
    try
        JLD2.jldsave(tmp;
            format_version = SPECTRAL_DAY_CACHE_VERSION,
            hours = spec.hours,
            lnsp_all = spec.lnsp_all,
            vo_by_hour = spec.vo_by_hour,
            d_by_hour = spec.d_by_hour,
            T = spec.T,
            n_times = spec.n_times)
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

"""
    read_day_spectral(vo_d_path, lnsp_path; T_target=0, cache_dir="")

Read one ERA5 spectral day, optionally using a persistent decoded-coefficient
cache. Empty `cache_dir` disables disk caching and preserves the historical
direct-GRIB path. Cache writes are best-effort: a failed write logs a warning
but never invalidates the decoded in-memory result.
"""
function read_day_spectral(vo_d_path::String, lnsp_path::String;
                           T_target::Int=0,
                           cache_dir::AbstractString="")
    if !isempty(cache_dir)
        path = spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target)
        if isfile(path)
            try
                spec = _load_spectral_day_cache(path)
                @info "  Spectral cache hit: $(path)"
                return spec
            catch err
                @warn "  Spectral cache unreadable; rebuilding from GRIB" path exception=(err, catch_backtrace())
            end
        end

        spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target)
        try
            _write_spectral_day_cache(path, spec)
            @info "  Spectral cache wrote: $(path)"
        catch err
            @warn "  Spectral cache write failed; continuing without cache" path exception=(err, catch_backtrace())
        end
        return spec
    end

    return read_day_spectral_streaming(vo_d_path, lnsp_path; T_target)
end

"""
Streaming spectral reader: reads one day's GRIB data, grouped by hour.
Returns a NamedTuple with hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times.
"""
function read_day_spectral_streaming(vo_d_path::String, lnsp_path::String;
                                     T_target::Int=0)
    # Read T from first LNSP message
    f = GribFile(lnsp_path)
    local T_file::Int
    try
        msg1 = first(f)
        T_file = msg1["J"]
    finally
        destroy(f)
    end
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    # Read all LNSP hours
    lnsp_all = Dict{Int, Matrix{ComplexF64}}()
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    vals_buf = Float64[]
    f = GribFile(lnsp_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            lnsp_all[hour] = copy(@view spec_buf[1:T + 1, 1:T + 1])
        end
    finally
        destroy(f)
    end

    # Read all VO/D hours
    vo_by_hour = Dict{Int, Array{ComplexF64, 3}}()
    d_by_hour  = Dict{Int, Array{ComplexF64, 3}}()
    f = GribFile(vo_d_path)
    try
        for msg in f
            name = msg["shortName"]
            level = msg["level"]
            hour = div(msg["dataTime"], 100)
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            if name == "vo"
                if !haskey(vo_by_hour, hour)
                    vo_by_hour[hour] = zeros(ComplexF64, T + 1, T + 1, Nlevels)
                end
                vo_by_hour[hour][:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            elseif name == "d"
                if !haskey(d_by_hour, hour)
                    d_by_hour[hour] = zeros(ComplexF64, T + 1, T + 1, Nlevels)
                end
                d_by_hour[hour][:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            end
        end
    finally
        destroy(f)
    end

    hours = sort(collect(keys(lnsp_all)))
    return (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times=length(hours))
end

function read_hour0_spectral(spectral_dir::String, date::Date;
                             T_target::Int=0,
                             cache_dir::AbstractString="")
    date_str = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    (!isfile(vo_d_path) || !isfile(lnsp_path)) && return nothing

    if !isempty(cache_dir)
        path = spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target)
        if isfile(path)
            try
                spec = _load_spectral_day_cache(path)
                hour0 = first(spec.hours)
                return (lnsp=spec.lnsp_all[hour0],
                        vo=spec.vo_by_hour[hour0],
                        d=spec.d_by_hour[hour0],
                        T=spec.T)
            catch err
                @warn "  Spectral hour-0 cache unreadable; falling back to GRIB" path exception=(err, catch_backtrace())
            end
        end
    end

    # Read T from first LNSP message
    f = GribFile(lnsp_path)
    local T_file::Int
    try
        msg1 = first(f)
        T_file = msg1["J"]
    finally
        destroy(f)
    end
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    # Read hour-0 LNSP
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    vals_buf = Float64[]
    lnsp_h0 = nothing
    f = GribFile(lnsp_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            if hour == 0
                read_spectral_coeffs!(spec_buf, msg, vals_buf)
                lnsp_h0 = copy(@view spec_buf[1:T + 1, 1:T + 1])
                break
            end
        end
    finally
        destroy(f)
    end
    lnsp_h0 === nothing && return nothing

    # Read hour-0 VO/D
    vo_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    d_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    f = GribFile(vo_d_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            hour == 0 || continue
            name = msg["shortName"]
            level = msg["level"]
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            if name == "vo"
                vo_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            elseif name == "d"
                d_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            end
        end
    finally
        destroy(f)
    end

    return (lnsp=lnsp_h0, vo=vo_h0, d=d_h0, T=T)
end
