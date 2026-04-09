"""
Read spectral coefficients from a GRIB message into a complex matrix.
Returns spec[n+1, m+1] for m=0..T, n=m..T (upper triangular).
"""
function read_spectral_coeffs!(spec::Matrix{ComplexF64}, msg)
    handle = msg.ptr
    sz = Ref{Csize_t}(0)
    ccall((:codes_get_size, GRIB.eccodes), Cint,
          (Ptr{Cvoid}, Cstring, Ref{Csize_t}), handle, "values", sz)

    vals = Vector{Float64}(undef, sz[])
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

"""
Streaming spectral reader: reads one day's GRIB data, grouped by hour.
Returns a NamedTuple with hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times.
"""
function read_day_spectral_streaming(vo_d_path::String, lnsp_path::String;
                                     T_target::Int=0)
    f = GribFile(lnsp_path)
    msg1 = first(f)
    T_file = msg1["J"]
    destroy(f)
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    f = GribFile(lnsp_path)
    lnsp_all = Dict{Int, Matrix{ComplexF64}}()
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    for msg in f
        hour = div(msg["dataTime"], 100)
        read_spectral_coeffs!(spec_buf, msg)
        lnsp_all[hour] = copy(@view spec_buf[1:T + 1, 1:T + 1])
    end
    destroy(f)

    vo_by_hour = Dict{Int, Array{ComplexF64, 3}}()
    d_by_hour  = Dict{Int, Array{ComplexF64, 3}}()
    f = GribFile(vo_d_path)
    for msg in f
        name = msg["shortName"]
        level = msg["level"]
        hour = div(msg["dataTime"], 100)
        read_spectral_coeffs!(spec_buf, msg)
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
    destroy(f)

    hours = sort(collect(keys(lnsp_all)))
    return (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times=length(hours))
end

function read_hour0_spectral(spectral_dir::String, date::Date; T_target::Int=0)
    date_str = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    (!isfile(vo_d_path) || !isfile(lnsp_path)) && return nothing

    f = GribFile(lnsp_path)
    msg1 = first(f)
    T_file = msg1["J"]
    destroy(f)
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    lnsp_h0 = nothing
    f = GribFile(lnsp_path)
    for msg in f
        hour = div(msg["dataTime"], 100)
        if hour == 0
            read_spectral_coeffs!(spec_buf, msg)
            lnsp_h0 = copy(@view spec_buf[1:T + 1, 1:T + 1])
            break
        end
    end
    destroy(f)
    lnsp_h0 === nothing && return nothing

    vo_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    d_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    f = GribFile(vo_d_path)
    for msg in f
        hour = div(msg["dataTime"], 100)
        hour == 0 || continue
        name = msg["shortName"]
        level = msg["level"]
        read_spectral_coeffs!(spec_buf, msg)
        if name == "vo"
            vo_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
        elseif name == "d"
            d_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
        end
    end
    destroy(f)

    return (lnsp=lnsp_h0, vo=vo_h0, d=d_h0, T=T)
end
