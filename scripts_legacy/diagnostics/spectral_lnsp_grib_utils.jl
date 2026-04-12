# Shared spectral LNSP → grid helpers for diagnostics (included by spectral_lnsp_*.jl).

using GRIB
using FFTW
using Dates
using Printf
using TOML
using Statistics

const _defaults = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config", "defaults.toml"))
const R_EARTH = Float64(_defaults["planet"]["radius"])
const GRAV = Float64(_defaults["planet"]["gravity"])

struct TargetGrid
    Nlon::Int
    Nlat::Int
    lons::Vector{Float64}
    lats::Vector{Float64}
    dlon::Float64
    dlat::Float64
    cos_lat::Vector{Float64}
    area::Matrix{Float64}
end

function TargetGrid(Nlon, Nlat)
    dlon_deg = 360.0 / Nlon
    dlat_deg = 180.0 / Nlat
    south_deg = -90.0 + dlat_deg / 2
    lons = [dlon_deg * (i - 0.5) for i in 1:Nlon]
    lats = [south_deg + dlat_deg * (j - 1) for j in 1:Nlat]
    dlon = deg2rad(dlon_deg)
    dlat = deg2rad(dlat_deg)
    cos_lat = [cosd(lat) for lat in lats]
    blat_deg = [south_deg + dlat_deg * (j - 1) - dlat_deg / 2 for j in 1:Nlat+1]
    blat_deg[1] = max(blat_deg[1], -90.0)
    blat_deg[end] = min(blat_deg[end], 90.0)
    area = zeros(Nlon, Nlat)
    for j in 1:Nlat
        cell_a = R_EARTH^2 * dlon * abs(sind(blat_deg[j+1]) - sind(blat_deg[j]))
        for i in 1:Nlon
            area[i, j] = cell_a
        end
    end
    return TargetGrid(Nlon, Nlat, lons, lats, dlon, dlat, cos_lat, area)
end

function compute_legendre_column!(P::Matrix{Float64}, T::Int, sin_lat::Float64)
    cos_lat = sqrt(1.0 - sin_lat^2)
    fill!(P, 0.0)
    P[1, 1] = 1.0
    for m in 1:T
        P[m+1, m+1] = sqrt((2m + 1.0) / (2m)) * cos_lat * P[m, m]
    end
    for m in 0:(T-1)
        P[m+2, m+1] = sqrt(2m + 3.0) * sin_lat * P[m+1, m+1]
    end
    for m in 0:T
        for n in (m+2):T
            n2 = n * n
            m2 = m * m
            a = sqrt((4.0 * n2 - 1.0) / (n2 - m2))
            b = sqrt(((2n + 1) * (n - m - 1) * (n + m - 1)) /
                     ((2n - 3) * (n2 - m2)))
            P[n+1, m+1] = a * sin_lat * P[n, m+1] - b * P[n-1, m+1]
        end
    end
    return nothing
end

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
            spec[n+1, m+1] = complex(vals[idx], vals[idx+1])
            idx += 2
        end
    end
    return T
end

function spectral_to_grid!(field::Matrix{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lats::Vector{Float64},
                           Nlon::Int,
                           P_buf::Matrix{Float64},
                           fft_buf::Vector{ComplexF64};
                           lon_shift_rad::Float64=0.0)
    Nlat = length(lats)
    Nfft = Nlon
    for j in 1:Nlat
        sin_lat = sind(lats[j])
        compute_legendre_column!(P_buf, T, sin_lat)
        fill!(fft_buf, zero(ComplexF64))
        for m in 0:min(T, div(Nfft, 2))
            Gm = zero(ComplexF64)
            @inbounds for n in m:T
                Gm += spec[n+1, m+1] * P_buf[n+1, m+1]
            end
            if lon_shift_rad != 0.0 && m > 0
                Gm *= exp(im * m * lon_shift_rad)
            end
            fft_buf[m+1] = Gm
        end
        for m in 1:min(T, div(Nfft, 2) - 1)
            fft_buf[Nfft - m + 1] = conj(fft_buf[m + 1])
        end
        f_lon = bfft(fft_buf)
        @inbounds for i in 1:Nlon
            field[i, j] = real(f_lon[i])
        end
    end
    return nothing
end

function grib_datetime(msg)::DateTime
    dd = Int(msg["dataDate"])
    dt = Int(msg["dataTime"])
    h = div(dt, 100)
    mi = mod(dt, 100)
    y, r = divrem(dd, 10000)
    mo, da = divrem(r, 100)
    return DateTime(y, mo, da, h, mi)
end

function lnsp_spec_to_ps!(ps::Matrix{Float64}, lnsp_spec::AbstractMatrix{ComplexF64}, T::Int,
                          grid::TargetGrid, P_buf, fft_buf, lnsp_grid::Matrix{Float64})
    center_shift = grid.dlon / 2
    spectral_to_grid!(lnsp_grid, lnsp_spec, T, grid.lats, grid.Nlon, P_buf, fft_buf;
                     lon_shift_rad=center_shift)
    @. ps = exp(lnsp_grid)
    return nothing
end

function total_mass_kg(lnsp_spec::Matrix{ComplexF64}, T::Int, grid::TargetGrid,
                       P_buf, fft_buf, field_2d::Matrix{Float64})
    center_shift = grid.dlon / 2
    spectral_to_grid!(field_2d, lnsp_spec, T, grid.lats, grid.Nlon, P_buf, fft_buf;
                     lon_shift_rad=center_shift)
    return sum(@. exp(field_2d) * grid.area) / GRAV
end

function collect_lnsp_series(spectral_dir::String;
                             Nlon::Int=720, Nlat::Int=361)
    grid = TargetGrid(Nlon, Nlat)
    T_target = div(Nlon, 2) - 1
    files = String[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb$", f)
        m !== nothing && push!(files, joinpath(spectral_dir, f))
    end
    sort!(files)
    isempty(files) && error("No era5_spectral_*_lnsp.gb in $spectral_dir")

    f0 = GribFile(first(files))
    msg0 = first(f0)
    T_file = msg0["J"]
    destroy(f0)
    T = min(T_target, T_file)

    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    lnsp_work = zeros(ComplexF64, T + 1, T + 1)
    P_buf = zeros(T + 1, T + 1)
    fft_buf = zeros(ComplexF64, grid.Nlon)
    field_2d = zeros(grid.Nlon, grid.Nlat)

    series = Tuple{DateTime,Float64}[]
    for path in files
        f = GribFile(path)
        for msg in f
            read_spectral_coeffs!(spec_buf, msg)
            lnsp_work .= @view spec_buf[1:T+1, 1:T+1]
            M = total_mass_kg(lnsp_work, T, grid, P_buf, fft_buf, field_2d)
            push!(series, (grib_datetime(msg), M))
        end
        destroy(f)
    end
    sort!(series, by=x -> x[1])
    out = Tuple{DateTime,Float64}[]
    for s in series
        if !isempty(out) && out[end][1] == s[1]
            continue
        end
        push!(out, s)
    end
    return out, T
end

"""Load every LNSP message into a time-ordered coefficient stack (RAM-heavy for long series)."""
function collect_lnsp_coeff_stack(spectral_dir::String;
                                  Nlon::Int=720, Nlat::Int=361)
    grid = TargetGrid(Nlon, Nlat)
    T_target = div(Nlon, 2) - 1
    files = String[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb$", f)
        m !== nothing && push!(files, joinpath(spectral_dir, f))
    end
    sort!(files)
    isempty(files) && error("No era5_spectral_*_lnsp.gb in $spectral_dir")

    f0 = GribFile(first(files))
    msg0 = first(f0)
    T_file = msg0["J"]
    destroy(f0)
    T = min(T_target, T_file)

    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    lnsp_work = zeros(ComplexF64, T + 1, T + 1)

    raw = Tuple{DateTime,Matrix{ComplexF64}}[]
    for path in files
        f = GribFile(path)
        for msg in f
            read_spectral_coeffs!(spec_buf, msg)
            lnsp_work .= @view spec_buf[1:T+1, 1:T+1]
            push!(raw, (grib_datetime(msg), copy(lnsp_work)))
        end
        destroy(f)
    end
    sort!(raw, by=x -> x[1])
    dedup = Tuple{DateTime,Matrix{ComplexF64}}[]
    for s in raw
        if !isempty(dedup) && dedup[end][1] == s[1]
            continue
        end
        push!(dedup, s)
    end
    N = length(dedup)
    coeff = Array{ComplexF64,3}(undef, T + 1, T + 1, N)
    times = Vector{DateTime}(undef, N)
    for k in 1:N
        times[k] = dedup[k][1]
        coeff[:, :, k] .= dedup[k][2]
    end
    return times, coeff, grid, T
end
