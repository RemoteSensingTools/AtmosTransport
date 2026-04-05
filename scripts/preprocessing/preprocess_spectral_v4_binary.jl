#!/usr/bin/env julia
# ===========================================================================
# Fused ERA5 spectral GRIB -> daily v4 binary preprocessor
#
# Goes directly from spectral VO/D/LNSP GRIB files to merged-level flat
# binary files, skipping the intermediate NetCDF step.
#
# Pipeline per day:
#   1. Read all hourly GRIB spectral data (VO, D, LNSP)
#   2. For each hour t (0-23):
#      a. Spectral -> gridpoint: VO/D -> U/V via SHT, compute am/bm/cm/m
#         at native 137 levels
#      b. Merge to target levels (merge_thin_levels with configurable min_dp)
#      c. Recompute cm from merged am/bm via continuity with B-correction
#      d. Compute deltas: dam = am[t+1] - am[t], dbm = bm[t+1] - bm[t],
#         dm = m[t+1] - m[t]
#         For the last hour of the day: use next day's hour 0 if available,
#         else zeros.
#      e. Write to daily v4 binary: m|am|bm|cm|ps|dam|dbm|dm per window
#   3. All spectral transforms done in Float64, stored as Float32.
#
# Produces ONLY core transport fields (m, am, bm, cm, ps) plus v4 deltas
# (dam, dbm, dm). No convection, surface, temperature, or QV.
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
#       config/preprocessing/era5_spectral_v4.toml [--day 2021-12-01]
#
# Binary layout (v4, core only):
#   [16384-byte JSON header | per-window data x Nt]
#   Per window: m(Nx*Ny*Nz) | am((Nx+1)*Ny*Nz) | bm(Nx*(Ny+1)*Nz) |
#               cm(Nx*Ny*(Nz+1)) | ps(Nx*Ny) | dam((Nx+1)*Ny*Nz) |
#               dbm(Nx*(Ny+1)*Nz) | dm(Nx*Ny*Nz)
# ===========================================================================

using GRIB
using FFTW
using LinearAlgebra: mul!
using JSON3
using Dates
using Printf
using TOML

# ===========================================================================
# Physical constants from defaults.toml (single source of truth)
# ===========================================================================
const _defaults = TOML.parsefile(joinpath(@__DIR__, "..", "..", "config", "defaults.toml"))
const R_EARTH = Float64(_defaults["planet"]["radius"])
const GRAV    = Float64(_defaults["planet"]["gravity"])

# ===========================================================================
# Vertical coordinate types + level merging
# ===========================================================================

struct HybridSigmaPressure{FT}
    A :: Vector{FT}   # Nz+1 interface values
    B :: Vector{FT}   # Nz+1 interface values
end

n_levels(vc::HybridSigmaPressure) = length(vc.A) - 1
pressure_at_interface(vc::HybridSigmaPressure, k, p_s) = vc.A[k] + vc.B[k] * p_s
level_thickness(vc::HybridSigmaPressure, k, p_s) =
    pressure_at_interface(vc, k + 1, p_s) - pressure_at_interface(vc, k, p_s)

function merge_thin_levels(vc::HybridSigmaPressure{FT};
                            min_thickness_Pa::Real = FT(1000),
                            p_surface::Real = FT(101325)) where FT
    Nz = n_levels(vc)
    ps = FT(p_surface); min_dp = FT(min_thickness_Pa)
    dp = [level_thickness(vc, k, ps) for k in 1:Nz]
    # Top-down pass
    top_ifaces = Int[1]; acc = zero(FT)
    for k in 1:Nz
        acc += dp[k]
        if acc >= min_dp; push!(top_ifaces, k + 1); acc = zero(FT); end
    end
    # Bottom-up pass
    bot_ifaces = Int[Nz + 1]; acc = zero(FT)
    for k in Nz:-1:1
        acc += dp[k]
        if acc >= min_dp; pushfirst!(bot_ifaces, k); acc = zero(FT); end
    end
    # Join
    first_bot = bot_ifaces[1]
    keep = Int[]
    for i in top_ifaces; i <= first_bot && push!(keep, i); end
    (isempty(keep) || keep[end] < first_bot) && push!(keep, first_bot)
    for i in bot_ifaces; i > keep[end] && push!(keep, i); end
    # Build merged vc
    merged_vc = HybridSigmaPressure(FT[vc.A[k] for k in keep], FT[vc.B[k] for k in keep])
    Nz_merged = n_levels(merged_vc)
    mm = Vector{Int}(undef, Nz); km = 1
    for k in 1:Nz
        while km < Nz_merged && keep[km + 1] <= k; km += 1; end
        mm[k] = km
    end
    return merged_vc, mm
end

function load_era5_vertical_coordinate(coeff_path::String, level_top::Int, level_bot::Int)
    isfile(coeff_path) || error("Coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    a_all = Float64.(cfg["coefficients"]["a"])
    b_all = Float64.(cfg["coefficients"]["b"])
    return HybridSigmaPressure(a_all[level_top:level_bot+1], b_all[level_top:level_bot+1])
end

# ===========================================================================
# Merge kernels
# ===========================================================================

function merge_cell_field!(merged::Array{FT,3}, native::Array{FT,3}, mm::Vector{Int}) where FT
    Threads.@threads for km in 1:size(merged, 3)
        @views merged[:, :, km] .= zero(FT)
    end
    @inbounds for k in 1:length(mm)
        @views merged[:, :, mm[k]] .+= native[:, :, k]
    end
end

"""
Recompute cm from merged horizontal divergence using TM5's hybrid-coordinate formula.

TM5 formula (advect_tools.F90, dynam0):
  cm[k+1] = cm[k] - div_h[k] + (B[k+1] - B[k]) * pit

where pit = column-integrated horizontal convergence and B are the
hybrid sigma-pressure interface coefficients. The B-correction accounts for the
pressure coordinate surfaces moving as surface pressure changes, naturally giving
cm[1]=0 (TOA) and cm[Nz+1]=0 (surface) without any residual correction.
"""
function recompute_cm_from_divergence!(cm::Array{FT,3}, am::Array{FT,3},
                                       bm::Array{FT,3}, m::Array{FT,3};
                                       B_ifc::Vector{<:Real}=Float64[]) where FT
    Nx = size(m, 1)
    Ny = size(m, 2)
    Nz = size(m, 3)
    fill!(cm, zero(FT))

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        # TM5-style: include B-coefficient correction for hybrid coordinates
        @inbounds for j in 1:Ny, i in 1:Nx
            # Column-integrated horizontal convergence
            pit = 0.0
            for k in 1:Nz
                pit += (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                       (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
            end
            # Build cm from TOA downward with B-correction
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h + (Float64(B_ifc[k+1]) - Float64(B_ifc[k])) * pit
                cm[i, j, k+1] = FT(acc)
            end
        end
    else
        # Fallback: simple accumulation (no B-correction)
        @inbounds for j in 1:Ny, i in 1:Nx
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h
                cm[i, j, k+1] = FT(acc)
            end
        end
    end
end

"""
Summarize the discrete mass-balance residual for one met window:

  residual = (m1 - m0) - scale * (div_h + div_v)

`am`, `bm`, and `cm` are stored as mass transported over one spectral half-step,
so a full met window corresponds to `scale = 2 * steps_per_met`.
"""
function summarize_mass_balance_residual(m0::Array{FT,3}, m1::Array{FT,3},
                                         am::Array{FT,3}, bm::Array{FT,3},
                                         cm::Array{FT,3}, grid;
                                         scale::Real,
                                         hour_start::Int,
                                         hour_end::Union{Nothing,Int}=nothing) where FT
    Nx, Ny, Nz = size(m0)
    scale_f = Float64(scale)
    n_cells = Nx * Ny * Nz
    eps64 = eps(Float64)

    res_sum = 0.0
    res_abs_sum = 0.0
    res_sq_sum = 0.0
    res_absmax = 0.0
    res_absmax_i = 1
    res_absmax_j = 1
    res_absmax_k = 1

    rel_sq_sum = 0.0
    rel_absmax = 0.0

    dm_sum = 0.0
    dm_abs_sum = 0.0
    dm_absmax = 0.0

    pred_sum = 0.0
    pred_abs_sum = 0.0
    pred_absmax = 0.0

    col_sq_sum = 0.0
    col_absmax = 0.0
    col_absmax_i = 1
    col_absmax_j = 1

    level_absmax = zeros(Float64, Nz)
    level_rms = zeros(Float64, Nz)
    level_mean_abs = zeros(Float64, Nz)
    level_rel_rms = zeros(Float64, Nz)

    @inbounds for j in 1:Ny, i in 1:Nx
        col_res = 0.0
        for k in 1:Nz
            div_half = (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
                       (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k])) +
                       (Float64(cm[i, j, k]) - Float64(cm[i, j, k + 1]))
            predicted = scale_f * div_half
            dm = Float64(m1[i, j, k]) - Float64(m0[i, j, k])
            residual = dm - predicted
            abs_res = abs(residual)

            res_sum += residual
            res_abs_sum += abs_res
            res_sq_sum += residual * residual
            if abs_res > res_absmax
                res_absmax = abs_res
                res_absmax_i = i
                res_absmax_j = j
                res_absmax_k = k
            end

            mass_scale = max(abs(Float64(m0[i, j, k])), abs(Float64(m1[i, j, k])), eps64)
            rel_res = residual / mass_scale
            rel_sq_sum += rel_res * rel_res
            rel_absmax = max(rel_absmax, abs(rel_res))

            abs_dm = abs(dm)
            dm_sum += dm
            dm_abs_sum += abs_dm
            dm_absmax = max(dm_absmax, abs_dm)

            abs_pred = abs(predicted)
            pred_sum += predicted
            pred_abs_sum += abs_pred
            pred_absmax = max(pred_absmax, abs_pred)

            level_absmax[k] = max(level_absmax[k], abs_res)
            level_rms[k] += residual * residual
            level_mean_abs[k] += abs_res
            level_rel_rms[k] += rel_res * rel_res

            col_res += residual
        end

        abs_col_res = abs(col_res)
        col_sq_sum += col_res * col_res
        if abs_col_res > col_absmax
            col_absmax = abs_col_res
            col_absmax_i = i
            col_absmax_j = j
        end
    end

    inv_n_cells = 1.0 / n_cells
    inv_n_cols = 1.0 / (Nx * Ny)
    inv_n_xy = 1.0 / (Nx * Ny)
    @inbounds for k in 1:Nz
        level_rms[k] = sqrt(level_rms[k] * inv_n_xy)
        level_mean_abs[k] *= inv_n_xy
        level_rel_rms[k] = sqrt(level_rel_rms[k] * inv_n_xy)
    end

    return Dict{String,Any}(
        "hour_start" => hour_start,
        "hour_end" => hour_end,
        "scale_factor" => scale_f,
        "residual_absmax_kg" => res_absmax,
        "residual_rms_kg" => sqrt(res_sq_sum * inv_n_cells),
        "residual_mean_abs_kg" => res_abs_sum * inv_n_cells,
        "residual_mean_kg" => res_sum * inv_n_cells,
        "residual_total_kg" => res_sum,
        "residual_rel_absmax" => rel_absmax,
        "residual_rel_rms" => sqrt(rel_sq_sum * inv_n_cells),
        "dm_absmax_kg" => dm_absmax,
        "dm_mean_abs_kg" => dm_abs_sum * inv_n_cells,
        "dm_total_kg" => dm_sum,
        "predicted_absmax_kg" => pred_absmax,
        "predicted_mean_abs_kg" => pred_abs_sum * inv_n_cells,
        "predicted_total_kg" => pred_sum,
        "column_residual_absmax_kg" => col_absmax,
        "column_residual_rms_kg" => sqrt(col_sq_sum * inv_n_cols),
        "hotspot" => Dict(
            "lon_deg" => grid.lons[res_absmax_i],
            "lat_deg" => grid.lats[res_absmax_j],
            "level" => res_absmax_k,
            "column_lon_deg" => grid.lons[col_absmax_i],
            "column_lat_deg" => grid.lats[col_absmax_j],
        ),
        "level_residual_absmax_kg" => level_absmax,
        "level_residual_rms_kg" => level_rms,
        "level_residual_mean_abs_kg" => level_mean_abs,
        "level_residual_rel_rms" => level_rel_rms,
    )
end

# ===========================================================================
# Target regular lat-lon grid (TM5 convention: poles at cell faces)
# ===========================================================================
struct TargetGrid
    Nlon::Int
    Nlat::Int
    lons::Vector{Float64}   # cell centers [degrees]
    lats::Vector{Float64}   # cell centers [degrees]
    dlon::Float64            # longitude spacing [radians]
    dlat::Float64            # latitude spacing [radians]
    cos_lat::Vector{Float64} # cos(lat) at cell centers
    area::Matrix{Float64}   # cell area [m^2] (Nlon x Nlat)
end

function TargetGrid(Nlon, Nlat)
    dlon_deg = 360.0 / Nlon
    # TM5 convention (grid_type_ll.F90:247-278): cell centers OFFSET from poles.
    # Poles (-90, +90) are at cell FACES, not centers.
    # Cell spacing: dlat = 180/Nlat (NOT 180/(Nlat-1)).
    # First center at -90 + dlat/2, last at +90 - dlat/2.
    # Pole cap cells (j=1, j=Nlat) span half the normal dphi -- their outer face
    # is clamped to +/-90 (blat[0]=-90, blat[Nlat]=+90).
    dlat_deg = 180.0 / Nlat
    south_deg = -90.0 + dlat_deg / 2
    lons = [dlon_deg * (i - 0.5) for i in 1:Nlon]
    lats = [south_deg + dlat_deg * (j - 1) for j in 1:Nlat]
    dlon = deg2rad(dlon_deg)
    dlat = deg2rad(dlat_deg)
    cos_lat = [cosd(lat) for lat in lats]

    # Cell faces (Nlat+1 boundaries, clamped to +/-90 at poles like TM5)
    blat_deg = [south_deg + dlat_deg * (j - 1) - dlat_deg / 2 for j in 1:Nlat+1]
    blat_deg[1]   = max(blat_deg[1],   -90.0)
    blat_deg[end]  = min(blat_deg[end],  90.0)

    # Cell areas: dlambda * R^2 * |sin(phi_top) - sin(phi_bot)| using actual faces
    area = zeros(Nlon, Nlat)
    for j in 1:Nlat
        cell_a = R_EARTH^2 * dlon * abs(sind(blat_deg[j+1]) - sind(blat_deg[j]))
        for i in 1:Nlon
            area[i, j] = cell_a
        end
    end
    return TargetGrid(Nlon, Nlat, lons, lats, dlon, dlat, cos_lat, area)
end

# ===========================================================================
# A/B hybrid coefficient loader
# ===========================================================================
function load_ab_coefficients(coeff_path::String, level_range)
    isfile(coeff_path) || error("Coefficients not found: $coeff_path")
    cfg = TOML.parsefile(coeff_path)
    a_all = Float64.(cfg["coefficients"]["a"])  # 138 values, n=0..137
    b_all = Float64.(cfg["coefficients"]["b"])
    Nz = length(level_range)
    # Extract interfaces for selected level range
    # Level k uses interfaces k and k+1 (1-based, where interface 1 = TOA)
    i_start = level_range[1]        # interface above first layer
    i_end   = level_range[end] + 1  # interface below last layer
    a_ifc = a_all[i_start:i_end]    # Nz+1 values
    b_ifc = b_all[i_start:i_end]
    # dA and dB for each layer: dp = dA + dB * ps
    dA = diff(a_ifc)  # Nz values
    dB = diff(b_ifc)
    # A/B at cell centers (for driver metadata)
    A_center = [(a_ifc[k] + a_ifc[k+1]) / 2 for k in 1:Nz]
    B_center = [(b_ifc[k] + b_ifc[k+1]) / 2 for k in 1:Nz]
    return (; a_ifc, b_ifc, dA, dB, A_center, B_center)
end

# ===========================================================================
# Fully normalized associated Legendre functions (ECMWF convention)
#
# P_n^m(x) such that integral_{-1}^{1} [P_n^m(x)]^2 dx = 2 / (2n+1) * (2n+1) = 2
# Uses stable three-term recurrence to avoid numerical overflow.
# ===========================================================================
function compute_legendre_column!(P::Matrix{Float64}, T::Int, sin_lat::Float64)
    cos_lat = sqrt(1.0 - sin_lat^2)

    # Starting value: P_0^0 = 1
    fill!(P, 0.0)
    P[1, 1] = 1.0

    # Sectoral (diagonal): P_m^m from P_{m-1}^{m-1}
    for m in 1:T
        P[m+1, m+1] = sqrt((2m + 1.0) / (2m)) * cos_lat * P[m, m]
    end

    # First off-diagonal: P_{m+1}^m
    for m in 0:(T-1)
        P[m+2, m+1] = sqrt(2m + 3.0) * sin_lat * P[m+1, m+1]
    end

    # General recurrence: P_n^m for n > m+1
    for m in 0:T
        for n in (m+2):T
            n2 = n * n
            m2 = m * m
            a = sqrt((4.0 * n2 - 1.0) / (n2 - m2))
            b = sqrt(((2n + 1.0) * (n - m - 1.0) * (n + m - 1.0)) /
                     ((2n - 3.0) * (n2 - m2)))
            P[n+1, m+1] = a * sin_lat * P[n, m+1] - b * P[n-1, m+1]
        end
    end
    return nothing
end

# ===========================================================================
# Read spectral coefficients from GRIB
# ===========================================================================
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
            spec[n+1, m+1] = complex(vals[idx], vals[idx+1])
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
    # First pass: determine T from LNSP
    f = GribFile(lnsp_path)
    msg1 = first(f)
    T_file = msg1["J"]
    destroy(f)
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    # Read ALL LNSP (small: n_times * (T+1)^2)
    f = GribFile(lnsp_path)
    lnsp_all = Dict{Int, Matrix{ComplexF64}}()
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    for msg in f
        hour = div(msg["dataTime"], 100)
        read_spectral_coeffs!(spec_buf, msg)
        lnsp_all[hour] = copy(@view spec_buf[1:T+1, 1:T+1])
    end
    destroy(f)

    # Read VO/D, grouped by hour
    vo_by_hour = Dict{Int, Array{ComplexF64,3}}()
    d_by_hour  = Dict{Int, Array{ComplexF64,3}}()
    f = GribFile(vo_d_path)
    for msg in f
        name  = msg["shortName"]
        level = msg["level"]
        hour  = div(msg["dataTime"], 100)
        read_spectral_coeffs!(spec_buf, msg)
        if name == "vo"
            if !haskey(vo_by_hour, hour)
                vo_by_hour[hour] = zeros(ComplexF64, T+1, T+1, Nlevels)
            end
            vo_by_hour[hour][:, :, level] .= @view spec_buf[1:T+1, 1:T+1]
        elseif name == "d"
            if !haskey(d_by_hour, hour)
                d_by_hour[hour] = zeros(ComplexF64, T+1, T+1, Nlevels)
            end
            d_by_hour[hour][:, :, level] .= @view spec_buf[1:T+1, 1:T+1]
        end
    end
    destroy(f)

    hours = sort(collect(keys(lnsp_all)))
    return (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times=length(hours))
end

# ===========================================================================
# VO/D -> U/V spectral conversion (port of TM5 grid_type_sh.F90 sh_vod2uv)
#
# U(m,n) = R * [delta(m,n)*VO(m,n-1) + i*sigma(m,n)*D(m,n) - delta(m,n+1)*VO(m,n+1)]
# V(m,n) = R * [-delta(m,n)*D(m,n-1) + i*sigma(m,n)*VO(m,n) + delta(m,n+1)*D(m,n+1)]
# ===========================================================================
@inline function _delta(m::Int, n::Int)
    n == 0 && return 0.0
    n2 = Float64(n * n)
    m2 = Float64(m * m)
    return -sqrt((n2 - m2) / (4n2 - 1)) / n
end

@inline function _sigma(m::Int, n::Int)
    (n == 0 || m == 0) && return 0.0
    return -Float64(m) / (Float64(n) * (n + 1))
end

"""
Convert spectral vorticity/divergence to spectral U/V winds.
Operates on a single level: vo_spec[n+1, m+1] -> u_spec, v_spec.
"""
function vod2uv!(u_spec::Matrix{ComplexF64}, v_spec::Matrix{ComplexF64},
                 vo_spec::AbstractMatrix{ComplexF64}, d_spec::AbstractMatrix{ComplexF64},
                 T::Int)
    fill!(u_spec, zero(ComplexF64))
    fill!(v_spec, zero(ComplexF64))

    for m in 0:T
        for n in m:T
            delta_mn  = _delta(m, n)
            delta_mn1 = _delta(m, n + 1)
            sigma_mn  = _sigma(m, n)

            # VO/D at neighboring n values (zero at boundaries)
            vo_nm1 = n > m     ? vo_spec[n, m+1]   : zero(ComplexF64)  # VO(m, n-1)
            vo_np1 = n < T     ? vo_spec[n+2, m+1] : zero(ComplexF64)  # VO(m, n+1)
            d_nm1  = n > m     ? d_spec[n, m+1]    : zero(ComplexF64)  # D(m, n-1)
            d_np1  = n < T     ? d_spec[n+2, m+1]  : zero(ComplexF64)  # D(m, n+1)
            vo_n   = vo_spec[n+1, m+1]
            d_n    = d_spec[n+1, m+1]

            u_spec[n+1, m+1] = R_EARTH * (delta_mn * vo_nm1 + im * sigma_mn * d_n - delta_mn1 * vo_np1)
            v_spec[n+1, m+1] = R_EARTH * (-delta_mn * d_nm1 + im * sigma_mn * vo_n + delta_mn1 * d_np1)
        end
    end
    return nothing
end

# ===========================================================================
# Inverse SHT: spectral -> regular lat-lon grid
# ===========================================================================
"""
Transform a single spectral field to a regular lat-lon grid.
- spec[n+1, m+1] = spectral coefficients (complex, m=0..T, n=m..T)
- T = spectral truncation
- lats = target latitudes [degrees]
- Nlon = number of target longitudes
Writes into field[Nlon, Nlat].
"""
function spectral_to_grid!(field::Matrix{Float64},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lats::Vector{Float64},
                           Nlon::Int,
                           P_buf::Matrix{Float64},
                           fft_buf::Vector{ComplexF64};
                           fft_out::Union{Nothing,Vector{ComplexF64}}=nothing,
                           bfft_plan=nothing)
    Nlat = length(lats)
    Nfft = Nlon

    for j in 1:Nlat
        sin_lat = sind(lats[j])

        # Compute Legendre functions P_n^m(sin_lat)
        compute_legendre_column!(P_buf, T, sin_lat)

        # Legendre sum: G_m = sum_n spec[n+1, m+1] * P[n+1, m+1]
        fill!(fft_buf, zero(ComplexF64))
        for m in 0:min(T, div(Nfft, 2))
            Gm = zero(ComplexF64)
            @inbounds for n in m:T
                Gm += spec[n+1, m+1] * P_buf[n+1, m+1]
            end
            fft_buf[m+1] = Gm
        end

        # Fill negative frequencies for real-valued field:
        # f(lambda) = sum_{m>=0} G_m exp(im*lambda) + sum_{m>=1} conj(G_m) exp(-im*lambda)
        # In DFT layout: negative freq m maps to index N-m+1
        for m in 1:min(T, div(Nfft, 2) - 1)
            fft_buf[Nfft - m + 1] = conj(fft_buf[m + 1])
        end

        # Fourier synthesis via backward FFT (unnormalized):
        # bfft(G)[k] = sum_m G[m] exp(2*pi*i*m*(k-1)/N) = f(lambda_k)
        # No 1/N scaling -- this is a direct evaluation, not an inverse DFT.
        f_lon = if bfft_plan !== nothing && fft_out !== nothing
            mul!(fft_out, bfft_plan, fft_buf)
            fft_out
        else
            bfft(fft_buf)
        end

        @inbounds for i in 1:Nlon
            field[i, j] = real(f_lon[i])
        end
    end
    return nothing
end

# ===========================================================================
# Mass flux computation on the target grid
#
# Following preprocess_mass_fluxes.jl conventions:
# - m  = air mass per cell [kg]
# - am = eastward mass flux at x-faces [kg per half-timestep]
# - bm = northward mass flux at y-faces [kg per half-timestep]
# - cm = vertical mass flux at z-interfaces [kg per half-timestep]
# ===========================================================================

"""
Stagger cell-center winds to faces (periodic in longitude, poles = 0).
"""
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        ip = i == Nlon ? 1 : i + 1
        u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u_stag[Nlon + 1, :, :] .= u_stag[1, :, :]

    @inbounds for k in 1:Nz, j in 2:Nlat, i in 1:Nlon
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0      # no meridional flux through South Pole
    v_stag[:, Nlat + 1, :] .= 0  # no meridional flux through North Pole
    return nothing
end

"""
Compute pressure thickness dp per cell from A/B coefficients and surface pressure.
dp[i,j,k] = |dA[k] + dB[k] * ps[i,j]|
"""
function compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        dp[i, j, k] = abs(dA[k] + dB[k] * ps[i, j])
    end
    return nothing
end

"""
Compute air mass per cell: m = dp * area / g
"""
function compute_air_mass!(m_arr, dp, area, Nlon, Nlat, Nz)
    inv_g = 1.0 / GRAV
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        m_arr[i, j, k] = dp[i, j, k] * area[i, j] * inv_g
    end
    return nothing
end

"""
Compute horizontal mass fluxes (staggered) and vertical mass flux from continuity.

am[i, j, k] = dp_stag * cos(lat) * dlat * R / g * u_stag * half_dt
bm[i, j, k] = dp_stag * dlon * R / g * v_stag * half_dt
cm from continuity: TM5 hybrid-coordinate formula with B-correction.
"""
function compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps,
                               dA, dB, grid::TargetGrid, half_dt, Nz)
    Nlon, Nlat = grid.Nlon, grid.Nlat
    R_g = R_EARTH / GRAV
    dlon = grid.dlon
    dlat = grid.dlat

    # --- Eastward mass flux at x-faces: am[Nlon+1, Nlat, Nz] ---
    # vod2uv returns U = u*cos(phi), not u. TM5 divides by cos(phi) in IntLat:
    #   mfu = R/g * integral U * dp / cos(phi) dphi = R/g * integral u * dp dphi
    # So: am = U/cos(phi) * dp/g * R * dphi * half_dt
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon+1)
        i_l = i == 1 ? Nlon : i - 1
        i_r = i <= Nlon ? i : 1
        dp_stag = (dp[i_l, j, k] + dp[i_r, j, k]) / 2
        cos_lat = max(grid.cos_lat[j], 1e-10)
        am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_stag * R_g * dlat * half_dt
    end

    # --- Northward mass flux at y-faces: bm[Nlon, Nlat+1, Nz] ---
    # vod2uv returns V = v*cos(phi). TM5 IntLon does NOT divide by cos:
    #   mfv = R/g * integral V * dp dlambda
    @inbounds for k in 1:Nz, j in 1:(Nlat+1), i in 1:Nlon
        if j == 1 || j == Nlat + 1
            bm[i, j, k] = 0  # No flux through poles
        else
            j_s = j - 1
            j_n = j
            dp_stag = (dp[i, j_s, k] + dp[i, j_n, k]) / 2
            bm[i, j, k] = v_stag[i, j, k] * dp_stag * R_g * dlon * half_dt
        end
    end

    # --- Vertical mass flux from hybrid-coordinate continuity (TM5 dynam0 formula) ---
    # cm[k+1] = cm[k] - div_h[k] + (B[k+1] - B[k]) * pit
    # Float64 accumulation prevents roundoff (residual drops from ~10^5 to ~0).
    fill!(cm, zero(eltype(cm)))
    @inbounds for j in 1:Nlat, i in 1:Nlon
        # Column-integrated horizontal convergence (Float64)
        pit = 0.0
        for k in 1:Nz
            pit += (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                   (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
        end
        # Build cm from TOA downward with B-correction (Float64 accumulation)
        acc = 0.0
        for k in 1:Nz
            div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                     (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
            acc = acc - div_h + Float64(dB[k]) * pit
            cm[i, j, k+1] = eltype(cm)(acc)
        end
    end

    return nothing
end

# ===========================================================================
# Binary write helper
# ===========================================================================

function write_array!(io::IO, arr::Array{FT}) where FT
    nb = sizeof(arr)
    GC.@preserve arr begin; written = unsafe_write(io, pointer(arr), nb); end
    written == nb || error("Short write: expected $nb bytes, got $written")
    return written
end

const HEADER_SIZE = 16384

# ===========================================================================
# Compute native-level fields from spectral data for a single hour
# ===========================================================================
"""
Transform spectral data for one hour into native-level gridpoint fields.
All computation in Float64; results stored in Float64 work arrays.
Returns ps (surface pressure, Float64 matrix).
"""
function spectral_to_native_fields!(
    m_arr::Array{Float64,3}, am::Array{Float64,3}, bm::Array{Float64,3},
    cm::Array{Float64,3}, sp::Matrix{Float64},
    u_cc::Array{Float64,3}, v_cc::Array{Float64,3},
    u_stag::Array{Float64,3}, v_stag::Array{Float64,3},
    dp::Array{Float64,3},
    lnsp_spec::Matrix{ComplexF64},
    vo_hour::Array{ComplexF64,3}, d_hour::Array{ComplexF64,3},
    T::Int, level_range::UnitRange{Int}, ab,
    grid::TargetGrid, half_dt::Float64,
    # Per-thread SHT buffers:
    P_buf::Matrix{Float64}, fft_buf::Vector{ComplexF64},
    field_2d::Matrix{Float64},
    P_buf_t, fft_buf_t, fft_out_t, u_spec_t, v_spec_t, field_2d_t,
    bfft_plans)

    Nlon = grid.Nlon
    Nlat = grid.Nlat
    Nz = length(level_range)

    # --- Transform LNSP -> SP on target grid ---
    spectral_to_grid!(field_2d, lnsp_spec, T, grid.lats, Nlon, P_buf, fft_buf)
    @. sp = exp(field_2d)

    # --- For each level: VO/D -> U/V spectral, then transform ---
    # Threaded over levels (each level is independent)
    Threads.@threads for k in 1:Nz
        level = level_range[k]
        tid = Threads.threadid()

        # VO/D -> U/V in spectral space (thread-local buffers)
        vod2uv!(u_spec_t[tid], v_spec_t[tid],
                @view(vo_hour[:, :, level]),
                @view(d_hour[:, :, level]),
                T)

        # U spectral -> gridpoint (thread-safe with pre-planned FFT)
        spectral_to_grid!(field_2d_t[tid], u_spec_t[tid], T,
                         grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                         fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid])
        u_cc[:, :, k] .= field_2d_t[tid]

        # V spectral -> gridpoint
        spectral_to_grid!(field_2d_t[tid], v_spec_t[tid], T,
                         grid.lats, Nlon, P_buf_t[tid], fft_buf_t[tid];
                         fft_out=fft_out_t[tid], bfft_plan=bfft_plans[tid])
        v_cc[:, :, k] .= field_2d_t[tid]
    end

    # --- Compute mass fluxes ---
    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nlon, Nlat, Nz)
    compute_dp!(dp, sp, ab.dA, ab.dB, Nlon, Nlat, Nz)
    compute_air_mass!(m_arr, dp, grid.area, Nlon, Nlat, Nz)
    compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, sp,
                         ab.dA, ab.dB, grid, half_dt, Nz)

    return nothing
end

# ===========================================================================
# Process one day: spectral GRIB -> merged v4 binary
# ===========================================================================
function process_day(date::Date, grid::TargetGrid, ab, level_range,
                     vc_native, merged_vc, merge_map,
                     spectral_dir::String, out_dir::String,
                     half_dt::Float64, dt::Float64, met_interval::Float64,
                     T_target::Int, min_dp::Float64;
                     next_day_hour0=nothing)
    FT = Float32
    Nz_native = n_levels(vc_native)
    Nz = n_levels(merged_vc)
    Nx = grid.Nlon
    Ny = grid.Nlat

    steps_per_met = max(1, round(Int, met_interval / dt))
    date_str = Dates.format(date, "yyyymmdd")

    # --- Read spectral GRIB data ---
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    hours = spec.hours

    # --- Element counts ---
    n_m  = Int64(Nx) * Ny * Nz
    n_am = Int64(Nx + 1) * Ny * Nz
    n_bm = Int64(Nx) * (Ny + 1) * Nz
    n_cm = Int64(Nx) * Ny * (Nz + 1)
    n_ps = Int64(Nx) * Ny
    n_dam = n_am
    n_dbm = n_bm
    n_dm  = n_m
    n_dcm = n_cm

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps + n_dam + n_dbm + n_dm + n_dcm
    bytes_per_window = elems_per_window * sizeof(FT)
    total_bytes = Int64(HEADER_SIZE) + bytes_per_window * Nt

    # --- Output path ---
    mkpath(out_dir)
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    bin_path = joinpath(out_dir, "era5_v4_$(date_str)_$(dp_tag)_float32.bin")
    diag_path = joinpath(out_dir, "era5_v4_$(date_str)_$(dp_tag)_float32_mass_balance.json")
    bin_ready = isfile(bin_path) && filesize(bin_path) == total_bytes
    diag_ready = isfile(diag_path)

    if bin_ready && diag_ready
        @info "  SKIP (exists, correct size): $(basename(bin_path))"
        return bin_path
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), total_bytes / 1e9, Nt)
    if bin_ready && !diag_ready
        @info "  Binary exists but mass-balance sidecar is missing; recomputing diagnostics"
    end

    # --- Build header ---
    header = Dict{String,Any}(
        "magic" => "MFLX", "version" => 4, "header_bytes" => HEADER_SIZE,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nz_native" => Nz_native, "Nt" => Nt,
        "float_type" => "Float32", "float_bytes" => sizeof(FT),
        "window_bytes" => bytes_per_window,
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => 0, "n_cmfmc" => 0,
        "n_entu" => 0, "n_detu" => 0, "n_entd" => 0, "n_detd" => 0,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "n_dam" => n_dam, "n_dbm" => n_dbm, "n_dm" => n_dm, "n_dcm" => n_dcm,
        "include_flux_delta" => true,
        "include_qv" => false, "include_cmfmc" => false,
        "include_tm5conv" => false, "include_surface" => false,
        "include_temperature" => false,
        "dt_seconds" => dt, "half_dt_seconds" => half_dt,
        "steps_per_met_window" => steps_per_met,
        "level_top" => level_range[1], "level_bot" => level_range[end],
        "lons" => Float64.(grid.lons), "lats" => Float64.(grid.lats),
        "A_ifc" => Float64.(merged_vc.A), "B_ifc" => Float64.(merged_vc.B),
        "merge_map" => merge_map, "merge_min_thickness_Pa" => min_dp,
        "var_names" => ["m","am","bm","cm","ps","dam","dbm","dm"],
        "date" => Dates.format(date, "yyyy-mm-dd"),
        "grid_convention" => "TM5",
        "spectral_half_dt_seconds" => half_dt,
    )
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large: $(length(header_json)) >= $(HEADER_SIZE)")

    # --- Allocate F64 work arrays for spectral computation ---
    u_cc    = Array{Float64}(undef, Nx, Ny, Nz_native)
    v_cc    = Array{Float64}(undef, Nx, Ny, Nz_native)
    u_stag  = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    v_stag  = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    dp      = Array{Float64}(undef, Nx, Ny, Nz_native)
    m_arr   = Array{Float64}(undef, Nx, Ny, Nz_native)
    am_arr  = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    bm_arr  = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    cm_arr  = Array{Float64}(undef, Nx, Ny, Nz_native + 1)
    sp      = Array{Float64}(undef, Nx, Ny)

    # SHT work buffers
    nt = Threads.nthreads()
    nt_max = max(nt, 2 * Threads.nthreads()) + 4
    P_buf    = zeros(Float64, spec.T + 1, spec.T + 1)
    fft_buf  = zeros(ComplexF64, Nx)
    field_2d = Array{Float64}(undef, Nx, Ny)
    P_buf_t    = [zeros(Float64, spec.T + 1, spec.T + 1) for _ in 1:nt_max]
    fft_buf_t  = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    fft_out_t  = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    u_spec_t   = [zeros(ComplexF64, spec.T + 1, spec.T + 1) for _ in 1:nt_max]
    v_spec_t   = [zeros(ComplexF64, spec.T + 1, spec.T + 1) for _ in 1:nt_max]
    field_2d_t = [Array{Float64}(undef, Nx, Ny) for _ in 1:nt_max]
    bfft_plans = [plan_bfft(fft_buf_t[i]) for i in 1:nt_max]

    # Merged-level work buffers (Float32) — reused per window for merging
    m_merged  = Array{FT}(undef, Nx, Ny, Nz)
    am_merged = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_merged = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_merged = Array{FT}(undef, Nx, Ny, Nz + 1)

    # Native-level Float32 temporaries for merging
    m_native  = Array{FT}(undef, Nx, Ny, Nz_native)
    am_native = Array{FT}(undef, Nx + 1, Ny, Nz_native)
    bm_native = Array{FT}(undef, Nx, Ny + 1, Nz_native)

    # Delta buffers
    dam_merged = Array{FT}(undef, Nx + 1, Ny, Nz)
    dbm_merged = Array{FT}(undef, Nx, Ny + 1, Nz)
    dm_merged  = Array{FT}(undef, Nx, Ny, Nz)
    dcm_merged = Array{FT}(undef, Nx, Ny, Nz + 1)

    # Next-window merged fields for last-hour delta via next day's hour 0
    m_next_merged  = Array{FT}(undef, Nx, Ny, Nz)
    am_next_merged = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_next_merged = Array{FT}(undef, Nx, Ny + 1, Nz)

    # Pre-compute all windows' merged fields so we can do forward deltas
    # Store merged m, am, bm for each window (ps is 2D, no delta needed)
    all_m  = Vector{Array{FT,3}}(undef, Nt)
    all_am = Vector{Array{FT,3}}(undef, Nt)
    all_bm = Vector{Array{FT,3}}(undef, Nt)
    all_cm = Vector{Array{FT,3}}(undef, Nt)
    all_ps = Vector{Array{FT,2}}(undef, Nt)

    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    for (win_idx, hour) in enumerate(hours)
        t0 = time()

        # Get spectral data for this hour
        lnsp_h = spec.lnsp_all[hour]
        vo_h   = spec.vo_by_hour[hour]
        d_h    = spec.d_by_hour[hour]

        # Spectral -> native gridpoint fields (all in Float64)
        spectral_to_native_fields!(
            m_arr, am_arr, bm_arr, cm_arr, sp,
            u_cc, v_cc, u_stag, v_stag, dp,
            lnsp_h, vo_h, d_h,
            spec.T, level_range, ab, grid, half_dt,
            P_buf, fft_buf, field_2d,
            P_buf_t, fft_buf_t, fft_out_t, u_spec_t, v_spec_t, field_2d_t,
            bfft_plans)

        # Cast to Float32 for merging
        @. m_native  = FT(m_arr)
        @. am_native = FT(am_arr)
        @. bm_native = FT(bm_arr)

        # Merge to target levels
        merge_cell_field!(m_merged, m_native, merge_map)
        merge_cell_field!(am_merged, am_native, merge_map)
        merge_cell_field!(bm_merged, bm_native, merge_map)

        # TM5 convention: zero bm at pole FACES only
        @views bm_merged[:, 1, :]    .= zero(FT)
        @views bm_merged[:, Ny+1, :] .= zero(FT)

        # Recompute cm from merged horizontal divergence (TM5 B-correction)
        recompute_cm_from_divergence!(cm_merged, am_merged, bm_merged, m_merged;
                                      B_ifc=merged_vc.B)

        # Enforce cm boundaries
        @views cm_merged[:, :, 1]      .= zero(FT)  # TOA
        @views cm_merged[:, :, Nz + 1] .= zero(FT)  # surface

        # Store for binary write + delta computation
        all_m[win_idx]  = copy(m_merged)
        all_am[win_idx] = copy(am_merged)
        all_bm[win_idx] = copy(bm_merged)
        all_cm[win_idx] = copy(cm_merged)
        all_ps[win_idx] = FT.(sp)

        elapsed = round(time() - t0, digits=2)
        if win_idx <= 3 || win_idx == Nt || win_idx % 8 == 0
            @info @sprintf("    Window %d/%d (hour %02d): %.2fs", win_idx, Nt, hour, elapsed)
        end
    end

    # --- Compute next-window merged fields for last hour's delta ---
    # If next_day_hour0 is provided, transform it to get the fields for delta
    last_hour_next_m  = nothing
    last_hour_next_am = nothing
    last_hour_next_bm = nothing
    last_hour_next_cm = nothing

    if next_day_hour0 !== nothing
        @info "  Computing next day hour 0 for last-window delta..."
        spectral_to_native_fields!(
            m_arr, am_arr, bm_arr, cm_arr, sp,
            u_cc, v_cc, u_stag, v_stag, dp,
            next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
            next_day_hour0.T, level_range, ab, grid, half_dt,
            P_buf, fft_buf, field_2d,
            P_buf_t, fft_buf_t, fft_out_t, u_spec_t, v_spec_t, field_2d_t,
            bfft_plans)
        @. m_native  = FT(m_arr)
        @. am_native = FT(am_arr)
        @. bm_native = FT(bm_arr)
        merge_cell_field!(m_next_merged, m_native, merge_map)
        merge_cell_field!(am_next_merged, am_native, merge_map)
        merge_cell_field!(bm_next_merged, bm_native, merge_map)
        @views bm_next_merged[:, 1, :]    .= zero(FT)
        @views bm_next_merged[:, Ny+1, :] .= zero(FT)
        # Recompute cm for the next day's merged fields
        recompute_cm_from_divergence!(cm_merged, am_next_merged, bm_next_merged, m_next_merged;
                                      B_ifc=merged_vc.B)
        cm_merged[1, :, 1]   .= zero(FT)
        cm_merged[:, :, end] .= zero(FT)
        last_hour_next_m  = copy(m_next_merged)
        last_hour_next_am = copy(am_next_merged)
        last_hour_next_bm = copy(bm_next_merged)
        last_hour_next_cm = copy(cm_merged)
    end

    # --- Segers-style mass-balance residual diagnostics ---
    @info "  Computing mass-balance residual diagnostics..."
    window_scale = 2 * steps_per_met
    mass_balance_windows = Vector{Dict{String,Any}}(undef, Nt)
    current_peak_absmax = -Inf
    current_peak_window = 0
    midpoint_peak_absmax = -Inf
    midpoint_peak_window = 0

    for win_idx in 1:Nt
        next_m = nothing
        next_am = nothing
        next_bm = nothing
        hour_end = nothing
        uses_next_day_hour0 = false

        if win_idx < Nt
            next_m = all_m[win_idx + 1]
            next_am = all_am[win_idx + 1]
            next_bm = all_bm[win_idx + 1]
            hour_end = hours[win_idx + 1]
        elseif last_hour_next_m !== nothing
            next_m = last_hour_next_m
            next_am = last_hour_next_am
            next_bm = last_hour_next_bm
            hour_end = 0
            uses_next_day_hour0 = true
        end

        if next_m === nothing
            mass_balance_windows[win_idx] = Dict{String,Any}(
                "window_index" => win_idx,
                "hour_start" => hours[win_idx],
                "hour_end" => hour_end,
                "available" => false,
                "reason" => "missing_next_window_state",
            )
            continue
        end

        current_summary = summarize_mass_balance_residual(
            all_m[win_idx], next_m, all_am[win_idx], all_bm[win_idx], all_cm[win_idx], grid;
            scale=window_scale, hour_start=hours[win_idx], hour_end=hour_end)

        dam_merged .= all_am[win_idx]
        dam_merged .+= next_am
        dam_merged .*= FT(0.5)
        dbm_merged .= all_bm[win_idx]
        dbm_merged .+= next_bm
        dbm_merged .*= FT(0.5)
        @views dam_merged[:, 1, :]    .= zero(FT)
        @views dam_merged[:, Ny, :]   .= zero(FT)
        @views dbm_merged[:, 1, :]    .= zero(FT)
        @views dbm_merged[:, Ny+1, :] .= zero(FT)
        recompute_cm_from_divergence!(cm_merged, dam_merged, dbm_merged, all_m[win_idx];
                                      B_ifc=merged_vc.B)
        @views cm_merged[:, :, 1]      .= zero(FT)
        @views cm_merged[:, :, Nz + 1] .= zero(FT)

        midpoint_summary = summarize_mass_balance_residual(
            all_m[win_idx], next_m, dam_merged, dbm_merged, cm_merged, grid;
            scale=window_scale, hour_start=hours[win_idx], hour_end=hour_end)

        current_absmax = Float64(current_summary["residual_absmax_kg"])
        if current_absmax > current_peak_absmax
            current_peak_absmax = current_absmax
            current_peak_window = win_idx
        end

        midpoint_absmax = Float64(midpoint_summary["residual_absmax_kg"])
        if midpoint_absmax > midpoint_peak_absmax
            midpoint_peak_absmax = midpoint_absmax
            midpoint_peak_window = win_idx
        end

        mass_balance_windows[win_idx] = Dict{String,Any}(
            "window_index" => win_idx,
            "hour_start" => hours[win_idx],
            "hour_end" => hour_end,
            "available" => true,
            "uses_next_day_hour0" => uses_next_day_hour0,
            "current_flux" => current_summary,
            "midpoint_flux" => midpoint_summary,
        )

        if win_idx <= 3 || win_idx == Nt || win_idx % 8 == 0
            @info @sprintf(
                "    Residual window %d/%d hour %02d->%02d endpoint_absmax=%.3e midpoint_absmax=%.3e",
                win_idx, Nt, hours[win_idx], hour_end === nothing ? -1 : hour_end,
                current_absmax, midpoint_absmax)
        end
    end

    diag_payload = Dict{String,Any}(
        "date" => Dates.format(date, "yyyy-mm-dd"),
        "bin_path" => bin_path,
        "window_scale_factor" => window_scale,
        "steps_per_met" => steps_per_met,
        "dt_seconds" => dt,
        "half_dt_seconds" => half_dt,
        "met_interval_seconds" => met_interval,
        "diagnostic_definition" => "residual = (m_next - m_current) - scale*(am_left-am_right + bm_south-bm_north + cm_top-cm_bottom)",
        "flux_units" => "kg per spectral half-step",
        "summary" => Dict(
            "current_flux_peak_absmax_kg" => current_peak_absmax,
            "current_flux_peak_window" => current_peak_window,
            "midpoint_flux_peak_absmax_kg" => midpoint_peak_absmax,
            "midpoint_flux_peak_window" => midpoint_peak_window,
        ),
        "windows" => mass_balance_windows,
    )
    open(diag_path, "w") do io
        write(io, JSON3.write(diag_payload))
    end
    @info @sprintf(
        "  Wrote mass-balance diagnostics: %s (peak midpoint residual %.3e kg in window %d)",
        basename(diag_path), midpoint_peak_absmax, midpoint_peak_window)

    # --- Write binary ---
    if !bin_ready
        @info "  Writing binary..."
        bytes_written = Int64(0)
        open(bin_path, "w") do io
            hdr_buf = zeros(UInt8, HEADER_SIZE)
            copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
            write(io, hdr_buf)
            bytes_written += HEADER_SIZE

            for win_idx in 1:Nt
                # Write core fields: m | am | bm | cm | ps
                bytes_written += write_array!(io, all_m[win_idx])
                bytes_written += write_array!(io, all_am[win_idx])
                bytes_written += write_array!(io, all_bm[win_idx])
                bytes_written += write_array!(io, all_cm[win_idx])
                bytes_written += write_array!(io, all_ps[win_idx])

                # Compute deltas: dam = next - current, dbm = next - current, dm = next - current
                if win_idx < Nt
                    # Next window is within this day
                    dam_merged .= all_am[win_idx + 1] .- all_am[win_idx]
                    dbm_merged .= all_bm[win_idx + 1] .- all_bm[win_idx]
                    dm_merged  .= all_m[win_idx + 1]  .- all_m[win_idx]
                elseif last_hour_next_m !== nothing
                    # Last window: use next day's hour 0
                    dam_merged .= last_hour_next_am .- all_am[win_idx]
                    dbm_merged .= last_hour_next_bm .- all_bm[win_idx]
                    dm_merged  .= last_hour_next_m  .- all_m[win_idx]
                else
                    # No next data available -> zero deltas
                    fill!(dam_merged, zero(FT))
                    fill!(dbm_merged, zero(FT))
                    fill!(dm_merged, zero(FT))
                end

                # Compute dcm = cm_next - cm_curr
                if win_idx < Nt
                    dcm_merged .= all_cm[win_idx + 1] .- all_cm[win_idx]
                elseif last_hour_next_cm !== nothing
                    dcm_merged .= last_hour_next_cm .- all_cm[win_idx]
                else
                    fill!(dcm_merged, zero(FT))
                end

                bytes_written += write_array!(io, dam_merged)
                bytes_written += write_array!(io, dbm_merged)
                bytes_written += write_array!(io, dm_merged)
                bytes_written += write_array!(io, dcm_merged)
            end
            flush(io)
        end

        actual = filesize(bin_path)
        @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path), actual / 1e9, time() - t_day)
        actual == total_bytes ||
            error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d", total_bytes, actual))
    else
        @info @sprintf("  Done diagnostics-only: %s (%.1fs)", basename(diag_path), time() - t_day)
    end

    # Return the last window's merged fields for the next day's prev_day_last_hour
    last_merged = (m=all_m[Nt], am=all_am[Nt], bm=all_bm[Nt])
    return bin_path, last_merged
end

# ===========================================================================
# Read hour 0 spectral data from a GRIB file (for cross-day delta)
# ===========================================================================
function read_hour0_spectral(spectral_dir::String, date::Date; T_target::Int=0)
    date_str = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    (!isfile(vo_d_path) || !isfile(lnsp_path)) && return nothing

    # Determine T from LNSP
    f = GribFile(lnsp_path)
    msg1 = first(f)
    T_file = msg1["J"]
    destroy(f)
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137

    # Read hour 0 LNSP
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    lnsp_h0 = nothing
    f = GribFile(lnsp_path)
    for msg in f
        hour = div(msg["dataTime"], 100)
        if hour == 0
            read_spectral_coeffs!(spec_buf, msg)
            lnsp_h0 = copy(@view spec_buf[1:T+1, 1:T+1])
            break
        end
    end
    destroy(f)
    lnsp_h0 === nothing && return nothing

    # Read hour 0 VO/D
    vo_h0 = zeros(ComplexF64, T+1, T+1, Nlevels)
    d_h0  = zeros(ComplexF64, T+1, T+1, Nlevels)
    f = GribFile(vo_d_path)
    for msg in f
        hour = div(msg["dataTime"], 100)
        hour == 0 || continue
        name  = msg["shortName"]
        level = msg["level"]
        read_spectral_coeffs!(spec_buf, msg)
        if name == "vo"
            vo_h0[:, :, level] .= @view spec_buf[1:T+1, 1:T+1]
        elseif name == "d"
            d_h0[:, :, level] .= @view spec_buf[1:T+1, 1:T+1]
        end
    end
    destroy(f)

    return (lnsp=lnsp_h0, vo=vo_h0, d=d_h0, T=T)
end

# ===========================================================================
# Main
# ===========================================================================
function main()
    if isempty(ARGS)
        println("""
        Fused ERA5 spectral GRIB -> v4 binary preprocessor

        Goes directly from spectral GRIB to merged-level daily binary,
        skipping the intermediate NetCDF step.

        Usage:
          julia -t8 --project=. $(PROGRAM_FILE) config.toml [--day 2021-12-01]

        Produces ONLY core transport fields (m, am, bm, cm, ps) plus v4
        flux deltas (dam, dbm, dm).
        """)
        return
    end

    config_path = expanduser(ARGS[1])
    isfile(config_path) || error("Config not found: $config_path")
    cfg = TOML.parsefile(config_path)

    # Optional --day filter
    day_filter = nothing
    for i in 1:length(ARGS)-1
        ARGS[i] == "--day" && (day_filter = Date(ARGS[i+1]))
    end

    # --- Config ---
    spectral_dir = expanduser(cfg["input"]["spectral_dir"])
    coeff_path   = expanduser(cfg["input"]["coefficients"])
    out_dir      = expanduser(cfg["output"]["directory"])

    Nlon      = cfg["grid"]["nlon"]
    Nlat      = cfg["grid"]["nlat"]
    level_top = Int(get(cfg["grid"], "level_top", 1))
    level_bot = Int(get(cfg["grid"], "level_bot", 137))
    min_dp    = Float64(cfg["grid"]["merge_min_thickness_Pa"])

    dt           = Float64(cfg["numerics"]["dt"])
    met_interval = Float64(cfg["numerics"]["met_interval"])
    half_dt      = dt / 2.0

    level_range = level_top:level_bot
    Nz_native = length(level_range)

    # --- Grid and coefficients ---
    grid = TargetGrid(Nlon, Nlat)
    ab = load_ab_coefficients(coeff_path, level_range)

    # Maximum spectral truncation for target grid
    T_target = div(Nlon, 2) - 1  # Nyquist: 720/2 - 1 = 359

    # --- Vertical merging ---
    vc_native = load_era5_vertical_coordinate(coeff_path, level_top, level_bot)
    merged_vc, merge_map = merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
    Nz_merged = n_levels(merged_vc)

    @info """
    Fused Spectral -> v4 Binary Preprocessor
    ==========================================
    Spectral dir:  $spectral_dir
    Output dir:    $out_dir
    Target grid:   $(Nlon) x $(Nlat) ($(360.0/Nlon) deg x $(180.0/Nlat) deg, TM5 convention)
    Native levels: $(Nz_native) ($(level_top)-$(level_bot))
    Merged levels: $(Nz_merged) (min_dp=$(min_dp) Pa)
    DT:            $(dt) s (half_dt=$(half_dt) s)
    Met interval:  $(met_interval) s
    T_target:      $(T_target) (Nyquist for Nlon=$(Nlon))
    Threads:       $(Threads.nthreads())
    """

    # --- Find available dates ---
    dates = Date[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    isempty(dates) && error("No spectral GRIB files found in $spectral_dir")

    if day_filter !== nothing
        dates = filter(==(day_filter), dates)
        isempty(dates) && error("Date $day_filter not found in spectral data")
    end

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))

    mkpath(out_dir)
    t_total = time()

    for (i, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", i, length(dates), date)

        # Try to get next day's hour 0 for the last-window delta
        next_day = date + Day(1)
        next_day_h0 = nothing
        if next_day in dates || isfile(joinpath(spectral_dir,
                "era5_spectral_$(Dates.format(next_day, "yyyymmdd"))_lnsp.gb"))
            next_day_h0 = read_hour0_spectral(spectral_dir, next_day; T_target)
            if next_day_h0 !== nothing
                @info "  Next day hour 0 available for last-window delta"
            end
        end

        result = process_day(date, grid, ab, level_range,
                             vc_native, merged_vc, merge_map,
                             spectral_dir, out_dir,
                             half_dt, dt, met_interval,
                             T_target, min_dp;
                             next_day_hour0=next_day_h0)

        result === nothing && continue
    end

    wall_total = round(time() - t_total, digits=1)
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), wall_total,
                   length(dates) > 0 ? wall_total / length(dates) : 0.0)
end

main()
