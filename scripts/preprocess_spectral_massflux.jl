#!/usr/bin/env julia
# ===========================================================================
# Convert ERA5 spectral VO/D/LNSP → mass fluxes on regular lat-lon grid
#
# Reads spectral spherical harmonic coefficients from GRIB files, converts
# vorticity/divergence → U/V winds via the Helmholtz decomposition, performs
# an inverse spherical harmonic transform to the target grid, then computes
# mass-conserving air mass, horizontal mass fluxes, and vertical mass flux
# (from continuity).
#
# This is the spectral equivalent of preprocess_mass_fluxes.jl, following
# TM5's approach (Bregman et al. 2003) for exact mass conservation.
# Gridpoint-derived mass fluxes have ~0.9% mass drift/day; spectral-derived
# fluxes have < 0.01%.
#
# Input:
#   ERA5 spectral GRIB files (VO param 138, D param 155, LNSP param 152)
#   Downloaded by: scripts/download_era5_grib_tm5.py
#
# Output:
#   NetCDF mass-flux file matching preprocess_mass_fluxes.jl format:
#   m, am, bm, cm, ps with staggered dimensions.
#
# Usage:
#   julia --project=. scripts/preprocess_spectral_massflux.jl <config.toml>
#
# Config TOML sections:
#   [input]    — spectral_dir, coefficients
#   [output]   — directory
#   [grid]     — nlon, nlat, level_top, level_bot
#   [numerics] — float_type, dt, met_interval
# ===========================================================================

using GRIB
using FFTW
using NCDatasets
using Dates
using Printf
using TOML

# ===========================================================================
# Parse configuration from TOML
# ===========================================================================
length(ARGS) >= 1 || error(
    "Usage: julia --project=. scripts/preprocess_spectral_massflux.jl <config.toml>")
const config = TOML.parsefile(ARGS[1])

# Physical constants from defaults.toml (single source of truth)
const _defaults = TOML.parsefile(joinpath(@__DIR__, "..", "config", "defaults.toml"))
const R_EARTH = Float64(_defaults["planet"]["radius"])
const GRAV    = Float64(_defaults["planet"]["gravity"])

# Input / output paths
const SPECTRAL_DIR = expanduser(config["input"]["spectral_dir"])
const OUTDIR       = expanduser(config["output"]["directory"])

# Grid configuration
const TARGET_NLON = config["grid"]["nlon"]
const TARGET_NLAT = config["grid"]["nlat"]
const LEVEL_TOP   = config["grid"]["level_top"]
const LEVEL_BOT   = config["grid"]["level_bot"]
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT
const Nz = length(LEVEL_RANGE)

# Numerics
const FT_STR = get(config["numerics"], "float_type", "Float32")
const FT = FT_STR == "Float32" ? Float32 : Float64
const DT = FT(config["numerics"]["dt"])
const MET_INTERVAL = Float64(config["numerics"]["met_interval"])

# ===========================================================================
# Load A/B hybrid coefficients
# ===========================================================================
function load_ab_coefficients(level_range)
    coeff_path = get(get(config, "input", Dict()), "coefficients",
                     "config/era5_L137_coefficients.toml")
    toml_path = joinpath(@__DIR__, "..", coeff_path)
    cfg = TOML.parsefile(toml_path)
    a_all = Float64.(cfg["coefficients"]["a"])  # 138 values, n=0..137
    b_all = Float64.(cfg["coefficients"]["b"])
    # Extract interfaces for selected level range
    # Level k uses interfaces k and k+1 (1-based, where interface 1 = TOA)
    i_start = level_range[1]        # interface above first layer
    i_end   = level_range[end] + 1  # interface below last layer
    a_ifc = a_all[i_start:i_end]    # Nz+1 values
    b_ifc = b_all[i_start:i_end]
    # dA and dB for each layer: dp = dA + dB * ps
    dA = diff(a_ifc)  # Nz values (negative for decreasing-with-height convention)
    dB = diff(b_ifc)
    # A/B at cell centers (for driver metadata)
    A_center = [(a_ifc[k] + a_ifc[k+1]) / 2 for k in 1:Nz]
    B_center = [(b_ifc[k] + b_ifc[k+1]) / 2 for k in 1:Nz]
    return (; a_ifc, b_ifc, dA, dB, A_center, B_center)
end

# ===========================================================================
# Target regular lat-lon grid
# ===========================================================================
struct TargetGrid
    Nlon::Int
    Nlat::Int
    lons::Vector{Float64}   # cell centers [degrees]
    lats::Vector{Float64}   # cell centers [degrees]
    dlon::Float64            # longitude spacing [radians]
    dlat::Float64            # latitude spacing [radians]
    cos_lat::Vector{Float64} # cos(lat) at cell centers
    area::Matrix{Float64}   # cell area [m²] (Nlon × Nlat)
end

function TargetGrid(Nlon, Nlat)
    dlon_deg = 360.0 / Nlon
    dlat_deg = 180.0 / (Nlat - 1)
    lons = [dlon_deg * (i - 0.5) for i in 1:Nlon]       # 0.25, 0.75, ... or 0.5, 1.0, ...
    lats = [-90.0 + dlat_deg * (j - 1) for j in 1:Nlat]  # -90, -89.5, ..., 90
    dlon = deg2rad(dlon_deg)
    dlat = deg2rad(dlat_deg)
    cos_lat = [cosd(lat) for lat in lats]

    # Cell areas: Δλ × R² × |sin(φ_top) - sin(φ_bot)|
    area = zeros(Nlon, Nlat)
    for j in 1:Nlat
        φ_top = min(lats[j] + dlat_deg/2,  90.0)
        φ_bot = max(lats[j] - dlat_deg/2, -90.0)
        cell_a = R_EARTH^2 * dlon * abs(sind(φ_top) - sind(φ_bot))
        for i in 1:Nlon
            area[i, j] = cell_a
        end
    end
    return TargetGrid(Nlon, Nlat, lons, lats, dlon, dlat, cos_lat, area)
end

# ===========================================================================
# Fully normalized associated Legendre functions (ECMWF convention)
#
# P_n^m(x) such that ∫₋₁¹ [P_n^m(x)]² dx = 2 / (2n+1) × (2n+1) = 2
# Uses stable three-term recurrence to avoid numerical overflow.
# ===========================================================================
function compute_legendre_column!(P::Matrix{Float64}, T::Int, sin_lat::Float64)
    # P[n+1, m+1] = P_n^m(sin_lat), for m=0..T, n=m..T
    # Computed using the recurrence:
    #   P_m^m = sqrt(1 + 1/(2m)) * cos_lat * P_{m-1}^{m-1}   (sectoral)
    #   P_{m+1}^m = sqrt(2m+3) * sin_lat * P_m^m              (first off-diagonal)
    #   P_n^m = a_nm * sin_lat * P_{n-1}^m - b_nm * P_{n-2}^m (recurrence)
    # where a_nm = sqrt((4n²-1)/(n²-m²)), b_nm = sqrt(((2n+1)*(n-m-1)*(n+m-1))/((2n-3)*(n²-m²)))

    cos_lat = sqrt(1.0 - sin_lat^2)

    # Starting value: P_0^0 = 1
    fill!(P, 0.0)
    P[1, 1] = 1.0

    # Sectoral (diagonal): P_m^m from P_{m-1}^{m-1}
    for m in 1:T
        # P_m^m = sqrt((2m+1)/(2m)) * cos_lat * P_{m-1}^{m-1}
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
Read all spectral fields from a day's GRIB files.
Returns (vo_spec, d_spec, lnsp_spec, T, n_times) where:
- vo_spec[n+1, m+1, level, time] and d_spec same shape
- lnsp_spec[n+1, m+1, time]
- T = spectral truncation
"""
function read_day_spectral(vo_d_path::String, lnsp_path::String;
                           T_target::Int=0)
    # First pass: determine T and count messages
    f = GribFile(lnsp_path)
    first_msg = true
    T_file = 0
    n_times = 0
    for msg in f
        if first_msg
            T_file = msg["J"]
            first_msg = false
        end
        n_times += 1
    end
    destroy(f)

    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = 137  # ERA5 always has 137 levels

    # Allocate spectral arrays (only up to T_target)
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    lnsp_spec = zeros(ComplexF64, T + 1, T + 1, n_times)
    vo_spec   = zeros(ComplexF64, T + 1, T + 1, Nlevels, n_times)
    d_spec    = zeros(ComplexF64, T + 1, T + 1, Nlevels, n_times)

    # Read LNSP
    f = GribFile(lnsp_path)
    t_idx = 0
    for msg in f
        t_idx += 1
        read_spectral_coeffs!(spec_buf, msg)
        lnsp_spec[:, :, t_idx] .= @view spec_buf[1:T+1, 1:T+1]
    end
    destroy(f)

    # Read VO/D (interleaved: vo lev=1, d lev=1, vo lev=2, d lev=2, ...)
    f = GribFile(vo_d_path)
    for msg in f
        name  = msg["shortName"]
        level = msg["level"]
        dtime = msg["dataTime"]

        # Map dataTime (0, 600, 1200, 1800) to time index (1..4)
        t_idx = div(dtime, 600) + 1

        read_spectral_coeffs!(spec_buf, msg)
        if name == "vo"
            vo_spec[:, :, level, t_idx] .= @view spec_buf[1:T+1, 1:T+1]
        elseif name == "d"
            d_spec[:, :, level, t_idx] .= @view spec_buf[1:T+1, 1:T+1]
        end
    end
    destroy(f)

    return (; vo_spec, d_spec, lnsp_spec, T, n_times)
end

# ===========================================================================
# VO/D → U/V spectral conversion (port of TM5 grid_type_sh.F90 sh_vod2uv)
#
# U(m,n) = R * [δ(m,n)*VO(m,n-1) + i*σ(m,n)*D(m,n) - δ(m,n+1)*VO(m,n+1)]
# V(m,n) = R * [-δ(m,n)*D(m,n-1) + i*σ(m,n)*VO(m,n) + δ(m,n+1)*D(m,n+1)]
#
# where δ(m,n) = -sqrt((n²-m²)/(4n²-1)) / n
#       σ(m,n) = -m / (n(n+1))
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
Operates on a single level: vo_spec[n+1, m+1] → u_spec, v_spec.
"""
function vod2uv!(u_spec::Matrix{ComplexF64}, v_spec::Matrix{ComplexF64},
                 vo_spec::AbstractMatrix{ComplexF64}, d_spec::AbstractMatrix{ComplexF64},
                 T::Int)
    fill!(u_spec, zero(ComplexF64))
    fill!(v_spec, zero(ComplexF64))

    for m in 0:T
        for n in m:T
            δ_mn  = _delta(m, n)
            δ_mn1 = _delta(m, n + 1)
            σ_mn  = _sigma(m, n)

            # VO/D at neighboring n values (zero at boundaries)
            vo_nm1 = n > m     ? vo_spec[n, m+1]   : zero(ComplexF64)  # VO(m, n-1)
            vo_np1 = n < T     ? vo_spec[n+2, m+1] : zero(ComplexF64)  # VO(m, n+1)
            d_nm1  = n > m     ? d_spec[n, m+1]    : zero(ComplexF64)  # D(m, n-1)
            d_np1  = n < T     ? d_spec[n+2, m+1]  : zero(ComplexF64)  # D(m, n+1)
            vo_n   = vo_spec[n+1, m+1]
            d_n    = d_spec[n+1, m+1]

            u_spec[n+1, m+1] = R_EARTH * (δ_mn * vo_nm1 + im * σ_mn * d_n - δ_mn1 * vo_np1)
            v_spec[n+1, m+1] = R_EARTH * (-δ_mn * d_nm1 + im * σ_mn * vo_n + δ_mn1 * d_np1)
        end
    end
    return nothing
end

# ===========================================================================
# Inverse Spherical Harmonic Transform: spectral → regular lat-lon grid
#
# f(λ,φ) = Σ_m exp(imλ) × Σ_n f_n^m × P_n^m(sin φ)
#
# For each latitude: compute Fourier coefficients via Legendre sum,
# then inverse FFT to longitude values.
# ===========================================================================
"""
Transform a single spectral field to a regular lat-lon grid.
- spec[n+1, m+1] = spectral coefficients (complex, m=0..T, n=m..T)
- T = spectral truncation
- lats = target latitudes [degrees]
- Nlon = number of target longitudes
Returns field[Nlon, Nlat] on the target grid.
"""
function spectral_to_grid!(field::Matrix{FT_out},
                           spec::AbstractMatrix{ComplexF64},
                           T::Int,
                           lats::Vector{Float64},
                           Nlon::Int,
                           P_buf::Matrix{Float64},
                           fft_buf::Vector{ComplexF64}) where FT_out
    Nlat = length(lats)
    Nfft = Nlon  # number of FFT points

    for j in 1:Nlat
        sin_lat = sind(lats[j])

        # Compute Legendre functions P_n^m(sin_lat)
        compute_legendre_column!(P_buf, T, sin_lat)

        # Legendre sum: G_m = Σ_n spec[n+1, m+1] * P[n+1, m+1]
        fill!(fft_buf, zero(ComplexF64))
        for m in 0:min(T, div(Nfft, 2))
            Gm = zero(ComplexF64)
            @inbounds for n in m:T
                Gm += spec[n+1, m+1] * P_buf[n+1, m+1]
            end
            fft_buf[m+1] = Gm
        end

        # Fill negative frequencies for real-valued field:
        # f(λ) = Σ_{m≥0} G_m exp(imλ) + Σ_{m≥1} conj(G_m) exp(-imλ)
        # In DFT layout: negative freq m maps to index N-m+1
        for m in 1:min(T, div(Nfft, 2) - 1)
            fft_buf[Nfft - m + 1] = conj(fft_buf[m + 1])
        end

        # Fourier synthesis via backward FFT (unnormalized):
        # bfft(G)[k] = Σ_m G[m] exp(2πi m(k-1)/N) = f(λ_k)
        # No 1/N scaling — this is a direct evaluation, not an inverse DFT.
        f_lon = bfft(fft_buf)

        @inbounds for i in 1:Nlon
            field[i, j] = FT_out(real(f_lon[i]))
        end
    end
    return nothing
end

# ===========================================================================
# Mass flux computation on the target grid
#
# Following preprocess_mass_fluxes.jl conventions:
# - m = air mass per cell [kg]
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
Compute pressure thickness Δp per cell from A/B coefficients and surface pressure.
Δp[i,j,k] = |dA[k] + dB[k] * ps[i,j]|
"""
function compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        dp[i, j, k] = abs(dA[k] + dB[k] * ps[i, j])
    end
    return nothing
end

"""
Compute air mass per cell: m = Δp * area / g
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

am[i, j, k] = Δp_stag × cos(lat) × Δlat × R / g × u_stag × half_dt
bm[i, j, k] = Δp_stag × Δlon × R / g × v_stag × half_dt
cm from continuity: cm[k+1] = cm[k] + Σ_horiz(flux divergence at level k)
"""
function compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, ps,
                               dA, dB, grid::TargetGrid, half_dt)
    Nlon, Nlat = grid.Nlon, grid.Nlat
    R_g = R_EARTH / GRAV
    dlon = grid.dlon
    dlat = grid.dlat

    # --- Eastward mass flux at x-faces: am[Nlon+1, Nlat, Nz] ---
    # Staggered Δp at x-faces = average of adjacent cells
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon+1)
        i_l = i == 1 ? Nlon : i - 1
        i_r = i <= Nlon ? i : 1
        dp_stag = (dp[i_l, j, k] + dp[i_r, j, k]) / 2
        # Mass flux = ρ × u × face_area × dt
        # face_area_x = R × dlat × cos(lat) ... wait, this is the face perpendicular to x
        # Actually: am = u × dp/g × R × Δφ × half_dt  (dp/g = mass per unit area per level)
        # The face width in y-direction is R × Δφ
        am[i, j, k] = u_stag[i, j, k] * dp_stag * R_g * dlat * half_dt
    end

    # --- Northward mass flux at y-faces: bm[Nlon, Nlat+1, Nz] ---
    @inbounds for k in 1:Nz, j in 1:(Nlat+1), i in 1:Nlon
        if j == 1 || j == Nlat + 1
            bm[i, j, k] = 0  # No flux through poles
        else
            j_s = j - 1
            j_n = j
            dp_stag = (dp[i, j_s, k] + dp[i, j_n, k]) / 2
            # Face width in x-direction = R × cos(lat_face) × Δλ
            lat_face = (grid.lats[j_s] + grid.lats[j_n]) / 2
            cos_face = cosd(lat_face)
            bm[i, j, k] = v_stag[i, j, k] * dp_stag * R_g * dlon * cos_face * half_dt
        end
    end

    # --- Vertical mass flux from continuity: cm[Nlon, Nlat, Nz+1] ---
    # cm[k=1] = 0 (no flux through TOA)
    # cm[k+1] = cm[k] + horizontal divergence at level k
    # div_h = (am[i+1] - am[i]) / ... + (bm[j+1] - bm[j]) / ...
    # But actually cm is computed to conserve mass:
    # m_new = m_old + (flux_in - flux_out) per cell per level
    # cm[k+1] = cm[k] + (am_in - am_out + bm_in - bm_out at level k)
    fill!(cm, zero(eltype(cm)))
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        # Horizontal flux divergence (out - in) at cell (i,j,k)
        # am: flux through east face (i+1) minus west face (i)
        # bm: flux through north face (j+1) minus south face (j)
        div_h = (am[i+1, j, k] - am[i, j, k]) + (bm[i, j+1, k] - bm[i, j, k])
        # Continuity: cm[k+1] = cm[k] - div_h  (positive cm = downward)
        cm[i, j, k+1] = cm[i, j, k] - div_h
    end

    return nothing
end

# ===========================================================================
# NetCDF output (matching preprocess_mass_fluxes.jl format)
# ===========================================================================
function create_output_file(outpath::String, grid::TargetGrid, ab,
                            half_dt, met_interval, steps_per_met, month_key)
    Nlon, Nlat = grid.Nlon, grid.Nlat

    year_str  = month_key[1:4]
    month_str = month_key[5:6]
    time_origin = "hours since $(year_str)-$(month_str)-01 00:00:00"

    rm(outpath; force=true)
    mkpath(dirname(outpath))

    ds = NCDataset(outpath, "c")
    ds.attrib["title"] = "Pre-computed mass fluxes for AtmosTransport (spectral)"
    ds.attrib["source"] = "ERA5 spectral VO/D/LNSP (TM5-style mass flux derivation)"
    ds.attrib["float_type"] = string(FT)
    ds.attrib["dt_seconds"] = Float64(DT)
    ds.attrib["half_dt_seconds"] = Float64(half_dt)
    ds.attrib["level_top"] = LEVEL_TOP
    ds.attrib["level_bot"] = LEVEL_BOT
    ds.attrib["steps_per_met_window"] = steps_per_met
    ds.attrib["month"] = month_key
    ds.attrib["history"] = "Created $(Dates.now()) by preprocess_spectral_massflux.jl"
    ds.attrib["spectral_truncation"] = "T639 (read), truncated for target grid"

    defDim(ds, "lon", Nlon)
    defDim(ds, "lat", Nlat)
    defDim(ds, "lev", Nz)
    defDim(ds, "lon_u", Nlon + 1)
    defDim(ds, "lat_v", Nlat + 1)
    defDim(ds, "lev_w", Nz + 1)
    defDim(ds, "time", Inf)

    defVar(ds, "lon", Float32, ("lon",);
           attrib = Dict("units" => "degrees_east"))[:] = Float32.(grid.lons)
    defVar(ds, "lat", Float32, ("lat",);
           attrib = Dict("units" => "degrees_north"))[:] = Float32.(grid.lats)
    defVar(ds, "time", Float64, ("time",);
           attrib = Dict("units" => time_origin))

    defVar(ds, "A_coeff", Float64, ("lev",);
           attrib = Dict("long_name" => "Hybrid A coefficient at level centers"))
    defVar(ds, "B_coeff", Float64, ("lev",);
           attrib = Dict("long_name" => "Hybrid B coefficient at level centers"))
    ds["A_coeff"][:] = ab.A_center
    ds["B_coeff"][:] = ab.B_center

    defVar(ds, "m", FT, ("lon", "lat", "lev", "time");
           attrib = Dict("units" => "kg", "long_name" => "Air mass per cell"),
           deflatelevel = 1)
    defVar(ds, "am", FT, ("lon_u", "lat", "lev", "time");
           attrib = Dict("units" => "kg",
                         "long_name" => "Eastward mass flux at x-faces per half-timestep"),
           deflatelevel = 1)
    defVar(ds, "bm", FT, ("lon", "lat_v", "lev", "time");
           attrib = Dict("units" => "kg",
                         "long_name" => "Northward mass flux at y-faces per half-timestep"),
           deflatelevel = 1)
    defVar(ds, "cm", FT, ("lon", "lat", "lev_w", "time");
           attrib = Dict("units" => "kg",
                         "long_name" => "Downward mass flux at z-interfaces per half-timestep"),
           deflatelevel = 1)
    defVar(ds, "ps", FT, ("lon", "lat", "time");
           attrib = Dict("units" => "Pa", "long_name" => "Surface pressure"),
           deflatelevel = 1)

    return ds
end

# ===========================================================================
# Main processing loop
# ===========================================================================
function preprocess()
    @info """
    Spectral Mass Flux Preprocessor
    ================================
    Spectral dir:  $SPECTRAL_DIR
    Output dir:    $OUTDIR
    Target grid:   $(TARGET_NLON) × $(TARGET_NLAT) ($(360.0/TARGET_NLON)° × $(180.0/(TARGET_NLAT-1))°)
    Levels:        $(LEVEL_TOP)-$(LEVEL_BOT) ($(Nz) layers)
    Float type:    $(FT)
    DT:            $(DT) s
    Met interval:  $(MET_INTERVAL) s
    """

    # Set up target grid
    grid = TargetGrid(TARGET_NLON, TARGET_NLAT)

    # Load A/B coefficients
    ab = load_ab_coefficients(LEVEL_RANGE)

    # Compute derived parameters
    half_dt = FT(DT / 2)
    steps_per_met = max(1, round(Int, MET_INTERVAL / DT))

    # Maximum spectral truncation for target grid
    T_target = div(TARGET_NLON, 2) - 1  # Nyquist: 720/2 - 1 = 359

    # Find spectral GRIB files
    dates = Date[]
    for f in readdir(SPECTRAL_DIR)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    isempty(dates) && error("No spectral GRIB files found in $SPECTRAL_DIR")

    @info "Found $(length(dates)) days of spectral data"

    # Group by month
    month_groups = Dict{String, Vector{Date}}()
    for d in dates
        key = Dates.format(d, "yyyymm")
        push!(get!(month_groups, key, Date[]), d)
    end

    # Allocate work arrays
    u_cc    = Array{FT}(undef, TARGET_NLON, TARGET_NLAT, Nz)
    v_cc    = Array{FT}(undef, TARGET_NLON, TARGET_NLAT, Nz)
    u_stag  = Array{FT}(undef, TARGET_NLON + 1, TARGET_NLAT, Nz)
    v_stag  = Array{FT}(undef, TARGET_NLON, TARGET_NLAT + 1, Nz)
    dp      = Array{FT}(undef, TARGET_NLON, TARGET_NLAT, Nz)
    m_arr   = Array{FT}(undef, TARGET_NLON, TARGET_NLAT, Nz)
    am      = Array{FT}(undef, TARGET_NLON + 1, TARGET_NLAT, Nz)
    bm      = Array{FT}(undef, TARGET_NLON, TARGET_NLAT + 1, Nz)
    cm      = Array{FT}(undef, TARGET_NLON, TARGET_NLAT, Nz + 1)
    sp      = Array{FT}(undef, TARGET_NLON, TARGET_NLAT)

    # SHT work buffers
    P_buf    = zeros(Float64, T_target + 1, T_target + 1)
    fft_buf  = zeros(ComplexF64, TARGET_NLON)
    u_spec   = zeros(ComplexF64, T_target + 1, T_target + 1)
    v_spec   = zeros(ComplexF64, T_target + 1, T_target + 1)
    field_2d = Array{FT}(undef, TARGET_NLON, TARGET_NLAT)

    for (month_key, month_dates) in sort(collect(month_groups))
        sort!(month_dates)
        outpath = joinpath(OUTDIR, "massflux_era5_spectral_$(month_key)_$(lowercase(FT_STR)).nc")

        @info "Processing month $month_key: $(length(month_dates)) days → $outpath"

        ds = create_output_file(outpath, grid, ab, half_dt, MET_INTERVAL, steps_per_met, month_key)

        tidx_out = 0
        t_month = time()

        for date in month_dates
            datestr = Dates.format(date, "yyyymmdd")
            vo_d_path = joinpath(SPECTRAL_DIR, "era5_spectral_$(datestr)_vo_d.gb")
            lnsp_path = joinpath(SPECTRAL_DIR, "era5_spectral_$(datestr)_lnsp.gb")

            if !isfile(vo_d_path) || !isfile(lnsp_path)
                @warn "Missing files for $datestr, skipping"
                continue
            end

            t_day = time()
            @info "  Reading spectral data for $datestr..."

            # Read all spectral data for this day
            spec = read_day_spectral(vo_d_path, lnsp_path; T_target)

            @info "  Spectral data read: T=$(spec.T), $(spec.n_times) timesteps ($(round(time()-t_day, digits=1))s)"

            for t in 1:spec.n_times
                t0 = time()
                tidx_out += 1

                # --- Transform LNSP → SP on target grid ---
                spectral_to_grid!(field_2d, @view(spec.lnsp_spec[:, :, t]),
                                  spec.T, grid.lats, grid.Nlon, P_buf, fft_buf)
                @. sp = FT(exp(field_2d))

                # --- For each level: VO/D → U/V spectral, then transform ---
                for k in 1:Nz
                    level = LEVEL_RANGE[k]

                    # VO/D → U/V in spectral space
                    vod2uv!(u_spec, v_spec,
                            @view(spec.vo_spec[:, :, level, t]),
                            @view(spec.d_spec[:, :, level, t]),
                            spec.T)

                    # U spectral → gridpoint
                    spectral_to_grid!(field_2d, u_spec, spec.T,
                                     grid.lats, grid.Nlon, P_buf, fft_buf)
                    u_cc[:, :, k] .= field_2d

                    # V spectral → gridpoint
                    spectral_to_grid!(field_2d, v_spec, spec.T,
                                     grid.lats, grid.Nlon, P_buf, fft_buf)
                    v_cc[:, :, k] .= field_2d
                end

                # --- Compute mass fluxes ---
                stagger_winds!(u_stag, v_stag, u_cc, v_cc, grid.Nlon, grid.Nlat, Nz)
                compute_dp!(dp, sp, ab.dA, ab.dB, grid.Nlon, grid.Nlat, Nz)
                compute_air_mass!(m_arr, dp, grid.area, grid.Nlon, grid.Nlat, Nz)
                compute_mass_fluxes!(am, bm, cm, u_stag, v_stag, dp, sp,
                                     ab.dA, ab.dB, grid, half_dt)

                # --- Write to NetCDF ---
                sim_hours = (tidx_out - 1) * MET_INTERVAL / 3600.0
                ds["time"][tidx_out] = sim_hours
                ds["m"][:, :, :, tidx_out]  = m_arr
                ds["am"][:, :, :, tidx_out] = am
                ds["bm"][:, :, :, tidx_out] = bm
                ds["cm"][:, :, :, tidx_out] = cm
                ds["ps"][:, :, tidx_out]    = sp

                elapsed = round(time() - t0, digits=2)
                @info @sprintf("    [%s] Window %d (hour %.0f): %.2fs",
                               month_key, tidx_out, sim_hours, elapsed)
            end

            @info @sprintf("  Day %s done in %.1fs", datestr, time() - t_day)
        end

        close(ds)
        wall_month = round(time() - t_month, digits=1)
        sz_gb = round(filesize(outpath) / 1e9, digits=2)
        @info @sprintf("  [%s] Complete: %d windows in %.1fs (%.1fs/win), %.2f GB",
                       month_key, tidx_out, wall_month,
                       tidx_out > 0 ? wall_month / tidx_out : 0.0, sz_gb)
    end

    @info "All done!"
end

preprocess()
