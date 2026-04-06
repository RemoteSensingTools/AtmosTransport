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
using NCDatasets

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

"""
    select_levels_echlevs(vc_native, echlevs)

TM5 r1112-style level selection: pick specific native levels by index.
No summation — am/bm at selected levels are the native values at those levels,
and cm at selected interfaces are the native cm at those interfaces.

`echlevs` is a vector of native level INTERFACE indices (0-based, bottom-up),
e.g., [137, 134, 129, ..., 7, 0] for TM5's ml137→tropo34 config.
Index 0 = TOA, 137 = surface.

Returns (selected_vc, merge_map) where merge_map[k] = merged level index
for native level k. Levels between selected interfaces are summed (like TM5).
"""
function select_levels_echlevs(vc_native::HybridSigmaPressure{FT},
                                echlevs::Vector{Int}) where FT
    Nz_native = n_levels(vc_native)
    # echlevs are 0-based interface indices; convert to 1-based Julia
    # TM5 convention: echlevs[0]=Nz (surface), echlevs[end]=0 (TOA)
    # We need the interface indices in top-to-bottom order (Julia convention)
    ifaces_0based = sort(echlevs)  # sorted: 0, 7, 12, ..., 134, 137
    # Convert to 1-based: interface k in Julia = native level boundary k
    # Native vc has Nz_native+1 interfaces (A[1]..A[Nz_native+1])
    # echlevs 0-based index i → Julia index Nz_native+1-i (flip for top-down)
    keep = Int[Nz_native + 1 - i for i in reverse(ifaces_0based)]
    # keep is now 1-based, top-to-bottom: [1, ..., Nz_native+1]

    selected_vc = HybridSigmaPressure(FT[vc_native.A[k] for k in keep],
                                       FT[vc_native.B[k] for k in keep])
    Nz_selected = n_levels(selected_vc)

    # Build merge_map: for each native level k, which selected level contains it?
    mm = Vector{Int}(undef, Nz_native)
    km = 1
    for k in 1:Nz_native
        while km < Nz_selected && keep[km + 1] <= k; km += 1; end
        mm[k] = km
    end

    @info "echlevs level selection: $(Nz_native) → $(Nz_selected) levels " *
          "($(length(echlevs)) interfaces)"
    return selected_vc, mm
end

# TM5 r1112 level configurations (from deps/tm5-mp-r1112/levels/)
const ECHLEVS_ML137_TROPO34 = [
    137, 134, 129, 124, 119, 114, 110, 105, 101,  97,
     93,  88,  84,  81,  78,  76,  73,  70,  67,  65,
     62,  59,  57,  54,  51,  46,  42,  37,  32,  27,
     22,  17,  12,   7,   0]

# 66 levels: fine BL (every native level k=0..30), mid-trop (every 2nd k=30..60),
# upper trop (every 3rd k=60..100), stratosphere (every 5th k=100..137)
const ECHLEVS_ML137_66L = [
    137, 135, 130, 125, 120, 115, 110, 105, 100,
     96,  93,  90,  87,  84,  81,  78,  75,  72,  69,  66,
     63,  60,  58,  56,  54,  52,  50,  48,  46,  44,
     42,  40,  38,  36,  34,  32,  30,  29,  28,  27,
     26,  25,  24,  23,  22,  21,  20,  19,  18,  17,
     16,  15,  14,  13,  12,  11,  10,   9,   8,   7,
      6,   5,   4,   3,   2,   1,   0]

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
    balance_mass_fluxes!(am, bm, dm_dt, Nx, Ny, Nz)

TM5-style Poisson mass flux balance (grid_type_ll.F90:2536-2653, r1112).
Adjusts am/bm so that horizontal convergence exactly matches the prescribed
mass tendency dm_dt at every cell and level.

For each level l:
  residual = convergence(am,bm) - dm_dt
  Solve ∇²ψ = residual (periodic in x, via 2D FFT)
  am += dψ/dx, bm += dψ/dy

After balancing, cm computed from the balanced am/bm via B-correction will
produce zero surface residual, eliminating the moist flux inconsistency that
caused 171 ppm SH std growth from uniform IC.
"""
function balance_mass_fluxes!(am::Array{FT,3}, bm::Array{FT,3},
                               dm_dt::Array{FT,3}) where FT
    Nx = size(am, 1) - 1
    Ny = size(bm, 2) - 1
    Nz = size(am, 3)

    # Precompute FFT eigenvalues: fac(i,j) = 2(cos(2π(i-1)/Nx) + cos(2π(j-1)/Ny) - 2)
    fac = Array{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nx) + cos(2π * (j - 1) / Ny) - 2.0)
    end
    fac[1, 1] = 1.0  # avoid division by zero (mean mode)

    psi = Array{Float64}(undef, Nx, Ny)
    residual = Array{Float64}(undef, Nx, Ny)

    n_balanced = 0
    max_residual = 0.0

    for k in 1:Nz
        # Compute residual: convergence(am,bm) - dm_dt
        @inbounds for j in 1:Ny, i in 1:Nx
            conv = (Float64(am[i, j, k]) - Float64(am[i+1, j, k])) +
                   (Float64(bm[i, j, k]) - Float64(bm[i, j+1, k]))
            residual[i, j] = conv - Float64(dm_dt[i, j, k])
        end

        max_res_k = maximum(abs, residual)
        max_residual = max(max_residual, max_res_k)
        max_res_k < 1e-10 && continue  # already balanced

        # 2D FFT Poisson solve (TM5 SolvePoissonEq_zoom)
        A = fft(complex.(residual))
        # Solve: fac * Ψ_hat = residual_hat → Ψ_hat = residual_hat / fac
        @inbounds for j in 1:Ny, i in 1:Nx
            A[i, j] /= fac[i, j]
        end
        A[1, 1] = 0.0 + 0.0im  # zero mean mode
        psi .= real.(ifft(A))

        # Correction fluxes from potential ψ (TM5 SolvePoissonEq_zoom:2873-2897)
        # u = dψ/dx, v = dψ/dy, then subtract boundary values to enforce zero-flux BCs.

        # X-direction: compute u[0:Nx] for each j, subtract boundary, apply
        @inbounds for j in 1:Ny
            # Compute all u corrections (periodic differences)
            u_wrap = psi[1, j] - psi[Nx, j]  # u[0] = u[Nx] (periodic)
            # Subtract boundary flux (TM5 line 2881: col = u(0,:))
            # All u values are shifted so u[0] = 0
            # u[i] - u[0] = (psi[i+1]-psi[i]) - (psi[1]-psi[Nx])
            for i in 2:Nx
                du = (psi[i, j] - psi[i-1, j]) - u_wrap
                am[i, j, k] += FT(du)  # TM5: pu += cqu (ADDITION)
            end
            # i=1: u[1]-u[0] = (psi[2]-psi[1]) - (psi[1]-psi[Nx]) — but we skip
            # i=1 because it matches i=Nx+1 (periodic); handle both:
            am[1, j, k] += FT(0)     # u[0]-u[0] = 0 (zeroed by boundary subtraction)
            am[Nx+1, j, k] += FT(0)  # same as i=1 (periodic)
        end

        # Y-direction: compute v[0:Ny] for each i, subtract boundary, apply
        @inbounds for i in 1:Nx
            v_wrap = psi[i, 1] - psi[i, Ny]  # v[0] = v[Ny] (periodic in TM5)
            for j in 2:Ny
                dv = (psi[i, j] - psi[i, j-1]) - v_wrap
                bm[i, j, k] += FT(dv)  # TM5: pv += cqv (ADDITION)
            end
            # j=1 and j=Ny+1: pole faces stay zero (no correction applied)
        end

        n_balanced += 1
    end

    @info "Poisson balance: corrected $n_balanced/$Nz levels, " *
          "max pre-balance residual: $(round(max_residual, sigdigits=3)) kg"
end

"""
Merge QV (specific humidity) from native to merged levels using mass-weighted averaging.
QV is an intensive quantity — merge by mass-weighted mean, not sum.
"""
function merge_qv!(qv_merged::Array{FT,3}, qv_native::Array{FT,3},
                    m_native::Array{FT,3}, mm::Vector{Int}) where FT
    Nx, Ny = size(qv_merged, 1), size(qv_merged, 2)
    Nz_merged = size(qv_merged, 3)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, Nx, Ny, Nz_merged)
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views qv_merged[:, :, km] .+= qv_native[:, :, k] .* m_native[:, :, k]
        @views m_sum[:, :, km]     .+= m_native[:, :, k]
    end
    @inbounds for km in 1:Nz_merged
        @views qv_merged[:, :, km] ./= max.(m_sum[:, :, km], FT(1))
    end
end

"""
Read QV from ERA5 thermo NetCDF file for a specific hour.
Returns Array{Float32, 3} of shape (Nx, Ny, Nz_native).
"""
function read_qv_from_thermo(thermo_path::String, hour_idx::Int, Nx::Int, Ny::Int, Nz::Int)
    NCDataset(thermo_path) do ds
        # q has shape (time, hybrid, lat, lon) or (lon, lat, hybrid, time)
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            # (lon, lat, hybrid, time)
            q = Float32.(q_var[:, :, :, hour_idx])  # Nx × Ny × Nz
        else
            # (time, hybrid, lat, lon) — need to permute
            q_raw = Float32.(q_var[hour_idx, :, :, :])  # Nz × Ny × Nx
            q = permutedims(q_raw, (3, 2, 1))  # Nx × Ny × Nz
        end
        # ERA5 lat is N→S, our grid is S→N: flip latitude
        if size(q, 2) == Ny && ds["latitude"][1] > ds["latitude"][end]
            q = q[:, end:-1:1, :]
        end
        return q
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
                     next_day_hour0=nothing,
                     thermo_dir::String="",
                     include_qv::Bool=false)
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
    n_qv  = include_qv ? n_m : Int64(0)

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps + n_qv + n_dam + n_dbm + n_dm
    bytes_per_window = elems_per_window * sizeof(FT)
    total_bytes = Int64(HEADER_SIZE) + bytes_per_window * Nt

    # --- Output path ---
    mkpath(out_dir)
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    bin_path = joinpath(out_dir, "era5_v4_$(date_str)_$(dp_tag)_float32.bin")

    if isfile(bin_path) && filesize(bin_path) == total_bytes
        @info "  SKIP (exists, correct size): $(basename(bin_path))"
        return bin_path
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), total_bytes / 1e9, Nt)

    # --- Build header ---
    header = Dict{String,Any}(
        "magic" => "MFLX", "version" => 4, "header_bytes" => HEADER_SIZE,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nz_native" => Nz_native, "Nt" => Nt,
        "float_type" => "Float32", "float_bytes" => sizeof(FT),
        "window_bytes" => bytes_per_window,
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => n_qv, "n_cmfmc" => 0,
        "n_entu" => 0, "n_detu" => 0, "n_entd" => 0, "n_detd" => 0,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "n_dam" => n_dam, "n_dbm" => n_dbm, "n_dm" => n_dm,
        "include_flux_delta" => true,
        "include_qv" => include_qv, "include_cmfmc" => false,
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
    all_qv = include_qv ? Vector{Array{FT,3}}(undef, Nt) : nothing

    # QV file path (thermo NetCDF)
    thermo_path = ""
    if include_qv
        thermo_path = joinpath(thermo_dir, "era5_thermo_ml_$(date_str).nc")
        isfile(thermo_path) || error("Thermo file not found: $thermo_path")
    end
    qv_native = include_qv ? Array{FT}(undef, Nx, Ny, Nz_native) : nothing
    qv_merged = include_qv ? Array{FT}(undef, Nx, Ny, Nz) : nothing

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

        # Read and merge QV from thermo file
        if include_qv
            qv_native .= read_qv_from_thermo(thermo_path, win_idx, Nx, Ny, Nz_native)
            merge_qv!(qv_merged, qv_native, m_native, merge_map)
        end

        # Store for binary write + delta computation
        all_m[win_idx]  = copy(m_merged)
        all_am[win_idx] = copy(am_merged)
        all_bm[win_idx] = copy(bm_merged)
        all_cm[win_idx] = copy(cm_merged)
        all_ps[win_idx] = FT.(sp)
        if include_qv; all_qv[win_idx] = copy(qv_merged); end

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
        last_hour_next_m  = copy(m_next_merged)
        last_hour_next_am = copy(am_next_merged)
        last_hour_next_bm = copy(bm_next_merged)
    end

    # --- Poisson mass flux balance (TM5 BalanceMassFluxes) ---
    # Adjust am/bm so horizontal convergence matches prescribed mass change dm_dt
    # at every cell and level. Then recompute cm from balanced am/bm.
    @info "  Applying Poisson mass flux balance..."
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    for win_idx in 1:Nt
        # dm_dt = m_next - m_curr
        if win_idx < Nt
            dm_dt_buf .= all_m[win_idx + 1] .- all_m[win_idx]
        elseif last_hour_next_m !== nothing
            dm_dt_buf .= last_hour_next_m .- all_m[win_idx]
        else
            fill!(dm_dt_buf, zero(FT))
        end
        # Balance am/bm against dm_dt
        balance_mass_fluxes!(all_am[win_idx], all_bm[win_idx], dm_dt_buf)
        # Re-zero pole bm after balance (corrections may have touched them)
        @views all_bm[win_idx][:, 1, :]    .= zero(FT)
        @views all_bm[win_idx][:, Ny+1, :] .= zero(FT)
        # Recompute cm from balanced am/bm
        recompute_cm_from_divergence!(all_cm[win_idx], all_am[win_idx], all_bm[win_idx],
                                      all_m[win_idx]; B_ifc=merged_vc.B)
        @views all_cm[win_idx][:, :, 1]      .= zero(FT)
        @views all_cm[win_idx][:, :, Nz + 1] .= zero(FT)
    end
    @info "  Poisson balance complete for $Nt windows"

    # --- Write binary ---
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
            if include_qv
                bytes_written += write_array!(io, all_qv[win_idx])
            end

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

            bytes_written += write_array!(io, dam_merged)
            bytes_written += write_array!(io, dbm_merged)
            bytes_written += write_array!(io, dm_merged)
        end
        flush(io)
    end

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path), actual / 1e9, time() - t_day)
    actual == total_bytes ||
        error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d", total_bytes, actual))

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
    thermo_dir   = expanduser(get(get(cfg, "input", Dict()), "thermo_dir", ""))
    out_dir      = expanduser(cfg["output"]["directory"])
    include_qv   = !isempty(thermo_dir)

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
    echlevs_name = get(get(cfg, "grid", Dict()), "echlevs", "")
    if !isempty(echlevs_name)
        # TM5 r1112-style index selection
        echlevs_map = Dict(
            "ml137_tropo34" => ECHLEVS_ML137_TROPO34,
            "ml137_66L" => ECHLEVS_ML137_66L,
            "ml137_full" => collect(137:-1:0),  # all 137 native levels
        )
        haskey(echlevs_map, echlevs_name) || error("Unknown echlevs config: $echlevs_name. " *
            "Available: $(join(keys(echlevs_map), ", "))")
        merged_vc, merge_map = select_levels_echlevs(vc_native, echlevs_map[echlevs_name])
    else
        # Pressure-threshold merging (original approach)
        merged_vc, merge_map = merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
    end
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
                             next_day_hour0=next_day_h0,
                             thermo_dir=thermo_dir,
                             include_qv=include_qv)

        result === nothing && continue
    end

    wall_total = round(time() - t_total, digits=1)
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), wall_total,
                   length(dates) > 0 ? wall_total / length(dates) : 0.0)
end

main()
