#!/usr/bin/env julia
# ===========================================================================
# Unified ERA5 daily preprocessor → merged-level flat binary (v3)
#
# Reads:
#   - Base mass fluxes from spectral-preprocessed NetCDF (m, am, bm, cm, ps)
#   - Convection from download_era5_physics.py output (params 235009-235012)
#   - Surface fields from download_era5_surface_fields.py output (ZIP-wrapped)
#   - Thermodynamics (T + Q) from download_era5_physics.py output
#
# Computes:
#   - TM5 convection fields (entu, detu, entd, detd) via ECconv_to_TMconv
#   - CMFMC = UDMF + DDMF (net convective mass flux at interfaces)
#   - Friction velocity (from download or U10/V10 fallback)
#   - De-accumulated SSHF → hflux [W/m²]
#
# Merges vertical levels and writes v3 daily binary files with all fields.
#
# Usage:
#   julia -t8 --project=. scripts/preprocessing/preprocess_era5_daily.jl \
#       config/preprocessing/catrine_era5_daily_v3.toml [--day 2021-12-01]
#
# Binary layout (v3):
#   [16384-byte JSON header | per-window data × Nt]
#   Per window: m|am|bm|cm|ps | qv | cmfmc | entu|detu|entd|detd |
#               pblh|t2m|ustar|hflux | temperature
# ===========================================================================

using NCDatasets
using JSON3
using Printf
using TOML
using Dates
using ZipFile

# ===========================================================================
# Minimal vertical coordinate types (same as convert_merged_massflux_to_binary.jl)
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
# Merge kernels (from convert_merged_massflux_to_binary.jl)
# ===========================================================================

function merge_cell_field!(merged::Array{FT,3}, native::Array{FT,3}, mm::Vector{Int}) where FT
    Threads.@threads for km in 1:size(merged, 3)
        @views merged[:, :, km] .= zero(FT)
    end
    @inbounds for k in 1:length(mm)
        @views merged[:, :, mm[k]] .+= native[:, :, k]
    end
end

function merge_interface_field!(merged::Array{FT,3}, native::Array{FT,3}, mm::Vector{Int}) where FT
    fill!(merged, zero(FT))
    @views merged[:, :, 1] .= native[:, :, 1]
    @inbounds for km in 1:maximum(mm)
        k_last = findlast(==(km), mm)
        @views merged[:, :, km + 1] .= native[:, :, k_last + 1]
    end
end

function merge_qv!(qv_m::Array{FT,3}, qv_n::Array{FT,3}, m_n::Array{FT,3}, mm::Vector{Int}) where FT
    fill!(qv_m, zero(FT))
    m_sum = zeros(FT, size(qv_m))
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views begin; qv_m[:,:,km] .+= qv_n[:,:,k] .* m_n[:,:,k]; m_sum[:,:,km] .+= m_n[:,:,k]; end
    end
    @inbounds for idx in eachindex(qv_m)
        qv_m[idx] = m_sum[idx] > zero(FT) ? qv_m[idx] / m_sum[idx] : zero(FT)
    end
end

"""
Recompute cm from merged horizontal divergence using TM5's hybrid-coordinate formula.

TM5 formula (advect_tools.F90, dynam0):
  cm[k+1] = cm[k] - div_h[k] + (B[k+1] - B[k]) × pit

where pit = Σ div_h (column-integrated horizontal convergence) and B are the
hybrid sigma-pressure interface coefficients. The B-correction accounts for the
pressure coordinate surfaces moving as surface pressure changes, naturally giving
cm[1]=0 (TOA) and cm[Nz+1]=0 (surface) without any residual correction.

Without the B-term, simple accumulation cm[k+1] = cm[k] - div_h[k] leaves a
non-zero residual at the surface.
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
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i+1, j, k] - am[i, j, k]) + (bm[i, j+1, k] - bm[i, j, k])
            end
            # Build cm from TOA downward with B-correction
            for k in 1:Nz
                div_h = (am[i+1, j, k] - am[i, j, k]) + (bm[i, j+1, k] - bm[i, j, k])
                cm[i, j, k+1] = cm[i, j, k] - div_h + FT(B_ifc[k+1] - B_ifc[k]) * pit
            end
        end
    else
        # Fallback: simple accumulation (no B-correction)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            div_h = (am[i+1, j, k] - am[i, j, k]) + (bm[i, j+1, k] - bm[i, j, k])
            cm[i, j, k+1] = cm[i, j, k] - div_h
        end
    end
end

function correct_cm_residual!(cm::Array{FT,3}, m::Array{FT,3}) where FT
    Nx, Ny, Nz_plus1 = size(cm); Nz = Nz_plus1 - 1
    Threads.@threads for j in 1:Ny
        @inbounds for i in 1:Nx
            res = cm[i, j, Nz + 1]; abs(res) < eps(FT) && continue
            col = zero(FT); for k in 1:Nz; col += m[i, j, k]; end
            col < eps(FT) && continue
            cm[i, j, 1] = zero(FT); cum = zero(FT)
            for k in 1:Nz; cum += m[i, j, k]; cm[i, j, k+1] -= res * cum / col; end
        end
    end
end

# ===========================================================================
# ECconv_to_TMconv (from preprocess_era5_tm5_convection.jl)
# ===========================================================================

function ecconv_to_tmconv!(entu::Vector{FT}, detu_out::Vector{FT},
                           entd::Vector{FT}, detd_out::Vector{FT},
                           mflu_in::Vector{FT}, mfld_in::Vector{FT},
                           detu_rate::Vector{FT}, detd_rate::Vector{FT},
                           dz::Vector{FT}, Nz::Int) where FT
    mflu = copy(mflu_in); detu = copy(detu_rate)
    mfld = copy(mfld_in); detd = copy(detd_rate)
    # Clean small values
    @inbounds for k in 1:Nz+1
        if mflu[k] < FT(1e-6); mflu[k] = zero(FT); end
        if mfld[k] > FT(-1e-6); mfld[k] = zero(FT); end
    end
    @inbounds for k in 1:Nz
        if detu[k] < FT(1e-10); detu[k] = zero(FT); end
        if detd[k] < FT(1e-10); detd[k] = zero(FT); end
        detu[k] *= dz[k]; detd[k] *= dz[k]  # rate → flux
    end
    # Updraft entrainment
    fill!(entu, zero(FT))
    uptop = 0
    @inbounds for k in 1:Nz; if mflu[k+1] > zero(FT); uptop = k; break; end; end
    if uptop > 0
        for k in 1:uptop-1; entu[k] = zero(FT); detu[k] = zero(FT); end
        for k in uptop:Nz; entu[k] = mflu[k] - mflu[k+1] + detu[k]; end
    else; fill!(detu, zero(FT)); end
    # Downdraft entrainment
    fill!(entd, zero(FT))
    dotop = 0
    @inbounds for k in 1:Nz; if mfld[k+1] < zero(FT); dotop = k; break; end; end
    if dotop > 0
        for k in 1:dotop-1; detd[k] = zero(FT); entd[k] = zero(FT); end
        for k in dotop:Nz; entd[k] = mfld[k] - mfld[k+1] + detd[k]; end
    else; fill!(detd, zero(FT)); end
    # Fix negative values
    @inbounds for k in 1:Nz
        if entu[k] < zero(FT); detu[k] -= entu[k]; entu[k] = zero(FT); end
        if detu[k] < zero(FT); entu[k] -= detu[k]; detu[k] = zero(FT); end
        if entd[k] < zero(FT); detd[k] -= entd[k]; entd[k] = zero(FT); end
        if detd[k] < zero(FT); entd[k] -= detd[k]; detd[k] = zero(FT); end
    end
    detu_out .= detu; detd_out .= detd
    return nothing
end

function massflux_to_tmconv!(entu::Vector{FT}, detu::Vector{FT},
                             entd::Vector{FT}, detd::Vector{FT},
                             mflu_in::Vector{FT}, mfld_in::Vector{FT}, Nz::Int) where FT
    mflu = copy(mflu_in); mfld = copy(mfld_in)
    @inbounds for k in 1:Nz+1
        if mflu[k] < FT(1e-6); mflu[k] = zero(FT); end
        if mfld[k] > FT(-1e-6); mfld[k] = zero(FT); end
    end
    @inbounds for k in 1:Nz
        net = mflu[k] - mflu[k+1]
        if net > zero(FT); entu[k] = zero(FT); detu[k] = net
        else; entu[k] = -net; detu[k] = zero(FT); end
        net_d = mfld[k] - mfld[k+1]
        if net_d < zero(FT); entd[k] = -net_d; detd[k] = zero(FT)
        else; entd[k] = zero(FT); detd[k] = net_d; end
    end
end

function compute_dz_hydrostatic(ps::FT, Nz::Int, ak::Vector{Float64}, bk::Vector{Float64}) where FT
    R = 287.058; g = 9.80665; T_ref = 260.0
    dz = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        p_top = FT(ak[k] + bk[k] * ps); p_bot = FT(ak[k+1] + bk[k+1] * ps)
        dz[k] = FT(R * T_ref / g * (p_bot - p_top) / (FT(0.5) * (p_top + p_bot)))
    end
    return dz
end

# ===========================================================================
# External file readers
# ===========================================================================

"""Read convection fields for a specific valid_time from daily NC file.
Handles cross-day lookup: hours 00-06 of day D are in day D-1's file.
Flips latitude N→S to S→N to match spectral preprocessed convention.
Returns (udmf_3d, ddmf_3d, udrf_3d, ddrf_3d) or nothing."""
function read_convection_window(conv_dir::String, date::Date, hour::Int)
    target = DateTime(date) + Hour(hour)

    # Try current day's file first, then previous day's (hours 00-06 span boundary)
    for try_date in [date, date - Day(1)]
        date_str = Dates.format(try_date, "yyyymmdd")
        conv_file = joinpath(conv_dir, "era5_convection_$(date_str).nc")
        isfile(conv_file) || continue

        result = _try_read_conv(conv_file, target)
        result !== nothing && return result
    end
    return nothing
end

function _try_read_conv(conv_file::String, target::DateTime)
    ds = NCDataset(conv_file, "r")
    try
        # Convection files have dims: (longitude, latitude, hybrid, step, time)
        # valid_time is 2D: (step, time) in Julia NCDatasets ordering
        vt = ds["valid_time"][:, :]  # (step, time) or (time, step) — check shape

        # Find matching (tidx, sidx) for our target datetime
        tidx = sidx = 0
        for ci in CartesianIndices(vt)
            if vt[ci] == target
                sidx = ci[1]; tidx = ci[2]; break
            end
        end
        (tidx == 0 || sidx == 0) && return nothing

        # Read 3D slice at this (step, time)
        read_var(name) = begin
            v = ds[name]
            nd = ndims(v)
            if nd == 5
                Float32.(v[:, :, :, sidx, tidx])
            elseif nd == 3
                Float32.(v[:, :, :])
            else
                error("Unexpected dims for $name: $(dimnames(v))")
            end
        end

        udmf = read_var("udmf")
        ddmf = read_var("ddmf")

        has_detr = haskey(ds, "udrf") && haskey(ds, "ddrf")
        udrf = has_detr ? read_var("udrf") : nothing
        ddrf = has_detr ? read_var("ddrf") : nothing

        # Flip latitude: CDS downloads are N→S (90→-90), preprocessed files are S→N
        lats = haskey(ds, "latitude") ? ds["latitude"][:] : nothing
        needs_flip = lats !== nothing && length(lats) > 1 && lats[1] > lats[end]
        if needs_flip
            udmf = udmf[:, end:-1:1, :]
            ddmf = ddmf[:, end:-1:1, :]
            if udrf !== nothing; udrf = udrf[:, end:-1:1, :]; end
            if ddrf !== nothing; ddrf = ddrf[:, end:-1:1, :]; end
        end

        return (udmf=udmf, ddmf=ddmf, udrf=udrf, ddrf=ddrf)
    finally
        close(ds)
    end
end

"""Read surface fields from ZIP-wrapped daily file for a specific hour.
Returns NamedTuple (blh, t2m, u10, v10, sshf, ustar?) or nothing."""
function read_surface_window(sfc_dir::String, date::Date, hour::Int)
    date_str = Dates.format(date, "yyyymmdd")

    # Try direct NC first, then ZIP-wrapped
    sfc_file = joinpath(sfc_dir, "era5_surface_$(date_str).nc")
    if isfile(sfc_file)
        return _read_surface_nc_or_zip(sfc_file, date, hour)
    end
    return nothing
end

function _read_surface_nc_or_zip(path::String, date::Date, hour::Int)
    # ZipFile imported at top level
    # Check if it's a ZIP
    magic = open(path, "r") do io; read(io, 2); end
    is_zip = magic == UInt8[0x50, 0x4b]  # PK header

    if is_zip
        return _read_surface_from_zip(path, date, hour)
    else
        # Direct NC
        ds = NCDataset(path, "r")
        try
            return _extract_surface_hour(ds, date, hour)
        finally
            close(ds)
        end
    end
end

function _read_surface_from_zip(path::String, date::Date, hour::Int)
    # ZipFile imported at top level
    result = Dict{Symbol, Matrix{Float32}}()

    zr = ZipFile.Reader(path)
    try
        for f in zr.files
            # Write to temp, open as NC
            tmp = tempname() * ".nc"
            open(tmp, "w") do io; write(io, read(f)); end
            ds = NCDataset(tmp, "r")
            try
                _extract_surface_vars!(result, ds, date, hour)
            finally
                close(ds); rm(tmp; force=true)
            end
        end
    finally
        close(zr)
    end

    isempty(result) && return nothing
    return result
end

function _extract_surface_vars!(result, ds, date, hour)
    # Find time index
    tkey = haskey(ds, "valid_time") ? "valid_time" : "time"
    times = ds[tkey][:]
    target = DateTime(date) + Hour(hour)
    tidx = findfirst(t -> t == target, times)
    tidx === nothing && return

    # Check if latitude needs flipping (CDS: N→S, preprocessed: S→N)
    needs_flip = false
    for lname in ("latitude", "lat")
        if haskey(ds, lname)
            lats = ds[lname][:]
            needs_flip = length(lats) > 1 && lats[1] > lats[end]
            break
        end
    end

    for (nc_name, sym) in [("blh", :blh), ("t2m", :t2m), ("u10", :u10), ("v10", :v10),
                            ("sshf", :sshf), ("zust", :ustar), ("friction_velocity", :ustar)]
        haskey(ds, nc_name) || continue
        haskey(result, sym) && continue  # don't overwrite
        data = Float32.(ds[nc_name][:, :, tidx])
        result[sym] = needs_flip ? data[:, end:-1:1] : data
    end
end

function _extract_surface_hour(ds, date, hour)
    result = Dict{Symbol, Matrix{Float32}}()
    _extract_surface_vars!(result, ds, date, hour)
    isempty(result) ? nothing : result
end

"""Read T + Q from thermodynamics daily file for a specific hour.
Flips latitude N→S to S→N to match spectral preprocessed convention."""
function read_thermo_window(thermo_dir::String, date::Date, hour::Int)
    date_str = Dates.format(date, "yyyymmdd")
    thermo_file = joinpath(thermo_dir, "era5_thermo_ml_$(date_str).nc")
    isfile(thermo_file) || return nothing

    ds = NCDataset(thermo_file, "r")
    try
        tkey = haskey(ds, "valid_time") ? "valid_time" : "time"
        times = ds[tkey][:]
        target = DateTime(date) + Hour(hour)
        tidx = findfirst(t -> t == target, times)
        tidx === nothing && return nothing

        # Check latitude flip
        needs_flip = false
        for lname in ("latitude", "lat")
            if haskey(ds, lname)
                lats = ds[lname][:]
                needs_flip = length(lats) > 1 && lats[1] > lats[end]
                break
            end
        end

        flip3d(arr) = needs_flip && arr !== nothing ? arr[:, end:-1:1, :] : arr

        t_data = haskey(ds, "t") ? flip3d(Float32.(ds["t"][:, :, :, tidx])) : nothing
        q_data = haskey(ds, "q") ? flip3d(Float32.(ds["q"][:, :, :, tidx])) : nothing
        return (temperature=t_data, qv=q_data)
    finally
        close(ds)
    end
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

const HEADER_SIZE_V3 = 16384

# ===========================================================================
# Process one day
# ===========================================================================

function process_day(date::Date, cfg, vc_native, merged_vc, merge_map)
    FT = Float32
    Nz_native = n_levels(vc_native)
    Nz = n_levels(merged_vc)

    # Config paths
    mf_dir       = expanduser(cfg["input"]["massflux_dir"])
    conv_dir     = expanduser(get(cfg["input"], "convection_dir", ""))
    sfc_dir      = expanduser(get(cfg["input"], "surface_dir", ""))
    thermo_dir   = expanduser(get(cfg["input"], "thermo_dir", ""))
    coeff_path   = expanduser(cfg["input"]["coefficients"])
    out_dir      = expanduser(cfg["output"]["directory"])

    dt          = Float64(cfg["numerics"]["dt"])
    met_interval = Float64(cfg["numerics"]["met_interval"])
    steps_per_met = max(1, round(Int, met_interval / dt))
    half_dt = dt / 2
    level_top = Int(get(cfg["grid"], "level_top", 1))
    level_bot = Int(get(cfg["grid"], "level_bot", 137))

    # Load native A/B for dz computation
    ak_native, bk_native = let c = TOML.parsefile(coeff_path)
        Float64.(c["coefficients"]["a"]), Float64.(c["coefficients"]["b"])
    end

    # Grid from first mass flux file
    date_str = Dates.format(date, "yyyymmdd")
    month_str = Dates.format(date, "yyyymm")

    # Find the mass flux NetCDF — try daily first, then monthly
    mf_file = ""
    for pattern in [
        "massflux_era5_spectral_$(date_str)_float32.nc",
        "massflux_era5_spectral_$(month_str)_float32.nc",
        "massflux_era5_spectral_$(date_str)_hourly_float32.nc",
    ]
        f = joinpath(mf_dir, pattern)
        if isfile(f); mf_file = f; break; end
    end
    # Also try glob for any matching file
    if isempty(mf_file)
        candidates = filter(f -> endswith(f, ".nc") && contains(f, month_str), readdir(mf_dir; join=true))
        !isempty(candidates) && (mf_file = first(candidates))
    end
    isempty(mf_file) && error("No mass flux file found for $date in $mf_dir")

    ds_mf = NCDataset(mf_file, "r")
    Nx = ds_mf.dim["lon"]; Ny = ds_mf.dim["lat"]
    mf_times = ds_mf["time"][:]
    lons = Float64.(ds_mf["lon"][:])
    lats = Float64.(ds_mf["lat"][:])

    # Find time indices for this date
    date_windows = [(i, DateTime(t)) for (i, t) in enumerate(mf_times)
                    if Date(t) == date]
    Nt = length(date_windows)
    Nt > 0 || (close(ds_mf); error("No windows found for $date in $mf_file"))
    @info @sprintf("  Found %d windows for %s in %s", Nt, date, basename(mf_file))

    # Detect available external data — probe a mid-day hour (12) since forecast
    # files cover 07:00-06:00+1d (hour 0 would need previous day's file)
    probe_hour = 12
    has_conv  = !isempty(conv_dir) && read_convection_window(conv_dir, date, probe_hour) !== nothing
    has_thermo = !isempty(thermo_dir) && read_thermo_window(thermo_dir, date, probe_hour) !== nothing
    has_sfc   = !isempty(sfc_dir) && isfile(joinpath(sfc_dir, "era5_surface_$(date_str).nc"))

    @info @sprintf("  External: conv=%s, thermo=%s, surface=%s", has_conv, has_thermo, has_sfc)

    # Element counts
    n_m  = Int64(Nx) * Ny * Nz
    n_am = Int64(Nx + 1) * Ny * Nz
    n_bm = Int64(Nx) * (Ny + 1) * Nz
    n_cm = Int64(Nx) * Ny * (Nz + 1)
    n_ps = Int64(Nx) * Ny
    n_3d = Int64(Nx) * Ny * Nz
    n_2d = Int64(Nx) * Ny
    n_ifc = Int64(Nx) * Ny * (Nz + 1)

    has_qv = has_thermo  # QV comes from thermo file
    has_cmfmc = has_conv
    has_tm5conv = has_conv
    has_temperature = has_thermo
    has_surface_out = has_sfc

    n_qv    = has_qv ? n_3d : Int64(0)
    n_cmfmc = has_cmfmc ? n_ifc : Int64(0)
    n_entu  = has_tm5conv ? n_3d : Int64(0)
    n_temp  = has_temperature ? n_3d : Int64(0)
    n_sfc   = has_surface_out ? n_2d : Int64(0)

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps +
                       n_qv + n_cmfmc + 4 * n_entu + 4 * n_sfc + n_temp
    bytes_per_window = elems_per_window * sizeof(FT)
    total_bytes = Int64(HEADER_SIZE_V3) + bytes_per_window * Nt

    # Output path
    mkpath(out_dir)
    min_dp = Float64(cfg["grid"]["merge_min_thickness_Pa"])
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    bin_path = joinpath(out_dir, "era5_v3_$(date_str)_$(dp_tag)_float32.bin")

    if isfile(bin_path) && filesize(bin_path) == total_bytes
        @info "  SKIP (exists, correct size): $(basename(bin_path))"
        close(ds_mf)
        return bin_path
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), total_bytes / 1e9, Nt)

    # Build header
    header = Dict{String,Any}(
        "magic" => "MFLX", "version" => 3, "header_bytes" => HEADER_SIZE_V3,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nz_native" => Nz_native, "Nt" => Nt,
        "float_type" => "Float32", "float_bytes" => sizeof(FT),
        "window_bytes" => bytes_per_window,
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => n_qv, "n_cmfmc" => n_cmfmc,
        "n_entu" => n_entu, "n_detu" => n_entu, "n_entd" => n_entu, "n_detd" => n_entu,
        "n_pblh" => n_sfc, "n_t2m" => n_sfc, "n_ustar" => n_sfc, "n_hflux" => n_sfc,
        "n_temperature" => n_temp,
        "dt_seconds" => dt, "half_dt_seconds" => half_dt,
        "steps_per_met_window" => steps_per_met,
        "level_top" => level_top, "level_bot" => level_bot,
        "lons" => lons, "lats" => lats,
        "A_ifc" => Float64.(merged_vc.A), "B_ifc" => Float64.(merged_vc.B),
        "merge_map" => merge_map, "merge_min_thickness_Pa" => min_dp,
        "include_qv" => has_qv, "include_cmfmc" => has_cmfmc,
        "include_tm5conv" => has_tm5conv, "include_surface" => has_surface_out,
        "include_temperature" => has_temperature,
        "var_names" => ["m","am","bm","cm","ps","qv","cmfmc",
                        "entu","detu","entd","detd",
                        "pblh","t2m","ustar","hflux","temperature"],
        "date" => Dates.format(date, "yyyy-mm-dd"),
    )
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE_V3 ||
        error("Header JSON too large: $(length(header_json))")

    # Allocate merged work arrays
    m_merged  = Array{FT}(undef, Nx, Ny, Nz)
    am_merged = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_merged = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_merged = Array{FT}(undef, Nx, Ny, Nz + 1)

    # Optional merged arrays
    qv_merged    = has_qv         ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    cmfmc_merged = has_cmfmc      ? Array{FT}(undef, Nx, Ny, Nz + 1) : nothing
    entu_merged  = has_tm5conv    ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    detu_merged  = has_tm5conv    ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    entd_merged  = has_tm5conv    ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    detd_merged  = has_tm5conv    ? Array{FT}(undef, Nx, Ny, Nz)     : nothing
    temp_merged  = has_temperature ? Array{FT}(undef, Nx, Ny, Nz)    : nothing
    sfc_buf      = has_surface_out ? Array{FT}(undef, Nx, Ny)        : nothing

    # Column work arrays for ECconv_to_TMconv
    mflu = Vector{FT}(undef, Nz_native + 1)
    mfld = Vector{FT}(undef, Nz_native + 1)
    detu_rate = Vector{FT}(undef, Nz_native)
    detd_rate = Vector{FT}(undef, Nz_native)
    entu_col = Vector{FT}(undef, Nz_native)
    detu_col = Vector{FT}(undef, Nz_native)
    entd_col = Vector{FT}(undef, Nz_native)
    detd_col = Vector{FT}(undef, Nz_native)

    # Native-level arrays for TM5 conv
    entu_native = has_tm5conv ? Array{FT}(undef, Nx, Ny, Nz_native) : nothing
    detu_native = has_tm5conv ? Array{FT}(undef, Nx, Ny, Nz_native) : nothing
    entd_native = has_tm5conv ? Array{FT}(undef, Nx, Ny, Nz_native) : nothing
    detd_native = has_tm5conv ? Array{FT}(undef, Nx, Ny, Nz_native) : nothing
    cmfmc_native = has_cmfmc ? Array{FT}(undef, Nx, Ny, Nz_native + 1) : nothing

    bytes_written = Int64(0)
    open(bin_path, "w") do io
        hdr_buf = zeros(UInt8, HEADER_SIZE_V3)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf); bytes_written += HEADER_SIZE_V3

        for (win_idx, (mf_tidx, dt_val)) in enumerate(date_windows)
            t0 = time()
            hour = Dates.hour(dt_val)

            # --- Core mass fluxes from spectral NetCDF ---
            m_native  = FT.(ds_mf["m"][:, :, :, mf_tidx])
            am_native = FT.(ds_mf["am"][:, :, :, mf_tidx])
            bm_native = FT.(ds_mf["bm"][:, :, :, mf_tidx])
            cm_native = FT.(ds_mf["cm"][:, :, :, mf_tidx])
            ps_data   = Array{FT}(FT.(ds_mf["ps"][:, :, mf_tidx]))

            merge_cell_field!(m_merged, m_native, merge_map)
            merge_cell_field!(am_merged, am_native, merge_map)
            merge_cell_field!(bm_merged, bm_native, merge_map)

            # Zero am at pole rows and bm at pole faces — the spectral SHT
            # produces extreme values at poles (U/cos(lat) → ∞).
            # Must zero BEFORE cm recomputation so continuity is consistent.
            @views am_merged[:, 1, :]    .= zero(FT)   # am at south pole row
            @views am_merged[:, Ny, :]   .= zero(FT)   # am at north pole row
            @views bm_merged[:, 1, :]    .= zero(FT)   # bm at south pole face
            @views bm_merged[:, Ny+1, :] .= zero(FT)   # bm at north pole face

            # Recompute cm from merged horizontal divergence using TM5's
            # hybrid-coordinate formula with B-coefficient correction.
            # This naturally gives cm=0 at both TOA and surface.
            recompute_cm_from_divergence!(cm_merged, am_merged, bm_merged, m_merged;
                                          B_ifc=merged_vc.B)

            bytes_written += write_array!(io, m_merged)
            bytes_written += write_array!(io, am_merged)
            bytes_written += write_array!(io, bm_merged)
            bytes_written += write_array!(io, cm_merged)
            bytes_written += write_array!(io, ps_data)

            # --- QV from thermo file ---
            if has_qv
                thermo = read_thermo_window(thermo_dir, date, hour)
                if thermo !== nothing && thermo.qv !== nothing
                    merge_qv!(qv_merged, thermo.qv, m_native, merge_map)
                else
                    fill!(qv_merged, zero(FT))
                end
                bytes_written += write_array!(io, qv_merged)
            end

            # --- Convection ---
            if has_conv
                conv = read_convection_window(conv_dir, date, hour)
                if conv !== nothing
                    # CMFMC = UDMF + DDMF at interfaces
                    cmfmc_native[:, :, 1] .= zero(FT)
                    @inbounds for k in 1:Nz_native
                        @views cmfmc_native[:, :, k+1] .= conv.udmf[:, :, k] .+ conv.ddmf[:, :, k]
                    end
                    merge_interface_field!(cmfmc_merged, cmfmc_native, merge_map)
                    bytes_written += write_array!(io, cmfmc_merged)

                    # TM5 conv: process column by column
                    has_detr = conv.udrf !== nothing
                    for j in 1:Ny, i in 1:Nx
                        mflu[1] = zero(FT); mfld[1] = zero(FT)
                        @inbounds for k in 1:Nz_native
                            mflu[k+1] = conv.udmf[i, j, k]
                            mfld[k+1] = conv.ddmf[i, j, k]
                        end
                        if has_detr
                            @inbounds for k in 1:Nz_native
                                detu_rate[k] = conv.udrf[i, j, k]
                                detd_rate[k] = conv.ddrf[i, j, k]
                            end
                            dz = compute_dz_hydrostatic(ps_data[i, j], Nz_native,
                                                         ak_native, bk_native)
                            ecconv_to_tmconv!(entu_col, detu_col, entd_col, detd_col,
                                              mflu, mfld, detu_rate, detd_rate, dz, Nz_native)
                        else
                            massflux_to_tmconv!(entu_col, detu_col, entd_col, detd_col,
                                                mflu, mfld, Nz_native)
                        end
                        @inbounds for k in 1:Nz_native
                            entu_native[i, j, k] = entu_col[k]
                            detu_native[i, j, k] = detu_col[k]
                            entd_native[i, j, k] = entd_col[k]
                            detd_native[i, j, k] = detd_col[k]
                        end
                    end
                    # Merge TM5 conv fields (sum, like mass fluxes)
                    merge_cell_field!(entu_merged, entu_native, merge_map)
                    merge_cell_field!(detu_merged, detu_native, merge_map)
                    merge_cell_field!(entd_merged, entd_native, merge_map)
                    merge_cell_field!(detd_merged, detd_native, merge_map)
                else
                    fill!(cmfmc_merged, zero(FT)); fill!(entu_merged, zero(FT))
                    fill!(detu_merged, zero(FT)); fill!(entd_merged, zero(FT))
                    fill!(detd_merged, zero(FT))
                    bytes_written += write_array!(io, cmfmc_merged)
                end
                bytes_written += write_array!(io, entu_merged)
                bytes_written += write_array!(io, detu_merged)
                bytes_written += write_array!(io, entd_merged)
                bytes_written += write_array!(io, detd_merged)
            end

            # --- Surface fields ---
            if has_surface_out
                sfc = read_surface_window(sfc_dir, date, hour)
                for (sym, default) in [(:blh, 500.0f0), (:t2m, 280.0f0),
                                        (:ustar, 0.3f0), (:sshf, 0.0f0)]
                    if sfc !== nothing && haskey(sfc, sym)
                        sfc_buf .= sfc[sym]
                        # De-accumulate SSHF → instantaneous heat flux (W/m²)
                        # ERA5 SSHF is accumulated J/m² over the hour → divide by 3600
                        if sym == :sshf
                            sfc_buf ./= FT(met_interval)
                        end
                    elseif sfc !== nothing && sym == :ustar && haskey(sfc, :u10) && haskey(sfc, :v10)
                        # Derive u* from U10/V10: u* ≈ κ√(u10²+v10²)/ln(10/z0)
                        # With z0=0.01 m (generic), κ=0.4: u* ≈ 0.058 × |V10|
                        @views sfc_buf .= FT(0.058) .* sqrt.(sfc[:u10].^2 .+ sfc[:v10].^2)
                    else
                        fill!(sfc_buf, FT(default))
                    end
                    bytes_written += write_array!(io, sfc_buf)
                end
            end

            # --- Temperature ---
            if has_temperature
                thermo = has_qv ? thermo : read_thermo_window(thermo_dir, date, hour)
                if thermo !== nothing && thermo.temperature !== nothing
                    merge_qv!(temp_merged, thermo.temperature, m_native, merge_map)
                else
                    fill!(temp_merged, zero(FT))
                end
                bytes_written += write_array!(io, temp_merged)
            end

            t_win = round(time() - t0, digits=2)
            if win_idx <= 3 || win_idx == Nt || win_idx % 8 == 0
                @info @sprintf("  Window %d/%d (hour %02d)  %.2fs  [%.1f GB]",
                               win_idx, Nt, hour, t_win, bytes_written / 1e9)
            end
        end
        flush(io)
    end

    close(ds_mf)

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB)", basename(bin_path), actual / 1e9)
    actual == total_bytes ||
        error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d", total_bytes, actual))

    return bin_path
end

# ===========================================================================
# Main
# ===========================================================================

function main()
    if isempty(ARGS)
        println("""
        Unified ERA5 daily preprocessor → v3 binary

        Usage:
          julia -t8 --project=. $(PROGRAM_FILE) config.toml [--day 2021-12-01]

        Config TOML example: config/preprocessing/catrine_era5_daily_v3.toml
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

    level_top = Int(get(cfg["grid"], "level_top", 1))
    level_bot = Int(get(cfg["grid"], "level_bot", 137))
    min_dp = Float64(cfg["grid"]["merge_min_thickness_Pa"])
    coeff_path = expanduser(cfg["input"]["coefficients"])

    vc_native = load_era5_vertical_coordinate(coeff_path, level_top, level_bot)
    merged_vc, merge_map = merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
    Nz_native = n_levels(vc_native)
    Nz_merged = n_levels(merged_vc)
    @info @sprintf("Level merging: %d → %d (min_dp=%.0f Pa)", Nz_native, Nz_merged, min_dp)

    # Determine date range from mass flux files
    mf_dir = expanduser(cfg["input"]["massflux_dir"])
    mf_files = sort(filter(f -> endswith(f, ".nc"), readdir(mf_dir; join=true)))
    isempty(mf_files) && error("No .nc files in $mf_dir")

    # Collect all unique dates across mass flux files
    all_dates = Date[]
    for f in mf_files
        ds = NCDataset(f, "r")
        times = ds["time"][:]
        close(ds)
        for t in times
            d = Date(t)
            d ∉ all_dates && push!(all_dates, d)
        end
    end
    sort!(all_dates)

    if day_filter !== nothing
        all_dates = filter(==(day_filter), all_dates)
        isempty(all_dates) && error("Date $day_filter not found in mass flux files")
    end

    @info @sprintf("Processing %d days: %s to %s", length(all_dates),
                   first(all_dates), last(all_dates))
    @info @sprintf("Output: %s", expanduser(cfg["output"]["directory"]))

    for (i, date) in enumerate(all_dates)
        @info @sprintf("[%d/%d] %s", i, length(all_dates), date)
        process_day(date, cfg, vc_native, merged_vc, merge_map)
    end

    @info "All done!"
end

main()
