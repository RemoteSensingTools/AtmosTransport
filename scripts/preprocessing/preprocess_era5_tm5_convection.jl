#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Preprocess ERA5 fields into TM5 convection inputs (entu, detu, entd, detd)
#
# Port of TM5's ECconv_to_TMconv (deps/tm5/base/src/phys_convec_ec2tm.F90)
#
# Reads:
#   - UDMF/DDMF files (updraft/downdraft mass flux at interfaces, from download_era5_cmfmc.py)
#   - UDRF/DDRF files (detrainment rates at full levels, from download_era5_detrainment.py)
#
# Computes dz from ERA5 hybrid coefficients + temperature/humidity or from
# geopotential if available. Converts detrainment rates (kg/m3/s) to fluxes
# (kg/m2/s) via dz, then derives entrainment from mass budget closure:
#   entu(k) = mflu(k-1) - mflu(k) + detu(k)
#
# Appends 4 variables (entu, detu, entd, detd) to the preprocessed mass flux
# NetCDF file, at full levels (lon, lat, lev, time).
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_era5_tm5_convection.jl \
#       <massflux.nc> <cmfmc_dir> <detr_dir> [--geopotential <geopotential_dir>]
#
# Level convention: top-to-bottom (k=1=TOA, k=137=surface), matching our
# preprocessed files. The TM5 matrix convection code handles the reversal
# internally.
# ---------------------------------------------------------------------------

using NCDatasets
using Dates
using Printf
using TOML

"""
    ecconv_to_tmconv!(entu, detu, entd, detd, mflu, mfld, detu_rate, detd_rate, dz, Nz)

Port of TM5's ECconv_to_TMconv (phys_convec_ec2tm.F90, 'd' variant for top-to-bottom).

All arrays are 1D column vectors in top-to-bottom convention (k=1=TOA).

Input:
- mflu[0:Nz]: updraft mass flux at half-levels (interfaces), positive upward
  - mflu[1] = TOA interface (should be ~0)
  - mflu[Nz+1] = surface interface (should be ~0)
  Note: Julia 1-indexed, so mflu[1] = f(0) in Fortran
- mfld[0:Nz]: downdraft mass flux at half-levels, negative downward
- detu_rate[1:Nz]: updraft detrainment RATE at full levels [kg/m3/s]
- detd_rate[1:Nz]: downdraft detrainment RATE at full levels [kg/m3/s]
- dz[1:Nz]: layer thickness [m]

Output (overwritten):
- entu[1:Nz], detu[1:Nz], entd[1:Nz], detd[1:Nz]: [kg/m2/s]
"""
function ecconv_to_tmconv!(entu::Vector{FT}, detu_out::Vector{FT},
                           entd::Vector{FT}, detd_out::Vector{FT},
                           mflu_in::Vector{FT}, mfld_in::Vector{FT},
                           detu_rate::Vector{FT}, detd_rate::Vector{FT},
                           dz::Vector{FT}, Nz::Int) where FT

    # Local copies to avoid modifying inputs
    mflu = copy(mflu_in)   # 1:Nz+1 (interfaces, 1=TOA, Nz+1=surface)
    detu = copy(detu_rate)  # 1:Nz (full levels)
    mfld = copy(mfld_in)
    detd = copy(detd_rate)

    # Step 1: Clean small values from GRIB noise
    @inbounds for k in 1:Nz+1
        if mflu[k] < FT(1e-6);  mflu[k] = zero(FT); end
        if mfld[k] > FT(-1e-6); mfld[k] = zero(FT); end
    end
    @inbounds for k in 1:Nz
        if detu[k] < FT(1e-10); detu[k] = zero(FT); end
        if detd[k] < FT(1e-10); detd[k] = zero(FT); end
    end

    # Step 2: Integrate detrainment rates over layer height
    # detu: kg/m3/s * m => kg/m2/s
    @inbounds for k in 1:Nz
        detu[k] *= dz[k]
        detd[k] *= dz[k]
    end

    # Step 3: Find updraft top (first level from TOA with mflu > 0)
    # In top-to-bottom convention, scan from k=1 (TOA) downward
    uptop = 0
    @inbounds for k in 1:Nz
        if mflu[k+1] > zero(FT)  # mflu[k+1] = flux at bottom of level k
            uptop = k
            break
        end
    end

    # Step 4: Compute updraft entrainment from mass budget
    fill!(entu, zero(FT))
    if uptop > 0
        # Zero above uptop
        @inbounds for k in 1:uptop-1
            entu[k] = zero(FT)
            detu[k] = zero(FT)
        end
        # From uptop to surface:
        # entu(k) = flux_out_above - flux_in_below + detu(k)
        # In top-to-bottom: mflu[k] is interface above level k, mflu[k+1] is below
        @inbounds for k in uptop:Nz
            entu[k] = mflu[k] - mflu[k+1] + detu[k]
        end
    else
        fill!(detu, zero(FT))
    end

    # Step 5: Find downdraft top (first level from TOA with mfld < 0)
    dotop = 0
    @inbounds for k in 1:Nz
        if mfld[k+1] < zero(FT)
            dotop = k
            break
        end
    end

    # Step 6: Compute downdraft entrainment
    fill!(entd, zero(FT))
    if dotop > 0
        @inbounds for k in 1:dotop-1
            detd[k] = zero(FT)
            entd[k] = zero(FT)
        end
        @inbounds for k in dotop:Nz
            entd[k] = mfld[k] - mfld[k+1] + detd[k]
        end
    else
        fill!(detd, zero(FT))
    end

    # Step 7: Fix negative values by redistribution
    @inbounds for k in 1:Nz
        if entu[k] < zero(FT)
            detu[k] -= entu[k]
            entu[k] = zero(FT)
        end
        if detu[k] < zero(FT)
            entu[k] -= detu[k]
            detu[k] = zero(FT)
        end
        if entd[k] < zero(FT)
            detd[k] -= entd[k]
            entd[k] = zero(FT)
        end
        if detd[k] < zero(FT)
            entd[k] -= detd[k]
            detd[k] = zero(FT)
        end
    end

    # Write output
    detu_out .= detu
    detd_out .= detd
    return nothing
end

"""
    massflux_to_tmconv!(entu, detu, entd, detd, mflu, mfld, Nz)

Derive TM5 entrainment/detrainment from mass flux profile alone
(no separate detrainment rate files needed).

Uses the monotonic decomposition:
- Where updraft mass flux increases: pure entrainment (entu > 0, detu = 0)
- Where updraft mass flux decreases: pure detrainment (entu = 0, detu > 0)
- Same for downdraft

This is a valid approximation when ERA5 detrainment rate fields (params
214/215) are not available. The TM5 matrix convection will still produce
physically reasonable transport, though with slightly different updraft
structure than the full ECconv_to_TMconv approach.

All arrays use top-to-bottom convention (k=1=TOA).
mflu/mfld are at interfaces (1:Nz+1), entu/detu/entd/detd at full levels (1:Nz).
"""
function massflux_to_tmconv!(entu::Vector{FT}, detu::Vector{FT},
                             entd::Vector{FT}, detd::Vector{FT},
                             mflu_in::Vector{FT}, mfld_in::Vector{FT},
                             Nz::Int) where FT
    # Clean small values
    mflu = copy(mflu_in)
    mfld = copy(mfld_in)
    @inbounds for k in 1:Nz+1
        if mflu[k] < FT(1e-6);  mflu[k] = zero(FT); end
        if mfld[k] > FT(-1e-6); mfld[k] = zero(FT); end
    end

    # Updraft: mass budget at each level
    # In top-to-bottom: mflu[k] = interface above level k, mflu[k+1] = below
    # Net change = mflu[k] - mflu[k+1] (positive = detraining into environment)
    @inbounds for k in 1:Nz
        net = mflu[k] - mflu[k+1]   # > 0 means updraft weakens (detrainment)
        if net > zero(FT)
            entu[k] = zero(FT)
            detu[k] = net
        else
            entu[k] = -net   # mass flux strengthened = entrainment
            detu[k] = zero(FT)
        end
    end

    # Downdraft: mfld is negative, |mfld| is the downward flux
    # Net change = mfld[k] - mfld[k+1]
    # mfld gets more negative going down = downdraft strengthening = entrainment
    @inbounds for k in 1:Nz
        net = mfld[k] - mfld[k+1]   # < 0 means downdraft strengthens
        if net < zero(FT)
            entd[k] = -net    # entrainment (positive)
            detd[k] = zero(FT)
        else
            entd[k] = zero(FT)
            detd[k] = net     # detrainment (positive)
        end
    end

    return nothing
end

"""
    compute_dz_hydrostatic(ps, Nz, ak, bk)

Compute layer thickness dz [m] from hybrid coefficients using
hydrostatic approximation with reference temperature.

Uses dz = R*T_ref/g * dln(p) with T_ref = 260 K (tropospheric mean).
"""
function compute_dz_hydrostatic(ps::FT, Nz::Int,
                                ak::Vector{Float64}, bk::Vector{Float64}) where FT
    R = 287.058   # dry air gas constant [J/kg/K]
    g = 9.80665   # gravitational acceleration [m/s2]
    T_ref = 260.0 # reference temperature [K]

    dz = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        p_top = FT(ak[k] + bk[k] * ps)
        p_bot = FT(ak[k+1] + bk[k+1] * ps)
        p_mid = FT(0.5) * (p_top + p_bot)
        dp = p_bot - p_top
        dz[k] = FT(R * T_ref / g * dp / p_mid)
    end
    return dz
end

"""Read ERA5 hybrid coefficients from TOML config."""
function load_era5_hybrid_coefficients()
    toml_path = joinpath(@__DIR__, "..", "..", "config", "era5_L137_coefficients.toml")
    if !isfile(toml_path)
        error("Cannot find ERA5 hybrid coefficients: $toml_path")
    end
    d = TOML.parsefile(toml_path)
    ak = Float64.(d["coefficients"]["a"])
    bk = Float64.(d["coefficients"]["b"])
    return ak, bk
end

"""
    read_cmfmc_column(ds_cmfmc, udmf_name, ddmf_name, tidx, i, j, Nz, needs_flip)

Read UDMF and DDMF for a single column from CMFMC NetCDF and return as
interface-level arrays mflu[1:Nz+1] and mfld[1:Nz+1].
"""
function detect_variable_names(ds, pairs_to_check)
    result = Dict{Symbol, String}()
    for (key, candidates) in pairs_to_check
        for vname in keys(ds)
            if lowercase(vname) in candidates
                result[key] = vname
                break
            end
        end
    end
    return result
end

function main()
    if length(ARGS) < 2
        println("""Usage: preprocess_era5_tm5_convection.jl <massflux.nc> <cmfmc_dir> [<detr_dir>]

Computes TM5 convection fields (entu, detu, entd, detd) and appends them
to the preprocessed mass flux file <massflux.nc>.

Two modes:
  With <detr_dir>:    Full ECconv_to_TMconv using UDMF/DDMF + UDRF/DDRF
  Without <detr_dir>: Mass-flux-only derivation from UDMF/DDMF profiles
                      (monotonic decomposition, valid approximation)
""")
        return
    end

    mf_path = ARGS[1]
    cmfmc_dir = expanduser(ARGS[2])
    detr_dir = length(ARGS) >= 3 ? expanduser(ARGS[3]) : nothing
    use_detr = detr_dir !== nothing

    if use_detr
        @info "Mode: full ECconv_to_TMconv (with detrainment rates)"
    else
        @info "Mode: mass-flux-only derivation (no detrainment rate files)"
    end

    # Read mass flux file metadata
    ds = NCDataset(mf_path, "r")
    mf_times = ds["time"][:]
    Nlon = ds.dim["lon"]
    Nlat = ds.dim["lat"]
    Nt = length(mf_times)
    # Get surface pressure for dz computation
    has_ps = haskey(ds, "ps")
    ps_data = has_ps ? Float32.(ds["ps"][:, :, :]) : nothing
    close(ds)

    Nz = 137  # ERA5 model levels
    @info "Mass flux file: $(Nlon)x$(Nlat), $Nt windows"

    # Load hybrid coefficients for dz computation
    ak, bk = load_era5_hybrid_coefficients()
    @assert length(ak) == Nz + 1 "Expected $(Nz+1) hybrid coefficients, got $(length(ak))"

    # Prepare output arrays
    entu_out = zeros(Float32, Nlon, Nlat, Nz, Nt)
    detu_out = zeros(Float32, Nlon, Nlat, Nz, Nt)
    entd_out = zeros(Float32, Nlon, Nlat, Nz, Nt)
    detd_out = zeros(Float32, Nlon, Nlat, Nz, Nt)

    # Column work arrays
    mflu = Vector{Float32}(undef, Nz + 1)   # interfaces
    mfld = Vector{Float32}(undef, Nz + 1)
    detu_rate = Vector{Float32}(undef, Nz)   # full levels
    detd_rate = Vector{Float32}(undef, Nz)
    entu_col = Vector{Float32}(undef, Nz)
    detu_col = Vector{Float32}(undef, Nz)
    entd_col = Vector{Float32}(undef, Nz)
    detd_col = Vector{Float32}(undef, Nz)

    matched = 0
    for (ti, dt) in enumerate(mf_times)
        date = Date(dt)
        hour = Dates.hour(dt)
        date_str = @sprintf("%04d%02d%02d", year(date), month(date), day(date))

        # Find CMFMC file for this date
        cmfmc_file = joinpath(cmfmc_dir, "era5_cmfmc_$(date_str).nc")
        if !isfile(cmfmc_file)
            @warn "No CMFMC file for $date ($cmfmc_file)"
            continue
        end

        # Optionally find detrainment file
        ds_detr = nothing
        if use_detr
            detr_file = joinpath(detr_dir, "era5_detr_$(date_str).nc")
            if !isfile(detr_file)
                @warn "No detrainment file for $date ($detr_file), using mass-flux-only"
            else
                ds_detr = NCDataset(detr_file, "r")
            end
        end

        ds_cmfmc = NCDataset(cmfmc_file, "r")
        try
            # Detect CMFMC variable names
            cmfmc_vars = detect_variable_names(ds_cmfmc,
                [:udmf => ("udmf", "mumf", "mu", "p71.162", "var71"),
                 :ddmf => ("ddmf", "mdmf", "md", "p72.162", "var72")])
            if !haskey(cmfmc_vars, :udmf) || !haskey(cmfmc_vars, :ddmf)
                @warn "Cannot find UDMF/DDMF in $cmfmc_file" keys=keys(ds_cmfmc)
                continue
            end

            # Find time index
            times_key = haskey(ds_cmfmc, "valid_time") ? "valid_time" : "time"
            times_cmfmc = ds_cmfmc[times_key][:]
            tidx_cmfmc = findfirst(t -> Dates.hour(t) == hour && Date(t) == date, times_cmfmc)
            if tidx_cmfmc === nothing
                @warn "Hour $hour not found in CMFMC file for $date"
                continue
            end

            # Read 3D fields helper
            function read_3d(ds, varname, tidx)
                var = ds[varname]
                nd = length(dimnames(var))
                nd == 4 ? Float32.(var[:, :, :, tidx]) :
                nd == 3 ? Float32.(var[:, :, :]) :
                error("Unexpected dims for $varname: $(dimnames(var))")
            end

            function ensure_lonlatlev(arr, Nz_expected)
                sz = size(arr)
                lev_dim = findfirst(d -> d == Nz_expected, sz)
                lev_dim == 1 ? permutedims(arr, (3, 2, 1)) :
                lev_dim == 3 ? arr :
                error("Cannot identify level dim in shape $sz")
            end

            udmf_3d = ensure_lonlatlev(read_3d(ds_cmfmc, cmfmc_vars[:udmf], tidx_cmfmc), Nz)
            ddmf_3d = ensure_lonlatlev(read_3d(ds_cmfmc, cmfmc_vars[:ddmf], tidx_cmfmc), Nz)

            # Read detrainment if available
            udrf_3d = nothing
            ddrf_3d = nothing
            if ds_detr !== nothing
                detr_vars = detect_variable_names(ds_detr,
                    [:udrf => ("udrf", "p214.162", "var214"),
                     :ddrf => ("ddrf", "p215.162", "var215")])
                if haskey(detr_vars, :udrf) && haskey(detr_vars, :ddrf)
                    times_key_d = haskey(ds_detr, "valid_time") ? "valid_time" : "time"
                    times_detr = ds_detr[times_key_d][:]
                    tidx_detr = findfirst(t -> Dates.hour(t) == hour && Date(t) == date, times_detr)
                    if tidx_detr !== nothing
                        udrf_3d = ensure_lonlatlev(read_3d(ds_detr, detr_vars[:udrf], tidx_detr), Nz)
                        ddrf_3d = ensure_lonlatlev(read_3d(ds_detr, detr_vars[:ddrf], tidx_detr), Nz)
                    end
                end
            end

            # Check if latitude needs flipping (N->S to S->N)
            needs_flip = false
            if haskey(ds_cmfmc, "latitude")
                lats = ds_cmfmc["latitude"][:]
                needs_flip = length(lats) > 1 && lats[1] > lats[end]
            end
            if needs_flip
                udmf_3d = udmf_3d[:, end:-1:1, :]
                ddmf_3d = ddmf_3d[:, end:-1:1, :]
                if udrf_3d !== nothing
                    udrf_3d = udrf_3d[:, end:-1:1, :]
                    ddrf_3d = ddrf_3d[:, end:-1:1, :]
                end
            end

            has_detr_3d = udrf_3d !== nothing
            mode_str = has_detr_3d ? "full" : "mflux-only"
            @info @sprintf("  Window %d/%d (%s): %dx%dx%d [%s]",
                           ti, Nt, dt, Nlon, Nlat, Nz, mode_str)

            # Process each column
            for j in 1:Nlat, i in 1:Nlon
                # Build interface-level mass flux arrays (Nz+1 interfaces)
                mflu[1] = Float32(0)  # TOA
                mfld[1] = Float32(0)
                @inbounds for k in 1:Nz
                    mflu[k+1] = udmf_3d[i, j, k]
                    mfld[k+1] = ddmf_3d[i, j, k]
                end

                if has_detr_3d
                    # Full ECconv_to_TMconv with detrainment rates
                    @inbounds for k in 1:Nz
                        detu_rate[k] = udrf_3d[i, j, k]
                        detd_rate[k] = ddrf_3d[i, j, k]
                    end
                    ps = has_ps ? ps_data[i, j, ti] : Float32(101325)
                    dz = compute_dz_hydrostatic(ps, Nz, ak, bk)
                    ecconv_to_tmconv!(entu_col, detu_col, entd_col, detd_col,
                                      mflu, mfld, detu_rate, detd_rate, dz, Nz)
                else
                    # Mass-flux-only derivation (monotonic decomposition)
                    massflux_to_tmconv!(entu_col, detu_col, entd_col, detd_col,
                                        mflu, mfld, Nz)
                end

                # Store
                @inbounds for k in 1:Nz
                    entu_out[i, j, k, ti] = entu_col[k]
                    detu_out[i, j, k, ti] = detu_col[k]
                    entd_out[i, j, k, ti] = entd_col[k]
                    detd_out[i, j, k, ti] = detd_col[k]
                end
            end

            matched += 1
        finally
            close(ds_cmfmc)
            ds_detr !== nothing && close(ds_detr)
        end
    end

    @info "Matched $matched/$Nt windows"

    if matched == 0
        @error "No data matched! Check file dates and directories."
        return
    end

    # Statistics
    @info @sprintf("entu: max=%.4e, mean=%.6e kg/m2/s",
                   maximum(entu_out), sum(entu_out) / length(entu_out))
    @info @sprintf("detu: max=%.4e, mean=%.6e kg/m2/s",
                   maximum(detu_out), sum(detu_out) / length(detu_out))
    @info @sprintf("entd: max=%.4e, mean=%.6e kg/m2/s",
                   maximum(entd_out), sum(entd_out) / length(entd_out))
    @info @sprintf("detd: max=%.4e, mean=%.6e kg/m2/s",
                   maximum(detd_out), sum(detd_out) / length(detd_out))

    # Write to mass flux file
    @info "Appending entu/detu/entd/detd to $mf_path ..."
    NCDataset(mf_path, "a") do ds
        attrib_common = Dict("units" => "kg/m2/s",
                             "source" => "ERA5 params 71.162+72.162+214.162+215.162 via ECconv_to_TMconv")

        for (name, data, desc) in [
            ("entu", entu_out, "Updraft entrainment rate"),
            ("detu", detu_out, "Updraft detrainment rate"),
            ("entd", entd_out, "Downdraft entrainment rate"),
            ("detd", detd_out, "Downdraft detrainment rate")]

            if haskey(ds, name)
                @info "  Variable $name already exists, overwriting"
                ds[name][:, :, :, :] = data
            else
                defVar(ds, name, data, ("lon", "lat", "lev", "time");
                       attrib=merge(attrib_common, Dict("long_name" => desc)))
            end
        end
    end

    @info "Done! Added entu/detu/entd/detd ($(Nlon)x$(Nlat)x$(Nz)x$(Nt)) to mass flux file."
    @info "To use: set [convection] type = \"tm5\" in your run config."
end

main()
