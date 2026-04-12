#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Merge ERA5 convective mass flux into preprocessed mass flux NetCDF
#
# Reads daily UDMF/DDMF files (from download_era5_cmfmc.py), computes
# CMFMC = UDMF + DDMF at model-level interfaces, and writes as
# `conv_mass_flux(lon, lat, lev_w, time)` into the preprocessed file.
#
# ERA5 model levels: 1 (top) to 137 (surface)
# CMFMC interfaces:  1 (TOA, =0) to 138 (surface, =0)
# Convention: positive = upward mass flux [kg/m²/s]
#
# UDMF (param 71.162): updraft mass flux, positive upward
# DDMF (param 72.162): downdraft mass flux, negative (downward)
# CMFMC = UDMF + DDMF (net convective mass flux)
#
# Usage:
#   julia --project=. scripts/preprocessing/merge_era5_cmfmc_to_massflux.jl \
#       /temp1/.../massflux_era5_spectral_202112_float32.nc \
#       ~/data/metDrivers/era5/cmfmc_catrine/
# ---------------------------------------------------------------------------

using NCDatasets
using Dates
using Printf

function main()
    if length(ARGS) < 2
        error("Usage: merge_era5_cmfmc_to_massflux.jl <massflux.nc> <cmfmc_dir>")
    end
    mf_path = ARGS[1]
    cmfmc_dir = expanduser(ARGS[2])

    # Read mass flux file metadata
    ds = NCDataset(mf_path, "r")
    mf_times = ds["time"][:]
    Nlon = ds.dim["lon"]
    Nlat = ds.dim["lat"]
    Nlev_w = ds.dim["lev_w"]  # 138 = 137 + 1 interfaces
    Nt = length(mf_times)
    close(ds)

    @info "Mass flux file: $(Nlon)×$(Nlat), lev_w=$Nlev_w, $Nt windows"
    @info "Time range: $(mf_times[1]) — $(mf_times[end])"
    Nz = Nlev_w - 1  # 137

    # Prepare output array: (lon, lat, lev_w, time)
    cmfmc_out = zeros(Float32, Nlon, Nlat, Nlev_w, Nt)

    # Process each time step
    matched = 0
    for (ti, dt) in enumerate(mf_times)
        date = Date(dt)
        hour = Dates.hour(dt)
        date_str = @sprintf("%04d%02d%02d", year(date), month(date), day(date))

        # Find CMFMC file for this date
        cmfmc_file = joinpath(cmfmc_dir, "era5_cmfmc_$(date_str).nc")
        if !isfile(cmfmc_file)
            @warn "No CMFMC file for $date ($cmfmc_file), filling with zeros"
            continue
        end

        NCDataset(cmfmc_file, "r") do ds_cmfmc
            # Detect variable names (cfgrib may use different names)
            udmf_name = nothing
            ddmf_name = nothing
            for vname in keys(ds_cmfmc)
                vl = lowercase(vname)
                if vl in ("udmf", "mu", "p71.162", "var71")
                    udmf_name = vname
                elseif vl in ("ddmf", "md", "p72.162", "var72")
                    ddmf_name = vname
                end
            end

            if udmf_name === nothing || ddmf_name === nothing
                @warn "Cannot find UDMF/DDMF variables in $cmfmc_file" keys=keys(ds_cmfmc)
                return
            end

            # Find matching time index in CMFMC file
            if haskey(ds_cmfmc, "valid_time")
                times_cmfmc = ds_cmfmc["valid_time"][:]
            elseif haskey(ds_cmfmc, "time")
                times_cmfmc = ds_cmfmc["time"][:]
            else
                @warn "No time variable found in $cmfmc_file"
                return
            end

            tidx = findfirst(t -> Dates.hour(t) == hour && Date(t) == date,
                             times_cmfmc)
            if tidx === nothing
                @warn "Hour $hour not found in CMFMC file for $date"
                return
            end

            # Read UDMF and DDMF: (lon, lat, level) or (lon, lat, level, time)
            udmf_var = ds_cmfmc[udmf_name]
            ddmf_var = ds_cmfmc[ddmf_name]

            # Handle different dimension orderings from cfgrib
            udmf_dims = dimnames(udmf_var)
            ndim = length(udmf_dims)

            if ndim == 4  # (lon, lat, level, time) or similar
                udmf_3d = Float32.(udmf_var[:, :, :, tidx])
                ddmf_3d = Float32.(ddmf_var[:, :, :, tidx])
            elseif ndim == 3  # (lon, lat, level) — single timestep
                udmf_3d = Float32.(udmf_var[:, :, :])
                ddmf_3d = Float32.(ddmf_var[:, :, :])
            else
                @warn "Unexpected dimensions for UDMF: $udmf_dims"
                return
            end

            # Determine data layout
            sz = size(udmf_3d)
            @info "  Window $ti/$Nt ($dt): UDMF $(size(udmf_3d)), DDMF $(size(ddmf_3d))"

            # Check if latitude needs flipping (N→S to S→N)
            # ERA5 raw data is N→S (90 to -90), preprocessed file is S→N (-90 to 90)
            needs_flip = false
            if haskey(ds_cmfmc, "latitude")
                lats = ds_cmfmc["latitude"][:]
                if length(lats) > 1 && lats[1] > lats[end]
                    needs_flip = true
                end
            end

            # Identify which dimension is levels (should be 137)
            lev_dim = findfirst(d -> d == Nz, sz)
            if lev_dim === nothing
                @warn "Cannot identify level dimension (expected $Nz): size=$sz"
                return
            end

            # Map to (lon, lat, lev) if needed
            if lev_dim == 3
                # Already (lon, lat, lev) — standard
                udmf = udmf_3d
                ddmf = ddmf_3d
            elseif lev_dim == 1
                # (lev, lat, lon) — cfgrib default
                udmf = permutedims(udmf_3d, (3, 2, 1))
                ddmf = permutedims(ddmf_3d, (3, 2, 1))
            else
                @warn "Unexpected level dimension position: $lev_dim in $sz"
                return
            end

            if needs_flip
                udmf = udmf[:, end:-1:1, :]
                ddmf = ddmf[:, end:-1:1, :]
            end

            # Compute CMFMC at interfaces:
            # Interface 1 (TOA) = 0
            # Interface k+1 = UDMF[k] + DDMF[k] for k=1..137
            # Interface 138 (surface) should naturally be ~0
            @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
                cmfmc_out[i, j, k + 1, ti] = udmf[i, j, k] + ddmf[i, j, k]
            end
            # TOA and surface BCs are already zero from initialization

            matched += 1
        end
    end

    @info "Matched $matched/$Nt windows"

    if matched == 0
        @error "No CMFMC data matched! Check file dates and directory."
        return
    end

    # Check that surface level is near-zero
    sfc_max = maximum(abs, @view cmfmc_out[:, :, Nlev_w, :])
    @info @sprintf("Surface CMFMC max = %.2e kg/m²/s (should be ~0)", sfc_max)
    if sfc_max > 0.01
        @warn "Surface CMFMC is unexpectedly large — check level ordering"
    end

    # Report statistics
    interior = @view cmfmc_out[:, :, 2:Nz, :]
    @info @sprintf("Interior CMFMC: min=%.4f max=%.4f mean=%.6f kg/m²/s",
                   minimum(interior), maximum(interior), sum(interior) / length(interior))

    # Write to mass flux file in append mode
    @info "Appending conv_mass_flux to $mf_path ..."
    NCDataset(mf_path, "a") do ds
        if haskey(ds, "conv_mass_flux")
            @info "  Variable conv_mass_flux already exists, overwriting"
            ds["conv_mass_flux"][:, :, :, :] = cmfmc_out
        else
            defVar(ds, "conv_mass_flux",
                   cmfmc_out, ("lon", "lat", "lev_w", "time");
                   attrib=Dict(
                       "long_name" => "Net convective mass flux (UDMF+DDMF)",
                       "units" => "kg/m2/s",
                       "positive" => "up",
                       "source" => "ERA5 model-level params 71.162 + 72.162"))
        end
    end

    @info "Done! Added conv_mass_flux ($(Nlon)×$(Nlat)×$(Nlev_w)×$(Nt)) to mass flux file."
end

main()
