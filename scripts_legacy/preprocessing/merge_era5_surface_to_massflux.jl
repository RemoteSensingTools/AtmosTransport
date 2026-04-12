#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Merge ERA5 surface fields into preprocessed mass flux NetCDF
#
# Adds PBLH, T2M, USTAR, HFLUX to the preprocessed spectral mass flux file.
# The surface data comes from separate ERA5 downloads (ZIP-wrapped daily NC).
#
# Surface files have hourly data, N→S latitudes; mass flux file has
# 6-hourly data, S→N latitudes. This script selects matching hours
# and flips the latitude axis.
#
# Usage:
#   julia --project=. scripts/preprocessing/merge_era5_surface_to_massflux.jl \
#       /temp1/atmos_transport/era5_spectral_catrine/massflux_era5_spectral_202112_float32.nc \
#       ~/data/metDrivers/era5/surface_catrine/
# ---------------------------------------------------------------------------

using NCDatasets
using Dates
using Printf

function _extract_surface_zips(sfc_dir::String, tmpdir::String)
    # Pre-extract ZIP-wrapped surface files to tmpdir (skip non-ZIP like HDF5)
    for f in readdir(sfc_dir)
        fpath = joinpath(sfc_dir, f)
        startswith(f, "era5_surface_") && endswith(f, ".nc") || continue
        # Check if it's a ZIP (PK signature)
        is_zip = open(fpath, "r") do io
            sig = read(io, 2)
            return sig == UInt8[0x50, 0x4b]
        end
        is_zip || continue
        # Extract into date-named subdir
        date_str = match(r"(\d{8})", f)
        date_str === nothing && continue
        outdir = joinpath(tmpdir, date_str.match)
        mkpath(outdir)
        run(`unzip -q -o $fpath -d $outdir`)
    end
end

function main()
    if length(ARGS) < 2
        error("Usage: merge_era5_surface_to_massflux.jl <massflux.nc> <surface_dir>")
    end
    mf_path = ARGS[1]
    sfc_dir = expanduser(ARGS[2])

    # Read mass flux file time axis
    ds = NCDataset(mf_path, "r")
    mf_times = ds["time"][:]  # Vector{DateTime}
    Nlon = ds.dim["lon"]
    Nlat = ds.dim["lat"]
    Nt = length(mf_times)
    close(ds)
    @info "Mass flux file: $(Nlon)×$(Nlat), $Nt windows"
    @info "Time range: $(mf_times[1]) — $(mf_times[end])"

    # Extract all ZIP surface files to temp directory
    tmpdir = mktempdir()
    @info "Extracting surface ZIPs to $tmpdir ..."
    _extract_surface_zips(sfc_dir, tmpdir)

    # Prepare output arrays
    pblh_out  = zeros(Float32, Nlon, Nlat, Nt)
    t2m_out   = zeros(Float32, Nlon, Nlat, Nt)
    ustar_out = zeros(Float32, Nlon, Nlat, Nt)
    hflux_out = zeros(Float32, Nlon, Nlat, Nt)

    # Process each time step
    for (ti, dt) in enumerate(mf_times)
        date = Date(dt)
        hour = Dates.hour(dt)
        date_str = @sprintf("%04d%02d%02d", year(date), month(date), day(date))

        # Find extracted files for this date
        date_dir = joinpath(tmpdir, date_str)
        if !isdir(date_dir)
            @warn "No extracted surface data for $date, filling with zeros"
            continue
        end

        # Find instant and accumulated NC files
        nc_files = filter(f -> endswith(f, ".nc"), readdir(date_dir; join=true))
        instant_path = nothing
        accum_path = nothing
        for f in nc_files
            if occursin("instant", f)
                instant_path = f
            elseif occursin("accum", f)
                accum_path = f
            end
        end

        # Read instantaneous fields (blh, t2m, u10, v10)
        if instant_path !== nothing
            NCDataset(instant_path, "r") do ds_inst
                # Time axis: seconds since 1970-01-01
                times_dt = ds_inst["valid_time"][:]

                # Find matching hour
                tidx = findfirst(t -> Dates.hour(t) == hour, times_dt)
                if tidx === nothing
                    @warn "Hour $hour not found in surface file for $date"
                    return
                end

                # NCDatasets presents as [lon, lat, time] in Julia
                blh_2d = Float32.(ds_inst["blh"][:, :, tidx])
                t2m_2d = Float32.(ds_inst["t2m"][:, :, tidx])
                u10_2d = Float32.(ds_inst["u10"][:, :, tidx])
                v10_2d = Float32.(ds_inst["v10"][:, :, tidx])

                # Flip latitude (axis 2: N→S to S→N)
                blh_2d = blh_2d[:, end:-1:1]
                t2m_2d = t2m_2d[:, end:-1:1]
                u10_2d = u10_2d[:, end:-1:1]
                v10_2d = v10_2d[:, end:-1:1]

                # Compute friction velocity: u* ≈ κ * V10 / ln(z/z0)
                # κ = 0.4, z = 10m, z0 = 0.01m (typical land)
                κ = Float32(0.4)
                log_z_z0 = Float32(log(10.0 / 0.01))  # ln(1000) ≈ 6.91
                V10 = @. sqrt(u10_2d^2 + v10_2d^2)
                ustar_2d = @. κ * V10 / log_z_z0

                pblh_out[:, :, ti]  .= blh_2d
                t2m_out[:, :, ti]   .= t2m_2d
                ustar_out[:, :, ti] .= ustar_2d
            end
        end

        # Read accumulated fields (sshf → hflux in W/m²)
        if accum_path !== nothing
            NCDataset(accum_path, "r") do ds_acc
                times_dt = ds_acc["valid_time"][:]

                tidx = findfirst(t -> Dates.hour(t) == hour, times_dt)
                if tidx === nothing
                    @warn "Hour $hour not found in accumulated file for $date"
                    return
                end

                # SSHF is in J/m² accumulated over forecast step (1 hour)
                # Convert to W/m² by dividing by step duration (3600 s)
                sshf_2d = Float32.(ds_acc["sshf"][:, :, tidx])
                sshf_2d = sshf_2d[:, end:-1:1]  # flip lat
                hflux_out[:, :, ti] .= sshf_2d ./ Float32(3600)
            end
        end

        ti % 4 == 0 && @info "  Processed window $ti/$Nt ($dt)"
    end

    # Clean up temp dir
    rm(tmpdir; recursive=true, force=true)

    # Write to mass flux file in append mode
    @info "Appending surface fields to $mf_path ..."
    NCDataset(mf_path, "a") do ds
        for (name, data, long_name, units) in [
            ("pblh",  pblh_out,  "Boundary layer height",       "m"),
            ("t2m",   t2m_out,   "2-metre temperature",         "K"),
            ("ustar", ustar_out, "Friction velocity",           "m/s"),
            ("hflux", hflux_out, "Surface sensible heat flux",  "W/m2"),
        ]
            if haskey(ds, name)
                @info "  Variable $name already exists, overwriting"
                ds[name][:, :, :] = data
            else
                defVar(ds, name, data, ("lon", "lat", "time");
                       attrib=Dict("long_name" => long_name, "units" => units))
            end
        end
    end

    @info "Done! Added pblh, t2m, ustar, hflux to mass flux file."
end

main()
