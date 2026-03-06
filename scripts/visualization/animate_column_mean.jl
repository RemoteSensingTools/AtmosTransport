#!/usr/bin/env julia
#=
Animate column-averaged CO2 mixing ratio (Δp-weighted) from the forward-run output.

Usage:
  julia --project=scripts scripts/animate_column_mean.jl [path_to_output.nc]
  OUTDIR=/path/to/output julia --project=scripts scripts/animate_column_mean.jl

Output: column_mean_animation.gif
=#

using NCDatasets
using CairoMakie
using GeoMakie
using Dates
using Printf

function main()
    outdir  = get(ENV, "OUTDIR",
                  joinpath(homedir(), "data", "output", "era5_edgar_preprocessed_f32"))
    nc_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(outdir, "era5_edgar_preprocessed.nc")
    isfile(nc_path) || error("NetCDF not found: $nc_path")

    NCDataset(nc_path, "r") do ds
        lon_raw  = Float64.(ds["lon"][:])   # may be 0..360 from forward run
        lat_raw  = Float64.(ds["lat"][:])
        time_raw = ds["time"][:]
        t0       = DateTime(2024, 6, 1)
        time_hr  = if eltype(time_raw) <: DateTime
            Float64.(Dates.value.(time_raw .- Ref(t0))) ./ 3_600_000
        else
            Float64.(time_raw)
        end

        nlon, nlat = length(lon_raw), length(lat_raw)
        ntime      = length(time_hr)

        # --- read field, reshape if needed ---
        co2 = ds["co2_column_mean"][:]
        if ndims(co2) == 1 && length(co2) == nlon * nlat * ntime
            co2 = reshape(co2, nlon, nlat, ntime)
        end
        dimnames  = NCDatasets.dimnames(ds["co2_column_mean"])
        time_dim  = findfirst(==("time"), dimnames)   # which axis is time?

        @info "Reading $nc_path: $ntime time steps (lon=$nlon, lat=$nlat)"

        # --- shift lon 0..360 → -180..180, sort ascending ---
        lon_shifted = [l > 180 ? l - 360.0 : l for l in lon_raw]
        lon_order   = sortperm(lon_shifted)
        lon         = lon_shifted[lon_order]           # -180..180 ascending

        # Ensure lat is ascending (S→N)
        if lat_raw[1] > lat_raw[end]
            lat_order = length(lat_raw):-1:1
        else
            lat_order = 1:length(lat_raw)
        end
        lat = lat_raw[lat_order]

        # --- helper: get a (nlon, nlat) slice at time index ti ---
        function get_slice(ti)
            if time_dim == 1
                z = Float32.(co2[ti, :, :])       # (nlon, nlat)
            else
                z = Float32.(co2[:, :, ti])
            end
            z = z[lon_order, lat_order]           # reorder lon/lat
            return z
        end

        vmin = 0f0
        vmax = 7f0

        # 2-D coordinate meshes for surface! on GeoAxis  (size: nlat × nlon)
        lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
        lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]

        # surface! expects (nlat, nlon) — transpose from get_slice's (nlon, nlat)
        get_z_ds(ti) = get_slice(ti)'

        # --- frame selection ---
        frame_step = max(1, ntime ÷ 120)
        idx        = 1:frame_step:ntime
        nframes    = length(idx)
        @info "Writing $nframes frames to GIF"

        # --- figure ---
        fig = Figure(; size = (1000, 520), fontsize = 13)
        ax  = GeoAxis(
            fig[1, 1];
            dest        = "+proj=robin",
            xlabel      = "Longitude",
            ylabel      = "Latitude",
            title       = "",
        )

        z_obs = Observable(get_z_ds(idx[1]))

        # surface! reprojects lon/lat → Robinson; no Resampler, no stripes
        sf = surface!(
            ax, lon_2d, lat_2d, z_obs;
            shading    = NoShading,
            colormap   = :YlOrRd,
            colorrange = (vmin, vmax),
            colorscale = sqrt,          # sqrt stretch: muted near 0, visible at enhancements
        )

        # Coastlines — GeoMakie 0.7 API
        lines!(ax, GeoMakie.coastlines(); color = (:black, 0.5), linewidth = 0.7)

        Colorbar(fig[1, 2], sf;
                 label     = "Column-mean CO₂ enhancement (ppm)",
                 width     = 18,
                 ticks     = range(0, vmax, length=6) .|> (x -> round(x, digits=2)),
        )

        gif_path = joinpath(dirname(nc_path), "column_mean_animation.gif")
        @info "Writing $nframes frames to $gif_path"

        record(fig, gif_path, 1:nframes;
               framerate = min(15, max(4, nframes ÷ 10))) do frame_num
            ti      = idx[frame_num]
            t_hr    = time_hr[ti]
            day     = t_hr / 24
            date_str = string(t0 + Hour(round(Int, t_hr)))
            z_obs[] = get_z_ds(ti)
            ax.title = "Column-mean CO₂ (ppm) — Day $(round(day; digits=1))  ($date_str)"
        end

        @info "Done: $gif_path"
    end
end

main()
