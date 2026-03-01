#!/usr/bin/env julia
#=
Dual-panel animation: column-mean CO₂ (top) + surface CO₂ (bottom).

Usage:
  julia --project=scripts scripts/animate_dual_panel.jl <nc_file> [region]

  region: asia (default), us, europe, north_america, south_america, africa, global

Example:
  julia --project=scripts scripts/animate_dual_panel.jl \
      /temp1/atmos_transport/output/geosfp_c720_june2024_fixed.nc asia
=#

using NCDatasets
using CairoMakie
using GeoMakie
using Dates
using Printf

const DOMAINS = Dict{Symbol, NamedTuple}(
    :global         => (lon_range=(-180.0, 180.0), lat_range=(-90.0, 90.0)),
    :us             => (lon_range=(-130.0, -60.0), lat_range=(24.0, 50.0)),
    :europe         => (lon_range=(-15.0, 35.0),   lat_range=(35.0, 72.0)),
    :asia           => (lon_range=(60.0, 150.0),    lat_range=(10.0, 55.0)),
    :north_america  => (lon_range=(-170.0, -50.0),  lat_range=(15.0, 75.0)),
    :south_america  => (lon_range=(-85.0, -30.0),   lat_range=(-60.0, 15.0)),
    :africa         => (lon_range=(-20.0, 55.0),    lat_range=(-40.0, 40.0)),
)

function main()
    nc_path = length(ARGS) >= 1 ? ARGS[1] :
        "/temp1/atmos_transport/output/geosfp_c720_june2024_fixed.nc"
    region  = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :asia
    isfile(nc_path) || error("NetCDF not found: $nc_path")
    haskey(DOMAINS, region) || error("Unknown region :$region. Available: $(join(keys(DOMAINS), ", "))")

    vmax_col = 7f0    # column-mean max [ppm]
    vmax_sfc = 70f0   # surface max [ppm]

    dom = DOMAINS[region]
    lon_min, lon_max = dom.lon_range
    lat_min, lat_max = dom.lat_range

    NCDataset(nc_path, "r") do ds
        lon_raw  = Float64.(ds["lon"][:])
        lat_raw  = Float64.(ds["lat"][:])
        time_raw = ds["time"][:]
        t0       = DateTime(2024, 6, 1)

        # Shift lon 0..360 → -180..180
        lon_shifted = [l > 180 ? l - 360.0 : l for l in lon_raw]
        lon_order   = sortperm(lon_shifted)
        lon_sorted  = lon_shifted[lon_order]

        # Ensure lat ascending (S→N)
        lat_order = lat_raw[1] > lat_raw[end] ?
                        collect(length(lat_raw):-1:1) : collect(1:length(lat_raw))
        lat_sorted = lat_raw[lat_order]

        # Subset indices for the region
        ilon_sub = findall(x -> lon_min <= x <= lon_max, lon_sorted)
        ilat_sub = findall(x -> lat_min <= x <= lat_max, lat_sorted)
        lon = lon_sorted[ilon_sub]
        lat = lat_sorted[ilat_sub]

        raw_ilon = lon_order[ilon_sub]
        raw_ilat = lat_order[ilat_sub]

        nlon, nlat = length(lon), length(lat)
        ntime = length(time_raw)

        time_hr = if eltype(time_raw) <: DateTime
            Float64.(Dates.value.(time_raw .- Ref(t0))) ./ 3_600_000
        else
            Float64.(time_raw)
        end

        # Read both variables (spatial subset only)
        @info "Reading co2_column_mean subset: $nlon × $nlat × $ntime"
        flush(stderr)
        t_read = @elapsed begin
            col_raw = ds["co2_column_mean"][raw_ilon, raw_ilat, :]
        end
        @info @sprintf("  Column-mean read: %.1f s (%.2f GB)", t_read, sizeof(col_raw) / 1e9)

        @info "Reading co2_surface subset: $nlon × $nlat × $ntime"
        flush(stderr)
        t_read = @elapsed begin
            sfc_raw = ds["co2_surface"][raw_ilon, raw_ilat, :]
        end
        @info @sprintf("  Surface read: %.1f s (%.2f GB)", t_read, sizeof(sfc_raw) / 1e9)
        flush(stderr)

        # Reorder so lon/lat ascending
        sub_lon_order = sortperm(lon_shifted[raw_ilon])
        sub_lat_order = lat_raw[raw_ilat][1] > lat_raw[raw_ilat][end] ?
                            (length(raw_ilat):-1:1) : (1:length(raw_ilat))
        col_data = Float32.(col_raw[sub_lon_order, collect(sub_lat_order), :])
        sfc_data = Float32.(sfc_raw[sub_lon_order, collect(sub_lat_order), :])

        # Frame selection: ~120 frames max
        frame_step = max(1, ntime ÷ 120)
        idx = 1:frame_step:ntime
        nframes = length(idx)

        # 2D coordinate meshes (nlat × nlon for surface!)
        lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
        lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]

        get_col(ti) = col_data[:, :, ti]'   # transpose to (nlat, nlon)
        get_sfc(ti) = sfc_data[:, :, ti]'

        # Projection
        proj_str = if region === :global
            "+proj=robin"
        elseif region === :asia
            "+proj=lcc +lon_0=105 +lat_1=25 +lat_2=45"
        elseif region === :us
            "+proj=lcc +lon_0=-95 +lat_1=30 +lat_2=45"
        elseif region === :europe
            "+proj=lcc +lon_0=10 +lat_1=40 +lat_2=60"
        else
            "+proj=eqc"
        end

        # Dual-panel figure
        fig = Figure(; size=(1100, 900), fontsize=14)

        # Top panel: column-mean
        ax_col = GeoAxis(fig[1, 1];
            dest   = proj_str,
            limits = (lon_min, lon_max, lat_min, lat_max),
            title  = "Column-mean CO₂",
        )
        z_col = Observable(get_col(idx[1]))
        sf_col = surface!(ax_col, lon_2d, lat_2d, z_col;
            shading    = NoShading,
            colormap   = :YlOrRd,
            colorrange = (0f0, vmax_col),
            colorscale = sqrt,
        )
        lines!(ax_col, GeoMakie.coastlines(); color=(:black, 0.6), linewidth=0.8)
        Colorbar(fig[1, 2], sf_col;
            label = "Column-mean CO₂ (ppm)",
            width = 16,
            ticks = range(0, vmax_col, length=8) .|> (x -> round(x, digits=1)),
        )

        # Bottom panel: surface
        ax_sfc = GeoAxis(fig[2, 1];
            dest   = proj_str,
            limits = (lon_min, lon_max, lat_min, lat_max),
            title  = "Surface CO₂",
        )
        z_sfc = Observable(get_sfc(idx[1]))
        sf_sfc = surface!(ax_sfc, lon_2d, lat_2d, z_sfc;
            shading    = NoShading,
            colormap   = :YlOrRd,
            colorrange = (0f0, vmax_sfc),
            colorscale = sqrt,
        )
        lines!(ax_sfc, GeoMakie.coastlines(); color=(:black, 0.6), linewidth=0.8)
        Colorbar(fig[2, 2], sf_sfc;
            label = "Surface CO₂ (ppm)",
            width = 16,
            ticks = range(0, vmax_sfc, length=8) .|> (x -> round(x, digits=0)),
        )

        # Supertitle
        title_label = Label(fig[0, :],
            "GEOS-FP C720 CO₂ Enhancement — Day 0.0",
            fontsize=18, font=:bold)

        gif_path = replace(nc_path, ".nc" => "_$(region)_dual.gif")
        @info "Writing $nframes frames to $gif_path"
        flush(stderr)

        t_start = time()
        record(fig, gif_path, 1:nframes;
               framerate=min(15, max(4, nframes ÷ 10))) do frame_num
            ti = idx[frame_num]
            t_hr = time_hr[ti]
            day = t_hr / 24
            date_str = string(t0 + Hour(round(Int, t_hr)))

            z_col[] = get_col(ti)
            z_sfc[] = get_sfc(ti)
            title_label.text = "GEOS-FP C720 CO₂ Enhancement — Day $(round(day; digits=1))  ($date_str)"

            elapsed = time() - t_start
            rate = frame_num / elapsed
            eta = (nframes - frame_num) / rate
            @printf(stderr, "\r  Frame %3d/%d  (%.1f fps, ETA %.0fs)  ", frame_num, nframes, rate, eta)
            flush(stderr)
        end

        println(stderr)
        @info @sprintf("Done: %s (%.1f s total)", gif_path, time() - t_start)
    end
end

main()
