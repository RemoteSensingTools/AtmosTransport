function AtmosTransport.Visualization.animate_output(
        nc_path::AbstractString, variable::Symbol;
        output_path    = nothing,
        framerate      = 10,
        frame_step     = nothing,
        projection     = :Robinson,
        domain         = :global,
        colormap       = :YlOrRd,
        colorrange     = nothing,
        colorscale     = identity,
        coastlines     = true,
        title_prefix   = "",
        colorbar_label = "",
        size           = (1000, 520))

    isfile(nc_path) || error("NetCDF file not found: $nc_path")

    out_path = output_path !== nothing ? output_path :
        replace(nc_path, r"\.nc$" => "_$(variable)_animation.gif")

    lon_range, lat_range = _resolve_domain(domain)
    proj_str = _projection_string(projection)

    NCDataset(nc_path, "r") do ds
        lon_raw  = Float64.(ds["lon"][:])
        lat_raw  = Float64.(ds["lat"][:])
        time_raw = ds["time"][:]

        var_str = string(variable)
        haskey(ds, var_str) || error("Variable '$var_str' not found in $nc_path")

        ncvar    = ds[var_str]
        dims     = NCDatasets.dimnames(ncvar)
        time_dim = findfirst(==("time"), dims)
        all_data = if length(dims) == 3
            ncvar[:, :, :]
        elseif length(dims) == 4
            ncvar[:, :, :, :]
        else
            ncvar[:]
        end

        nlon, nlat = length(lon_raw), length(lat_raw)
        ntime = length(time_raw)

        # Normalize coordinates
        lon, lat, _ = _normalize_lonlat(lon_raw, lat_raw, zeros(Float32, nlon, nlat))
        lon_order   = sortperm([l > 180 ? l - 360.0 : l for l in lon_raw])
        lat_order   = lat_raw[1] > lat_raw[end] ?
                          collect(length(lat_raw):-1:1) : collect(1:length(lat_raw))

        # Frame selection
        step = frame_step !== nothing ? frame_step : max(1, ntime ÷ 120)
        idx  = 1:step:ntime

        # Auto colorrange from full dataset
        cr = if colorrange !== nothing
            colorrange
        else
            valid = filter(!isnan, all_data)
            isempty(valid) ? (0f0, 1f0) : (Float32(minimum(valid)), Float32(maximum(valid)))
        end

        # Build coordinate meshes
        lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
        lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]

        # Figure
        fig = Figure(; size, fontsize=13)
        ax  = GeoAxis(fig[1, 1];
            dest   = proj_str,
            limits = (lon_range..., lat_range...))

        function get_slice(ti)
            z = if time_dim == 1
                Float32.(all_data[ti, :, :])
            else
                Float32.(all_data[:, :, ti])
            end
            return z[lon_order, lat_order]'
        end

        z_obs = Observable(get_slice(idx[1]))

        sf = surface!(ax, lon_2d, lat_2d, z_obs;
            shading    = NoShading,
            colormap   = colormap,
            colorrange = cr,
            colorscale = colorscale)

        if coastlines
            lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
        end

        Colorbar(fig[1, 2], sf; label=colorbar_label, width=18)

        @info "Writing $(length(idx)) frames to $out_path"

        record(fig, out_path, 1:length(idx); framerate) do frame_num
            ti    = idx[frame_num]
            t_val = time_raw[ti]
            t_str = if t_val isa DateTime
                string(t_val)
            else
                @sprintf("%.1f s", Float64(t_val))
            end
            z_obs[]  = get_slice(ti)
            ax.title = isempty(title_prefix) ? "$var_str — $t_str" :
                                               "$title_prefix — $t_str"
        end

        @info "Animation saved: $out_path"
    end

    return out_path
end
