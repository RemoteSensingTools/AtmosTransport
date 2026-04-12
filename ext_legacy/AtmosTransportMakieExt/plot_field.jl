function AtmosTransport.Visualization.plot_field(
        data::AbstractMatrix, lons::AbstractVector, lats::AbstractVector;
        projection     = :Robinson,
        domain         = :global,
        colormap       = :YlOrRd,
        colorrange     = nothing,
        colorscale     = identity,
        coastlines     = true,
        title          = "",
        size           = (1000, 520),
        colorbar_label = "",
        figure         = nothing)

    lon_range, lat_range = _resolve_domain(domain)
    proj_str = _projection_string(projection)

    # Normalize lon/lat ordering
    lon, lat, z = _normalize_lonlat(Float64.(lons), Float64.(lats), Float32.(data))

    # Build 2D coordinate meshes for surface! (nlat × nlon layout)
    nlon, nlat = length(lon), length(lat)
    lon_2d = Float32[lon[i] for _j in 1:nlat, i in 1:nlon]
    lat_2d = Float32[lat[j] for  j in 1:nlat, _i in 1:nlon]

    # Auto colorrange from non-NaN values
    cr = if colorrange !== nothing
        colorrange
    else
        valid = filter(!isnan, z)
        isempty(valid) ? (0f0, 1f0) : (minimum(valid), maximum(valid))
    end

    fig = figure !== nothing ? figure : Figure(; size, fontsize=13)
    ax = GeoAxis(fig[1, 1];
        dest   = proj_str,
        title  = title,
        limits = (lon_range..., lat_range...))

    # surface! reprojects lon/lat to the target projection
    sf = surface!(ax, lon_2d, lat_2d, z';
        shading    = NoShading,
        colormap   = colormap,
        colorrange = cr,
        colorscale = colorscale)

    if coastlines
        lines!(ax, GeoMakie.coastlines(); color=(:black, 0.5), linewidth=0.7)
    end

    Colorbar(fig[1, 2], sf; label=colorbar_label, width=18)

    return fig
end
