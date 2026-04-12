function AtmosTransport.Visualization.plot_output(
        nc_path::AbstractString, variable::Symbol;
        time_index = nothing,
        kwargs...)

    isfile(nc_path) || error("NetCDF file not found: $nc_path")

    NCDataset(nc_path, "r") do ds
        lon = Float64.(ds["lon"][:])
        lat = Float64.(ds["lat"][:])
        var_str = string(variable)
        haskey(ds, var_str) || error("Variable '$var_str' not found in $nc_path")

        raw   = ds[var_str]
        dims  = NCDatasets.dimnames(raw)
        ntime = length(ds["time"])

        # Default to last time step
        ti = time_index !== nothing ? time_index : ntime

        time_dim = findfirst(==("time"), dims)
        data_slice = if time_dim == 1
            Float32.(raw[ti, :, :])
        elseif time_dim == 3
            Float32.(raw[:, :, ti])
        elseif time_dim === nothing
            # No time dimension — 2D field
            Float32.(raw[:, :])
        else
            error("Unexpected dimension layout: $dims")
        end

        # Build a default title from the time coordinate
        t_val = ds["time"][ti]
        t_str = if t_val isa DateTime
            string(t_val)
        else
            @sprintf("t = %.1f s", Float64(t_val))
        end
        default_title = "$var_str — $t_str"

        # Merge default title if user didn't provide one
        kw = Dict{Symbol,Any}(kwargs)
        if !haskey(kw, :title) || isempty(kw[:title])
            kw[:title] = default_title
        end

        return AtmosTransport.Visualization.plot_field(data_slice, lon, lat; kw...)
    end
end
