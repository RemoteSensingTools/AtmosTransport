# ---------------------------------------------------------------------------
# read_met! implementation for MetDataSource
#
# Reads meteorological fields from OPeNDAP or local NetCDF files into
# the MetDataSource buffers. Fields are read using the variable mappings
# from the TOML configuration.
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

"""
    read_met!(met::MetDataSource{FT}, time_seconds::Real;
              variables::Union{Nothing, Vector{Symbol}}=nothing) where {FT}

Read meteorological data for the given time into internal buffers.

# Arguments
- `met` — the met data source (with TOML config)
- `time_seconds` — time in seconds since model epoch, or a DateTime
- `variables` — optional subset of canonical variables to read (default: all mapped)

Dispatches on the access protocol:
- `"opendap"` → reads directly from OPeNDAP URL
- `"local"` → reads from local NetCDF files (in `met.local_path`)
"""
function read_met!(met::MetDataSource{FT}, time_seconds::Real;
                   variables::Union{Nothing, Vector{Symbol}}=nothing) where {FT}
    proto = protocol(met)
    if proto == "opendap" && isempty(met.local_path)
        _read_met_opendap!(met, time_seconds; variables)
    else
        _read_met_local!(met, time_seconds; variables)
    end
    met.current_time[] = Float64(time_seconds)
    return nothing
end

"""
    read_met!(met::MetDataSource, dt::Dates.DateTime; kw...)

Read met data for a DateTime.
"""
function read_met!(met::MetDataSource, dt::Dates.DateTime; kw...)
    epoch = Dates.DateTime(1, 1, 1, 0, 0, 0)
    seconds = Dates.value(dt - epoch) / 1000.0  # milliseconds → seconds
    return read_met!(met, seconds; kw...)
end

# ---------------------------------------------------------------------------
# OPeNDAP reader
# ---------------------------------------------------------------------------

function _read_met_opendap!(met::MetDataSource{FT}, time_seconds::Real;
                            variables=nothing) where {FT}
    vars_to_read = something(variables, collect(keys(met.config.variables)))

    # Group variables by collection to minimize OPeNDAP connections
    by_collection = Dict{String, Vector{Symbol}}()
    for vname in vars_to_read
        mapping = met.config.variables[vname]
        coll_key = mapping.collection
        if !haskey(by_collection, coll_key)
            by_collection[coll_key] = Symbol[]
        end
        push!(by_collection[coll_key], vname)
    end

    for (coll_key, var_names) in by_collection
        url = build_opendap_url(met.config, coll_key)
        coll = met.config.collections[coll_key]

        ds = NCDataset(url)
        try
            # Find time index
            times = ds["time"][:]
            tidx = _find_time_index(times, time_seconds, met.config.name)

            for vname in var_names
                mapping = met.config.variables[vname]
                native = mapping.native_name

                if !haskey(ds, native)
                    @warn "Variable '$native' not found in $(coll.dataset) for $(met.config.name)"
                    continue
                end

                raw = _read_variable(ds, native, tidx)
                met.buffers[vname] = FT.(raw) .* FT(mapping.unit_conversion)
            end
        finally
            close(ds)
        end
    end
end

# ---------------------------------------------------------------------------
# Local file reader
# ---------------------------------------------------------------------------

function _read_met_local!(met::MetDataSource{FT}, time_seconds::Real;
                          variables=nothing) where {FT}
    vars_to_read = something(variables, collect(keys(met.config.variables)))
    data_dir = met.local_path

    isdir(data_dir) || error("Local met data directory not found: $data_dir")

    # Group by collection
    by_collection = Dict{String, Vector{Symbol}}()
    for vname in vars_to_read
        mapping = met.config.variables[vname]
        coll_key = mapping.collection
        if !haskey(by_collection, coll_key)
            by_collection[coll_key] = Symbol[]
        end
        push!(by_collection[coll_key], vname)
    end

    for (coll_key, var_names) in by_collection
        # Find matching file in local directory
        filepath = _find_local_file(data_dir, coll_key, met.config)
        filepath === nothing && continue

        ds = NCDataset(filepath)
        try
            # Check if file has a time dimension
            has_time = haskey(ds.dim, "time")
            tidx = if has_time
                times = ds["time"][:]
                _find_time_index_local(times, time_seconds)
            else
                1
            end

            for vname in var_names
                mapping = met.config.variables[vname]
                native = mapping.native_name

                if !haskey(ds, native)
                    @warn "Variable '$native' not found in $filepath"
                    continue
                end

                raw = if has_time
                    _read_variable(ds, native, tidx)
                else
                    Array(ds[native])
                end
                met.buffers[vname] = FT.(raw) .* FT(mapping.unit_conversion)
            end
        finally
            close(ds)
        end
    end
end

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""Read a variable at a specific time index, handling 2D vs 3D."""
function _read_variable(ds::NCDataset, varname::String, tidx::Int)
    var = ds[varname]
    ndims_var = ndims(var)
    if ndims_var == 4      # (lon, lat, lev, time)
        return Array(var[:, :, :, tidx])
    elseif ndims_var == 3  # (lon, lat, time)
        return Array(var[:, :, tidx])
    elseif ndims_var == 2  # (lon, lat) — no time dim
        return Array(var[:, :])
    else
        error("Unexpected number of dimensions ($ndims_var) for variable $varname")
    end
end

"""Find the nearest time index for OPeNDAP datasets.
OPeNDAP GEOS-FP/MERRA-2 time is in 'days since 1-1-1'."""
function _find_time_index(times::AbstractVector, time_seconds::Real, source_name::String)
    # Convert seconds since epoch to days since 1-1-1
    # This is approximate — proper implementation would use Dates
    time_days = time_seconds / 86400.0
    idx = argmin(abs.(times .- time_days))
    return idx
end

"""Find time index for local files (may use different time encoding)."""
function _find_time_index_local(times::AbstractVector, time_seconds::Real)
    if length(times) == 1
        return 1
    end
    # Try direct match first, then nearest
    idx = argmin(abs.(times .- time_seconds))
    return idx
end

"""Find a local NetCDF file matching a collection key."""
function _find_local_file(dir::String, coll_key::String, config::MetSourceConfig)
    coll = config.collections[coll_key]
    dataset_name = coll.dataset
    collection_name = coll.collection_name

    # Search for files containing the collection/dataset name
    for f in readdir(dir)
        if (endswith(f, ".nc") || endswith(f, ".nc4"))
            if contains(f, dataset_name) || contains(f, collection_name) || contains(f, coll_key)
                return joinpath(dir, f)
            end
        end
    end

    @warn "No local file found for collection '$coll_key' (dataset=$dataset_name) in $dir"
    return nothing
end
