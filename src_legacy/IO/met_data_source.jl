# ---------------------------------------------------------------------------
# MetDataSource — generic config-driven met data reader
#
# A single type that works with any met data source defined by a TOML file.
# The three convenience constructors (GEOSFPMetData, MERRAMetData,
# ERA5MetData) simply load the appropriate TOML config.
#
# This replaces the previous per-source structs that had hardcoded
# variable mappings.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Config-driven meteorological data reader. Reads its variable mappings,
collection definitions, and access URLs from a TOML configuration file.

$(FIELDS)

# Construction
```julia
# From a specific TOML file:
met = MetDataSource(Float64, "config/met_sources/geosfp.toml")

# Using built-in configs:
met = MetDataSource(Float64, "geosfp")

# Convenience aliases:
met = GEOSFPMetData(Float64)
met = MERRAMetData(Float64)
met = ERA5MetData(Float64)
```
"""
struct MetDataSource{FT} <: AbstractMetData{FT}
    "parsed TOML configuration"
    config       :: MetSourceConfig
    "cached field data (canonical name → array)"
    buffers      :: Dict{Symbol, Array}
    "time of the currently loaded data"
    current_time :: Base.RefValue{Float64}
    "local data directory (overrides OPeNDAP if non-empty)"
    local_path   :: String
end

function MetDataSource(::Type{FT}, config::MetSourceConfig;
                       local_path::String = "") where {FT}
    buffers = Dict{Symbol, Array}()
    return MetDataSource{FT}(config, buffers, Ref(-Inf), local_path)
end

function MetDataSource(::Type{FT}, source::String;
                       local_path::String = "") where {FT}
    # If source looks like a file path, load directly; otherwise treat as short name
    config = if isfile(source)
        load_met_config(source)
    else
        default_met_config(source)
    end
    return MetDataSource(FT, config; local_path)
end

# Default to Float64
MetDataSource(source::String; kw...) = MetDataSource(Float64, source; kw...)

# ---------------------------------------------------------------------------
# Convenience constructors — backward-compatible names
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Create a GEOS-FP met data reader from the built-in TOML config.
Optionally override with a custom config path.
"""
function GEOSFPMetData(; FT::Type = Float64,
                        local_path::String = "",
                        config_path::Union{String, Nothing} = nothing)
    config = if config_path !== nothing
        load_met_config(config_path)
    else
        default_met_config("geosfp")
    end
    return MetDataSource(FT, config; local_path)
end

"""
$(TYPEDSIGNATURES)

Create a MERRA-2 met data reader from the built-in TOML config.
"""
function MERRAMetData(; FT::Type = Float64,
                       local_path::String = "",
                       config_path::Union{String, Nothing} = nothing)
    config = if config_path !== nothing
        load_met_config(config_path)
    else
        default_met_config("merra2")
    end
    return MetDataSource(FT, config; local_path)
end

"""
$(TYPEDSIGNATURES)

Create an ERA5 met data reader from the built-in TOML config.
"""
function ERA5MetData(; FT::Type = Float64,
                      local_path::String = "",
                      config_path::Union{String, Nothing} = nothing)
    config = if config_path !== nothing
        load_met_config(config_path)
    else
        default_met_config("era5")
    end
    return MetDataSource(FT, config; local_path)
end

# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Return the buffered field for the given canonical variable name.
"""
function get_field(met::MetDataSource, name::Symbol)
    haskey(met.buffers, name) || error(
        "Variable :$name not loaded. Available: $(keys(met.buffers)). " *
        "Call read_met! first, or check that $(met.config.name) provides :$name."
    )
    return met.buffers[name]
end

"""
$(SIGNATURES)

Check whether the met data source provides a given canonical variable.
"""
has_variable(met::MetDataSource, name::Symbol) = haskey(met.config.variables, name)

"""
$(SIGNATURES)

Return the native variable name for a canonical variable.
"""
function native_name(met::MetDataSource, canonical::Symbol)
    haskey(met.config.variables, canonical) || error(
        "Canonical variable :$canonical not mapped in $(met.config.name)"
    )
    return met.config.variables[canonical].native_name
end

"""
$(SIGNATURES)

Return the collection that contains a given canonical variable.
"""
function collection_for(met::MetDataSource, canonical::Symbol)
    mapping = met.config.variables[canonical]
    return met.config.collections[mapping.collection]
end

"""
$(SIGNATURES)

Return the native grid specification from the config.
"""
function met_grid(met::MetDataSource)
    return met.config.grid_info
end

"""
$(SIGNATURES)

Return the data access protocol (opendap, cds, local).
"""
protocol(met::MetDataSource) = get(met.config.access, "protocol", "local")

"""
$(SIGNATURES)

Return the time interval between snapshots (seconds).
"""
time_interval(met::MetDataSource) = get(met.config.access, "time_interval", 10800.0)

"""
$(SIGNATURES)

Return the human-readable name of the met data source.
"""
source_name(met::MetDataSource) = met.config.name

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

function Base.show(io::Base.IO, met::MetDataSource{FT}) where {FT}
    nvars = length(met.config.variables)
    ncolls = length(met.config.collections)
    proto = protocol(met)
    loaded = length(met.buffers)
    print(io, "MetDataSource{$FT}(\"$(met.config.name)\", $nvars vars, $ncolls collections, protocol=$proto")
    if loaded > 0
        print(io, ", $loaded fields loaded")
    end
    if !isempty(met.local_path)
        print(io, ", local_path=\"$(met.local_path)\"")
    end
    print(io, ")")
end

export MetDataSource, GEOSFPMetData, MERRAMetData, ERA5MetData
export get_field, has_variable, native_name, collection_for
export protocol, time_interval, source_name
