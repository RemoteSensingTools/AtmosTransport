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
    MetDataSource{FT} <: AbstractMetData{FT}

Config-driven meteorological data reader. Reads its variable mappings,
collection definitions, and access URLs from a TOML configuration file.

# Fields
- `config :: MetSourceConfig` — parsed TOML configuration
- `buffers :: Dict{Symbol, Array}` — cached field data (canonical name → array)
- `current_time :: Base.RefValue{Float64}` — time of the currently loaded data
- `local_path :: String` — local data directory (overrides OPeNDAP if non-empty)

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
    config       :: MetSourceConfig
    buffers      :: Dict{Symbol, Array}
    current_time :: Base.RefValue{Float64}
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
    GEOSFPMetData(; FT=Float64, local_path="", config_path=nothing)

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
    MERRAMetData(; FT=Float64, local_path="", config_path=nothing)

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
    ERA5MetData(; FT=Float64, local_path="", config_path=nothing)

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
    get_field(met::MetDataSource, name::Symbol) -> Array

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
    has_variable(met::MetDataSource, name::Symbol) -> Bool

Check whether the met data source provides a given canonical variable.
"""
has_variable(met::MetDataSource, name::Symbol) = haskey(met.config.variables, name)

"""
    native_name(met::MetDataSource, canonical::Symbol) -> String

Return the native variable name for a canonical variable.
"""
function native_name(met::MetDataSource, canonical::Symbol)
    haskey(met.config.variables, canonical) || error(
        "Canonical variable :$canonical not mapped in $(met.config.name)"
    )
    return met.config.variables[canonical].native_name
end

"""
    collection_for(met::MetDataSource, canonical::Symbol) -> CollectionInfo

Return the collection that contains a given canonical variable.
"""
function collection_for(met::MetDataSource, canonical::Symbol)
    mapping = met.config.variables[canonical]
    return met.config.collections[mapping.collection]
end

"""
    met_grid(met::MetDataSource)

Return the native grid specification from the config.
"""
function met_grid(met::MetDataSource)
    return met.config.grid_info
end

"""
    protocol(met::MetDataSource) -> String

Return the data access protocol ("opendap", "cds", "local").
"""
protocol(met::MetDataSource) = get(met.config.access, "protocol", "local")

"""
    time_interval(met::MetDataSource) -> Float64

Return the time interval between snapshots (seconds).
"""
time_interval(met::MetDataSource) = get(met.config.access, "time_interval", 10800.0)

"""
    source_name(met::MetDataSource) -> String

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
