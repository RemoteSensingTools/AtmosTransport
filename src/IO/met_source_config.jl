# ---------------------------------------------------------------------------
# TOML-driven Met Data Source Configuration
#
# Parses a met source TOML file (e.g., config/met_sources/geosfp.toml)
# into typed Julia structs. The Julia model code is generic — all
# source-specific details (URLs, variable names, collections) live
# in the TOML files.
#
# See config/met_sources/*.toml for the schema and examples.
# ---------------------------------------------------------------------------

using TOML
using ..Grids: HybridSigmaPressure

"""
    VarMapping

Mapping from a canonical variable name to its native representation
in a specific met data source.

# Fields
- `canonical_name :: Symbol` — canonical name (e.g., :u_wind)
- `native_name :: String` — name in the met file (e.g., "u" or "U")
- `collection :: String` — key into the collections dict (e.g., "asm_Nv_inst")
- `unit_conversion :: Float64` — multiplicative factor: canonical = native × factor
- `cds_name :: String` — CDS API variable name (ERA5 only; empty otherwise)
"""
struct VarMapping
    canonical_name  :: Symbol
    native_name     :: String
    collection      :: String
    unit_conversion :: Float64
    cds_name        :: String
end

"""
    CollectionInfo

Metadata for a dataset collection within a met data source.

# Fields
- `key :: String` — lookup key (e.g., "asm_Nv_inst")
- `dataset :: String` — OPeNDAP dataset name or ESDT name
- `collection_name :: String` — file-level collection name (for MERRA-2 file naming)
- `frequency :: String` — temporal frequency (e.g., "inst3", "tavg3", "tavg1")
- `vertical :: String` — vertical grid type ("Nv", "Ne", "Nx", "pressure")
- `levels :: Int` — number of vertical levels
- `description :: String`
"""
struct CollectionInfo
    key             :: String
    dataset         :: String
    collection_name :: String
    frequency       :: String
    vertical        :: String
    levels          :: Int
    description     :: String
end

"""
    VerticalConfig

Vertical coordinate configuration parsed from the `[vertical]` section of a
met source TOML. All met sources use hybrid sigma-pressure coordinates;
this struct records the source-specific details (level count, coefficient file).

# Fields
- `coordinate_type :: String` — always "HybridSigmaPressure" for now
- `coefficients_file :: String` — path to the TOML file with A/B coefficients
- `n_levels :: Int` — number of model levels
- `n_interfaces :: Int` — number of level interfaces (n_levels + 1)
- `surface_pressure_var :: String` — canonical variable for surface pressure
- `log_surface_pressure_var :: String` — optional: ln(ps) variable (ERA5 model levels)
"""
struct VerticalConfig
    coordinate_type         :: String
    coefficients_file       :: String
    n_levels                :: Int
    n_interfaces            :: Int
    surface_pressure_var    :: String
    log_surface_pressure_var :: String
end

"""
    MetSourceConfig

Complete configuration for a meteorological data source, parsed from TOML.

# Fields
- `name :: String` — human-readable name (e.g., "GEOS-FP")
- `description :: String`
- `institution :: String`
- `grid_info :: Dict{String, Any}` — grid type, resolution, dimensions
- `vertical :: VerticalConfig` — hybrid sigma-pressure vertical coordinate
- `access :: Dict{String, Any}` — protocol, base URL, auth settings
- `collections :: Dict{String, CollectionInfo}` — collection key → info
- `variables :: Dict{Symbol, VarMapping}` — canonical name → mapping
- `toml_path :: String` — path to the source TOML file (for error messages)
"""
struct MetSourceConfig
    name         :: String
    description  :: String
    institution  :: String
    grid_info    :: Dict{String, Any}
    vertical     :: VerticalConfig
    access       :: Dict{String, Any}
    collections  :: Dict{String, CollectionInfo}
    variables    :: Dict{Symbol, VarMapping}
    toml_path    :: String
end

"""
    load_met_config(toml_path::String) -> MetSourceConfig

Parse a met source TOML file into a `MetSourceConfig`.

# Example
```julia
config = load_met_config("config/met_sources/geosfp.toml")
config.name            # "GEOS-FP"
config.variables[:u_wind].native_name  # "u"
config.collections["asm_Nv_inst"].dataset  # "inst3_3d_asm_Nv"
```
"""
function load_met_config(toml_path::String)
    isfile(toml_path) || error("Met source config not found: $toml_path")
    raw = TOML.parsefile(toml_path)

    source = get(raw, "source", Dict())
    name        = get(source, "name", "Unknown")
    description = get(source, "description", "")
    institution = get(source, "institution", "")

    grid_info = get(raw, "grid", Dict{String, Any}())
    access    = get(raw, "access", Dict{String, Any}())

    # Parse vertical coordinate config
    raw_vert = get(raw, "vertical", Dict())
    vertical = VerticalConfig(
        get(raw_vert, "coordinate_type", "HybridSigmaPressure"),
        get(raw_vert, "coefficients_file", ""),
        get(raw_vert, "n_levels", get(grid_info, "Nz", 0)),
        get(raw_vert, "n_interfaces", get(grid_info, "Nz", 0) + 1),
        get(raw_vert, "surface_pressure_var", "surface_pressure"),
        get(raw_vert, "log_surface_pressure_var", ""),
    )

    # Parse collections
    raw_collections = get(raw, "collections", Dict())
    collections = Dict{String, CollectionInfo}()
    for (key, coll) in raw_collections
        collections[key] = CollectionInfo(
            key,
            get(coll, "dataset", ""),
            get(coll, "collection_name", get(coll, "dataset", "")),
            get(coll, "frequency", ""),
            get(coll, "vertical", ""),
            get(coll, "levels", 0),
            get(coll, "description", ""),
        )
    end

    # Parse variable mappings
    raw_vars = get(raw, "variables", Dict())
    variables = Dict{Symbol, VarMapping}()
    for (canonical_str, var_info) in raw_vars
        canonical = Symbol(canonical_str)
        variables[canonical] = VarMapping(
            canonical,
            get(var_info, "native_name", canonical_str),
            get(var_info, "collection", ""),
            get(var_info, "unit_conversion", 1.0),
            get(var_info, "cds_name", ""),
        )
    end

    return MetSourceConfig(
        name, description, institution,
        grid_info, vertical, access, collections, variables,
        toml_path
    )
end

"""
    load_canonical_config(toml_path::String) -> Dict{Symbol, Dict{String, Any}}

Parse the canonical variables TOML file. Returns a dict mapping
canonical variable names to their properties (units, dimensions, required, etc.).
"""
function load_canonical_config(toml_path::String)
    isfile(toml_path) || error("Canonical variables config not found: $toml_path")
    raw = TOML.parsefile(toml_path)
    raw_vars = get(raw, "variables", Dict())
    result = Dict{Symbol, Dict{String, Any}}()
    for (name, info) in raw_vars
        result[Symbol(name)] = info
    end
    return result
end

"""
    default_config_dir()

Return the path to the default config directory (relative to package root).
"""
function default_config_dir()
    return normpath(joinpath(@__DIR__, "..", "..", "config"))
end

"""
    default_met_config(source::String) -> MetSourceConfig

Load a built-in met source config by short name.

# Examples
```julia
config = default_met_config("geosfp")
config = default_met_config("merra2")
config = default_met_config("era5")
```
"""
function default_met_config(source::String)
    toml_path = joinpath(default_config_dir(), "met_sources", "$(source).toml")
    return load_met_config(toml_path)
end

"""
    validate_met_config(config::MetSourceConfig; canonical_path=nothing)

Validate a met source config against the canonical variables definition.
Warns about missing required variables and unknown canonical names.
"""
function validate_met_config(config::MetSourceConfig; canonical_path=nothing)
    if canonical_path === nothing
        canonical_path = joinpath(default_config_dir(), "canonical_variables.toml")
    end
    canonical = load_canonical_config(canonical_path)

    warnings = String[]

    # Check for missing required canonical variables
    for (name, info) in canonical
        required = get(info, "required", false)
        if required && !haskey(config.variables, name)
            push!(warnings, "Required variable :$name missing from $(config.name) config")
        end
    end

    # Check for unknown variables (mapped but not canonical)
    for name in keys(config.variables)
        if !haskey(canonical, name)
            push!(warnings, "Variable :$name in $(config.name) config is not a canonical variable")
        end
    end

    # Check collection references
    for (name, mapping) in config.variables
        if !isempty(mapping.collection) && !haskey(config.collections, mapping.collection)
            push!(warnings, "Variable :$name references unknown collection '$(mapping.collection)'")
        end
    end

    for w in warnings
        @warn w
    end

    return isempty(warnings)
end

"""
    merra2_stream(year::Int) -> Int

Determine the MERRA-2 production stream code for a given year.
Returns 100, 200, 300, or 400.
"""
function merra2_stream(year::Int)
    year < 1980 && error("MERRA-2 data not available before 1980 (requested year $year)")
    year ≤ 1991 && return 100
    year ≤ 2000 && return 200
    year ≤ 2010 && return 300
    return 400
end

"""
    build_opendap_url(config::MetSourceConfig, collection_key::String) -> String

Construct the OPeNDAP URL for a specific collection.
"""
function build_opendap_url(config::MetSourceConfig, collection_key::String)
    base = get(config.access, "base_url", "")
    coll = config.collections[collection_key]
    return "$base/$(coll.dataset)"
end

"""
    build_merra2_file_url(config::MetSourceConfig, collection_key::String, year::Int, month::Int, day::Int) -> String

Construct the OPeNDAP URL for a specific MERRA-2 file.
"""
function build_merra2_file_url(config::MetSourceConfig, collection_key::String,
                                year::Int, month::Int, day::Int)
    base = get(config.access, "base_url", "")
    coll = config.collections[collection_key]
    stream = merra2_stream(year)
    datestr = string(year, lpad(month, 2, '0'), lpad(day, 2, '0'))
    filename = "MERRA2_$(stream).$(coll.collection_name).$(datestr).nc4"
    return "$base/$(coll.dataset)/$(year)/$(lpad(month, 2, '0'))/$filename"
end

"""
    load_vertical_coefficients(config::MetSourceConfig; FT=Float64) -> (A::Vector{FT}, B::Vector{FT})

Load hybrid sigma-pressure A/B coefficients from the file referenced in the
met source configuration. Works for any met source (ERA5, GEOS-FP, MERRA-2)
because every source's TOML has a `[vertical]` section pointing to its
coefficient file.

The returned vectors have length `n_interfaces` (= n_levels + 1).

# Example
```julia
config = default_met_config("era5")
A, B = load_vertical_coefficients(config)   # 138-element vectors
```
"""
function load_vertical_coefficients(config::MetSourceConfig; FT::Type{<:AbstractFloat}=Float64)
    coeff_path = config.vertical.coefficients_file
    isempty(coeff_path) && error("No coefficients_file specified in $(config.name) vertical config")

    if !isabspath(coeff_path)
        coeff_path = normpath(joinpath(default_config_dir(), "..", coeff_path))
    end
    isfile(coeff_path) || error("Vertical coefficients file not found: $coeff_path")

    raw = TOML.parsefile(coeff_path)
    coeffs = raw["coefficients"]
    A = FT.(coeffs["a"])
    B = FT.(coeffs["b"])

    expected = config.vertical.n_interfaces
    if length(A) != expected
        @warn "A coefficient count $(length(A)) ≠ expected $expected interfaces for $(config.name)"
    end
    if length(B) != expected
        @warn "B coefficient count $(length(B)) ≠ expected $expected interfaces for $(config.name)"
    end

    return A, B
end

"""
    build_vertical_coordinate(config::MetSourceConfig; FT=Float64, level_range=nothing)

Construct a `HybridSigmaPressure` from a met source configuration.
Optionally pass `level_range` (e.g., `50:137`) to select a subset of levels.

This is the universal entry point for building the vertical coordinate
regardless of met source — the same code works for ERA5 (137 levels),
GEOS-FP (72 levels), or MERRA-2 (72 levels).

# Example
```julia
config = default_met_config("geosfp")
vc = build_vertical_coordinate(config)   # 72-level HybridSigmaPressure
```
"""
function build_vertical_coordinate(config::MetSourceConfig;
                                    FT::Type{<:AbstractFloat}=Float64,
                                    level_range=nothing)
    A, B = load_vertical_coefficients(config; FT)
    if level_range !== nothing
        first_idx = first(level_range)
        last_idx  = last(level_range) + 1   # +1 because we need interfaces
        A = A[first_idx:last_idx]
        B = B[first_idx:last_idx]
    end
    return HybridSigmaPressure(A, B)
end

export VerticalConfig, VarMapping, CollectionInfo, MetSourceConfig
export load_met_config, load_canonical_config, default_met_config
export validate_met_config, merra2_stream
export load_vertical_coefficients, build_vertical_coordinate
export build_opendap_url, build_merra2_file_url
