"""
    IO

Input/output for meteorological data, diagnostics, and configuration.

# Met data sources (TOML-configured)

All met data sources are configured via TOML files in `config/met_sources/`:
- `geosfp.toml` — NASA GEOS-FP (OPeNDAP, no auth, near real-time)
- `merra2.toml` — NASA MERRA-2 (OPeNDAP, Earthdata auth, 1980–present)
- `era5.toml`   — ECMWF ERA5 (CDS API, ECMWF auth, 1940–present)

Canonical variable names are defined in `config/canonical_variables.toml`.
Adding a new met data source requires only a new TOML file — no Julia
code changes for OPeNDAP or local-file sources.

# Convenience constructors

```julia
met = GEOSFPMetData(; FT=Float64)   # loads geosfp.toml
met = MERRAMetData(; FT=Float64)    # loads merra2.toml
met = ERA5MetData(; FT=Float64)     # loads era5.toml

# Or point to any TOML:
met = MetDataSource(Float64, "path/to/my_source.toml")
```

# Output writers

- `NetCDFOutputWriter` — schedule-based NetCDF output
- Extensible via `AbstractOutputWriter`

# Configuration

- TOML-based run configuration (`configuration.jl`)
"""
module IO

using DocStringExtensions

export AbstractMetData
export MetDataSource, GEOSFPMetData, MERRAMetData, ERA5MetData
export MetSourceConfig, VarMapping, CollectionInfo, VerticalConfig
export load_met_config, load_canonical_config, default_met_config, validate_met_config
export load_vertical_coefficients, build_vertical_coordinate
export get_field, has_variable, native_name, collection_for
export read_met!, write_output!, initialize_output!, prepare_met_for_physics, compute_continuity_omega
export protocol, time_interval, source_name
export merra2_stream, build_opendap_url, build_merra2_file_url
export canonical_variables, canonical_units, canonical_required, canonical_dimensions
export AbstractOutputWriter, AbstractOutputSchedule
export NetCDFOutputWriter, TimeIntervalSchedule, IterationIntervalSchedule
export TemporalInterpolator, interpolation_weight
export load_configuration

# Core abstract interface (must come first)
include("abstract_met_data.jl")

# TOML config loader (must come before variable_mapping and met_data_source)
include("met_source_config.jl")

# Variable mapping (now reads from canonical_variables.toml)
include("variable_mapping.jl")

# Generic config-driven met data reader + convenience constructors
include("met_data_source.jl")

# read_met! implementation (OPeNDAP + local file)
include("met_reader.jl")

# Met-to-physics bridge (stagger winds, prepare NamedTuple for operators)
include("met_fields_bridge.jl")

# Legacy per-source files (now just reference comments)
include("era5_met_data.jl")
include("merra2_met_data.jl")
include("geosfp_met_data.jl")

# Temporal interpolation
include("temporal_interpolation.jl")

# Output
include("output_writers.jl")

# Run configuration
include("configuration.jl")

end # module IO
