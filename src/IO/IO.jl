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
export load_initial_conditions!
export MetDataSource, GEOSFPMetData, MERRAMetData, ERA5MetData
export MetSourceConfig, VarMapping, CollectionInfo, VerticalConfig
export load_met_config, load_canonical_config, default_met_config, validate_met_config
export load_vertical_coefficients, build_vertical_coordinate
export get_field, has_variable, native_name, collection_for
export read_met!, write_output!, initialize_output!, prepare_met_for_physics, compute_continuity_omega
export protocol, time_interval, source_name
export merra2_stream, build_opendap_url, build_merra2_file_url
export canonical_variables, canonical_units, canonical_required, canonical_dimensions
export AbstractOutputWriter, AbstractOutputSchedule, AbstractOutputGrid, LatLonOutputGrid
export NetCDFOutputWriter, BinaryOutputWriter, TimeIntervalSchedule, IterationIntervalSchedule
export finalize_output!, convert_binary_to_netcdf
export TemporalInterpolator, interpolation_weight
export load_configuration, build_model_from_config, get_pending_ic,
       finalize_ic_vertical_interp!, has_deferred_ic_vinterp
export GeosFPCubedSphereTimestep
export read_geosfp_cs_timestep, to_haloed_panels, cgrid_to_staggered_panels
export read_geosfp_cs_cmfmc, read_geosfp_cs_surface_fields
export geosfp_cs_url, geosfp_cs_asm_url, geosfp_cs_tavg_url, geosfp_cs_local_path
export inspect_geosfp_cs_file, read_geosfp_cs_grid_info

# Binary readers (mmap-based)
export AbstractBinaryReader, MassFluxBinaryReader, CSBinaryReader
export load_window!, load_cs_window!, load_cs_qv_ps_window!, load_edgar_cs_binary, window_count

# File discovery
export find_massflux_shards, find_preprocessed_cs_files
export find_geosfp_cs_files, find_era5_files, ensure_local_cache

# Met driver abstraction (Phase 4)
export AbstractMetDriver, AbstractRawMetDriver, AbstractMassFluxMetDriver
export total_windows, window_dt, steps_per_window, load_met_window!
export load_cmfmc_window!, load_dtrain_window!, load_tm5conv_window!, load_qv_window!, load_surface_fields_window!, load_all_window!, load_physics_window!
export load_qv_and_ps_pair!, load_ps_from_ctm_i1!, load_cx_cy_window!

# Met buffer types
export AbstractMetBuffer, AbstractCPUStagingBuffer
export LatLonMetBuffer, LatLonCPUBuffer, CubedSphereMetBuffer, CubedSphereCPUBuffer
export upload!

# Wind processing utilities
export stagger_winds!, load_era5_timestep, get_era5_grid_info

# Concrete met drivers
export ERA5MetDriver, PreprocessedLatLonMetDriver, GEOSFPCubedSphereMetDriver

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

# Binary output writer (fast sequential writes)
include("binary_output_writer.jl")

# Initial conditions loader (before configuration.jl which uses PendingInitialConditions)
include("initial_conditions.jl")

# Run configuration
include("configuration.jl")

# GEOS-FP native cubed-sphere reader
include("geosfp_cubed_sphere_reader.jl")

# Binary readers (mmap-based, zero-copy)
include("binary_readers.jl")

# File discovery utilities
include("file_discovery.jl")

# Met driver abstraction — abstract types + interface (Phase 4)
include("abstract_met_driver.jl")

# Met buffer types (CPU staging + GPU resident)
include("met_buffers.jl")

# Wind processing utilities (stagger_winds!, load_era5_timestep, etc.)
include("wind_processing.jl")

# Concrete met drivers
include("era5_met_driver.jl")
include("preprocessed_latlon_driver.jl")
include("geosfp_cs_met_driver.jl")

end # module IO
