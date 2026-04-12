# ---------------------------------------------------------------------------
# GEOS-FP met data — now configured via config/met_sources/geosfp.toml
#
# This file is kept for reference. The actual implementation is in
# met_data_source.jl which provides the generic MetDataSource type.
# Use GEOSFPMetData(; FT=Float64) to construct — it loads geosfp.toml.
#
# NASA GEOS Forward Processing:
# - Grid: lat-lon 0.3125° × 0.25° (1152 × 721)
# - Vertical: 72 hybrid sigma-pressure levels (same as MERRA-2)
# - Temporal: 3-hourly (3D), hourly (2D)
# - Access: OPeNDAP at opendap.nccs.nasa.gov (NO authentication required)
# - Native variable names: u, v, omega, t, qv, ps, delp, pl, h, phis, kh, ple
# ---------------------------------------------------------------------------
# GEOSFPMetData is now a convenience constructor defined in met_data_source.jl.
# It returns MetDataSource{FT} loaded from geosfp.toml.
