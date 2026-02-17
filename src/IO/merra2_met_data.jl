# ---------------------------------------------------------------------------
# MERRA-2 met data — now configured via config/met_sources/merra2.toml
#
# This file is kept for reference. The actual implementation is in
# met_data_source.jl which provides the generic MetDataSource type.
# Use MERRAMetData(; FT=Float64) to construct — it loads merra2.toml.
#
# NASA MERRA-2 reanalysis:
# - Grid: regular lat-lon 0.5° × 0.625°
# - Vertical: 72 hybrid sigma-pressure levels (same as GEOS-FP)
# - Temporal: 3-hourly (3D), hourly (2D)
# - Access: OPeNDAP via goldsmr5.gesdisc.eosdis.nasa.gov (Earthdata auth)
# - Native variable names: U, V, OMEGA, T, QV, PS, DELP, KH, PLE, ...
# - Streams: 100 (1980-91), 200 (1992-2000), 300 (2001-10), 400 (2011+)
# ---------------------------------------------------------------------------
# MERRAMetData is now a convenience constructor defined in met_data_source.jl.
# It returns MetDataSource{FT} loaded from merra2.toml.
