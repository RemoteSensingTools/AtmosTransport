# ---------------------------------------------------------------------------
# ERA5 met data — now configured via config/met_sources/era5.toml
#
# This file is kept for reference. The actual implementation is in
# met_data_source.jl which provides the generic MetDataSource type.
# Use ERA5MetData(; FT=Float64) to construct — it loads era5.toml.
#
# ECMWF ERA5 reanalysis:
# - Grid: regular lat-lon (e.g. 0.25° × 0.25°), lon starts at 0°, lat N→S
# - Vertical: 137 hybrid sigma-pressure levels
# - Temporal: hourly
# - Access: Copernicus CDS API (requires ~/.cdsapirc)
# - Native variable names: u, v, w, t, q, sp, z, ...
# ---------------------------------------------------------------------------
# ERA5MetData is now a convenience constructor defined in met_data_source.jl.
# It returns MetDataSource{FT} loaded from era5.toml.
