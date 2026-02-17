# ---------------------------------------------------------------------------
# Abstract met data interface
#
# Interface contract for all met data readers:
#
#   read_met!(met::AbstractMetData, time) → populate internal buffers
#   get_field(met, :u_wind)               → return the u-wind field
#   met_grid(met)                         → return the met data's native grid
#   met_times(met)                        → available time steps
#
# Physics code NEVER calls met data readers directly. Instead, the model
# setup constructs a regridder (if needed) that maps met data fields to
# the model grid before physics operators see them.
# ---------------------------------------------------------------------------

"""
    AbstractMetData{FT}

Supertype for meteorological data readers. Parametric on float type.

# Canonical variable names

All met data types must provide these fields (via `get_field`):
- `:u_wind` — zonal wind [m/s]
- `:v_wind` — meridional wind [m/s]
- `:w_wind` — vertical wind (pressure velocity) [Pa/s]
- `:temperature` — temperature [K]
- `:specific_humidity` — specific humidity [kg/kg]
- `:surface_pressure` — surface pressure [Pa]
- `:diffusivity` — vertical diffusivity Kz [m²/s] (optional)
- `:conv_mass_flux_up` — convective updraft mass flux [kg/m²/s] (optional)
- `:conv_mass_flux_down` — convective downdraft mass flux [kg/m²/s] (optional)
"""
abstract type AbstractMetData{FT} end

"""Read met data for the given `time` into internal buffers."""
function read_met! end

"""Return the field for canonical variable `name` (a Symbol)."""
function get_field end

"""Return the native grid of the met data."""
function met_grid end

"""Return available time steps as a vector of Float64 (seconds since epoch)."""
function met_times end
