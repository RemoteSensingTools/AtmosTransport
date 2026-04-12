using Dates

# ---------------------------------------------------------------------------
# Abstract met driver type hierarchy
#
# All met data ingestion is abstracted behind AbstractMetDriver.
# Multiple dispatch on (driver, grid) resolves how met data is loaded,
# and on (driver) for metadata queries.
#
# Two branches:
#   AbstractRawMetDriver      — reads raw winds, computes mass fluxes on the fly
#   AbstractMassFluxMetDriver — reads pre-computed mass fluxes directly
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for all meteorological data drivers.

# Interface contract

Concrete subtypes must implement:

    total_windows(driver)           → Int
    window_dt(driver)               → FT   (seconds per met window)
    steps_per_window(driver)        → Int   (advection sub-steps per window)
    load_met_window!(buf, driver, grid, win_index)  → nothing

Optional:
    met_interval(driver)            → FT   (time between met updates, seconds)
    start_date(driver)              → Date  (simulation start date for output timestamps)
    date_range(driver)              → (start_date, end_date)
"""
abstract type AbstractMetDriver{FT} end

"""
$(TYPEDEF)

Met driver that reads raw wind fields and computes mass fluxes on the fly.
Examples: ERA5MetDriver, GEOSFPWindMetDriver.
"""
abstract type AbstractRawMetDriver{FT} <: AbstractMetDriver{FT} end

"""
$(TYPEDEF)

Met driver that reads pre-computed mass fluxes (am, bm, cm, m).
Examples: PreprocessedLatLonMetDriver, GEOSFPCubedSphereMetDriver.
"""
abstract type AbstractMassFluxMetDriver{FT} <: AbstractMetDriver{FT} end

# --- Interface stubs (error on unimplemented) ---

function total_windows end
function window_dt end
function steps_per_window end
function load_met_window! end

"""Default met interval: same as window_dt."""
met_interval(d::AbstractMetDriver) = window_dt(d)

"""Default start date: 2000-01-01 (override per driver for correct output timestamps)."""
start_date(d::AbstractMetDriver) = Dates.Date(2000, 1, 1)

"""Default transport mass basis for met drivers."""
mass_basis(::AbstractMetDriver) = :moist

"""Fallback: QV not available for this driver. Override per driver to enable dry-air transport."""
load_qv_window!(qv, driver::AbstractMetDriver, grid, win) = false
