# ---------------------------------------------------------------------------
# Abstract met driver types and capability traits
#
# Met drivers are responsible for:
#   1. Reading native meteorological fields
#   2. Reconstructing pressures and layer masses
#   3. Computing basis-aware mass fluxes
#   4. Diagnosing or ingesting vertical fluxes
#   5. Enforcing continuity / closure
#
# The transport core should never see raw met fields.
# Met drivers produce AbstractFaceFluxState + CellState.air_mass.
# ---------------------------------------------------------------------------

using Dates

"""
    AbstractMetDriver

Supertype for all meteorological data drivers.

## Required methods
    total_windows(driver) -> Int
    window_dt(driver) -> FT (seconds per met window)
    steps_per_window(driver) -> Int
    load_transport_window(driver, win_index)
    driver_grid(driver)
    air_mass_basis(driver)
"""
abstract type AbstractMetDriver end
const AbstractDriver = AbstractMetDriver

"""
    AbstractRawMetDriver <: AbstractMetDriver

Reads raw wind fields and computes mass fluxes on the fly.
"""
abstract type AbstractRawMetDriver <: AbstractMetDriver end

"""
    AbstractMassFluxMetDriver <: AbstractMetDriver

Reads pre-computed mass fluxes (am, bm, cm, m).
"""
abstract type AbstractMassFluxMetDriver <: AbstractMetDriver end

# --- Interface stubs ---
function total_windows end
function window_dt end
function steps_per_window end
function load_transport_window end
function driver_grid end
function air_mass_basis end
function flux_interpolation_mode end
function load_met_window! end

met_interval(d::AbstractMetDriver) = window_dt(d)
start_date(::AbstractMetDriver) = Date(2000, 1, 1)

# ---------------------------------------------------------------------------
# Capability traits — what physics the met data supports
# ---------------------------------------------------------------------------

"""Does this driver provide diffusivity fields for boundary-layer diffusion?"""
supports_diffusion(::AbstractMetDriver) = false

"""Does this driver provide convective mass flux / detrainment for convection?"""
supports_convection(::AbstractMetDriver) = false

"""Does this driver provide native vertical mass fluxes (vs diagnosing from continuity)?"""
supports_native_vertical_flux(::AbstractMetDriver) = false

"""How should flux forcing vary within a met window?"""
flux_interpolation_mode(::AbstractMetDriver) = :constant

"""Does this driver provide specific humidity for dry-mass correction?"""
supports_moisture(::AbstractMetDriver) = false

export AbstractMetDriver, AbstractRawMetDriver, AbstractMassFluxMetDriver
export AbstractDriver
export total_windows, window_dt, steps_per_window, load_transport_window, load_met_window!
export driver_grid, air_mass_basis, flux_interpolation_mode
export supports_diffusion, supports_convection
export supports_native_vertical_flux, supports_moisture
