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

"""
    current_time(meteo) -> Float64

Simulation time [s] at the start of the next step. Threaded through
operator `apply!` methods by plan 17 Commit 4:

    apply!(state, meteo, grid, op, dt; workspace)

Every operator that consumes time (`ExponentialDecay` rates,
`ImplicitVerticalDiffusion` Kz refresh, future emission-rate
`StepwiseField`s, etc.) reads `current_time(meteo)` once per call
and passes the resulting scalar to each `update_field!(f, t)`.

# Canonical usage (plan 18 A3)

- **Production**: `meteo = sim::DrivenSimulation`; returns `sim.time`,
  advanced by `sim.time += sim.Δt` at the end of each `step!(sim)`.
  See `src/Models/DrivenSimulation.jl`.
- **Unit tests without a sim**: `meteo = nothing`; returns `0.0`.
- **Legacy driver stub** (`meteo = ::AbstractMetDriver`): returns
  `0.0`. Retained for backward compatibility but should not be
  relied upon — the driver is stateless (struct holds only the
  reader + grid) and cannot provide real time information. Any code
  that previously passed `meteo = sim.driver` silently got `0.0`;
  plan 18 A3 changes `DrivenSimulation.step!` to pass `meteo = sim`
  so the sim's clock is the canonical source.
"""
current_time(::AbstractMetDriver) = 0.0
current_time(::Nothing) = 0.0

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
export current_time
