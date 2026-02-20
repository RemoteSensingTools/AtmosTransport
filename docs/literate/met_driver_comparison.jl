# # Met Driver Intercomparison Tutorial
#
# One of the key design features of AtmosTransportModel.jl is that **the
# same model code runs with different meteorological driving data** simply
# by pointing to a different TOML configuration file. This enables systematic
# studies of how transport uncertainties from meteorological reanalyses
# propagate into tracer simulations and flux inversions.
#
# ## Motivation
#
# Quantifying met driver uncertainty is essential for inverse modeling of
# greenhouse gas fluxes. Differences in winds, boundary-layer mixing, and
# convection between ERA5, MERRA-2, and GEOS-FP propagate directly into:
#
# - Simulated tracer distributions (XCO₂, XCH₄)
# - Retrieved surface fluxes via 4D-Var
# - Inter-hemispheric exchange times
# - Vertical profiles at validation sites
#
# With legacy models (TM5 uses only ECMWF; GEOS-Chem uses only GEOS),
# disentangling model errors from met driver errors requires running two
# entirely different codebases. AtmosTransportModel.jl eliminates this
# confounding factor.

# ## Example: Three-Way Met Driver Comparison
#
# The workflow below runs the same tracer transport simulation with three
# different meteorological data sources and compares the results.

# ### Step 1: Define common model setup
#
# ```julia
# using AtmosTransportModel
# using AtmosTransportModel.Grids
# using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate
# using Dates
#
# # Build vertical coordinate from the GEOS-FP config (72 levels);
# # ERA5 would give 137 levels — choose whichever matches your comparison.
# config = default_met_config("geosfp")
# vert = build_vertical_coordinate(config; FT=Float64)
#
# # Common grid — all met drivers will be regridded to this
# grid = LatitudeLongitudeGrid(CPU();
#     FT   = Float64,
#     size = (360, 180, n_levels(vert)),
#     longitude = (-180, 180),
#     latitude  = (-90, 90),
#     vertical  = vert)
#
# # Common tracer setup — uniform initial condition (NamedTuple of 3D arrays)
# Nx, Ny, Nz = 360, 180, n_levels(vert)
# c = fill(Float64(400.0e-6), Nx, Ny, Nz)
# tracers = (; CO2 = c)
#
# # Common time window
# t_start = DateTime(2024, 3, 1, 0, 0, 0)
# t_end   = DateTime(2024, 3, 8, 0, 0, 0)   # one week
# ```

# ### Step 2: Run with each met driver
#
# The workflow for each met source follows the same pattern used in
# `scripts/run_era5_edgar_forward.jl`:
# 1. Load met data for each window (3-hourly or 6-hourly)
# 2. Precompute mass fluxes via `compute_air_mass!` and `compute_mass_fluxes!`
# 3. Run `strang_split_massflux!` for TM5-faithful advection
#
# ```julia
# using AtmosTransportModel.Advection: build_geometry_cache, allocate_massflux_workspace,
#     compute_air_mass!, compute_mass_fluxes!, strang_split_massflux!
#
# results = Dict{String, Any}()
#
# for met_name in ["geosfp", "merra2", "era5"]
#     cfg = default_met_config(met_name)
#     vc  = build_vertical_coordinate(cfg; FT=Float64)
#
#     g = LatitudeLongitudeGrid(CPU();
#         FT=Float64, size=(360, 180, n_levels(vc)),
#         longitude=(-180, 180), latitude=(-90, 90), vertical=vc)
#
#     # Allocate arrays (zero-allocation inner loop)
#     Δp  = zeros(Float64, 360, 180, n_levels(vc))
#     m   = similar(Δp)
#     am  = zeros(Float64, 361, 180, n_levels(vc))
#     bm  = zeros(Float64, 360, 181, n_levels(vc))
#     cm  = zeros(Float64, 360, 180, n_levels(vc)+1)
#     gc  = build_geometry_cache(g, Δp)
#     ws  = allocate_massflux_workspace(m, am, bm, cm)
#
#     run_tracers = (; CO2 = copy(c))  # deep copy per met source
#
#     # ... per-timestep loop: load met → compute fluxes → strang_split_massflux! ...
#
#     results[met_name] = copy(run_tracers.CO2)
# end
# ```

# ### Step 3: Compare results
#
# Key diagnostics to examine:
#
# ```julia
# using Statistics: mean
#
# # Global mean mixing ratio
# for (name, field) in results
#     println("$name: global mean = $(mean(field)) ppm")
# end
#
# # Zonal mean cross-sections
# for (name, field) in results
#     zonal = mean(field, dims=1)  # average over longitude
#     # Plot latitude-altitude cross-section...
# end
#
# # Differences
# diff_geos_merra = results["geosfp"] .- results["merra2"]
# diff_geos_era5  = results["geosfp"] .- results["era5"]
# diff_merra_era5 = results["merra2"] .- results["era5"]
# ```

# ## What to Compare
#
# ### Horizontal Transport (Winds)
# - Zonal mean u, v differences
# - Implications for inter-hemispheric exchange
# - Storm-track representation
#
# ### Vertical Transport (Omega)
# - Tropical upwelling (Brewer-Dobson circulation)
# - Subsidence patterns
# - Critical for CO₂ seasonal cycle amplitude
#
# ### Boundary Layer Mixing (Kz / kh)
# - PBL height differences
# - Mixing efficiency near the surface
# - **Directly impacts surface flux inversions** — this is often the
#   dominant source of transport error for ground-based measurements
#
# ### Deep Convection (Mass Fluxes)
# - Tropical convective redistribution
# - Critical for CH₄ and CO vertical profiles
# - Often the most poorly constrained transport process
#
# ### Surface Pressure
# - Mass budget consistency across reanalyses
# - Affects column-average quantities (XCO₂)

# ## Regridding Considerations
#
# The three met sources have different native grids:
#
# | Source  | Resolution      | Lon start | Lat direction |
# |:--------|:----------------|:----------|:--------------|
# | GEOS-FP | 0.3125° × 0.25° | -180°     | S → N         |
# | MERRA-2 | 0.625° × 0.5°   | -180°     | S → N         |
# | ERA5    | 0.25° × 0.25°   | 0°        | N → S         |
#
# When comparing, either:
# 1. **Regrid to a common grid** (e.g., 2° × 2°) using conservative regridding
# 2. **Compare area-weighted statistics** (global means, zonal means, vertical profiles)
#
# The model's `Regridding` module handles both conservative and bilinear
# regridding. See `src/Regridding/Regridding.jl`.

# ## Extension: Adjoint Sensitivity to Met Drivers
#
# The adjoint model can quantify how met driver differences affect the
# gradient of a cost function (e.g., in a CO₂ flux inversion):
#
# ```julia
# # Each forward operator (advect_x_massflux!, etc.) has a paired adjoint;
# # running the adjoint with different met drivers quantifies sensitivity.
# for met_name in ["geosfp", "merra2", "era5"]
#     # ... run adjoint_strang_split_massflux! with same met data ...
#     # Compare adjoint gradient maps across met drivers...
# end
# ```
#
# This provides a rigorous framework for estimating "transport model
# uncertainty" in flux inversions — a quantity that is typically estimated
# ad hoc by comparing TM5 and GEOS-Chem results (confounding model
# structural differences with met driver differences).
