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
#
# # Common grid — all met drivers will be regridded to this
# grid = LatitudeLongitudeGrid(Float64;
#     Nx = 360, Ny = 180, Nz = 72,
#     lon_start = -180.0, lat_start = -90.0,
#     vertical_coordinate = HybridSigmaPressure(Nz=72)
# )
#
# # Common tracer setup — uniform initial condition
# tracers = TracerFields(grid, (:CO2,))
# fill!(interior(tracers[:CO2]), 400.0e-6)  # 400 ppm
#
# # Common time window
# t_start = DateTime(2024, 3, 1, 0, 0, 0)
# t_end   = DateTime(2024, 3, 8, 0, 0, 0)   # one week
# ```

# ### Step 2: Run with each met driver
#
# ```julia
# results = Dict{String, Any}()
#
# for met_source in ["geosfp", "merra2", "era5"]
#     config_path = joinpath(@__DIR__, "../../config/met_sources/$(met_source).toml")
#
#     met = MetDataSource(Float64, config_path)
#
#     model = TransportModel(;
#         grid       = grid,
#         met_data   = met,
#         advection  = SlopesAdvection(),
#         diffusion  = BoundaryLayerDiffusion(),
#         Δt         = 1800.0,   # 30-minute time step
#     )
#
#     # Deep-copy tracers so each run starts from the same initial condition
#     run_tracers = deepcopy(tracers)
#
#     # Run forward model
#     run!(model, run_tracers, t_start, t_end)
#
#     results[met_source] = interior(run_tracers[:CO2]) |> copy
# end
# ```

# ### Step 3: Compare results
#
# Key diagnostics to examine:
#
# ```julia
# # Global mean column CO2 (proxy for XCO2)
# for (name, field) in results
#     col_mean = mean(field, dims=(1,2,3))
#     println("$name: global mean = $(col_mean) ppm")
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
# # Run adjoint with different met drivers
# for met_source in ["geosfp", "merra2", "era5"]
#     gradient = compute_gradient(model, cost_function, met_source)
#     # Compare gradient maps across met drivers...
# end
# ```
#
# This provides a rigorous framework for estimating "transport model
# uncertainty" in flux inversions — a quantity that is typically estimated
# ad hoc by comparing TM5 and GEOS-Chem results (confounding model
# structural differences with met driver differences).
