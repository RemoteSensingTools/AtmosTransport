# Completed-experiment run configs

Run configurations for experiments that are done — plan studies,
scheme comparisons, one-off validations. Kept for reference and
reproducibility but not part of any active workflow.

Active production / validation configs live in `config/runs/`.

## Groupings

**v2-era (pre-restructure, reference src_v2 paths that no longer exist):**
- `era5_cs_c90_v2_*`, `era5_latlon_transport_v2_*`, `era5_latlon_v2_*`,
  `era5_reduced_gaussian_transport_v2_*`, `era5_reduced_transport_v2_*`,
  `era5_rg_v2_*`.

**F64 debug + mass-conservation validation (Invariant 11, now stable):**
- `era5_f64_debug_moist*`, `era5_f64_debug_v4_startCO2`,
  `era5_mass_conservation_test*`.

**Experimental advection schemes (not shipped):**
- `*_prather_*`, `*_qspace_*`.

**Plan 14 vertical remap / perremap studies (integrated into unified
pipeline):**
- `*_vremap_*`, `*_perremap_*`, `test_vertical_remap*`.

**Advection-only diagnostics:**
- `era5_v4_advonly_*`, `era5_v4_diag_*`, `era5_v4_test_*`,
  `catrine_c180_dryfix_*_advonly`.

**Catrine 2-day scheme comparison suite (done, results captured):**
- `catrine_2day_cs_*`, `catrine_2day_ll*`, `catrine_2day_o090_*`,
  `catrine_2day_rg_*`.

**Catrine linrood intermediate variants (canonical linrood kept in
`config/runs/catrine_geosit_c180_linrood.toml`):**
- `catrine_geosit_c180_linrood_noEmissions*`,
  `catrine_geosit_c180_linrood_advectionOnly`,
  `catrine_geosit_c180_linrood_advonly`,
  `catrine_geosit_c180_linrood_7d`,
  `catrine_geosit_c180_linrood_v4_7d*`,
  `catrine_geosit_c180_linrood_conv`,
  `catrine_geosit_c180_linrood_diff`.

**Older ERA5 catrine variants (superseded by current spectral v4 chain):**
- `catrine_era5_spectral_1week`,
  `catrine_era5_spectral_dec2021*`,
  `catrine_era5_merged_dec2021_pbl`,
  `catrine_era5_hourly_merged_dec2021`.

**GEOS-FP / GEOS-IT intermediate variants (superseded by current
`geosfp_c720_june2024.toml` and `geosit_c180_june2023.toml`):**
- `catrine_geosfp_cs`, `geosfp_c720_2day_linrood`,
  `geosfp_c720_flat_test`, `geosfp_c720_june2024_{fixed,merge,smalldt}`,
  `geosfp_cs_*`, `geosit_c180_1day_test`,
  `geosit_c180_advection_only`,
  `catrine_geosit_c180_{advonly,flat_ic}`.
