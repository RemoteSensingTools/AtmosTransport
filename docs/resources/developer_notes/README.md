# Archived developer notes

These dated investigations are retained for scientific provenance. They are
not part of the maintained newcomer manual and are intentionally excluded from
the documentation navigation and strict link checks. File paths, run status,
and implementation details inside them may be historical.

Use `docs/src/` for current contracts and workflows. Consult these notes when
tracing the evidence behind GEOS mass-flux decisions or the original CATRINE
validation protocol.

## Contents

Meteorology and preprocessing:

- `ERA5_LNSP_MASS_DRIFT_SPATIAL.md` — spatial diagnostics supporting the
  global-mean surface-pressure correction.
- `ERA5_SPECTRAL_HUMIDITY_CONSISTENCY.md` — humidity and pressure consistency
  in spectral ERA5 conversion.
- `GEOSIT_C180_GCHP_ATMOS_VARIABLE_MAPPING.md` — GEOS-IT C180 variable mapping.
- `GEOS_MASS_FLUX_UTLS_FINGERING.md` — investigation of native GEOS mass-flux
  artifacts in the UTLS.
- `GEOS_PREPROCESSING_MASS_BALANCE.md` — mass-flux balance and global dry-air
  conservation analysis.
- `CATRINE_DEC2021_VALIDATION.md` — original December 2021 validation protocol.

GCHP, TM5, and numerical alignment:

- `GCHP_ADVECTION_TIMING_NOTES.md`
- `GCHP_C180_FORTRAN_PARITY_AUDIT.md`
- `PANEL_BOUNDARY_HALO_DEEP_DIVE.md`
- `PBL_CONVECTION_COMPARISON_OVERVIEW.md`
- `TM5_ADJOINT_CONTROLS.md`
- `TM5_ADVECTION_TRACER_MASS_AND_MULTITRACER.md`
- `advection_massflux_gchp_atmos_tm5.md`

Build and setup:

- `TM5_LOCAL_SETUP.md`
