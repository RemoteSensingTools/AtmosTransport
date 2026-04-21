# Developer notes

Technical reference notes migrated from `docs_legacy/src/developer/` during
the pre-main cleanup. Each note documents a specific audit, comparison, or
deep-dive that is too narrow for `docs/reference/` but too valuable to
lose to git history. Plan 20 (documentation overhaul) will decide which
of these graduate into Layer 3 (`docs/src/under_the_hood/`) and which stay
as developer references.

## Contents

**Meteorology / preprocessing:**
- [ERA5_LNSP_MASS_DRIFT_SPATIAL.md](ERA5_LNSP_MASS_DRIFT_SPATIAL.md) — spatial diagnostics of ERA5 LNSP → ps drift; supports the global-mean-ps fix in `preprocess_spectral_v4_binary.jl`.
- [ERA5_SPECTRAL_HUMIDITY_CONSISTENCY.md](ERA5_SPECTRAL_HUMIDITY_CONSISTENCY.md) — humidity / pressure consistency when converting spectral ERA5 to grid.
- [GEOSIT_C180_GCHP_ATMOS_VARIABLE_MAPPING.md](GEOSIT_C180_GCHP_ATMOS_VARIABLE_MAPPING.md) — variable-by-variable mapping from GEOS-IT C180 (GCHP 14.7.0) into AtmosTransport's canonical names.

**GCHP / TM5 alignment:**
- [GCHP_ADVECTION_TIMING_NOTES.md](GCHP_ADVECTION_TIMING_NOTES.md) — GCHP C720/C180 advection timing baseline.
- [GCHP_C180_FORTRAN_PARITY_AUDIT.md](GCHP_C180_FORTRAN_PARITY_AUDIT.md) — line-by-line parity audit against GCHP Fortran for `catrine_geosit_c180_gchp.toml`.
- [PANEL_BOUNDARY_HALO_DEEP_DIVE.md](PANEL_BOUNDARY_HALO_DEEP_DIVE.md) — panel-boundary halo treatment in GCHP (FV3) vs AtmosTransport.
- [PBL_CONVECTION_COMPARISON_OVERVIEW.md](PBL_CONVECTION_COMPARISON_OVERVIEW.md) — three-way comparison of PBL + convection between AtmosTransport, GCHP, and TM5.
- [TM5_ADVECTION_TRACER_MASS_AND_MULTITRACER.md](TM5_ADVECTION_TRACER_MASS_AND_MULTITRACER.md) — how TM5 handles tracer mass vs mixing ratio and multi-tracer advection.
- [advection_massflux_gchp_atmos_tm5.md](advection_massflux_gchp_atmos_tm5.md) — three-way mass-flux handling comparison.

**Build / setup:**
- [TM5_LOCAL_SETUP.md](TM5_LOCAL_SETUP.md) — instructions for building TM5 locally for validation.
