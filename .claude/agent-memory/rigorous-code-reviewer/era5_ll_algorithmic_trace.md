---
name: ERA5 LL algorithmic trace
description: Complete step-by-step execution trace for ERA5 lat-lon transport path from run! through all physics phases
type: reference
---

Reference document: Complete algorithmic flow for ERA5 lat-lon (preprocessed_latlon) transport.
See `/home/cfranken/code/gitHub/AtmosTransportModel/docs/ERA5_LL_TRANSPORT_TRACE.md` for the full trace.

Key structural facts discovered during trace:
- LL path has NO air mass allocation (allocate_air_mass returns nothing); m_ref/m_dev live in LatLonMetBuffer
- LL transport is purely MOIST basis (TM5 convention): rm = c * m_moist; dry correction only at output
- cm is pre-computed by preprocessor and stored in binary; NOT computed at runtime for LL (unlike CS)
- Strang split order: X-Y-Z-Z-Y-X with per-direction CFL subcycling (cap at 50)
- Z-advection runs TWICE in Strang (inner pair of symmetric sweep)
- Multi-tracer: each tracer advected independently with m_save/restore between tracers
- v4 flux deltas: per-substep linear interpolation of am/bm/cm with cm recomputed from divergence
- v3 (no deltas): constant fluxes with cm clamped to CFL<0.95
- Convection runs ONCE per window AFTER all advection substeps (not per-substep)
- Emissions and diffusion operate in VMR space (rm/m_ref roundtrip)
- Output uses DRY mass: c_dry = rm / (m_ref * (1-QV))
