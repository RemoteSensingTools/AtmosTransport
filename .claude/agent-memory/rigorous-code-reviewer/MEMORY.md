# Rigorous Code Reviewer - Memory Index

- [TM5 cm boundary invariant](tm5_cm_boundary_invariant.md) — cm=0 at surface/TOA enforced by dynam0 construction + dynamw/dynamwm fatal assertions
- [TM5 Z-advection CFL analysis](tm5_z_advection_analysis.md) — TM5 never allows Z-CFL >= 1 (halves ndyn instead), prognostic slopes don't help at gamma > 1
- [CFL subcycling architecture](cfl_subcycling_architecture.md) — Outer-loop pilot is wrong approach; per-direction flux-remaining is correct but buggy (per-cell fractions needed)
- [TM5 Check_CFL outer loop design](tm5_check_cfl_outer_loop.md) — Full TM5 Check_CFL trace + Julia mapping for ERA5 LL global refinement
- [Z-column kernel review](z_column_kernel_review.md) — Two critical bugs: stale-value single-pass + multi-tracer double mass update
- [LL advection chain review](ll_advection_chain_review.md) — Full call chain trace from run! to GPU kernels, TM5 comparison, missing n_sub cap bug
- [Fused spectral v4 review](fused_spectral_v4_review.md) — APPROVED: faithful copy from two refs, Float64-only improvement, binary layout matches reader
- [CS transport full trace](cs_transport_trace.md) — Complete run! to output trace for CS grid: Strang/LinRood/GCHP paths, file:line refs, unit conventions
- [Pole bm clamp review](pole_bm_clamp_review.md) — clamp_bm_at_poles! correctness: indexing OK, but clamps inflow too + cm not recomputed
