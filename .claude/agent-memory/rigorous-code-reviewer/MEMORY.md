# Rigorous Code Reviewer - Memory Index

This directory holds review notes from prior code-review sessions. Not
all are current. Trust each entry only after verifying its file:line
citations against the current `src/` tree.

## TM5 Fortran reference knowledge (durable, external)

These describe TM5 Fortran source code at `deps/tm5/base/src/` (external
reference). Physics principles and Fortran line citations remain valid
regardless of AtmosTransport refactors.

- [TM5 cm boundary invariant](tm5_cm_boundary_invariant.md) — cm=0 at surface/TOA enforced by dynam0 construction + dynamw/dynamwm fatal assertions
- [TM5 Z-advection CFL analysis](tm5_z_advection_analysis.md) — TM5 never allows Z-CFL >= 1 (halves ndyn instead)
- [TM5 TimeInterpolation trace](tm5_time_interpolation_trace.md) — Full TM5 field interpolation architecture

## Historical reviews (STALE — verify before trusting)

Principles may still apply; file:line citations are known-stale. Each
file carries a STALE header with specifics.

- [TM5 Check_CFL outer loop design](tm5_check_cfl_outer_loop.md) — Full TM5 Check_CFL trace; TM5 Fortran section durable, Julia mapping table cites `src_legacy/`
- [CS cm diagnosis convention](cs_cm_diagnosis_convention.md) — CS vs LL cm formula distinction (citations drifted, concept durable)
- [CS Poisson balance review](cs_poisson_balance_review.md) — Global multi-panel CG balance; subsystem still live, specific claims unverified
- [CS runner flux scaling](cs_runner_flux_scaling.md) — Three named bugs in `scripts/run_cs_transport.jl`; **bugs may or may not still exist — re-verify before trusting OR fixing**
- [V4 flux interpolation NaN analysis](v4_interpolation_nan_analysis.md) — Runtime cm recomputation amplifying CFL; runtime path now in `src_legacy/`
- [Preprocessing pipeline review](preprocessing_pipeline_review.md) — 14-file review; scale/content drifted
- [TM5 polar audit findings](tm5_polar_audit_findings.md) — TM5 Fortran citations durable; AtmosTransport-side claims stale

## Removed

The following memory files were removed during 2026-04-21 cleanup
because they described AtmosTransport code that now lives in
`src_legacy/` or `scripts_legacy/`, or functions that no longer exist.
Keeping them risked misleading future agents about current behavior.
Git history preserves the content.

- `ll_advection_chain_review.md` — cites `run_loop.jl`, `physics_phases.jl`, `run_helpers.jl`, `mass_flux_advection.jl` (all `src_legacy/`)
- `era5_ll_algorithmic_trace.md` — legacy runtime trace (`src_legacy/Models/`)
- `cs_transport_trace.md` — 8 of 10 file citations dead (legacy CS kernels + legacy Models)
- `cs_pipeline_final_review.md` — cites `normalize_regridded_intensive!` which no longer exists; `cs_poisson_balance.jl` line 524 reference invalid (line numbers drifted post-refactor)
- `fused_spectral_v4_review.md` — cites `scripts/preprocessing/preprocess_spectral_v4_binary.jl` (now in `scripts_legacy/preprocessing/`)
- `z_column_kernel_review.md` — `_massflux_z_column_kernel!` removed entirely
- `cfl_subcycling_architecture.md` — describes legacy "outer-loop pilot" CFL; TM5 content already in `tm5_check_cfl_outer_loop.md`; Julia citations all `src_legacy/`
- `pole_bm_clamp_review.md` — `clamp_bm_at_poles!` function removed entirely

For the most current view of a subsystem, prefer:

- [`src/Operators/Advection/README.md`](../../../src/Operators/Advection/README.md) — advection, CFL subcycling
- [`src/Operators/Convection/README.md`](../../../src/Operators/Convection/README.md) — convection (plan 18 + 22D)
- [`src/Preprocessing/`](../../../src/Preprocessing/) source — preprocessing pipeline
- [`src/State/README.md`](../../../src/State/README.md) — state storage per topology
