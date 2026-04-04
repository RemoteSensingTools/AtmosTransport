---
name: CFL subcycling architecture analysis
description: Red team analysis of TM5-style outer-loop vs per-direction CFL subcycling for ERA5 LL — outer loop is wrong approach, per-direction is correct but buggy
type: reference
---

## Three CFL Approaches in Codebase

1. **TM5 outer-loop (Check_CFL)**: advectm_cfl.F90:154-302. Runs FULL mass-only Strang pilot evolving m. If any direction fails, halves ndyn globally and rescales ALL fluxes. Also re-interpolates winds temporally via Setup_MassFlow. X has per-row nloop subcycling (dynamum), Y/Z have no inner subcycling — reject on CFL>=1.

2. **Julia find_mass_cfl_refinement** (mass_cfl_pilot.jl): GPU port of TM5 pilot. Returns global r. DISABLED since commit 939ee22 — r=16 at 75s/win too slow. Has bugs: global mutable state (`Ref{Any}`), O(r*n_sub*6) GPU sync points.

3. **Julia flux-remaining per-direction** (mass_flux_advection.jl:1138-1233): Correct architecture but buggy. Commit 9bb1874 has -1408% mass loss and NaN. Root cause per commit 0b5f556: global fraction too conservative, extreme polar cells deplete → CFL grows → frac shrinks → 100-iter cap → undelivered flux → mass loss cascade.

## Key Finding: TM5 is NOT Just a Pre-Check

TM5 Check_CFL evolves m through the FULL Strang sequence (XYZ...ZYX), checking CFL at every sweep against the then-current m. A pre-check at the start misses cascade failures (X depletes polar cells → Z-CFL spikes against depleted m). The `find_mass_cfl_refinement` correctly replicates this, but a "simple CFL check" would not.

## Flux Division Approximation

Dividing am/bm/cm by r IS mathematically equivalent to running dynam0 with ndyn/r IF winds are constant. But TM5's Setup_MassFlow re-interpolates winds temporally for each sub-interval, getting different pu/pv. For 6-hourly ERA5 this is likely a small effect (linear interp), but it's an unverified approximation.

## Correct Fix Direction

Per-direction subcycling is architecturally superior (doesn't penalize X when only Z has CFL issues). The fix is per-CELL safe fractions instead of the current global `frac = cfl_limit / max_cfl`. This way 99.99% of cells proceed normally while extreme polar/thin-layer cells get the subcycling they need.

## ERA5 CFL Magnitudes

- X (polar, 0.5deg): CFL > 20 at high latitudes (J=1,2,360,361)
- Y: typically CFL < 1 everywhere
- Z (L137 merged 68L): CFL 2-10 at thin top layers
- Z (L137 unmerged): CFL 4-19 at layers 130-137

## Performance

n_sub = 24 (dt=900s, met_interval=21600s). Baseline ~5s/win on GPU.
Global r=16: 64 substeps, 75s/win (15x overhead).
Per-direction: subcycles only where needed, ~0 overhead for well-behaved cells.
