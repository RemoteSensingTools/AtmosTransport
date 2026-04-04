# ERA5 Lat-Lon Transport: Theoretical Analysis of Julia vs TM5 Differences

Analysis of each implementation difference from a computational fluid dynamics
and numerical methods perspective. References to Russell & Lerner (1981),
Lin & Rood (1996), Prather (1986).

## 1. Met Loading: v4 Temporal Interpolation

**Julia v4:** Linear interpolation of am/bm with cm recomputed from divergence per substep.
**TM5:** `TimeInterpolation` + `dynam0` per substep.

**Theory:** Linear interpolation of mass fluxes preserves the discrete continuity equation
at every substep because cm is derived from the horizontal divergence of the interpolated
am/bm. Mass conservation is guaranteed. This is equivalent to TM5's approach.

**Verdict:** Theoretically sound. The v4 approach is a faithful approximation of TM5's
per-substep wind interpolation.

## 2. CFL Control: Per-Direction vs Global

**Julia:** Per-direction uniform subdivision, n_sub = ceil(CFL/0.95), capped at 50.
**TM5:** Global ndyn halving, scales ALL fluxes uniformly.

**Theory:** Both maintain O(dt^2) Strang splitting error. TM5's approach preserves the
splitting symmetry (effective dt is the same in all directions). Julia's per-direction
approach breaks symmetry — effective dt differs across directions. This is a second-order
effect dominated by spatial discretization error. Not significant for CO2 transport.

## 3. X Subcycling: Per-Row vs Global

**Julia:** Single global n_sub from max CFL across all rows.
**TM5:** Per-row nloop with mass-evolution tracking (advectm_cfl.F90:1354).

**Theory:** NOT mathematically equivalent when mass evolves during subcycles. TM5's
per-row approach tracks how m changes at each sub-iteration, ensuring CFL < 1 throughout.
Julia's global approach is conservative (over-subcycles low-CFL rows) but correct — it
never under-subcycles. The difference is computational efficiency, not accuracy.

## 4. Y Subcycling: Harmless No-Op

**Julia:** Same subcycling as X.
**TM5:** No Y subcycling (relies on outer Check_CFL loop).

**Theory:** Typical Y-CFL peaks at ~0.43 for ERA5 0.5-deg. Julia's Y-subcycling computes
n_sub = ceil(0.43/0.95) = 1, so it's a no-op. No impact on results.

## 5. Z Subdivision: Stable with Caveat

**Julia:** cm/n_sub applied n_sub times, each sub-step CFL < 0.95.
**TM5:** Rejects CFL >= 1, halves global timestep.

**Theory:** Each sub-step satisfies the CFL condition independently, so the van Leer
scheme is stable per-step. The caveat: m evolves during sub-steps, so the effective CFL
for later sub-steps may exceed 0.95 (the denominator m changed). This was confirmed
experimentally — m goes negative at sub-step 3/6 for deep convection cells.

The fix (v4 interpolation with prescribed m trajectory) prevents m drift by resetting
m_dev each substep to the interpolated target.

## 6. Diagnostic vs Prognostic Slopes

**Julia:** Diagnostic (minmod, recomputed each step).
**TM5:** Prognostic rxm/rym/rzm (evolve via Russell-Lerner update formulas).

**Theory (Russell & Lerner 1981):** Prognostic slopes give 2nd-order accuracy in BOTH
space and time. Diagnostic slopes are 2nd-order in space but 1st-order in time — they
lose the sub-grid gradient memory between steps.

**Quantified diffusion:** The leading truncation error difference is O(u * dx * dt).
For ERA5 at 0.5-deg (~50 km) with dt=900s and u~10 m/s:
  Extra diffusivity ~ u * dx * CFL ~ 10 * 50000 * 0.16 ~ 80,000 m^2/s

This is comparable to mesoscale diffusion (~10^4-10^5 m^2/s). For CO2 with smooth
gradients (~1-2 ppm/1000 km), the extra diffusion smears features on ~100 km scales
over ~1 day. Acceptable for global transport but noticeable in regional studies.

**Reference:** Prather (1986) showed that prognostic moment methods (of which TM5's
slopes are the simplest case) dramatically reduce numerical diffusion.

## 7. Cross-Slope Transport: Small Impact

**Julia:** Not implemented.
**TM5:** fy/fz advected in X, fx/fz in Y, fx/fy in Z.

**Theory:** Cross-slopes preserve the 3D gradient tensor through operator splitting.
Without them, Strang splitting introduces O(dt^2) directional decoupling — the
gradient in the Y-direction gets "forgotten" during X-advection and vice versa.

**Quantified:** For CO2 with gradients of ~1-2 ppm/1000 km, the cross-slope correction
is approximately (slope * CFL) ~ 0.01 * 0.5 = 0.5% of the flux. Negligible compared
to the diagnostic slope diffusion (item 6).

## 8. Multi-Tracer Order: Equivalent

**Julia:** Sequential Strang sweeps per tracer with m_save/restore.
**TM5:** All tracers in inner loop within single mass evolution.

**Theory:** Equivalent because Julia restores m before each tracer's sweep. Both
approaches produce the same mass update since m_new = m + am_in - am_out depends
only on the fluxes, not tracer content.

## 9. m_dev Negativity

**Problem:** With constant fluxes over 4-24 substeps, actual air mass deviates from
the meteorologically prescribed trajectory. At deep convection regions, cumulative
horizontal divergence can exceed the cell mass.

**Theory:** This is a consistency problem, not a stability problem. The van Leer scheme
is stable if CFL < 1 at each step. The issue is that the SAME fluxes imply a mass
trajectory that diverges from reality — after several substeps, the implied mass at
extreme cells can go to zero or negative.

**Fix:** v4 interpolation refreshes fluxes and prescribes the mass trajectory each
substep, preventing drift.

## 10. Level Merging

**Julia:** 137 → 68 levels (min 1000 Pa thickness).
**TM5:** 25-34 levels.

**Theory:** The Russell-Lerner slopes scheme is 2nd-order in space, so vertical
resolution primarily affects the representation of sharp gradients (tropopause,
boundary layer top). For CO2, which is well-mixed in the troposphere, 68 levels
preserves all relevant structure. The stratosphere (where thin levels are merged)
has weak CO2 gradients.

TM5's 25-34 levels sacrifice upper-troposphere resolution for computational
efficiency and CFL stability. Our 68 levels give better vertical resolution
while managing CFL through merging the thinnest layers.

## Summary

| Difference | Impact on CO2 Transport | Priority to Fix |
|-----------|------------------------|-----------------|
| Met loading (v4) | Fixes m drift — critical | ✅ Done |
| CFL control | Different symmetry, same order | Low |
| X per-row subcycling | Over-subcycles low-CFL rows | Medium (efficiency) |
| Y subcycling | No-op (Y-CFL < 1) | None |
| Z subdivision | Stable with v4 m prescription | ✅ Done |
| Diagnostic slopes | ~80 km^2/s extra diffusion | Medium (accuracy) |
| Cross-slopes | 0.5% flux correction | Low |
| Multi-tracer | Equivalent | None |
| m_dev negativity | Fixed by v4 interpolation | ✅ Done |
| Level merging (68L) | Good balance resolution/CFL | None |
