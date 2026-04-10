# Poisson Balance Target Bug (2026-04-09)

## Summary

This memo records the **first** forcing bug found in the ERA5 lat-lon `src_v2`
reference path: an incorrect normalization in the preprocessing Poisson-balance
target.

The preprocessor was balancing the stored horizontal flux fields `am` and `bm`
against the full forward window mass difference

`m_next - m_curr`

while the runtime consumes each stored window over `2 * steps_per_window`
horizontal half-sweeps per met window.

For the current reference setup:
- `dt = 900 s`
- `dt_met = 3600 s`
- `steps_per_window = 4`
- horizontal sweeps per window = `2 * steps_per_window = 8`

So the correct Poisson-balance target is:

`(m_next - m_curr) / (2 * steps_per_window)`

not the raw full-window difference.

## Where The Bug Was

### Preprocessing path

The buggy target was formed in:
- [binary_pipeline.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/binary_pipeline.jl)

Specifically, `fill_window_mass_tendency!` used:
- `storage.all_m[win_idx + 1] - storage.all_m[win_idx]`

without dividing by `2 * steps_per_window`.

That target was then used by:
- `balance_mass_fluxes!`

inside:
- `apply_poisson_balance!`

The same issue also existed in the fast repair path:
- [repair_v2_binary_cm.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/repair_v2_binary_cm.jl)

## Why This Is Wrong

The stored structured fluxes are written as half-sweep transport amounts.

That contract comes from the spectral synthesis path:
- `compute_mass_fluxes!` uses `half_dt`

The runtime then applies those fluxes in a Strang sequence with repeated horizontal sweeps:
- `X Y Z Z Y X`

So each met window consumes the horizontal fluxes over `2 * steps_per_window` horizontal half-sweeps.

Balancing the stored fluxes against the full window mass difference overdrives the corrective part of the forcing by that factor.

## Evidence

### 1. The runtime mismatch is local, not global

On the real `2021-12-01` ERA5 day-1 lat-lon binary:
- worst cellwise relative hourly endpoint mismatch after one window: `5.09%`
- global relative air-mass mismatch after one window: `5.67e-7`

So the issue is a strong local endpoint mismatch with cancellation, not a global conservation blow-up.

### 2. The problem is not pole-row specific

The strongest mismatch is around:
- `lat ≈ ±60°`
- upper levels `k ≈ 28-32`

It is not dominated by the pole rows or the longitude wrap seam.

### 3. The current forcing drives the window mass tendency at about 7x amplitude

Using the current binary and runtime, the one-window model air-mass change projects onto the stored `dm` with coefficient:
- `a ≈ 6.95`

That is close to the expected over-application when a full-window target is consumed over 8 horizontal half-sweeps.

### 4. Consistent two-window in-memory patch confirms the normalization bug

I patched both the window-1 base fluxes and the window-1 deltas consistently across the first two windows, rebalancing both windows against scaled targets.

Results for one-window upwind run:

- current target scale `1.0`:
  - local endpoint mismatch `5.09%`
  - projected mass-change amplitude `a = 6.95`
- scaled target `0.5`:
  - local endpoint mismatch `2.20%`
  - `a = 3.47`
- scaled target `0.25`:
  - local endpoint mismatch `0.91%`
  - `a = 1.74`
- scaled target `0.125 = 1 / (2 * 4)`:
  - local endpoint mismatch `0.50%`
  - `a = 0.87`

That is the strongest direct evidence that the present normalization is the primary bug.

### 5. Fresh rebuilt day-1 binary confirms the fix

After patching the preprocessing code and rebuilding the real `2021-12-01` ERA5 lat-lon binary, the same runtime endpoint diagnostic improved from:

- old binary:
  - `window 1 -> 2`: `2.98e-02` local mismatch
  - `window 2 -> 3`: `3.52e-02` local mismatch

To:

- rebuilt binary with fixed Poisson target:
  - `window 1 -> 2`: `2.86e-03` local mismatch
  - `window 2 -> 3`: `3.96e-03` local mismatch

So the binary-level verification agreed with the in-memory two-window experiment: the normalization fix improved hourly endpoint fidelity by about an order of magnitude.

## Fix Implemented

The preprocessing and repair paths now use:

`poisson_balance_target_scale = 1 / (2 * steps_per_window)`

and write that scale into the binary provenance.

Files changed:
- [binary_pipeline.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/binary_pipeline.jl)
- [preprocess_era5_latlon_transport_binary_v2.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_era5_latlon_transport_binary_v2.jl)
- [repair_v2_binary_cm.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/repair_v2_binary_cm.jl)

## Status After Follow-Up

This fix turned out to be necessary but not sufficient.

Follow-up verification showed:

- the Poisson normalization fix reduced hourly endpoint mismatch by about an
  order of magnitude
- but the remaining mass-loss bug was still in runtime forcing interpretation:
  the balanced fluxes had to be treated as `window_constant`, not interpolated
  toward the next hour inside the current window

That second bug is documented separately in:

- [37_WINDOW_CONSTANT_FLUX_INTERPRETATION_BUG_2026-04-09.md](/home/cfranken/code/gitHub/AtmosTransportModel/docs_v2/37_WINDOW_CONSTANT_FLUX_INTERPRETATION_BUG_2026-04-09.md)
