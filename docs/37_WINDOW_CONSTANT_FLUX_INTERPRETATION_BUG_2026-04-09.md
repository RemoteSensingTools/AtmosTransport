# Window-Constant Flux Interpretation Bug (2026-04-09)

## Summary

After fixing the Poisson-balance normalization bug, the remaining ERA5 lat-lon
`src` mass-loss problem came from the runtime, not the kernels.

The rebuilt reference binaries were still being advanced with intra-window
interpolation of `am`, `bm`, and `cm` toward the next hour. That interpretation
was wrong for the current reference contract.

For the current ERA5 lat-lon reference path, the stored balanced fluxes must be
treated as:

- `flux_kind = substep_mass_amount`
- `flux_sampling = window_constant`

That is, they are prepared transport amounts reused across the substeps within
the current met window, not endpoint states to interpolate toward the next one.

## Evidence

### Constant-vs-interpolated replay

Using the same rebuilt day-1 binary:

- with intra-window interpolation:
  - `window 1`: local mismatch `2.859e-03`
  - `window 4`: local mismatch `5.216e-03`
  - `window 10`: local mismatch `2.892e-03`
  - `window 20`: local mismatch `2.445e-03`
- with constant within-window flux forcing:
  - `window 1`: local mismatch `4.571e-07`
  - `window 4`: local mismatch `4.987e-07`
  - `window 10`: local mismatch `7.671e-06`
  - `window 20`: local mismatch `9.238e-07`

That isolates the remaining bug very cleanly: the forcing interpretation was
wrong even after the binary itself had been rebuilt correctly.

### Real 2-day run verification

On the corrected binaries for `2021-12-01` and `2021-12-02`:

- day-boundary mismatch before day 2: `9.153e-07`
- day-1 worst hourly endpoint mismatch over all 23 handoffs:
  - local `7.671e-06`
  - global `9.492e-06`
- uniform-tracer 2-day runs:
  - `UpwindScheme`: tracer drift `0`
  - `SlopesScheme`: tracer drift `1.219e-16`

## Fix

The current reference path now does three things explicitly:

1. preprocessing writes
   - `source_flux_sampling = window_start_endpoint`
   - `flux_sampling = window_constant`
   - `poisson_balance_target_scale = 1 / (2 * steps_per_window)`
2. `TransportBinaryDriver` validates that contract
3. `DrivenSimulation` derives the flux-interpolation mode from the driver and
   keeps fluxes constant within the window for this path

## Implication

The remaining mass-loss problem was **not** due to the advection kernels.

For the current ERA5 lat-lon reference path, the forcing contract is now:

- balanced in preprocessing against the correct scaled target
- consumed at runtime as window-constant substep transport amounts
- kept free of closure logic inside the advection kernels

That is the contract that should be carried into the next reduced-Gaussian
preprocessing/runtime work.
