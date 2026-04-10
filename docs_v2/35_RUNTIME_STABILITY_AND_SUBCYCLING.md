# Runtime Stability, Subcycling, and Precompile Notes

## Scope

This memo records the first real-data `src_v2` ERA5 lat-lon results and the
conclusions they force on the next implementation steps.

It is intentionally narrow:

- pure advection only
- `src_v2` only
- ERA5 lat-lon reference path first
- `UpwindScheme` as the first runtime truth baseline
- `SlopesScheme` only after the `UpwindScheme` path is stable enough to serve as
  the comparison target

## Current Real-Data Status

The first standalone `src_v2` ERA5 lat-lon transport binaries for
`2021-12-01` and `2021-12-02` were successfully produced and can be read and
executed through the new runtime.

What now works:

- a full real-data `2021-12-01` run completes with `UpwindScheme`
- the same full real-data `2021-12-01` run completes with `SlopesScheme`
- both runs stay finite for all 24 windows
- both runs preserve a uniform tracer exactly to the reported precision
- the kernel comparison suite passes again, including legacy
  `RussellLernerAdvection` vs `SlopesScheme`

So the original day-scale blow-up on the real `0.5°` lat-lon binary was fixed
well enough to produce a usable single-day baseline.

## What Still Fails: Multi-Day Endpoint Fidelity

A two-day carry-forward run is **not** yet a clean reference path.

Observed behavior:

- the binary day handoff itself is exact:
  - `day1.window24.air_mass + day1.window24.deltas.dm == day2.window1.air_mass`
- but the runtime state at the end of day 1 does **not** land close to the day-2
  window-1 air-mass state
- the measured local boundary mismatch before day 2 is about `5.0e-2`

This means:

- the remaining bug is **not** in the binary payload continuity
- the remaining bug is in the runtime integration of the window forcing
- single-day stability is therefore necessary but not sufficient to claim the
  lat-lon path is ready for multi-day reference runs

## Why This Matters

The current runner was originally reporting a final "air-mass drift" relative to
initial state. That number is not a pure conservation metric because the driver
resets air mass from the met window at each new window boundary.

The important diagnostic for multi-day correctness is instead:

- does the runtime state at the end of window `n` land near the stored air-mass
  state at window `n+1`?

That is now the key acceptance criterion for the lat-lon runtime.

## What The Current Flux Contract Means

The new transport-binary runtime contract currently says:

- `flux_kind = :substep_mass_amount`
- flux interpolation is done upstream in the driver/runtime layer
- advection kernels only consume prepared flux fields

That architectural split is still correct.

However, the current ERA5 lat-lon preprocessor still reuses the legacy spectral
pipeline convention where `am`, `bm`, and `cm` are scaled to `half_dt`.

Relevant source:

- spectral mass-flux construction uses `half_dt` in
  [spectral_synthesis.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/spectral_synthesis.jl#L190)
- the runtime then applies these prepared sweep fluxes through the Strang
  sequence in
  [StrangSplitting.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src_v2/Operators/Advection/StrangSplitting.jl#L631)

So the right reading of the current ERA5 lat-lon contract is:

- the binary stores prepared Strang-sweep transport amounts for one substep
- the runtime should not multiply fluxes by `dt`
- the remaining question is whether midpoint interpolation of hourly endpoint
  sweep fluxes is endpoint-faithful enough over a full hour

The current two-day mismatch says: **not yet**.

## Structured Subcycling Status

The structured `src_v2` path now includes a first CPU-side evolving-mass pilot
for x, y, and z directional subcycling.

That was enough to remove the original full-day instability on the real
`0.5°` lat-lon binary, but it is still only a first stage:

- it is global per direction, not row-adaptive
- it is CPU-only; GPU arrays currently fall back to a single pass
- it improves stability, but it does not guarantee endpoint fidelity across
  hourly windows

So the current state is:

- stable enough for a one-day upwind/slopes baseline
- not yet accurate enough for a trusted two-day carry-forward reference run

## Where TM5 Still Matters

TM5 does not rely only on one global subdivision factor.

In the newer MPI transport:

- x transport uses evolving-mass local `nloop(j,l)` in
  [advectx__slopes.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advectx__slopes.F90#L441)
- y transport uses evolving-mass local `nloop(l)` in
  [advecty__slopes.F90](/home/cfranken/code/gitHub/AtmosTransportModel/deps/tm5-mp-r1112/tm5-moguntia-r1112-revised/base/advecty__slopes.F90#L236)

Important implications:

- x-direction stiffness on lat-lon is naturally latitude dependent
- a row- or row/level-adaptive x policy is closer to TM5 and more efficient than
  one global x subdivision based on the polar maximum
- y-direction refinement can often stay simpler than x

But `src_v2` should still stay generic. That argues for:

1. a clean subcycling policy layer
2. scheme-independent CFL handling
3. topology-aware implementations behind a stable API

## Likely Next Numerical Step

The next bug to solve is no longer "make the day run finite". It is:

- make the runtime land near the next stored window state

The most likely places to inspect next are:

1. structured mass-only endpoint fidelity over one window
2. consistency between midpoint flux interpolation and stored `dm`
3. whether the current number of substeps per hour is sufficient for the
   Strang-split mass path

The first concrete experiment should be:

- keep the same binary payload structure
- measure end-of-window mismatch systematically
- test whether a smaller `dt` / more substeps per hour reduces the day-boundary
  mismatch materially

If it does, the remaining issue is primarily runtime integration error, not
binary semantics.

## PrecompileTools: Useful, But Not For This Bug

`PrecompileTools.jl` can help with:

- first-run latency of `src_v2` smoke tests
- JIT time for `TransportModel`, `DrivenSimulation`, and scheme dispatch
- repeated local development loops

It does **not** solve:

- multi-day endpoint mismatch
- large x-direction CFL by itself
- slow GRIB or NetCDF I/O
- expensive spectral transforms

So it is still worth adding later for ergonomics, but it is not the next
numerical priority.

## Preprocessing Speed Notes

The first real `2021-12-01` and `2021-12-02` lat-lon preprocesses showed that
the biggest costs are not binary packing:

- GRIB spectral reads are expensive
- repeated QV reads are also expensive

Representative timing for one day was about:

- total: `~313 s`
- spectral GRIB read: `~125 s`
- spectral transforms: `~76 s`
- QV loads: `~55 s`
- write: `~17 s`

So the first preprocessing optimizations should be:

1. keep the daily thermo/QV file open across the day
2. keep payload packing threaded and deterministic
3. parallelize across days before trying fine-grained variable-level process
   fan-out

The natural parallel units are:

- first: day-level parallelism across processes
- second: small worker-pool hourly-window parallelism

Not recommended as a first move:

- splitting `VO`, `D`, and `LNSP` across separate workers

Those fields belong to one coupled spectral transform path.

## Immediate Next Step

The next implementation step should be:

- keep the current stable single-day lat-lon path
- add explicit diagnostics for end-of-window air-mass mismatch
- test whether more substeps per hour materially improve endpoint fidelity
- only after that start treating the lat-lon path as ready for two-day
  reference runs

ReducedGaussian should stay on hold until this lat-lon endpoint-fidelity issue is
understood, because that same question will otherwise just reappear on the next
mesh.

## Useful Diagnostic Command

To measure runtime endpoint fidelity against the next stored window state:

```bash
julia --project=. scripts/diagnostics/check_window_endpoint_v2.jl \
  ~/data/AtmosTransport/met/era5/0.5x0.5/transport_binary_v2_tropo34_dec2021_f64/\
  era5_transport_v2_20211201_merged1000Pa_float64.bin upwind 6
```

The current baseline result is:

- local hourly air-mass mismatch: roughly `2.6e-2` to `3.5e-2`
- global hourly air-mass mismatch: roughly `1e-6`

That is the metric to watch while changing `dt`, subcycling policy, or the
window-forcing interpolation scheme.
