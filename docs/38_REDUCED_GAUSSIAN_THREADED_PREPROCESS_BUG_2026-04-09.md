# Reduced-Gaussian Threaded Preprocessing Bug (2026-04-09)

## Summary

The first `src` native reduced-Gaussian transport-binary path produced catastrophic mass-loss in runtime checks, but the root cause was not the advection kernel. The corruption was introduced earlier, in the threaded native spectral-to-face preprocessing path.

The bug was:
- nondeterministic
- present before level merging and before runtime
- specific to the threaded reduced-grid preprocessing path

## Symptom

The first reduced-grid endpoint check on the day-1 binary showed impossible air-mass mismatch after only a few windows, with local relative mismatch reaching values like `1e77` to `1e116`.

Direct binary inspection then showed impossible stored face fluxes:
- `|hflux| ~ 4e89` to `5e96` in the written day-1 file
- while cell masses were only `~1e8` to `1e12`

That immediately ruled out a small timestep or limiter issue. The forcing written to disk was already corrupted.

## What Was Ruled Out

These were checked and found not to be the primary cause:
- merged-level packing
- runtime vertical closure
- runtime `UpwindScheme` stepping
- reduced-grid face indexing alone
- meridional staggering alone

Important narrowing result:
- the first impossible face flux was a **zonal** face, so the dominant failure was already present in the zonal `u`-derived flux path

## Key Evidence

### 1. Native synthesis was nondeterministic under threading

A repeated hour-0 native reduced-grid synthesis test with `-t24` gave different maxima on each run for the same inputs.

Examples observed:
- `max |hflux| ~ 3.7e106`
- `max |hflux| ~ 2.7e104`
- `max |hflux| ~ 4.4e125`

The same repeated test with `-t1` was stable across runs:
- `max |hflux| = 1.7954091852374457e11`
- same index every run
- same sample face value every run

That isolated the bug to the threaded preprocessing implementation.

### 2. Raw zonal wind synthesis at the offending ring/level was sane

For the pathological zonal face region:
- ring latitude about `-56.63°`
- `nlon = 729`
- level `137`

Directly synthesizing the raw zonal wind on that ring gave reasonable values:
- `u ring min/max = (-4.846, 9.974)`
- offending face sample `u = 0.6943`

Rebuilding the face-flux formula by hand for the sample face gave a sane value:
- stored `hflux = 4.2886326610303503e8`
- recomputed `calc = 4.2886326610303503e8`

So the formula itself was fine. The corruption came from the threaded path, not from the scalar flux formula.

## Root Cause

The reduced-grid helper used thread-local caches, but those caches were not actually safe under the threaded execution pattern.

Two specific problems existed in
[reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl):

1. Per-thread FFT/real buffers were allocated lazily via mutable `Dict`s inside the threaded native-level loop.
2. The native-level loop used default `Threads.@threads` scheduling.

That combination allowed nondeterministic corruption in the threaded native spectral-to-face synthesis path.

## Fix Applied

### 1. Preallocate all needed thread-local buffers at workspace construction

At workspace allocation time, the code now precomputes the full set of ring and boundary lengths and allocates all FFT/real buffers upfront:
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L49)
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L60)
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L61)

### 2. Make buffer lookup non-mutating inside the threaded loop

The buffer helpers now only index preallocated storage:
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L109)
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L113)

### 3. Use static threading for the native-level loop

The native reduced-grid loop now uses `Threads.@threads :static`:
- [reduced_transport_helpers.jl](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/preprocessing/preprocess_spectral_v4_binary/reduced_transport_helpers.jl#L305)

## Verification After Fix

After the fix, the repeated `-t24` hour-0 reduced-grid synthesis became stable and matched the single-thread result.

Repeated threaded runs now give the same result every time:
- `max |hflux| = 1.7954091852374457e11`
- same max location each run
- same sample face value each run

That removed the nondeterministic corruption.

## Additional Reduced-Grid Issues Found Along The Way

These were also fixed while bringing the first native reduced-grid path up:
- missing mesh API imports in the reduced-grid preprocessor script
- local `cell_area` name shadowing the imported function
- native coefficient indexing needed `ab.dA[level]` / `ab.dB[level]`, not `ab.dA[kk]` / `ab.dB[kk]`
- reduced-grid preprocessing must iterate over `spec.hours`, not dictionary iteration order
- the reduced-grid binary header needed a larger header budget than the structured default
- the binary reader needed a larger initial header probe to parse the larger reduced-grid metadata block
- `load_transport_window` needed `AtmosGrid{<:ReducedGaussianMesh}` dispatch, not exact `AtmosGrid{ReducedGaussianMesh}`

## Remaining Work

This fix removes the catastrophic threaded corruption, but it does not prove the reduced-grid reference path is fully correct yet.

Still to verify:
- regenerate the reduced-grid day-1 binary with the fixed threaded path
- rerun the reduced-grid endpoint check on the clean binary
- rerun a short reduced-grid `UpwindScheme` smoke test
- then decide whether the current meridional staggering is acceptable or should be changed to a cell-center-then-average construction analogous to the structured path

## Design Lesson

For the new `src` preprocessing/runtime split, the important lesson is:

- correctness-critical preprocessors should not mutate supposedly thread-local cache maps inside hot threaded loops
- all required scratch for threaded spectral transforms should be fully allocated at workspace construction time
- if we rely on thread-local buffers, the scheduling semantics should be explicit
