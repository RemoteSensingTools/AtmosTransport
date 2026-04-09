# Binary And Driver Contract

## Why timing semantics matter

Meteorological sources may provide different things:

- instantaneous values at time boundaries
- interval means
- interval-integrated transport amounts

Those are not interchangeable.

If the runtime treats interval-integrated fields as endpoint states, the forcing
can be wrong even when the kernel is correct.

## Current `src_v2` stored runtime contract

The current standalone runtime expects stored transport binaries with explicit
semantics that match this contract:

- `air_mass_sampling = "window_start_endpoint"`
- `flux_sampling = "window_start_endpoint"` when deltas are present
- `flux_kind = "substep_mass_amount"`
- `delta_semantics = "forward_window_endpoint_difference"`

Humidity semantics:

- `qv_start/qv_end` imply `humidity_sampling = "window_endpoints"`
- legacy `qv` implies `humidity_sampling = "single_field"`

## Provenance vs stored semantics

The binary should distinguish:

- what the raw source product originally was
- what the stored transport payload now means

That is why the header should carry provenance like:

- `source_flux_sampling = "instantaneous_endpoint"`
- `source_flux_sampling = "interval_mean"`
- `source_flux_sampling = "interval_integrated"`

But the runtime should only consume normalized stored semantics that it
explicitly supports.

## Driver responsibility

Drivers should:

- validate header semantics early
- refuse unsupported files
- prepare substep forcing cleanly

Drivers should not:

- silently reinterpret ambiguous semantics
- push timing/closure guesses down into kernels

## Dry-air output note

For moist-basis transport that may later write dry VMR:

- carry `qv_start`
- carry `qv_end`
- use end-of-window moist mass plus `qv_end` for end-of-window dry-air output

Do not reuse stale `qv_start` for output conversion.
