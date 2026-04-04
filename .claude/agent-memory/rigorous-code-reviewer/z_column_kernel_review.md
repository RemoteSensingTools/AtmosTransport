---
name: Z-column kernel review findings
description: Critical bugs in _massflux_z_column_kernel! — stale-value contamination from single-pass in-place update, multi-tracer double mass update
type: reference
---

## _massflux_z_column_kernel! (mass_flux_advection.jl:1216-1358)

Reviewed 2026-04-03. Two critical bugs:

1. **Stale-value contamination:** Single-pass k=1:Nz loop updates rm[k] in-place before computing flux at k+1. Flux at interface k+1 reads rm[k] which is post-update, not pre-update. TM5 dynamw_1d avoids this via two-pass (compute all fluxes, then apply all updates). KA kernels can't allocate local arrays, so two-pass requires either double-buffer (existing _massflux_z_kernel!) or fixed-size NTuple stack storage.

2. **Multi-tracer double mass update:** advect_z_massflux_subcycled! (line 1404-1409) calls column kernel once per tracer. Each call updates m in-place (lines 1353-1357). With N tracers, m is updated N*r times instead of r times.

**Correct approach:** Use existing double-buffered _massflux_z_kernel! in a subcycling loop with cm_eff = cm/r, launched r times. This is already proven correct.

**Key invariant:** TM5 dynamw_1d computes ALL fluxes from the pre-update state before ANY rm update. Any in-place single-pass approach that reads neighbors of already-updated cells violates this.
