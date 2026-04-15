---
name: Preprocessing pipeline review findings
description: Comprehensive review of src/Preprocessing/ — 5924 lines across 14 files, covering LL/RG/CS paths. Key bugs and architecture patterns.
type: project
---

Preprocessing pipeline review completed 2026-04-13. Key findings:

**Architecture**: Three grid-dispatched `process_day` methods (LL, RG, CS). LL is batch (stores all windows, then balances + writes). RG uses streaming 2-slot sliding buffer. CS uses streaming with per-window balance-and-write. The module docstring claims `GriddedFluxSource` and `GriddedWindSource` types exist — they do NOT. Only `SpectralSource` (ERA5 GRIB) is implemented.

**LL Poisson balance flux correction bug (MEDIUM)**: In `mass_support.jl:112-121`, the zonal correction subtracts `u_wrap` from every `du` but never applies it to `am[1]` or `am[Nx+1]`. The correction for `am[1]` should be `+u_wrap` (the periodic wrap gradient). Lines 119-120 add `FT(0)` as a no-op. Since pole constraints zero am at j=1 and j=Ny immediately after, and the FFT solution is exact for periodic grids, the error is only at non-pole latitudes for the wrap-around face. In practice, the correction IS included because `u_wrap = psi[1] - psi[Nx]` and the correction is `du[i] = (psi[i]-psi[i-1]) - u_wrap`. For i=2..Nx this subtracts the wrap; for i=1 (not handled in the loop) the wrap IS psi[1]-psi[Nx]=u_wrap, so the correction for am[1] should be u_wrap - u_wrap = 0, and am[Nx+1]=am[1] by periodicity. So the `+FT(0)` lines are actually CORRECT but misleadingly documented.

**RG mass fix inconsistency (LOW)**: RG `process_window!` uses the `qv_global_climatology` fallback only (no hourly qv path), while the LL path has both. The RG path does not support `mass_basis = :dry`.

**CS path: no next-day delta support**: The CS `process_day` does not process next-day hour-0 spectral data. The last window always uses zero-tendency fallback.

**Memory**: LL path stores ALL Nt windows before any writing (peak ~4x single window). RG and CS stream with O(1) windows in memory.

**How to apply**: When reviewing future preprocessing changes, check (1) which `process_day` method is affected, (2) whether the balance correction is applied consistently across paths, (3) whether mass fix mode matches between paths.
