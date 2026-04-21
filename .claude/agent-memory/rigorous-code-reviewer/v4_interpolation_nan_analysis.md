---
name: V4 flux interpolation NaN root cause
description: Runtime cm recomputation from interpolated am/bm amplifies Z-CFL 3.3x via B-correction; 590K cells exceed CFL=1; clamp destroys continuity balance
type: project
---

> **STALE — DO NOT TRUST FILE:LINE CITATIONS**
>
> Written before plans 18 and 22 shipped. File and line references may
> point to code that has been moved to `src_legacy/`, refactored, or
> renamed. Physics principles and design rationale may still apply.
> Verify any current-code claim against actual `src/` before acting.
>
> For a current trace of the corresponding subsystem, prefer the
> relevant module README under `src/` over this memory file.
>
> Note: `recompute_cm_from_divergence!` is still live at
> `src/Preprocessing/binary_pipeline.jl:684,926` and
> `src/Preprocessing/mass_support.jl:223,228`, but the `physics_phases.jl`
> runtime referenced below lives in `src_legacy/Models/`. The NaN
> scenario described may no longer apply after preprocessing refactors.

V4 flux delta interpolation (physics_phases.jl:1603-1626) produces 100% NaN.

**Root cause**: `_compute_cm_from_divergence_gpu!` (line 1441-1482) recomputes cm from
interpolated am/bm using the B-correction formula. The B-correction amplifies column-integrated
divergence differences between interpolated and original fluxes, producing cm values 7x larger
and Z-CFL 3.3x worse (276 vs 82) than the preprocessor's balanced cm. 590K cells (12.6%)
have Z-CFL > 1, and `_clamp_cm_cfl!` reduces them by up to 290x, destroying the continuity
balance (div_h + div_z ≠ 0).

**Why**: The operator `continuity(interpolate(am,bm))` ≠ `interpolate(continuity(am,bm))`
in Float32 arithmetic because the B-correction accumulates column sums that amplify
interpolation artifacts.

**How to apply**: Fix 1 (best): interpolate cm directly like am/bm (add dcm to v4 binary).
Fix 2: don't prescribe m_dev — let advection evolve it naturally (TM5 approach).
The m_dev prescription at line 1614 has no TM5 analog and is unnecessary.
