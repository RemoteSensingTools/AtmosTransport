---
name: CS cm diagnosis convention differs from LL
description: CS preprocessor uses conv_h - dm for cm (mass-budget residual), LL uses B-coefficient hybrid formula. Both self-consistent but different vertical transport profiles.
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
> Current paths (line numbers drifted):
> - `diagnose_cs_cm!` now at `src/Preprocessing/cs_poisson_balance.jl:768`
>   (memory says 638)
> - `diagnose_cm_from_continuity!` now at
>   `src/MetDrivers/ERA5/VerticalClosure.jl:21` (memory says
>   `VerticalClosure.jl:32`)
>
> The conceptual CS-vs-LL convention distinction below is durable; only
> the citations have drifted.

The CS `diagnose_cs_cm!` (cs_poisson_balance.jl:638) uses:
  `cm[k+1] = cm[k] + conv_h - dm`

The LL `recompute_cm_from_divergence!` (mass_support.jl:228) with B_ifc uses:
  `cm[k+1] = cm[k] + conv_h - Δb * pit`

Where `pit = Σ conv_h` (total column convergence).

After a perfect Poisson balance (conv_h ≈ dm at each cell):
- CS: cm ≈ 0 everywhere (horizontal transport alone achieves target)
- LL: cm follows sigma-pressure (B-coefficient) vertical redistribution

Both are self-consistent with the runtime Z-sweep: `m_new = m + cm[k] - cm[k+1]`.
The total per-substep mass change is `conv_h + cm[k] - cm[k+1]`.

**Why:** The runtime `diagnose_cm_from_continuity!` (VerticalClosure.jl:32) uses the B-coefficient formula. The LL preprocessor's binary cm is overwritten at runtime by DryFluxBuilder. For CS grids, no equivalent runtime re-diagnosis exists yet, so the binary cm is used directly. The CS binary's near-zero cm means almost no vertical transport from the preprocessor data.

**How to apply:** This is not a bug but a design difference. When the CS met driver is implemented, either (a) the binary cm will be re-diagnosed at runtime using B-coefficients (matching LL), or (b) the near-zero cm from the mass-budget formula is intentional. Need to clarify intent with the user.
