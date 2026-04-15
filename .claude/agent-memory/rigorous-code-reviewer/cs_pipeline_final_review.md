---
name: CS pipeline final review findings
description: Two critical bugs found in spectral-to-CS preprocessing pipeline — double wind normalization and Poisson balance sign error
type: project
---

**Final review of CS transport binary pipeline (2026-04-13)**

Two CRITICAL bugs found:

1. **Double wind normalization** (`cs_transport_helpers.jl:310-324`): `normalize_regridded_intensive!` divides by `cell_areas`, but `ConservativeRegridding.regrid!` ALWAYS divides by `dst_areas` (which equals `cell_areas` to machine precision). This makes regridded winds ~5e12x too small (1e-12 m/s instead of 5 m/s). The fix in the prior review was based on the incorrect assumption that `normalize=false` means `regrid!` returns raw area-weighted sums -- it does not.

2. **Poisson balance target sign** (`cs_poisson_balance.jl:524`): The target is `(m_next - m_cur) * inv_scale` but should be `(m_cur - m_next) * inv_scale`. The CS divergence convention is `div = net_outflow` (opposite sign from the LL convergence convention). Confirmed numerically: LL gives `conv/projected_dm = +1`, CS gives `conv/projected_dm = -1`. This causes cm to be O(dm) instead of O(0) at interior levels.

**Why:** Bug 1 from misunderstanding the ConservativeRegridding API. Bug 2 from the CS face table using divergence (outflow) convention while the target formula was copied from the LL path which uses convergence (inflow).

**How to apply:** Both bugs must be fixed before any CS transport binary can be used for science.
