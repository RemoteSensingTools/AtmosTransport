---
name: CS global Poisson balance review
description: Review of global multi-panel CG Poisson balance for cubed-sphere grids — sign conventions, cm diagnosis, test coverage
type: project
---

Reviewed `scripts/preprocessing/cs_global_poisson_balance.jl` (2026-04-12).

**Architecture**: Global face table with 12*Nc^2 faces, CG solver identical to RG path. Cross-panel faces created only from outgoing edges (north, east), mirrors on incoming edges (south, west). Sign convention: mirror = +canonical for all CS cross-panel connections (verified: every outgoing edge pairs with an incoming edge in the GEOS-FP connectivity).

**Why:** Per-panel FFT was wrong (treated panels as doubly periodic). Global CG handles cross-panel coupling correctly.

**Key finding — cm diagnosis uses different formula than LL/RG paths**: The CS `diagnose_cs_cm!` uses `cm[k+1] = cm[k] + conv[k] - dm[k]` with explicit dm_dt from m_next-m_cur. The LL path uses TM5 B-coefficient splitting: `cm[k+1] = cm[k] + conv[k] - Δb[k]*pit`. Both are correct but produce different intermediate cm values. The CS approach is more accurate when mass change doesn't follow B-coefficient proportions.

**How to apply:** When the CS runtime eventually gets a `build_dry_fluxes!` with B-coefficient cm rediagnosis, the stored cm from this preprocessor will be overwritten. Until then, the preprocessor's cm IS the runtime cm. If discrepancies appear in CS transport, check whether cm rediagnosis has been added.

**Test gap**: Test imports are incomplete (needs `PanelConnectivity`, `EDGE_NORTH`, `EDGE_EAST` in scope). Test zeroes all boundary fluxes before balance, so doesn't exercise pre-existing inconsistent cross-panel flux reading.
