---
name: Pole bm clamp review
description: Analysis of clamp_bm_at_poles! for ERA5 LL pole drainage fix — key indexing, mass conservation, and cm inconsistency findings
type: project
---

## Pole bm clamp (clamp_bm_at_poles!) — April 2026

Reviewed the pole cell drainage fix. Key findings:

**Correct aspects:**
- Face indexing: clamps bm[i,2,k] (south) and bm[i,Ny,k] (north) — matches the faces that affect pole cell mass updates
- bm is (Nx, Ny+1, Nz): face j=1 and j=Ny+1 are boundary faces (zeroed by preprocessor), j=2 and j=Ny are pole-adjacent faces
- Mass conservation: in-place clamp means both pole cell and adjacent cell see the same reduced flux
- Linear cumulative analysis: 2*n_sub applications × max_cfl=1/(2*n_sub) = exactly 1.0 cumulative CFL → m_final >= 0

**Issues identified:**
1. MEDIUM: Clamps inflow as well as outflow (abs(bm) > threshold). Only outflow causes negativity. Clamping inflow is unnecessarily dissipative at poles.
2. MEDIUM: cm is NOT recomputed after bm clamp. Preprocessed cm was computed from original bm. After clamp, cm+bm mass balance at poles is inconsistent (divergence doesn't match).
3. MEDIUM: Z-advection between Y-passes in Strang splitting (X-Y-Z-Z-Y-X) can extract mass from pole cell. The cumulative CFL analysis assumes only Y-advection touches pole mass, but Z can drain mass too. If delta_z > 0, m_final could go slightly negative.
4. LOW: When has_deltas is eventually enabled, bm is recomputed each substep from bm0+t*dbm, bypassing the clamp. Need to re-apply clamp there.

**TM5 comparison:**
- TM5 calls Setup_MassFlow → TimeInterpolation → dynam0 each substep (advectm_cfl.F90:242)
- dynam0 recomputes bm=dtv*pv(interpolated) each substep with time-interpolated winds
- TM5 does NOT dynamically limit bm based on evolving pole mass; it relies on time-varying winds and ndyn halving for CFL control
- TM5 pole treatment in advecty.F90:520-538: upwind with no slope (zero rym at poles), not a CFL clamp
