# Session 2 Status: Advection Runtime — Blocked on NaN

**Commits**: `590aa8a` (Session 1), `efefaba` (Session 1 fixup), `8f23dd0` (Session 2 WIP)  
**Date**: 2026-04-03  
**Status**: BLOCKED — run produces NaN with current no-/n_sub convention

---

## What Session 1 Fixed (VALIDATED)

- **Grid convention**: Preprocessor now matches runtime (TM5: centers at ±89.75°, faces at ±90°)
- **cm precision**: Float64 accumulation + B-coefficient correction → residual 0.0005 kg (was 32,800)
- **Diagnostic validation**: `compare_transport_binaries.jl` confirms lat_convention=REJECTED, cm_reconstruction=REJECTED
- **New binaries**: Dec 1-2 at `/temp1/atmos_transport/era5_daily_v3/` with correct grid

## What Session 2 Changed

### 1. Removed /n_sub division from advection_phase! (LL)

**File**: `src/Models/physics_phases.jl` advection_phase! for LatitudeLongitudeGrid

**Current code** (no division):
```julia
copyto!(gpu.m_dev, gpu.m_ref)
for _ in 1:n_sub
    _apply_advection_latlon!(tracers, gpu.m_dev, gpu.am, gpu.bm, gpu.cm, ...)
end
```

**Rationale**: The CS path does not divide by n_sub. The old `run_forward_preprocessed.jl` did not divide. The red/blue team analysis confirmed TM5 computes am = pu × ndyn/(2×tref) = pu × 1800s, and each Strang cycle applies am twice. With n_sub=4 and am per 450s: total = 4 × 2 × 450 = 3600s ✓.

**Problem**: Run produces NaN after a few windows. Mass diagnostics show NaN for all tracers.

### 2. Z-subcycling kept at cfl_limit=0.95

**Evidence** (red/blue team, documented in commit `8f23dd0`):
- TM5 has NO Z-subcycling (advectz.F90 `dynamw_1d` has no CFL check)
- TM5 allows gamma = cm/m > 1, relying on prognostic slopes (rzm)
- Our diagnostic slopes (recomputed via minmod) produce slope ≈ 0 for uniform fields
- With gamma > 1 and slope = 0: flux = gamma × rm > rm → negative rm
- Tested: removing Z-subcycling → 66,406 negative cells at t=1
- Kept subcycling at 0.95 (3 inner iterations for Z-CFL=2.5)

### 3. Output uses m_ref (not m_dev)

m_dev diverges from m_ref over n_sub Strang cycles. Without /n_sub division, full am applied each substep creates large m changes. c = rm/m_dev produces NaN. Reverted to m_ref.

### 4. Pole am zeroed at runtime

GPU kernel caps reduced-grid clusters at 4 (`_MAX_GPU_CLUSTER` in run_helpers.jl:134). TM5 spec is 720 at poles. Effective CFL with cluster=4 is 89/4=22, too high. Pole am zeroing restored as safeguard.

---

## The Blocking Issue: NaN with no /n_sub

**Config**: dt=900, n_sub=4, no /n_sub division, no scaling

**Symptom**: All tracer masses become NaN during the first day. No CFL warnings logged.

**Hypothesis**: The accumulated Strang splitting error over 4 substeps with full am creates m_dev values that cause division-by-zero or overflow in the advection kernel. With /n_sub division (the old unified loop behavior), each substep sees 1/4 of the flux, keeping intermediate m_dev changes small.

**The fundamental question**: The old working code (`run_forward_preprocessed.jl`) also applied full am without /n_sub and worked fine. What was different?

Possible differences:
1. The old code might have used a different `strang_split_massflux!` implementation
2. The old code might have reset m_dev differently between substeps
3. The old spectral data had different am magnitudes (old grid with cos(-90°)≈0)
4. The old code might have had /n_sub in a different form

**Codex**: Please check `scripts/legacy/run_forward_preprocessed.jl` (if it exists) or git history for the old advection loop. Key question: did the old code apply full am per Strang cycle, or did it divide am somehow?

---

## Current Code State Summary

| Component | State | Evidence |
|-----------|-------|----------|
| Spectral preprocessor grid | ✓ TM5 convention | Diagnostic validated |
| cm B-coefficient + Float64 | ✓ Correct | Residual < 0.001 kg |
| v3 daily preprocessor | ✓ No am zeroing, bm zeroed at pole faces | TM5 reference |
| Binary header metadata | ✓ grid_convention + spectral_half_dt | Self-documenting |
| Spectral file validation | ✓ Rejects old pole-centered files | Guard against stale data |
| Runtime pole am zeroing | ✓ Safeguard (GPU cluster cap=4) | Codex review finding |
| /n_sub division | ✗ REMOVED but causes NaN | Needs investigation |
| Z-subcycling | ✓ Kept at CFL=0.95 | Red/blue team: diagnostic slopes require it |
| Output m convention | ✓ m_ref (m_dev causes NaN) | Strang splitting error too large |

## Files Changed Since Last Stable Commit

| File | Lines | Change |
|------|-------|--------|
| `src/Advection/mass_flux_advection.jl` | 1290-1296, 1335-1339 | Z-subcycling comments, cfl_limit=0.95 kept |
| `src/Models/physics_phases.jl` | 331-345, 568-582 | No /n_sub, pole am zeroed, pole avg removed |
| `src/Models/run_loop.jl` | 215-218 | m_ref for output |

## What Needs Resolution

1. **Why does no-/n_sub cause NaN?** — Is it the flux magnitude, m_dev drift, or kernel instability?
2. **What did the old code do?** — The legacy runner that worked at 2 s/win without /n_sub
3. **Should we restore /n_sub with the correct scaling?** — If the old code actually DID divide, the correct approach is /n_sub + appropriate scaling to match the total transport

## Preprocessed Data Available

| Data | Path | Grid | Status |
|------|------|------|--------|
| Spectral NetCDF (Dec 1-2) | `~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine_hourly/` | TM5 ✓ | Clean |
| v3 daily binary (Dec 1-2) | `/temp1/atmos_transport/era5_daily_v3/` | TM5 ✓ | Clean |
| GEOS-IT CS v4 (Dec 1-31) | `/temp1/catrine/met/geosit_c180/massflux_v4_nfs/massflux_v4/` | CS ✓ | Reference |
