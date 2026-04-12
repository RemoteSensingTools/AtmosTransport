# Mass-Flux Advection: Design Evolution and Lessons Learned

This document records the evolution from concentration-based advection to TM5's
mass-flux formulation. **Read this before modifying the advection core** to
avoid repeating past mistakes.

## Timeline

### Phase 1: Concentration-based advection (initial)

The first implementation advected concentration `c` directly using Russell-Lerner
slopes with velocity-based Courant numbers (`α = u·Δt/Δx`). This matched TM5's
**stencil algebra** perfectly: the flux formula, slope limiter, and boundary
treatment were verified to produce identically zero differences on 1D test cases.

**What worked:** Stencil-level parity with TM5 (verified in
`scripts/compare_advection_stencil.jl`).

**What failed:** When embedded in 3D operator splitting (X-Y-Z-Z-Y-X), the model
produced extreme tracer values and poor mass conservation (~8% drift over 48h).
The catastrophic failure typically appeared suddenly (around time step 40 in one
test case), with good behavior before that.

### Phase 2: Post-hoc mass correction ("Mass-Flux Pressure Fixer")

Diagnosis suggested the issue was `Δp` accumulating drift across Strang splits.
A "pressure fixer" was introduced that reset `Δp` from surface pressure at the
start of each Strang split and scaled concentrations to compensate.

**What worked:** Eliminated the catastrophic instability.

**What failed:** Mass conservation improved but was still imperfect (0.91% drift).
More importantly, this was a **symptom treatment**, not a root cause fix. The
approach diverged from TM5's architecture rather than converging toward it.

### Phase 3: TM5-faithful mass-flux advection (current)

Deep comparison with TM5's Fortran source (`advectx.F90`, `advecty.F90`,
`advectz.F90`, `advect_tools.F90`, `advectm_cfl.F90`) revealed **four
architectural differences** — not bugs in the stencil, but fundamental
formulation mismatches:

1. **Prognostic variable**: TM5 advects tracer mass (`rm`) and air mass (`m`),
   not concentration (`c`).

2. **Continuous mass tracking**: TM5 passes `m` through the entire operator
   split without resetting. Each directional step updates `m` in place, and the
   next step uses that updated `m` for its Courant numbers and fluxes.

3. **Mass-based fluxes**: TM5 uses mass fluxes (`am`, `bm`, `cm`) with
   mass-based Courant numbers (`α = am / m_donor`), not velocity-based fluxes.

4. **Vertical flux from continuity**: The vertical mass flux `cm` is derived
   from horizontal convergence, not from meteorological vertical velocity. This
   guarantees column mass conservation by construction.

The new implementation (`src/Advection/mass_flux_advection.jl`) addresses all
four points and achieves machine-precision mass conservation.

## Lessons Learned

### 1. Stencil parity ≠ formulation parity

Getting the 1D flux formula right is necessary but not sufficient. The **outer
loop** — how mass is tracked through operator splitting, how Courant numbers
are computed, what variables are passed between directional steps — matters
equally. Future comparisons should verify the full operator-splitting loop,
not just individual stencils.

### 2. Post-hoc corrections are warning signs

If you need a correction step after each advection pass (scaling, clamping,
resetting `Δp`), the formulation is likely wrong at a deeper level. TM5
requires no such corrections because its formulation is inherently
mass-conserving.

### 3. Slopes must be computed from concentration, not tracer mass

When slopes are computed from `rm` (tracer mass), a spatially uniform
concentration field has non-zero slopes (because `rm = m·c` varies with `m`).
This causes spurious transport. Computing slopes from `c = rm/m` and then
scaling by `m` gives `s_rm = m · (c_{i+1} - c_{i-1})/2`, which correctly
produces zero slopes for uniform `c`.

This subtlety caused a test failure (max deviation ~1.0 instead of machine
epsilon) that was diagnosed by running the new test suite.

### 4. The continuity equation is not optional

The vertical mass flux must be derived from horizontal convergence to ensure
that each column's mass budget closes. Using meteorological `w` directly would
break this closure because gridpoint winds don't satisfy the discrete continuity
equation exactly.

### 5. Always write tests before the run script

The comprehensive test suite (`test/test_mass_flux_advection.jl`) caught two
bugs before the code ever touched real meteorological data:
- The slope-from-concentration issue (lesson 3)
- A CFL subcycling test that used too-low velocities, not a code bug but a
  test design issue that, if left unfixed, would have hidden future regressions.

## Files

| File | Role |
|:-----|:-----|
| `src/Advection/mass_flux_advection.jl` | Core mass-flux advection |
| `src/Advection/mass_correction.jl` | DEPRECATED — post-hoc fixer |
| `test/test_mass_flux_advection.jl` | Unit/integration tests |
| `docs/literate/advection_theory.jl` | Mathematical theory (Literate) |
| `docs/TM5_CODE_ALIGNMENT.md` | Checklist for TM5 comparison |
| `docs/VALIDATION.md` | Test results and validation status |

## TM5 Source Files Used for Comparison

| TM5 file | What we extracted |
|:---------|:------------------|
| `deps/tm5/base/src/advectx.F90` | x-advection: `mnew` (line 630), `alpha` (663), tracer+slope updates (706-716) |
| `deps/tm5/base/src/advecty.F90` | y-advection: `mnew` (478-492), tracer+slope updates (617-630), pole handling |
| `deps/tm5/base/src/advectz.F90` | z-advection (`dynamw_1d`): `mnew` (441), `gamma` (462,468), updates (479-491), sign convention |
| `deps/tm5/base/src/advect_tools.F90` | `dynam0`: mass flux computation from ECMWF `pu`, `pv`, `sd` |
| `deps/tm5/base/src/advectm_cfl.F90` | `dynamvm`, `advectx_get_nloop`: CFL subcycling |
