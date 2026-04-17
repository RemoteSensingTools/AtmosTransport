# Operator Composition — Strategic Architecture Reference

**Status:** Strategic reference document, not an executable plan.
**Audience:** Plan agent writing implementation plans for operators
  beyond pure advection (vertical diffusion, convection, emissions,
  deposition, slow chemistry).
**Companion docs:**
- `ARCHITECTURAL_SKETCH.md` — advection-subsystem-specific shape
- Individual operator plans (future) — will reference this doc for
  cross-operator constraints

## 1. Purpose

AtmosTransport currently implements advection only. Future work will
add vertical diffusion, convection, emissions/deposition, and simple
chemistry (slow decay, with possible extension to stiff chemistry).

This document defines HOW these operators compose into a complete
transport timestep. It specifies operator order, sub-stepping
hierarchy, data layout requirements, and the invariants each
operator must respect.

The document is grounded in two reference models:
- **TM5** (Krol et al., Uppsala/Utrecht) — offline CTM using symmetric
  Strang operator splitting across the full operator palindrome
- **GEOS-Chem / GCHP** (Harvard/NASA) — hierarchical timestep with
  transport inside a 2×-longer chemistry timestep

The chosen AtmosTransport architecture borrows TM5's palindrome
structure for its first-principles simplicity, with GEOS-Chem's
hierarchical timestep as a future extension point for stiff chemistry.

## 2. The physics that drives the design

Before discussing operator order as a numerical choice, the physics
it must respect:

**Surface emissions are highly localized in the vertical.** A power
plant, a forest, a city — these emit into the first ~100m of the
atmosphere (the surface layer) which corresponds to the first
model layer on typical vertical grids. If the model emits a tracer
and then immediately advects it horizontally, the tracer is "stuck"
in the bottom layer until vertical mixing operators run. In nature,
vertical turbulent mixing is fast (boundary layer turnover ~20-60
min), so real-world freshly-emitted mass spreads vertically on
similar timescales to horizontal advection.

**This creates a numerical hazard: operator ordering matters.**
- Emit → advect horizontally → (next step) diffuse vertically:
  tracer stays near surface longer than physical
- Emit → diffuse vertically → advect horizontally:
  tracer mixes before being advected, closer to physical

**The palindrome structure fixes this by symmetry.** In the sequence
`X Y Z  V  S  V  Z Y X`, emissions (`S`) sit at the palindrome
center. Vertical diffusion (`V`) runs BEFORE emissions (mixing the
state that will receive emissions) AND AFTER emissions (mixing the
freshly-emitted mass before the next round of advection). This is
second-order Strang accurate AND physically sensible: freshly-emitted
mass sees vertical mixing before seeing horizontal transport.

**Multi-layer emission distribution is orthogonal to this.** Some
real-world emissions go into layer 1 (diffuse surface sources,
wildfires at low altitude, cities). Others go into higher layers
(tall stack sources, aviation, volcanic plumes). The emissions
operator takes a 3D source field `S[i,j,k,tracer]` where most
tracers have non-zero source only in k=1 but the interface supports
arbitrary vertical distribution. The palindrome structure works
identically for both cases.

**Deposition has two distinct physical mechanisms:**
- **Dry deposition**: gas/particle uptake at surface, parameterized
  via deposition velocity `v_d`, flux = `-v_d · C_surface`. This is
  a BOUNDARY FLUX at z=0 — naturally expressed in the advection
  operator's bottom face.
- **Wet deposition**: in-cloud and below-cloud scavenging by
  precipitation, acts throughout the column wherever precipitation
  is present. This is a COLUMN SOURCE process, not a boundary flux.

These are different code paths for physically different things.

**Slow chemistry (current stage)**: pointwise cell-local decay,
`dq/dt = -k·q`, independent for each tracer, exact integration.
No stability constraint, no sub-stepping needed, trivially parallel.

**Stiff chemistry (future)**: species-coupled reactions with fast
timescales, needs implicit or semi-implicit solvers (Rosenbrock,
backward Euler). This is expensive and wants a COARSER timestep
than transport — hence GEOS-Chem's 2:1 chemistry:transport ratio.

## 3. The operator sequence

### 3.1 Top-level structure

```
step(dt_chem):
    transport_block(dt_transport)     # called N_chem_substeps times
    ...
    chemistry_block(dt_chem)
```

Where `N_chem_substeps = dt_chem / dt_transport`.

**Current stage default:** `dt_chem == dt_transport`, so
`N_chem_substeps = 1`. One transport block per step, then one
chemistry block.

**Future (stiff chemistry) default:** `dt_chem == 2 * dt_transport`
(GEOS-Chem standard), so `N_chem_substeps = 2`. This gives
second-order Strang splitting across the chemistry timestep:
```
transport(dt_trans) → transport(dt_trans) → chemistry(2*dt_trans)
```

The code SHOULD implement the general N_chem_substeps case and
default N=1, so that enabling stiff chemistry later is a
configuration change, not a code restructure.

### 3.2 The transport block — TM5-style palindrome

```
transport_block(dt):
    for l = 1 : n_cfl_iterations(dt, meteorology):
        dt_sub = dt / l

        # Forward palindrome half
        advection_x(dt_sub / 2)
        advection_y(dt_sub / 2)
        advection_z(dt_sub / 2)
        vertical_diffusion(dt_sub / 2)
        convection(dt_sub / 2)

        # Palindrome center (non-symmetric operators)
        emissions(dt_sub)
        # (wet deposition, if fast and in transport block, goes here)

        # Reverse palindrome half
        convection(dt_sub / 2)
        vertical_diffusion(dt_sub / 2)
        advection_z(dt_sub / 2)
        advection_y(dt_sub / 2)
        advection_x(dt_sub / 2)
```

Notes:

- `n_cfl_iterations` is determined by reading advective mass fluxes
  and checking the Courant limit. Matches existing `strang_split!`
  behavior; generalized to the full operator palindrome.

- The HALF-timestep pattern (`dt_sub / 2`) for each operator on each
  side of the center is the Strang splitting signature. Symmetric
  in both directions around the palindrome center.

- **Emissions run at FULL timestep** (`dt_sub`), not half. The
  palindrome symmetry provides second-order accuracy for operators
  that run twice with half-steps; operators at the palindrome center
  run once with the full step. Sources/sinks are the natural center
  operator because they are not "transport" operators — they don't
  move mass between cells, they add/remove mass.

- **Advection currently uses its own internal palindrome** (`X Y Z
  Z Y X`) within a single `strang_split!` call. With the full
  palindrome sequence above, the call chain looks like:
  `advection_x(dt/2)` does a single X-sweep, NOT the full internal
  palindrome. Plan 14's restructure of `strang_split!` should
  expose per-direction sweeps as first-class operators, so the
  outer palindrome can interleave them with V and C correctly.

- **"Combined convection/diffusion" (TM5 legacy)**: TM5's
  documentation notes that vertical diffusion and convection are
  combined into a single implementation. AtmosTransport should
  keep them SEPARATE (GEOS-Chem style). Rationales:
    1. They are mathematically different: diffusion is
       `∂q/∂t = ∂/∂z(K ∂q/∂z)` (parabolic); convection is a
       mass-flux updraft/downdraft system (hyperbolic-ish with
       entrainment).
    2. Separate modules are easier to develop, test, benchmark,
       and replace independently.
    3. TM5's combined solver is a Fortran-era design artifact;
       modern Julia code benefits from composability.

### 3.3 The chemistry block

```
chemistry_block(dt):
    # Cell-local operators, trivially parallel
    for each tracer t:
        apply_chemistry(tracer[t], dt)
```

Where `apply_chemistry` for current stage is:
```
apply_chemistry(tracer, dt):
    tracer .*= exp(-k_decay * dt)   # exact integration
```

For slow decay, this is exact regardless of `dt`. No sub-stepping,
no stability concern. The kernel is embarrassingly parallel —
every cell independent.

**Future extension for stiff chemistry:**
```
apply_chemistry(tracer, dt):
    implicit_solve(reaction_system, tracer, dt)
    # Rosenbrock, backward Euler, or similar
```

This is where the GEOS-Chem 2:1 ratio pays off: stiff solver runs
at the coarser chemistry timestep.

**Design implication for data layout:** chemistry is column-local
AND cell-local. It doesn't care about grid topology. Whatever 4D
layout advection settles on (see §4), chemistry iterates over cells
and applies per-tracer kernels. The layout constraint from chemistry
is weak — whatever advection prefers works for chemistry.

## 4. Data layout requirements

All operators need to agree on how tracers are stored. This is
the single most important cross-operator decision.

**Advection needs:**
- 4D tracer array `(Nx, Ny, Nz, Nt)` or similar
- Direct CUDA.jl / KernelAbstractions.jl compatible
- Per-sweep indexing: `Q[i,j,k,t]` contiguous along whichever axis
  is being swept (column-major, so k is fastest → good for vertical
  sweeps; need to transpose or restructure for x-sweeps — see plan
  14)

**Vertical diffusion needs:**
- Column-wise access: operates on `Q[i,j,:,t]` vectors
- Would prefer `k` fastest for column sweeps → COLUMN-MAJOR WITH K
  FASTEST is natural
- Operates one column at a time in parallel (all columns
  simultaneously on GPU)

**Convection needs:**
- Column-wise access (same as diffusion)
- Additional: mass-flux fields at column faces (updraft, downdraft,
  entrainment rates) as 3D fields
- Sparse in practice (convection only active where clouds exist)
  but the easiest implementation is "run the operator everywhere,
  it's a no-op where no clouds" — revisit for performance if needed

**Emissions needs:**
- 3D source field `S[i,j,k,t]` (most entries zero, layer-1 entries
  non-zero for surface sources)
- For stack sources, non-zero at higher k
- Applied per-tracer, per-cell: `Q[i,j,k,t] += dt * S[i,j,k,t] / mass[i,j,k]`

**Dry deposition needs:**
- Surface-layer access (k=1 only)
- Deposition velocity `v_d[i,j,t]` (2D, per-tracer)
- Applied as flux BC at advection bottom face OR as
  surface-layer loss term

**Wet deposition needs (future):**
- Column access (similar to diffusion/convection)
- Precipitation rate field (3D)
- Soluble-gas uptake coefficients per tracer

**Slow chemistry needs:**
- Per-cell, per-tracer access: `Q[i,j,k,t] *= exp(-k_decay[t] * dt)`
- No stencil, trivially parallel

**Stiff chemistry (future) needs:**
- Column-local AT MOST (most atmospheric chemistry is column-local
  for photolysis reasons, even though the reactions themselves are
  cell-local)
- Needs `Q[i,j,:,:]` as a (Nz, Nt) slab for implicit solvers
  that couple species

**The unifying layout:**

```
tracers::Array{FT, 4}         # (Nx, Ny, Nz, Nt), column-major
                              # k fastest-varying after (i, j)
tracer_names::Vector{Symbol}  # length Nt, for dispatch on per-tracer kernels
```

This layout works for ALL operators:
- Advection sweeps: X-sweep reads `Q[:, j, k, t]` (strided, need transpose
  buffer OR specialized kernel — plan 14 addresses this)
- Y-sweep reads `Q[i, :, k, t]` (strided, similar)
- Z-sweep reads `Q[i, j, :, t]` (contiguous, easy)
- Column operators (V, C, chemistry): `Q[i, j, :, t]` contiguous
- Emissions: `Q[i, j, k, t]` pointwise
- Per-tracer chemistry: iterate t outer, pointwise inner

**Plan 14's CellState restructure** should produce exactly this
layout. The composition requirement is:
1. Single 4D array, not NamedTuple of 3D arrays
2. `(i, j, k, t)` index order (not `(t, i, j, k)` — keeps spatial
   indices together for stencil operations)
3. `tracer_names` as separate vector, used for kernel dispatch and
   introspection

## 5. Sub-stepping composition

Different operators have different stability constraints:

- **Advection:** CFL limit, `u·dt/dx < 1`. Typical timestep at
  0.25° is 5-10 minutes.
- **Vertical diffusion (explicit):** `K·dt/dz² < 0.5` — VERY tight
  near the surface where layers are thin and K is large. In
  practice, diffusion uses IMPLICIT integration (Crank-Nicolson
  or backward Euler) to avoid this.
- **Convection:** natural timescale of updraft is the cumulus
  overturning time (~minutes to hours). With mass-flux formulation,
  stability is similar to advection.
- **Slow chemistry:** unconditionally stable, no sub-stepping.
- **Stiff chemistry:** requires implicit solver, operates at
  coarser timestep (hence 2:1 ratio).

**The AtmosTransport design:**

1. **Global outer timestep** `dt_chem` (e.g., 20 minutes, matches
   GEOS-Chem's chemistry timestep).

2. **Transport block substeps** at `dt_transport = dt_chem /
   N_chem_substeps` (default N=1, future N=2).

3. **CFL iterations WITHIN each transport block** drive the
   palindrome repetition. If `dt_transport` is too large for the
   local wind field, the full palindrome repeats `l` times at
   `dt_transport / l`.

4. **Each operator integrates internally** for its own stability.
   Vertical diffusion uses implicit Crank-Nicolson regardless of
   the outer timestep — it's stable for any dt. Chemistry uses
   exact exponential integration for slow decay, implicit solver
   for stiff chemistry.

5. **No independent operator sub-cycling** (at least initially).
   If convection needs finer timesteps in some regions, that's
   handled inside the convection operator (local mass-flux
   iteration), not by running convection multiple times in the
   outer loop.

This matches TM5's "global iteration" philosophy. It's simpler
than GEOS-Chem's more decoupled approach, and it's appropriate
for an offline model where the dynamical timestep is set by the
meteorological input frequency (typically hourly), not by
operator-specific stability.

## 6. Interface between operators

Each operator implements a standard signature:

```julia
apply!(state::CellState, meteo::Meteorology, grid::Grid,
       operator::AbstractOperator, dt; workspace)
```

Where:
- `state`: contains the 4D tracer array + cell-mass field `m`
- `meteo`: the meteorological inputs relevant to this operator
  (e.g., velocity for advection, K_z for diffusion, mass fluxes
  for convection, emission rates for emissions)
- `grid`: geometric information
- `operator`: dispatch type (e.g., `CrankNicolsonDiffusion`,
  `TiedtkeConvection`, `ExponentialDecay`)
- `dt`: the timestep for this operator call (not necessarily the
  outer timestep — could be a half-step within the palindrome)
- `workspace`: pre-allocated scratch arrays

**Workspace sharing vs. per-operator workspaces:**

Two options:
- **Single shared workspace**: one `Workspace` struct with fields
  for every operator's scratch needs. Simpler, but the workspace
  grows large and every operator must know about the others.
- **Per-operator workspace**: each operator has its own workspace
  type (`AdvectionWorkspace`, `DiffusionWorkspace`, etc.),
  grouped in a `TransportWorkspace` container.

**Recommendation: per-operator workspaces, grouped.** Matches the
existing `AdvectionWorkspace` pattern. Each operator's workspace
is designed for that operator. The container is a simple
NamedTuple or struct that holds all of them.

```julia
struct TransportWorkspace
    advection::AdvectionWorkspace
    diffusion::DiffusionWorkspace
    convection::ConvectionWorkspace
    # emissions and chemistry typically don't need scratch
end
```

This is extensible — adding a new operator adds a field. Existing
operators don't change.

**State mutation contract:**

Every operator's `apply!` mutates `state.tracers` (and possibly
`state.mass` for mass-flux operators). The operator reads the
state, computes fluxes/sources, writes back. Post-condition:
`state` reflects the tracers after `dt` of this operator's
action.

**Conservation invariant:**

Transport operators (advection, convection, vertical diffusion)
MUST conserve total tracer mass. Post-apply:
`sum(state.tracers[:,:,:,t] .* state.mass)` unchanged for each
tracer `t`.

Sources/sinks operators (emissions, deposition, chemistry) DO
NOT conserve mass — they add or remove mass. The invariant for
these operators:
- Emissions: `sum_new - sum_old = dt * sum(emission_field)`
  (exactly, to floating-point precision)
- Decay: `sum_new = sum_old * exp(-k * dt)` (exactly)

Violations of these invariants indicate bugs. Unit tests for
each operator MUST check conservation/decay invariants
bit-identically or to ULP tolerance.

## 7. Order within the transport block

Why the specific order `X Y Z V C | S | C V Z Y X`?

**Advection before V and C:** TM5 convention. The idea is that
advection resolves the resolved-scale transport first, then the
sub-grid processes (diffusion, convection) handle what's left.
GEOS-Chem does the same (`transport → mixing → convection` order
historically).

**V before C:** TM5 puts V before C in the forward half
(implicitly, since the combined operator runs once). GEOS-Chem
runs PBL mixing before cloud convection. Rationale: vertical
diffusion redistributes boundary-layer mass, then convection
takes over above the PBL.

**S at center:** Emissions and their counterpart fluxes
(deposition, if treated as source term) are the "non-transport"
part of the palindrome. They don't need symmetric application
because they're not moving mass between cells.

**Palindrome symmetry:** Everything except the center operator
runs twice, once on each side of the center, each with half the
timestep. This gives Strang splitting's second-order accuracy.

**Alternative orderings considered:**

- `X Y V Z C | S | C Z V Y X`: putting V between Y and Z. No
  strong physical reason; breaks the "horizontal advection
  together" grouping. Rejected for simplicity.

- `V X Y Z C | S | C Z Y X V`: V outermost. Would mean
  vertical mixing happens first and last, sandwiching everything.
  Plausible alternative. Rejected because horizontal advection
  is the "main" transport operator and TM5 convention puts it
  outermost.

- `X Y Z | S | Z Y X` with V and C outside the palindrome:
  would require separate sub-stepping for V and C. More complex
  for no obvious benefit.

**Conclusion:** use the TM5 order `X Y Z V C | S | C V Z Y X`.
It's conservative (follows a well-validated precedent), it's
physically motivated (mix then emit then mix), and it's symmetric
(Strang second-order).

## 8. Boundary conditions

Surface emissions and dry deposition are physically boundary
conditions (fluxes through z=0). Two implementation options:

**Option A (TM5): as part of emissions operator in palindrome
center.** Emissions add to cell `k=1`, deposition removes from
cell `k=1`. Clean separation, but treats a flux BC as a source
term (small inconsistency — the emission "flux" is applied as a
cell-volume source, which introduces a grid-dependent factor).

**Option B (clean): as flux BC inside advection's z-sweep.** The
advection operator's Z-sweep uses face fluxes `F[i,j,k]` at
`k=1/2, 3/2, ..., Nz+1/2`. The surface flux `F[i,j,1/2]` is
currently set to zero (wall boundary). Instead, set it to
`emission_rate - deposition_rate`. Now advection handles surface
exchange NATIVELY, conservatively, without a separate operator.

**Recommendation: Option B for surface exchange.** It's closer to
the physics (emissions ARE a flux through the surface), it's
perfectly conservative, and it unifies the treatment.

**But keep a separate "sources" operator for:**
- Stack emissions (not at surface, need 3D source field)
- Chemistry-like processes (decay, reactions — not fluxes)
- Wet deposition (column process, not surface)

So the final structure:

```
advection_z(dt):
    # Z-sweep with custom bottom-face flux
    bottom_flux[i,j] = emission_rate[i,j] - dry_dep_velocity[i,j] * C[i,j,1]
    sweep_z!(state, fluxes, bottom_flux, dt)

emissions(dt):
    # Stack emissions (non-surface), treated as source terms
    # Applied where emission_field[i,j,k,t] is non-zero and k > 1
    state.tracers .+= dt .* emission_field ./ state.mass

chemistry(dt):
    # Pointwise decay, exact integration
    state.tracers .*= exp.(-k_decay .* dt)
```

Where most tracers have zero `emission_field` for `k > 1`,
effectively making the emissions operator a no-op for surface-only
sources. The expensive term is inside advection's bottom-face
flux.

## 9. Implementation order recommendation

Suggested sequence for adding operators after advection ships:

1. **Plan 14 — finish advection cleanup.** 4D tracer layout,
   unified pipeline. Expected benefit: code cleanliness; perf
   depends on config (see plan 14 v2).

2. **Slow chemistry.** Simplest non-trivial operator. Already
   needed for current usage (decay). Adding it:
   - Validates the operator interface (`apply!` contract)
   - Establishes the operator dispatch pattern
   - Adds essentially zero complexity (cell-local, exact integration)
   - Good shakedown for the operator composition structure

3. **Surface emissions as flux BC.** Modify advection's Z-sweep
   to accept a bottom-face flux field. Minimal change to existing
   advection; small new code path.

4. **Dry deposition as flux BC.** Extension of #3 — same bottom
   face, negative flux proportional to surface concentration.

5. **Vertical diffusion.** First operator that's genuinely new
   machinery. Column-implicit Crank-Nicolson. Tests: mass
   conservation, known analytic solution of diffusion equation.

6. **Convection.** Most complex operator. Mass-flux formulation
   (Tiedtke 1989 or similar). Tests: mass conservation,
   entrainment/detrainment balance.

7. **Stack emissions as 3D source.** Extension to emissions
   operator for non-surface sources.

8. **(Future) Stiff chemistry, photolysis, aerosol microphysics.**
   Enable 2:1 chemistry:transport timestep ratio. Major project,
   separate design phase needed.

9. **(Future) Wet deposition.** Column loss term, either in
   transport block or chemistry block depending on timescale.

Each step is a separate plan document. This doc is the strategic
reference each of those plans refers to.

## 10. Invariants and tests

Every operator plan should include these tests:

### Conservation tests

For transport operators (advection, diffusion, convection):
```
# Total mass preserved
∑ state.tracers[i,j,k,t] * state.mass[i,j,k]  # before
  == ∑ state.tracers[i,j,k,t] * state.mass[i,j,k]  # after
# (exactly, for conservative schemes)
```

For source operators (emissions):
```
# Mass change equals integrated source
∑ state.tracers[i,j,k,t] * state.mass[i,j,k]_after
  - ∑ state.tracers[i,j,k,t] * state.mass[i,j,k]_before
  == dt * ∑ emission_field[i,j,k,t]
```

For decay operators (slow chemistry):
```
# Exact exponential decay
∑ state.tracers[i,j,k,t]_after
  == ∑ state.tracers[i,j,k,t]_before * exp(-k * dt)
# (to ULP tolerance)
```

### Consistency tests

For the full palindrome transport block:
```
# Symmetric Strang: running with dt twice equals running with 2*dt once
# (to some tolerance, not bit-exact because of substep CFL iteration)
step(state, 2*dt) ≈ step(step(state, dt), dt)
```

### Comparison tests

Cross-check against known analytic solutions:
- Advection of a Gaussian on uniform flow → unchanged shape
- Diffusion of a delta function → Gaussian with variance
  `σ² = 2 K t`
- Convection mass flux balance → sum of updraft - downdraft =
  net vertical mass transport

### Splitting error tests

Compare palindrome against a reference non-split solution
(e.g., Runge-Kutta 4 on the full ADR equation). Strang should
give O(dt²) error; first-order splitting gives O(dt) error.

## 11. Open questions

These are deliberately left open — future plans will resolve them:

1. **Sub-stepping within operators.** Does vertical diffusion need
   internal substeps for unstable K profiles? Probably not with
   implicit solver, but TBD. Resolve when writing plan for
   diffusion.

2. **Convection sparsity.** Real-world convection is non-zero in
   <10% of grid columns. Is it worth short-circuiting the operator
   where mass fluxes are zero? Measure first, optimize second.

3. **GPU vs CPU for chemistry.** Slow decay is trivially GPU-able.
   Stiff chemistry typically runs on CPU (complex control flow
   in stiff solvers is not GPU-friendly). How does the architecture
   handle this CPU-GPU split? Resolve when stiff chemistry becomes
   real.

4. **Adjoint considerations.** Future 4D-Var or adjoint-based
   inversion may impose additional constraints on operator
   structure. Not an immediate concern; revisit if/when inversions
   become a goal.

## 12. Cross-references

- Advection subsystem shape: `ARCHITECTURAL_SKETCH.md`
- Current performance reality: `CLAUDE.md` § "Performance tips"
- Plan 14 (advection unification): `14_SINGLE_PIPELINE_PLAN.md` (v2)
- Plan 11, 12, 13 retrospectives: `docs/plans/11.../NOTES.md` etc.

## 13. Reference models — what TM5 and GEOS-Chem actually do

### TM5 (post-2005)

Sequence: `(xyz vsc csv zyx)^l`
- `x, y, z` = advection in each direction
- `v` = vertical diffusion
- `s` = sources/sinks (emissions, deposition)
- `c` = chemistry (combined with convection in TM5's impl)
- `l` = CFL-driven iteration count

External timestep: 3 hours (CarbonTracker setting)
Inner palindrome effective frequency: ~10 min
Splitting: symmetric Strang across entire palindrome

### GEOS-Chem / GCHP

Hierarchical timestep:
- Dynamic timestep (10 min typical): transport, convection,
  PBL mixing, wet deposition
- Chemistry timestep (20 min, 2× dynamic): chemistry,
  emissions, dry deposition

Strang: `transport(dt) + transport(dt) + chemistry(2*dt)`
Recommended settings (Philip et al. 2016): dt_chem = 20 min,
dt_transport = 10 min at typical 2°×2.5° resolution.

### Key differences AtmosTransport borrows from each

From TM5:
- Palindrome structure
- Single global iteration count (CFL-driven)
- Emissions at palindrome center
- Flux-based surface exchange

From GEOS-Chem:
- Separate vertical diffusion and convection (NOT combined)
- Hierarchical timestep support (dt_chem ≥ dt_transport)
- Transport-before-chemistry ordering

What AtmosTransport does NOT borrow:
- GEOS-Chem's historical "chemistry includes emissions" coupling
  (unclean separation)
- TM5's combined V+C operator (legacy artifact)
- Fortran-era Makefile-based module compilation ordering :)

---

## 14. Summary

Target architecture:

```
step(dt_chem):
    for i in 1 : N_chem_substeps:   # default N=1
        transport_block(dt_chem / N_chem_substeps)
    chemistry_block(dt_chem)

transport_block(dt):
    for l = 1 : n_cfl_iterations(dt):
        dt_sub = dt / l
        # Forward half
        advection_x(dt_sub/2)
        advection_y(dt_sub/2)
        advection_z(dt_sub/2)        # with surface flux BC
        vertical_diffusion(dt_sub/2)
        convection(dt_sub/2)
        # Center
        emissions(dt_sub)            # stack sources, non-surface
        # Reverse half
        convection(dt_sub/2)
        vertical_diffusion(dt_sub/2)
        advection_z(dt_sub/2)
        advection_y(dt_sub/2)
        advection_x(dt_sub/2)

chemistry_block(dt):
    for each tracer t:
        apply decay (exact)   # current
        # future: stiff solver at dt_chem
```

Data layout: single 4D `tracers` array with companion
`tracer_names` vector.

Each operator is a separate module, separate type, separate
workspace. Composed via the sequence above.

This is the reference architecture. Individual plans implement
pieces. Every plan references this doc for the "how do I fit in?"
question.

---

**End of document.**
