# Vertical Diffusion Operator — Implementation Plan for Plan Agent (v1)

**Status:** Ready for execution after plan 15 has shipped.
**Target branch:** new branch from wherever plan 15's shipped work
  lives. Verify in §4.1.
**Estimated effort:** 1-2 weeks, single engineer / agent.
**Primary goal:** Implement a vertical diffusion operator conforming
  to the operator interface from `OPERATOR_COMPOSITION.md` §6. Use
  `TimeVaryingField{FT, 3}` abstraction for Kz sourcing, enabling
  both pre-computed (GCHP-style) and derived (TM5-style) Kz with
  the same diffusion kernel. Preserve adjoint-friendliness
  (transposable tridiagonal structure) even though adjoint shipping
  is deferred. Position V in the palindrome per composition doc.

Self-contained document. ~25 minutes to read.

**Dependencies verified:**
- Plan 11 (ping-pong) shipped
- Plan 12 (scheme consolidation) shipped
- Plan 13 (sync + CFL + rename) shipped
- Plan 14 (advection unification) shipped — 4D `tracers_raw`
  layout, storage-agnostic accessor API
- Plan 15 (slow chemistry) shipped — operator interface validated,
  `AbstractChemistryOperator`, `chemistry_block!`, `step!`
  orchestration established

**Companion documents:**
- `OPERATOR_COMPOSITION.md` §3.1 (step-level), §3.2 (transport
  block), §6 (operator interface) — authoritative
- `TIME_VARYING_FIELD_MODEL.md` — for Kz field abstraction
- Plan 15 NOTES — operator interface validation lessons
- Plan 14 NOTES — test contract, `parent(arr)` dispatch

**Legacy code available:**
- `src_legacy/Diffusion/Diffusion.jl` — abstract type + three
  concrete implementations (`BoundaryLayerDiffusion`,
  `PBLDiffusion`, `NonLocalPBLDiffusion`) plus `NoDiffusion`
- `src_legacy/Diffusion/boundary_layer_diffusion.jl` — static
  exponential Kz + Thomas solve
- `src_legacy/Diffusion/boundary_layer_diffusion_adjoint.jl` —
  transposed Thomas solve (for adjoint, deferred this plan)
- `src_legacy/Diffusion/pbl_diffusion.jl` — Beljaars & Viterbo
  1998 / TM5 revised LTG, derives Kz from PBLH + u* + HFLUX + T2M
- `src_legacy/Diffusion/nonlocal_pbl_diffusion.jl` — Holtslag &
  Boville 1993 / GEOS-Chem VDIFF with counter-gradient γ_c

Legacy is a **starting point, not a reference implementation**.
The agent can restructure when modern design is cleaner, subject
to preserving physics correctness and adjoint-friendliness.

---

# Revisions note

v1 (this version). Written after plan 15's shipping. Incorporates
lessons from plans 14-15 retrospectives:

1. **Measurement-first Commit 0** — establish diffusion-enabled
   vs advection-only performance baseline before any refactor.
2. **Test contract: accessor API observations only** — plan 14
   lesson. Diffusion mutates `state.tracers_raw`; tests must
   observe through `get_tracer(state, name)`.
3. **`parent(arr) isa Array` for CPU/GPU dispatch** — SubArrays
   of 4D `tracers_raw` aren't `isa Array`.
4. **Operator interface validated in plan 15** — `apply!(state,
   meteo, grid, op, dt; workspace)` works. Use it unchanged.
5. **Survey-first Commit 1 (from plan 15)** — the "mostly
   greenfield" assumption was wrong in plan 15; existing
   `src/Operators/Chemistry/` was substantial. Plan 16 explicitly
   surveys `src_legacy/Diffusion/` (known substantial) AND
   `src/Operators/Diffusion/` (may not exist yet). Do NOT assume
   greenfield until grep confirms.
6. **Pre-Commit-0 memory compaction (from plan 15 D3)** — update
   MEMORY.md, plan-completion notes, and stale-path warnings
   BEFORE Commit 0 so downstream agents see fresh state. Part
   of plan 16's Commit 0 scope.
7. **`apply!` arg-type dispatch requires meteorology for
   diffusion** — unlike chemistry (which accepts `meteo=nothing`),
   `ImplicitVerticalDiffusion` NEEDS meteorology when using
   `DerivedKzField` (to update surface fields). Test helpers
   must provide a real meteorology even for constant-Kz tests.
   See Decision 10 and known pitfall 11.
8. **Chemistry rates retrofit as part of scope** — plan 15
   deferred `TimeVaryingField` because the abstraction didn't
   exist; plan 15 D2 noted "when plan 16+ introduces
   `TimeVaryingField`, migrate `ExponentialDecay.decay_rates`
   field type." Plan 16 DOES introduce the abstraction (for Kz)
   and SHOULD retrofit chemistry rates to use it. See §3.2 and
   Commit 2a.

---

# Part 1 — Orientation

## 1.1 The problem in one paragraph

AtmosTransport needs vertical diffusion to capture boundary-layer
mixing. The legacy `src_legacy/Diffusion/` module has three
implementations (static, derived-from-surface-fields, non-local
counter-gradient) with a tridiagonal Thomas solve and adjoint
support. Plan 16 ports this into the modern `src/Operators/`
hierarchy, with three key modernizations:

1. **Kz sourcing via `TimeVaryingField{FT, 3}` abstraction.** The
   diffusion kernel no longer reads meteorology directly; it reads
   a Kz field that can be pre-computed (GCHP, MERRA-2 style) OR
   derived at kernel time from surface fields (TM5 / ERA5 style).
   Same kernel, different backends.
2. **Integration with operator interface.** Diffusion becomes a
   first-class operator implementing `apply!(state, meteo, grid,
   op, dt; workspace)`, composable with advection and chemistry.
3. **Palindrome position.** Diffusion V integrates into the
   transport block at the palindrome center, between forward and
   reverse Z-sweeps. Structure: `X Y Z V Z Y X` (current shape;
   `C | S | C` chemistry/emissions wrapping added later in
   plans 15 integration and plan 17).

## 1.2 Scope keywords

- **Operator type:** `AbstractDiffusionOperator`
- **Concrete types:** `ImplicitVerticalDiffusion` (primary),
  `NoDiffusion` (no-op for tests)
- **Interface:** `apply!(state, meteo, grid, op, dt; workspace)`
- **Kz field types:**
  - `ConstantKzField` — scalar Kz (tests, idealized runs)
  - `PreComputedKzField` — 3D array from binary (GCHP path)
  - `DerivedKzField` — wraps surface fields + computes via
    Beljaars-Viterbo (TM5 path)
- **Time integration:** Backward Euler (implicit, first-order,
  unconditionally stable, adjoint = Forward Euler)
- **Spatial discretization:** Tridiagonal Thomas solve per column
- **Palindrome position:** `X Y Z V Z Y X`

## 1.3 What's new vs. legacy

The legacy code reads meteorology directly inside the diffusion
kernel. Plan 16 inserts the `TimeVaryingField` abstraction between
meteorology and the kernel:

**Legacy pattern:**
```
PBLDiffusion(β_h, Kz_bg, ...) + diffuse!(tracers, met, grid, diff, dt)
  kernel reads met.pblh, met.ustar, met.hflux, met.t2m directly
  kernel computes Kz inline per (i,j,k)
```

**Plan 16 pattern:**
```
ImplicitVerticalDiffusion(kz_field::TimeVaryingField{FT, 3})
  + apply!(state, meteo, grid, diff, dt; workspace)
  update_field!(kz_field, t)   # on CPU, may read meteorology
  kernel reads field_value(kz_field, (i,j,k)) — kernel-safe
```

The indirection has two wins:

1. **Unifies runtime.** Same kernel for pre-computed and derived
   Kz. Kernel no longer branches on met source type.
2. **Preserves backend flexibility.** Different met sources provide
   Kz differently (GCHP provides directly; ERA5 requires derivation).
   Plan 16 supports both via different `TimeVaryingField` concrete
   types.

## 1.4 What plan 16 preserves

- **Physics of Beljaars-Viterbo** (TM5 revised LTG scheme) —
  preserved in `DerivedKzField` concrete type
- **Tridiagonal structure** — preserved for adjoint-friendliness
- **Backward Euler time integration** — unconditionally stable,
  simple adjoint (Forward Euler)
- **Coefficient caching** — Thomas factors can be cached and
  reused between timesteps if Kz is constant
- **Legacy physical constants** (Kz_min, Kz_max, Kz_bg, β_h, etc.)
  — same meanings, same default values

## 1.5 What plan 16 defers

- **Adjoint kernel** (transposed Thomas) — legacy has it; plan 16
  preserves tridiagonal STRUCTURE but ships forward-only
- **Non-local (counter-gradient) diffusion** — legacy has
  `NonLocalPBLDiffusion`; plan 16 ships only local diffusion
- **Diagnostic K_z computation for every met source** — plan 16
  implements one `DerivedKzField` concrete type (Beljaars-Viterbo
  from ERA5-style surface fields). Other met sources handled via
  `PreComputedKzField` or future concrete types.
- **Surface flux BCs** — plan 17
- **Ordering study (V before or after advection?)** — plan 17
  after emissions are in place

## 1.6 Test suite discipline

Same as plans 11-15:
- Baseline failures captured at Commit 0
- Per-commit test runs compare pass/fail to baseline
- Pre-existing 77 failures remain unchanged through plans 11-15

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before:
- Diffusion lives in `src_legacy/`, not integrated with modern
  `CellState` 4D storage
- Diffusion has its own `diffuse!` interface, not the unified
  `apply!`
- Kz sourcing baked into each implementation (TM5 vs GCHP code
  paths different)
- No palindrome structure — legacy diffusion is called ad hoc in
  driver code

After:
- Diffusion is a first-class operator in `src/Operators/Diffusion/`
- Implements `apply!(state, meteo, grid, op, dt; workspace)` —
  unified interface
- Kz sourcing via `TimeVaryingField` — one kernel, multiple
  backends
- Palindrome position formalized: `X Y Z V Z Y X`
- Composable with chemistry (plan 15) through the transport/
  chemistry block structure

## 2.2 What this enables

- **Plan 17 (surface fluxes):** can build on V at palindrome center.
  Emissions ENTER at palindrome center, V mixes them upward before
  reverse Z-sweep takes over horizontally.
- **Future non-local diffusion:** another `AbstractDiffusionOperator`
  subtype, same interface.
- **Future diagnostic Kz computations:** another `TimeVaryingField`
  concrete type, same diffusion kernel.

## 2.3 What this does NOT enable

- Column-specific physics (convection, etc.) — plan 18
- Time-varying surface fluxes with complex temporal structure —
  plan 17
- Photolysis or other radiation-dependent chemistry — deferred

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- Advection machinery — plan 14 settled it
- Chemistry machinery — plan 15 settled it
- `OPERATOR_COMPOSITION.md` §3-6 structure — diffusion conforms to
  existing interface
- `TimeVaryingField` abstract type or existing concrete types —
  plan 16 ADDS `PreComputedKzField` and `DerivedKzField` as new
  concrete types, doesn't modify the abstract
- LinRood, CubedSphereStrang — diffusion is a vertical column
  operation, doesn't interact with horizontal grid logic

## 3.2 Do NOT add

- Adjoint kernel — legacy has one (`boundary_layer_diffusion_adjoint.jl`),
  structure preserved so it can be added later. Don't port the
  adjoint file itself yet.
- Non-local counter-gradient diffusion — defer
- Implicit time integration other than Backward Euler — Crank-
  Nicolson is tempting for second-order accuracy, but introduces
  oscillation issues at discontinuities and is rarely used in
  atmospheric transport. Stick with BE.
- Column mixing beyond the tridiagonal operator — no non-local
  fluxes, no convective transilient matrix
- Vertical diffusion substepping — BE is unconditionally stable,
  no CFL constraint
- Surface emission BCs in the Z-sweep — plan 17's scope. Until
  plan 17 ships, emissions continue to be handled at the
  DrivenSimulation level (via the existing sim-level workaround
  from plan 15). Plan 16 does NOT attempt to resolve the
  TM5-ordering tension noted in plan 15 NOTES; that's plan 17.

## 3.2.1 DO add (not strictly diffusion, but related)

- **`AbstractTimeVaryingField{FT, 0}` scalar fields** — needed for
  `ConstantKzField` in current form AND for retrofitting chemistry
  rates. Plan 15 D2 explicitly deferred this retrofit to "when
  plan 16+ introduces `TimeVaryingField`." Plan 16 introduces the
  abstraction; it should also ship the migration of
  `ExponentialDecay.decay_rates` from `NTuple{N, FT}` to
  `NTuple{N, AbstractTimeVaryingField{FT, 0}}`. This is a small
  scope addition (~50 lines + tests), fits in Commit 2a.

## 3.3 Potential confusion — clarified

**Diffusion is implicit; no CFL constraint.** Backward Euler is
unconditionally stable. The time step can be as large as the
outer transport timestep. No sub-stepping needed.

**Diffusion is column-local, not cell-local.** Unlike chemistry
(plan 15, cell-local) and advection (stencil-local), diffusion
requires solving a tridiagonal system over the entire column.
Kernel dispatch is `(i, j)` with `k` as inner loop.

**Adjoint is the TRANSPOSE, not the INVERSE.** For
`M · x_new = x_old`, forward is `x_new = M⁻¹ x_old`, adjoint is
`x_old_adj = M⁻ᵀ x_new_adj`. For tridiagonal M, the transpose is
obtained by swapping sub- and super-diagonals. Thomas algorithm
can be adapted. Plan 16 preserves this structure for future
adjoint support.

**Kz has units of m²/s in physics. Legacy uses Pa²/s when working
in pressure coordinates.** Be careful about unit conventions when
porting — the tridiagonal coefficients involve `Kz / Δz²` where
Δz can be m (geometric) or Pa (pressure). Decision below.

**`TimeVaryingField{FT, 3}` for Kz.** Per
`TIME_VARYING_FIELD_MODEL.md`, Kz is a 3D field. The time-varying
part matters because Kz changes with meteorology. For constant-Kz
test cases, use `ConstantKzField` (a thin concrete type).

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch
git branch -a | head -20
git log --oneline --all | grep -i "plan 15\|chemistry\|slow[-_]chem" | head -10
git checkout <parent-branch>
git pull
git log --oneline | head -20
# Expected: plans 11-15 commits visible

git checkout -b vertical-diffusion

# 2. Clean working tree
git status

# 3. Verify dependency state
grep -c "tracers_raw" src/State/CellState.jl
# Expected: non-zero (plan 14)
grep -rn "AbstractChemistryOperator\|ExponentialDecay" src/ --include="*.jl" | head -5
# Expected: multiple (plan 15 shipped)
grep -rn "TimeVaryingField\|ConstantField" src/ --include="*.jl" | head -5
# Expected: multiple (plan 15 used ConstantField)
grep -rn "chemistry_block\|transport_block\|step!" src/Models/ --include="*.jl" | head -5
# Expected: present (plan 15)

# 4. Inventory legacy diffusion code (starting point)
ls -la src_legacy/Diffusion/ | tee artifacts/legacy_diffusion_inventory.txt
wc -l src_legacy/Diffusion/*.jl | tee -a artifacts/legacy_diffusion_inventory.txt

# 5. Inventory existing diffusion-like code in modern src/
grep -rn -i "diffus\|kz\|k_z\|thomas\|tridiag" src/ --include="*.jl" | \
    tee artifacts/modern_diffusion_refs.txt
# Review. The modern src/ should have minimal diffusion refs.
# If there's substantial existing infrastructure, flag for review.

# 6. Capture baseline failure set
for testfile in test/test_basis_explicit_core.jl \
                test/test_advection_kernels.jl \
                test/test_structured_mesh_metadata.jl \
                test/test_reduced_gaussian_mesh.jl \
                test/test_driven_simulation.jl \
                test/test_cubed_sphere_advection.jl \
                test/test_poisson_balance.jl \
                test/test_chemistry_operator.jl; do
    # last test file is from plan 15; include if present
    [ -f "$testfile" ] || continue
    echo "=== $testfile ==="
    julia --project=. $testfile 2>&1 | tail -20
done | tee artifacts/baseline_test_summary.log

# 7. Record baseline
git rev-parse HEAD > artifacts/baseline_commit.txt
mkdir -p artifacts/perf/plan16
```

If preconditions fail, STOP.

## 4.2 Change scope — the expected file list

**Files to ADD (new):**

Core operator:
- `src/Operators/Diffusion/` — new directory
- `src/Operators/Diffusion/Diffusion.jl` — module file,
  `include`s the files below
- `src/Operators/Diffusion/AbstractDiffusionOperator.jl` — type
  hierarchy
- `src/Operators/Diffusion/ImplicitVerticalDiffusion.jl` —
  concrete operator type + `apply!`
- `src/Operators/Diffusion/thomas_solve.jl` — tridiagonal Thomas
  algorithm (forward only; structure preserved for future adjoint)
- `src/Operators/Diffusion/diffusion_kernels.jl` — KA kernel for
  column-wise diffusion

Kz field concrete types (extend `TimeVaryingField`):
- `src/State/Fields/ConstantKzField.jl` — scalar Kz wrapped as
  `TimeVaryingField{FT, 3}`
- `src/State/Fields/PreComputedKzField.jl` — thin wrapper over 3D
  array
- `src/State/Fields/DerivedKzField.jl` — Beljaars-Viterbo
  computation from surface fields

(Directory `src/State/Fields/` may already exist from plan 15's
`ConstantField` work. If not, create it.)

Tests:
- `test/test_diffusion_kernels.jl` — unit tests for Thomas solve
  and diffusion kernel
- `test/test_diffusion_operator.jl` — integration tests through
  `apply!`
- `test/test_kz_fields.jl` — unit tests for the three Kz
  `TimeVaryingField` types

Docs:
- `docs/plans/16_VERTICAL_DIFFUSION_PLAN/NOTES.md`

**Files to MODIFY:**

- `src/AtmosTransport.jl` — include new `Operators/Diffusion/`
  module
- `src/Operators/Advection/StrangSplitting.jl` — palindrome
  integration (see §4.3 Decision 7)
- `src/Models/TransportModel.jl` — carry diffusion operator
  alongside advection scheme
- `src/Models/DrivenSimulation.jl` — pass diffusion operator into
  the step! call

**Files NOT to modify** (but read for context):
- `src_legacy/Diffusion/*.jl` — reference only; no edits
- `TIME_VARYING_FIELD_MODEL.md` — interface already specified

## 4.3 Design decisions (pre-answered)

Every decision final. If ambiguous, STOP and ask.

**Decision 1: Type hierarchy.**

```julia
abstract type AbstractDiffusionOperator end

struct NoDiffusion <: AbstractDiffusionOperator end

struct ImplicitVerticalDiffusion{FT, KzF} <: AbstractDiffusionOperator
    kz_field::KzF   # ::AbstractTimeVaryingField{FT, 3}
    Kz_min::FT      # safety floor, default 0.01 m²/s
    Kz_max::FT      # safety ceiling, default 500 m²/s
end
```

Single concrete type `ImplicitVerticalDiffusion` parameterized by
the Kz source. Variation in behavior comes from the `TimeVaryingField`
backend, not from multiple operator subtypes. Clean separation.

**Decision 2: Backward Euler time integration.**

The implicit system is `(I - Δt · D) Q_new = Q_old` where D is
the discrete diffusion operator. Backward Euler:
- Unconditionally stable
- First-order accurate in time (acceptable; diffusion is usually
  not the dominant source of time-truncation error)
- Adjoint is Forward Euler (simple)
- Legacy code uses BE — consistent

Reject Crank-Nicolson despite second-order accuracy: CN can
produce oscillations at discontinuities, and atmospheric tracer
profiles have sharp gradients at the PBL top. BE's dissipation
is a feature for tracer transport.

**Decision 3: Three Kz concrete types, all as `TimeVaryingField{FT, 3}`.**

```julia
# Option 1: scalar Kz, fully constant in space and time
struct ConstantKzField{FT} <: AbstractTimeVaryingField{FT, 3}
    value::FT
    dims::NTuple{3, Int}   # for interface compatibility
end

# Option 2: pre-computed 3D field (possibly time-varying)
struct PreComputedKzField{FT, D, A, T} <: AbstractTimeVaryingField{FT, 3}
    inner::StepwiseField{FT, 3, A, T}   # delegate to existing type
end
# Or: direct wrapper if the underlying data is a single 3D snapshot

# Option 3: derived from surface fields (Beljaars-Viterbo)
struct DerivedKzField{FT, SF, GP, P} <: AbstractTimeVaryingField{FT, 3}
    surface_fields::SF   # holds PBLH, USTAR, HFLUX, T2M as
                         # TimeVaryingField{FT, 2} each
    grid_params::GP      # Δz profile for the grid
    physics_params::P    # (β_h, Kz_bg, Kz_min, Kz_max, ...)
    cache::Array{FT, 3}  # internal 3D cache of computed Kz
end
```

The `DerivedKzField.cache` holds the computed Kz values.
`update_field!(f, t)` recomputes the cache when surface fields
change (at each meteorology update cycle); `field_value(f, idx)`
just reads from the cache.

This separation matters for GPU: the Beljaars-Viterbo computation
is complex and might be easier to run on CPU than to write a
kernel for. The cache is then copied to GPU. `field_value` always
reads from the cache, kernel-safe.

**Decision 4: Kernel-level Kz access.**

Inside the diffusion kernel, Kz is accessed via `field_value`:

```julia
# In apply!:
update_field!(op.kz_field, t)   # CPU, may be expensive

# Inside kernel:
Kz_k = field_value(op.kz_field, (i, j, k))
```

For `ConstantKzField`, `field_value` returns the stored scalar
(zero-cost dispatch). For `PreComputedKzField` and
`DerivedKzField`, it reads from a 3D array (cache). All three are
kernel-safe.

**Decision 5: Tridiagonal solve on columns in registers.**

One kernel thread per (i, j) column. Thread loops from k=1 to
k=Nz forward, then k=Nz to k=1 backward, solving the tridiagonal
system in registers (Thomas algorithm).

For GPU register pressure: Nz=72 (C180 grids) means ~72 doubles
per thread for the g[k] intermediate. Plus a[k], b[k], c[k]
coefficients (cached outside kernel if possible; recomputed per
column if Kz varies).

If Nz > ~100 and register pressure becomes an issue, spill to
global memory. For initial implementation, assume register-based
solve is adequate.

**Decision 6: Coefficient handling.**

Options for computing tridiagonal coefficients `(a, b, c)`:

- **Option A:** Precompute once per timestep on CPU, upload to GPU.
  Efficient if Kz doesn't change during the timestep.
- **Option B:** Compute inside the kernel per column. Simpler, no
  precompute phase.

For `ConstantKzField` and `PreComputedKzField`, Option A is
efficient (coefficients identical across columns if Δz is grid-
uniform; column-specific if Δz varies with surface pressure).

For `DerivedKzField`, Option B is natural (Kz varies per column).

Decision: **Option B always** (compute coefficients inside kernel).
Simpler, uniform across Kz types. If profiling shows precomputation
helps, add it as an optimization later.

**Decision 7: Palindrome position — `X Y Z V Z Y X`.**

Current (post plan 14): `X Y Z Z Y X` (advection Strang).

After plan 16: insert V between the two Z sweeps:
```
X Y Z V Z Y X
```

This is the palindrome center for the transport block. Future
plans add:
- Plan 17 emissions as S: `X Y Z V S V Z Y X` (emissions applied
  between two V half-steps for physical coupling)
- Plan 18 convection as C: `X Y Z V C S C V Z Y X` or similar

For plan 16, implement: `X Y Z V Z Y X`. The single V in the
middle means `V` runs with full dt (not half-steps).

**Decision 8: V runs with full dt at palindrome center.**

Since V is implicit and unconditionally stable, there's no need
to split V into two half-steps. A single V application between
the forward and reverse Z-sweeps is correct:

```
forward:  X(dt/2) Y(dt/2) Z(dt/2)
center:   V(dt)
reverse:  Z(dt/2) Y(dt/2) X(dt/2)
```

Wait — this doesn't quite work for Strang second-order accuracy.
The correct Strang splitting for `L = L1 + L2 + L3 + L4 + L5`
(where L4 is V, L5 is XYZ advection half-steps) is symmetric
around the midpoint. Let me re-examine.

**Actually: for the palindrome `X Y Z V Z Y X`, each operator
runs with `dt/2`, and V runs twice (once forward, once reverse),
giving:**

```
X(dt/2) Y(dt/2) Z(dt/2) V(dt/2)   V(dt/2) Z(dt/2) Y(dt/2) X(dt/2)
```

The two V(dt/2) calls are equivalent to one V(dt) for linear
operators (which V is — diffusion is linear). So the effective
pattern for V is:

```
X(dt/2) Y(dt/2) Z(dt/2)   V(dt)   Z(dt/2) Y(dt/2) X(dt/2)
```

Implementation: ONE V(dt) call at palindrome center. For linear
operators, this is mathematically identical to V(dt/2) V(dt/2)
with no wasted work.

For nonlinear operators (chemistry, convection), the two-half-step
version would be needed. Diffusion is linear (with state-
independent Kz), so single-call is fine.

**This is consistent with OPERATOR_COMPOSITION.md §3.2.**

**Decision 9: `apply!` signature unchanged from plan 15.**

```julia
function apply!(state::CellState,
                meteo::AbstractMeteorology,
                grid::AbstractGrid,
                op::ImplicitVerticalDiffusion,
                dt::Real;
                workspace=nothing)
    update_field!(op.kz_field, current_time(meteo))
    # ... kernel launch
    return state
end
```

Plan 15 validated this interface. Diffusion uses it without
modification. Workspace may carry pre-allocated tridiagonal
coefficient arrays (optimization, not required for correctness).

**Decision 10: Meteorology is passed to `apply!` for time lookup.**

`update_field!(kz_field, t)` needs the current time `t`. The
meteorology object carries current time. For `DerivedKzField`,
`update_field!` also reads surface fields from the meteorology
(if the surface fields are themselves `TimeVaryingField`s, they're
updated first).

If meteorology doesn't yet have a `current_time` accessor, add
one. Small scope.

**Decision 11: Tests use accessor API.**

Same as plans 14, 15: observe tracer state via
`get_tracer(state, :name)` post-operator, never via input array.

**Decision 12: Preserve tridiagonal structure for future adjoint.**

The Thomas solve computes `(a, b, c)` tridiagonal coefficients
and solves. For the adjoint, the transposed tridiagonal is
obtained by:
- `a_transpose[k] = c[k-1]` (sub-diagonal becomes shifted super)
- `b_transpose[k] = b[k]` (main diagonal unchanged)
- `c_transpose[k] = a[k+1]` (super-diagonal becomes shifted sub)

Plan 16 does NOT implement the adjoint, but the forward
implementation MUST:
- Keep `(a, b, c)` accessible (not fused into a single
  pre-factorized form that can't be transposed)
- Document the transposition rule in a comment
- Structure the solve as `solve_tridiagonal(a, b, c, d) → x`
  with a clean interface that can be called from an adjoint kernel

The Thomas solve CAN use optimizations (precomputed `w[k]` and
`inv_denom[k]`) for the forward path, but the `(a, b, c)` values
must also be retrievable or recomputable.

**Decision 13: Kz_min / Kz_max safety clamps stay in the operator,
not in the field.**

The `TimeVaryingField` returns whatever Kz the source provides.
The diffusion operator applies `clamp(Kz, Kz_min, Kz_max)` inside
the kernel. Rationale:
- Keeps `TimeVaryingField` pure (no domain-specific clamping)
- Lets different operators using the same field apply different
  constraints (e.g., a future operator might use Kz without
  clamping for diagnostics)
- Matches legacy `PBLDiffusion` which carries Kz_min/Kz_max in
  the struct

## 4.4 Atomic commit sequence

### Commit 0: NOTES.md + baseline

```bash
mkdir -p docs/plans/16_VERTICAL_DIFFUSION_PLAN
cat > docs/plans/16_VERTICAL_DIFFUSION_PLAN/NOTES.md << 'EOF'
# Plan 16 Execution Notes — Vertical Diffusion Operator

Plan: `docs/plans/16_VERTICAL_DIFFUSION_PLAN.md` (v1)

## Baseline
Commit: (fill in)
Pre-existing test failures: (fill in)

## Legacy code survey (from §4.1)
(fill in: line counts, which files ported, which deferred)

## Commit-by-commit notes
(fill in)

## Decisions beyond the plan
(fill in)

## Surprises
(fill in)

## Interface validation findings
(fill in: did apply! signature work unchanged?)

## Adjoint-structure preservation
(fill in: is tridiagonal structure transposable without rewrite?)
EOF

# Baseline: see §4.1

git add docs/ artifacts/
git commit -m "Commit 0: NOTES.md + baseline for plan 16"
```

### Commit 1: Kz field concrete types (no diffusion operator yet)

Add the three `TimeVaryingField{FT, 3}` concrete types:

- `src/State/Fields/ConstantKzField.jl`
- `src/State/Fields/PreComputedKzField.jl`
- `src/State/Fields/DerivedKzField.jl`

Each implements the full `TimeVaryingField` interface
(`value_at`, `integral_between`, `update_field!`, `field_value`,
`time_bounds`, `requires_subfluxing`).

`DerivedKzField` includes the Beljaars-Viterbo computation from
legacy `pbl_diffusion.jl`. This is the substantive port — ~300
lines of physics translated from legacy kernel into a `TimeVaryingField`.

Tests in `test/test_kz_fields.jl`:
1. `ConstantKzField`: `field_value` returns scalar, invariants
   hold
2. `PreComputedKzField`: `field_value` reads from 3D array
3. `DerivedKzField`: given test surface fields (PBLH, u*, HFLUX,
   T2M), Beljaars-Viterbo formula produces the expected Kz
   profile at a known column (reference data from legacy test
   case if available)
4. Integral invariants (per `TIME_VARYING_FIELD_MODEL.md` §8)
5. CPU vs GPU consistency (ULP tolerance)

This is a substantial commit (~800 lines of new code + tests).
Split into 1a (ConstantKzField), 1b (PreComputedKzField), 1c
(DerivedKzField) if it's too large.

```bash
git commit -m "Commit 1: Kz field concrete types (ConstantKzField, PreComputedKzField, DerivedKzField)"
```

### Commit 2a: Retrofit chemistry rates to `TimeVaryingField{FT, 0}`

Plan 15 D2 deferred this pending plan 16's introduction of
`TimeVaryingField`. Now that scalar 0D fields exist (as the
trivial case of the abstraction from
`TIME_VARYING_FIELD_MODEL.md`), migrate `ExponentialDecay`:

**Before (plan 15):**
```julia
struct ExponentialDecay{FT, N} <: AbstractChemistryOperator
    decay_rates::NTuple{N, FT}
    tracer_names::NTuple{N, Symbol}
end
```

**After (plan 16):**
```julia
struct ExponentialDecay{FT, N, K} <: AbstractChemistryOperator
    decay_rates::K   # ::NTuple{N, AbstractTimeVaryingField{FT, 0}}
    tracer_names::NTuple{N, Symbol}
end
```

Update `apply!(ExponentialDecay)` to:
1. Call `update_field!(rate, t)` on each rate field
2. Resolve to scalars via `field_value(rate, ())` for kernel use
3. Pass the NTuple of scalars to the existing chemistry kernel

Existing tests in `test/test_chemistry.jl` need small updates:
convert test configurations from `(2.1e-6, 0.0, 0.0)` to
`(ConstantField(2.1e-6), ConstantField(0.0), ConstantField(0.0))`.

Scope: ~50 lines change + test updates. Small commit.

Why in plan 16 and not deferred further: leaving chemistry rates
as raw scalars while introducing `TimeVaryingField` for Kz creates
architectural inconsistency. Either ALL rate-like inputs are
`TimeVaryingField` or NONE are. Plan 15 D2 committed to the former.

```bash
git commit -m "Commit 2a: Retrofit ExponentialDecay decay rates to TimeVaryingField{FT, 0}"
```

### Commit 2: Thomas solve + diffusion kernel (no palindrome yet)

Implement the low-level solve machinery:

- `src/Operators/Diffusion/thomas_solve.jl` — forward Thomas
  algorithm, structured for future adjoint (Decision 12)
- `src/Operators/Diffusion/diffusion_kernels.jl` — KA kernel for
  column-wise diffusion

```julia
@kernel function _vertical_diffusion_kernel!(
    tracers_raw,       # 4D: (Nx, Ny, Nz, Nt)
    kz_values,         # 3D: (Nx, Ny, Nz), pre-fetched from TimeVaryingField
    delp,              # 3D: layer thicknesses (or Δz)
    dt,
    Kz_min, Kz_max     # safety clamps
)
    i, j, t = @index(Global, NTuple)
    # Inside: loop over k, build (a, b, c) tridiagonal,
    #         solve Thomas per column per tracer
    # ...
end
```

Note: one thread per (i, j, tracer). Inside thread: full
tridiagonal solve for this column + tracer. Register pressure
scales with Nz. For Nz=72 (C180), tractable.

Alternative: one thread per (i, j), loop over tracers inside.
Benchmark both patterns in Commit 6.

Tests in `test/test_diffusion_kernels.jl`:
1. Gaussian bump diffuses to a broader Gaussian (analytic)
2. Step function smooths to tanh (analytic)
3. Conservation: `sum(Q_new) == sum(Q_old)` within ULP (no
   sources/sinks in pure diffusion)
4. Adjoint-structure test: given `(a, b, c)`, can we construct
   the transposed system and get expected behavior? Document
   but don't implement the adjoint solve yet.
5. Convergence to steady state for Neumann BCs
6. CPU vs GPU agreement (ULP tolerance)

```bash
git commit -m "Commit 2: Thomas solve + vertical diffusion kernel"
```

### Commit 3: `ImplicitVerticalDiffusion` operator + `apply!`

Implement the operator type:

```julia
struct ImplicitVerticalDiffusion{FT, KzF} <: AbstractDiffusionOperator
    kz_field::KzF
    Kz_min::FT
    Kz_max::FT
end

function apply!(state::CellState{FT},
                meteo, grid,
                op::ImplicitVerticalDiffusion,
                dt::Real;
                workspace=nothing) where FT

    # Update Kz cache for current time
    update_field!(op.kz_field, current_time(meteo))

    # Pre-fetch Kz values into a dense 3D array for kernel
    Nx, Ny, Nz, Nt = size(state.tracers_raw)
    kz_values = _materialize_kz(op.kz_field, Nx, Ny, Nz)

    # Get delp for the grid (layer thicknesses)
    delp = layer_thickness_array(grid, state)

    backend = get_backend(state.tracers_raw)
    kernel = _vertical_diffusion_kernel!(backend, (8, 8, 1))
    kernel(state.tracers_raw, kz_values, delp, FT(dt),
           op.Kz_min, op.Kz_max;
           ndrange=(Nx, Ny, Nt))
    synchronize(backend)

    return state
end
```

Tests in `test/test_diffusion_operator.jl`:
1. Single tracer, `ConstantKzField(Kz=1.0)`, 1D Gaussian bump:
   expected diffusion per analytic solution (ULP tolerance for
   small dt, looser for larger dt)
2. Multi-tracer: all tracers diffuse independently
3. Mass conservation: `sum(get_tracer(state, name)) ≈ const` over
   many steps
4. Observation through accessor API (Decision 11)
5. CPU/GPU agreement

```bash
git commit -m "Commit 3: ImplicitVerticalDiffusion operator + apply!"
```

### Commit 4: Palindrome integration in `strang_split!`

Modify `src/Operators/Advection/StrangSplitting.jl` to optionally
call V at palindrome center. The change is scoped to the
`strang_split!` body:

```julia
function strang_split!(state, fluxes, grid, scheme, dt;
                       workspace,
                       diffusion_op::AbstractDiffusionOperator = NoDiffusion())
    # Forward: X Y Z with dt/2
    sweep_x_mt!(..., dt/2)
    sweep_y_mt!(..., dt/2)
    sweep_z_mt!(..., dt/2)

    # Center: V with dt (linear operator, single call)
    apply!(state, nothing, grid, diffusion_op, dt; workspace)

    # Reverse: Z Y X with dt/2
    sweep_z_mt!(..., dt/2)
    sweep_y_mt!(..., dt/2)
    sweep_x_mt!(..., dt/2)
end
```

Note: when `diffusion_op::NoDiffusion`, the `apply!` is a no-op —
behavior equivalent to pre-plan-16 `strang_split!`.

Tests (add to existing advection test suite):
1. With `NoDiffusion`, results identical to pre-plan-16 behavior
   (bit-exact regression check)
2. With `ImplicitVerticalDiffusion`, vertical profiles of a pulse
   broaden per diffusion analytic solution
3. Strang accuracy: second-order convergence in dt with full
   palindrome

```bash
git commit -m "Commit 4: Diffusion integrated into palindrome center"
```

### Commit 5: TransportModel + DrivenSimulation integration

Update `src/Models/TransportModel.jl` to carry diffusion:

```julia
struct TransportModel{T, D, C}
    transport_op::T       # advection scheme
    diffusion_op::D       # ::AbstractDiffusionOperator
    chemistry_ops::C      # ::NTuple (plan 15)
end

function step!(model::TransportModel, state, meteo, grid, dt;
               workspace=nothing, N_chem_substeps::Int = 1)

    dt_transport = dt / N_chem_substeps

    for _ in 1:N_chem_substeps
        transport_block!(state, meteo, grid,
                         model.transport_op,
                         model.diffusion_op,
                         dt_transport; workspace)
    end

    chemistry_block!(state, meteo, grid, model.chemistry_ops, dt;
                     workspace)

    return state
end

function transport_block!(state, meteo, grid, advection_op,
                          diffusion_op, dt; workspace)
    # Advection with diffusion in palindrome center
    strang_split!(state, fluxes_from_meteo(meteo), grid,
                  advection_op, dt; workspace, diffusion_op)
end
```

`DrivenSimulation.jl` updated to construct `TransportModel` with
a default `NoDiffusion` (preserves pre-plan-16 behavior when not
configured), or with `ImplicitVerticalDiffusion(...)` when
diffusion is desired.

Tests:
1. TransportModel with `NoDiffusion`: bit-exact match to prior
   behavior
2. TransportModel with `ImplicitVerticalDiffusion + ConstantKzField`:
   end-to-end test, Rn-222 emitted uniformly, diffuses upward
3. End-to-end CATRINE-like scenario: CO2 + fossil CO2 + SF6 +
   Rn-222 with diffusion, verify mass conservation and
   qualitative behavior

```bash
git commit -m "Commit 5: TransportModel and DrivenSimulation integration"
```

### Commit 6: Benchmarks

Measure:
- Advection-only per-step time (baseline)
- Advection + `NoDiffusion` (should be identical to baseline)
- Advection + `ImplicitVerticalDiffusion(ConstantKzField)`
- Advection + `ImplicitVerticalDiffusion(DerivedKzField)`

Configurations: CPU medium, GPU medium, GPU large; F64; Nt=10;
cfl=0.4.

Expected: diffusion adds 5-20% to per-step time on GPU. On CPU,
probably similar.

If diffusion is >30% of per-step time, investigate: Thomas solve
inefficient? Register pressure? Kernel occupancy?

Document findings in NOTES.md.

```bash
git commit -m "Commit 6: Diffusion benchmarks"
```

### Commit 7: Documentation

Update:
- `ARCHITECTURAL_SKETCH.md` — add diffusion to operator overview,
  document palindrome position
- `CLAUDE.md` — performance note on diffusion cost
- `OPERATOR_COMPOSITION.md` — if any deviation from spec, note it
- NOTES.md retrospective

```bash
git commit -m "Commit 7: Documentation"
```

## 4.5 Test plan per commit

After EACH commit:
```bash
julia --project=. -e 'using AtmosTransport'
julia --project=. test/runtests.jl
```

Compare to baseline. New failures → STOP, revert.

GPU tests:
```bash
julia --project=. test/runtests.jl   # includes GPU tests
```

## 4.6 Acceptance criteria

**Correctness (hard):**
- All pre-existing tests pass (77 pre-existing failures unchanged)
- Diffusion operator produces correct diffused profiles per
  analytic solutions (Gaussian, step function tests)
- Mass conservation exact within ULP for pure diffusion (no
  sources/sinks)
- With `NoDiffusion`, `strang_split!` bit-exact to pre-plan-16
- Beljaars-Viterbo Kz computation matches legacy output (ULP
  tolerance) for reference surface fields

**Code cleanliness (hard):**
- `src/Operators/Diffusion/` directory exists with files per §4.2
- `AbstractDiffusionOperator` defined; `ImplicitVerticalDiffusion`
  and `NoDiffusion` are concrete subtypes
- Three Kz field concrete types in `src/State/Fields/`
- `apply!(state, meteo, grid, op, dt; workspace)` unchanged signature
- Palindrome structure `X Y Z V Z Y X` in `strang_split!`
- `TransportModel` carries diffusion operator

**Performance (soft):**
- Diffusion adds ≤ 30% to per-step time on GPU at Nt=10 (typical)
- CPU and GPU give bit-close results (ULP tolerance)
- No regression in advection-only per-step time

**Interface validation (hard):**
- `apply!` signature works for diffusion unchanged (plan 15 validated)
- `TimeVaryingField` works for 3D Kz (validates the abstraction
  beyond plan 15's scalar use)
- Tridiagonal structure preserved — transposition is clearly
  identifiable in the code (future adjoint kernel can be written
  without restructuring)

**Documentation:**
- NOTES.md complete with:
  - Legacy code inventory (what ported, what deferred)
  - Interface validation findings
  - Adjoint-structure preservation verification
  - Benchmark results

## 4.7 Rollback plan

Standard:
- Do not "fix forward"
- Revert to last-known-good commit
- Write failure in NOTES.md
- Stop if stuck >30 min

Specific rollback points:
- **Commit 1 `DerivedKzField` Beljaars-Viterbo port fails unit
  test.** Likely sign error, unit confusion (m²/s vs Pa²/s), or
  boundary condition handling. Cross-reference with legacy
  `pbl_diffusion.jl` line by line.
- **Commit 2 Thomas solve fails analytic test.** Most likely:
  boundary condition (zero-flux at top/bottom, or Dirichlet),
  coefficient sign error, or index-off-by-one. Debug with small
  Nz (e.g., Nz=5) and verify tridiagonal system by hand.
- **Commit 3 kernel launch fails on GPU.** Register pressure
  issue likely. Reduce work per thread (split tracer dimension
  into separate launches, or use shared memory for intermediates).
- **Commit 4 Strang accuracy test fails.** Check: is V at correct
  palindrome position? Is dt_transport = dt or dt/2 in the reverse
  half? Verify by comparing to a 2-operator reference
  (X(dt/2) V(dt/2) V(dt/2) X(dt/2) — should be second-order accurate
  in dt).
- **Commit 5 DrivenSimulation integration breaks non-diffusion
  test cases.** Default to `NoDiffusion` should preserve behavior
  bit-exactly. If not, investigate whether TransportModel
  construction is failing on old configurations.

## 4.8 Known pitfalls

1. **"I'll skip the `TimeVaryingField` wrapper for Kz — just pass
   a 3D array to `ImplicitVerticalDiffusion`."** NO per Decision 3.
   The `TimeVaryingField` abstraction is the whole point: unifies
   multiple Kz sources. Passing a raw array loses the time-varying
   semantics and makes the operator source-specific.

2. **"Crank-Nicolson is second-order; let me use it instead of
   Backward Euler."** NO per Decision 2. BE's dissipation is a
   feature for sharp tracer profiles; CN oscillates. Legacy uses
   BE; stay consistent.

3. **"I'll use `Vector{AbstractDiffusionOperator}` in TransportModel."**
   Type-unstable. Use a single field `diffusion_op::D` with D
   being the concrete type, or `Tuple` if multiple diffusion
   operators are ever needed (unlikely).

4. **"The adjoint test should actually verify the adjoint solve
   works."** NO per §3.2. Plan 16 preserves tridiagonal STRUCTURE
   for future adjoint but does NOT ship the adjoint kernel.
   Document the transposition rule, verify it conceptually, defer
   actual adjoint port.

5. **"Kz values in legacy are in Pa²/s, let me just scale to m²/s
   by multiplying by some factor."** Check units carefully. Kz
   in physics is m²/s. Legacy `BoundaryLayerDiffusion` works in
   pressure coordinates where Δz is Pa and Kz is Pa²/s. When
   porting, choose ONE convention (recommend m²/s geometric) and
   convert at the meteorology boundary if needed.

6. **"`DerivedKzField.cache` should be GPU memory so
   `field_value` is kernel-callable."** YES — when state is on
   GPU, cache must be GPU array. Allocate with `similar(tracers_raw,
   Nx, Ny, Nz)` to match backend. The `update_field!` computation
   may happen on CPU then copy to GPU, or happen directly on GPU
   if kernelizable.

7. **"Beljaars-Viterbo is only for ERA5; other reanalyses need
   different formulas."** YES. `DerivedKzField` as defined here
   uses B-V. Other parameterizations become additional concrete
   types (e.g., `LouisDerivedKzField` for Louis 1979). Future
   plans can add these; plan 16 ships only B-V.

8. **"Palindrome center means V runs between Z and Z, so V gets
   dt/2 not dt."** NO per Decision 8. V is linear; V(dt/2) + V(dt/2)
   = V(dt). Ship single V(dt) call at palindrome center. Same
   math, simpler code.

9. **"Let me also port `NonLocalPBLDiffusion` from legacy."** NO
   per §3.2. Defer. It's a significant additional scope (counter-
   gradient term, different RHS) and validating local diffusion
   first is more important.

10. **"The tests should construct `ImplicitVerticalDiffusion` once
    and reuse."** MAYBE. For unit tests, fine. For integration
    tests, be aware that `DerivedKzField` has internal state
    (cache); reusing an operator across unrelated test scenarios
    can mask bugs. Prefer constructing fresh operators per test
    unless specifically testing cache behavior.

11. **"I can pass `meteo = nothing` to diffusion's `apply!`, like
    chemistry does."** NO when using `DerivedKzField`.
    `update_field!(kz_field, current_time(meteo))` needs a real
    meteorology; `nothing` breaks dispatch AND can't supply
    current time or surface fields. Test helpers using
    `ConstantKzField` MAY pass `nothing` (the kz_field's
    `update_field!` is a no-op), but this creates an inconsistency
    with the `DerivedKzField` path.
    Recommendation: require real meteorology for all diffusion
    tests; use a `FakeMeteorology` stub for unit tests that
    doesn't actually need meteorological data. This is
    inconsistent with plan 15's "accept nothing" pattern for
    chemistry, but chemistry genuinely doesn't need meteorology
    — diffusion does (for the primary use case).

12. **"Plan 16 should also solve the emissions ordering issue."**
    NO per §3.2. The tension between chemistry-at-sim-level
    (plan 15 workaround) and the TM5 ordering
    `advection → emissions → chemistry` is plan 17's concern.
    Plan 16 ships diffusion as `V` at palindrome center; plan 17
    adds emissions as `S` and resolves the ordering.

13. **"The retrofit of chemistry rates to `TimeVaryingField` is
    out of scope."** NO per §3.2.1. Plan 15 D2 deferred this
    explicitly BECAUSE `TimeVaryingField` didn't exist. Plan 16
    creates it. Deferring the retrofit further leaves chemistry
    inconsistent with the rest of the architecture. Ship it in
    Commit 2a.

---

# Part 5 — How to Work

## 5.1 Session cadence

- Session 1: Commit 0 + Commit 1a (ConstantKzField + PreComputedKzField)
- Session 2: Commit 1c (DerivedKzField — the physics port)
- Session 3: Commit 2a (chemistry retrofit) + Commit 2 (Thomas solve + kernel)
- Session 4: Commit 3 (operator + apply!)
- Session 5: Commit 4 (palindrome integration)
- Session 6: Commits 5-6 (TransportModel + benchmarks)
- Session 7: Commit 7 (docs)

Plan 16 is larger than plan 15 but smaller than plan 14. The
`DerivedKzField` port (Commit 1c) is the biggest risk — porting
300 lines of physics with reference to legacy.

## 5.2 When to stop and ask

- `DerivedKzField` reference validation fails (legacy output
  differs from ported implementation beyond ULP)
- Tridiagonal solve fails analytic diffusion test
- GPU kernel has register spill issues that can't be easily fixed
- TransportModel refactor breaks non-diffusion tests
- Scope creep toward non-local diffusion or adjoint kernel
- Performance regression >30% on advection-only path

## 5.3 NOTES.md discipline

Three specific items to capture:

1. **Interface validation** — did `apply!` signature work for
   diffusion without modification? If yes, that's confirmation
   plans 17-18 can rely on it. If no, document the reason and
   propose signature extension.

2. **Adjoint-structure preservation** — verify by inspection that
   `(a, b, c)` coefficients are accessible in the forward path.
   Document the transposition rule so a future adjoint port is
   mechanical.

3. **TimeVaryingField for 3D Kz** — does the abstraction work
   cleanly for 3D time-varying fields? If any concrete type needed
   extending the abstract interface, document.

---

# End of Plan

After this refactor ships:
- `ImplicitVerticalDiffusion` operator in `src/Operators/Diffusion/`
- Three `TimeVaryingField{FT, 3}` concrete types for Kz
- Palindrome structure `X Y Z V Z Y X` in transport block
- TransportModel carries diffusion alongside advection and chemistry
- Beljaars-Viterbo computation preserved for ERA5-style met sources
- Tridiagonal structure preserved for future adjoint port
- Non-local diffusion and adjoint kernel deferred

The next plans:
- Plan 17: Surface emissions + ordering study.
  - Adds S to palindrome: `X Y Z V S V Z Y X` (two V half-steps
    wrap emissions, physical coupling)
  - Ordering study: compare `V S V` (palindrome) vs `S V` (post-
    emission diffusion) vs `V S` (pre-emission diffusion), quantify
    layer-1 pileup in each
- Plan 18: Convection C.
- Plans 17+18 both reference plan 16's diffusion operator.
