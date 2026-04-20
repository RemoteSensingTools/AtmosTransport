# Plan 18 Corrections Addendum (v2) — Authoritative

**Status:** supersedes my earlier corrections addendum (which was
written against a stale local snapshot and got several things
wrong). This v2 is written against the current working tree
(`AtmosTransport-surface-emissions.zip`, plans 11-17 merged).

Reviewed against: Codex review (5 items on plan 18 coherence),
GPT-5.4 review (3 items on current-tree regressions), direct
inspection of `src/` in the new snapshot.

---

## Executive summary

| Item | Severity | Type | Where to fix |
|------|----------|------|--------------|
| Face-indexed `apply!` signature drift | HIGH | current tree bug | Pre-plan-18 fix commit |
| `current_time` has no concrete overrides | MEDIUM | current tree gap | Pre-plan-18 fix commit |
| Stale v1-vs-v2 test harnesses | MEDIUM | current tree bug | Pre-plan-18 fix commit |
| Plan 18 basis contract contradictions | HIGH | plan 18 defect | §B below |
| Plan 18 missing driver/window wiring | HIGH | plan 18 scope gap | §C below |
| Plan 18 missing CFL sub-cycling | HIGH | plan 18 algorithm gap | §D below |
| Plan 18 artifact path typo | LOW | plan 18 cosmetic | §E below |

**Prior addendum H1 (tracer state interface) is WITHDRAWN.** My
earlier claim that the state uses per-tracer `NamedTuple` mass
arrays was based on stale code. Current `src/State/CellState.jl`
uses `tracers_raw` packed storage exactly as the base plan
assumes — that interface is correct as originally specified.

---

## Part I — Current-tree fixes (PRECEDE plan 18)

These three items break `main` right now. Plan 18 cannot assume
the contract works across all grid topologies until they're
fixed. Recommend shipping as a single "plan 17.5" cleanup commit
or three separate fixes, then starting plan 18.

### A1. Face-indexed `apply!` signature drift

**Current bug:** `TransportModel.step!` forwards `diffusion_op`,
`emissions_op`, `meteo` to `apply!`. The structured-lat-lon
`apply!` accepts these (StrangSplitting.jl:1008-1015). The
face-indexed `apply!` does NOT (StrangSplitting.jl:1035-1039) —
it takes only `workspace` and `cfl_limit`. Face-indexed runtime
throws `MethodError` on any call from `step!`.

Regression: `test_transport_binary_reader.jl:201` fails.

**Fix:**

1. Update face-indexed `apply!` signature at `StrangSplitting.jl:1035`
   and `:1081` (there are two — Constant schemes, and Linear/
   Quadratic schemes) to match structured:
   ```julia
   @eval function apply!(state::CellState{B}, fluxes::FaceIndexedFluxState{B},
                         grid::AtmosGrid{<:AbstractHorizontalMesh},
                         scheme::$scheme_type, dt;
                         workspace::AdvectionWorkspace,
                         cfl_limit::Real = one(eltype(state.air_mass)),
                         diffusion_op::AbstractDiffusionOperator = NoDiffusion(),
                         emissions_op::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                         meteo = nothing) where {B <: AbstractMassBasis}
   ```

2. Implement the Strang palindrome inside the face-indexed path.
   Structured path's `strang_split!` does `X Y Z V S V Z Y X` (per
   plan 17). Face-indexed needs the same semantics. Likely requires:
   - Vertical diffusion operator works face-indexed already (plan
     16b shipped it — verify `ImplicitVerticalDiffusion.apply!`
     works for both topologies)
   - Surface flux operator works face-indexed already (plan 17
     shipped it — verify)
   - The face-indexed horizontal sweep needs to be wrapped in the
     palindrome just like structured

3. Regression test must pass:
   `julia --project=. test/test_transport_binary_reader.jl`

**Time estimate:** 1-2 days depending on how much face-indexed
plumbing needs extending.

### A2. Stale v1-vs-v2 test harnesses

**Current bug:** After `src_v2 → src` promotion, these files
still double-include:
- `test_dry_flux_interface.jl:23-24`
- `test_real_era5_v1_vs_v2.jl:29-30`

Second file is segfaulting on rerun per GPT-5.4's review.

**Fix:** either drop the duplicate `include` lines, or delete the
v1-vs-v2 harnesses entirely (the v1/v2 distinction no longer
exists). Prefer deletion — `test_real_era5_v1_vs_v2.jl` especially
is obsolete.

**Time estimate:** 1-2 hours.

### A3. `current_time` has no concrete driver overrides

**Current gap:** `AbstractMetDriver.current_time` is a stub that
returns `0.0` (AbstractMetDriver.jl:85). No concrete override in
`src/`. Any `StepwiseField` / `AbstractTimeVaryingField` that
reads `current_time(meteo)` silently gets time=0 throughout the
run.

This is not an active bug — shipped tests don't exercise real
time-varying forcing, and `ConstantField` doesn't care about
time. But it's a landmine for any genuine time-varying-field
use, including plan 17 emissions with `StepwiseField` across
multiple windows and plan 18's convection if it's held as
`TimeVaryingField` (see §B below).

**Fix:**

1. Add `current_time(::TransportBinaryDriver)` override backed
   by the driver's clock.
2. Decide the unit (seconds since epoch? hours since run start?
   — whatever `StepwiseField.boundaries` uses).
3. Verify `DrivenSimulation.step!` threads a real driver through
   (it does, per DrivenSimulation.jl:275).
4. Add a regression test: construct a driver, advance through
   two windows, verify `current_time(driver)` advances
   accordingly. Currently not tested.

**Time estimate:** 0.5-1 day.

---

## Part II — Plan 18 corrections

### B. Basis contract: single authoritative rule

**Base plan defect:** Decisions 10, 13, 18 in the plan 18 text
contradict each other. Decision 13 says moist retained; Decision
18 says CMFMC is dry and TM5 is moist; Decision 10 says both
run on dry for cross-scheme test. All three coexist.

**Authoritative rule — replaces all three:**

The basis follows the state's type parameter. Both convection
operators are basis-polymorphic. Driver is responsible for
basis-appropriate forcing upstream.

```julia
struct CMFMCConvection end   # no basis parameter
struct TM5Convection end     # no basis parameter

function apply!(
    state::CellState{B}, met::AbstractTransportWindow{B}, grid,
    op::Union{CMFMCConvection, TM5Convection}, dt; workspace
) where B <: AbstractMassBasis
    # B is compile-time known. Kernel operates on whatever basis
    # the state is on. Inputs (cmfmc, dtrain or entu/detu/entd/detd)
    # must be on the same basis as state.air_mass.
end
```

Driver responsibility: the `DryFluxBuilder` (or new
`DryConvFluxBuilder`) applies `cmfmc_dry = cmfmc_moist × (1 - qv)`
when the driver is configured for dry basis. Existing horizontal-
flux dry-correction plumbing is the template.

**Cross-scheme test (replaces Decision 10):** run both operators
on the state's basis (dry in CATRINE use case). Construct two
windows with matching physical forcing on the same basis. No
basis mismatch in the comparison. Tolerance ~5% on column-
integrated mass. Remaining 5% comes from explicit vs implicit
discretization (~2-3%) and well-mixed sub-cloud treatment
(~1-2%).

**Drop from plan 18:**
- Decision 10 (as worded) — replaced above
- Decision 13 — dropped
- Decision 18 (dual-basis per operator) — dropped
- All "7-10% tolerance" language — replaced by ~5%

**Retain in plan 18:**
- Basis audit of upstream references in Commit 0 NOTES (useful
  documentation work)
- Operator docstrings stating that basis follows state

### C. Met-driver and window wiring

**Base plan defect:** the plan claims no driver API changes but
expects `CMFMC`/`DTRAIN`/`entu/detu/entd/detd` to reach the
kernel. Current state (verified against `src/`):

- `AbstractMetDriver.supports_convection()` returns `false`; no
  overrides
- `StructuredTransportWindow` has no convection fields
- `load_transport_window!` doesn't call `load_cmfmc_window!` or
  `load_tm5conv_window!`

The ERA5 reader CAN load convection blocks (`BinaryReader.jl:422`,
`:463`). Nothing routes them through the driver contract.

**Scope expansion — Commit 7 becomes substantive:**

1. **Add `ConvectionForcing` struct** as a field of the window:
   ```julia
   struct ConvectionForcing{CM, DT, TM}
       cmfmc      :: CM   # (Nx, Ny, Nz+1) or Nothing
       dtrain     :: DT   # (Nx, Ny, Nz) or Nothing
       tm5_fields :: TM   # ::NamedTuple{(:entu,:detu,:entd,:detd)} or Nothing
   end
   ```

2. **Extend window structs** (both structured and face-indexed):
   ```julia
   struct StructuredTransportWindow{Basis, M, PS, F, Q, D, C} <: ...
       air_mass :: M
       ...existing fields...
       convection :: C   # ::Union{Nothing, ConvectionForcing}
   end
   ```

3. **Extend `load_transport_window!`:** when
   `supports_convection(driver)`, call
   `load_cmfmc_window!(reader, win)` and populate
   `window.convection.cmfmc/dtrain`.

4. **Add driver method overrides:**
   ```julia
   supports_convection(d::TransportBinaryDriver) = has_cmfmc(d.reader)
   ```

5. **Back-compat:** runs without convection get
   `convection === nothing` in their windows. No operator wired
   in the model. Existing tests unchanged.

6. **Don't wrap cmfmc/dtrain in `TimeVaryingField`.** The window
   already carries all forcing as plain arrays (air_mass, fluxes,
   qv_start, qv_end). Convection forcing follows the same
   pattern. `PreComputedConvMassFluxField` from the base plan
   §3 and Commit 2 is not needed — just add plain arrays to
   `ConvectionForcing`.

   Rationale: the window IS the time-windowed forcing container.
   Wrapping its fields in an extra `TimeVaryingField` layer
   duplicates the window contract with no benefit. Also avoids
   relying on `current_time(meteo)` for convection (A3 concern
   becomes moot for convection specifically).

**Drop from plan 18:**
- Commit 2 as written (`PreComputedConvMassFluxField`). Replace
  with "extend `StructuredTransportWindow` and
  `FaceIndexedTransportWindow` to carry optional `ConvectionForcing`"
- §3 description of convection forcing as `AbstractTimeVaryingField`

**Commit 7 sizing:** was sketched as a small wiring commit;
needs to be ~1.5-2 days. Expanded scope absorbs all of §C.

### D. CMFMC CFL sub-cycling is part of the algorithm

**Base plan defect:** the CMFMCConvection kernel signature takes
a single `dt`, no sub-stepping. GCHP and legacy both sub-cycle.
Without sub-cycling, positivity/stability goals fail at
realistic timesteps.

**Corrected scope for Commit 3:**

```julia
function apply!(state, met, grid, op::CMFMCConvection, dt; workspace)
    # 1. Compute CFL-safe sub-step count (cached per window)
    n_sub, sdt = get_or_compute_cmfmc_subcycling(op, met, state, workspace, dt)
    
    # 2. Sub-step loop
    for _ in 1:n_sub
        _cmfmc_kernel!(state.air_mass, state.tracers_raw,
                       met.convection.cmfmc, met.convection.dtrain,
                       workspace.qc_scratch, sdt, workspace)
    end
end
```

**CFL rule:** `n_sub = max(1, ceil(max(cmfmc × dt / air_mass) / cfl_safety))`
with `cfl_safety = 0.5`. Typical values: 5-15 sub-steps per
CATRINE 30-minute dynamic dt in deep-convective regions.

**Caching:** cmfmc/dtrain constant within a window. Cache n_sub
in the workspace:

```julia
struct CMFMCWorkspace{FT, QC}
    qc_scratch   :: QC
    cached_n_sub :: Base.RefValue{Int}
    cache_valid  :: Base.RefValue{Bool}
end

function invalidate_cmfmc_cache!(ws) ; ws.cache_valid[] = false ; end
```

Driver calls `invalidate_cmfmc_cache!(ws)` on window roll. First
`apply!` in the window computes n_sub; subsequent calls reuse.

**Acceptance criterion (add to §4.6):** for a column with known
CFL n_sub=N, a single `apply!` call must match N manual calls of
sdt = dt/N to machine precision.

**This applies only to CMFMCConvection.** TM5Convection is
unconditionally stable (implicit matrix solve) and does not need
sub-cycling. The matrix-build cost already accounts for the full
dt.

### E. Artifact path

The filesystem path is `docs/plans/18_ConvectionPlan/` (camelCase),
not `docs/plans/18_CONVECTION_PLAN/` (base plan typo). All
artifacts go to:

- `docs/plans/18_ConvectionPlan/notes.md`
- `docs/plans/18_ConvectionPlan/upstream_fortran_notes.md`
- `docs/plans/18_ConvectionPlan/upstream_gchp_notes.md`
- `docs/plans/18_ConvectionPlan/validation_report.md`
- `docs/plans/18_ConvectionPlan/benchmark_results.md`

---

## Summary table — commit impact

| Commit | Base plan | With this addendum |
|--------|-----------|-------------------|
| **Pre-18: Fix A1** | N/A | Face-indexed `apply!` signature extension, palindrome implementation, `test_transport_binary_reader.jl` regression pass (~1-2 days) |
| **Pre-18: Fix A2** | N/A | Clean up stale v1-vs-v2 harnesses (~2h) |
| **Pre-18: Fix A3** | N/A | `current_time` driver overrides + regression test (~0.5-1 day) |
| 0 (NOTES) | Upstream Fortran survey | Same + verify current state against plan assumptions |
| 1 (AbstractConvectionOperator) | Type hierarchy | Same — basis-polymorphic (no basis parameter) |
| 2 (PreComputedConvMassFluxField) | **DROPPED** — not needed | Replaced by §C window extension |
| 3 (CMFMCConvection) | Kernel | Kernel + **mandatory CFL sub-cycling** + well-mixed sub-cloud + inline helpers (§D) |
| 4 (TM5Convection) | Kernel | Kernel (no sub-cycling; matrix is unconditionally stable) |
| 5 (cross-scheme test) | ~7-10% mixed-basis | ~5% same-basis (§B) |
| 6 (TransportModel) | Wire operator | Same |
| **7 (driver integration)** | Small wiring | **Major expansion (§C): ConvectionForcing, window struct, load methods, supports_convection overrides** (~1.5-2 days) |
| 8 (DrivenSimulation) | Wiring | Same |
| 9 (position study) | Keep | Same |
| 10 (benchmarks) | Keep | Same |
| 11 (retrospective) | Keep | Same |

**Net delta vs base plan:** +3 pre-18 fix commits (~2.5-4 days),
Commit 2 dropped (savings ~1 day), Commit 3 and 7 expanded
(+2-3 days). Overall plan 18 grows by ~2-5 days including
pre-fixes.

---

## Decisions: what stays, what drops, what's new

**DROP:**
- Decision 10 (as worded)
- Decision 13 ("moist basis retained throughout")
- Decision 18 (dual-basis per operator) 
- All "7-10% tolerance" language
- `PreComputedConvMassFluxField` as a `TimeVaryingField`
- All H1 from my earlier addendum (tracer state interface
  concerns) — withdrawn

**STAND:**
- Decision 1 (convection as separate block in step!)
- Decision 2 (two concrete types)
- Decision 8/19 (inline helpers with dispatch-ready signatures)
- Decision 9 (three-tier validation)
- Decision 11 (adjoint-structure preservation — see adjoint addendum)
- Decision 14 (upstream Fortran read at Commit 0)
- Decision 15 (medium cleanup CMFMC, light TM5)
- Decision 16 (no scavenging infrastructure)
- Decision 17 (well-mixed sub-cloud layer — ADD vs legacy)

**NEW:**
- **Decision 20: Basis follows state, not operator.** Operators
  are basis-polymorphic. Driver handles basis conversion upstream.
- **Decision 21: CFL sub-cycling in CMFMCConvection is mandatory.**
  Cached per window. Not optional.
- **Decision 22: Convection forcing lives in the transport
  window** as plain arrays in `ConvectionForcing`, not wrapped
  in `TimeVaryingField`.

---

## For the execution agent

Read in this order:

1. `18_CONVECTION_PLAN.md` (base plan, v3)
2. This corrections addendum (authoritative on overlap)
3. `18_CONVECTION_ADJOINT_ADDENDUM.md` (additive, no overlap)

Where base plan and this addendum conflict, this addendum wins.

**Ship the three pre-18 fixes (A1, A2, A3) first, in a separate
commit or small commit series.** Then start plan 18 Commit 0.

If during plan 18 execution another contradiction surfaces
between documents, stop and ask. Do not resolve ad-hoc.

---

**End of v2 corrections addendum.**
