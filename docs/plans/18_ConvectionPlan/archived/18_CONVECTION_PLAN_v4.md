# Plan 18 — Convection Operators (v4, authoritative)

**Status:** Ready for execution after prerequisites (A1, A2, A3 from
`PRE_PLAN_18_FIXES.md`) ship.
**Target branch:** `convection`, forked from the tip of `surface-emissions`
(plan 17) *after* the pre-plan-18 fix commits land.
**Estimated effort:** 3-4 weeks, single engineer / agent.
**Primary goal:** ship two concrete convection operators —
`CMFMCConvection` (GCHP path, CMFMC + DTRAIN) and `TM5Convection`
(TM5 path, four-field matrix scheme) — as first-class operators
conforming to the plans 15 / 16b / 17 operator interface.
Validate physics against upstream Fortran references (TM5
`tm5_conv.F90`, GEOS-Chem `convection_mod.F90`) and paper
equations (Tiedtke 1989, Moorthi-Suarez 1992), not just against
legacy Julia. Integrate at step level as a separate block after
the transport block.

This document consolidates and supersedes:

- `18_CONVECTION_PLAN.md` (v1 → v3, 1834 lines, had contradictions at
  Decisions 10 / 13 / 18)
- `18_CONVECTION_CORRECTIONS_ADDENDUM_v2.md` (Codex + GPT-5.4 review,
  introduces Decisions 20 / 21 / 22, drops v3 Commit 2)
- `18_CONVECTION_ADJOINT_ADDENDUM.md` (purely additive: adjoint-identity
  test, no-positivity-clamp rule, Plan 19 registration)

Precedence rules go away: this document is the single source of truth.

**Companion documents (still authoritative):**

- `PRE_PLAN_18_FIXES.md` — three prerequisite current-tree fixes
- `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` — TM5 Fortran reference
- `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` — GCHP Fortran reference
  (v4-corrected basis section)
- `CLAUDE_additions_v2.md` — accumulated lessons (validation
  discipline, basis-convention section)

---

# Part 1 — Context

## 1.1 Purpose and scope

AtmosTransport needs convective transport for CATRINE intercomparison.
Two distinct met-data sources are supported in production:

- **TM5 path:** ECMWF ERA5-driven, four fields per column (entu, detu,
  entd, detd) derived from Tiedtke 1989 mass-flux scheme.
- **GCHP path:** GEOS-Chem High-Performance, two fields per column
  (CMFMC + DTRAIN) from MERRA-2 / GEOS-FP / GEOS-IT.

These use different met fields and different underlying physics, but
the tracer-transport operator must present a uniform interface either
way.

Plan 18 ships two concrete convection operators:

1. `CMFMCConvection` — GCHP-style mass-flux redistribution with
   updraft mixing. Two-pass column kernel with mandatory
   CFL sub-cycling.
2. `TM5Convection` — TM5-style four-field matrix scheme with
   in-kernel LU solve. Implicit, unconditionally stable.

Both take their required fields through a new `ConvectionForcing`
struct carried by the transport window. Both integrate at step level
as a dedicated convection block between the transport block and the
chemistry block. Validation follows the three-tier hierarchy
(analytic / paper / cross-implementation) with upstream Fortran as
ground truth, not legacy Julia.

## 1.2 Prerequisites

`PRE_PLAN_18_FIXES.md` specifies three current-tree fixes that must
ship **before** plan 18 Commit 0:

- **A1:** face-indexed `apply!` contract catch-up. Current
  `TransportModel.step!` forwards `diffusion_op`, `emissions_op`,
  `meteo` kwargs. Structured path accepts them; face-indexed does not.
  Fix: extend face-indexed `apply!` signature and implement the
  palindrome inside the face-indexed body. Gates reduced-Gaussian
  runtime.
- **A2:** stale v1-vs-v2 test harness cleanup. Delete or deduplicate
  `test/test_dry_flux_interface.jl` and `test/test_real_era5_v1_vs_v2.jl`.
- **A3:** `current_time` concrete driver overrides. Current
  `AbstractMetDriver.current_time` is a `0.0` stub; plan 17 threads it
  through operator `apply!` methods. Add a real
  `current_time(::TransportBinaryDriver)` backed by the driver's
  window clock and decide a unit convention (recommend: seconds since
  `t_start`).

Plan 18 assumes these have shipped. Starting plan 18 Commit 0 before
A1 ships breaks Commit 8 (DrivenSimulation integration) because
face-indexed runs can't execute the full operator suite.

## 1.3 Current state of `src/` — verified interface claims

Before writing a line of kernel code, verify these claims still hold.
All are verified against current tree at the time this document was
written.

| Claim | Source |
|---|---|
| `CellState{Basis, A, Raw, Names}` with packed `tracers_raw`, `tracer_names::NTuple`, `air_mass` (not `air_mass_dry`), accessor API `get_tracer(state, :CO2)` / `state.tracers.CO2` | `src/State/CellState.jl:42-46` |
| `AbstractMassBasis`, `DryBasis`, `MoistBasis` tags on `CellState{B}` | `src/State/Basis.jl:10-24` |
| `AbstractTransportWindow{Basis}` supertype; `StructuredTransportWindow{Basis, M, PS, F, Q, D}` and `FaceIndexedTransportWindow{Basis, M, PS, F, Q, D}` each have 6 type parameters | `src/MetDrivers/TransportBinaryDriver.jl:18, 33, 42` |
| Window carries `air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas` | `src/MetDrivers/TransportBinaryDriver.jl:33-49` |
| `Adapt.adapt_structure` methods for both window types at `:66` (structured) and `:77` (face-indexed) | as above |
| ERA5 reader exposes `has_cmfmc`, `has_tm5conv`, `load_cmfmc_window!(reader, win; cmfmc=...)`, `load_tm5conv_window!(reader, win; entu=..., detu=..., entd=..., detd=...)` | `src/MetDrivers/ERA5/BinaryReader.jl:173-175, 428-436, 469-483` |
| `supports_convection(::AbstractMetDriver) = false` — no overrides yet | `src/MetDrivers/AbstractMetDriver.jl:95` |
| `current_time(::AbstractMetDriver) = 0.0` — stub (A3 replaces) | `src/MetDrivers/AbstractMetDriver.jl:85` |
| `TransportModel.step!` forwards `diffusion_op`, `emissions_op`, `meteo` and then runs `chemistry_block!` | `src/Models/TransportModel.jl:162-170` |
| Structured `apply!` palindrome `X Y Z [V(dt/2) S V(dt/2)] Z Y X` at `strang_split_mt!` | `src/Operators/Advection/StrangSplitting.jl:1199-1246` |
| Submodule convention: single `operators.jl` + `*_kernels.jl` per submodule | `src/Operators/Diffusion/operators.jl`, `src/Operators/SurfaceFlux/operators.jl` |
| Diffusion workspace contract: required (`ArgumentError` when nothing); caller-owned `dz_scratch`, `w_scratch` | `src/Operators/Diffusion/operators.jl:108-116, 152-185` |
| SurfaceFlux plan 17: `apply_surface_flux!(q_raw, op, ws, dt, meteo, grid; tracer_names)` array-level entry | `src/Operators/SurfaceFlux/operators.jl:122-143` |
| Legacy `ras_convection.jl` comment lines 41-46 claiming "no dry conversion applied before convection" is stale (inconsistent with legacy's own dry-correction kernels) | `src_legacy/Convection/ras_convection.jl:41-46` |
| Legacy RAS test coverage: zero tests (`test_legacy/test_convection.jl` covers Tiedtke + TM5 only) | `test_legacy/test_convection.jl` |
| Baseline 77 pre-existing test failures across `test_basis_explicit_core.jl`, `test_structured_mesh_metadata.jl`, `test_poisson_balance.jl` | plan 12 lineage |

Commit 0 re-runs the verification: if any claim has changed since this
document was written, update the plan before proceeding.

## 1.4 Upstream references

### TM5 Fortran (available in repo)

- `tm5_conv.F90:37-191` — `TM5_Conv_Matrix`: matrix builder.
  Updraft section `:97-118`, downdraft section `:122-135`, assembly
  `:140-149`, final matrix `:167-183`.
- `tm5_conv.F90:197-341` — `TM5_Conv_Apply`: LU solve.
  `dgetrf` + `dgetrs` with `trans='N'` for forward, `'T'` for adjoint.
- `convection.F90:227-747` — driver `convec`; column loop,
  `lmax_conv` handling, matrix build + solve dispatch.
- `tmphys_convec.F90:39-132` — `ConvCloudDim`: cloud top, cloud base,
  level of free sinking.
- `phys_convec_ec2tm.F90:82-400` — ECMWF → TM5 field conversion;
  explains sign / level / full-vs-half conventions (entu, detu, entd,
  detd all at full levels after conversion, `kg / m² / s`).

### GCHP Fortran (available in repo)

- `convection_mod.F90:422-1419` — `DO_RAS_CLOUD_CONVECTION` (RAS
  scheme). Cloud base determination `:626-634`, sub-stepping
  `:1601-1612` (`NS = NDT / 300` standard, `NDT / 60` hi-res),
  well-mixed sub-cloud layer `:725-782`, updraft pass `:785-920`,
  four-term tendency `:988-1007`.
- `convection_mod.F90:1424-2282` — `DO_GF_CLOUD_CONVECTION`
  (Grell-Freitas scheme). Consumes the same CMFMC + DTRAIN interface;
  plan 18's `CMFMCConvection` works for both RAS and GF data.
- `calc_met_mod.F90:185-200` — basis comment block. `DELP_DRY` is
  "needed for transport calculations ... to conserve mass."
- `calc_met_mod.F90:647-660` — `BMASS = DELP_DRY * 100 / g` basis
  derivation.

See `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` and
`18_CONVECTION_UPSTREAM_GCHP_NOTES.md` for per-line algorithm
breakdowns and cross-references to legacy Julia. Commit 0 extends
these notes with any discoveries and saves final versions to
`artifacts/plan18/upstream_fortran_notes.md` /
`.../upstream_gchp_notes.md`.

## 1.5 Non-goals

Plan 18 does NOT ship any of the following. Each is a candidate for
a follow-up plan.

- **Wet scavenging** in any form. No `AbstractTracerSolubility`, no
  `InertTracer` / `SolubleTracer` trait hierarchy, no Henry's-law
  hooks. Inline kernel helpers leave scavenging-ready function
  signatures but the `# TODO:` hooks are signatures, not comments.
- **`TiedtkeConvection` as a standalone type.** `CMFMCConvection`
  with `dtrain === nothing` covers the single-net-flux case via
  runtime fallback (preserves legacy behavior at
  `src_legacy/Convection/ras_convection.jl:303-370`).
- **Derived convective mass flux fields.** Computing CMFMC / DTRAIN
  online from parent T, q, p is out of scope. Operational path
  reads pre-computed fields from the binary. Future plan for
  online (`DerivedConvMassFluxField`).
- **Convection adjoint kernel.** Adjoint *structure* is preserved per
  Decision 11; a dedicated `Plan 19: Adjoint operator suite` ports
  the backward-mode kernels, reuses forward factorizations, and
  wires observation injection.
- **`AbstractLayerOrdering{TopDown, BottomUp}` abstraction.** Deferred
  from plan 17. Convection kernels continue the codebase convention
  `k=1=TOA, k=Nz=surface` (CLAUDE.md Invariant 2), with per-kernel
  column reversal at the TM5 / AtmosTransport interface (legacy
  pattern).
- **Palindrome-centered convection.** Plan 18 ships as a separate
  block after the transport block. Commit 9 (position study)
  compares against palindrome-internal options; any migration is a
  follow-up plan.
- **Performance optimization beyond correctness.** Matrix LU solve
  is column-serial. GPU-batched BLAS or shared-memory factorization
  are future optimization plans. Plan 18 targets correctness.

---

# Part 2 — Design decisions (consolidated)

Every decision is final. If ambiguous during execution, STOP and ask.

## 2.1 Operator hierarchy and types

**Decision 2.** `AbstractConvectionOperator` hierarchy with a
`NoConvection` dead-branch default and two concrete types.

```julia
abstract type AbstractConvectionOperator end

struct NoConvection <: AbstractConvectionOperator end

struct CMFMCConvection <: AbstractConvectionOperator end   # no basis parameter

struct TM5Convection <: AbstractConvectionOperator
    lmax_conv :: Int   # 0 sentinel = use full Nz
end
```

No basis type parameter on either concrete type. Basis follows state
(§2.3). Field access goes through the transport window's
`ConvectionForcing` (§2.4), not through struct fields — the operator
is a pure compute kernel keyed by type.

## 2.2 Composition — convection as separate block

**Decision 1.** Convection runs as a dedicated block after the
transport block and before the chemistry block, not inside the
palindrome.

```
step!(model, dt; meteo):
  transport_block  → X Y Z [V(dt/2) S V(dt/2)] Z Y X          # plan 17
  convection_block → apply!(state, meteo, grid, model.convection, dt;
                            workspace)
  chemistry_block  → chemistry(dt)                             # plan 15
```

Rationale:

- Convection is non-local in the vertical (matrix transport); a
  different operator character from column-local diffusion or
  point-local emissions.
- Matches TM5's operator-splitting pattern (convection as a separate
  phase, not interleaved with horizontal transport).
- Simpler to reason about and to test in isolation.
- Commit 9 (position study) compares Option 1 (this decision) against
  Option 2 (palindrome-internal, C around S) and two alternates. If
  any alternate wins clearly, it's documented as a follow-up plan;
  plan 18 does not reorganize mid-flight.

## 2.3 Basis handling — basis follows state

**Decision 20 (replaces base-plan Decisions 10, 13, 18).** The basis
follows the state's type parameter. Both convection operators are
basis-polymorphic. The driver is responsible for basis-appropriate
CMFMC / DTRAIN / entu / detu / entd / detd upstream.

```julia
function apply!(
    state::CellState{B},
    met::AbstractTransportWindow{B},   # B matches state
    grid::AbstractGrid,
    op::Union{CMFMCConvection, TM5Convection},
    dt::Real;
    workspace,
) where {B <: AbstractMassBasis}
    # B is compile-time known. Kernel operates on whatever basis
    # the state carries. Inputs must be on the same basis.
end
```

Driver responsibility: when the driver is configured for dry basis,
it applies `cmfmc_dry = cmfmc_moist × (1 - qv_interface)` (and
analogous corrections for DTRAIN, entu / detu / entd / detd, using
appropriate qv evaluation per field placement). Existing horizontal-
flux dry-correction plumbing (`DryFluxBuilder`, `latlon_dry_air.jl`,
`cubed_sphere_mass_flux.jl`) is the template. Commit 7 extends this
for convection.

**Cross-scheme test tolerance:** both operators run on the state's
basis. No basis mismatch. Tolerance **~5%** on column-integrated
mass: explicit sub-step (~2-3%) + well-mixed sub-cloud treatment
(~1-2%) = ~5%. Any pre-v4 mention of 7-10% or mixed-basis tolerances
is drift artifact.

**Operator docstrings state explicitly that basis follows state.**
Legacy stale comment at `src_legacy/Convection/ras_convection.jl:41-46`
("No dry conversion is applied before convection") is inconsistent
with the legacy's own dry-correction kernels and is removed / rewritten
during port (Commit 3 housekeeping).

## 2.4 Convection forcing as a window field

**Decision 22 (replaces base-plan `PreComputedConvMassFluxField`).**
Convection forcing is a plain-array struct carried by the transport
window, not wrapped in `AbstractTimeVaryingField`.

```julia
struct ConvectionForcing{CM, DT, TM}
    cmfmc      :: CM   # ::Union{Nothing, AbstractArray{FT, 3}}  (Nx, Ny, Nz+1)
    dtrain     :: DT   # ::Union{Nothing, AbstractArray{FT, 3}}  (Nx, Ny, Nz)
    tm5_fields :: TM   # ::Union{Nothing, NamedTuple{(:entu,:detu,:entd,:detd)}}  each (Nx, Ny, Nz)
end
```

Both window structs extend with a `C` type parameter and an optional
`convection` field:

```julia
struct StructuredTransportWindow{Basis, M, PS, F, Q, D, C} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C   # ::Union{Nothing, ConvectionForcing}
end
# mirror for FaceIndexedTransportWindow
```

Back-compat: runs without convection get `convection === nothing`.
Existing constructors default `convection = nothing`. Existing tests
unchanged.

Rationale: the window IS the time-windowed forcing container. Wrapping
`cmfmc`, `dtrain`, etc. in an extra `TimeVaryingField` layer would
duplicate the window contract with zero benefit — it would also make
convection forcing depend on `current_time(meteo)`, a pathway with a
known gap (A3) and no added expressiveness vs. the window's existing
window-rolled lifecycle. `air_mass`, `fluxes`, `qv_start` / `qv_end`,
`deltas` are already plain arrays held by the window; convection
forcing follows the same pattern.

`Adapt.adapt_structure` methods for both window structs are extended
to thread the `convection` field through device conversion.

## 2.5 Three-tier validation discipline

**Decision 9.** Each operator's test file includes tests at three
tiers. Per `CLAUDE_additions_v2.md` § "Validation discipline for
physics ports".

- **Tier A — Analytic** (~5 tests per operator). Mass conservation,
  zero-forcing identity, positivity, stability, adjoint identity
  (§2.9).
- **Tier B — Paper / Fortran formulas** (~7-10 tests per operator).
  Hand-expand paper equations and / or upstream Fortran for a 5-layer
  idealized column; compare port output to hand expansion at
  machine-precision for linear operators.
- **Tier C — Cross-implementation** (~2-3 tests per operator).
  Standard deep-tropical-convective column test case. Compare against
  upstream Fortran reference calculation or published reference data.
  Agreement ~10% on column-integrated mass, similar qualitative
  vertical profile. Opt-in via `ENV["ATMOSTR_TIER_C_REFS"]` — harness
  lives in repo, reference data gated (not always available at CI
  time).

## 2.6 Inline helpers with dispatch-ready signatures

**Decisions 8 and 19.** Kernel math lives in named `@inline`
helpers taking an implicit scavenging argument that plan 18 omits
(inert default). Scavenging becomes a method dispatch in a future
plan, not a kernel rewrite.

For `CMFMCConvection`, helpers are:

- `_cmfmc_updraft_mix(qc_below, q_env, cmfmc_below, entrn, cmout, tiny)`
  — returns `(qc_post_mix, qc_scav)`; inert version returns
  `(qc, zero(qc))`.
- `_cmfmc_apply_tendency(q_env, q_above, qc_post_mix, qc_pres,
  cmfmc_below, cmfmc_above, dtrain, bmass, dt)` — inert version uses
  two-term form (§5.3 of GCHP notes proves equivalence to four-term
  form for inert tracers).
- `_cmfmc_wellmix_subcloud(q_slice, delp_slice, q_cldbase,
  cmfmc_at_base, dt)` — pressure-weighted average applied uniformly
  to sub-cloud layers (§2.7).

For `TM5Convection`, helper is:

- `_tm5_scavenging_matrix(entu, detu, cp, sceffdeep, fu)` — inert
  version returns zero matrix (dispatches away to no-op). Future
  wet-deposition plan adds a soluble method.

Future plan adds methods on these helpers keyed by a solubility
parameter. No type hierarchy shipped in plan 18.

## 2.7 Well-mixed sub-cloud layer — ADD vs legacy

**Decision 17.** `CMFMCConvection` implements the well-mixed sub-cloud
layer treatment from GCHP `convection_mod.F90:742-782`. This is a
**deliberate improvement over legacy Julia**, which lacks it entirely.

```julia
# Below cloud base: pressure-weighted well-mixed
if cldbase > 1 && cmfmc_at_cldbase_interface > tiny
    qb, mb = pressure_weighted_avg(q, delp, 1:cldbase-1)
    qc_mixed = (mb * qb + cmfmc_at_cldbase * q[cldbase] * dt) /
               (mb + cmfmc_at_cldbase * dt)
    q[1:cldbase-1] .= qc_mixed
end
```

Rationale: for surface-source tracers (Rn-222 especially, also CO2
over tropical continents, SF6 over strong sources), the sub-cloud
well-mixed layer redistributes surface tracer mass before convective
uplift. Without it:

- Surface concentrations bias HIGH (source-region emissions don't
  mix horizontally within the sub-cloud layer before convection).
- Free-troposphere concentrations bias LOW (less well-mixed source
  air entering the updraft).

Commit 3 Tier C verifies the improvement — port with sub-cloud fix
should agree better with GCHP output than port without.

## 2.8 CFL sub-cycling in `CMFMCConvection` — mandatory

**Decision 21.** The `CMFMCConvection` kernel sub-cycles internally to
maintain CFL stability. This is mandatory, not optional. TM5Convection
does not sub-cycle (implicit matrix solve is unconditionally stable).

```julia
function apply!(
    state::CellState{B}, met::AbstractTransportWindow{B}, grid,
    op::CMFMCConvection, dt; workspace,
) where {B <: AbstractMassBasis}
    n_sub, sdt = get_or_compute_cmfmc_subcycling(op, met, state, workspace, dt)
    for _ in 1:n_sub
        _cmfmc_kernel!(state.air_mass, state.tracers_raw,
                       met.convection.cmfmc, met.convection.dtrain,
                       workspace.qc_scratch, sdt, workspace)
    end
end
```

**CFL rule:**
`n_sub = max(1, ceil(Int, max_over_grid(cmfmc_k × dt / air_mass_k) / cfl_safety))`
with `cfl_safety = 0.5`. Typical values at CATRINE configuration:
5-15 sub-steps per 30-minute dynamic `dt` in deep-convective regions,
1 in quiet regions.

**Caching.** CMFMC / DTRAIN are constant within a met window. `n_sub`
is cached in the workspace:

```julia
struct CMFMCWorkspace{FT, QC}
    qc_scratch   :: QC                   # (Nx, Ny, Nz), updraft conc
    cached_n_sub :: Base.RefValue{Int}
    cache_valid  :: Base.RefValue{Bool}
end
invalidate_cmfmc_cache!(ws::CMFMCWorkspace) = (ws.cache_valid[] = false; nothing)
```

Driver calls `invalidate_cmfmc_cache!` on window roll (Commit 7). First
`apply!` in a window computes `n_sub`; subsequent `apply!` calls within
the same window reuse.

**Acceptance criterion** (added to §4 and Commit 3): for a column with
CFL-derived `n_sub = N`, a single `apply!` call at `dt` must match
`N` manual calls of `sdt = dt/N` to machine precision. Regression test
ships in Commit 3 Tier A.

## 2.9 Adjoint structure preservation

**Decision 11 (consolidated with adjoint addendum).** Both operators
preserve the structure needed for a mechanical adjoint port in a
dedicated `Plan 19: Adjoint operator suite`. Plan 18 does not ship
adjoint kernels but preserves linearity and stores coefficients in
transposable form.

**Three concrete requirements, each testable:**

1. **Kernel is linear in tracer mixing ratio.** No positivity
   clamps, no `max(q, 0)` conditionals, no `if q_new < 0` branches
   inside the core kernel. GCHP's Fortran includes a clamp at
   `convection_mod.F90:1002-1004`:
   ```fortran
   IF ( Q(K) + DELQ < 0 ) DELQ = -Q(K)
   Q(K) = Q(K) + DELQ
   ```
   **Do NOT port this clamp.** It breaks the adjoint-identity test
   (nonlinear dependence on forward trajectory) and would require
   storing the forward state to adjoint. Alternative strategies:
   - **Accept tiny negativities** and let the global mass fixer
     absorb them. This is what legacy Julia does
     (`src_legacy/Convection/ras_convection.jl:208-214` comment:
     "The global mass fixer in the run loop handles the small column
     drift from entrainment clipping").
   - **Pre-step met-data validation** (optional, upstream):
     `_check_mass_flux_consistency` detects `cmfmc / dtrain / delp`
     combinations that would remove more mass than present, and
     either warns or caps the forcing before the kernel sees it.
     This keeps the nonlinearity in preprocessing, where it doesn't
     affect the tracer-state adjoint.
   Plan 18 ships option 1 (accept + mass fixer) with the option to
   add option 2 upstream at driver level if negativities become
   observable in production.

2. **Coefficients stored in transposable form.** For `TM5Convection`,
   the matrix `conv1` is explicit in workspace `conv1_ws[row, col, i, j]`.
   Adjoint solve reuses the same LU factorization with `trans='T'`
   (mirrors TM5 Fortran `TM5_Conv_Apply` dispatching on `trans`
   character arg at `tm5_conv.F90:197-341`). For `CMFMCConvection`,
   mass-flux coefficients at each level (`cmfmc_k`, `dtrain_k`,
   `entrn_k`) are named locals in the kernel, not fused.

3. **Adjoint-identity test per operator (Tier A).** Verify
   `⟨y, L·x⟩ ≈ ⟨L^T·y, x⟩` for random `x, y` on a small column.
   Tests (Commits 3 Tier A and 4 Tier A):
   - `TM5Convection`: extract `conv1` from workspace, compute dense
     `L⁻¹`, test `⟨y, L⁻¹x⟩ ≈ ⟨(L^T)⁻¹y, x⟩` via LU transpose solve.
     No extra kernel calls.
   - `CMFMCConvection`: build dense `L` by applying the kernel to
     each unit vector `e_k` in an Nz=8 column (8 kernel calls);
     compute `L^T`; verify identity. Negligible cost.

**Operator docstrings** include a "# Adjoint path (not shipped in
plan 18)" paragraph documenting how the future Plan 19 adjoint
kernel reuses forward structure. Required text specified in Commit 3
and Commit 4.

## 2.10 Cleanup aggressiveness

**Decision 15.**

- **`CMFMCConvection`:** MEDIUM cleanup vs legacy `ras_convection.jl`
  (~400 lines). Room for clearer organization: modernize variable
  names; separate "compute coefficients" from "apply coefficients"
  into distinct inline helpers (§2.6); split compound expressions
  for readability; remove the stale "no dry conversion" comment at
  legacy lines 41-46. ADD the well-mixed sub-cloud layer (§2.7).
  DO NOT port the positivity clamp (§2.9).
- **`TM5Convection`:** LIGHT cleanup vs legacy
  `tm5_matrix_convection.jl` (~625 lines). Matrix algorithm has
  subtle invariants (column reversal, Sander Houweling subsidence
  correction, diagonal dominance). Stay close to legacy structure.
  Use modern naming and `ConvectionForcing` for inputs. Port the
  matrix builder faithfully with comments linking to Fortran line
  numbers.

## 2.11 No scavenging infrastructure

**Decision 16.** No `AbstractTracerSolubility`, `InertTracer`,
`SolubleTracer`, `wet_scavenge_fraction`, or Henry's-law code ships
in plan 18. The inline helpers (§2.6) are signature-ready for future
scavenging but do not parameterize on solubility traits. Wet-deposition
trait system, scavenging coefficients (`cp`, `sceffdeep`, `lbdcv`,
`cvsfac`), and the four-term tendency form for soluble tracers are
all deferred to a future wet-deposition plan.

## 2.12 Read upstream Fortran before writing Julia

**Decision 14.** Commit 0 MUST include reading the upstream Fortran
references listed in §1.4 and extending the pre-populated notes
(`18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md`,
`18_CONVECTION_UPSTREAM_GCHP_NOTES.md`) with any discoveries. Save
final versions to `artifacts/plan18/upstream_fortran_notes.md` and
`.../upstream_gchp_notes.md`.

Per `CLAUDE_additions_v2.md` § "Validation discipline", this is not
optional. Physics ports require upstream validation to avoid
inheriting legacy bugs silently.

**Pre-verified findings from initial spot-check (Commit 0 re-verifies):**

1. Legacy `tm5_matrix_convection.jl:60-122` appears faithful to
   `tm5_conv.F90:37-191` after accounting for 1-based index shift.
   Including the Sander Houweling subsidence correction at Fortran
   line 147. Matrix-builder math is correct; light cleanup OK.
2. Legacy `ras_convection.jl:188-193` uses a two-term tendency.
   GCHP `convection_mod.F90:988-1007` uses four terms. Careful
   algebraic expansion (`18_CONVECTION_UPSTREAM_GCHP_NOTES.md` §5.3)
   proves equivalence for inert tracers via the updraft-balance
   substitution `CMFMC_BELOW · old_QC = CMOUT · new_QC - ENTRN · Q(K)`.
   Plan 18 adopts the two-term form.
3. Legacy is MISSING the well-mixed sub-cloud layer from GCHP
   `:742-782`. Plan 18 ADDS it (§2.7).

## 2.13 `lmax_conv` handling for TM5

**Decision 12.** `TM5Convection.lmax_conv::Int` preserves legacy
`0` sentinel ("use full Nz"). Constructor:

```julia
function TM5Convection(; lmax_conv::Int = 0)
    lmax_conv >= 0 || throw(ArgumentError("lmax_conv must be >= 0"))
    TM5Convection(lmax_conv)
end
```

Actual convection-top level is returned by the matrix builder as `lmc`
(per Fortran `tm5_conv.F90:170-183`); the solver runs on levels
`1..lmc` only. `lmax_conv` is the user-specified upper bound.

## 2.14 `apply!` signature matches plans 15 / 16b / 17

**Decision 3.**

```julia
function apply!(
    state::CellState{B},
    meteo::AbstractTransportWindow{B},
    grid::AbstractGrid,
    op::AbstractConvectionOperator,
    dt::Real;
    workspace,
) where {B <: AbstractMassBasis}
    # ...
end
```

`meteo` required because convection consumes
`met.convection::ConvectionForcing`. Unlike chemistry and diffusion,
`meteo = nothing` is NOT acceptable — without forcing there's nothing
to do, and `NoConvection` is the correct way to express
"no convection in this model."

`workspace` required (no lazy allocation). `CMFMCWorkspace` or
`TM5Workspace` as appropriate. Matches plan 16b diffusion contract.

**Runtime dispatch:**
- `op::NoConvection` → `apply!(state, meteo, grid, ::NoConvection, dt; workspace=nothing) = state`, pure identity, no kernel launch. Default path is bit-exact with pre-plan-18 behavior when `model.convection = NoConvection()`.
- `op::CMFMCConvection` → sub-cycling loop calling `_cmfmc_kernel!`.
- `op::TM5Convection` → matrix build + LU solve per column.

## 2.15 Array-level entry point

**Decision 4.** Mirror the plan 16b / 17 pattern so the convection
block in `TransportModel.step!` can be called on
`state.tracers_raw` directly:

```julia
apply_convection!(q_raw::AbstractArray{FT, 4}, ::NoConvection, ws, dt, meteo, grid) = nothing

function apply_convection!(q_raw::AbstractArray{FT, 4},
                            op::CMFMCConvection, ws, dt,
                            meteo::AbstractTransportWindow, grid) where FT
    # ...
end

function apply_convection!(q_raw::AbstractArray{FT, 4},
                            op::TM5Convection, ws, dt,
                            meteo::AbstractTransportWindow, grid) where FT
    # ...
end
```

State-level `apply!` delegates to `apply_convection!(state.tracers_raw,
...)`. Used by Commit 6's `step!` orchestration. Not inserted into the
palindrome — plan 18 ships convection as a separate block (§2.2).

---

# Part 3 — Commit sequence

Draft, not contract (CLAUDE.md plan execution rhythm). NOTES.md's
"Deviations from plan doc §3" section is mandatory — record every
commit merge, split, or skip.

### Commit 0 — NOTES + baseline + upstream survey

Standard pattern. No source changes.

1. Create `docs/plans/18_ConvectionPlan/notes.md` with baseline
   section (parent commit, branch name, pre-existing failure count).
2. Run precondition verification (§4.1 of legacy plan): grep survey
   for existing infrastructure (`convection`, `cmfmc`, `tiedtke`,
   `ras`, `entu`, `detu`), test-suite sanity, verify A1/A2/A3 fixes
   merged.
3. Capture baseline test summary to
   `artifacts/plan18/baseline_test_summary.log`.
4. Record baseline git hash to `artifacts/plan18/baseline_commit.txt`.
5. Read and extend upstream Fortran references per §2.12. Save final
   notes to `artifacts/plan18/upstream_fortran_notes.md` and
   `.../upstream_gchp_notes.md`.
6. **Re-verify interface claims** against current `src/` tree. If
   any claim in §1.3 has changed, update v4 before proceeding.
7. Register follow-up plan candidates (see §6).
8. Memory compaction: update `MEMORY.md` "Current State" with plan
   17 completion note (already shipped), add `plan18_start.md` memo.

```bash
git commit -m "Commit 0: NOTES + baseline + upstream Fortran survey for plan 18"
```

### Commit 1 — `AbstractConvectionOperator` + `NoConvection`

Minimal: type hierarchy and no-op operator. No kernels yet.

**Files added:**

- `src/Operators/Convection/Convection.jl` — module file
- `src/Operators/Convection/operators.jl` — stub: defines
  `AbstractConvectionOperator`, `NoConvection`, `apply!` / `apply_convection!`
  no-op methods
- `test/test_convection_types.jl`

**Files modified:**

- `src/AtmosTransport.jl` — include + export
- `src/Operators/Operators.jl` — include order: Diffusion → SurfaceFlux →
  Convection → Advection → Chemistry (Convection has no cross-deps)

**Tests in `test/test_convection_types.jl`** (~6 tests):

1. Type hierarchy: `NoConvection() isa AbstractConvectionOperator`.
2. State-level identity: `apply!(state, meteo, grid, NoConvection(), dt; workspace=nothing)` returns state bit-exact (`==`, not `≈`).
3. Array-level identity: `apply_convection!(q_raw, NoConvection(), ws, dt, meteo, grid) === nothing`.
4. Dispatch correctness: no kernel launched for `NoConvection`.
5. Exported from `AtmosTransport` (symbols visible).
6. Works on both `CellState{DryBasis}` and `CellState{MoistBasis}`.

```bash
git commit -m "Commit 1: AbstractConvectionOperator hierarchy + NoConvection"
```

### Commit 2 — Extend transport windows for `ConvectionForcing`

**REPLACES base-plan Commit 2 (which shipped `PreComputedConvMassFluxField`).**

This commit extends the existing window infrastructure to carry
optional convection forcing as plain arrays. No `TimeVaryingField`
wrapper.

**Files added:**

- `src/MetDrivers/ConvectionForcing.jl` — the struct (§2.4)

**Files modified:**

- `src/MetDrivers/TransportBinaryDriver.jl`:
  - Add `C` type parameter to `StructuredTransportWindow` and
    `FaceIndexedTransportWindow` structs (line 33 and 42).
  - Add `convection :: C` field, default `nothing` in outer
    constructors.
  - Extend `Adapt.adapt_structure` methods (line 66, 77) to thread
    the `convection` field.
- `src/MetDrivers/MetDrivers.jl` — include + export `ConvectionForcing`.
- `src/AtmosTransport.jl` — export `ConvectionForcing`.

**Back-compat:**

- All existing window constructors keep their current signatures and
  default `convection = nothing`.
- Runs without convection are bit-exact vs pre-Commit-2.
- `test_transport_binary_reader.jl` passes unchanged.
- `has_convection_forcing(window) = window.convection !== nothing`
  helper added for convenience.

**Tests** (`test/test_convection_forcing.jl`, ~10 tests):

1. Default construction: `window.convection === nothing`.
2. With `ConvectionForcing(cmfmc=..., dtrain=..., tm5_fields=nothing)`:
   `window.convection.cmfmc` correctly shaped `(Nx, Ny, Nz+1)`.
3. With `ConvectionForcing(cmfmc=nothing, dtrain=nothing, tm5_fields=(entu=..., detu=..., entd=..., detd=...))`: TM5 fields correctly shaped.
4. `has_convection_forcing` trait.
5. `Adapt.adapt_structure` round-trip preserves `convection`.
6. CPU / CUDA round-trip with Array → CuArray on `cmfmc` field.
7. Both structured and face-indexed windows support the new field.
8. Mixed: some fields populated, others `nothing`.
9. Back-compat: `test_transport_binary_reader.jl` unchanged.
10. Type parameter `C` is `Nothing` when no convection; concrete
    `ConvectionForcing{CM, DT, TM}` when present.

```bash
git commit -m "Commit 2: Extend transport windows with optional ConvectionForcing"
```

### Commit 3 — `CMFMCConvection` port (GCHP path)

Substantial commit. Port from upstream `convection_mod.F90`
`DO_RAS_CLOUD_CONVECTION` with medium cleanup plus:

- Well-mixed sub-cloud layer addition (§2.7)
- Mandatory CFL sub-cycling (§2.8)
- Inline helpers (§2.6)
- Tier A / B / C validation (§2.5)
- Tier A adjoint-identity test (§2.9)
- NO positivity clamp (§2.9)

**Files added:**

- `src/Operators/Convection/CMFMCConvection.jl` — struct (one-liner,
  no fields)
- `src/Operators/Convection/cmfmc_kernels.jl` — KA kernel + inline
  helpers
- `src/Operators/Convection/convection_workspace.jl` — `CMFMCWorkspace`
  (Decision 21)
- `test/test_cmfmc_convection.jl`

**Files modified:**

- `src/Operators/Convection/Convection.jl` — include new files,
  re-export

**Kernel signature (exposed, not buried):**

```julia
function apply_convection!(q_raw::AbstractArray{FT, 4},
                            op::CMFMCConvection,
                            ws::CMFMCWorkspace,
                            dt::Real,
                            meteo::AbstractTransportWindow,
                            grid) where FT
    met_conv = meteo.convection
    met_conv === nothing && throw(ArgumentError("CMFMCConvection requires ConvectionForcing in window"))
    met_conv.cmfmc === nothing && throw(ArgumentError("CMFMCConvection requires cmfmc field"))

    # Compute / reuse sub-cycling
    n_sub = _get_or_compute_n_sub!(ws, meteo, state_air_mass, dt)
    sdt   = dt / n_sub

    # Sub-step loop
    for _ in 1:n_sub
        _cmfmc_kernel!(q_raw, <state.air_mass shape>,
                       met_conv.cmfmc, met_conv.dtrain,
                       ws.qc_scratch, FT(sdt), grid)
    end
end
```

The sub-cycling loop is visible in the signature; n_sub and its
cache appear in the workspace.

**Kernel structure (two-pass per column, per GCHP and legacy):**

- Pass 1 (bottom-to-top): updraft concentration `qc`.
- **New:** Well-mixed sub-cloud layer (§2.7) between cloud-base
  detection and Pass 1.
- Pass 2 (top-to-bottom): environment tendency using `_cmfmc_apply_tendency`.
- No positivity clamp.

Column reversal: codebase convention `k=1=TOA, k=Nz=surface`. Kernel
iterates `k = Nz:-1:cldbase` for Pass 1 and `k = cldbase:Nz` for
Pass 2 (surface → top for updraft, top → surface for tendency, same
physical sense as GCHP's `k=1=surface` loops).

**DTRAIN-missing fallback (legacy behavior preserved):**

When `met_conv.dtrain === nothing`, the kernel runs Tiedtke-style
single-flux transport: `entrn = 0`, `cmout = cmfmc_below`,
tendency reduces to `cmfmc[k] · (q_above - q_env)` only. No
separate operator type — runtime path within `CMFMCConvection`.

**Cleanup item:** remove the stale legacy comment at
`src_legacy/Convection/ras_convection.jl:41-46` during the port.
`CMFMCConvection` docstring states basis follows state.

**Tier A tests (~5 tests)**:

1. Mass conservation (column sum preserved to machine precision, no
   positivity clamp to break linearity).
2. Zero forcing identity (`cmfmc = dtrain = 0` → no change, `==`
   bit-exact).
3. Positivity under physical forcing (non-negative q in with
   consistent met data → non-negative q out).
4. Sub-cycling bit-exactness: single `apply!` at dt matches
   `n_sub` manual calls of `sdt = dt/n_sub` to machine precision
   (Decision 21 regression).
5. **Adjoint identity** (§2.9 Addition A): build dense `L` for
   Nz=8 column by applying kernel to unit vectors; verify
   `⟨y, L·x⟩ ≈ ⟨L^T·y, x⟩` for random x, y to rtol=1e-10.

**Tier B tests (~7-10 tests)**:

- Updraft formula: hand-expand `qc = (cmfmc_below · qc_below + entrn · q_env) / cmout` for a 5-layer column (per GCHP `convection_mod.F90:917-920`).
- Two-term tendency equivalence to four-term: hand-expand both forms,
  verify port's two-term matches four-term after updraft-balance
  substitution (GCHP notes §5.3).
- Well-mixed sub-cloud layer: hand-expand for 3-layer sub-cloud case
  (cldbase=3, q=[1,2,3]×mass), verify port reproduces GCHP's QB
  weighting and uniform application.
- Mass balance under arbitrary cmfmc / dtrain profile.
- DTRAIN-missing fallback matches Tiedtke-style path on a hand-built
  case.
- Cloud-base detection: lowest level with `cmfmc > tiny`.
- Column reversal: compare explicitly to a reference in TM5 /
  AtmosTransport convention.

**Tier C tests (~2-3 tests, opt-in)**:

- Standard deep-tropical-convective column (CMFMC peaking at
  500 hPa, DTRAIN detraining at 200 hPa, surface-source tracer).
- Compare to GEOS-Chem Fortran reference calculation or published
  Rn-222 profile from a GEOS-Chem paper.
- Agreement ~10% relative on column-integrated mass, similar
  qualitative shape.
- Port WITH sub-cloud fix should agree BETTER with GCHP reference
  than port WITHOUT (documents the improvement).
- Gated via `ENV["ATMOSTR_TIER_C_REFS"]` pointing to reference data;
  skipped at CI by default.

**`CMFMCConvection` docstring includes:**

```
"""
    CMFMCConvection()

GEOS-Chem RAS / Grell-Freitas convective transport.

Basis follows the state's type parameter (DryBasis or MoistBasis).
Driver is responsible for basis-appropriate CMFMC and DTRAIN upstream.

# Fields required on the transport window

- `window.convection.cmfmc :: AbstractArray{FT, 3}` at interfaces,
  shape (Nx, Ny, Nz+1). Units kg / m² / s on basis matching state.
- `window.convection.dtrain :: Union{AbstractArray{FT, 3}, Nothing}`
  at centers, shape (Nx, Ny, Nz). When `nothing`, reduces to
  Tiedtke-style single-flux transport (DTRAIN-missing fallback).

# CFL sub-cycling

The kernel sub-cycles internally based on the CMFMC profile:
`n_sub = max(1, ceil(max(cmfmc × dt / air_mass) / 0.5))`. Cached per
window in the workspace; driver invalidates on window roll.

# Well-mixed sub-cloud layer

Applies GCHP's pressure-weighted well-mixed treatment below cloud
base (`convection_mod.F90:742-782`). Absent in legacy Julia;
deliberate improvement for surface-source tracers.

# Adjoint path (not shipped in plan 18)

The forward operator is linear in tracer mixing ratio (verified by
Tier A adjoint-identity test). No positivity clamp. A future
`Plan 19: Adjoint operator suite` kernel will:

1. Reverse the two-pass loop order (tendency → updraft).
2. Apply transposed coefficients: `adj_q_cloud` accumulated
   top-down, `adj_q_env` tendency bottom-up.
3. Operate in the reversed step sequence of the adjoint time
   integration.

For INERT tracers (plan 18 scope), the two-term tendency
`cmfmc · (q_above - q_env) + dtrain · (qc - q_env)` is linear in q.
Wet-deposition plan introduces the four-term form whose
`q_post_mix = (CMFMC_BELOW · qc + ENTRN · q) / CMOUT` division
would require forward-state storage for adjoint — see
GCHP_NOTES §5.3 for the algebra and Plan 19 for the discrete-adjoint
approach.
"""
```

```bash
git commit -m "Commit 3: CMFMCConvection — GCHP path + sub-cloud fix + CFL sub-cycling + Tier A/B/C"
```

### Commit 4 — `TM5Convection` port (TM5 path)

Substantial commit. Light cleanup (preserving matrix-algorithm
invariants).

**Files added:**

- `src/Operators/Convection/TM5Convection.jl` — struct with
  `lmax_conv` field
- `src/Operators/Convection/tm5_matrix_kernels.jl` — two kernels:
  build + factorize, solve
- `src/Operators/Convection/convection_workspace.jl` (extended):
  `TM5Workspace{FT}` holding `conv1_ws :: Array{FT, 4}` shape
  `(lmax, lmax, Nx, Ny)` and `pivot_ws :: Array{Int, 3}` shape
  `(lmax, Nx, Ny)`
- `test/test_tm5_convection.jl`

**Files modified:**

- `src/Operators/Convection/Convection.jl` — include, re-export.

**Kernel split (Decision 7 from v3, preserved):**

- `_tm5_conv_build_factorize_kernel!(conv1_ws, entu, detu, entd, detd,
  air_mass, Nz, lmax, dt, grav)` — builds matrix, does Gaussian
  elimination. Matrix is diagonal-dominant; no pivoting needed.
- `_tm5_conv_solve_kernel!(q_raw, conv1_ws, Nz, Nt, lmax)` — forward /
  back substitution per tracer.

Build once, solve `Nt` times (tracer count). Fusion is a future
perf plan.

Column reversal at `apply_convection!` boundary (not inside the
kernel): map AtmosTransport `k=1=TOA` to TM5 `k=1=surface` before
building, and back after solving. Legacy pattern preserved.

**Tier A tests (~5)**: mass conservation (exact, implicit solve),
zero forcing, positivity, matrix diagonal dominance (structural),
**adjoint identity** — extract `conv1` from workspace, verify
`⟨y, conv1⁻¹x⟩ ≈ ⟨(conv1^T)⁻¹y, x⟩` using LU transpose solve
(no extra factorization).

**Tier B tests (~7)**: 5-layer column, specified entu / detu / entd / detd,
hand-build matrix per Tiedtke 1989 + `tm5_conv.F90:37-183`, hand-compute
LU factorization, hand-solve for specific RHS. Verify port's matrix
entries, LU, and solve each match hand expansion.

**Tier C tests (~2, opt-in)**: same deep-convective column as Commit 3,
entu / detu / entd / detd derived from CMFMC + DTRAIN (documented
relationship, §2.4 of GCHP notes). Compare to TM5 Fortran reference
or published TM5 output. Agreement ~10%.

**Docstring** (parallel structure to CMFMCConvection):

```
"""
    TM5Convection(; lmax_conv=0)

TM5-style four-field matrix convective transport.

Basis follows the state's type parameter. Driver responsible for
basis-appropriate entu / detu / entd / detd upstream.

# Fields required on the transport window

- `window.convection.tm5_fields :: NamedTuple{(:entu, :detu, :entd, :detd)}`,
  each `AbstractArray{FT, 3}` shape (Nx, Ny, Nz). Units kg / m² / s.

# lmax_conv

Maximum level for convection. 0 sentinel = use full Nz. The matrix
builder returns `lmc` (actual convection top) and the solver runs on
levels 1..lmc only.

# Adjoint path (not shipped in plan 18)

Forward operator `L = I - dt·D` is explicit in `conv1_ws` after the
matrix-build kernel. Adjoint operator is `L^T`; adjoint solve reuses
the same LU factorization with `trans='T'`, mirroring TM5's
`TM5_Conv_Apply` at `tm5_conv.F90:197-341`. A future
`Plan 19: Adjoint operator suite` kernel:

1. Runs forward: LU factorize `conv1_ws` once per column; solve for
   each tracer with `trans='N'`.
2. Runs adjoint: reuse the SAME LU factorization; solve for each
   tracer with `trans='T'`. No re-factorization.
3. Reverses the step ordering in the adjoint time integration
   (TM5's `adj_modelIntegration.F90:569-731`).

The Tier A adjoint-identity test in `test_tm5_convection.jl` verifies
the forward operator is transposable. This property is preserved
throughout plan 18.
"""
```

```bash
git commit -m "Commit 4: TM5Convection — matrix scheme with Tier A/B/C + adjoint identity"
```

### Commit 5 — Cross-scheme consistency

**Scope:** `test/test_convection_cross_scheme.jl`. Verify the two
operators agree to ~5% on column-integrated mass for matched forcing,
both run on the same basis as state.

- Specify idealized deep-convective CMFMC(k) profile (peak 500 hPa,
  zero above tropopause and at surface).
- Specify DTRAIN(k) profile (detraining at cloud top).
- Derive equivalent entu / detu / entd / detd via documented mass-
  balance relationship (GCHP notes §11):
  ```
  entu(k) = max(0, cmfmc(k+1) - cmfmc(k) + dtrain(k))
  detu(k) = dtrain(k)
  entd(k) = 0    # no downdraft in GCHP path
  detd(k) = 0
  ```
- **Both operators run on the same basis as state** (dry in CATRINE
  use case). `ConvectionForcing` carries whichever basis matches.
  No mixed-basis comparison.
- Tolerance: **~5% relative on column-integrated mass.** Rationale:
  explicit sub-step (~2-3%) + well-mixed sub-cloud (~1-2%) = ~5%.
- Vertical profile: similar qualitative shape, maximum mid-
  troposphere, surface mass lofted to cloud-top detrainment layer.

**Shared test-forcing harness** (lives in `test/shared/convection_forcing_harness.jl`
or similar): generates the deep-convective CMFMC / DTRAIN profile
and derived entu / detu / entd / detd for reuse across Commits 3,
4, 5, and 9.

Expected test count: ~8 tests.

```bash
git commit -m "Commit 5: Cross-scheme consistency test (~5% same-basis)"
```

### Commit 6 — `TransportModel.convection` + `step!` orchestration

**Files modified:**

- `src/Models/TransportModel.jl`: add `convection::ConvT` field, default
  `NoConvection()`. Add `with_convection(model, op)` helper paralleling
  `with_diffusion`, `with_chemistry`, `with_emissions`.

```julia
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT, ChemT, DiffT, EmT, ConvT}
    state      :: StateT
    fluxes     :: FluxT
    grid       :: GridT
    advection  :: SchemeT
    workspace  :: WorkspaceT
    chemistry  :: ChemT
    diffusion  :: DiffT
    emissions  :: EmT
    convection :: ConvT   # NEW
end

with_convection(model::TransportModel, op::AbstractConvectionOperator) = ...
```

**`step!` orchestration:**

```julia
function step!(model::TransportModel, dt; meteo = nothing)
    # Transport block (plan 17 palindrome includes diffusion + emissions)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)

    # Convection block (new — plan 18)
    if !(model.convection isa NoConvection)
        apply_convection!(model.state.tracers_raw, model.convection,
                          model.workspace.convection,
                          dt, meteo, model.grid)
    end

    # Chemistry block (plan 15)
    chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end
```

The `if !(model.convection isa NoConvection)` branch is compile-time
removable when `model.convection` is concretely typed as `NoConvection`
— zero runtime overhead in default configuration.

**Workspace integration:**

- Extend `AdvectionWorkspace` (or parallel field in `TransportModel.workspace`)
  to include `workspace.convection :: Union{Nothing, CMFMCWorkspace, TM5Workspace}`.
- Default `nothing` when `model.convection isa NoConvection`.
- `with_convection` allocates the appropriate workspace.

**Tests (`test/test_transport_model_convection.jl`, ~10 tests)**:

1. Default `TransportModel` carries `NoConvection`.
2. `with_convection(model, CMFMCConvection())` returns new model
   with only convection replaced (other fields `===` identity).
3. `with_convection(model, TM5Convection(; lmax_conv=30))` similarly.
4. **Bit-exact regression:** default `step!` `==` pre-plan-18
   no-convection path (`==`, not `≈`). Critical test.
5. `step!` with `CMFMCConvection + ConvectionForcing` produces
   measurable vertical redistribution.
6. `step!` with `TM5Convection` similarly.
7. Plan 17's `test_transport_model_emissions.jl` passes unchanged
   (regression).
8. Plan 16b's `test_transport_model_diffusion.jl` passes unchanged.
9. Plan 15's `test_transport_model_chemistry.jl` passes unchanged.
10. `with_convection(NoConvection())` round-trip works.

```bash
git commit -m "Commit 6: TransportModel.convection + step! orchestration (block-level)"
```

### Commit 7 — Driver / window integration (EXPANDED)

**Significantly expanded vs base plan.** This commit wires the binary
reader's existing `load_cmfmc_window!` / `load_tm5conv_window!`
through the driver contract and into window population.

**Files added:**

- `src/MetDrivers/DryConvFluxBuilder.jl` (if dry-basis correction
  needed as standalone module) or extend `DryFluxBuilder.jl` with
  convection corrections.

**Files modified:**

- `src/MetDrivers/AbstractMetDriver.jl`: `supports_convection(::AbstractMetDriver) = false` already defined; add docstring clarifying the contract and a mode parameter for `:cmfmc` vs `:tm5`.
  ```julia
  """
      supports_convection(driver) -> Bool
      convection_mode(driver) -> Symbol  # :cmfmc, :tm5, or :none
  """
  supports_convection(::AbstractMetDriver) = false
  convection_mode(::AbstractMetDriver) = :none
  ```
- `src/MetDrivers/TransportBinaryDriver.jl`:
  - `supports_convection(d::TransportBinaryDriver) = has_cmfmc(d.reader) || has_tm5conv(d.reader)`
  - `convection_mode(d::TransportBinaryDriver) = has_cmfmc(d.reader) ? :cmfmc : (has_tm5conv(d.reader) ? :tm5 : :none)`
  - Extend `load_transport_window!` (or `load_met_window!` per current
    name) to populate `window.convection` when `supports_convection(d)`:
    ```julia
    if supports_convection(d)
        mode = convection_mode(d)
        cmfmc_arr = mode == :cmfmc ? load_cmfmc_window!(d.reader, win) : nothing
        dtrain_arr = mode == :cmfmc ? load_dtrain_window!(d.reader, win) : nothing
        tm5_nt = mode == :tm5 ? load_tm5conv_window!(d.reader, win) : nothing

        # Basis correction if driver configured dry
        if driver_basis(d) === DryBasis
            cmfmc_arr, dtrain_arr = apply_dry_conv_correction!(cmfmc_arr, dtrain_arr, qv_interface)
            # or analogous for tm5_nt
        end

        forcing = ConvectionForcing(cmfmc_arr, dtrain_arr, tm5_nt)
        window = @set window.convection = forcing  # or reconstruct
    end
    ```
- `src/Models/DrivenSimulation.jl`:
  - On window roll, call `invalidate_cmfmc_cache!(model.workspace.convection)`
    if the workspace is a `CMFMCWorkspace`.

**Basis correction (Decision 20):**

When the driver is configured for dry basis but the binary stores moist
CMFMC / DTRAIN (typical case), apply:
```
cmfmc_dry(i, j, k_interface) = cmfmc_moist(i, j, k_interface) × (1 - qv_at_interface(i, j, k_interface))
dtrain_dry(i, j, k_center)   = dtrain_moist(i, j, k_center)   × (1 - qv_at_center(i, j, k_center))
```
and analogous for TM5 fields at full levels. qv interpolation follows the existing `DryFluxBuilder` pattern.

**Binary may already store dry fields:** in that case the header flag
indicates basis and the correction is skipped.

**Tests (`test/test_driver_convection_wiring.jl`, ~12 tests)**:

1. `supports_convection` / `convection_mode` for a CMFMC-only binary.
2. Same for a TM5conv-only binary.
3. Same for a binary with neither (baseline stays `false` / `:none`).
4. Window population from CMFMC binary: `window.convection.cmfmc` shape correct, `dtrain` shape correct, `tm5_fields === nothing`.
5. Window population from TM5 binary: `tm5_fields` NamedTuple with correct shapes, `cmfmc === nothing`.
6. Basis correction applied when dry driver + moist binary.
7. No correction applied when dry driver + dry binary (header flag).
8. Cache invalidation: first `apply!` in new window recomputes
   `n_sub`; subsequent `apply!` in the same window reuses cached
   value (observable via a counter or by asserting the cache-valid
   flag).
9. Cache invalidation on window roll: driver calls
   `invalidate_cmfmc_cache!`.
10. Back-compat: runs without convection capability get
    `window.convection === nothing` and operator defaults to `NoConvection`.
11. Adapt round-trip for a window with `ConvectionForcing`.
12. End-to-end: driver construction → window load → `apply!` with
    `CMFMCConvection`.

```bash
git commit -m "Commit 7: Driver/window integration for convection forcing"
```

### Commit 8 — DrivenSimulation integration

Minimal since plan 17 Commit 6 already cleaned up sim-level
orchestration. Plan 18's work here:

- Ensure `DrivenSimulation` constructor accepts a convection operator
  (via kwarg or pre-constructed `TransportModel`).
- Verify `DrivenSimulation.step!` delegates entirely to
  `step!(model)`; convection rides through automatically.
- Hook `invalidate_cmfmc_cache!` into the window-roll path so the
  CFL cache is refreshed per window.
- Add a test that sim-level runs with convection produce the
  expected trajectory.

**Files modified:**

- `src/Models/DrivenSimulation.jl`: add convection to constructor
  kwargs; wire cache invalidation on window roll (already extended
  in Commit 7's driver path; Commit 8 ensures the DrivenSimulation
  call site invokes it).
- `src/AtmosTransport.jl`: exports already cover operator types from
  Commits 1 / 3 / 4.

**Tests (`test/test_driven_simulation_convection.jl`, ~8 tests)**:

1. Construction with `convection = NoConvection()` (default), no driver
   convection capability → unchanged behavior vs plan 17.
2. Construction with `convection = CMFMCConvection()`, binary has
   CMFMC capability → full path runs.
3. Construction with `convection = TM5Convection()`, binary has TM5
   capability → full path runs.
4. Error: `convection = CMFMCConvection()` with a driver that doesn't
   support convection → clear error message.
5. Error: `convection = TM5Convection()` with a CMFMC-only binary →
   clear error message.
6. End-to-end 10-step run preserves total mass to machine precision.
7. Cache invalidation occurs on window roll (counter test).
8. Existing `test_driven_simulation.jl` 57-test suite passes unchanged.

```bash
git commit -m "Commit 8: DrivenSimulation integration for convection"
```

### Commit 9 — Convection block position study

Analogous to plan 17's ordering study, but for block position rather
than palindrome position.

Four configurations compared on a standardized 24-hour tropical
convective scenario (idealized CMFMC / DTRAIN profile, surface
emissions, decay chemistry):

| Label | Block order | Rationale |
|---|---|---|
| 1 (recommended — shipped) | transport → convection → chemistry | §2.2 Decision 1 |
| 2 | transport → chemistry → convection | chemistry first |
| 3 | convection → transport → chemistry | convection first |
| 4 (palindrome-internal) | X Y Z V(dt/4) C(dt/2) V(dt/4) S V(dt/4) C(dt/2) V(dt/4) Z Y X | palindrome internal, study-only |

Config 4 requires modified palindrome plumbing and is "study only;
not shipped."

**Tracked metrics:**

- Column-integrated mass distribution per level at t=0, 6h, 12h, 24h.
- Surface fraction over time (plan 17 pattern for pathological-case
  detection).
- Vertical profile snapshots at selected grid cells.

**Expected findings (hypothesis):**

- Configs 1, 2, 3 differ by ~1-3% relative (small).
- Config 4 may differ more (palindrome mixing with convection).
- No pathological "Config D-style pileup" expected (unlike plan 17
  emissions where D had catastrophic layer-1 accumulation).

**Writeup** at `docs/plans/18_ConvectionPlan/position_study_results.md`
with plots and recommendation.

Follow-up plan candidate: if Config 4 clearly wins, register "move
convection inside palindrome" as a future plan. Plan 18 ships as
planned regardless.

Expected test count: ~10 tests (configuration validation, not new
physics).

```bash
git commit -m "Commit 9: Convection block position study"
```

### Commit 10 — Benchmarks

`scripts/benchmarks/bench_convection_overhead.jl`:

- Configurations: `NoConvection` (baseline), `CMFMCConvection` with
  representative `ConvectionForcing`, `TM5Convection` similarly.
- Sizes: small CPU, medium CPU, medium GPU, large GPU (L40S F32
  per plan 16b convention).
- Metrics: median step time, overhead vs baseline, CFL sub-cycling
  cost breakdown for CMFMC.

**Expected results:**

- `CMFMCConvection`: moderate overhead (~10-30% base + sub-cycling
  factor of ~5-15 sub-steps at peak). Kernel is column-serial but
  arithmetic-light; sub-step count dominates cost.
- `TM5Convection`: higher overhead (~30-70%). Matrix build + LU
  solve is the heaviest convective operator. Roughly constant cost
  per dynamic dt (no sub-cycling).

**Soft targets** (no hard perf constraint in plan 18):
- `CMFMCConvection` overhead < 50% on GPU at medium grid.
- `TM5Convection` overhead < 100% on GPU at medium grid.

Document performance characteristics in
`artifacts/plan18/perf/SUMMARY.md`. Per
`CLAUDE_additions_v2.md` § "Measurement is the decision gate,"
re-frame the value proposition around measured numbers rather than
defending the pre-registered prediction.

```bash
git commit -m "Commit 10: Convection overhead benchmarks"
```

### Commit 11 — Retrospective + ARCHITECTURAL_SKETCH_v4 + validation_report

- Fill in `notes.md` retrospective sections: decisions beyond the
  plan, surprises, interface validation findings, template usefulness
  for plan 19+.
- Update `ARCHITECTURAL_SKETCH.md` to v4:
  - Convection operator hierarchy added.
  - Step-level composition documented (transport block → convection
    block → chemistry block).
  - Window extension (`ConvectionForcing` field) documented.
  - File-level map updated.
- Update `OPERATOR_COMPOSITION.md` with block-vs-palindrome decision
  and reference to position study results.
- Write `docs/plans/18_ConvectionPlan/validation_report.md`
  summarizing three-tier test coverage, adjoint-identity results,
  legacy bugs found and fixed.
- Document any scope deviations (§7 mandatory: split / merged / reshuffled
  commits).
- Move superseded v3 base plan, corrections addendum v2, and adjoint
  addendum to `docs/plans/18_ConvectionPlan/archived/` with a brief
  `README.md` explaining v4 supersedes them. Preserves provenance
  without confusing future agents.

```bash
git commit -m "Commit 11: Retrospective + ARCHITECTURAL_SKETCH_v4 + validation_report"
```

---

# Part 4 — Acceptance criteria

## 4.1 Correctness (hard)

- All pre-existing tests pass; baseline 77-failure count unchanged
  across all 12 commits.
- Plans 11-17 regression tests pass bit-exact (plans 16b emissions
  palindrome, 17 emissions ordering, 15 chemistry).
- Mass conservation in both operators to machine precision
  (F64 / ULP F32).
- Tier A + Tier B tests all pass (required).
- Tier C tests pass when reference data available
  (`ENV["ATMOSTR_TIER_C_REFS"]` set); documented as skipped
  otherwise.
- CFL sub-cycling acceptance (§2.8): `apply!(dt)` matches `n_sub`
  manual `apply!(sdt = dt/n_sub)` calls to machine precision.
- No positivity clamp introduced inside either kernel (grep for
  `max(..., 0)`, `if q < 0`, etc. in `cmfmc_kernels.jl` and
  `tm5_matrix_kernels.jl` returns no matches).

## 4.2 Cross-scheme consistency (hard)

- `CMFMCConvection` and `TM5Convection` agree to ~5% on column-
  integrated mass for matched forcing on the same basis
  (Commit 5 test).
- Divergences > 5% investigated and documented before Commit 6
  merges — either as legacy bugs fixed or as scheme-specific
  physics differences.

## 4.3 Code cleanliness (hard)

- `src/Operators/Convection/` directory exists with the files
  listed in Commits 1-4.
- `ConvectionForcing` struct in `src/MetDrivers/`.
- No `AbstractTracerSolubility` or scavenging code anywhere in
  plan 18.
- Inline helper functions (§2.6) with inert-only signatures.
  Each helper docstring contains "FUTURE: scavenging overload adds
  solubility parameter."
- Legacy `src_legacy/Convection/ras_convection.jl:41-46` stale
  comment is removed or corrected during port.
- Operator docstrings state basis follows state (§2.3).
- Convection block placed after transport, before chemistry, in
  `step!` (Commit 6).

## 4.4 Interface validation (hard)

- `apply!(state, meteo, grid, op, dt; workspace)` signature works
  for plan 18 operators (plans 15 / 16b / 17 pattern preserved).
- `AbstractTransportWindow{B}` extension with `ConvectionForcing`
  works for both `StructuredTransportWindow` and
  `FaceIndexedTransportWindow`.
- Array-level `apply_convection!` entry point mirrors plan 16b / 17.
- `supports_convection` / `convection_mode` driver contract works
  for `TransportBinaryDriver` and falls back cleanly for drivers
  that don't implement it.

## 4.5 Adjoint structure preservation (hard)

- Both operators pass the Tier A adjoint-identity test
  (Commit 3 Tier A, Commit 4 Tier A).
- No positivity clamp inside either kernel.
- Coefficients (matrix entries for TM5, mass-flux terms for CMFMC)
  stored in named-locals or explicit-array form amenable to
  transposition.
- Docstrings document the adjoint path with the "# Adjoint path
  (not shipped in plan 18)" section.

## 4.6 Validation discipline (hard)

- Upstream Fortran references read and annotated in
  `artifacts/plan18/upstream_fortran_notes.md` and `.../upstream_gchp_notes.md`.
- Three-tier validation structure (Tier A / B / C) in both
  `test_cmfmc_convection.jl` and `test_tm5_convection.jl`.
- Any legacy bugs found documented with source-of-truth hierarchy
  (paper > Fortran > legacy Julia) applied.

## 4.7 Performance (soft)

- `CMFMCConvection` overhead < 50% on GPU at medium grid (soft).
- `TM5Convection` overhead < 100% on GPU at medium grid (soft).
- No regression in non-convection paths (bit-exact default
  regression passes — Commit 6 test 4).

## 4.8 Documentation (hard)

- `ARCHITECTURAL_SKETCH_v4.md` committed.
- `position_study_results.md` with plots and recommendation.
- `validation_report.md` summarizing tier coverage.
- `notes.md` complete with legacy-bug section (if any bugs found),
  decisions beyond plan, surprises, interface findings.

---

# Part 5 — Known pitfalls

From accumulated plan-surprise patterns (CLAUDE.md + plans 11-17
retrospectives) plus plan-18-specific traps.

1. **Porting TM5 matrix scheme "faithfully without reading the Fortran."**
   Legacy `tm5_matrix_convection.jl` may have bugs. Upstream
   `tm5_conv.F90` is authoritative. Per §2.12, Commit 0 reads and
   extends the upstream notes.

2. **Porting legacy scavenging code as commented-out.** No. Per §2.11,
   no scavenging code at all. Signature-ready inline helpers, no
   comment blocks with old algorithm.

3. **Making the DTRAIN-missing fallback a separate operator type.**
   No. Runtime path within `CMFMCConvection` via `met_conv.dtrain === nothing`.

4. **Skipping Tier C "because it's hard."** No. Per §2.5, Tier C is
   required but opt-in via environment gate. If upstream Fortran
   isn't runnable, use published reference data. Cross-scheme
   consistency test (Commit 5) is ALSO a form of Tier C (two
   independent implementations).

5. **Treating legacy Julia as ground truth for Tier B tests.** No.
   Per §2.5, Tier B is against PAPER formulas, not legacy. Hand-expand
   from Tiedtke 1989 and Moorthi-Suarez 1992, not legacy code.

6. **Palindrome-internal convection because that's where plan 17 put
   emissions.** No, per §2.2 Decision 1. Plan 18 ships as separate
   block. Commit 9 study may suggest a follow-up plan; plan 18 does
   not mid-flight reorganize.

7. **Dynamic allocation in `apply!` for the matrix workspace.** No.
   Per §2.8, workspaces pre-allocated at `TransportModel` construction.

8. **Adding `InertTracer` as a vestigial type parameter.** No. No
   solubility parameters at all. Kernel applies mass-flux
   redistribution without trait queries.

9. **Shipping adjoint kernels "since they're in legacy."** No. Plan 18
   preserves adjoint-able structure and ships the adjoint-identity
   test. Adjoint kernel itself is `Plan 19: Adjoint operator suite`.

10. **Loosening cross-scheme tolerance to 10% because "5% is strict."**
    No. With both operators run on same basis (§2.3), 5% is the
    correct tolerance. 10% was pre-v4 legacy language from mixed-basis
    comparisons.

11. **Skipping the well-mixed sub-cloud layer because legacy Julia
    skipped it.** No. Per §2.7, plan 18 ADDS it. Commit 3 Tier C
    verifies the improvement.

12. **"Four-term tendency is right, I'll restore it."** No. Per
    §2.12 finding 2, for inert tracers the two-term form is
    algebraically equivalent. Four-term becomes necessary when
    scavenging is added — future plan.

13. **Porting the positivity clamp from GCHP's `convection_mod.F90:1002-1004`.**
    **CRITICAL: NO.** Per §2.9. The clamp breaks linearity and the
    adjoint-identity test. Accept tiny negativities; rely on global
    mass fixer. Optional pre-step met-data validation (preprocessing,
    not kernel) if negativities become observable.

14. **Benchmark overhead "too high; optimize before shipping."** No.
    Plan 18 is correctness-first. TM5Convection at 70% overhead is
    expected (matrix solve is expensive). File follow-up perf plan
    for cuBLAS-batched or shared-memory BLAS.

15. **GEOS-FP June 2020 RAS→GF data discontinuity "should be handled
    in the operator."** No. It's a DATA problem. Operator consumes
    whatever CMFMC + DTRAIN the driver provides. Documented in user
    docs.

16. **Wrapping `cmfmc` / `dtrain` in `TimeVaryingField` "to match the
    Kz / emissions pattern."** No, per §2.4 Decision 22. Window
    carries them as plain arrays. TimeVaryingField would duplicate
    the window contract and depend on `current_time(meteo)` (A3
    concern).

17. **Adding CFL sub-cycling "later if needed."** No, per §2.8
    Decision 21. Mandatory from Commit 3. Without it, production
    timesteps cause positivity failures and stability violations.

18. **Starting Commit 0 before PRE_PLAN_18_FIXES ship.** No. Commit 8
    (DrivenSimulation) needs A1's face-indexed apply! path working.
    A3's `current_time` fix prevents silent wrong-answer on any
    StepwiseField. Ship prerequisites first.

---

# Part 6 — Follow-up plan candidates

Register at Commit 0 in `notes.md`; retrospective (Commit 11)
closes or defers each.

- **Plan 19: Adjoint operator suite.** Backward-mode kernel ports
  for advection, diffusion, surface flux, convection, chemistry.
  Adjoint emission accumulator (`adj_em += adj_rm × dt` pattern
  from TM5 `emission_adj.F90:447`). Adjoint kernels reuse forward
  factorizations / coefficients. Observation injection as adjoint
  boundary conditions. Memory strategy for forward-trajectory
  storage (advection limiter is the hard case). Reference: TM5-4DVAR
  `adj_*.F90` files; GEOS-Chem Adjoint (GCAdj) as alternative.
  Estimated 3-4 weeks, parallel complexity to plan 18.

- **`DerivedConvMassFluxField`** — online computation of CMFMC / DTRAIN
  (or entu / detu / entd / detd) from parent T, q, p using the full
  Tiedtke 1989 or Grell-Freitas scheme. Would add the fields to
  `ConvectionForcing` via a `DerivedConvectionForcing` type or similar.

- **Wet deposition plan.** Re-introduce `QC_PRES` / `QC_SCAV` split
  in updraft pass. Restore four-term tendency form for soluble
  tracers (necessary when `q_post_mix ≠ old_QC`). Add solubility
  trait system (possibly `AbstractTracerSolubility`, `InertTracer`,
  `SolubleTracer`) and parameterize operators. Add precip flux
  fields to `ConvectionForcing`. Henry's-law hooks.

- **Multi-tracer fusion in TM5 matrix solve.** Build coefficients
  once per column, back-substitute Nt times. Currently solve per-tracer.

- **GPU-batched BLAS paths.** cuBLAS / rocBLAS `getrfBatched` +
  `getrsBatched` for TM5 matrix solve. Currently in-kernel Gaussian
  elimination on all backends.

- **Shared-memory LU factorization.** For Nz=72+ columns, shared-memory
  LU may outperform register-based in-kernel version on GPU.

- **`AbstractLayerOrdering{TopDown, BottomUp}` abstraction.** Deferred
  from plan 17. Would clean up the per-kernel column-reversal dance.

- **Palindrome-internal convection.** If Commit 9 study suggests
  Config 4 benefits, ship as follow-up plan. Requires palindrome
  modification.

- **TiedtkeConvection as standalone type.** If `CMFMCConvection`
  DTRAIN-missing fallback gets heavily used in production, promote
  to its own type.

- **Plan 16c: retroactive Tier B/C validation for diffusion.** Plan
  16b shipped Tier A + shallow Tier B; no Tier C vs GCHP `vdiff_mod`
  or TM5 `diffusion.F90`. Scope should include basis audit: do
  `ProfileKzField` / `PreComputedKzField` / `DerivedKzField` expect
  dry- or moist-basis input? Does `ImplicitVerticalDiffusion`'s
  column Thomas operate on dry or moist tracer? Tests sensitive to
  basis mismatch? Apply plan 18's basis-audit discipline
  retroactively.

- **Upstream-basis audit for existing transport code.** `src/`
  may have dry-correction paths inherited from legacy. Verify each
  matches GCHP convention. If absent where needed, add (referencing
  `src_legacy/Advection/latlon_dry_air.jl` and `cubed_sphere_mass_flux.jl`).

---

# Part 7 — How to work

## 7.1 Session cadence

- Session 1: Commits 0-1 (NOTES + baseline + upstream survey + type
  hierarchy)
- Session 2: Commit 2 (window extension for `ConvectionForcing`)
- Session 3: Commit 3 Part 1 (CMFMCConvection scaffolding + Tier A)
- Session 4: Commit 3 Part 2 (kernel + sub-cycling + sub-cloud + Tier B)
- Session 5: Commit 3 Part 3 (Tier C + adjoint identity + completion)
- Session 6: Commit 4 Part 1 (TM5Convection scaffolding + matrix workspace + Tier A)
- Session 7: Commit 4 Part 2 (matrix kernel + Tier B)
- Session 8: Commit 4 Part 3 (Tier C + adjoint identity + completion)
- Session 9: Commit 5 (cross-scheme consistency)
- Session 10: Commit 6 (TransportModel wiring)
- Session 11: Commit 7 (driver / window integration — EXPANDED, may
  span 2 sessions)
- Session 12: Commit 8 (DrivenSimulation)
- Session 13: Commit 9 (position study)
- Session 14: Commits 10-11 (benchmarks + retrospective)

Longer than plans 15-17 because of:
- Two substantive physics ports.
- Three-tier validation per port.
- Cross-scheme consistency.
- Expanded driver integration (Commit 7).
- Position study.

## 7.2 Validation per commit

After EACH commit:

```bash
julia --project=. -e 'using AtmosTransport'
julia --project=. test/runtests.jl
```

Baseline 77-failure count invariant. All prior plans' test suites
pass unchanged. If any new failure appears, STOP and revert.

Tier A + Tier B tests run in the normal suite (CI-friendly).
Tier C tests are opt-in:

```bash
ATMOSTR_TIER_C_REFS=/path/to/refs julia --project=. test/test_cmfmc_convection.jl
```

## 7.3 When to stop and ask

- Commit 0 upstream Fortran inaccessible. Per §2.12, this is not
  optional. If refs missing, STOP.
- Commit 3 Tier A mass conservation fails. Port has a bug.
- Commit 3 adjoint-identity test fails. Some nonlinearity crept
  in — typically a positivity clamp or a conditional branch.
- Commit 4 Tier A LU factorization fails. Matrix scheme port has
  a bug — compare matrix entries to hand-built reference.
- Commit 5 cross-scheme > 10% divergence. One operator has a bug.
- Commit 6 breaks bit-exact default-path regression. `NoConvection`
  dispatch isn't a compile-time dead branch. Check type stability
  at the `step!` call site.
- Commit 7 driver capability negotiation produces unclear errors
  when binary lacks CMFMC / DTRAIN. Clarify error message, don't
  paper over with a fallback.
- Legacy bug found during port. Document, stop, confirm with user
  whether port-follows-paper or port-preserves-bug.
- Scope creep toward scavenging, derived fields, or adjoint kernels.

## 7.4 NOTES.md discipline

Capture in `docs/plans/18_ConvectionPlan/notes.md` during execution
(not at the end):

- Upstream Fortran / legacy Julia discrepancies discovered in
  Commit 0 survey.
- Decisions on port-follows-paper vs port-preserves-legacy when
  discrepancies exist.
- Legacy bugs found and fix rationale.
- Deviations from Part 3 (commits split / merged / reshuffled).
- Performance observations.
- Position study outcomes (which config wins, by how much).
- Cross-scheme test outcomes (are schemes consistent, on what
  scales).
- Adjoint-identity test numerics (tolerance achieved, any surprises).

## 7.5 Artifacts directory

```
artifacts/plan18/
├── baseline_commit.txt
├── baseline_test_summary.log
├── existing_convection_survey.txt
├── upstream_fortran_notes.md
├── upstream_gchp_notes.md
├── validation/
│   ├── tier_a_cmfmc.log
│   ├── tier_a_tm5.log
│   ├── tier_b_cmfmc.log
│   ├── tier_b_tm5.log
│   ├── tier_c_cmfmc.log
│   ├── tier_c_tm5.log
│   ├── cross_scheme.log
│   └── adjoint_identity.log
├── perf/
│   └── SUMMARY.md
└── position_study/
    └── results.json
```

---

# Part 8 — Revisions note

- **v1** (initial): scoping conversation output; 10-commit sequence; 19 decisions.
- **v2**: post-GCHP-Fortran-read corrections; Findings 1-3 (two-term tendency equivalence, missing sub-cloud layer, stale "no dry conversion" comment).
- **v3**: added basis-convention analysis; Findings 3-5; inline helpers; decisions 18-19. Contained contradictions at Decisions 10 / 13 / 18 — resolved in v4.
- **v4** (this document): consolidation of v3 + corrections addendum v2 + adjoint addendum into single authoritative plan. Drops Decisions 10 / 13 / 18 (replaced by Decision 20: basis follows state). Drops `PreComputedConvMassFluxField` (replaced by Decision 22: `ConvectionForcing` on window). Adds Decision 21: mandatory CFL sub-cycling. Adds adjoint-identity Tier A tests (adjoint addendum §A). Adds no-positivity-clamp rule (adjoint addendum §D). Registers Plan 19 (adjoint suite) in follow-ups. Corrects file-layout convention (single `operators.jl` per submodule, matching Diffusion / SurfaceFlux; NOT the per-type filenames of base §4.2). Corrects directory path casing to `18_ConvectionPlan/`. Single ~5% cross-scheme tolerance, single rationale. All interface claims verified against current `src/` tree at v4 authoring time.

Source documents v3 base, corrections addendum v2, and adjoint addendum are moved to `docs/plans/18_ConvectionPlan/archived/` by Commit 11 with a `README.md` explaining that v4 supersedes them.

---

# End of Plan

After this refactor ships:

- `CMFMCConvection` in `src/Operators/Convection/` (GCHP path, dry- or moist-basis per state).
- `TM5Convection` in `src/Operators/Convection/` (TM5 path, dry- or moist-basis per state).
- Convection block positioned after transport, before chemistry in `step!`.
- Three-tier validation completed (Tier A required, Tier B required, Tier C opt-in).
- Cross-scheme consistency verified (~5% same-basis agreement).
- CATRINE intercomparison possible for convection-sensitive tracers (Rn-222, tropical CO2, SF6 over sources).
- Full offline atmospheric transport model with emissions, transport, convection, diffusion, chemistry — operational.
- Adjoint structure preserved throughout; `Plan 19: Adjoint operator suite` can proceed mechanically.

This is the final physics operator needed for CATRINE. After plan 18, the model has the full operator suite for offline atmospheric transport.
