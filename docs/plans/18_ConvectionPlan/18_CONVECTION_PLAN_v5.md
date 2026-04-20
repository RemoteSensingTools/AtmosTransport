# Plan 18 — Convection Operators (v5.1, authoritative)

**Status:** Ready for execution after prerequisites (A1, A2, A3 from
`PRE_PLAN_18_FIXES.md`, updated for v5) ship.

**v5.1 patch note.** This document is v5 with post-review corrections
for three architectural gaps flagged by GPT-5.4: model-side forcing
allocation (Decision 26), `dtrain` capability invariance (Decision 27),
and independent driver capability traits (Decision 28). Changes are
concentrated in §§2.20-2.22 (three new decisions) and Commits 2, 7,
8. All v5 physics decisions (basis rule, CFL sub-cycling, adjoint
structure, well-mixed sub-cloud, ~5% tolerance, three-tier validation,
Fortran references) carry forward unchanged. See Part 8 Revisions note
for the full change list.
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

This document consolidates and supersedes v1–v4. The v4 → v5
delta is concentrated on the **runtime data flow** from window
forcing through to operator execution: v4 assumed operators
receive the window at `apply!` time; reality is that the window
lives on `DrivenSimulation`, flux forcing is copied into
`TransportModel.fluxes` by `_refresh_forcing!` every substep, and
the operator only sees model-level forcing. v5 adds a parallel
`model.convection_forcing` field populated the same way and
changes the `apply!` signature to take `ConvectionForcing`
directly. All v4 physics decisions, validation discipline, and
Fortran references carry forward unchanged.

Precedence rules go away: this document is the single source of truth.

**Companion documents (still authoritative):**

- `PRE_PLAN_18_FIXES.md` — prerequisite current-tree fixes (A3
  rewritten for v5: `current_time` is a sim-level query).
- `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` — TM5 Fortran reference.
- `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` — GCHP Fortran reference
  (v4-corrected basis section).
- `CLAUDE_additions_v2.md` — accumulated lessons (validation
  discipline, basis-convention section).

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
struct. The struct lives in two places at runtime: on the
transport window (populated by the driver when loading a new met
window) and on `TransportModel` (refreshed every substep by
`DrivenSimulation._refresh_forcing!`, mirroring how
`model.fluxes` already works). Operators are called via an
`apply!` signature that takes `ConvectionForcing` directly — they
do not see the window or the driver.

Both integrate at step level as a dedicated convection block
between the transport block and the chemistry block. Validation
follows the three-tier hierarchy (analytic / paper / cross-
implementation) with upstream Fortran as ground truth, not legacy
Julia.

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
- **A3 (rewritten in v5):** `current_time` sim-level override.
  Current `AbstractMetDriver.current_time` is a `0.0` stub; plan 17
  threads it through operator `apply!` methods. The v4 PRE_PLAN_18_FIXES
  text proposed `current_time(::TransportBinaryDriver)` via a
  non-existent `d.current_window_index` field — the driver is stateless
  (struct has only `reader` + `grid`, verified at
  `src/MetDrivers/TransportBinaryDriver.jl:109-112`). v5 A3 instead
  adds `current_time(::DrivenSimulation) = sim.time` (sim already
  tracks this at `src/Models/DrivenSimulation.jl:26` and advances it
  per step at `:276`) and updates `DrivenSimulation.step!` to pass
  `meteo = sim` instead of `meteo = sim.driver`. Effort ~2 hours
  instead of 0.5-1 day.

Plan 18 assumes these have shipped. Starting plan 18 Commit 0 before
A1 ships breaks Commit 8 (DrivenSimulation integration) because
face-indexed runs can't execute the full operator suite. Starting
before A3 ships means the operator suite sees `t=0` throughout,
making any multi-window time-varying-forcing test silently wrong.

## 1.3 Current state of `src/` — verified interface claims

Before writing a line of kernel code, verify these claims still hold.
All are verified against current tree at v5 authoring time.

| Claim | Source |
|---|---|
| `CellState{Basis, A, Raw, Names}` with packed `tracers_raw`; structured `(Nx, Ny, Nz, Nt)`, face-indexed `(ncells, Nz, Nt)`; `air_mass` (not `air_mass_dry`); accessor API `get_tracer(state, :CO2)` / `state.tracers.CO2` | `src/State/CellState.jl:42-46` |
| `AbstractMassBasis`, `DryBasis`, `MoistBasis` tags on `CellState{B}` | `src/State/Basis.jl:10-24` |
| Horizontal-mesh supertype is `AbstractHorizontalMesh`; `AbstractStructuredMesh <: AbstractHorizontalMesh`; **no `AbstractFaceIndexedMesh` type exists** (face-indexed = any `AbstractHorizontalMesh` not `<: AbstractStructuredMesh`, e.g. `ReducedGaussianMesh`) | `src/Grids/AbstractMeshes.jl:33-43`; `src/Grids/ReducedGaussianMesh.jl:38` |
| Grid type name is **`AtmosGrid{H <: AbstractHorizontalMesh, ...}`** — not `AbstractGrid` (v4 used the wrong name) | `src/Grids/AbstractMeshes.jl:111` |
| `AbstractTransportWindow{Basis}` supertype; `StructuredTransportWindow{Basis, M, PS, F, Q, D}` and `FaceIndexedTransportWindow{Basis, M, PS, F, Q, D}` each have 6 type parameters | `src/MetDrivers/TransportBinaryDriver.jl:18, 33, 42` |
| Window carries `air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas` | `src/MetDrivers/TransportBinaryDriver.jl:33-49` |
| `Adapt.adapt_structure` methods for both window types at `:66` (structured) and `:77` (face-indexed) | as above |
| ERA5 reader exposes `has_cmfmc`, `has_tm5conv`, `load_cmfmc_window!`, `load_tm5conv_window!` | `src/MetDrivers/ERA5/BinaryReader.jl:173-175, 428-436, 469-483` |
| **`TransportBinaryReader` (generic; used by production CATRINE binaries) has NO convection loaders and `TransportBinaryHeader` has NO convection payload fields** — v5 Commit 7 adds them | `src/MetDrivers/TransportBinary.jl:35-77, 88` |
| **`TransportBinaryDriver` struct has ONLY `reader :: ReaderT` and `grid :: GridT`** — no `current_window_index`, no time, no clock. Stateless. | `src/MetDrivers/TransportBinaryDriver.jl:109-112` |
| `supports_convection(::AbstractMetDriver) = false` — no overrides yet | `src/MetDrivers/AbstractMetDriver.jl:95` |
| `current_time(::AbstractMetDriver) = 0.0` stub; A3 v5 adds `current_time(::DrivenSimulation) = sim.time` | `src/MetDrivers/AbstractMetDriver.jl:85` |
| `DrivenSimulation` has fields `model`, `driver`, `window`, `expected_air_mass`, `qv_buffer`, `Δt`, `window_dt`, `steps_per_window`, `time`, `iteration`, `current_window_index`, `stop_window`, `final_iteration`, `callbacks`, ... | `src/Models/DrivenSimulation.jl:17-38` |
| `DrivenSimulation._refresh_forcing!` copies `window.fluxes → model.fluxes` (+ `expected_air_mass`, `qv_buffer`); runs every substep | `src/Models/DrivenSimulation.jl:119-131` |
| `DrivenSimulation._maybe_advance_window!` loads new window when `substep == 1 && iteration > 0` | `src/Models/DrivenSimulation.jl:137-152` |
| `DrivenSimulation.step!` currently passes `meteo = sim.driver` at line 275 — v5 A3 changes to `meteo = sim` | `src/Models/DrivenSimulation.jl:262-282` |
| `sim.time` exists as mutable `FT` field, advanced by `sim.time += sim.Δt` at line 276 | `src/Models/DrivenSimulation.jl:26, 276` |
| `TransportModel` struct has `state, fluxes, grid, advection, workspace, chemistry, diffusion, emissions` (8 fields); v5 Commit 6 adds `convection` and `convection_forcing` | `src/Models/TransportModel.jl:28-37` |
| `TransportModel.step!` forwards `diffusion_op, emissions_op, meteo` to advection `apply!`, then calls `chemistry_block!` | `src/Models/TransportModel.jl:162-170` |
| Structured `apply!` palindrome `X Y Z [V(dt/2) S V(dt/2)] Z Y X` | `src/Operators/Advection/StrangSplitting.jl:1199-1246` |
| Submodule convention: single `operators.jl` + `*_kernels.jl` per submodule | `src/Operators/Diffusion/operators.jl`, `src/Operators/SurfaceFlux/operators.jl` |
| Diffusion workspace contract: required (`ArgumentError` when nothing); caller-owned scratch fields | `src/Operators/Diffusion/operators.jl:108-116, 152-185` |
| Legacy `ras_convection.jl:41-46` "no dry conversion" comment is stale (inconsistent with legacy's own dry-correction kernels) | `src_legacy/Convection/ras_convection.jl:41-46` |
| Baseline 77 pre-existing test failures across `test_basis_explicit_core.jl`, `test_structured_mesh_metadata.jl`, `test_poisson_balance.jl` | plan 12 lineage |

Commit 0 re-runs the verification: if any claim has changed since v5
was written, update the plan before proceeding.

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

- **Face-indexed (reduced-Gaussian) convection.** Plan 18 is
  structured-only. Face-indexed kernel work is a dedicated
  follow-up plan (Plan 18b). CATRINE production uses structured
  lat-lon; face-indexed is not on the critical path. A safety
  error-stub dispatched on state shape prevents accidental
  face-indexed use (§2.18 Decision 25).
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

v4 Decisions 1-22 are preserved with minor clarifications on where
runtime lifecycle sits. v5 adds Decisions 23, 24, 25 to pin the
runtime data flow.

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
(§2.3). Field access goes through the model-level
`ConvectionForcing` that was populated by `_refresh_forcing!`
(§2.17 Decision 23), not through struct fields — the operator is a
pure compute kernel keyed by type.

## 2.2 Composition — convection as separate block

**Decision 1.** Convection runs as a dedicated block after the
transport block and before the chemistry block, not inside the
palindrome.

```
step!(model, dt; meteo):
  transport_block  → X Y Z [V(dt/2) S V(dt/2)] Z Y X          # plan 17
  convection_block → if !(model.convection isa NoConvection)
                        apply!(state, model.convection_forcing, grid,
                               model.convection, dt;
                               workspace = model.workspace.convection_ws)
                     end
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
    forcing::ConvectionForcing,
    grid::AtmosGrid,
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

## 2.4 Convection forcing — window field AND model field

**Decision 22 (revised in v5, extending v4).** Convection forcing
lives at **two** runtime slots, mirroring how `fluxes` already works:

1. **On the transport window** (`window.convection :: Union{Nothing, ConvectionForcing}`),
   populated by the driver's `load_transport_window!` once per met
   window.
2. **On the model** (`model.convection_forcing :: ConvectionForcing`),
   refreshed every substep by `DrivenSimulation._refresh_forcing!`
   via `copy_convection_forcing!(model.convection_forcing, sim.window.convection)`.

Both slots carry the same struct type:

```julia
struct ConvectionForcing{CM, DT, TM}
    cmfmc      :: CM   # ::Union{Nothing, AbstractArray{FT, 3}}  (Nx, Ny, Nz+1)
    dtrain     :: DT   # ::Union{Nothing, AbstractArray{FT, 3}}  (Nx, Ny, Nz)
    tm5_fields :: TM   # ::Union{Nothing, NamedTuple{(:entu,:detu,:entd,:detd)}}  each (Nx, Ny, Nz)
end

ConvectionForcing() = ConvectionForcing(nothing, nothing, nothing)
```

Invariants (enforced at construction):
- `cmfmc` non-nothing (CMFMC mode, with `dtrain` optionally also
  non-nothing) OR `tm5_fields` non-nothing (TM5 mode) OR all are
  nothing (no-convection placeholder).
- Not a mix of CMFMC fields with TM5 fields.
- **Capability (which set of fields is non-nothing) is INVARIANT
  for the lifetime of a `DrivenSimulation`.** See §2.21 Decision 27.
  In particular, the `dtrain === nothing` Tiedtke-style fallback is
  allowed, but it must apply to the WHOLE run — not toggle per
  window. Sim construction locks capability based on the driver's
  reported traits (§2.22 Decision 28) and the first window.

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

Back-compat: runs without convection get `convection === nothing` on
the window and `ConvectionForcing()` (all-nothing placeholder) on
the model. Existing constructors default `convection = nothing`.
Existing tests unchanged.

**Allocation lifecycle.** `ConvectionForcing()` on the model is a
placeholder — device memory for the actual `cmfmc`, `dtrain`, or
`tm5_fields` arrays is allocated once at `DrivenSimulation`
construction, after the first window has loaded and the required
shape / capability / backend are known. `with_convection(model, op)`
does NOT allocate forcing buffers; it only swaps the operator type.
See §2.20 Decision 26.

Rationale: convection forcing is constant within a met window (no
sub-step interpolation of CMFMC in either RAS or Tiedtke schemes).
The model-side slot could, in principle, just hold a reference. But
the two-slot pattern:

- Mirrors how `model.fluxes` / `window.fluxes` already work, so the
  operator integration stays uniform with advection's.
- Keeps the device-memory lifecycle consistent: the model-side array
  is the one the operator kernel reads; the window-side is the
  staging area after a `load_transport_window!`.
- Leaves a natural slot if sub-window-varying convection is ever
  added — `_refresh_forcing!` becomes a real interpolation rather
  than a reference copy.

`Adapt.adapt_structure` methods for both window structs and for
`TransportModel` (plan 17 Commit 6) are extended to thread the new
fields through device conversion.

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
    state::CellState{B}, forcing::ConvectionForcing, grid,
    op::CMFMCConvection, dt; workspace,
) where {B <: AbstractMassBasis}
    n_sub, sdt = get_or_compute_cmfmc_subcycling(op, forcing, state, workspace, dt)
    for _ in 1:n_sub
        _cmfmc_kernel!(state.air_mass, state.tracers_raw,
                       forcing.cmfmc, forcing.dtrain,
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

Cache invalidation happens in `DrivenSimulation._maybe_advance_window!`
(not in `_refresh_forcing!`). Cache is only stale when the cmfmc
arrays actually change — i.e., on window boundary. Mid-window
substeps reuse cached `n_sub` even though `_refresh_forcing!` runs
every substep, because the window-side and model-side arrays
point to the same (or identical) data.

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

## 2.14 `apply!` signature — takes `ConvectionForcing`, not window

**Decision 3 (revised in v5).** The operator-level `apply!` takes a
`ConvectionForcing` directly, not an `AbstractTransportWindow`. This
matches what the runtime actually has at the call site (see §2.17
Decision 23) and mirrors plans 15 / 16b / 17 in spirit:

```julia
function apply!(
    state::CellState{B},
    forcing::ConvectionForcing,
    grid::AtmosGrid,
    op::AbstractConvectionOperator,
    dt::Real;
    workspace,
) where {B <: AbstractMassBasis}
    # ...
end
```

`workspace` required (no lazy allocation). `CMFMCWorkspace` or
`TM5Workspace` as appropriate. Matches plan 16b diffusion contract.

**No `meteo` kwarg.** Unlike diffusion (which needs
`current_time(meteo)` for time-varying Kz) and surface flux (which
needs `current_time(meteo)` for StepwiseField emission rates),
convection does not need a simulation-time handle: the forcing
arrays ARE the time information. They are refreshed from the window
once per substep by `DrivenSimulation._refresh_forcing!` (§2.17
Decision 23).

**Runtime dispatch:**

- `op::NoConvection` → `apply!(state, forcing, grid, ::NoConvection, dt; workspace=nothing) = state`, pure identity, no kernel launch. Default path is bit-exact with pre-plan-18 behavior when `model.convection = NoConvection()`.
- `op::CMFMCConvection` → sub-cycling loop calling `_cmfmc_kernel!`.
- `op::TM5Convection` → matrix build + LU solve per column.

## 2.15 Array-level entry point — structured-only

**Decision 4 (revised in v5).** Mirror the plan 16b / 17 pattern so
the convection block in `TransportModel.step!` can be called on
`state.tracers_raw` directly. Plan 18 ships the structured-only
signature:

```julia
apply_convection!(q_raw::AbstractArray{FT,4},
                  air_mass::AbstractArray{FT,3},
                  forcing::ConvectionForcing,
                  ::NoConvection, dt, workspace, grid::AtmosGrid) = nothing

function apply_convection!(q_raw::AbstractArray{FT,4},        # (Nx, Ny, Nz, Nt) — structured
                            air_mass::AbstractArray{FT,3},     # (Nx, Ny, Nz)
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt::Real,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid) where FT
    # ...
end

function apply_convection!(q_raw::AbstractArray{FT,4},
                            air_mass::AbstractArray{FT,3},
                            forcing::ConvectionForcing,
                            op::TM5Convection,
                            dt::Real,
                            workspace::TM5Workspace,
                            grid::AtmosGrid) where FT
    # ...
end
```

Both `q_raw` AND `air_mass` are passed. v4 left `air_mass` as a
placeholder; that's a real omission fixed in v5.

State-level `apply!` delegates to `apply_convection!(state.tracers_raw,
state.air_mass, forcing, op, dt, workspace, grid)`. Used by Commit 6's
`step!` orchestration. Not inserted into the palindrome — plan 18
ships convection as a separate block (§2.2).

Face-indexed topologies are rejected via Decision 25 (§2.18) with a
clear error.

## 2.16 (reserved — keeping decision numbering aligned with v4)

## 2.17 Decision 23 — Runtime data flow via `_refresh_forcing!`

**New in v5.** Convection forcing reaches the operator through
`TransportModel.convection_forcing`, populated per substep by
`DrivenSimulation._refresh_forcing!`. The operator sees only the
model-side field; it never sees the window or the driver.

**The two-slot lifecycle:**

```
Driver (once per window)
  │  load_transport_window!(driver, win_idx)
  │    → returns AbstractTransportWindow with .convection::ConvectionForcing
  ▼
DrivenSimulation.window.convection  (window-scoped)
  │  _refresh_forcing!(sim, substep)  — every substep
  │    → copy_convection_forcing!(sim.model.convection_forcing, sim.window.convection)
  ▼
TransportModel.convection_forcing  (per-step, read by operator)
  │  step!(model, dt; meteo = sim)
  │    → apply!(state, model.convection_forcing, grid, op, dt; workspace)
  ▼
Operator kernel
```

Because CMFMC / DTRAIN / entu etc. are constant within a met window,
`copy_convection_forcing!` can in practice be a reference-swap or a
no-op after the first substep; the semantic contract is "ensure
`model.convection_forcing` reflects the current window." The
implementation can be trivially fast.

**`_refresh_forcing!` extension** (existing location: `src/Models/DrivenSimulation.jl:119-131`):

```julia
function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
    else
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    end
    expected_air_mass!(sim.expected_air_mass, sim.window, λ)
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end

    # v5: refresh convection forcing (constant within window)
    if has_convection_forcing(sim.model.convection_forcing) &&
       sim.window.convection !== nothing
        copy_convection_forcing!(sim.model.convection_forcing, sim.window.convection)
    end

    return λ
end
```

`copy_convection_forcing!(dst::ConvectionForcing, src::ConvectionForcing)`
copies array-to-array where both sides have non-nothing fields; errors
if the capability set (CMFMC-mode vs TM5-mode) doesn't match.

**CFL cache invalidation happens in `_maybe_advance_window!`**,
NOT in `_refresh_forcing!`:

```julia
function _maybe_advance_window!(sim::DrivenSimulation, substep::Int)
    if sim.iteration > 0 && substep == 1
        next_window = sim.current_window_index + 1
        next_window <= sim.stop_window ||
            throw(ArgumentError("DrivenSimulation attempted to step past stop_window"))
        sim.window = _adapt_window_to_model_backend(_load_window(sim.driver, next_window),
                                                     sim.model.state.air_mass)
        sim.current_window_index = next_window

        # v5: invalidate CMFMC CFL cache — the cmfmc arrays just changed
        if sim.model.workspace.convection_ws isa CMFMCWorkspace
            invalidate_cmfmc_cache!(sim.model.workspace.convection_ws)
        end

        # ...existing qv + air_mass handling...
    end
end
```

Cache only stales on window boundary because that's when the CMFMC
arrays change. Mid-window substeps reuse the cached `n_sub`.

**TransportModel.step! orchestration** (§2.2 repeated here for
completeness — Decision 23 dictates the data source):

```julia
function step!(model::TransportModel, dt; meteo = nothing)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)

    if !(model.convection isa NoConvection)
        apply!(model.state, model.convection_forcing, model.grid,
               model.convection, dt;
               workspace = model.workspace.convection_ws)
    end

    chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end
```

The convection `apply!` does NOT receive `meteo` — the forcing is
already in `model.convection_forcing`. This is a deliberate
asymmetry with diffusion (which reads `meteo` for Kz time lookup)
and emissions (which reads `meteo` for StepwiseField refresh).

## 2.18 Decision 24 — `meteo` is the sim, not the driver

**New in v5.** `DrivenSimulation.step!` passes `meteo = sim` (not
`meteo = sim.driver`) to `TransportModel.step!`. This gives operators
access to both driver methods AND time state:

```julia
# src/Models/DrivenSimulation.jl:275 — changed in A3:
step!(sim.model, sim.Δt; meteo = sim)   # v5: was `meteo = sim.driver`
sim.time += sim.Δt
sim.iteration += 1
```

`current_time(sim::DrivenSimulation) = sim.time`. The sim field
`sim.time` already exists (`src/Models/DrivenSimulation.jl:26`) and
advances per step at `:276`. No new state needed.

**Fallback stub retained** for operators that receive `meteo = nothing`
(tests, sim-less contexts): `current_time(::Nothing) = 0.0` (new).
The legacy `current_time(::AbstractMetDriver) = 0.0` stub stays as-is
for backward compatibility but is deprecated in favor of the sim
form — old code that passed `meteo = sim.driver` continues to get
`0.0` (silent wrong answer avoided because the sim is now the
canonical `meteo`).

This decision also makes operator access to driver capabilities
(`supports_convection`, `supports_cmfmc`, `supports_dtrain`,
`supports_tm5conv` — see §2.22 Decision 28) straightforward via
`meteo.driver`, while time queries use `current_time(meteo)`.

## 2.19 Decision 25 — Face-indexed convection is out of scope

**New in v5.** Plan 18 is structured-only. Face-indexed
(reduced-Gaussian) convection is a follow-up plan (Plan 18b:
"Face-indexed convection").

Rationale:

- CATRINE production uses structured lat-lon; face-indexed is not
  on the critical path.
- Face-indexed kernel work adds complexity (different indexing,
  different reduction patterns) without near-term production users.
- Scope containment on an already-substantial plan.
- When face-indexed is needed, a small follow-up plan extends the
  dispatch once the structured path is validated.

**Error-stub dispatch.** A method catches face-indexed states
(3D `tracers_raw`) and errors explicitly:

```julia
function apply!(
    state::CellState{B, A, Raw, Names},
    forcing::ConvectionForcing,
    grid::AtmosGrid,
    op::Union{CMFMCConvection, TM5Convection},
    dt::Real;
    workspace,
) where {B <: AbstractMassBasis, A, Raw <: AbstractArray{<:Any, 3}, Names}
    throw(ArgumentError(
        "Face-indexed convection is not in plan 18 scope. " *
        "Plan 18 supports structured lat-lon only. " *
        "See follow-up 'Plan 18b: Face-indexed convection'."
    ))
end
```

Dispatch on `Raw <: AbstractArray{_,3}` catches 3D `tracers_raw`
(face-indexed `(ncells, Nz, Nt)`) vs 4D (structured `(Nx, Ny, Nz, Nt)`).
This is grid-topology-agnostic — it works regardless of which
`AbstractHorizontalMesh` subtype the grid carries. (Note: v4's brief
mentioned `grid::AtmosGrid{<:AbstractFaceIndexedMesh}` — no such type
exists in `src/Grids/AbstractMeshes.jl`; the correct supertype is
`AbstractHorizontalMesh` with structured as `AbstractStructuredMesh`.
The `Raw <: AbstractArray{_,3}` state-shape dispatch works cleanly
without needing a mesh-side trait.)

Commit 0 verifies this dispatch against current `CellState` storage
layout.

## 2.20 Decision 26 — Model-side forcing allocation at sim construction

**New in v5.1.** Post-review correction: v5's naive "default to
`ConvectionForcing()` (all-nothing) and let `_refresh_forcing!` copy
in" design had no place that actually allocated destination buffers.
`with_convection(model, op)` only swaps the operator; `_refresh_forcing!`
checks `has_convection_forcing(model.convection_forcing)` and does
nothing when buffers are missing. Result: the sim path never populated
forcing. Reviewer flag, HIGH.

**The rule.** `DrivenSimulation` owns model-side forcing allocation.
The flow is:

1. User builds a `TransportModel` with `with_convection(model, op)`.
   `model.convection_forcing` stays at its default placeholder
   `ConvectionForcing()` (all-nothing). No device memory yet.
2. User calls `DrivenSimulation(model, driver; ...)`.
3. Constructor loads the first window. If `sim.window.convection !== nothing`
   AND `!(model.convection isa NoConvection)`:
   a. Validate operator ↔ capability (§2.22 Decision 28): the
      operator's required trait must be satisfied by the driver.
   b. Call `allocate_convection_forcing_like(sim.window.convection,
      sim.model.state.air_mass)` to create `similar(...)` arrays on
      the model's backend with matching shapes and matching capability.
   c. Install the allocated `ConvectionForcing` into the model via
      an internal `_install_convection_forcing!` (a private cousin
      of `with_convection_forcing`, used by sim construction only).
4. All subsequent `_refresh_forcing!` calls copy src → allocated dst
   in place. No allocation after step 3.

**Helper.** Add to `src/MetDrivers/ConvectionForcing.jl`:

```julia
"""
    allocate_convection_forcing_like(src::ConvectionForcing, backend_hint) -> ConvectionForcing

Build a destination `ConvectionForcing` whose array fields are
`similar(src_field)` — same shape, same element type, same backend
(inferred from `backend_hint`, typically `sim.model.state.air_mass`).
Capability (which fields are non-nothing) exactly matches `src`.

Used by `DrivenSimulation` construction to seed `model.convection_forcing`
from the first loaded window.
"""
function allocate_convection_forcing_like(src::ConvectionForcing, backend_hint)
    adaptor = _window_backend_adapter(backend_hint)   # already in DrivenSimulation.jl

    cmfmc = src.cmfmc === nothing ? nothing :
            adaptor === Array ? similar(src.cmfmc) : adaptor(similar(src.cmfmc))
    dtrain = src.dtrain === nothing ? nothing :
             adaptor === Array ? similar(src.dtrain) : adaptor(similar(src.dtrain))
    tm5_fields = src.tm5_fields === nothing ? nothing :
        (; entu = (adaptor === Array ? similar(src.tm5_fields.entu) : adaptor(similar(src.tm5_fields.entu))),
           detu = (adaptor === Array ? similar(src.tm5_fields.detu) : adaptor(similar(src.tm5_fields.detu))),
           entd = (adaptor === Array ? similar(src.tm5_fields.entd) : adaptor(similar(src.tm5_fields.entd))),
           detd = (adaptor === Array ? similar(src.tm5_fields.detd) : adaptor(similar(src.tm5_fields.detd))))
    return ConvectionForcing(cmfmc, dtrain, tm5_fields)
end
```

(The `_window_backend_adapter` helper already exists at
`src/Models/DrivenSimulation.jl:81-89`; reuse it rather than
re-implementing.)

**Implication for `with_convection`.** `with_convection(model, op)`
continues to preserve `convection_forcing` unchanged — it's the
user-facing operator-swap helper. Test 13 in Commit 2 stays valid.
Allocation is a sim-construction responsibility, not a model-
construction one. `with_convection_forcing` stays as a test-helper
for direct injection (tests that skip the sim layer).

## 2.21 Decision 27 — Convection forcing capability is invariant per run

**New in v5.1.** Post-review correction: v5 allowed
`dtrain === nothing` mid-run Tiedtke-style fallback AND used a
permissive `copy_convection_forcing!` that silently skipped dtrain
when `src.dtrain === nothing`. That combination means: if the
destination has dtrain preallocated and a fallback window arrives
(src.dtrain nothing), the destination keeps stale values live. If
destination lacks dtrain and a full-path window arrives, capability
mismatch. Reviewer flag, HIGH.

**The rule.** `ConvectionForcing` capability — the set of fields
that are non-nothing — is fixed at sim construction and does NOT
change window-to-window.

Concretely:

- If driver reports `supports_dtrain(driver) == true`, then for
  EVERY loaded window, `window.convection.dtrain !== nothing`.
  Preprocessing enforces this: a binary with `include_dtrain == true`
  stores a DTRAIN block for every window.
- If driver reports `supports_dtrain(driver) == false`, then for
  EVERY loaded window, `window.convection.dtrain === nothing`. The
  CMFMCConvection kernel runs its Tiedtke-style single-flux path
  for the whole run.
- No window can toggle between modes.

**`copy_convection_forcing!` semantic update.** Stricter than v5's
version. Requires EXACT capability match between src and dst, not
just one-directional compatibility:

```julia
function copy_convection_forcing!(dst::ConvectionForcing, src::ConvectionForcing)
    # Capability must match exactly — no silent capability change.
    # This catches:
    #   - dst preallocated with dtrain, src has dtrain=nothing → stale values bug
    #   - dst without dtrain, src has dtrain → missing destination
    #   - etc.
    _check_capability_match(dst, src)

    if src.cmfmc !== nothing
        copyto!(dst.cmfmc, src.cmfmc)
    end
    if src.dtrain !== nothing
        copyto!(dst.dtrain, src.dtrain)
    end
    if src.tm5_fields !== nothing
        for name in (:entu, :detu, :entd, :detd)
            copyto!(getfield(dst.tm5_fields, name), getfield(src.tm5_fields, name))
        end
    end
    return dst
end

@inline _cap(f::ConvectionForcing) = (f.cmfmc !== nothing, f.dtrain !== nothing, f.tm5_fields !== nothing)

function _check_capability_match(dst::ConvectionForcing, src::ConvectionForcing)
    _cap(dst) == _cap(src) ||
        throw(ArgumentError(
            "ConvectionForcing capability mismatch (dst: $(_cap(dst)), src: $(_cap(src))). " *
            "Per Decision 27, capability is invariant for the lifetime of a DrivenSimulation. " *
            "Check preprocessing: binaries must write a consistent set of convection blocks across all windows."
        ))
end
```

**Tests (extend Commit 2's `test_convection_forcing.jl`):**

- Capability mismatch `(cmfmc=true, dtrain=false)` dst vs
  `(cmfmc=true, dtrain=true)` src → `ArgumentError`.
- Capability mismatch both directions (src without dtrain, dst with
  dtrain, and vice versa) → `ArgumentError`.
- Capability match, all-nothing placeholder → no-op succeeds.

**Tests (extend Commit 8's `test_driven_simulation_convection.jl`):**

- End-to-end: dtrain-present run across window boundary, capability
  preserved, mass-conserved bit-exact.
- End-to-end: dtrain-absent run (Tiedtke fallback) across window
  boundary, capability preserved.

**Preprocessing contract implication.** The preprocessing pipeline
writes either `(cmfmc + dtrain)` blocks OR `cmfmc` alone OR
`tm5conv` blocks OR nothing — not a per-window decision. Commit 7.2
enforces this at the header level via independent `include_cmfmc`,
`include_dtrain`, `include_tm5conv` flags decided once per output
binary (§2.22 Decision 28).

## 2.22 Decision 28 — Independent driver capabilities; no `convection_mode`

**New in v5.1.** Post-review correction: v5 had `convection_mode(driver) -> Symbol`
returning `:cmfmc` or `:tm5` with `:cmfmc` preferred — implicit
prioritization that can't represent binaries with both payloads.
Preprocessing already allows independent `include_cmfmc_in_output`
and `include_tm5conv_in_output` flags, so the driver contract was
narrower than the data schema. Reviewer flag, MEDIUM.

**The rule.** Replace `convection_mode` with three independent
capability traits, one per field / field-group:

```julia
# In src/MetDrivers/AbstractMetDriver.jl:
supports_cmfmc(::AbstractMetDriver)   = false
supports_dtrain(::AbstractMetDriver)  = false   # only meaningful when supports_cmfmc
supports_tm5conv(::AbstractMetDriver) = false

# Aggregate trait retained for convenience:
supports_convection(driver::AbstractMetDriver) =
    supports_cmfmc(driver) || supports_tm5conv(driver)

# `convection_mode` is REMOVED. Don't reintroduce.
```

**Driver overrides for `TransportBinaryDriver`:**

```julia
supports_cmfmc(d::TransportBinaryDriver)   = has_cmfmc(d.reader)
supports_dtrain(d::TransportBinaryDriver)  = has_dtrain(d.reader)
supports_tm5conv(d::TransportBinaryDriver) = has_tm5conv(d.reader)
```

`has_dtrain` is new on the reader; Commit 7.1 adds it alongside
`has_cmfmc` / `has_tm5conv`.

**Allowed combinations (all supported):**

| `supports_cmfmc` | `supports_dtrain` | `supports_tm5conv` | Use |
|---|---|---|---|
| false | false | false | No convection (no operator active) |
| true  | true  | false | CMFMC+DTRAIN path (full RAS/GF) |
| true  | false | false | CMFMC-only path (Tiedtke-style fallback, Decision 2) |
| false | false | true  | TM5 path |
| true  | true  | true  | Both paths available; sim selects by operator type |
| true  | false | true  | CMFMC (fallback) + TM5 both available |

**Invalid:** `supports_dtrain=true && supports_cmfmc=false`. Driver
constructor should error if the binary header has `include_dtrain=true`
but `include_cmfmc=false`.

**Operator ↔ capability validation at sim construction.** Sim
construction picks which capability matters for the user's chosen
operator:

```julia
function _validate_convection_capability(model::TransportModel, driver::AbstractMetDriver)
    op = model.convection
    if op isa NoConvection
        return nothing   # no capability required
    elseif op isa CMFMCConvection
        supports_cmfmc(driver) ||
            throw(ArgumentError("CMFMCConvection operator requires driver with supports_cmfmc=true; got $(typeof(driver))"))
        # supports_dtrain is informational only: its value determines whether
        # CMFMCConvection runs full-path (true) or Tiedtke fallback (false).
    elseif op isa TM5Convection
        supports_tm5conv(driver) ||
            throw(ArgumentError("TM5Convection operator requires driver with supports_tm5conv=true; got $(typeof(driver))"))
    else
        throw(ArgumentError("unknown convection operator: $(typeof(op))"))
    end
    return nothing
end
```

Called from `DrivenSimulation` constructor before allocating
`model.convection_forcing` (§2.20 Decision 26).

**`load_transport_window` behavior.** Loads WHICHEVER payloads the
driver advertises:

```julia
# src/MetDrivers/TransportBinaryDriver.jl (sketch):
if supports_convection(d)
    cmfmc_arr = supports_cmfmc(d) ? load_cmfmc_window!(d.reader, win) : nothing
    dtrain_arr = supports_dtrain(d) ? load_dtrain_window!(d.reader, win) : nothing
    tm5_nt = supports_tm5conv(d) ? load_tm5conv_window!(d.reader, win) : nothing
    # Apply dry-basis correction to whichever fields are present
    ...
    forcing = ConvectionForcing(cmfmc_arr, dtrain_arr, tm5_nt)
    window = _window_with_convection(window, forcing)
end
```

The window may carry BOTH CMFMC and TM5 payloads when the binary
has both. The sim's allocated `model.convection_forcing` carries
whichever subset the sim's chosen operator needs — determined at
sim construction via `allocate_convection_forcing_like`, which
matches the FULL window capability. The unused payload is
allocated and copied each substep but not read by the operator; the
overhead is acceptable (cheap device-to-device copy) and keeps the
`copy_convection_forcing!` contract strict per Decision 27.

If this overhead is unacceptable in production (it is not in plan 18
scope), a follow-up optimization can trim allocated forcing to the
operator's actual capability by intersecting window capability with
operator need.

**Tests** (extend Commit 7.4's `test_driver_convection_wiring.jl`):

- Dual-capability binary: `supports_cmfmc`, `supports_dtrain`,
  `supports_tm5conv` all true; `CMFMCConvection` sim works;
  `TM5Convection` sim works (from the same binary).
- CMFMC-only with dtrain: the standard full-RAS path.
- CMFMC-only without dtrain: Tiedtke-style fallback path runs.
- TM5-only: only TM5 path available.
- Invalid: `supports_dtrain=true` without `supports_cmfmc` at
  reader / header construction → error.
- Operator-capability mismatch at sim construction: `TM5Convection`
  with a CMFMC-only driver → clear error.

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
   any claim in §1.3 has changed, update v5 before proceeding.
   Particular attention:
   - `DrivenSimulation` field names (especially `time`, `iteration`,
     `current_window_index`, and that nothing else has moved).
   - `TransportBinaryDriver` is still stateless (`reader` + `grid`
     only).
   - `TransportBinaryHeader` still lacks convection payload fields.
   - No `AbstractFaceIndexedMesh` type (dispatch must be state-shape-
     based per §2.19).
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
2. State-level identity: `apply!(state, ConvectionForcing(), grid, NoConvection(), dt; workspace=nothing)` returns state bit-exact (`==`, not `≈`).
3. Array-level identity: `apply_convection!(q_raw, air_mass, ConvectionForcing(), NoConvection(), dt, nothing, grid) === nothing`.
4. Dispatch correctness: no kernel launched for `NoConvection`.
5. Exported from `AtmosTransport` (symbols visible).
6. Works on both `CellState{DryBasis}` and `CellState{MoistBasis}`.

```bash
git commit -m "Commit 1: AbstractConvectionOperator hierarchy + NoConvection"
```

### Commit 2 — `ConvectionForcing` + window extension + model-side slot

**Extends v4 Commit 2** to also add the model-side slot (the v5 new
piece). Still a single commit since both window and model-side
extensions land together for back-compat.

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
- `src/Models/TransportModel.jl`:
  - Add `convection :: ConvT` and `convection_forcing :: CF` fields.
    `convection` defaults to `NoConvection()`; `convection_forcing`
    defaults to `ConvectionForcing()` (all-nothing).
  - Extend all three constructor methods (LatLonMesh, FaceIndexed,
    CubedSphereMesh error stub) to accept `convection` and
    `convection_forcing` kwargs.
  - Extend `Adapt.adapt_structure` for `TransportModel` to thread
    the two new fields.
- `src/Models/TransportModel.jl` (`with_convection` helper):
  - `with_convection(model, op::AbstractConvectionOperator)` returns
    a new model with only `convection` replaced (parallel to
    `with_diffusion`, `with_chemistry`, `with_emissions`).
  - `with_convection_forcing(model, forcing::ConvectionForcing)`
    returns a new model with only `convection_forcing` replaced
    (useful in tests that want to inject forcing directly without
    a sim).
- `src/MetDrivers/MetDrivers.jl` — include + export
  `ConvectionForcing`, `copy_convection_forcing!`,
  `allocate_convection_forcing_like`, `has_convection_forcing`.
- `src/AtmosTransport.jl` — export `ConvectionForcing`,
  `with_convection`, `with_convection_forcing`.

**`copy_convection_forcing!` semantics (Decision 27 enforcement):**

```julia
function copy_convection_forcing!(dst::ConvectionForcing, src::ConvectionForcing)
    # Strict capability match (Decision 27 — capability is invariant
    # for the lifetime of a sim). Catches both directions: silent
    # stale values (dst has dtrain but src doesn't) and missing dst
    # buffers (src has dtrain but dst doesn't).
    _check_capability_match(dst, src)

    if src.cmfmc !== nothing
        copyto!(dst.cmfmc, src.cmfmc)
    end
    if src.dtrain !== nothing
        copyto!(dst.dtrain, src.dtrain)
    end
    if src.tm5_fields !== nothing
        for name in (:entu, :detu, :entd, :detd)
            copyto!(getfield(dst.tm5_fields, name), getfield(src.tm5_fields, name))
        end
    end
    return dst
end

@inline _cap(f::ConvectionForcing) = (f.cmfmc !== nothing, f.dtrain !== nothing, f.tm5_fields !== nothing)

function _check_capability_match(dst, src)
    _cap(dst) == _cap(src) ||
        throw(ArgumentError(
            "ConvectionForcing capability mismatch (dst: $(_cap(dst)), src: $(_cap(src))). " *
            "Per Decision 27, capability is invariant for the lifetime of a DrivenSimulation."
        ))
end
```

**`allocate_convection_forcing_like` helper:**

```julia
function allocate_convection_forcing_like(src::ConvectionForcing, backend_hint)
    adaptor = _window_backend_adapter(backend_hint)   # existing helper in DrivenSimulation.jl
    _like(arr) = adaptor === Array ? similar(arr) : adaptor(similar(arr))

    cmfmc = src.cmfmc === nothing ? nothing : _like(src.cmfmc)
    dtrain = src.dtrain === nothing ? nothing : _like(src.dtrain)
    tm5_fields = src.tm5_fields === nothing ? nothing :
        (; entu = _like(src.tm5_fields.entu),
           detu = _like(src.tm5_fields.detu),
           entd = _like(src.tm5_fields.entd),
           detd = _like(src.tm5_fields.detd))
    return ConvectionForcing(cmfmc, dtrain, tm5_fields)
end
```

Used only by `DrivenSimulation` construction (Commit 8). Allocation
happens once per sim lifecycle.

**Allocation lifecycle** (per §2.20 Decision 26):

1. `TransportModel` construction: `convection_forcing` defaults to
   placeholder `ConvectionForcing()` (all-nothing). No device memory.
2. `with_convection(model, op)` swaps the operator; leaves
   `convection_forcing` unchanged.
3. `DrivenSimulation` constructor: loads first window, validates
   operator ↔ capability (§2.22 Decision 28), calls
   `allocate_convection_forcing_like(sim.window.convection, model.state.air_mass)`,
   installs allocated forcing into the model via internal helper.
4. `_refresh_forcing!` reuses the allocated buffers on every
   subsequent substep. No further allocation.

**Back-compat:**

- All existing window constructors keep their current signatures and
  default `convection = nothing`.
- All existing `TransportModel` constructors keep their current
  signatures and default `convection = NoConvection()`,
  `convection_forcing = ConvectionForcing()` (all-nothing).
- Runs without convection are bit-exact vs pre-Commit-2.
- `test_transport_binary_reader.jl` passes unchanged.
- `has_convection_forcing(window) = window.convection !== nothing`
  helper added for convenience.
- `has_convection_forcing(forcing::ConvectionForcing) = forcing.cmfmc !== nothing || forcing.tm5_fields !== nothing`
  helper for the model-side field.

**Tests** (`test/test_convection_forcing.jl`, ~20 tests):

1. Default construction: `window.convection === nothing`.
2. With `ConvectionForcing(cmfmc=..., dtrain=..., tm5_fields=nothing)`:
   `window.convection.cmfmc` correctly shaped `(Nx, Ny, Nz+1)`.
3. With `ConvectionForcing(cmfmc=nothing, dtrain=nothing, tm5_fields=(entu=..., detu=..., entd=..., detd=...))`: TM5 fields correctly shaped.
4. Invariant: mixing CMFMC + TM5 fields in one forcing throws.
5. CMFMC-only fallback: `ConvectionForcing(cmfmc=..., dtrain=nothing, tm5_fields=nothing)` is valid.
6. `has_convection_forcing` trait for both window and model-side.
7. `Adapt.adapt_structure` round-trip preserves `convection` on the window.
8. `Adapt.adapt_structure` round-trip preserves `convection` AND `convection_forcing` on the model.
9. CPU / CUDA round-trip with Array → CuArray on `cmfmc` field.
10. Both structured and face-indexed windows support the new field.
11. `copy_convection_forcing!` copies CMFMC + DTRAIN when both capabilities match.
12. `copy_convection_forcing!` copies CMFMC-only (no dtrain) when both sides lack dtrain.
13. `copy_convection_forcing!` copies TM5 fields correctly.
14. **Decision 27 — capability invariance:** `copy_convection_forcing!` errors when dst has dtrain but src doesn't (stale-values regression).
15. **Decision 27 — capability invariance:** `copy_convection_forcing!` errors when src has dtrain but dst doesn't (missing-destination regression).
16. `copy_convection_forcing!` errors on CMFMC vs TM5 mode mismatch.
17. `allocate_convection_forcing_like` produces a `ConvectionForcing` with identical capability and `similar(...)`-shaped arrays on the correct backend.
18. `allocate_convection_forcing_like` on a dtrain-absent src produces a dtrain-absent dst.
19. `with_convection(model, CMFMCConvection())` returns new model with only convection replaced; `convection_forcing` preserved (still the placeholder unless previously installed).
20. `with_convection_forcing` returns new model with only convection_forcing replaced.
21. Type parameter `C` is `Nothing` when no convection on window; concrete `ConvectionForcing{CM, DT, TM}` when present.

```bash
git commit -m "Commit 2: ConvectionForcing + window and model-side slots"
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
- Face-indexed error stub (§2.19 Decision 25)

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
function apply_convection!(q_raw::AbstractArray{FT,4},
                            air_mass::AbstractArray{FT,3},
                            forcing::ConvectionForcing,
                            op::CMFMCConvection,
                            dt::Real,
                            workspace::CMFMCWorkspace,
                            grid::AtmosGrid) where FT
    forcing.cmfmc === nothing && throw(ArgumentError("CMFMCConvection requires cmfmc in ConvectionForcing"))

    # Compute / reuse sub-cycling
    n_sub = _get_or_compute_n_sub!(workspace, forcing.cmfmc, air_mass, dt)
    sdt   = dt / n_sub

    # Sub-step loop
    for _ in 1:n_sub
        _cmfmc_kernel!(q_raw, air_mass,
                       forcing.cmfmc, forcing.dtrain,
                       workspace.qc_scratch, FT(sdt), grid)
    end
end
```

The sub-cycling loop is visible in the signature; `n_sub` and its
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

When `forcing.dtrain === nothing`, the kernel runs Tiedtke-style
single-flux transport: `entrn = 0`, `cmout = cmfmc_below`,
tendency reduces to `cmfmc[k] · (q_above - q_env)` only. No
separate operator type — runtime path within `CMFMCConvection`.

**Cleanup item:** remove the stale legacy comment at
`src_legacy/Convection/ras_convection.jl:41-46` during the port.
`CMFMCConvection` docstring states basis follows state.

**Tier A tests (~6 tests)**:

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
6. **Face-indexed rejection** (§2.19 Decision 25): construct a
   face-indexed `CellState` (3D `tracers_raw`), call `apply!`,
   verify `ArgumentError` with the expected message. Exercises the
   state-shape dispatch explicitly.

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

# Fields required on `ConvectionForcing`

- `forcing.cmfmc :: AbstractArray{FT, 3}` at interfaces, shape
  (Nx, Ny, Nz+1). Units kg / m² / s on basis matching state.
- `forcing.dtrain :: Union{AbstractArray{FT, 3}, Nothing}` at centers,
  shape (Nx, Ny, Nz). When `nothing`, reduces to Tiedtke-style
  single-flux transport (DTRAIN-missing fallback).

The `ConvectionForcing` value reaches the operator via
`TransportModel.convection_forcing`, populated each substep by
`DrivenSimulation._refresh_forcing!`.

# CFL sub-cycling

The kernel sub-cycles internally based on the CMFMC profile:
`n_sub = max(1, ceil(max(cmfmc × dt / air_mass) / 0.5))`. Cached per
window in the workspace; driver invalidates on window roll (in
`_maybe_advance_window!`).

# Well-mixed sub-cloud layer

Applies GCHP's pressure-weighted well-mixed treatment below cloud
base (`convection_mod.F90:742-782`). Absent in legacy Julia;
deliberate improvement for surface-source tracers.

# Scope

Plan 18 is structured-only. Face-indexed state raises
`ArgumentError`; use the forthcoming Plan 18b follow-up for face-
indexed support.

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

**Face-indexed error stub** (§2.19 Decision 25) shared with
CMFMCConvection via the parametric dispatch on `Raw <: AbstractArray{_,3}`.

**Tier A tests (~6)**: mass conservation (exact, implicit solve),
zero forcing, positivity, matrix diagonal dominance (structural),
**adjoint identity** — extract `conv1` from workspace, verify
`⟨y, conv1⁻¹x⟩ ≈ ⟨(conv1^T)⁻¹y, x⟩` using LU transpose solve
(no extra factorization), face-indexed rejection test.

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

# Fields required on `ConvectionForcing`

- `forcing.tm5_fields :: NamedTuple{(:entu, :detu, :entd, :detd)}`,
  each `AbstractArray{FT, 3}` shape (Nx, Ny, Nz). Units kg / m² / s.

# lmax_conv

Maximum level for convection. 0 sentinel = use full Nz. The matrix
builder returns `lmc` (actual convection top) and the solver runs on
levels 1..lmc only.

# Scope

Plan 18 is structured-only. Face-indexed state raises
`ArgumentError`; use the forthcoming Plan 18b follow-up for face-
indexed support.

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

### Commit 6 — TransportModel wiring + `step!` orchestration

**Scope refined in v5.** Commit 2 already added the model-level
fields (`convection`, `convection_forcing`) and the `with_convection`/`with_convection_forcing`
helpers. Commit 6 wires `step!` orchestration and verifies end-to-end
runtime bit-exactness.

**Files modified:**

- `src/Models/TransportModel.jl`:
  - Update `step!` to invoke the convection block between transport
    and chemistry blocks (§2.17 Decision 23):
    ```julia
    function step!(model::TransportModel, dt; meteo = nothing)
        apply!(model.state, model.fluxes, model.grid, model.advection, dt;
               workspace = model.workspace,
               diffusion_op = model.diffusion,
               emissions_op = model.emissions,
               meteo = meteo)

        if !(model.convection isa NoConvection)
            apply!(model.state, model.convection_forcing, model.grid,
                   model.convection, dt;
                   workspace = model.workspace.convection_ws)
        end

        chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
        return nothing
    end
    ```

**Workspace integration:**

- Extend `AdvectionWorkspace` (or `TransportModel.workspace`) to
  include `workspace.convection_ws :: Union{Nothing, CMFMCWorkspace, TM5Workspace}`.
- Default `nothing` when `model.convection isa NoConvection`.
- `with_convection(model, op)` allocates the appropriate workspace.

The `if !(model.convection isa NoConvection)` branch is compile-time
removable when `model.convection` is concretely typed as `NoConvection`
— zero runtime overhead in default configuration.

**Tests (`test/test_transport_model_convection.jl`, ~12 tests)**:

1. Default `TransportModel` carries `NoConvection`.
2. `with_convection(model, CMFMCConvection())` returns new model
   with only convection replaced (other fields `===` identity).
3. `with_convection(model, TM5Convection(; lmax_conv=30))` similarly.
4. **Bit-exact regression:** default `step!` `==` pre-plan-18
   no-convection path (`==`, not `≈`). Critical test.
5. `step!` with `CMFMCConvection` + populated `model.convection_forcing`
   produces measurable vertical redistribution.
6. `step!` with `TM5Convection` similarly.
7. **Forcing source is model, not window/driver:** manually set
   `model.convection_forcing` to test values, run `step!`, verify
   the kernel saw those values (not the sim's window).
8. Plan 17's `test_transport_model_emissions.jl` passes unchanged.
9. Plan 16b's `test_transport_model_diffusion.jl` passes unchanged.
10. Plan 15's `test_transport_model_chemistry.jl` passes unchanged.
11. `with_convection(NoConvection())` round-trip works.
12. Operator does NOT receive `meteo` — verify by constructing a
    dummy `meteo` with a counter and confirming it's not touched
    by the convection block.

```bash
git commit -m "Commit 6: TransportModel step! orchestration for convection"
```

### Commit 7 — Driver / window integration (EXPANDED, sub-committed)

**Significantly expanded vs v4.** This commit wires convection
forcing from the preprocessing pipeline through the generic
`TransportBinaryReader` (used by CATRINE production binaries),
through the driver contract, and into window population. It's
naturally four sub-commits that may be split at session boundaries
but land as a single logical unit.

**Effort: ~2.5 days total.**

#### Commit 7.1 — `TransportBinary.jl` header + reader methods

**Files modified:**

- `src/MetDrivers/TransportBinary.jl`:
  - Add to `TransportBinaryHeader` (line 35 block). Note three
    independent flags per Decision 28 — `include_dtrain` is separate
    from `include_cmfmc` so the binary can be CMFMC-only (Tiedtke-
    style fallback) without DTRAIN:
    ```julia
    include_cmfmc    :: Bool   # CMFMC block present
    include_dtrain   :: Bool   # DTRAIN block present (meaningful only when include_cmfmc)
    include_tm5conv  :: Bool   # entu/detu/entd/detd blocks present
    n_cmfmc          :: Int    # nelements per window (for cmfmc block); 0 if absent
    n_dtrain         :: Int    # nelements per window (for dtrain block); 0 if absent
    n_tm5conv        :: Int    # nelements per window per field (×4 in total)
    ```
  - Constructor-level validation: `include_dtrain && !include_cmfmc`
    → `ArgumentError`. DTRAIN without CMFMC is meaningless.
  - Extend header serialization / deserialization following the
    `include_qv` / `n_qv` template.
  - Add to `TransportBinaryReader` capability traits (Decision 28):
    ```julia
    has_cmfmc(r::TransportBinaryReader)   = r.header.include_cmfmc
    has_dtrain(r::TransportBinaryReader)  = r.header.include_dtrain
    has_tm5conv(r::TransportBinaryReader) = r.header.include_tm5conv
    ```
  - Add loaders following `load_qv_window!` template (to be verified;
    check for existing qv loader and mirror):
    ```julia
    function load_cmfmc_window!(r::TransportBinaryReader, win::Int;
                                cmfmc = Array{FT}(undef, Nx, Ny, Nz+1))
        has_cmfmc(r) || return nothing
        # ... compute offset, copy from r.data ...
        return cmfmc
    end

    function load_dtrain_window!(r::TransportBinaryReader, win::Int;
                                 dtrain = Array{FT}(undef, Nx, Ny, Nz))
        has_dtrain(r) || return nothing   # independent of has_cmfmc — Decision 28
        # ... compute offset, copy ...
        return dtrain
    end

    function load_tm5conv_window!(r::TransportBinaryReader, win::Int;
                                   entu = ..., detu = ..., entd = ..., detd = ...)
        has_tm5conv(r) || return nothing
        # ...
        return (; entu, detu, entd, detd)
    end
    ```

**Tests (`test/test_transport_binary_convection.jl`, ~12 tests):**

- Header round-trip with each combination of payload flags.
- Constructor rejects `include_dtrain && !include_cmfmc`.
- Load CMFMC + DTRAIN from a binary with `include_cmfmc=true, include_dtrain=true`.
- Load CMFMC-only (Tiedtke fallback) from a binary with
  `include_cmfmc=true, include_dtrain=false`: `load_cmfmc_window!`
  returns array; `load_dtrain_window!` returns `nothing`.
- Load TM5 conv fields from a binary with `include_tm5conv=true`.
- Load from a DUAL-capability binary (`include_cmfmc && include_tm5conv`):
  both `load_cmfmc_window!` and `load_tm5conv_window!` return arrays
  (Decision 28 — no mutual exclusion).
- Absent-capability paths return `nothing` for each loader.
- Back-compat: existing test binaries (no convection fields)
  continue to work; new fields default to `false` / `0`.

#### Commit 7.2 — Preprocessing pipeline writes convection blocks

**Files modified:**

- `src/Preprocessing/binary_pipeline.jl` (or whichever preprocessing
  module generates the transport binary from raw met):
  - Add three independent flags to the preprocessing config
    (Decision 28):
    ```julia
    include_cmfmc_in_output   :: Bool   # write CMFMC blocks
    include_dtrain_in_output  :: Bool   # write DTRAIN blocks (requires include_cmfmc_in_output)
    include_tm5conv_in_output :: Bool   # write entu/detu/entd/detd blocks
    ```
  - Validate: `include_dtrain && !include_cmfmc` → error. DTRAIN
    without CMFMC is meaningless (CMFMC is the updraft mass flux,
    DTRAIN the detrainment from that flux).
  - **Decision 27 invariance:** if a flag is true, it's true for
    EVERY window in the output. No per-window toggling.
  - When `include_cmfmc_in_output`:
    ```julia
    cmfmc = load_cmfmc_window!(era5_reader, win)
    write_cmfmc_block(output_stream, cmfmc)
    if include_dtrain_in_output
        dtrain = _derive_dtrain_from_tm5conv(era5_reader, win)
        dtrain === nothing && error(
            "include_dtrain_in_output=true but source binary has no tm5conv " *
            "to derive dtrain from; either disable dtrain output or source a binary with tm5conv"
        )
        write_dtrain_block(output_stream, dtrain)
    end
    ```
  - When `include_tm5conv_in_output`:
    ```julia
    tm5_fields = load_tm5conv_window!(era5_reader, win)
    write_tm5conv_block(output_stream, tm5_fields)
    ```
  - All three can be true simultaneously (dual-capability binary,
    Decision 28).

**DTRAIN sourcing.** ERA5 binaries currently store CMFMC but not
DTRAIN. Plan 18 approach: DTRAIN is derived from the tm5conv `detu`
field when source has tm5conv. If source lacks tm5conv, preprocessing
either (a) writes a CMFMC-only binary (Tiedtke-style fallback at
runtime) or (b) errors if `include_dtrain_in_output=true` was set
— no silent synthesis.

```julia
function _derive_dtrain_from_tm5conv(reader, win)
    if has_tm5conv(reader)
        tm5 = load_tm5conv_window!(reader, win)
        return tm5.detu   # detrainment from updraft is the DTRAIN analog
    else
        return nothing
    end
end
```

This approach makes the CMFMC + DTRAIN full-path testable against
both ERA5 sources (via tm5conv's detu) and future GCHP sources
(when native DTRAIN imports are added).

**Tests:** expand `test_preprocessing.jl` or add a dedicated test
for the convection-writing path.

#### Commit 7.3 — `TransportBinaryDriver` convection integration

**Files modified:**

- `src/MetDrivers/AbstractMetDriver.jl`:
  - Keep `supports_convection(::AbstractMetDriver) = false` (already
    exists at `:95`); add three independent capability traits per
    Decision 28:
    ```julia
    """
        supports_cmfmc(driver)   -> Bool
        supports_dtrain(driver)  -> Bool   # meaningful only when supports_cmfmc
        supports_tm5conv(driver) -> Bool

    Independent capability traits. Drivers override these to advertise
    which convection-forcing fields they supply. `supports_convection`
    is the aggregate.

    Plan 18 Decision 28 — capabilities are independent; a binary may
    carry both CMFMC and TM5 payloads. The sim selects which capability
    to consume based on the operator type (`CMFMCConvection` reads
    cmfmc/dtrain; `TM5Convection` reads tm5_fields).

    `supports_dtrain=true` requires `supports_cmfmc=true` (DTRAIN
    without CMFMC is meaningless).
    """
    supports_cmfmc(::AbstractMetDriver)   = false
    supports_dtrain(::AbstractMetDriver)  = false
    supports_tm5conv(::AbstractMetDriver) = false

    # Aggregate — replaces any aggregate-only code paths:
    supports_convection(driver::AbstractMetDriver) =
        supports_cmfmc(driver) || supports_tm5conv(driver)
    ```
  - **Remove `convection_mode`.** It was a single-symbol trait with
    implicit `:cmfmc`-wins prioritization that can't describe dual-
    capability binaries. Any call site using it updates to check the
    independent traits.

- `src/MetDrivers/TransportBinaryDriver.jl`:
  - Override the three traits:
    ```julia
    supports_cmfmc(d::TransportBinaryDriver)   = has_cmfmc(d.reader)
    supports_dtrain(d::TransportBinaryDriver)  = has_dtrain(d.reader)
    supports_tm5conv(d::TransportBinaryDriver) = has_tm5conv(d.reader)
    ```
  - Extend `load_transport_window` to populate whichever payloads the
    driver supports, capability-independently:
    ```julia
    if supports_convection(d)
        cmfmc_arr  = supports_cmfmc(d)   ? load_cmfmc_window!(d.reader, win)  : nothing
        dtrain_arr = supports_dtrain(d)  ? load_dtrain_window!(d.reader, win) : nothing
        tm5_nt     = supports_tm5conv(d) ? load_tm5conv_window!(d.reader, win) : nothing

        # Apply dry-basis correction per Decision 20 to whichever
        # fields are present. Each helper is a no-op when its input
        # is nothing.
        if _driver_basis(d) === DryBasis && _source_basis_is_moist(d)
            cmfmc_arr, dtrain_arr = _apply_dry_conv_correction(cmfmc_arr, dtrain_arr, qv)
            tm5_nt = _apply_dry_tm5_correction(tm5_nt, qv)
        end

        forcing = ConvectionForcing(cmfmc_arr, dtrain_arr, tm5_nt)
        window = _window_with_convection(window, forcing)
    end
    ```
  - Decision 27 invariance: on every window, the capability set
    (which traits are true) is the same. Every window's
    `ConvectionForcing` has the same `_cap(forcing)` tuple. This is
    automatic because capability is driven by the binary header's
    flags, which are constant across windows.

- `src/MetDrivers/DryFluxBuilder.jl` (or new `DryConvFluxBuilder.jl`):
  - Add `_apply_dry_conv_correction(cmfmc_moist, dtrain_moist, qv)`
    handling `nothing` inputs (pass through as `nothing`).
  - Add `_apply_dry_tm5_correction(tm5_nt_moist, qv)` similarly
    handling `tm5_nt_moist === nothing`.

**Basis correction** (Decision 20):

When the driver is configured for dry basis but the binary stores moist
CMFMC / DTRAIN (typical case), apply:
```
cmfmc_dry(i, j, k_interface) = cmfmc_moist × (1 - qv_at_interface)
dtrain_dry(i, j, k_center)   = dtrain_moist × (1 - qv_at_center)
```
and analogous for TM5 fields at full levels. qv interpolation follows
the existing `DryFluxBuilder` pattern.

**Binary may already store dry fields:** in that case the header flag
`mass_basis` (already in `TransportBinaryHeader:59`) indicates basis
and the correction is skipped.

#### Commit 7.4 — End-to-end integration test

**Tests (`test/test_driver_convection_wiring.jl`, ~14 tests):**

1. Driver traits for CMFMC-only-with-dtrain binary:
   `supports_cmfmc=true, supports_dtrain=true, supports_tm5conv=false`.
2. Driver traits for CMFMC-only-without-dtrain binary (Tiedtke
   fallback): `supports_cmfmc=true, supports_dtrain=false, supports_tm5conv=false`.
3. Driver traits for TM5-only binary:
   `supports_cmfmc=false, supports_dtrain=false, supports_tm5conv=true`.
4. Driver traits for **dual-capability binary** (both CMFMC+DTRAIN
   and TM5 blocks present): all three traits true.
5. Driver traits for a binary with neither: all three false,
   `supports_convection=false`.
6. Window population from CMFMC+DTRAIN binary:
   `window.convection.cmfmc` shape correct, `dtrain` shape correct,
   `tm5_fields === nothing`.
7. Window population from CMFMC-only-without-dtrain binary:
   `cmfmc` populated, `dtrain === nothing`, `tm5_fields === nothing`.
8. Window population from TM5 binary: `tm5_fields` NamedTuple with
   correct shapes, `cmfmc === nothing`, `dtrain === nothing`.
9. Window population from DUAL-capability binary: `cmfmc` and
   `dtrain` populated AND `tm5_fields` populated. Both paths
   available.
10. Basis correction applied when dry driver + moist binary.
11. No correction applied when dry driver + dry binary (header flag).
12. Decision 27 invariance: two consecutive windows from the same
    driver have the same `_cap(window.convection)` tuple.
13. Back-compat: runs without convection capability get
    `window.convection === nothing` and operator defaults to `NoConvection`.
14. Adapt round-trip for a window with `ConvectionForcing`.
15. End-to-end: preprocessing pipeline → binary → reader → driver → window → `model.convection_forcing` (after Commit 8's allocation + `_refresh_forcing!`). Verify forcing arrives at operator.

```bash
git commit -m "Commit 7: Driver/window integration (TransportBinary header + preprocessing + driver + e2e)"
```

### Commit 8 — DrivenSimulation integration (EXPANDED in v5 / v5.1)

**Scope.** Commit 8 owns four pieces:

- Model-side `ConvectionForcing` allocation at sim construction
  (§2.20 Decision 26).
- Operator ↔ capability validation at sim construction
  (§2.22 Decision 28).
- `_refresh_forcing!` extension that copies window → model forcing
  every substep (§2.17 Decision 23).
- `_maybe_advance_window!` extension that invalidates the CMFMC
  CFL cache on window roll (Decision 21).
- `DrivenSimulation.step!` change to `meteo = sim` (Decision 24,
  A3).
- `current_time(sim)` override (A3).

Effort ~1.5 days (up from v5's "~1 day" — v5.1 allocation +
validation work adds a few hours).

**Files modified:**

- `src/Models/DrivenSimulation.jl`:
  - **Constructor extension.** After the first window loads and
    after basis/grid compatibility checks, before any `copy_fluxes!`:
    ```julia
    # v5.1 Decision 26 + 28: validate operator ↔ capability, then
    # allocate model.convection_forcing buffers to match the first
    # window's capability.
    if !(model.convection isa NoConvection)
        _validate_convection_capability(model, driver)   # Decision 28
        if window.convection === nothing
            throw(ArgumentError(
                "model.convection = $(typeof(model.convection)) but the loaded " *
                "window has no convection forcing. Check driver supports_cmfmc/supports_tm5conv."
            ))
        end
        forcing_alloc = allocate_convection_forcing_like(
            window.convection, model.state.air_mass
        )
        model = _install_convection_forcing(model, forcing_alloc)
    end
    # If model.convection isa NoConvection, convection_forcing stays
    # at the all-nothing placeholder. `_refresh_forcing!` is a no-op
    # for that branch.
    ```
    `_install_convection_forcing(model, forcing)` is a private helper
    that returns a new `TransportModel` with only `convection_forcing`
    replaced — parallel to `with_convection_forcing` but intended for
    sim-internal use.

  - **`_validate_convection_capability(model, driver)`** (Decision 28):
    ```julia
    function _validate_convection_capability(model::TransportModel, driver::AbstractMetDriver)
        op = model.convection
        if op isa CMFMCConvection
            supports_cmfmc(driver) ||
                throw(ArgumentError("CMFMCConvection requires driver with supports_cmfmc=true; got $(typeof(driver))"))
            # supports_dtrain is informational: false → Tiedtke-style fallback at runtime
        elseif op isa TM5Convection
            supports_tm5conv(driver) ||
                throw(ArgumentError("TM5Convection requires driver with supports_tm5conv=true; got $(typeof(driver))"))
        end
        return nothing
    end
    ```

  - **Extend `_refresh_forcing!`** (lines 119-131) to copy
    window → model.convection_forcing per §2.17 Decision 23:
    ```julia
    if has_convection_forcing(sim.model.convection_forcing) &&
       sim.window.convection !== nothing
        copy_convection_forcing!(sim.model.convection_forcing, sim.window.convection)
    end
    ```
    The first substep post-allocation performs the first real copy;
    subsequent substeps reuse the same device buffers.
    `copy_convection_forcing!` enforces capability match (Decision 27)
    — if a later window has different capability (preprocessing bug),
    it errors loudly.

  - **Extend `_maybe_advance_window!`** (lines 137-152) to invalidate
    the CMFMC CFL cache on window roll (Decision 21 / §2.17):
    ```julia
    if sim.model.workspace.convection_ws isa CMFMCWorkspace
        invalidate_cmfmc_cache!(sim.model.workspace.convection_ws)
    end
    ```

  - **Change `step!`** (line 275) per Decision 24: `meteo = sim`
    instead of `meteo = sim.driver`. (This is A3's deliverable per
    PRE_PLAN_18_FIXES; Commit 8 depends on it but doesn't re-ship
    it if A3 is already in.)

  - Add `current_time(sim::DrivenSimulation) = sim.time` override
    (also an A3 deliverable).

**`DrivenSimulation` constructor flow summary:**

```
user builds model with `with_convection(model, op)`           # placeholder forcing
user calls DrivenSimulation(model, driver; ...)
├── grid + basis compatibility checks (existing)
├── load first window (existing)
├── adapt window to model backend (existing)
├── [NEW v5.1] validate operator ↔ capability per Decision 28
├── [NEW v5.1] if model.convection isa concrete operator:
│   ├── assert sim.window.convection !== nothing
│   ├── allocate_convection_forcing_like(window.convection, air_mass)
│   └── install into model
├── copy_fluxes!, expected_air_mass!, qv init (existing)
└── return sim
```

**Tests (`test/test_driven_simulation_convection.jl`, ~14 tests)**:

1. Construction with `convection = NoConvection()` (default), no driver
   convection capability → unchanged behavior vs plan 17.
2. Construction with `model = with_convection(model, CMFMCConvection())`,
   binary supports cmfmc + dtrain → full path runs; allocated
   `model.convection_forcing` has `cmfmc` + `dtrain` non-nothing,
   `tm5_fields` nothing.
3. Construction with `model = with_convection(model, CMFMCConvection())`,
   binary supports cmfmc only (no dtrain) → Tiedtke-style fallback
   runs; allocated forcing has `cmfmc` non-nothing, `dtrain` nothing.
4. Construction with `model = with_convection(model, TM5Convection(;lmax_conv=...))`,
   binary supports tm5conv → full path runs; allocated forcing has
   `tm5_fields` non-nothing.
5. **Dual-capability binary (Decision 28):** `CMFMCConvection` sim
   AND `TM5Convection` sim can both be constructed from the same
   driver (separate sim objects).
6. Error: `CMFMCConvection` with a driver that has neither cmfmc
   nor tm5conv → clear error at sim construction with "requires
   driver with supports_cmfmc=true" message.
7. Error: `TM5Convection` with a CMFMC-only binary → clear error.
8. Error: `model.convection` is concrete but window has
   `convection === nothing` → "loaded window has no convection forcing"
   error at sim construction.
9. Allocation correctness: allocated `model.convection_forcing` arrays
   have the correct shape, element type, and backend (CUDA when
   `model.state.air_mass isa CuArray`).
10. **Decision 27 capability invariance:** run across window boundary,
    `copy_convection_forcing!` succeeds because capability is
    invariant.
11. End-to-end 10-step run preserves total mass to machine precision.
12. Cache invalidation occurs on window roll (counter test on
    `ws.cache_valid`).
13. `current_time(sim)` returns `sim.time` and advances per step.
14. Existing `test_driven_simulation.jl` 57-test suite passes unchanged.

```bash
git commit -m "Commit 8: DrivenSimulation convection refresh + cache invalidation + meteo=sim"
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
  - Runtime data flow: `ConvectionForcing` on window + model
    (§2.17 Decision 23), refreshed per substep.
  - Window extension (`ConvectionForcing` field) documented.
  - File-level map updated.
- Update `OPERATOR_COMPOSITION.md` with block-vs-palindrome decision
  and reference to position study results.
- Write `docs/plans/18_ConvectionPlan/validation_report.md`
  summarizing three-tier test coverage, adjoint-identity results,
  legacy bugs found and fixed.
- Document any scope deviations (§7 mandatory: split / merged / reshuffled
  commits).
- Move superseded v3 base plan, corrections addendum v2, adjoint
  addendum, v4 plan, and V5 architectural brief to
  `docs/plans/18_ConvectionPlan/archived/` with the `README.md`
  updated to reflect v5 as authoritative.

```bash
git commit -m "Commit 11: Retrospective + ARCHITECTURAL_SKETCH_v4 + validation_report"
```

---

# Part 4 — Acceptance criteria

## 4.1 Correctness (hard)

- All pre-existing tests pass; baseline 77-failure count unchanged
  across all 12 commits.
- Plans 11-17 regression tests pass bit-exact.
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
  merges.

## 4.3 Runtime data flow (hard — new in v5)

- `TransportModel.convection_forcing` is the operator's sole source
  of convection data — the operator does not reference the window
  or driver. Verified by Commit 6 test 12 (dummy meteo counter).
- `DrivenSimulation._refresh_forcing!` populates
  `sim.model.convection_forcing` from `sim.window.convection` every
  substep when convection is active. Verified by Commit 8 test 7.
- `DrivenSimulation._maybe_advance_window!` invalidates the CMFMC
  CFL cache on window roll. Verified by Commit 8 test 8.
- `DrivenSimulation.step!` passes `meteo = sim` (not `sim.driver`).
  `current_time(sim) = sim.time` advances per step.

## 4.4 Code cleanliness (hard)

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
- Face-indexed error-stub dispatch lands via state-shape parametric
  (`Raw <: AbstractArray{_,3}`), not via a nonexistent mesh trait.

## 4.5 Interface validation (hard)

- `apply!(state, forcing::ConvectionForcing, grid, op, dt; workspace)`
  signature works for plan 18 operators (no window at the operator
  call site — see §2.14).
- `AbstractTransportWindow{B}` extension with optional
  `ConvectionForcing` field works for both `StructuredTransportWindow`
  and `FaceIndexedTransportWindow`.
- `TransportModel` has `convection` and `convection_forcing` fields,
  `with_convection` and `with_convection_forcing` helpers.
- Array-level `apply_convection!` entry point takes both `q_raw` and
  `air_mass`.
- Independent driver capability traits `supports_cmfmc`,
  `supports_dtrain`, `supports_tm5conv` (Decision 28) work for
  `TransportBinaryDriver` and fall back cleanly for drivers that
  don't implement them. `supports_convection` aggregate retained.
  `convection_mode` is not re-introduced.
- `TransportBinaryReader` supports `has_cmfmc`, `has_dtrain`,
  `has_tm5conv`, and the three loaders after Commit 7.1.
  Preprocessing pipeline populates the corresponding blocks after
  Commit 7.2 with independent `include_cmfmc_in_output`,
  `include_dtrain_in_output`, `include_tm5conv_in_output` flags.
- Model-side `ConvectionForcing` allocation happens at
  `DrivenSimulation` construction (§2.20 Decision 26), not at
  `TransportModel` construction. Verified by Commit 8 test 9.
- Capability invariance (§2.21 Decision 27) is enforced at
  `copy_convection_forcing!` boundary with a strict tuple match.
  Verified by Commit 2 tests 14-15.

## 4.6 Adjoint structure preservation (hard)

- Both operators pass the Tier A adjoint-identity test
  (Commit 3 Tier A, Commit 4 Tier A).
- No positivity clamp inside either kernel.
- Coefficients (matrix entries for TM5, mass-flux terms for CMFMC)
  stored in named-locals or explicit-array form amenable to
  transposition.
- Docstrings document the adjoint path with the "# Adjoint path
  (not shipped in plan 18)" section.

## 4.7 Validation discipline (hard)

- Upstream Fortran references read and annotated in
  `artifacts/plan18/upstream_fortran_notes.md` and `.../upstream_gchp_notes.md`.
- Three-tier validation structure (Tier A / B / C) in both
  `test_cmfmc_convection.jl` and `test_tm5_convection.jl`.
- Any legacy bugs found documented with source-of-truth hierarchy
  (paper > Fortran > legacy Julia) applied.

## 4.8 Performance (soft)

- `CMFMCConvection` overhead < 50% on GPU at medium grid (soft).
- `TM5Convection` overhead < 100% on GPU at medium grid (soft).
- No regression in non-convection paths (bit-exact default
  regression passes — Commit 6 test 4).

## 4.9 Documentation (hard)

- `ARCHITECTURAL_SKETCH_v4.md` committed (includes v5 runtime
  data-flow diagram).
- `position_study_results.md` with plots and recommendation.
- `validation_report.md` summarizing tier coverage.
- `notes.md` complete with legacy-bug section (if any bugs found),
  decisions beyond plan, surprises, interface findings.

---

# Part 5 — Known pitfalls

From accumulated plan-surprise patterns (CLAUDE.md + plans 11-17
retrospectives) plus plan-18-specific traps. Items 1-15 from v4,
new v5 items 16-18 flagged below.

1. **Porting TM5 matrix scheme "faithfully without reading the Fortran."**
   Legacy `tm5_matrix_convection.jl` may have bugs. Upstream
   `tm5_conv.F90` is authoritative. Per §2.12, Commit 0 reads and
   extends the upstream notes.

2. **Porting legacy scavenging code as commented-out.** No. Per §2.11,
   no scavenging code at all. Signature-ready inline helpers, no
   comment blocks with old algorithm.

3. **Making the DTRAIN-missing fallback a separate operator type.**
   No. Runtime path within `CMFMCConvection` via
   `forcing.dtrain === nothing`.

4. **Skipping Tier C "because it's hard."** No. Per §2.5, Tier C is
   required but opt-in via environment gate. If upstream Fortran
   isn't runnable, use published reference data.

5. **Treating legacy Julia as ground truth for Tier B tests.** No.
   Per §2.5, Tier B is against PAPER formulas, not legacy. Hand-expand
   from Tiedtke 1989 and Moorthi-Suarez 1992, not legacy code.

6. **Palindrome-internal convection because that's where plan 17 put
   emissions.** No, per §2.2 Decision 1. Plan 18 ships as separate
   block.

7. **Dynamic allocation in `apply!` for the matrix workspace.** No.
   Per §2.8, workspaces pre-allocated at `TransportModel` construction.

8. **Adding `InertTracer` as a vestigial type parameter.** No. No
   solubility parameters at all.

9. **Shipping adjoint kernels "since they're in legacy."** No. Plan 18
   preserves adjoint-able structure and ships the adjoint-identity
   test. Adjoint kernel itself is `Plan 19: Adjoint operator suite`.

10. **Loosening cross-scheme tolerance to 10% because "5% is strict."**
    No. With both operators run on same basis (§2.3), 5% is the
    correct tolerance.

11. **Skipping the well-mixed sub-cloud layer because legacy Julia
    skipped it.** No. Per §2.7, plan 18 ADDS it.

12. **"Four-term tendency is right, I'll restore it."** No. Per
    §2.12 finding 2, for inert tracers the two-term form is
    algebraically equivalent.

13. **Porting the positivity clamp from GCHP's `convection_mod.F90:1002-1004`.**
    **CRITICAL: NO.** Per §2.9. The clamp breaks linearity and the
    adjoint-identity test. Accept tiny negativities; rely on global
    mass fixer. Optional pre-step met-data validation (preprocessing,
    not kernel) if negativities become observable.

14. **Benchmark overhead "too high; optimize before shipping."** No.
    Plan 18 is correctness-first.

15. **Starting Commit 0 before PRE_PLAN_18_FIXES ship.** No. Commit 8
    (DrivenSimulation) needs A1's face-indexed apply! path working.
    A3's `current_time(sim)` fix prevents silent wrong-answer on any
    StepwiseField with `meteo = sim` (Decision 24). Ship prerequisites
    first.

16. **(New in v5) Assuming the convection operator takes a window.**
    NO — the operator takes a `ConvectionForcing` at its `apply!`
    signature, not an `AbstractTransportWindow` (§2.14 Decision 3
    revised). The `ConvectionForcing` reaches the operator via
    `TransportModel.convection_forcing`, populated by
    `DrivenSimulation._refresh_forcing!` (§2.17 Decision 23). At
    the call site the window is not in scope — only the
    model-side forcing.

17. **(New in v5) Passing `meteo = sim.driver` from the sim.** NO —
    Decision 24 changed this to `meteo = sim`. The driver is
    stateless (verified: `TransportBinaryDriver` struct has only
    `reader` + `grid`); it cannot supply `current_time`. The sim
    is the canonical time source. Code that needs driver
    capabilities uses `meteo.driver` or equivalent.

18. **(New in v5) Dispatching face-indexed rejection on a
    nonexistent mesh trait.** NO — `AbstractFaceIndexedMesh` does
    not exist in `src/Grids/`. The dispatch is on state shape:
    `Raw <: AbstractArray{_,3}` catches face-indexed
    `tracers_raw::(ncells, Nz, Nt)` vs 4D structured. This is
    grid-topology-agnostic.

19. **(New in v5) Porting convection forcing as a `TimeVaryingField`.**
    NO — Decision 22 keeps it as plain arrays in `ConvectionForcing`.
    The model-side slot (§2.17 Decision 23) handles the per-substep
    lifecycle directly, without the `AbstractTimeVaryingField`
    layer. This avoids dependency on `current_time(meteo)` for
    convection specifically and keeps the lifecycle uniform with
    `model.fluxes`.

20. **(New in v5.1) Assuming the all-nothing placeholder
    `ConvectionForcing()` on `TransportModel` will be populated
    automatically.** NO — per §2.20 Decision 26, allocation happens
    only at `DrivenSimulation` construction, after the first window
    is loaded. `with_convection(model, op)` does NOT allocate forcing
    buffers; tests that skip the sim layer and call `apply!` directly
    must either inject allocated forcing via `with_convection_forcing`
    or build `model.convection_forcing` by hand. `_refresh_forcing!`
    checks `has_convection_forcing(dst)` and is a no-op when the
    placeholder hasn't been replaced — so running `step!` on a model
    whose convection_forcing is the default placeholder silently
    skips the convection kernel. Sim construction catches this
    (Decision 28 validation + allocation); unit tests that bypass
    the sim must allocate explicitly.

21. **(New in v5.1) Letting `dtrain` toggle between nothing and
    populated across windows.** NO — per §2.21 Decision 27, capability
    (the set of non-nothing fields in `ConvectionForcing`) is
    invariant for the lifetime of a `DrivenSimulation`. The
    `dtrain === nothing` Tiedtke-style fallback (Decision 2) is
    allowed, but only if it applies to the WHOLE run. Preprocessing
    writes `include_dtrain=true` consistently or `include_dtrain=false`
    consistently — never per-window. `copy_convection_forcing!`
    enforces this with a strict capability-tuple match and errors
    loudly on any mismatch.

22. **(New in v5.1) Adding back `convection_mode` as a single-symbol
    trait.** NO — per §2.22 Decision 28, capabilities are independent
    traits (`supports_cmfmc`, `supports_dtrain`, `supports_tm5conv`).
    The single-symbol `convection_mode` from v5 had implicit
    `:cmfmc`-wins prioritization that can't describe dual-capability
    binaries. Any code that needs "is this a CMFMC-only path?"
    computes `supports_cmfmc(driver) && !supports_tm5conv(driver)`
    explicitly; don't re-introduce the aggregate.

---

# Part 6 — Follow-up plan candidates

Register at Commit 0 in `notes.md`; retrospective (Commit 11)
closes or defers each.

- **Plan 18b: Face-indexed convection.** Extend `apply!` and the
  array-level `apply_convection!` to face-indexed state shapes
  (3D `tracers_raw`, `(ncells, Nz)` air_mass). Kernel dispatches on
  cell-indexed rather than (i, j) pairs; column-internal physics
  unchanged. Decision 25 defers this; the error stub is the
  placeholder. Estimated: 1-2 weeks once structured is validated.

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
  Tiedtke 1989 or Grell-Freitas scheme. Would replace the window /
  model `ConvectionForcing` arrays with a derived type that recomputes
  on each window load. For sub-window-varying forcing, this is where
  the model-side slot pays for itself (real interpolation, not a
  reference copy).

- **Wet deposition plan.** Re-introduce `QC_PRES` / `QC_SCAV` split
  in updraft pass. Restore four-term tendency form for soluble
  tracers. Add solubility trait system (possibly `AbstractTracerSolubility`,
  `InertTracer`, `SolubleTracer`). Add precip flux fields to
  `ConvectionForcing`. Henry's-law hooks.

- **Multi-tracer fusion in TM5 matrix solve.** Build coefficients
  once per column, back-substitute Nt times. Currently solve per-tracer.

- **GPU-batched BLAS paths.** cuBLAS / rocBLAS `getrfBatched` +
  `getrsBatched` for TM5 matrix solve.

- **Shared-memory LU factorization.** For Nz=72+ columns, shared-memory
  LU may outperform register-based in-kernel version on GPU.

- **`AbstractLayerOrdering{TopDown, BottomUp}` abstraction.** Deferred
  from plan 17. Would clean up the per-kernel column-reversal dance.

- **Palindrome-internal convection.** If Commit 9 study suggests
  Config 4 benefits, ship as follow-up plan.

- **TiedtkeConvection as standalone type.** If `CMFMCConvection`
  DTRAIN-missing fallback gets heavily used, promote to its own type.

- **Plan 16c: retroactive Tier B/C validation for diffusion.** Plan
  16b shipped Tier A + shallow Tier B; no Tier C vs GCHP `vdiff_mod`
  or TM5 `diffusion.F90`. Scope should include basis audit. Apply
  plan 18's basis-audit discipline retroactively.

- **Sub-window-varying forcing infrastructure.** If the pattern of
  model-side forcing refresh generalizes (e.g., sub-window-varying
  emissions, time-varying Kz), extract `_refresh_forcing!` into a
  composable per-forcing-type interface.

---

# Part 7 — How to work

## 7.1 Session cadence

- Session 1: Commits 0-1 (NOTES + baseline + upstream survey + type
  hierarchy)
- Session 2: Commit 2 (ConvectionForcing + window and model slots)
- Session 3: Commit 3 Part 1 (CMFMCConvection scaffolding + Tier A)
- Session 4: Commit 3 Part 2 (kernel + sub-cycling + sub-cloud + Tier B)
- Session 5: Commit 3 Part 3 (Tier C + adjoint identity + completion)
- Session 6: Commit 4 Part 1 (TM5Convection scaffolding + matrix workspace + Tier A)
- Session 7: Commit 4 Part 2 (matrix kernel + Tier B)
- Session 8: Commit 4 Part 3 (Tier C + adjoint identity + completion)
- Session 9: Commit 5 (cross-scheme consistency)
- Session 10: Commit 6 (TransportModel step! orchestration)
- Session 11: Commit 7.1 + 7.2 (TransportBinary header + preprocessing)
- Session 12: Commit 7.3 + 7.4 (driver + e2e integration test)
- Session 13: Commit 8 (DrivenSimulation refresh + cache invalidation)
- Session 14: Commit 9 (position study)
- Session 15: Commits 10-11 (benchmarks + retrospective)

Longer than plans 15-17 because of:
- Two substantive physics ports.
- Three-tier validation per port.
- Cross-scheme consistency.
- Expanded Commit 7 (four sub-commits) for binary format extension.
- Expanded Commit 8 (sim-level refresh wiring).
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
  a bug.
- Commit 5 cross-scheme > 10% divergence. One operator has a bug.
- Commit 6 breaks bit-exact default-path regression. `NoConvection`
  dispatch isn't a compile-time dead branch.
- Commit 6 test 7 fails (operator reads window instead of
  model-side forcing). The data-flow contract (§2.17 Decision 23)
  is broken — investigate before proceeding.
- Commit 7 driver capability negotiation produces unclear errors
  when binary lacks CMFMC / DTRAIN. Clarify error message.
- Commit 8 runs but `_refresh_forcing!` doesn't propagate forcing
  changes across windows. Cache-invalidation hook is in the wrong
  place.
- Legacy bug found during port. Document, stop, confirm with user
  whether port-follows-paper or port-preserves-bug.
- Scope creep toward scavenging, derived fields, adjoint kernels,
  or face-indexed convection.

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
- Cross-scheme test outcomes.
- Adjoint-identity test numerics.
- Runtime data flow surprises (e.g., if `_refresh_forcing!` needed
  a real interpolation instead of a reference copy).

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
- **v4**: consolidation of v3 + corrections addendum v2 + adjoint addendum into single authoritative plan. Dropped Decisions 10 / 13 / 18 (replaced by Decision 20: basis follows state). Replaced `PreComputedConvMassFluxField` with `ConvectionForcing` on the window (Decision 22). Added Decision 21: mandatory CFL sub-cycling. Added adjoint-identity Tier A tests (adjoint addendum §A). Added no-positivity-clamp rule (adjoint addendum §D). Registered Plan 19 (adjoint suite) in follow-ups.
- **v5** (initial): runtime-data-flow correction based on GPT-5.4 architectural review. v4 assumed `apply!(state, window, ...)` — reality is that `DrivenSimulation` holds the window, and `TransportModel.fluxes` is populated per substep by `_refresh_forcing!`. v5 mirrors that pattern for convection: `ConvectionForcing` lives in BOTH the window (Decision 22 unchanged) AND the model (new Decision 23), with `_refresh_forcing!` copying per substep. `apply!` takes `ConvectionForcing` directly (not the window; Decision 3 revised). `meteo` kwarg is changed from `sim.driver` to `sim` (new Decision 24; also the corrected A3); `current_time(sim) = sim.time` replaces the impossible driver-level override. Face-indexed convection is scoped out via state-shape dispatch (new Decision 25). Commit 2 expanded to include the model-side field. Commit 6 now focuses on step orchestration. Commit 7 expanded to four sub-commits. Commit 8 expanded to include `_refresh_forcing!` extension. PRE_PLAN_18_FIXES A3 rewritten to target `current_time(::DrivenSimulation)`.
- **v5.1** (this document): post-review corrections from GPT-5.4 second pass. Three architectural gaps in v5:
  - **Allocation gap (HIGH):** v5's model-side `ConvectionForcing` defaulted to all-nothing placeholder; `with_convection` didn't fix it; `_refresh_forcing!` only copied when dst already had buffers. Net effect: allocation never happened, operator never saw forcing. Fixed by new **Decision 26** — `DrivenSimulation` constructor owns allocation. New helper `allocate_convection_forcing_like(src, backend_hint)` creates matching-capability `similar(...)` arrays on the right backend after the first window loads. `with_convection` continues to only swap the operator; allocation is a sim-construction step.
  - **dtrain invariance gap (HIGH):** v5 allowed `dtrain === nothing` Tiedtke-style fallback per operator AND used a permissive `copy_convection_forcing!` that silently skipped dtrain when src didn't have it. Combined with a preallocated dst that HAD dtrain, this left stale values live. Fixed by new **Decision 27** — capability is invariant per run; `copy_convection_forcing!` asserts exact capability match; preprocessing writes `include_dtrain` consistently for every window in the output binary.
  - **Dual-capability contract gap (MEDIUM):** v5's `convection_mode(driver) -> Symbol` couldn't describe a binary with both CMFMC+TM5 payloads even though preprocessing allowed writing both. Fixed by new **Decision 28** — independent capability traits `supports_cmfmc`, `supports_dtrain`, `supports_tm5conv`. `convection_mode` removed. Drivers can advertise both capabilities; sim selects which to consume based on operator type. A new reader trait `has_dtrain` distinguishes dtrain presence from CMFMC presence in the header.
  - **Commit 2** adds `allocate_convection_forcing_like` helper and tightens `copy_convection_forcing!` semantics.
  - **Commit 7.1** adds `has_dtrain`, `load_dtrain_window!` as first-class (not gated by `has_cmfmc`); header adds `include_dtrain :: Bool` with validation that `include_dtrain => include_cmfmc`.
  - **Commit 7.2** splits preprocessing flags: `include_cmfmc_in_output`, `include_dtrain_in_output`, `include_tm5conv_in_output`.
  - **Commit 7.3** replaces `convection_mode` with three independent traits; driver populates whichever payloads it has, capability-independently.
  - **Commit 7.4** adds tests for dual-capability binaries + dtrain presence/absence permutations (expanded from 10 to ~14 tests).
  - **Commit 8** adds sim-construction allocation path + operator ↔ capability validation + stricter window/model capability linkage.
  - **Pitfalls** 20, 21, 22 added covering these three traps.

Net effort vs v5: ~0.5 day (mostly in Commit 8's allocation /
validation path and Commit 7's preprocessing flag split). Total
plan 18 effort unchanged; v5.1 removes debug-the-missing-allocation
time from execution rather than adding it. No v5.1-specific
execution-ordering change — ship as if v5.1 were v5 from the start.

Source documents v3 base, corrections addendum v2, adjoint addendum,
v4 plan, and V5 architectural brief are moved to
`docs/plans/18_ConvectionPlan/archived/` by Commit 11 with the
`README.md` updated to reflect v5 as authoritative.

---

# End of Plan

After this refactor ships:

- `CMFMCConvection` in `src/Operators/Convection/` (GCHP path, dry- or moist-basis per state, structured-only).
- `TM5Convection` in `src/Operators/Convection/` (TM5 path, dry- or moist-basis per state, structured-only).
- Convection block positioned after transport, before chemistry in `step!`.
- `ConvectionForcing` as the per-step forcing container on `TransportModel`, populated per substep by `DrivenSimulation._refresh_forcing!` from `sim.window.convection`.
- `meteo = sim` as the canonical operator-time threading; `current_time(sim) = sim.time`.
- `TransportBinaryReader` supports convection payloads; preprocessing pipeline writes them.
- Three-tier validation completed (Tier A required, Tier B required, Tier C opt-in).
- Cross-scheme consistency verified (~5% same-basis agreement).
- CATRINE intercomparison possible for convection-sensitive tracers (Rn-222, tropical CO2, SF6 over sources).
- Full offline atmospheric transport model with emissions, transport, convection, diffusion, chemistry — operational.
- Adjoint structure preserved throughout; `Plan 19: Adjoint operator suite` can proceed mechanically.
- Face-indexed convection deferred to follow-up Plan 18b; error stub prevents accidental use.

This is the final structured-lat-lon physics operator needed for CATRINE. After plan 18, the model has the full operator suite for offline atmospheric transport on structured grids.
