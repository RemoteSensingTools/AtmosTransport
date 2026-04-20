# Convection Operator — Implementation Plan for Plan Agent (v1)

**Status:** Ready for execution after plan 17 has shipped.
**Target branch:** new branch from `surface-emissions` (or wherever
17's final tip lives). Verify in §4.1.
**Estimated effort:** 3-4 weeks, single engineer / agent.
**Primary goal:** Ship two concrete convection operators —
`CMFMCConvection` (GCHP path, CMFMC + DTRAIN fields) and
`TM5Convection` (TM5 path, four-field matrix scheme) — as
first-class operators conforming to the plan 16b/17 operator
interface. Validate physics against upstream Fortran references
(TM5 `tm5_conv.F90`, GEOS-Chem `convection_mod.F90`) and paper
equations (Tiedtke 1989, Moorthi & Suarez 1992), not just against
legacy Julia. Integrate at step level as a separate block after
the transport block.

Self-contained document. ~45 minutes to read.

**Dependencies verified:**
- Plan 11-17 shipped
- `AbstractTimeVaryingField` + concrete types (plans 16a/16b/17)
- `ImplicitVerticalDiffusion` + palindrome (plan 16b)
- `SurfaceFluxOperator` + palindrome integration (plan 17)
- `current_time(meteo)` threaded through `apply!` (plan 17 Commit 4)
- Plan 15 chemistry workaround removed from `DrivenSimulation`
  (plan 17 Commit 6)

**Companion documents:**
- `OPERATOR_COMPOSITION.md` — step-level composition; plan 18
  revises the convection-position decision
- `TIME_VARYING_FIELD_MODEL.md` — for convective mass flux
  field shapes
- `ARCHITECTURAL_SKETCH_v3.md` — current architecture
- `CLAUDE_additions_v2.md` — accumulated lessons including new
  § "Validation discipline for physics ports"
- Plan 17 NOTES — immediate predecessor, validation of palindrome
  + block composition pattern
- Plan 16b NOTES — adjoint-structure preservation pattern that
  plan 18 inherits
- `src_legacy/Convection/` — prior art (4 implementations,
  ~1,768 lines) as starting point, NOT as reference
- **TM5 Upstream Fortran (AVAILABLE):** `tm5_source/toupload/base/src/`
  — `tm5_conv.F90` (344 lines, matrix builder + solver),
  `convection.F90` (750 lines, driver), `tmphys_convec.F90`
  (138 lines, cloud dimensions), `phys_convec_ec2tm.F90`
  (497 lines, ECMWF→TM5 conversion). Authoritative for TM5 port.
- **GCHP Upstream Fortran (AVAILABLE):** `gchp_source/.../geos-chem/GeosCore/`
  — `convection_mod.F90` (2282 lines, RAS + Grell-Freitas schemes),
  `calc_met_mod.F90` (1677 lines, met field conventions).
  Authoritative for CMFMCConvection port.
- **`18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md`** — pre-populated
  TM5 Fortran reference notes with line-by-line algorithm breakdown.
- **`18_CONVECTION_UPSTREAM_GCHP_NOTES.md`** — pre-populated
  GCHP Fortran reference notes covering RAS algorithm, tendency
  derivation, and legacy-Julia comparison.
  
  Both documents extended during execution, final versions
  saved to `artifacts/plan18/`.

---

# Revisions note

**v3 (current version).** Three additional findings from deeper
basis-convention analysis and a design pattern discussion:

- **Finding 3: Dry vs moist basis differs by scheme AND by model
  convention.** GCHP uses dry-air basis throughout tracer
  transport (tracer in kg/kg dry air, BMASS = DELP_DRY × 100/g).
  TM5 uses moist-air basis throughout (tracer-in-moist mass,
  m = phlb_top - phlb_bot / g with phlb = total pressure). Legacy
  AtmosTransport has dry-air correction kernels
  (`latlon_dry_air.jl`, `cubed_sphere_mass_flux.jl`) that convert
  moist → dry BEFORE convection — matching GCHP convention when
  applied. **The comment in `ras_convection.jl:41-46` ("No dry
  conversion is applied before convection") appears to contradict
  the legacy's own dry-correction kernels and is likely stale.**
  Plan 18 documents this explicitly and makes basis expectations
  part of each operator's interface contract (§4.3 Decision 18).

- **Finding 4: Inline helper functions with implicit dispatch.**
  Rather than monolithic kernels with `# TODO` comments scattered
  through, plan 18 structures kernel math as named inline helpers
  (`_updraft_mix`, `_apply_tendency`, `_cloud_base_wellmix`)
  taking a `solubility` parameter. For plan 18 scope, this
  parameter is always implicitly inert and compiles away to
  identity. Future wet-deposition plan adds a method dispatch
  on a non-inert solubility trait; no kernel rewrite needed.
  This gets dispatch-friendly structure without shipping empty
  type hierarchies (§4.3 Decision 19).

- **Finding 5: Misleading legacy comment documented for cleanup.**
  The "No dry conversion is applied before convection" comment
  in `ras_convection.jl:41-46` is inconsistent with the actual
  legacy code (which DOES apply dry corrections via separate
  kernels). During plan 18 port, remove or correct this comment.

**v2.** Extended after reading GCHP `convection_mod.F90` and
deriving the relationship between GEOS-Chem's four-term tendency
and the legacy Julia's two-term tendency. See
`18_CONVECTION_UPSTREAM_GCHP_NOTES.md` §5 for the algebraic
derivation.

**Findings incorporated in v2:**

- **Finding 1: Legacy two-term tendency is mathematically correct
  for inert tracers.** Legacy `ras_convection.jl:188-193` uses a
  simplified two-term form that initially appeared to deviate
  from GEOS-Chem's four-term form. Careful substitution shows
  equivalence for inert tracers; the forms differ only when
  wet scavenging is active. Plan 18 adopts the two-term form.
  
  For FUTURE wet-deposition plan: the four-term form must be
  restored because QC_PRES ≠ old_QC breaks the algebra. Plan 18's
  `# TODO: scavenging hook` must be placed at BOTH the updraft
  pass AND the tendency pass.

- **Finding 2: Legacy is missing well-mixed sub-cloud layer
  treatment.** GCHP `convection_mod.F90:742-782` implements a
  pressure-weighted well-mixed layer below cloud base that
  legacy Julia lacks. Likely produces surface-concentration
  bias for CATRINE tracers with strong surface sources (Rn-222
  especially). Plan 18 Commit 3 ADDS this treatment as a
  deliberate improvement over legacy.

**v1 (initial version).** Written after plan 17's shipping and
after the plan-18 scoping conversation that clarified:

1. **No wet scavenging in plan 18 scope.** `AbstractTracerSolubility` /
   `InertTracer` / `SolubleTracer` infrastructure from legacy is
   out of scope. All tracers treated as inert. Placeholder
   `# TODO: scavenging hook` in kernels for future wet deposition
   plan.
2. **Cleaner kernels than legacy.** Legacy (`src_legacy/Convection/`)
   is a starting point with known code smell and may have bugs.
   Port follows paper equations and upstream Fortran; legacy is
   proximate reference only.
3. **Three-tier validation hierarchy.** Tier A (analytic), Tier B
   (paper/literature formulas), Tier C (cross-implementation
   comparison against upstream Fortran). See CLAUDE_additions_v2
   § "Validation discipline for physics ports".
4. **Two concrete types, not three.** `CMFMCConvection` and
   `TM5Convection`. `TiedtkeConvection` simplified path
   (single net CMFMC) deferred — `CMFMCConvection` with
   `dtrain = nothing` fallback covers the simpler case.
5. **Convection as separate block, not palindrome-centered.**
   `step!(model, dt)` runs transport_block → convection_block →
   chemistry_block. Rationale: convection is non-local (matrix
   transport), unlike diffusion (column-local) or emissions
   (point-local). Cleaner physics boundary.

---

# Part 1 — Orientation

## 1.1 The problem in two paragraphs

AtmosTransport needs convective transport for CATRINE intercomparison.
Two distinct data sources are supported: TM5 (ECMWF ERA5-driven,
provides four fields entu/detu/entd/detd from Tiedtke 1989) and
GCHP (GEOS-Chem High-Performance, provides CMFMC + DTRAIN from
MERRA-2, GEOS-FP, or GEOS-IT). These use different met fields
and different physics, but the tracer-transport operator must
provide a uniform interface either way.

Plan 18 ships two concrete convection operators —
`CMFMCConvection` (handles GCHP data via mass-flux redistribution
with updraft mixing) and `TM5Convection` (handles TM5 data via
full Nz×Nz transfer matrix with LU solve). Each takes its
required fields as `AbstractTimeVaryingField{FT, 3}` concrete
types. Both integrate at step level as a dedicated convection
block between transport and chemistry. Validation follows the
three-tier hierarchy (analytic / paper / cross-implementation)
with upstream Fortran as ground truth, not legacy Julia.

## 1.2 What plan 18 is really about

Four distinct pieces of work:

1. **Infrastructure:** `AbstractConvectionOperator` type hierarchy
   with `NoConvection` dead-branch default. Step-level composition
   as a separate block after transport. `TransportModel.convection`
   field + `with_convection` helper. `TimeVaryingField{FT, 3}`
   concrete types for convective mass flux fields.
2. **Physics: CMFMC/DTRAIN path (`CMFMCConvection`).** Port from
   paper equations (Moorthi-Suarez 1992 RAS) and upstream
   `convection_mod.F90` DO_CONVECTION. Legacy `ras_convection.jl`
   is starting point. Kernel cleaner than legacy; no scavenging
   branch. Adjoint-structure preserved.
3. **Physics: Four-field matrix path (`TM5Convection`).** Port
   from paper equations (Tiedtke 1989) and upstream
   `tm5_conv.F90`. Matrix builder + LU solve with pre-allocated
   workspace. Legacy `tm5_matrix_convection.jl` is starting point.
4. **Validation: three tiers.** Tier A (mass conservation,
   positivity, zero-forcing identity). Tier B (hand-expand paper
   formulas for 5-layer test column, compare port). Tier C
   (cross-reference against upstream Fortran output for standard
   test cases). Cross-scheme consistency test:
   `CMFMCConvection` and `TM5Convection` should agree to within
   ~5-10% on column-integrated tracer mass for matched forcing.

## 1.3 Scope keywords

- **Operator types:** `AbstractConvectionOperator`, `NoConvection`,
  `CMFMCConvection`, `TM5Convection`
- **Field types shipped:** `PreComputedConvMassFluxField{FT}`
  (for operational binary reads),
  `ConstantField{FT, 3}` (reuse from plan 16a for tests)
- **Composition:** step-level block after transport, before
  chemistry. NOT palindrome-centered.
- **Validation:** three-tier hierarchy (Tier A / B / C)

## 1.4 What plan 18 defers

- **Wet scavenging entirely.** `AbstractTracerSolubility`,
  `InertTracer`, `SolubleTracer`, Henry's law hooks — all out of
  scope. Kernels have `# TODO: scavenging hook here` placeholders.
- **TiedtkeConvection simplified path.** Degraded `CMFMCConvection`
  (with `dtrain = nothing` fallback per legacy) covers single-
  field case. Standalone `TiedtkeConvection` not shipped.
- **Derived convective mass flux fields.** Online computation of
  CMFMC/DTRAIN from parent meteorology (full Tiedtke or
  Grell-Freitas running live). Operational path reads
  pre-computed fields. Future plan for online computation.
- **Convection adjoint kernel.** Structure preserved (same as
  plan 16b), adjoint port mechanical when needed.
- **`AbstractLayerOrdering{TopDown, BottomUp}`** abstraction
  — deferred from plan 17 follow-ups. Convection kernels use
  `k=1=TOA, k=Nz=surface` convention per CLAUDE.md Invariant 2,
  with per-kernel column reversal at the interface (as legacy
  does).
- **Palindrome-centered convection.** Option to interleave
  convection inside the palindrome is discussed in §2.3 and
  deferred. Plan 18 ships as a separate block. If ordering study
  (Commit 8) shows a clear benefit from interleaving, a follow-up
  plan can move it.
- **Perf optimization beyond correctness.** Matrix LU solve is
  column-serial. GPU-batched BLAS or shared-memory factorization
  are future optimization plans.

---

# Part 2 — Why This Specific Change

## 2.1 What gets cleaner

Before:
- No convection at all in new `src/` (legacy `src_legacy/Convection/`
  untouched since the restructure)
- Met-field bridge (`met_fields_bridge.jl`) has CMFMC hints but
  no operator consumes them
- CATRINE intercomparison can't run because convective transport
  is absent
- Legacy code is unvetted — may have bugs that would poison
  intercomparison results

After:
- `CMFMCConvection` for GCHP data path
- `TM5Convection` for TM5 data path
- Both validated against paper equations AND upstream Fortran
  (not just legacy Julia)
- Three-tier test suite catches legacy-inherited bugs before
  CATRINE surfaces them
- Convection block positioned in `step!` sequence between
  transport and chemistry — matches TM5's operator-splitting
  pattern

## 2.2 What this enables

- **CATRINE intercomparison for convective-transport-sensitive
  tracers.** Rn-222 especially; its short half-life and
  ground-level source make convective vertical transport the
  dominant signal. CO2 over tropical convective regions. SF6
  to a lesser extent.
- **Comparison between ERA5-driven and MERRA-2-driven runs.**
  Both paths now supported; user selects met product and the
  right convection operator is wired automatically.
- **Future online convection.** `AbstractTimeVaryingField` slot
  for convective mass fluxes makes `DerivedConvMassFluxField`
  (online Tiedtke/Grell-Freitas computation) a mechanical
  addition.
- **Future wet deposition.** Scavenging hooks in the convection
  kernel mean adding wet deposition is a branch addition, not
  a kernel rewrite.

## 2.3 The composition question: block or palindrome?

Plan 17's ordering study showed that for emissions, palindrome
positioning (`V(dt/2) S V(dt/2)` vs `S V(dt)` vs `V(dt) S`)
mattered at the 3-4% relative level — small but detectable.
Plan 18 faces a similar question for convection.

**Option 1: Convection as separate block after transport.**

```
step!(model, dt):
  transport_block    → X Y Z V(dt/2) S V(dt/2) Z Y X  (plan 17)
  convection_block   → C(dt)
  chemistry_block    → chemistry(dt)
```

Rationale:
- Convection is non-local (matrix transport) — different
  operator character from column-local diffusion or point-local
  emissions
- Matches TM5's operator-splitting pattern (convection as
  separate phase)
- Simpler to reason about; no interleaving with palindrome
- If needed, can move inside palindrome later based on ordering
  study

**Option 2: Convection inside palindrome, wrapping C around S.**

```
transport_block:
  X Y Z V(dt/4) C(dt/2) V(dt/4) S V(dt/4) C(dt/2) V(dt/4) Z Y X
```

Rationale:
- Symmetric around S matches physics (fresh emissions lifted by
  convection before horizontal transport)
- Consistent with plan 17's V-around-S pattern
- But: C is non-linear (depends on column state), so halving C
  changes truncation error unpredictably

**Decision (§4.3 Decision 1): Option 1.** Ship convection as
separate block. Run ordering-position study (Commit 8) comparing
Option 1 to Option 2 to inform a potential follow-up reorganization.

Note on composition: the block boundary between transport and
convection is where TM5's operator splitting separates them,
and CATRINE's reference implementations generally follow this
structure. Going palindrome-internal would be a
physics-significant choice, not just a code-organization one —
worth validating before committing.

---

# Part 3 — Out of Scope

## 3.1 Do NOT touch

- Advection machinery (plan 14)
- Chemistry operator (plan 15)
- Diffusion operator or Kz field types (plan 16b)
- Surface emissions, `StepwiseField`, `SurfaceFluxOperator`
  (plan 17)
- `AbstractTimeVaryingField` or existing concrete types
- Palindrome structure itself (if Commit 8 ordering study
  suggests a change, document as follow-up plan candidate)
- `current_time(meteo)` threading — already done (plan 17 Commit 4)
- `DrivenSimulation` chemistry workaround — already removed
  (plan 17 Commit 6)

## 3.2 Do NOT add

- **Wet scavenging in any form.** No `AbstractTracerSolubility`,
  no `wet_scavenge_fraction`, no Henry's law. Kernels have
  `# TODO:` placeholders for future plan.
- **TiedtkeConvection as a separate concrete type.** Use
  `CMFMCConvection` with `dtrain = nothing` fallback per legacy.
- **Derived convective mass flux fields.** `PreComputedConvMassFluxField`
  for operational; tests use `ConstantField{FT, 3}`.
- **Online convection schemes.** Reading CMFMC from binary is
  fine. Computing CMFMC online from T/q/p is future plan.
- **Adjoint kernels.** Structure preserved per plan 16b; kernel
  port deferred.
- **Non-local (counter-gradient) corrections to convection.**
  Stay with bulk mass-flux redistribution.
- **Multi-tracer fusion optimization inside the matrix solve.**
  Solve per-tracer for now; fusion is a perf plan.
- **GPU-batched BLAS paths (cuBLAS getrfBatched).** KA kernel
  works on all backends; batched BLAS is future optimization.

## 3.3 Potential confusion — clarified

**Mass flux conventions.** CMFMC is at level interfaces
`(Nx, Ny, Nz+1)`; DTRAIN is at layer centers `(Nx, Ny, Nz)`.
TM5's entu/detu/entd/detd are all at layer centers `(Nx, Ny, Nz)`.
The `AbstractTimeVaryingField{FT, 3}` rank parameter is still
`N=3` for all — the shape differs within that. Concrete type
must document its expected shape.

**Layer ordering.** Codebase convention: `k=1=TOA, k=Nz=surface`
(CLAUDE.md Invariant 2). TM5's internal algorithm uses
`k=1=surface, k=lm=TOA`. Port reverses columns at the interface,
same as legacy does (`tm5_matrix_convection.jl`:17). No
`AbstractLayerOrdering` abstraction in plan 18.

**Moist vs dry basis.** CMFMC and DTRAIN from GEOS are on MOIST
(total air) basis, as is DELP. Convective transport works on
moist basis; no dry conversion applied before convection.
Matches GeosChem (`calc_met_mod.F90`). Documented in kernel
header comment.

**Offline vs online.** Plan 18's convection operator consumes
pre-computed mass flux fields from meteorology (offline path).
Met-field bridge already reads these (see
`met_fields_bridge.jl`). Computing mass fluxes online from T/q/p
is a separate future plan.

**CMFMCConvection DTRAIN-missing fallback.** Per legacy
`ras_convection.jl`:34-36, if DTRAIN is unavailable at runtime,
the operator falls back to Tiedtke-style CMFMC-only transport.
Plan 18 preserves this fallback. Not a separate concrete type —
it's a runtime path within `CMFMCConvection`.

**GEOS-FP June 2020 discontinuity.** GEOS-FP switched from RAS
to Grell-Freitas on 1 June 2020 (near-doubling of CMFMC values).
This is a DATA problem, not an algorithm problem — the operator
consumes whatever CMFMC+DTRAIN the met driver provides.
Documented in user-facing docs; NOT plan 18's concern.

---

# Part 4 — Implementation Plan

## 4.1 Precondition verification

```bash
# 1. Determine parent branch
git branch -a | head -20
git log --oneline --all | grep -i "plan 17\|surface-emissions" | head -10
git checkout <parent-branch>
git pull
git log --oneline | head -25
# Expected: plans 11-17 commits visible

git checkout -b convection

# 2. Clean working tree
git status

# 3. Verify dependency state
grep -c "SurfaceFluxOperator\|NoSurfaceFlux\|StepwiseField" src/ \
    --include="*.jl" -r
# Expected: non-zero (plan 17)
grep -c "apply_surface_flux!" src/ --include="*.jl" -r
# Expected: non-zero (plan 17 Commit 3 array-level entry)
grep -c "current_time(meteo)" src/Operators/ --include="*.jl" -r
# Expected: non-zero (plan 17 Commit 4)
grep "with_chemistry.*NoChemistry" src/Models/DrivenSimulation.jl
# Expected: NO match (plan 17 Commit 6 removed)

# 4. Survey existing convection code (the BIG one)
ls -la src_legacy/Convection/
wc -l src_legacy/Convection/*.jl
# Expected: 7 files, ~1,768 lines total
grep -rn "conv\|convection\|tiedtke\|ras\|cmfmc" src/ \
    --include="*.jl" | tee artifacts/plan18/existing_convection_survey.txt
# Expected: mostly references in comments / met field bridge,
# no active implementation in new src/

# 5. Read upstream Fortran references
# Plan 18 §4.3 Decision 14 requires reading these BEFORE Commit 1:
#
# TM5 (AVAILABLE): tm5_source/toupload/base/src/
#   - tm5_conv.F90 lines 37-191 (matrix builder TM5_Conv_Matrix)
#   - tm5_conv.F90 lines 197-341 (solver TM5_Conv_Apply)
#   - convection.F90 lines 227-747 (driver convec)
#   - tmphys_convec.F90 lines 39-132 (ConvCloudDim)
#   - phys_convec_ec2tm.F90 lines 82-400 (EC→TM5 field conversion,
#     explains sign/level conventions)
#
# Pre-populated notes exist at
# 18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md — extend with
# discoveries during Commit 0.
#
# GEOS-Chem (TO BE OBTAINED):
#   - github.com/geoschem/geos-chem, GeosCore/convection_mod.F90
#     DO_CONVECTION routine
#   - calc_met_mod.F90 for met field conventions
# Save annotated notes to artifacts/plan18/upstream_fortran_notes.md
# noting any discrepancies with legacy Julia ports.

# 6. Survey convective mass flux plumbing
grep -rn "conv_mass_flux\|CMFMC\|DTRAIN\|entu\|detu\|entd\|detd" src/ \
    --include="*.jl" | tee artifacts/plan18/existing_conv_mf_survey.txt

# 7. Capture baseline
for testfile in $(ls test/*.jl | sort); do
    echo "=== $testfile ==="
    julia --project=. $testfile 2>&1 | tail -20
done | tee artifacts/plan18/baseline_test_summary.log

# 8. Record baseline
git rev-parse HEAD > artifacts/plan18/baseline_commit.txt
mkdir -p artifacts/plan18/perf artifacts/plan18/validation

# 9. Memory compaction
# Update MEMORY.md with plan 17 shipped note
# Add plan 17 completion summary to planNN_complete.md
```

If preconditions fail, STOP.

## 4.2 Change scope — expected file list

**Files to ADD (new):**

Field types:
- `src/State/Fields/PreComputedConvMassFluxField.jl` — thin
  wrapper around `AbstractArray{FT, 3}` for convective mass flux
  fields. Parallels `PreComputedKzField` (plan 16b) but named
  explicitly for convection context.

Convection operator hierarchy:
- `src/Operators/Convection/` — new directory
- `src/Operators/Convection/Convection.jl` — module file
- `src/Operators/Convection/AbstractConvectionOperator.jl`
- `src/Operators/Convection/NoConvection.jl`
- `src/Operators/Convection/CMFMCConvection.jl` — GCHP path
- `src/Operators/Convection/cmfmc_kernels.jl` — KA kernel for
  CMFMC/DTRAIN mass-flux redistribution
- `src/Operators/Convection/TM5Convection.jl` — TM5 path
- `src/Operators/Convection/tm5_matrix_kernels.jl` — KA kernel
  for TM5 matrix builder + LU solve
- `src/Operators/Convection/convection_workspace.jl` — scratch
  storage (matrix workspace for TM5, pivots, etc.)

Tests:
- `test/test_convection_types.jl` — operator type hierarchy
- `test/test_cmfmc_convection.jl` — Tier A, B, C tests for GCHP
  path
- `test/test_tm5_convection.jl` — Tier A, B, C tests for TM5 path
- `test/test_convection_cross_scheme.jl` — consistency between
  the two operators for matched forcing
- `test/test_convection_palindrome_position.jl` — ordering study
- `test/test_transport_model_convection.jl` — TransportModel
  wiring

Benchmarks:
- `scripts/benchmarks/bench_convection_overhead.jl`

Validation artifacts:
- `artifacts/plan18/validation/tier_a_*.log`
- `artifacts/plan18/validation/tier_b_*.log`
- `artifacts/plan18/validation/tier_c_*.log`
- `artifacts/plan18/validation/cross_scheme_*.log`
- `artifacts/plan18/upstream_fortran_notes.md` — Fortran
  divergence notes

Docs:
- `docs/plans/18_CONVECTION_PLAN/NOTES.md`
- `docs/plans/18_CONVECTION_PLAN/position_study_results.md`
- `docs/plans/18_CONVECTION_PLAN/validation_report.md`

**Files to MODIFY:**

- `src/AtmosTransport.jl` — include Convection module, export
  types
- `src/Operators/Operators.jl` — include order:
  `Diffusion → SurfaceFlux → Convection → Advection → Chemistry`
  (Convection module is independent of Advection and Chemistry
  submodules; no cross-dep issues)
- `src/Models/TransportModel.jl` — add `convection::ConvT` field
  default `NoConvection()`, add `with_convection` helper. `step!`
  runs transport_block → convection_block → chemistry_block.
- `src/MetDrivers/AbstractMetDriver.jl` — no API changes; concrete
  drivers already provide convective mass flux via
  `met_fields_bridge.jl`. Plan 18 operators consume these via
  `PreComputedConvMassFluxField` wrappers.

**Files NOT to modify:**

- `src/Operators/Advection/StrangSplitting.jl` — palindrome
  unchanged
- `src/Models/DrivenSimulation.jl` — delegates to
  `TransportModel.step!`; convection rides through automatically
- `src/State/CellState.jl` — no storage changes

## 4.3 Design decisions (pre-answered)

Every decision final. If ambiguous, STOP and ask.

**Decision 1: Convection as separate block, not palindrome-centered.**

```
step!(model, dt):
  transport_block    → plan 17 palindrome
  convection_block   → C(dt)
  chemistry_block    → chemistry(dt)
```

See §2.3 rationale. Ordering position study in Commit 8 validates
this choice against palindrome-internal alternatives.

**Decision 2: `AbstractConvectionOperator` hierarchy.**

```julia
abstract type AbstractConvectionOperator end

struct NoConvection <: AbstractConvectionOperator end

struct CMFMCConvection{FT, MF, DT} <: AbstractConvectionOperator
    cmfmc::MF          # ::AbstractTimeVaryingField{FT, 3}, shape (Nx, Ny, Nz+1)
    dtrain::DT         # ::Union{AbstractTimeVaryingField{FT, 3}, Nothing}
                       # shape (Nx, Ny, Nz) when present
    # Runtime fallback: when dtrain === nothing, reduces to
    # Tiedtke-style single-flux transport (legacy behavior)
end

struct TM5Convection{FT, EU, DU, ED, DD} <: AbstractConvectionOperator
    entu::EU           # ::AbstractTimeVaryingField{FT, 3}, (Nx, Ny, Nz)
    detu::DU
    entd::ED
    detd::DD
    lmax_conv::Int     # maximum level for convection (0 = use full Nz)
end
```

Each concrete type carries exactly the fields it needs. Type
parameters `MF, DT, EU, DU, ED, DD` allow field backing to vary
(ConstantField, PreComputedConvMassFluxField, future
DerivedConvMassFluxField).

**Decision 3: `apply!` signature matches plan 15/16b/17 pattern.**

```julia
apply!(state::CellState{FT},
       meteo,
       grid::AbstractGrid,
       op::AbstractConvectionOperator,
       dt::Real;
       workspace) -> state
```

`meteo` required for `current_time(meteo)` call (plan 17 pattern);
test helpers may pass `meteo = nothing` if fields are `ConstantField`
(update_field! is no-op).

Workspace: `CMFMCConvection` requires only small column-scratch
(updraft concentration cache). `TM5Convection` requires the matrix
workspace `conv1_ws::Array{FT, 4}` of shape `(lmax, lmax, Nx, Ny)`.
Plan 18 introduces a new `ConvectionWorkspace` struct carrying
these allocations; `TransportModel` extends `AdvectionWorkspace`
to include it.

**Decision 4: Array-level entry point mirrors plans 16b/17.**

```julia
apply_convection!(q_raw, op::NoConvection, ws, dt, meteo, grid) = nothing

apply_convection!(q_raw, op::CMFMCConvection, ws, dt, meteo, grid)
    # ...

apply_convection!(q_raw, op::TM5Convection, ws, dt, meteo, grid)
    # ...
```

Called from `step!` convection block on `state.tracers_raw`
directly. State-level `apply!` delegates.

**Decision 5: Mass flux fields are `TimeVaryingField{FT, 3}`.**

Concrete type for operational path:

```julia
struct PreComputedConvMassFluxField{FT, A} <: AbstractTimeVaryingField{FT, 3}
    data::A       # ::AbstractArray{FT, 3}, caller-owned storage
end

# update_field!(f, t) — no-op (caller refreshes data from met driver)
# field_value(f, (i, j, k)) — direct indexed read
# Adapt.adapt_structure converts backing array to device type
```

Parallels `PreComputedKzField` from plan 16b. CATRINE configs
wrap met-driver arrays in this type at simulation setup.

Tests use `ConstantField{FT, 3}` (from plan 16a) for uniform
mass flux configurations.

`DerivedConvMassFluxField` (online computation from T/q/p) is
deferred to a future plan.

**Decision 6: TM5 matrix storage — per-column in 4D workspace.**

Legacy uses `conv1_ws[row, col, i, j]` of shape `(lmax, lmax, Nx, Ny)`.
Plan 18 preserves this layout. Matrix is diagonal-dominant, no
pivoting needed.

Storage allocation: `ConvectionWorkspace` holds
`conv1_ws::Array{FT, 4}` and `pivot_ws::Array{Int, 3}`. Sized
for `lmax_conv` (or Nz if `lmax_conv = 0`). Adapt.adapt_structure
converts to device arrays.

**Decision 7: Kernel structure — one KA kernel per (column,
scheme).**

- `_cmfmc_convection_kernel!(q_raw, cmfmc_data, dtrain_data, delp,
  Nz, Nt, dt, grav)` — one thread per (i, j), loops k and t_idx
  inside.
- `_tm5_conv_build_factorize_kernel!(conv1_ws, entu, detu, entd,
  detd, delp, Nz, lmax, dt, grav)` — builds matrix, does Gaussian
  elimination.
- `_tm5_conv_solve_kernel!(q_raw, conv1_ws, Nz, Nt, lmax)` —
  forward/back substitution per tracer.

Two-kernel split for TM5 allows matrix reuse across tracers
(build once, solve Nt times). Plan 18 ships this split; fusion
is a perf plan.

**Decision 8: Scavenging hooks via dispatch-ready inline helpers.**

Per Decision 19, kernel math lives in named inline helpers
(`_cmfmc_updraft_mix`, `_cmfmc_apply_tendency`,
`_cmfmc_wellmix_subcloud`, `_tm5_scavenging_matrix`). Plan 18
ships helpers with inert-only signatures. Scavenging is a
future-plan addition as a new method dispatch, not a kernel
rewrite.

**Hook sites are function signatures**, not `# TODO:` comments
inside kernel bodies. More discoverable. Each helper's docstring
lists which future parameter the scavenging overload will take.

For `CMFMCConvection`, two helper hook sites:

- `_cmfmc_updraft_mix`: future overload returns
  `(qc_pres, qc_scav)` split per solubility; inert version
  returns `(qc, 0)`.
- `_cmfmc_apply_tendency`: future overload uses four-term form
  when `qc_pres != qc_post_mix`; inert version uses two-term
  equivalent (GCHP_NOTES §5.3).

For `TM5Convection`, one helper hook site:

- `_tm5_scavenging_matrix`: future overload computes `lbdcv`
  from `cp, sceffdeep, fu` per Fortran `tm5_conv.F90:154-165`;
  inert version returns zero matrix (can be omitted entirely
  via dispatch).

Plan 18 ships NO `AbstractTracerSolubility` type hierarchy (per
Decision 16). Future wet-deposition plan adds the trait and new
helper methods.

**Decision 9: Validation tier structure.**

Per CLAUDE_additions_v2 § "Validation discipline for physics
ports". Each scheme's test file includes:

- **Tier A — Analytic:** 3-5 tests
  - Mass conservation: Σq after step == Σq before (machine precision)
  - Zero forcing identity: with all mass fluxes = 0, state
    unchanged bit-exact
  - Positivity: non-negative q in → non-negative q out
  - Stability: no growth under repeated application with fixed
    fluxes
- **Tier B — Paper formulas:** 5-10 tests
  - Hand-expand paper equations for 5-layer idealized column
  - Compute expected tendency / matrix entries by hand
  - Compare port output to hand expansion (<ULP agreement for
    linear ops)
- **Tier C — Cross-implementation:** 1-3 tests
  - Standard "deep tropical convective column" test case
    (specify CMFMC profile peaking at 500 hPa, DTRAIN detraining
    at cloud top, corresponding entu/detu/entd/detd for TM5)
  - Compare port output to upstream Fortran reference (if
    runnable) OR published reference calculation
  - Agreement expected ~5-10% on column-integrated mass, similar
    qualitative vertical profile

**Decision 10: Cross-scheme consistency test.**

New test file `test/test_convection_cross_scheme.jl`:
- Idealized deep-convective column (specified CMFMC profile on
  DRY basis)
- Derive equivalent entu/detu/entd/detd from CMFMC+DTRAIN via
  documented relationship (per GCHP_NOTES §11):
  ```
  entu(k) = max(0, CMFMC(k) + DTRAIN(k) - CMFMC(k-1))
  detu(k) = DTRAIN(k)
  entd(k) = 0   # no downdraft in GCHP path
  detd(k) = 0   # no downdraft in GCHP path
  ```
- **Both operators run on the same (dry) basis** per Decision 18
  Option A: `TM5Convection` receives the dry-basis entu/detu
  (contra its usual moist-basis contract) for the cross-scheme
  test mode. Cleanly isolates the discretization difference
  (explicit vs implicit) from the basis difference.
- Run `CMFMCConvection` on (cmfmc, dtrain) and `TM5Convection`
  on derived (entu, detu, entd, detd) — all dry-basis.
- **Tolerance: ~5% relative on column-integrated mass.** Rationale:
  with basis mismatch eliminated (Option A), the remaining
  disagreement is from:
  - Explicit sub-step (GCHP) vs implicit matrix solve (TM5):
    ~2-3% typically at standard CATRINE dt
  - Sub-cloud layer treatment (in CMFMCConvection, not in TM5):
    ~1-2% for surface-source tracers
  Total ~5% is the correct tolerance with same-basis inputs.
- Vertical profile: similar shape, maximum in mid-troposphere,
  tracer lofted from surface to cloud-top detrainment layer

Passing this test is strong evidence that both operators
implement consistent physics. Failing it (>5% divergence)
means at least one has a bug beyond the known discretization
differences.

**Note for tests on native basis:** separate Tier C tests
(Commit 3 and Commit 4) run each operator on its NATIVE
convention (GCHP dry / TM5 moist) against upstream Fortran
output. Those tests verify the port correctness within each
ecosystem. The cross-scheme test is additional, specifically
targeting inter-operator consistency.

**Decision 11: Adjoint-structure preservation.**

Per plan 16b pattern. Both kernels write coefficients as named
locals at each level, not pre-factored. TM5 matrix is explicit
(`conv1_ws[row, col, i, j]`), making transpose mechanical.
`CMFMCConvection` kernel's mass-flux coefficients are explicit
(`cmfmc_k, dtrain_k, entrain_k`).

Adjoint kernel not shipped in plan 18. Documented in header
comments of each kernel file: transposition rule for future
adjoint port.

**Decision 12: `lmax_conv` handling for TM5.**

Legacy `TM5MatrixConvection(; lmax_conv=0)` uses `0` as sentinel
for "use full Nz." Plan 18 preserves this.

Constructor:

```julia
function TM5Convection{FT}(; entu, detu, entd, detd, lmax_conv::Int = 0) where FT
    # Validate field types are TimeVaryingField{FT, 3}
    # ...
end
```

Outer convenience: `TM5Convection(entu, detu, entd, detd; lmax_conv=0)`
infers FT from the field types.

**Decision 13: Moist basis retained throughout.**

Per legacy convention (`ras_convection.jl`:41-46). CMFMC, DTRAIN,
DELP all on moist basis; no conversion in the convection operator.
Matches GeosChem. Document in kernel headers.

**Decision 14: Read upstream Fortran before writing Julia.**

Per CLAUDE_additions_v2 § "Validation discipline". Commit 0's
survey MUST include reading:

**TM5 (AVAILABLE):**
- `tm5_conv.F90`:37-191 matrix builder (TM5_Conv_Matrix)
- `tm5_conv.F90`:197-341 solver (TM5_Conv_Apply)
- `convection.F90`:227-747 driver (convec)
- `tmphys_convec.F90`:39-132 cloud dimensions (ConvCloudDim)
- `phys_convec_ec2tm.F90`:82-400 field conversions (explains
  sign/level conventions)

Pre-populated Fortran reference notes at
`18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md`. **Commit 0 extends
this** with any additional findings, then saves final version
to `artifacts/plan18/upstream_fortran_notes.md`.

**Pre-verified finding from initial spot-check (see notes §8):**
Legacy Julia `tm5_matrix_convection.jl:60-122` appears faithful
to Fortran `tm5_conv.F90:37-191` after accounting for 1-based
index shift. Matrix-builder math is correct. Port can proceed
with light cleanup confidence. This does NOT exempt plan 18
from Tier B/C validation — driver logic, column reversal, and
kernel boundary handling remain unverified.

**GEOS-Chem (AVAILABLE):**
- `convection_mod.F90` (2282 lines) — both RAS and Grell-Freitas
  schemes. DO_RAS_CLOUD_CONVECTION at lines 422-1419.
- `calc_met_mod.F90` (1677 lines) — met field conventions.

**Key findings from initial GCHP analysis (see GCHP_NOTES):**

1. Legacy Julia's simplified two-term tendency is
   mathematically equivalent to GCHP's four-term form for
   inert tracers (GCHP_NOTES §5.3). Light cleanup OK.
2. Legacy Julia MISSES the well-mixed sub-cloud layer
   treatment present in GCHP (`convection_mod.F90:742-782`).
   Plan 18 ADDS this (Commit 3 scope).
3. Both RAS and GF consume the same CMFMC + DTRAIN interface.
   No separate operator type needed for GF.
4. GCHP uses DRY air basis (BMASS = DELP_DRY * 100/g); TM5
   uses MOIST basis. ~1-3% systematic difference in
   cross-scheme comparison.

This is NOT optional — physics ports require upstream validation
to satisfy CLAUDE_additions_v2 validation discipline.

**Decision 15: Cleanup aggressiveness.**

Per plan 18 scoping conversation:
- `CMFMCConvection`: medium cleanup. Legacy RAS (~400 lines) has
  room for cleaner organization. Modernize variable names,
  separate "compute coefficients" from "apply coefficients" into
  distinct internal helpers, use `field_value` for mass flux
  reads. Split compound expressions for clarity.
- `TM5Convection`: light cleanup. Matrix algorithm has subtle
  invariants (column reversal, diagonal dominance). Stay close
  to legacy structure but use modern naming and TimeVaryingField
  for inputs. Port the matrix builder faithfully with comments
  linking to Fortran line numbers.

**Decision 16: No scavenging infrastructure.**

Per plan 18 scoping conversation: no `AbstractTracerSolubility`,
no `InertTracer`/`SolubleTracer` types, no `wet_scavenge_fraction`.
Kernels have `# TODO: scavenging hook here` at future-insertion
sites but do not parameterize on solubility traits.

**Decision 17: Well-mixed sub-cloud layer — ADD vs legacy.**

GCHP `convection_mod.F90:742-782` implements a pressure-weighted
well-mixed treatment below the cloud base:

```fortran
! Below cloud base: well-mixed treatment
IF (CLDBASE > 1 .AND. CMFMC(CLDBASE-1) > TINYNUM) THEN
    ! Weighted avg Q below cloud base
    QB = sum(Q(1:CLDBASE-1) * DELP_DRY(1:CLDBASE-1)) / sum(DELP_DRY(1:CLDBASE-1))
    MB = sum(DELP_DRY(1:CLDBASE-1)) * 100/g
    ! Mix with updraft flux from cloud base
    QC = (MB*QB + CMFMC(CLDBASE-1) * Q(CLDBASE) * SDT) / (MB + CMFMC(CLDBASE-1) * SDT)
    ! Apply uniformly to sub-cloud layers
    Q(1:CLDBASE-1) = QC
ENDIF
```

**Legacy Julia `ras_convection.jl` does NOT implement this.**

**Plan 18 ADDS this to `CMFMCConvection`**, as a deliberate
improvement over legacy. Rationale:

- For surface-source tracers (Rn-222, CO2 in tropics, SF6 over
  source regions), the sub-cloud well-mixed layer is essential
  for correct convective tracer transport
- Without it, surface concentrations are biased HIGH (source
  emissions don't redistribute before convection lifts them)
  and free-troposphere concentrations are biased LOW (less
  well-mixed source air entering updraft)
- Rn-222 is particularly sensitive (short half-life ~3.8 days,
  strong land-surface source, tropical convective lofting is
  the dominant vertical transport pathway)

Commit 3 Tier C test verifies the improvement: port WITH
sub-cloud fix should agree better with GCHP output than port
WITHOUT (and better than legacy Julia).

This is plan 18's first "legacy fix" — the validation discipline
surfaced it during Commit 0 survey, and plan 18 fixes it rather
than preserving the bug.

When wet deposition plan ships (future), it will add:
- Solubility trait system (possibly different from legacy shape)
- Parameter on `CMFMCConvection` / `TM5Convection` for
  per-tracer solubility (likely `NTuple{N, AbstractTracerSolubility}`
  or similar)
- Kernel branches activated by the new solubility parameters

Plan 18's structure should make this addition mechanical.

**Decision 18: Operator basis expectations (dry vs moist).**

Each operator's interface contract documents which basis it
expects. Meteorology layer is responsible for conversions
upstream.

**`CMFMCConvection` expects DRY-basis fields.**
- `cmfmc`: dry-corrected CMFMC [kg dry-air / m² / s]
- `dtrain`: dry-corrected DTRAIN [kg dry-air / m² / s]
- `delp`: dry pressure thickness `DELP_DRY` [Pa]
- Tracer storage: kg of tracer per kg of dry air in the layer
  (via `state.tracers_raw` / `state.air_mass_dry`)

Matches GCHP convention. Apply dry-air correction
(`cmfmc_dry = cmfmc_moist × (1 - qv_interface)`, etc.) upstream
via `latlon_dry_air.jl` / `cubed_sphere_mass_flux.jl` equivalents
if consuming raw GEOS met.

**`TM5Convection` expects MOIST-basis fields.**
- `entu`, `detu`, `entd`, `detd`: moist-basis mass fluxes
  [kg moist-air / m² / s]
- `m`: moist air mass per layer [kg]; `m = (phlb_top - phlb_bot)/g × area`
- Tracer storage: kg of tracer per moist-air mass (via native
  TM5 convention)

Matches TM5 convention. No dry correction upstream.

**Docstring contract:** each operator's constructor docstring
states the basis expectation explicitly:

```julia
"""
    CMFMCConvection(cmfmc, dtrain; lmax_conv=0)

GEOS-Chem RAS/Grell-Freitas convective transport on DRY-AIR BASIS.

# Field expectations

- `cmfmc::TimeVaryingField{FT, 3}`: dry-corrected CMFMC at
  interfaces (Nx, Ny, Nz+1). Units kg dry-air / m² / s.
- `dtrain::Union{TimeVaryingField{FT, 3}, Nothing}`: dry-corrected
  DTRAIN at centers (Nx, Ny, Nz). Units kg dry-air / m² / s.

If fields are from raw GEOS met (moist basis), apply dry-air
correction upstream via `apply_dry_cmfmc!` / `apply_dry_dtrain!`
before wrapping in `PreComputedConvMassFluxField`.

Tracer state assumed dry-basis (kg tracer per kg dry air).

# See also

- `TM5Convection`: moist-basis equivalent
- GCHP_NOTES.md §4 for basis derivation
"""
```

**Legacy cleanup item:** the comment in `ras_convection.jl:41-46`
("No dry conversion is applied before convection") is inconsistent
with the legacy's own dry-correction kernels and should be
removed or corrected during port. Flag in Commit 3 NOTES.

**Implications for cross-scheme test (Decision 10):**

Cross-scheme consistency test (Commit 5) must pass matched
physical forcing, not matched raw numbers. Either:
- Option A: pass dry-corrected fields to BOTH operators (convert
  entu/detu/entd/detd to dry via `(1-qv_interface)` scaling)
- Option B: pass native moist fields to BOTH (no dry correction;
  `CMFMCConvection` gets moist CMFMC/DTRAIN contra its usual
  contract — document as "cross-scheme test mode")

Recommend Option A: closer to real-use conditions, operator
contracts are respected. Tolerance stays at ~7-10% due to
discretization (explicit sub-step vs implicit matrix) but the
~1-3% basis difference is eliminated.

**Decision 19: Inline helpers with implicit dispatch for future
scavenging.**

Rather than monolithic kernels with `# TODO` comments scattered
through, plan 18 structures kernel math as named inline helpers
that take an implicit scavenging argument. For plan 18 scope,
the argument is absent (inert default); adding scavenging becomes
a method dispatch.

**Recommended structure:**

```julia
# --- CMFMCConvection kernel helpers ---

@inline function _cmfmc_updraft_mix(qc_below, q_env, cmfmc_below,
                                    entrn, cmout, tiny)
    # INERT VERSION: returns (qc_post_mix, zero_scav)
    if cmout > tiny
        qc = (cmfmc_below * qc_below + entrn * q_env) / cmout
    else
        qc = q_env
    end
    return qc, zero(qc)    # qc_pres, qc_scav (scav=0 for inert)
end

@inline function _cmfmc_apply_tendency(q_env, q_above, qc_post_mix,
                                       qc_pres, cmfmc_below, cmfmc_above,
                                       dtrain, bmass, dt)
    # INERT VERSION: two-term form (GCHP_NOTES §5.3 equivalent)
    tsum = cmfmc_above * (q_above - q_env) +
           dtrain * (qc_post_mix - q_env)
    return q_env + (dt / bmass) * tsum
end

@inline function _cmfmc_wellmix_subcloud(q_slice, delp_slice,
                                         q_cldbase, cmfmc_at_base, dt)
    # Per GCHP convection_mod.F90:742-782
    qb_num = zero(eltype(q_slice))
    mb_num = zero(eltype(q_slice))
    for k in eachindex(q_slice)
        qb_num += q_slice[k] * delp_slice[k]
        mb_num += delp_slice[k]
    end
    qb = qb_num / mb_num
    mb = mb_num  # already in units of dry mass (after G0_100 factor)
    return (mb * qb + cmfmc_at_base * q_cldbase * dt) /
           (mb + cmfmc_at_base * dt)
end
```

**Future scavenging plan adds methods:**

```julia
# FUTURE — when wet deposition plan adds solubility parameter:
@inline function _cmfmc_updraft_mix(qc_below, q_env, cmfmc_below,
                                    entrn, cmout, tiny,
                                    sol::SolubleTracer, F, ...)
    # Full QC_PRES / QC_SCAV split per Fortran convection_mod.F90
    ...
end

@inline function _cmfmc_apply_tendency(...,
                                       sol::SolubleTracer, ...)
    # Four-term form — necessary when QC_PRES != old_QC
    ...
end
```

**For plan 18 execution:**
- Ship the helpers with inert-only signatures
- Docstring each helper with "FUTURE: scavenging overload adds
  solubility parameter" comment
- No `AbstractTracerSolubility` type shipped (per Decision 16)
- Future plan adds the type hierarchy and new methods; no
  rewrite of plan 18's inline helpers

**Benefits:**
- Cleaner kernel body (math logic in named helpers; dispatch
  ready for future)
- Scavenging hook sites are function signatures, not comments —
  more discoverable
- No empty type hierarchies cluttering plan 18
- `@inline` gives same performance as monolithic kernel
- Helper functions can be unit-tested independently (Tier A/B
  pure-function tests)

**Applies to `TM5Convection` too:** matrix builder has one
scavenging hook site at `lbdcv` assembly (Fortran lines 154-165).
Helper: `_tm5_scavenging_matrix` returning `lbdcv = 0` for inert,
computed from `cp, sceffdeep, fu` for soluble in future.

## 4.4 Atomic commit sequence

### Commit 0: NOTES + baseline + upstream Fortran survey

Standard pattern plus:

- Read TM5 `tm5_conv.F90` lines 32-186 and GEOS-Chem
  `convection_mod.F90` DO_CONVECTION
- Save annotated notes to `artifacts/plan18/upstream_fortran_notes.md`
  capturing line-by-line formulas and any discrepancies with
  legacy Julia
- Survey legacy `src_legacy/Convection/` (4 files, ~1,768 lines)
  and map each to its upstream Fortran origin
- Register follow-up plan candidates (derived fields, scavenging,
  adjoint kernel, perf tuning)
- Memory compaction per plan 15 D3

```bash
mkdir -p docs/plans/18_CONVECTION_PLAN
# ... (standard NOTES.md scaffold with upstream_fortran_notes
# pre-loaded)
git commit -m "Commit 0: NOTES + baseline + upstream Fortran survey for plan 18"
```

### Commit 1: `AbstractConvectionOperator` + `NoConvection`

Minimal: type hierarchy, no-op operator, wiring through
`Operators.jl` and `AtmosTransport.jl`. No kernels yet.

Tests in `test/test_convection_types.jl`:

1. Type hierarchy: `NoConvection` subtypes `AbstractConvectionOperator`
2. `apply!(state, meteo, grid, ::NoConvection, dt; workspace) = state`
   (bit-exact identity, both state and array-level)
3. `apply_convection!(q_raw, ::NoConvection, ...) = nothing`
4. Dispatch correctness: no kernel launch for `NoConvection`

```bash
git commit -m "Commit 1: AbstractConvectionOperator hierarchy + NoConvection"
```

### Commit 2: `PreComputedConvMassFluxField`

- Create `src/State/Fields/PreComputedConvMassFluxField.jl`
- Thin wrapper around `AbstractArray{FT, 3}`
- Inner constructor validates rank 3
- `update_field!` no-op (caller refreshes)
- `field_value(f, (i, j, k))` direct indexed read
- Adapt support

Wire through `Fields → State → AtmosTransport`. Tests in
`test/test_fields.jl`:

- Construction: rank validation, type stability
- `field_value` reads correctly
- `update_field!` is no-op
- Adapt/GPU path converts backing array

~15 new tests. Pattern matches `PreComputedKzField` from plan 16b.

```bash
git commit -m "Commit 2: PreComputedConvMassFluxField — TimeVaryingField for convective mass fluxes"
```

### Commit 3: `CMFMCConvection` port (GCHP path)

Substantial commit — port from upstream `convection_mod.F90`
DO_RAS_CLOUD_CONVECTION with medium cleanup PLUS one deliberate
addition over legacy Julia.

- `src/Operators/Convection/CMFMCConvection.jl` — struct
- `src/Operators/Convection/cmfmc_kernels.jl` — KA kernel
- State-level `apply!(state, meteo, grid, op::CMFMCConvection, dt; workspace)`
- Array-level `apply_convection!(q_raw, op, ws, dt, meteo, grid)`
- Runtime fallback when `dtrain === nothing` (Tiedtke-style
  single-flux)

**Kernel structure (two-pass, per GCHP and legacy):**

Pass 1 (bottom-to-top): updraft concentration
```julia
# At each level k (surface to top):
#   qc = (cmfmc_below * q_cloud_below + entrn * q_env) / cmout
# TODO: scavenging hook here (site 1) — future wet deposition
# will split qc into qc_pres and qc_scav using F(k, tracer).
# For inert tracers, qc_pres = qc, qc_scav = 0.
```

Pass 2 (top-to-bottom): environment tendency
```julia
# At each level k:
#   tsum = cmfmc[k] * (q_env_above - q_env)
#        + dtrain[k] * (q_cloud - q_env)
# TODO: scavenging hook here (site 2) — future wet deposition
# will restore the four-term form:
#   tsum = cmfmc_below * qc_pres - cmfmc[k] * qc_post_mix
#        + cmfmc[k] * q_above    - cmfmc_below * q_env
# For inert tracers, the two-term form above is algebraically
# equivalent (see GCHP_NOTES §5.3).
```

**NEW vs legacy: well-mixed sub-cloud layer.**

Between the cloud base identification and Pass 1 updraft, add:
```julia
# Well-mixed sub-cloud layer treatment (per GCHP convection_mod.F90:742-782)
# NOT in legacy Julia — added as a deliberate improvement.
#
# Rationale: for tracers with strong surface sources, the sub-cloud
# layer well-mixing redistributes tracer mass before convective
# uplift. Without this, surface concentrations are biased high
# and free-troposphere concentrations are biased low.
if cldbase > 1 && cmfmc_at_cldbase_interface > tiny
    # Pressure-weighted average Q below cloud base
    qb, mb = pressure_weighted_avg(q, delp, 1:cldbase-1)
    # Mix with updraft flux from cloud base
    qc_mixed = (mb * qb + cmfmc_at_cldbase * q[cldbase] * dt) /
               (mb + cmfmc_at_cldbase * dt)
    # Apply uniformly to sub-cloud layers
    q[1:cldbase-1] .= qc_mixed
end
```

The `pressure_weighted_avg` helper is trivial (sum q·delp / sum delp).

Three-tier tests in `test/test_cmfmc_convection.jl`:

**Tier A — Analytic (5 tests):**
- Mass conservation (column sum preserved to machine precision)
- Zero forcing identity (cmfmc = dtrain = 0 → no change)
- Positivity (non-negative q in → non-negative q out)
- Stability (repeated application with fixed fluxes stays bounded)
- DTRAIN-missing fallback (nothing vs zeros: check they match
  Tiedtke-style path to machine precision)

**Tier B — Paper / Fortran formulas (7-10 tests):**
- Updraft formula: `qc = (cmfmc_below·q_cloud_below + entrn·q_env)/cmout`
  — hand-expand for 5-layer column (per GCHP convection_mod.F90:917-920
  as documented in GCHP_NOTES §3.4)
- Two-term tendency vs four-term algebraic equivalence:
  hand-expand both forms, verify port's two-term matches four-term
  after substitution (GCHP_NOTES §5.3 derivation)
- **Well-mixed sub-cloud layer:** hand-expand for 3-layer
  sub-cloud case (cldbase=3, q=[1,2,3]*mass_below), verify
  port reproduces GCHP's QB weighting and uniform application
- Mass balance: total column mass conserved under arbitrary
  cmfmc/dtrain profile

**Tier C — Cross-implementation (2-3 tests):**
- Standard deep-tropical-convective column test case (CMFMC
  peaking at 500 hPa, DTRAIN detraining at 200 hPa, idealized
  tracer initial profile including strong surface maximum)
- Compare port output to:
  - GEOS-Chem Fortran reference calculation (if runnable offline)
  - OR published Rn-222 profile from a GEOS-Chem paper
- Agreement within ~10% relative on column-integrated mass,
  similar qualitative shape
- **Compare port with sub-cloud fix vs port without sub-cloud fix
  vs GCHP output.** Port with fix should agree BETTER with GCHP
  than port without fix — documents the improvement.

All tests use accessor API (plan 14 contract).

Expected test count: ~30-35 new tests.

```bash
git commit -m "Commit 3: CMFMCConvection port — GCHP path with sub-cloud fix + Tier A/B/C validation"
```

### Commit 4: `TM5Convection` port (TM5 path)

Also substantial. Light cleanup (preserving matrix-algorithm
invariants).

- `src/Operators/Convection/TM5Convection.jl` — struct
- `src/Operators/Convection/tm5_matrix_kernels.jl` — two kernels:
  build+factorize, solve
- `src/Operators/Convection/convection_workspace.jl` — matrix
  workspace allocation
- State-level `apply!`
- Array-level `apply_convection!`

Port faithful to `tm5_conv.F90` lines 32-186. Column reversal
at the interface (k=1=TOA our convention vs k=1=surface TM5
convention) handled per legacy `tm5_matrix_convection.jl`:17.

Three-tier tests in `test/test_tm5_convection.jl`:

**Tier A — Analytic (5 tests):**
- Mass conservation (exact — implicit LU solve)
- Zero forcing identity
- Positivity
- Matrix diagonal dominance (structural: verify built matrix is
  diagonally dominant)
- LU factorization correctness (on small Nz=5 column with known
  matrix, verify port's factorization matches hand-computed LU)

**Tier B — Paper formulas (7 tests):**
- 5-layer column, specified entu/detu/entd/detd profile
- Hand-build matrix per Tiedtke 1989 + `tm5_conv.F90`:32-186
- Hand-compute LU factorization
- Hand-solve for specific RHS
- Tests: port's matrix entries match hand build; port's LU
  matches hand LU; port's solve matches hand solve

**Tier C — Cross-implementation (2 tests):**
- Same deep-convective column as Commit 3, with entu/detu/entd/detd
  derived from CMFMC+DTRAIN (documented relationship)
- Compare to TM5 Fortran reference (if runnable) or published
  TM5 output
- Agreement ~10% relative on column-integrated mass

Expected test count: ~35 new tests (matrix tests bump count).

```bash
git commit -m "Commit 4: TM5Convection port — matrix scheme with Tier A/B/C validation"
```

### Commit 5: Cross-scheme consistency test

`test/test_convection_cross_scheme.jl`:

- Specify idealized deep-convective CMFMC(k) profile (peak at
  500 hPa, zero above tropopause and at surface)
- Specify DTRAIN(k) profile (detraining at cloud top)
- Derive equivalent entu/detu/entd/detd profile via documented
  mass-balance relationship:
  `entu(k) = max(0, cmfmc(k+1) - cmfmc(k) + dtrain(k))`
  `detu(k) = dtrain(k)` (or similar, per TM5 ↔ GCHP mapping)
  `entd(k), detd(k) = 0` (no downdraft in this test case)
- Run both operators on matched forcing
- Column-integrated tracer mass should agree to ~5% relative
- Vertical profile: similar qualitative shape

This test is a strong cross-check: if both schemes implement
consistent physics, they should agree on the column-integrated
redistribution. Divergence flags a bug in one or both.

Expected test count: ~8 new tests.

```bash
git commit -m "Commit 5: Cross-scheme consistency test for CMFMCConvection vs TM5Convection"
```

### Commit 6: Convection workspace + TransportModel wiring

- Extend `AdvectionWorkspace` to include `ConvectionWorkspace` or
  add a separate field (TBD during implementation — pick the
  cleaner option)
- `TransportModel` gains `convection::ConvT` field, default
  `NoConvection()`
- `with_convection` helper parallels `with_chemistry` / `with_diffusion`
- `step!(model, dt)` orchestration:
  ```julia
  function step!(model::TransportModel, dt; meteo = nothing)
      # Transport block: advection + diffusion + emissions
      apply_advection_block!(model, dt; meteo)
      # Convection block
      apply_convection!(model.state.tracers_raw, model.convection,
                        model.workspace, dt, meteo, model.grid)
      # Chemistry block
      apply!(model.state, meteo, model.grid, model.chemistry, dt;
             workspace = model.workspace)
  end
  ```

Tests in `test/test_transport_model_convection.jl`:

- Default `TransportModel` carries `NoConvection` (1 test)
- `with_convection` returns new model with only convection
  replaced (~5 tests)
- Bit-exact regression: default `step!` bit-exact to pre-18
  no-convection path (1 critical test — `==` not `≈`)
- `step!` with `CMFMCConvection + ConstantField` fluxes produces
  measurable vertical redistribution (~5 tests)
- `step!` with `TM5Convection` similar (~5 tests)
- Bit-exact regression: plan 17's
  `test_diffusion_palindrome.jl` / `test_transport_model_diffusion.jl`
  / `test_emissions_palindrome.jl` suites pass unchanged (regression
  check, not new tests)

Expected new test count: ~15 new tests in convection-specific
file; regression checks confirmed on existing files.

```bash
git commit -m "Commit 6: TransportModel.convection + step! orchestration (block-level)"
```

### Commit 7: DrivenSimulation integration

`DrivenSimulation` wiring — but note this should be minimal
since plan 17 Commit 6 already cleaned up the sim-level
orchestration. Plan 18's work:

- Ensure `DrivenSimulation` constructor accepts convection
  operator (via kwarg or via pre-constructed TransportModel)
- Verify `DrivenSimulation.step!` delegates entirely to
  `step!(model)` and convection rides through automatically
- Add a test that sim-level run with convection produces expected
  trajectory

Also: `AbstractMetDriver` no API changes. Concrete drivers
(CATRINE driver, if exists) may need to expose convective mass
flux fields; that's driver-specific work outside plan 18's scope
beyond documenting what the driver should provide.

Tests in `test/test_driven_simulation_convection.jl`:

- Sim construction with `convection = NoConvection()` (default)
- Sim construction with `CMFMCConvection` using ConstantField
  forcing
- End-to-end 10-step run preserves mass
- Existing `test_driven_simulation.jl` 57-test suite passes
  unchanged

Expected new test count: ~8 new tests.

```bash
git commit -m "Commit 7: DrivenSimulation integration for convection"
```

### Commit 8: Convection block position study

Analogous to plan 17's ordering study, but for block position
rather than palindrome position.

Four configurations compared:

| Label | Block order | Rationale |
|---|---|---|
| 1 (recommended) | transport → convection → chemistry | §4.3 Decision 1 |
| 2 | transport → chemistry → convection | chem first |
| 3 | convection → transport → chemistry | conv first |
| 4 (palindrome-internal) | X Y Z V(dt/4) C(dt/2) V(dt/4) S V(dt/4) C(dt/2) V(dt/4) Z Y X | palindrome internal |

Note: Config 4 requires palindrome modification — this is
scoped as "study only; not shipped." The study runs each config
through a standardized 24-h idealized tropical convective
scenario (tropical CMFMC/DTRAIN profile, emissions at surface,
decay chemistry). Tracks column-integrated mass distribution.

Expected findings (hypothesized):
- Configs 1, 2, 3 differ by ~1-3% relative (small)
- Config 4 may differ more (palindrome mixing with convection)
- No dramatic "Config D-style pileup" expected (unlike plan 17
  where D had catastrophic layer-1 accumulation)

Writeup in `docs/plans/18_CONVECTION_PLAN/position_study_results.md`
with plots and recommendation.

If Config 1 is clearly best (or within noise), plan 18's shipped
default stands. If Config 4 clearly wins, document as follow-up
plan candidate: "move convection inside palindrome."

Expected test count: ~10 new tests.

```bash
git commit -m "Commit 8: Convection block position study — four configurations compared"
```

### Commit 9: Benchmarks

`scripts/benchmarks/bench_convection_overhead.jl`:

- Configurations: `NoConvection` (baseline), `CMFMCConvection +
  ConstantField`, `TM5Convection + ConstantField`
- Sizes: small CPU, medium CPU, medium GPU, large GPU (L40S F32
  per plan 16b convention)
- Metrics: median step time, overhead vs baseline

Expected results:
- `CMFMCConvection`: moderate overhead (~10-30%). Kernel is
  column-serial but arithmetic-light.
- `TM5Convection`: higher overhead (~30-70%). Matrix build + LU
  solve is the heaviest convective operator.

Target: correctness > perf for plan 18. No hard perf targets
(all "soft"). Document performance characteristics in
`artifacts/plan18/perf/SUMMARY.md`.

```bash
git commit -m "Commit 9: Convection overhead benchmarks"
```

### Commit 10: Retrospective + ARCHITECTURAL_SKETCH_v4

- Fill in NOTES.md retrospective sections
- Update `ARCHITECTURAL_SKETCH.md` to v4:
  - Convection operator hierarchy added
  - Step-level composition documented (transport_block →
    convection_block → chemistry_block)
  - File-level map updated
- Update `OPERATOR_COMPOSITION.md` with block-vs-palindrome
  decision and reference to position study results
- Write `docs/plans/18_CONVECTION_PLAN/validation_report.md`
  summarizing three-tier test coverage
- Document any legacy bugs found and fixed during port
- Document any scope deviations (split/merged commits, reshuffled
  decisions)

```bash
git commit -m "Commit 10: Retrospective + ARCHITECTURAL_SKETCH_v4 + validation_report"
```

## 4.5 Test plan per commit

After EACH commit:
```bash
julia --project=. -e 'using AtmosTransport'
julia --project=. test/runtests.jl
```

Baseline 77-failure count must be preserved. All prior plans'
test suites (plan 11-17) must continue to pass unchanged.

Validation tier checks:
- Tier A tests run in normal test suite (CI-friendly, fast)
- Tier B tests run in normal test suite (CI-friendly, moderate)
- Tier C tests may require external reference data or runnable
  Fortran — may be opt-in (`@testset "Tier C (requires refs)"`
  skipped by default, run manually in validation sessions)

## 4.6 Acceptance criteria

**Correctness (hard):**
- All pre-existing tests pass (77 baseline failures unchanged)
- Plans 11-17 regression tests pass bit-exact
- Mass conservation in both operators to machine precision
- Tier A + Tier B tests all pass
- Tier C tests pass if reference data available; documented
  otherwise

**Cross-scheme consistency (hard):**
- `CMFMCConvection` and `TM5Convection` agree to ~5% on column-
  integrated mass for matched forcing (Commit 5 test)
- Divergences >5% investigated and documented as legacy bugs
  fixed or scheme-specific physics differences

**Code cleanliness (hard):**
- `src/Operators/Convection/` directory exists per §4.2
- `PreComputedConvMassFluxField` in `src/State/Fields/`
- No `AbstractTracerSolubility` / scavenging code in plan 18
- Inline helper functions (`_cmfmc_updraft_mix`,
  `_cmfmc_apply_tendency`, `_cmfmc_wellmix_subcloud`,
  `_tm5_scavenging_matrix`) with inert-only signatures per
  Decision 19
- Each helper docstring documents "FUTURE: scavenging overload
  adds solubility parameter"
- Legacy `ras_convection.jl:41-46` comment removed/corrected
  during port (per Decision 18 cleanup item)
- Operator docstrings explicitly state basis expectation
  (CMFMC: dry; TM5: moist) per Decision 18
- Convection block after transport, before chemistry, in
  `step!`

**Interface validation (hard):**
- `apply!` signature unchanged (plan 15 + 16b + 17 + 18
  validated; plan 18 confirms for non-local operator)
- `AbstractTimeVaryingField{FT, 3}` works for convective mass
  flux fields at kernel scale
- Array-level `apply_convection!` entry point mirrors plan 16b
  / 17 pattern

**Performance (soft):**
- `CMFMCConvection` overhead <50% on GPU at medium grid
- `TM5Convection` overhead <100% on GPU at medium grid
- No regression in non-convection paths (bit-exact default
  regression passes)

**Validation discipline (hard):**
- Upstream Fortran references read and annotated in
  `artifacts/plan18/upstream_fortran_notes.md`
- Three-tier validation structure in test files
- Any legacy bugs found documented with source-of-truth
  hierarchy (paper > Fortran > legacy Julia) applied

**Documentation (hard):**
- `ARCHITECTURAL_SKETCH_v4.md` committed
- `position_study_results.md` with plots and recommendation
- `validation_report.md` summarizing tier coverage
- NOTES.md complete with legacy-bug section (if any bugs found)

## 4.7 Rollback plan

Standard. Specific rollback points:

- **Commit 3 (CMFMCConvection) Tier B fails.** Port has physics
  error. Re-read `convection_mod.F90` DO_CONVECTION;
  `ras_convection.jl` legacy for hints; paper for definitive
  form. Do NOT ship a Commit 3 that fails Tier B.
- **Commit 4 (TM5Convection) Tier A mass conservation fails.**
  Matrix port has a bug. Compare matrix entries to hand-built
  reference. LU factorization sign errors are common.
- **Commit 5 (cross-scheme) shows >20% divergence.** One operator
  has a bug, or the mass-balance relationship between CMFMC/DTRAIN
  and entu/detu is wrong. Investigate before proceeding.
- **Commit 6 (TransportModel wiring) breaks bit-exact regression.**
  Fix: ensure `NoConvection` dispatch is a compile-time dead
  branch, not a runtime no-op with side effects. Check that
  `step!`'s convection-block branch is type-stable.
- **Commit 8 (position study) shows palindrome-internal dominates
  block version.** Document as finding; file a follow-up plan.
  Do NOT reorganize plan 18 mid-flight — ship as planned,
  re-architect in a future plan.

## 4.8 Known pitfalls

1. **"I'll port TM5 matrix scheme faithfully without reading the
   Fortran."** NO per §4.3 Decision 14. Legacy `tm5_matrix_convection.jl`
   may have bugs. Upstream `tm5_conv.F90` is authoritative.

2. **"Legacy Julia has scavenging code — I'll port it as commented-
   out."** NO per §4.3 Decision 16. No scavenging code at all in
   plan 18. `# TODO:` comment markers at insertion sites is the
   full extent of scavenging acknowledgment.

3. **"DTRAIN-missing fallback deserves its own operator type."**
   NO. Runtime path within `CMFMCConvection` via
   `dtrain === nothing`. Same type parameter structure;
   conditional code path.

4. **"Tier C tests are hard; I'll skip them."** NO. Tier C is
   REQUIRED per CLAUDE_additions_v2 validation discipline. If
   upstream Fortran isn't runnable, use published reference
   data. If no published data, construct a Tier C test using
   cross-scheme consistency (Commit 5's test IS a form of Tier
   C — both schemes are independent implementations).

5. **"I'll treat legacy Julia as ground truth for Tier B tests."**
   NO. Tier B tests are against PAPER formulas, not legacy Julia.
   Hand-expand from Tiedtke 1989 and Moorthi-Suarez 1992, not
   from legacy code.

6. **"I'll put convection inside the palindrome since that's
   where plan 17 put emissions."** NO per §4.3 Decision 1. Plan
   18 ships convection as separate block. Position study
   (Commit 8) may suggest palindrome-internal; if so, file as
   follow-up plan, don't reorganize plan 18 mid-flight.

7. **"The matrix workspace is big; I'll dynamically allocate
   per-step."** NO. Pre-allocate in `ConvectionWorkspace` at
   TransportModel construction. Dynamic allocation in `apply!`
   violates the "workspace supplied by caller" pattern from plan
   16b.

8. **"I'll add `InertTracer` default as a vestigial type
   parameter."** NO. No solubility parameters at all. Kernel
   applies mass flux redistribution without trait queries.

9. **"I'll port adjoint kernels since they're in legacy."** NO.
   Plan 18 preserves adjoint-able structure but does not ship
   adjoint kernels. Follow-up plan.

10. **"Cross-scheme test is too strict at 5% — I'll use 20%."**
    NO per §4.3 Decision 10. With both operators run on
    same-basis inputs (Option A per Decision 18), the remaining
    ~5% tolerance covers discretization (explicit vs implicit)
    and sub-cloud layer differences. If schemes disagree more,
    there's a real inconsistency worth finding. Conversely, do
    not LOOSEN to 10% — that was pre-Option-A value that mixed
    basis and discretization error.

11. **"I'll skip the well-mixed sub-cloud layer since legacy
    Julia skipped it."** NO per §4.3 Decision 17. Plan 18 ADDS
    the sub-cloud layer treatment. Legacy's omission is a real
    bug for CATRINE surface-source tracers. Tier C test verifies
    the improvement.

12. **"Two-term tendency is wrong, I'll restore the four-term
    form."** NO per §4.3 Decision 8. For INERT tracers (plan 18
    scope), the two-term form is algebraically equivalent to the
    four-term form (see GCHP_NOTES §5.3 derivation). Both forms
    are correct; the simpler two-term form is preferred. The
    four-term form becomes necessary ONLY when scavenging is
    added in a future plan.

13. **"Benchmark overhead is too high; I need to optimize before
    shipping."** NO. Plan 18 is correctness-first. `TM5Convection`
    being 70% overhead at medium grid is expected (matrix solve
    is expensive). File a perf-tuning plan for shared-memory
    BLAS or cuBLAS-batched paths.

14. **"The GEOS-FP June 2020 discontinuity should be handled in
    the operator."** NO. It's a DATA problem. Document in user
    docs. Operator consumes whatever CMFMC+DTRAIN the driver
    provides.

---

# Part 5 — How to Work

## 5.1 Session cadence

- Session 1: Commits 0-1 (NOTES + baseline + upstream Fortran
  survey + type hierarchy)
- Session 2: Commit 2 (PreComputedConvMassFluxField)
- Session 3: Commit 3 Part 1 (CMFMCConvection scaffolding + Tier A)
- Session 4: Commit 3 Part 2 (CMFMCConvection kernel + Tier B)
- Session 5: Commit 3 Part 3 (CMFMCConvection Tier C + completion)
- Session 6: Commit 4 Part 1 (TM5Convection scaffolding + matrix
  workspace + Tier A)
- Session 7: Commit 4 Part 2 (TM5Convection kernel + Tier B)
- Session 8: Commit 4 Part 3 (TM5Convection Tier C + completion)
- Session 9: Commit 5 (cross-scheme consistency)
- Session 10: Commits 6-7 (TransportModel + DrivenSimulation)
- Session 11: Commit 8 (position study)
- Session 12: Commits 9-10 (benchmarks + retrospective)

This is longer than plans 15-17 because:
- Two substantive physics ports (CMFMCConvection + TM5Convection)
- Three-tier validation per port
- Cross-scheme consistency test
- Position study

## 5.2 When to stop and ask

- Commit 0 upstream Fortran not accessible (no `deps/tm5/` source;
  no GEOS-Chem Fortran available). MUST stop and ask for
  alternative reference (paper expansion may suffice for some
  Tier B, but Tier C needs code or published data).
- Commit 3 Tier A mass conservation fails. Port has a bug in
  CMFMCConvection.
- Commit 4 Tier A LU factorization fails. Port has a bug in TM5
  matrix scheme.
- Commit 5 cross-scheme test shows >20% divergence. One operator
  has a significant bug. Investigate before continuing.
- Commit 6 breaks plan 17 regression test. Default dispatch is
  not dead-branch optimized; investigate.
- Legacy bug found: document and stop to confirm with user before
  deciding port-follows-paper vs port-preserves-bug.
- Scope creep toward scavenging, derived fields, or adjoint kernels.

## 5.3 NOTES.md discipline

Specific items to capture:

- Any upstream Fortran / legacy Julia discrepancies discovered
  during Commit 0 survey
- Decisions on port-follows-paper vs port-preserves-legacy when
  discrepancies exist
- Legacy bugs found (if any) and their fix rationale
- Deviations from plan doc §4.4 (commits split, merged, reshuffled)
- Performance observations (where is time spent in each operator?)
- Position study outcomes (which config wins, by how much)
- Cross-scheme test outcomes (are schemes consistent? on what
  scales?)

## 5.4 Follow-up plan candidates (register at Commit 0)

Per plan 17 pattern, register at Commit 0 for retrospective (Commit
10) to close:

- `DerivedConvMassFluxField` — online computation of CMFMC/DTRAIN
  from parent meteorology
- Convection adjoint kernel port
- Wet deposition operator (scavenging, Henry's law,
  precipitation flux fields)
- Multi-tracer fusion in TM5 matrix solve
- Shared-memory / cuBLAS-batched LU for performance
- `AbstractLayerOrdering{TopDown, BottomUp}` abstraction (also
  deferred from plan 17)
- Palindrome-internal convection (if position study Commit 8
  suggests benefit)
- TiedtkeConvection as standalone type (if CMFMCConvection
  fallback path gets heavily used)
- Plan 16c: retroactive Tier B/C validation for diffusion
  operator (deferred from plan 16b). Scope should include
  **basis audit** — verify `ImplicitVerticalDiffusion` /
  Beljaars-Viterbo port is consistent about whether Kz fields
  and tracer state are on dry or moist basis. Tier C comparison
  against GCHP vdiff_mod and TM5 diffusion.F90 for a standard
  PBL column. Plan 18's basis-audit discipline (Decision 18)
  should retroactively apply.
- Upstream basis documentation in AtmosTransport. `src/` may
  have a dry-air correction path inherited from legacy; verify
  it matches GCHP convention. If absent, add it (referencing
  `src_legacy/Advection/latlon_dry_air.jl` and
  `cubed_sphere_mass_flux.jl` as starting points).

---

# End of Plan

After this refactor ships:
- `CMFMCConvection` in `src/Operators/Convection/` (GCHP path)
- `TM5Convection` in `src/Operators/Convection/` (TM5 path)
- Convection block positioned after transport, before chemistry
  in `step!`
- Three-tier validation completed (Tier A, B, C)
- Cross-scheme consistency verified (~5% agreement)
- CATRINE intercomparison possible for convection-sensitive
  tracers (Rn-222, tropical CO2)
- Full offline atmospheric transport model with emissions,
  transport, convection, diffusion, chemistry — operational

The next plan candidates (see §5.4):
- Wet deposition (scavenging + Henry's law + precip fluxes)
- Retroactive Tier B/C validation for plans 11-16b operators
- Performance tuning (multi-tracer fusion, shared-memory BLAS,
  cuBLAS-batched solve)
- Adjoint kernels for sensitivity/inversion work
- Online convection schemes (DerivedConvMassFluxField)
- Layer-ordering abstraction cleanup

This is the final physics operator needed for CATRINE. After
plan 18, the model has the full operator suite for offline
atmospheric transport.
