# Plan 18 Consolidation Brief — For Plan Agent

**Purpose:** consolidate the plan 18 package (currently 4 documents
with precedence rules) into a single authoritative `18_CONVECTION_PLAN_v4.md`
that an execution agent can work from without cross-referencing.

**You (the plan agent) are the right layer for this** because:

1. The planning-chat assistant has made at least two errors from
   relying on summaries and stale snapshots. You can read the
   current tree directly and verify.
2. Incremental edits through chat have accumulated contradictions
   between decisions (Decisions 10, 13, 18 conflicted in the v3
   base plan; this has since been corrected but the pattern is
   diagnostic). Single-pass consolidation by a disciplined
   planning process avoids recurrence.
3. An execution agent shouldn't have to read three documents with
   precedence rules. One authoritative document.

---

## Inputs you have to work with

Four source documents at `/mnt/user-data/outputs/plan/`:

1. **`18_CONVECTION_PLAN.md`** (v3) — base plan, 1835 lines, 19
   decisions, 10-commit sequence. Has known contradictions
   (Decisions 10/13/18 on basis conventions) that later addenda
   corrected.

2. **`18_CONVECTION_CORRECTIONS_ADDENDUM_v2.md`** — Codex + GPT-5.4
   review findings. Authoritative on overlap with base plan.
   Key corrections:
   - Basis rule: operators are basis-polymorphic, basis follows
     state. Decisions 10/13/18 replaced by single Decision 20.
   - `PreComputedConvMassFluxField` dropped (Commit 2 removed);
     convection forcing lives in window as plain arrays inside
     `ConvectionForcing` struct. New Decision 22.
   - CFL sub-cycling in CMFMCConvection is mandatory, not
     optional. New Decision 21.
   - Commit 7 (driver integration) significantly expanded.
   - My earlier H1 about tracer state interface was WRONG — the
     base plan's `tracers_raw` assumption is correct.

3. **`18_CONVECTION_ADJOINT_ADDENDUM.md`** — purely additive.
   - Tier A adjoint-identity test per operator
   - Docstring paragraph on adjoint path
   - Acceptance criterion on adjoint structure
   - Warning against porting GCHP's positivity clamp (breaks
     linearity)
   - Plan 19 candidate registration

4. **`PRE_PLAN_18_FIXES.md`** — three current-tree bugs that must
   ship BEFORE plan 18 starts. NOT part of plan 18 itself;
   reference it in the plan's "prerequisites" section.

Also relevant:

- `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` — TM5 Fortran
  reference, unchanged
- `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` — GCHP Fortran reference,
  v3 was corrected in §4 (basis convention). Current version is
  authoritative.
- `CLAUDE_additions_v2.md` — accumulated lessons including case
  study and basis-conventions section

## Ground truth: current `src/` tree

The current AtmosTransport source is at
`/home/claude/AtmosTransport/` (symlink to the zipped snapshot
`AtmosTransport-surface-emissions/`). Plans 11-17 are merged.

**You must verify every interface claim against this tree.**
Do not rely on the base plan or addenda for interface shape.
Where the documents and the tree disagree, the tree is correct.

Specifically, verify by direct inspection:

- `src/State/CellState.jl` — confirms `tracers_raw` packed storage,
  `air_mass`, `tracer_names`
- `src/State/Basis.jl` — `AbstractMassBasis`, `DryBasis`,
  `MoistBasis`
- `src/State/FaceFluxState.jl` — flux state with basis tag
- `src/Models/TransportModel.jl` — step! forwards diffusion/
  emissions/meteo
- `src/Models/DrivenSimulation.jl` — driver threading
- `src/MetDrivers/AbstractMetDriver.jl` — supports_convection,
  current_time stubs
- `src/MetDrivers/TransportBinaryDriver.jl` — window struct
  definition
- `src/Operators/Advection/StrangSplitting.jl` — structured
  (line 962-) vs face-indexed (line 1035-) apply! methods;
  palindrome structure at line 1199-1246
- `src/Operators/Diffusion/` — plan 16b
- `src/Operators/SurfaceFlux/` — plan 17
- `src/State/Fields/` — time-varying field abstractions (plan 16a/17)

If anything in the consolidated plan references code that doesn't
exist or has a different shape than the documents describe, fix
the plan — don't just copy forward.

## What to produce

**Deliverable:** `18_CONVECTION_PLAN_v4.md` — single document,
~1500-2000 lines, authoritative.

**Structure suggestion:**

```
Part 1 — Context
  1.1  Purpose and scope
  1.2  Prerequisites (link to PRE_PLAN_18_FIXES.md)
  1.3  Current state of src/ (verified)
  1.4  Upstream references (TM5 + GCHP Fortran)
  1.5  Non-goals

Part 2 — Design decisions (consolidated)
  2.1  Operator hierarchy and types (Decisions 1, 2)
  2.2  Composition and palindrome position (Decisions 1, ...)
  2.3  Basis handling (Decision 20 replaces 10/13/18)
  2.4  Convection forcing as window field (Decision 22)
  2.5  Validation discipline (Decision 9)
  2.6  Scavenging dispatch-ready structure (Decisions 8, 19)
  2.7  Well-mixed sub-cloud layer (Decision 17)
  2.8  CFL sub-cycling (Decision 21)
  2.9  Adjoint structure preservation (Decision 11 + adjoint addendum)

Part 3 — Commit sequence
  Commit 0  NOTES + baseline + upstream survey
  Commit 1  AbstractConvectionOperator + NoConvection
  Commit 2  Window extension (ConvectionForcing struct)
            [replaces base plan's PreComputedConvMassFluxField]
  Commit 3  CMFMCConvection (port + sub-cycling + sub-cloud +
            inline helpers + Tier A/B/C + adjoint identity)
  Commit 4  TM5Convection (matrix port + Tier A/B/C + adjoint
            identity)
  Commit 5  Cross-scheme consistency test (~5% tolerance,
            same basis)
  Commit 6  TransportModel wiring
  Commit 7  Driver/window integration (supports_convection
            overrides, load_transport_window! extension,
            ConvectionForcing population)
  Commit 8  DrivenSimulation wiring
  Commit 9  Position study
  Commit 10 Benchmarks
  Commit 11 Retrospective + ARCHITECTURAL_SKETCH_v4

Part 4 — Acceptance criteria (hard/soft)
Part 5 — Known pitfalls
Part 6 — Follow-up plan candidates (register at Commit 0)
Part 7 — How to work (session cadence, validation per commit)
```

Feel free to restructure. The above is a starting point.

## Specific consolidation rules

### Decisions — keep, drop, replace

**DROP entirely:**
- Base plan Decision 10 as worded (replaced by corrections §B)
- Base plan Decision 13 ("moist basis retained throughout")
- Base plan Decision 18 (dual-basis per operator)
- All "7-10% tolerance" language

**NEW decisions from corrections addendum v2:**
- Decision 20: Basis follows state. Operators basis-polymorphic.
- Decision 21: CFL sub-cycling in CMFMCConvection mandatory.
- Decision 22: Convection forcing in window as plain arrays.

**NEW decisions from adjoint addendum:**
- Tier A adjoint-identity test (not strictly a new decision;
  extension of Decision 11)
- Warning against positivity clamp that breaks linearity (same)

**KEEP from base plan (v3):**
- Decisions 1, 2, 8, 9, 11, 14, 15, 16, 17, 19

### Tolerance numbers — one value, one rationale

Cross-scheme test tolerance is **~5%** on column-integrated mass,
both operators run on the same basis as state. Rationale: explicit
vs implicit discretization (~2-3%) + well-mixed sub-cloud
(~1-2%). Use this one number everywhere. Any appearance of "7-10%"
or "10%" or "5%" with different rationale is a drift artifact —
pick one and use it.

### File paths — one convention

Filesystem path is `docs/plans/18_ConvectionPlan/` (camelCase).
All references in the plan use this path consistently.

### State interface — verify, don't assume

The base plan's mental model (`state.tracers_raw(Nx,Ny,Nz,Nt)`,
`state.air_mass`, accessor API) IS correct in current tree. My
earlier addendum that said otherwise was wrong. Verify with
`src/State/CellState.jl` — if any field name has changed since
plans 11-17 merged, update the plan to match.

Specifically:
- `state.air_mass` (NOT `state.air_mass_dry`)
- `state.tracers_raw` with `tracer_names::NTuple`
- Basis via type parameter `B <: AbstractMassBasis`
- Mixing ratio NOT pre-computed; kernels do `tracer_mass[k] / air_mass[k]`
  per cell

### ConvectionForcing — replaces PreComputedConvMassFluxField

Key change from base plan:

```julia
# BASE PLAN (dropped): separate TimeVaryingField
struct PreComputedConvMassFluxField{FT, A} <: AbstractTimeVaryingField{FT, 3}
    data :: A
    ...
end

# v4: plain arrays in ConvectionForcing, field of the window
struct ConvectionForcing{CM, DT, TM}
    cmfmc      :: CM   # ::Union{Nothing, AbstractArray{FT, 3}}
    dtrain     :: DT   # ::Union{Nothing, AbstractArray{FT, 3}}
    tm5_fields :: TM   # ::Union{Nothing, NamedTuple{...}}
end

struct StructuredTransportWindow{Basis, M, PS, F, Q, D, C} <: ...
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C  # ::Union{Nothing, ConvectionForcing}
end
```

Rationale: the window IS the time-windowed container. Wrapping
cmfmc/dtrain in a separate TimeVaryingField duplicates this
contract and creates dependency on `current_time(driver)` which
is stub-only in current tree (fix A3).

Plain arrays inside `ConvectionForcing` follow the same pattern
as `air_mass`, `fluxes`, `qv_start`, etc. Same life-cycle.

### Adjoint support — not deferred, minimally shipped

Per adjoint addendum:
- Each operator's Tier A suite includes an adjoint-identity test
  `⟨y, L·x⟩ = ⟨L^T·y, x⟩`
- Operator docstrings include "Adjoint path (not shipped in plan
  18)" paragraph documenting how a future adjoint kernel would
  reuse forward structure (especially TM5: reuse LU with
  trans='T')
- CMFMCConvection kernel does NOT port GCHP's positivity clamp
  at convection_mod.F90:1002-1004 — breaks linearity. Alternative:
  pre-step validation of met data, or accept tiny negativities
  and rely on global mass fixer.

### Pre-plan-18 fixes — prerequisite

The plan has a "Prerequisites" section that references
`PRE_PLAN_18_FIXES.md` as a separate document that must ship
before plan 18 Commit 0. Don't inline the fix specs; they're
separate deliverables.

## Coherence checklist — verify in final document

Before declaring v4 done, verify by search:

- [ ] No occurrence of "PreComputedConvMassFluxField" except as
      historical reference in Revisions Note
- [ ] No occurrence of "7-10%", "10% relative", "~7%" for
      cross-scheme tolerance. Single value ~5% throughout.
- [ ] No occurrence of "tracers_raw_v2", "air_mass_dry",
      or other interface names that don't exist in current tree
- [ ] No occurrence of "Decision 10" (as worded in v3), "Decision 13",
      "Decision 18" — replaced by Decision 20. If referenced, it's
      only in Revisions Note.
- [ ] File path "docs/plans/18_CONVECTION_PLAN/" replaced
      everywhere with "docs/plans/18_ConvectionPlan/"
- [ ] All interface references (CellState fields, window fields,
      driver methods) match current `src/` verified by direct
      inspection
- [ ] Adjoint-identity test specified for each operator's Tier A
      suite
- [ ] CFL sub-cycling in CMFMCConvection kernel interface, not
      buried in a paragraph — should be visible in the kernel
      signature in §3.3 or equivalent
- [ ] ConvectionForcing struct shape appears in Commit 2 description
- [ ] Each acceptance criterion references a specific test file
      or verifiable property, not vague goals

## Estimated effort

~3-5 hours for a careful plan agent, including tree verification.
The source material is substantial but the consolidation work is
structural, not creative. Think of it as a senior-engineer doc
review + rewrite pass.

## What NOT to do

- Don't accept the base plan as-is and just apply patches. The
  base plan has accumulated drift; re-derive each section from
  inputs.
- Don't preserve "historical context" in the main body of v4.
  Put the revision history in a compact section at the end
  (`# Revisions note: v1 → v2 → v3 → v4`) and move on.
- Don't introduce new decisions without clear provenance. Every
  decision in v4 traces to either: (a) a decision in base plan
  that wasn't dropped, (b) a correction from the v2 addendum,
  or (c) an addition from the adjoint addendum. If you find
  yourself wanting to add (d), stop and ask.
- Don't assume any interface. Verify against `src/`.

---

**After you produce v4, the execution agent gets ONE document
plus the prerequisites file. That's the clean handoff.**

## Deliverables summary

At completion of consolidation:

1. `/mnt/user-data/outputs/plan/18_CONVECTION_PLAN_v4.md`
   (the consolidated plan)
2. `/mnt/user-data/outputs/plan/PRE_PLAN_18_FIXES.md`
   (unchanged, prerequisites)
3. `/mnt/user-data/outputs/plan/18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md`
   (unchanged, TM5 reference)
4. `/mnt/user-data/outputs/plan/18_CONVECTION_UPSTREAM_GCHP_NOTES.md`
   (unchanged, GCHP reference)

Everything else (v3 base plan, corrections addenda, adjoint
addendum) becomes archival. You can move them to a
`archived/` subdirectory or delete them — the execution agent
doesn't need them once v4 exists.

---

**End of consolidation brief.**
