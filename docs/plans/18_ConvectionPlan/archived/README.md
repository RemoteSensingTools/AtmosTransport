# Archived Plan 18 source documents

These documents are **superseded by `18_CONVECTION_PLAN_v5.md`** in
the parent directory. They are preserved here for provenance and as
historical record of the plan's evolution. Execution agents should
work from v5 directly — reading these archived files is not required
and may cause confusion (v3 has known contradictions at Decisions
10 / 13 / 18; v4 had an architectural mistake in the runtime data
flow that v5 corrects).

| File | Role before supersession | Superseded by |
|---|---|---|
| `18_CONVECTION_PLAN_v3.md` | Base plan, 1834 lines, 19 decisions | v4, then v5 |
| `18_CONVECTION_CORRECTIONS_ADDENDUM_v2.md` | Codex + GPT-5.4 review; introduced Decisions 20 / 21 / 22; dropped Commit 2 as written | v4 Part 2 §§2.3, 2.4, 2.8; Part 3 Commit 2, Commit 7 (all carried into v5) |
| `18_CONVECTION_ADJOINT_ADDENDUM.md` | Additive: Tier A adjoint-identity test, no-positivity-clamp rule, Plan 19 registration | v4/v5 Part 2 §2.9; Commits 3 and 4 Tier A; Part 6 |
| `CONSOLIDATION_BRIEF_FOR_PLAN_AGENT.md` | Task specification for producing v4 | v4 shipped; retained as record |
| `18_CONVECTION_PLAN_v4.md` | Consolidated single-source plan (1788 lines); had runtime-data-flow mistake (operator `apply!` assumed it would receive a window; reality: window lives on `DrivenSimulation` and model holds per-step forcing) | v5 Part 2 §§2.14, 2.17-2.19 (Decisions 3 revised, 23, 24, 25); Part 3 Commits 2, 6, 7, 8 revised |
| `V5_ARCHITECTURAL_BRIEF.md` | GPT-5.4 review findings specifying v5 corrections | v5 shipped; retained as record |

## What still lives in the parent directory

- `18_CONVECTION_PLAN_v5.md` — authoritative plan (execution agents read this)
- `PRE_PLAN_18_FIXES.md` — prerequisite current-tree fixes (A1/A2/A3; A3 rewritten in v5 to target sim-level `current_time`)
- `18_CONVECTION_UPSTREAM_FORTRAN_NOTES.md` — TM5 Fortran reference
- `18_CONVECTION_UPSTREAM_GCHP_NOTES.md` — GCHP Fortran reference
- `CLAUDE_additions_v2.md` — accumulated lessons (plans 13-17 + plan 18 design)

## Key changes from v3 → v4 → v5

### v3 → v4 (consolidation)

**Dropped decisions:**

- v3 Decision 10 (mixed-basis cross-scheme test)
- v3 Decision 13 ("moist basis retained throughout")
- v3 Decision 18 (dual-basis per operator)

**New decisions in v4:**

- **Decision 20**: basis follows state; operators are basis-polymorphic; driver handles basis conversion upstream.
- **Decision 21**: CFL sub-cycling in `CMFMCConvection` is mandatory, cached per window, with bit-exact regression test.
- **Decision 22**: convection forcing lives in the transport window as plain arrays inside `ConvectionForcing`, not wrapped in `TimeVaryingField`. Replaces v3 Commit 2's `PreComputedConvMassFluxField`.

**Adjoint addendum absorbed:** Tier A adjoint-identity test per
operator, no-positivity-clamp rule, Plan 19 follow-up.

**Structural corrections:** single `operators.jl` file per submodule
matching existing Diffusion / SurfaceFlux convention; camelCase
directory path throughout; single ~5% cross-scheme tolerance.

### v4 → v5 (runtime data-flow correction)

GPT-5.4 architectural review flagged that v4's `apply!(state, met::AbstractTransportWindow, ...)`
signature and Commit 6 `apply_convection!(..., meteo, ...)` threading
both assume data that isn't at the call site. Reality (verified in
v5 against current tree):

- **The window lives on `DrivenSimulation`, not on `TransportModel`.**
- **`TransportModel.fluxes` is the per-step forcing slot**, populated
  by `DrivenSimulation._refresh_forcing!` (lines 119-131) copying
  from `sim.window.fluxes`.
- **The meteo kwarg at `step!(sim.model, sim.Δt; meteo = sim.driver)`
  gives operators a stateless driver** (driver struct at
  `src/MetDrivers/TransportBinaryDriver.jl:109-112` has only
  `reader` + `grid`; no current_window_index, no time).

v5 mirrors the `fluxes` pattern for convection:

- **Decision 22 extended:** `ConvectionForcing` lives at two runtime
  slots — on the window (populated once per met window by the driver)
  AND on `TransportModel.convection_forcing` (refreshed every
  substep by `_refresh_forcing!`).
- **New Decision 23:** Runtime data flow. Operator reads only from
  `TransportModel.convection_forcing`, never from the window or
  driver. `_refresh_forcing!` extension copies window → model each
  substep.
- **New Decision 24:** `meteo = sim`, not `sim.driver`.
  `current_time(sim::DrivenSimulation) = sim.time` (sim already
  tracks this at `DrivenSimulation.jl:26, 276`). Driver stays
  stateless.
- **New Decision 25:** Face-indexed convection is out of scope.
  Error-stub dispatch on state shape (`Raw <: AbstractArray{_,3}`),
  not on a nonexistent `AbstractFaceIndexedMesh` trait (brief's
  proposed dispatch was wrong — the type doesn't exist in
  `src/Grids/`).
- **Decision 3 revised:** `apply!` takes `ConvectionForcing` directly,
  not `AbstractTransportWindow`. Uses `AtmosGrid`, not non-existent
  `AbstractGrid`.

**Commit sequence adjusted:**

- Commit 2 expanded to add the model-side field and `with_convection`/`with_convection_forcing` helpers.
- Commit 6 now focuses on `step!` orchestration.
- Commit 7 split into 4 sub-commits (TransportBinary header, preprocessing pipeline, driver methods, e2e test) — total ~2.5 days.
- Commit 8 expanded to include `_refresh_forcing!` extension, cache invalidation in `_maybe_advance_window!`, and the `meteo = sim` change.

**PRE_PLAN_18_FIXES A3 rewritten:** target `current_time(::DrivenSimulation) = sim.time` + `DrivenSimulation.step!` meteo=sim, NOT `current_time(::TransportBinaryDriver)` using a non-existent `d.current_window_index` field. Effort drops from 0.5-1 day to ~2 hours.

**Interface claims verified against current `src/` tree** at v5
authoring time. If claims drift, Commit 0 re-verifies.

---

End of archived README.
