# Plan 17 Execution Notes — Surface Emissions Operator

**Plan:** [17_SURFACE_EMISSIONS_PLAN.md](../17_SURFACE_EMISSIONS_PLAN.md) (v1)
— builds on plan 16b's palindrome-centered `V` integration to add
`S` at palindrome center (`X Y Z V(dt/2) S V(dt/2) Z Y X`). Ships
`StepwiseField{FT, N}` for piecewise-constant monthly/annual inventories.
Resolves plan-15's `DrivenSimulation` chemistry-ordering workaround.

## Baseline

- **Parent commit:** `6bbf512` (plan 16b Commit 7, tip of `vertical-diffusion`).
- **Branch:** `surface-emissions` created from `vertical-diffusion`
  HEAD per CLAUDE_additions.md §"Branch hygiene" stack-branches rule.
- **Pre-existing test failures:** 77, stable since plan 12
  (`test_basis_explicit_core: 2`, `test_structured_mesh_metadata: 3`,
  `test_poisson_balance: 72`). Plan 17 preserves this count.
- **Fast-path baseline** captured in
  [../../../artifacts/plan17/baseline_test_summary.log](../../../artifacts/plan17/baseline_test_summary.log).

## Pre-Commit-0 survey findings

Contrary to plan doc §4.1 assumption of "minimal existing emissions
code", an active `SurfaceFluxSource` infrastructure exists. Full survey
in [../../../artifacts/plan17/existing_emissions_survey.txt](../../../artifacts/plan17/existing_emissions_survey.txt).
Summary:

1. **Primary path** — [SurfaceFluxSource](../../../src/Models/DrivenSimulation.jl):
   `cell_mass_rate::RateT` in **kg/s per cell**, applied at `rm[:,:,Nz]`
   (k=Nz surface). Called inline from `DrivenSimulation.step!` between
   transport and chemistry.
2. **Secondary path (dead)** — [`_inject_source_kernel!`](../../../src/Kernels/CellKernels.jl#L43):
   exported but uncalled in `src/`. Per-area `kg/m²/s × areas[j] × dt`
   convention. Only `scripts_legacy/` uses a similarly-named function.
   Plan 17 does NOT consume or remove it; a future cleanup plan can.
3. **`StepwiseField` is genuinely new** — only hint is a comment at
   [PreComputedKzField.jl:13](../../../src/State/Fields/PreComputedKzField.jl#L13)
   anticipating this type.

[StepwiseField survey](../../../artifacts/plan17/existing_stepwise_survey.txt).

## Deviations from plan doc §4.4

Three scope revisions recorded up front, resolved with user on
2026-04-19 (plan execution file [~/.claude/plans/you-can-compact-context-mutable-token.md](file:///home/cfranken/.claude/plans/you-can-compact-context-mutable-token.md)):

1. **Commit 3 MIGRATES existing `SurfaceFluxSource`** into
   `src/Operators/SurfaceFlux/` rather than building greenfield.
   Re-export from original location for backward compat for one
   release. `PerTracerFluxMap` is `NTuple{N, SurfaceFluxSource}`-backed,
   matching existing `sim.surface_sources::Tuple` storage — plan doc
   recommended Dict, but the existing NTuple is simpler and type-stable.

2. **`k = Nz` = surface convention kept** (plan doc Decision 4 had
   `k = 1` which would silently emit into the top model layer). Matches
   existing [DrivenSimulation.jl:126](../../../src/Models/DrivenSimulation.jl#L126)
   and all LatLon callers. No `AbstractLayerOrdering{TopDown, BottomUp}`
   abstraction in plan 17 — driver-level flip at read time is the
   current contract (CLAUDE.md Invariant 2). A dedicated layer-ordering
   plan is a follow-up candidate (sibling to plan 18).

3. **Flux kept as `kg/s` per cell** (plan doc Decision 4 had
   `kg/m²/s × cell_area × dt`). Prognostic tracer is mass
   ([CellState.jl:23](../../../src/State/CellState.jl#L23)), so
   `rm[i, j, Nz, t] += rate[i, j] * dt` is dimensionally consistent.
   Acceptance: Commit 3 test asserts that summed `rate × dt × step_count`
   equals global Earth-total emissions over the run window. Per-area
   flux variant is a follow-up candidate (some CATRINE inventories
   arrive in kg/m²/s).

4. **Commit 6 simpler than plan doc** — since existing `DrivenSimulation`
   already owns `surface_sources`, Commit 6 is "move the call site into
   the palindrome via `TransportModel.emissions`, install via
   `with_emissions` at sim construction, delete the 2 sim-level
   application lines + 1 workaround line" rather than building
   `SurfaceFluxOperator` plumbing from zero.

## Follow-up plan candidates (not shipped in 17)

Recorded here at Commit 0 so the retrospective at Commit 9 has a
pre-registered list:

- **`AbstractLayerOrdering{TopDown, BottomUp}`** — cross-cutting type
  for operators to dispatch on vertical-layer convention. Motivated
  by GEOS-FP (top-down) vs GEOS-IT (bottom-up). Touches every
  vertically-indexed kernel.
- **Per-area flux variant** — `kg/m²/s` × `cell_area` sibling to
  `SurfaceFluxSource`, for inventories that arrive per-area.
- **Stack emissions** (3D source fields) — non-surface emission layers
  (tall stacks, aviation, volcanic plumes).
- **`DepositionOperator`** — dry deposition as a first-class operator.
- **Remove `_inject_source_kernel!`** — dead code in
  [src/Kernels/CellKernels.jl](../../../src/Kernels/CellKernels.jl).

## Commit sequence (initial draft; will evolve)

- **Commit 0** — NOTES.md + baseline + surveys + memory compaction.
- **Commit 1** — `StepwiseField{FT, N, A, T}` + tests.
- **Commit 2** — `PerTracerFluxMap{FT, Sources}` + tests.
- **Commit 3** — Migrate `SurfaceFluxSource` + `AbstractSurfaceFluxOperator`,
  `NoSurfaceFlux`, `SurfaceFluxOperator`, kernel, `apply!`,
  `apply_surface_flux!` array-level.
- **Commit 4** — `current_time(meteo)` threading through chemistry
  and diffusion `apply!`.
- **Commit 5** — Palindrome integration in `strang_split_mt!`.
- **Commit 6** — `TransportModel.emissions` + `with_emissions` +
  `DrivenSimulation` cleanup.
- **Commit 7** — Ordering study.
- **Commit 8** — Benchmarks.
- **Commit 9** — Retrospective + `ARCHITECTURAL_SKETCH_v3.md`.

## Commit-by-commit notes

### Commit 0 — NOTES + baseline

This file. Surveys saved to `artifacts/plan17/`. Baseline test summary
log captured. Plan docs `17_SURFACE_EMISSIONS_PLAN.md`,
`ARCHITECTURAL_SKETCH_v2.md`, `CLAUDE_additions.md` (pre-existing
untracked, authored by user) included in Commit 0.

### Commit 1 — `StepwiseField{FT, N, A, B, W}`

Created [src/State/Fields/StepwiseField.jl](../../../src/State/Fields/StepwiseField.jl).
Wired through `Fields → State → AtmosTransport` export chain; also
exported `integral_between` (first concrete type to ship this
previously-reserved optional method).

**Design surprises encountered:**

1. **`Ref{Int}` is not kernel-safe.** Initial draft stored
   `current_window::Base.RefValue{Int}`. Adapt would have carried a
   host-side Ref into kernel closures — subsequent `current_window[]`
   reads on GPU would pull from host memory (or error). Switched to a
   1-element `AbstractVector{Int}` which adapts naturally to
   `CuArray{Int, 1}`.

2. **Adapt.adapt_structure cannot use the validating inner constructor.**
   `issorted(boundaries::CuDeviceVector)` tries to iterate device
   memory on the host and errors. Added a second inner constructor
   gated by `Val(:unchecked)` that `Adapt.adapt_structure` calls
   after the host-side object already validated. Pattern to reuse
   in future field types that hold device-convertible metadata.

3. **5-arg outer constructor needed for Adapt.** Adapt's generic
   reconstruction calls `StepwiseField(samples_new, boundaries_new, current_window_new)`
   (3 args). The original outer constructor was 2-arg
   `(samples, boundaries)`, adding a fresh `current_window = [1]`.
   Added a 3-arg outer that preserves the adapted `current_window`.

**Test state** (relative to Commit 0 baseline):

- `test_fields.jl`: 300 pass → 343 pass (+43 new `StepwiseField` tests).
- `test_chemistry.jl`: 37 pass (unchanged).
- `test_diffusion_palindrome.jl`: 27 pass (unchanged).
- `test_transport_model_diffusion.jl`: 24 pass (unchanged).
- Baseline 77 failure count preserved.

**Deviation from plan doc §4.4 Commit 1 spec:** none. Plan doc's eight
tests become our 19 testsets totaling 43 assertions. The mass-accounting
invariant (user acceptance criterion for plan 17 Decision 1) was added
as an extra test here rather than deferred to Commit 3 — it exercises
only `StepwiseField` + `integral_between` and has no other dependencies.

### Commit 2 — Migrate `SurfaceFluxSource` + `PerTracerFluxMap`

Created [src/Operators/SurfaceFlux/](../../../src/Operators/SurfaceFlux/)
and moved `SurfaceFluxSource` + surface-slice helpers
(`_surface_shape`, `_check_surface_source_compatibility`,
`_apply_surface_source!`) out of
[src/Models/DrivenSimulation.jl](../../../src/Models/DrivenSimulation.jl).
Added `PerTracerFluxMap{S <: Tuple}` NTuple-backed lookup structure.

**Surprises:**

1. **`using ..Operators.SurfaceFlux: ...` at file scope (not module
   scope).** DrivenSimulation.jl needed the migrated helpers. Added
   the `using` mid-file inside the non-module DrivenSimulation.jl
   rather than in the enclosing `Models.jl`. Julia accepts this at
   top-level file scope, but the IDE lint-analyzer loses track of
   subsequent name resolution and emits spurious "unused binding"
   warnings for everything below the mid-file import. False positives,
   no functional issue.

2. **Exports must be surfaced explicitly at `AtmosTransport.jl` level.**
   `using .Operators` in `src/AtmosTransport.jl` brings Operators-
   exported names into scope, but those names are only re-exported
   from AtmosTransport if added to the `export` line. `SurfaceFluxSource`
   was already listed (from the old Models export); `PerTracerFluxMap`
   and `flux_for` had to be added explicitly. Easy to miss during a
   migration because the module loads fine either way.

3. **`unique(names) == names` is the right duplicate check**, not just
   `length(unique) == length(names)`. Both work, but the length form
   is idiomatic Julia for this use case and catches subtle iteration-
   order bugs if the NTuple ordering is ever changed.

**Test state** (relative to Commit 1):

- `test_per_tracer_flux_map.jl`: 36 new tests (construction variants,
  duplicate rejection, lookup, iteration, Adapt CPU + GPU).
- `test_driven_simulation.jl`: 57 pass unchanged. `SurfaceFluxSource`
  still usable by-name via backward-compat export.
- `test_fields.jl`, `test_chemistry.jl`,
  `test_diffusion_palindrome.jl`, `test_transport_model_diffusion.jl`:
  all pass unchanged.
- 77 baseline failure count preserved.

**Deviation from plan doc §4.4 Commit 2 spec:** plan doc suggested
Dict-backed map; chose NTuple-backed instead. Rationale in NOTES §
"Deviations from plan doc §4.4" item 1 above. Plan doc also listed
Commit 3's migration of `SurfaceFluxSource` as a separate task;
pulling it into Commit 2 was necessary because the module include
order (Operators before Models) forced the type to move before any
`SurfaceFluxOperator` could reference it.

### Commit 3 — `SurfaceFluxOperator` + kernel + `apply!`

Created `AbstractSurfaceFluxOperator` hierarchy with `NoSurfaceFlux`
(identity) and `SurfaceFluxOperator{M}` (wraps `PerTracerFluxMap`).
Shipped:

- [surface_flux_kernels.jl](../../../src/Operators/SurfaceFlux/surface_flux_kernels.jl)
  — `_surface_flux_kernel!(q_raw, rate, dt, tracer_idx, Nz)` writes
  `q_raw[i, j, Nz, tracer_idx] += rate[i, j] × dt` over the (Nx, Ny)
  grid.
- [operators.jl](../../../src/Operators/SurfaceFlux/operators.jl) —
  type hierarchy, state-level `apply!`, array-level
  `apply_surface_flux!` (for Commit 5 palindrome use), Adapt
  structures, `emitting_tracer_indices` introspection helper.

**Design choices:**

1. **One kernel launch per emitting source, not a fused multi-tracer
   kernel.** CPU-side `findfirst(==(name), tracer_names)` resolves
   each source's index before launch. For typical N ≤ 10 emitting
   tracers the launch overhead is negligible compared to O(Nx · Ny)
   kernel work. A fused kernel would need the rate tuple as a
   captured argument; tuples of arrays are supported on GPU via
   Adapt but add complexity without measurable perf benefit at
   small N.

2. **Array-level `apply_surface_flux!` takes `tracer_names` as a
   kwarg.** The palindrome (Commit 5) operates on a raw 4D buffer
   that is NOT necessarily `state.tracers_raw` — it can be the
   workspace's ping-pong alternate. Passing `tracer_names`
   explicitly lets the entry point work without a CellState
   reference.

3. **Tracers in map but absent from state are silently skipped.**
   CATRINE configs carry a superset of sources (fossil_co2, SF6,
   Rn222, CH4, …); an individual run might only advect a subset.
   Throwing an error on missing tracers would force configs to
   match runs exactly. Silent skip is the friendlier default;
   users who want strict checking can call
   `_check_surface_source_compatibility` explicitly.

4. **Empty-variadic `SurfaceFluxOperator()` is allowed.** Returns
   an operator with a zero-length map. Semantically identical to
   `NoSurfaceFlux` but type-distinguishable (useful for configs that
   toggle emission presence without rebuilding the TransportModel).

**Surprises:**

1. **`@kernel function _surface_flux_kernel!(..., @Const(rate), ...)`
   triggers the IDE static analyzer's "unused argument" warning.**
   The `@kernel` macro expands `rate` into indexed access inside
   the generated struct, but the analyzer walks the pre-macro AST.
   Same false positive as plan 16b Commit 2's diffusion kernel.

2. **IDE also complains about `NoSurfaceFlux`'s explicit signature
   arguments (`meteo`, `grid`, `workspace`, `dt`, `tracer_names`)
   being "unused" in the dead-branch method.** Interface-consistency
   arguments required by the operator contract are still hints-
   flagged as unused by the linter. False positive; these are
   necessary for multiple-dispatch correctness.

**Test state** (relative to Commit 2):

- `test_surface_flux_operator.jl`: 38 new tests covering type
  hierarchy, constructor variants, `NoSurfaceFlux` dead branch (state
  and array levels), k=Nz surface-only write, upper layers unchanged,
  rate × dt arithmetic (F64 + F32), per-tracer index resolution,
  skipping absent tracers, untouching non-emitting tracers, global
  mass accounting (acceptance criterion), array-level
  `apply_surface_flux!` on arbitrary 4D buffers, `emitting_tracer_indices`,
  repeated apply accumulation, Adapt CPU + GPU, GPU vs CPU agreement.
- All pre-existing test files unchanged. 77 baseline failures preserved.

**Cumulative new tests plan 17 Commits 1-3:** 43 + 36 + 38 = 117.

---

Subsequent commit sections will be filled in as execution proceeds
(per CLAUDE.md §"Plan execution rhythm — Retrospective sections are
cumulative").
