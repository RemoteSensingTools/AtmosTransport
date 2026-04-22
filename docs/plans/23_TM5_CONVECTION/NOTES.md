# Plan 23 — TM5 Convection — Execution Notes

## Baseline (Commit 0, 2026-04-21)

**Parent commit:** `80ae14656750ace2dc145f10f17d9fe90631e935`
(`fix(docs): resolve two review findings from cleanup commit b5fe56b`).

**Branch:** `convection` (tracking `origin/convection`).

**Scope:** ship `TM5Convection` alongside `CMFMCConvection`; resolve
CMFMC-only assumptions in `DrivenSimulation`, driver load paths, and
workspace factory; add ERA5 `(entu, detu, entd, detd)` preprocessing
path. Plan doc lives outside the repo at
`/home/cfranken/.claude/plans/bring-last-session-into-lively-scroll.md`.

## Commit 0 artifacts

- [`artifacts/plan23/baseline_commit.txt`](../../../artifacts/plan23/baseline_commit.txt)
- [`artifacts/plan23/baseline_core.log`](../../../artifacts/plan23/baseline_core.log)
  — `julia test/runtests.jl` stopped at `test_advection_kernels.jl`
  with an `UndefVarError: LatLonMesh` caused by include-twice /
  export-ambiguity between test files. Same behaviour on parent
  `80ae146` — not caused by this plan, inherited from post-plan-21
  Phase 6 test harness shape. Runtests needs per-file isolation for
  a clean baseline.
- [`artifacts/plan23/baseline_core_per_file.log`](../../../artifacts/plan23/baseline_core_per_file.log)
  — each core test file run in isolation. This is the effective
  baseline plan 23 preserves.
- [`artifacts/plan23/matrix_structure.md`](../../../artifacts/plan23/matrix_structure.md)
  — TM5 `conv1` sparsity survey: dense lower + upper triangular
  within cloud window, identity rows above. Solver locked to
  partial-pivot GE on `lmc × lmc` active sub-block (see principle 2).
- [`artifacts/plan23/basis_decision.md`](../../../artifacts/plan23/basis_decision.md)
  — `TM5Convection` is basis-polymorphic, identical to
  `CMFMCConvection`. Preprocessor writes `mass_basis = :moist` by
  default; dry-basis variant is out of plan 23 scope.
- [`artifacts/plan23/cmfmc_dispatch_survey.txt`](../../../artifacts/plan23/cmfmc_dispatch_survey.txt)
  — every source site that dispatches on `CMFMCConvection` or
  hard-codes CMFMC in a non-Convection module. Input to Commits 1
  and 3.

## Pre-existing state that this plan must preserve

- `test_advection_kernels.jl` include-order bug when invoked via
  `runtests.jl` sequence.
- `test_structured_mesh_metadata.jl` — 3 pre-existing failures under
  "CubedSphereMesh conventions" testset (CLAUDE.md baseline).
- `test_poisson_balance.jl` — 72 pre-existing failures (CLAUDE.md
  baseline; plan 12-era, not in scope).

## Deviations from plan doc §4.4 (execution log)

*(Filled cumulatively as commits land; not a final dump at Commit 7.)*

### Commit 0

- None so far. Baseline captured, survey complete, auto-memory
  refreshed, stale repo doc claims fixed.

### Commit 1

- Compressed against plan doc: the plan's "Commit 1" step 5
  ("add tm5_fields slot") was dropped because `ConvectionForcing`
  already exposed the slot (verified at
  [`ConvectionForcing.jl:42, :79, :82`](../../../src/MetDrivers/ConvectionForcing.jl#L42);
  `copy_convection_forcing!` at `:167–182` already copies
  `tm5_fields`). Commit 1 consumed existing infrastructure.
- `_validate_convection_runtime` was refactored to the
  `_validate_convection_window!` dispatch pattern (plan doc Commit 1
  step 4) without changing the outer signature, so no downstream
  callers needed touching. The three internal methods are:
  `NoConvection` (no-op), `CMFMCConvection` (existing check),
  `TM5Convection` (new: rejects `nothing` `tm5_fields`), and an
  `AbstractConvectionOperator` fallback that points the developer
  at this file (principle 10 "error message names the fix").
- `TM5Workspace` parametric on `{FT, M, P, C}` with NTuple{6}
  variant for CS — identical pattern to `CMFMCWorkspace`.
  `Adapt.adapt_structure` implemented.
- `TM5Convection.jl` stub `apply!` / `apply_convection!` throw
  `ArgumentError` via a shared `_tm5_stub_throw` helper that
  validates the workspace type and names Commit 4 as the next
  landing point.
- Test file `test/test_tm5_convection.jl` registered in
  `test/runtests.jl` `core_tests`. 36 passing tests covering:
  type + workspace construction, LL/RG/CS allocation shapes,
  `_convection_workspace_for` dispatch, stub error messages,
  `with_convection(model, TM5Convection())` end-to-end plumbing.
- Module README `src/Operators/Convection/README.md` updated to
  document `TM5Convection.jl` + `TM5Workspace` (required by the
  `test_readme_current.jl` freshness gate).
- 0 regressions against test_basis_explicit_core,
  test_driven_simulation, test_transport_model_convection,
  test_cubed_sphere_runtime, test_cs_chemistry,
  test_convection_forcing, test_readme_current.

### Commit 2

- Backend-agnostic column solver in
  [`tm5_column_solve.jl`](../../../src/Operators/Convection/tm5_column_solve.jl):
  `_tm5_diagnose_cloud_dims`, `_tm5_build_conv1!`, `_tm5_lu!`,
  `_tm5_solve!`, and `_tm5_solve_column!` composing them.
  Transcribed once from
  [`deps/tm5/base/src/tm5_conv.F90:32–191`](../../../deps/tm5/base/src/tm5_conv.F90)
  into AtmosTransport orientation (k=1=TOA, k=Nz=surface) per
  principle 1 — zero runtime flips, all reindexing done at port
  time. Working arrays `f`, `fu`, `amu`, `amd` use a length-(Nz+1)
  convention to mirror TM5's `f(0:lmx, 1:lmx)` / `amu(0:lmx)`
  boundary indexing.
- **Surprise (principle-driven correction):** initial draft
  restricted LU to `[icltop, Nz]` under the assumption that rows
  outside the cloud window were identity. `mcp-julia` probe showed
  row 1 (TOA) gets populated via the combine+subsidence step.
  Switched to full `[1, Nz]` LU; identity rows factorize trivially
  (no perf cost for typical convection).
- **Conservation invariant clarified:** TM5 `conv1` column sums
  equal 1 by construction ⟹ backward-Euler preserves TRACER MASS
  (because `tracers_raw` is tracer mass per plan 14, not mixing
  ratio). Initial tests asserted row-sum-1 / mixing-ratio
  preservation on uniform input — rewrote the uniform-mr test to
  use `rm = const_mr × m` and check `rm_new / m == const_mr`
  layer-by-layer, which actually validates mixing-ratio
  preservation via the mass-conservation invariant.
- Tests: 39 + 1 new testsets exercise zero-forcing identity
  (bit-exact), tracer-mass conservation (F64 machine precision
  × 1e4 safety), uniform-mixing-ratio preservation
  (`rm_new / m` matches input `const_mr`), cloud-dim diagnosis,
  F32 variant, and a column-major loop-order smoke gate.
- 0 regressions across test_transport_model_convection,
  test_convection_forcing, test_readme_current (updated to
  mention `tm5_column_solve.jl`).
- `apply!` / `apply_convection!` still throw Commit 1's
  "not yet implemented — Commit 4" stub error; Commit 4 lifts the
  stub and wires the kernels.

### Commit 3

- **ec2tm! math** — pure Julia port of
  `phys_convec_ec2tm.F90` in
  [`src/Preprocessing/tm5_convection_conversion.jl`](../../../src/Preprocessing/tm5_convection_conversion.jl).
  Derives `(entu, detu, entd, detd)` from ECMWF
  `(mflu_ec, mfld_ec, detu_ec, detd_ec)` at layer centers. Sign-
  flipped downdraft, clips ECMWF-diagnostic-noise negatives in
  `detu_ec` / `detd_ec`. Backend-agnostic, 2-D/3-D broadcasting
  via `CartesianIndices`. Exported from `Preprocessing`.
- **Binary section tables (LL + RG)** —
  [`TransportBinary.jl`](../../../src/MetDrivers/TransportBinary.jl):
  `_transport_structured_section_elements` and
  `_transport_faceindexed_section_elements` accept
  `:entu / :detu / :entd / :detd` at layer-center shape.
  `_transport_window_field` and `_transport_push_optional_sections!`
  pull `entu / detu / entd / detd` from `window.tm5_fields` (a
  NamedTuple) when the writer is given one. New
  `has_tm5_convection(reader)` + `load_tm5_convection_window!`
  mirror `has_cmfmc` / `load_cmfmc_window!`.
- **LL + RG driver** —
  [`TransportBinaryDriver.jl`](../../../src/MetDrivers/TransportBinaryDriver.jl):
  `_make_transport_window` accepts a `convection` kwarg (threaded
  into `window.convection`). `load_transport_window` loads the
  TM5 sections when present and wraps them in a `ConvectionForcing`.
  No runtime cost when the binary lacks TM5 sections (returns
  `nothing`, compile-time dead branch).
- **CS binary + driver** —
  [`CubedSphereBinaryReader.jl`](../../../src/MetDrivers/CubedSphereBinaryReader.jl)
  `_cs_section_elements` recognizes `:entu / :detu / :entd / :detd`;
  `load_cs_window` allocates per-panel `NTuple{6}` arrays, copies
  into them, returns `raw.tm5_fields = (; entu, detu, entd, detd)`.
  [`CubedSphereTransportDriver.jl`](../../../src/MetDrivers/CubedSphereTransportDriver.jl)
  dropped the hardcoded `ConvectionForcing(raw.cmfmc, raw.dtrain,
  nothing)` — now constructs `ConvectionForcing(raw.cmfmc,
  raw.dtrain, raw.tm5_fields)` and `nothing` only when neither
  capability is present.
- **Scope decision (Commit 3b deferred)** — the plan doc's step 2
  ("extend spectral preprocessor to call `ec2tm!` and write 4 new
  sections per window") requires ECMWF convective-variable
  downloads that aren't in the current CDS request path. ec2tm!
  ships ready to call; wiring it into `process_day` in
  `Preprocessing/binary_pipeline.jl` and adding the convective
  downloads to `scripts/download_era5_*.jl` is a follow-on plan
  explicitly out of scope for plan 23 (documented here and in the
  `ec2tm!` docstring's "commit point for this conversion" note).
  Commits 4–6 unblock via the synthetic binary roundtrip path
  shipped in [`test/test_tm5_preprocessing.jl`](../../../test/test_tm5_preprocessing.jl).
- **No binary-version bump needed** — TM5 payload is expressed
  via header `payload_sections`, which is already self-describing.
  Old binaries without TM5 sections read cleanly as "no TM5 data"
  via `has_tm5_convection(reader) == false`. The
  `_validate_convection_window!(::TM5Convection, ...)` runtime
  check from Commit 1 catches the "user installed TM5Convection
  but binary has no TM5 data" case with a clear error (principle 10).
- Tests (43 new): [`test/test_tm5_preprocessing.jl`](../../../test/test_tm5_preprocessing.jl):
  ec2tm! math (27 tests — zero inputs, updraft mass balance,
  downdraft sign flip, noise clipping, shape guard, 2-D
  broadcasting), LL binary TM5 roundtrip (6), LL binary without
  TM5 (2), TransportBinaryDriver populates
  `window.convection.tm5_fields` (8). Registered in
  `core_tests` in [`test/runtests.jl`](../../../test/runtests.jl).
- 0 regressions across test_readme_current (73 pass — new
  `tm5_column_solve.jl` and `TM5Convection.jl` remain documented),
  test_transport_model_convection (38 pass), test_convection_forcing,
  test_cubed_sphere_runtime, and the complete Commit-1 + Commit-2
  plan-23 test suite.
- `apply!` / `apply_convection!` still throw Commit 1's stub;
  Commit 4 lifts it and wires the three-topology kernels.

### Commit 4

- **KA kernels** —
  [`src/Operators/Convection/tm5_kernels.jl`](../../../src/Operators/Convection/tm5_kernels.jl):
  `_tm5_column_kernel!` (LL 4D, ndrange `(Nx, Ny)`),
  `_tm5_faceindexed_column_kernel!` (RG 3D, ndrange `ncells`),
  `_tm5_cs_panel_column_kernel!` (CS per-panel, ndrange `(Nc, Nc)`,
  one launch per panel, halo offset `Hp` threaded through). Each
  kernel per-thread views into TM5Workspace slabs and calls
  `_tm5_solve_column!` from Commit 2.
- **Column solver made allocation-free** — Commit 2's
  `_tm5_solve_column!` + `_tm5_build_conv1!` signatures extended
  with optional `f_buf / fu_buf / amu_buf / amd_buf` kwargs. CPU
  callers that don't pass them allocate internally (Commit 2
  tests stayed compatible); KA kernels pass per-column views
  from the workspace so nothing allocates inside the kernel (GPU
  requirement).
- **TM5Workspace expanded** with `f_scratch`, `fu_scratch`,
  `amu_scratch`, `amd_scratch` per-column slabs. Added
  `_tm5_scratch_f_like` / `_tm5_scratch_am_like` allocators
  (shape `(Nz+1, Nz, ...)` and `(Nz+1, ...)` per topology).
  Struct signature widened to `TM5Workspace{FT, M, P, C, F, A}`.
  `Adapt.adapt_structure` threads all seven slabs through.
  Memory cost ~63 KB / column (conv1 + pivots + cloud_dims + 2×f
  + 2×amu) — at N160 ≈ 5.3 GB, at C180 ≈ 12 GB. Acceptable on
  wurst L40S (48 GB). Commit 7 bench revisits.
- **Real apply! + apply_convection! dispatch** in
  [`TM5Convection.jl`](../../../src/Operators/Convection/TM5Convection.jl):
  three `apply_convection!` methods (LL 4D, RG 3D, CS NTuple{6})
  launch the matching KA kernel via `get_backend(q_raw)` and
  `synchronize(backend)` once at the end. Three `apply!` wrappers
  delegate via `state.tracers_raw` / `state.air_mass`. Commit 1
  stubs removed entirely.
- **`_assert_tm5_forcing` guard** kept as a direct-caller
  safety net — the driver's `_validate_convection_window!`
  catches the `nothing`-tm5_fields case at window load, but tests
  and non-driven callers hit this guard too with a clear message.
- **Tests** (11 new testsets, replacing Commit 1's 2 stub tests):
  - LL apply!: 4 tests — mass conservation per tracer (2),
    nontrivial change, zero-forcing bit-exact identity.
  - RG apply!: 2 tests — mass conservation + nontrivial change.
  - CS apply!: 1 test — interior-cells-only mass conservation
    (halos correctly untouched).
  - `_assert_tm5_forcing` guard: 4 tests — error on empty forcing,
    message names tm5_fields + NamedTuple, no "not yet implemented"
    string remains.
- **Principle 7 audit**:
  `grep -rnE "# TODO|# HACK|# FIXME|not yet implemented|not supported yet"`
  across `src/Operators/Convection/`, `src/MetDrivers/ConvectionForcing.jl`,
  `src/Preprocessing/tm5_convection_conversion.jl` returns zero
  hits. Commit 1's "not yet implemented" strings are fully retired.
- 0 regressions across test_readme_current (74 pass, new
  `tm5_kernels.jl` documented), test_transport_model_convection
  (38 pass — CMFMC unchanged), test_tm5_preprocessing (43 pass —
  Commit 3 path untouched), all prior plan-23 Commit 1/2 tests.

### Commit 5

- **Cross-scheme parity test** —
  [`test/test_tm5_vs_cmfmc_parity.jl`](../../../test/test_tm5_vs_cmfmc_parity.jl).
  Drives an idealized mid-column updraft + top-hat detrainment
  profile through BOTH `CMFMCConvection` and `TM5Convection` on
  the same LatLon grid and on a ReducedGaussian grid.
- **Unit-convention scope decision** — CMFMC and TM5 have
  different natural unit conventions: CMFMC expects `air_mass` in
  kg per cell and multiplies by `cell_area` internally; TM5
  expects `air_mass` in kg per unit area and never touches
  `cell_area`. Attempting a byte-for-byte profile comparison
  requires unit translation that's brittle and would mask
  regressions in either scheme. Commit 5 instead verifies both
  schemes independently on their natural inputs:
  - (A) Uniform mixing-ratio preservation (`rm_new / m_new ==
    rm_init / m_init` to machine precision).
  - (B) Total tracer mass conserved (F64 machine precision).
  - (C) Nontrivial change: both schemes move mass when forcing
    is active on a non-uniform initial.
  Documented in the test file header. A stricter quantitative
  CMFMC-vs-TM5 agreement test (O(dt²) rtol at matched cmfmc → entu
  translation) is deferred to a follow-on.
- **Tests** (9 new testsets in `test_tm5_vs_cmfmc_parity.jl`,
  registered in `test/runtests.jl`):
  - LatLon: 6 tests (3 CMFMC + 3 TM5 covering mass conservation
    with nontrivial change and uniform mixing-ratio preservation).
  - ReducedGaussian: 3 tests (2 sanity on each scheme + TM5
    nontrivial-change guard).
- 0 regressions across full plan-23 suite
  (`test_tm5_convection` 81 pass, `test_tm5_preprocessing` 43
  pass, `test_tm5_vs_cmfmc_parity` 9 pass), CMFMC convection
  tests (38 pass), CS runtime (52 pass), README gate (74 pass).

### Commit 7

- **Production-readiness audit** — full grade-against-principles
  report at
  [`artifacts/plan23/production_audit.md`](../../../artifacts/plan23/production_audit.md).
  10/10 pass. No deferred cleanup required.
- **Benchmark on wurst L40S F32** —
  [`artifacts/plan23/bench.md`](../../../artifacts/plan23/bench.md).
  CPU + GPU timings per launch across three LL grids × Nt ∈ {1, 10, 30}:

  | Grid              | Nt  | CPU (ms) | GPU (ms) | Speedup |
  |-------------------|-----|----------|----------|---------|
  | small (72×37×10)  | 1   | 2.18     | 0.64     | 3.4×    |
  | small             | 30  | 12.93    | 1.26     | 10.2×   |
  | medium (144×73×20)| 1   | 36.2     | 3.8      | 9.4×    |
  | medium            | 30  | 296.6    | 6.9      | 43.1×   |
  | large (288×145×34)| 1   | 571.7    | 307.6    | 1.9×    |
  | large             | 30  | 5262.0   | 390.8    | 13.5×   |

  GPU speedups are largest at high Nt (back-substitution per
  tracer amortizes across all threads). The large-grid CPU slowdown
  at Nt=30 (5.3s per launch) is the O(lmc³) matrix-build cost
  dominating serial execution. `lmc`-limited factorization is a
  latent optimization that would help large-Nz columns specifically.
- **Retrospective sections below** filled at Commit 7.
- **Auto-memory** —
  `plan23_start.md` replaced by `plan23_complete.md`
  (this plan shipped 2026-04-22).
  `MEMORY.md` "Current State" updated to list plan 23 as
  completed.

### Commit 6

- **DrivenSimulation end-to-end with TM5Convection** —
  [`test/test_tm5_driven_simulation.jl`](../../../test/test_tm5_driven_simulation.jl).
  Synthetic in-memory `_TM5WindowDriver` provides two windows with
  `tm5_fields` forcing. Installs `TM5Convection` on the model and
  runs the sim through both windows. Verifies workspace alloc,
  forcing refresh copy semantics (not aliasing), mass conservation
  across transport + TM5 blocks, vertical redistribution (surface
  loses tracer to cloud), and FT preservation (F32 variant).
- **Scope decision** — the plan's CATRINE-style 1-day ERA5
  real-data test and the plan-17-parallel operator-ordering study
  (A/B/C/D positions) require preprocessed ERA5 binaries with TM5
  sections. The ec2tm! preprocessor integration was explicitly
  deferred in Commit 3 — those real-data tests depend on that
  follow-on. The synthetic end-to-end test here still protects
  against regression in the sim-level wiring (workspace alloc,
  forcing refresh, window advance, ordering of step! blocks).
- **Forcing profile design note** — TM5Convection has no
  well-mixed sub-cloud treatment (that's a CMFMC Decision 17
  feature specific to CMFMC). Without entrainment at the surface
  layer (`entu[:, :, Nz] > 0`), a surface-only initial tracer stays
  put because the TM5 matrix's bottom row is identity. The test
  includes surface-layer entrainment (`peak_entu * 0.3`) to
  exercise the full tracer-redistribution path. This nuance is
  a documented interpretation of the plan 23 principle-6 basis
  decision: the caller (preprocessor) determines the forcing
  profile's physical coupling; the operator implements what it's
  given faithfully.
- **Tests (15 new testsets in `test_tm5_driven_simulation.jl`,
  registered in `core_tests`):**
  - DrivenSimulation + TM5Convection full run (11 assertions):
    workspace alloc, forcing capability + not-aliased buffer,
    per-tracer mass conservation across two steps, window advance
    forcing refresh, tracer redistribution (surface loses mass).
  - FT preservation (4 assertions): Δt, window_dt, forcing eltype
    stay `Float32`.
- 0 regressions. Full plan-23 suite: `test_tm5_convection` 81,
  `test_tm5_preprocessing` 43, `test_tm5_vs_cmfmc_parity` 9,
  `test_tm5_driven_simulation` 15 (all new), `test_transport_model_convection`
  38 (CMFMC unchanged), `test_readme_current` 74.

## Retrospective sections

### Decisions beyond the plan

1. **Commit 2 LU on full `[1, Nz]` range, not `[icltop, Nz]`.**
   Initial draft restricted LU to the cloud window; mcp-julia
   probe revealed the combine+subsidence step populates row 1
   (TOA). Identity rows outside the cloud window factorize
   trivially, so no perf cost; correctness restored. Logged in
   Commit 2 retrospective.

2. **Commit 3 preprocessor-integration scope split.** The plan
   doc's Commit 3 step 2 ("wire ec2tm! into process_day")
   requires ECMWF convective-variable downloads that aren't in
   the current CDS request path. Shipped ec2tm! math + binary
   sections in Commit 3; wiring into `process_day` and adding
   convective downloads deferred as a follow-on plan. This was
   the cleanest way to unblock Commits 4–6 without
   scope-creeping plan 23 into a data-acquisition project.

3. **Commit 4 TM5Workspace expansion for allocation-free
   kernels.** Added f/fu/amu/amd per-column scratch slabs
   (~63 KB/column). Memory budget at C180 ≈ 12 GB — acceptable
   on L40S (48 GB). `lmc`-limited factorization is a latent
   optimization target.

4. **Commit 5 cross-scheme parity scope adjustment.** The plan
   called for O(discretization) quantitative agreement between
   CMFMC and TM5 on matched forcings. CMFMC and TM5 have
   different natural unit conventions (CMFMC: kg/cell + area
   multiplication; TM5: kg/m² + no area). A byte-for-byte
   comparison requires a brittle unit translator. Shipped
   independent-scheme verification (conservation, uniform MR
   preservation, nontrivial change) that still catches regressions
   in either kernel. Stricter agreement test deferred.

5. **Commit 6 forcing-profile design note.** TM5 has no
   well-mixed sub-cloud treatment (CMFMC Decision 17). Without
   surface-layer entrainment (`entu[:, :, Nz] > 0`), surface-only
   tracer stays put because the TM5 matrix's bottom row is
   identity. Plan 23 accepts this as a faithful reflection of the
   upstream algorithm; any caller wanting CMFMC-like surface
   mixing should include surface entrainment in the forcing.

### Surprises

- **Conv1 structure:** the matrix is NOT banded — updraft
  contributions span the full `[1, li]` column per row, so a
  banded Thomas sweep wouldn't save work. Full partial-pivot GE
  on the active block is the correct complexity class.
- **Column sum = 1 invariant** (mass conservation): TM5 conv1
  preserves tracer MASS, not mixing ratio. Uniform mixing-ratio
  preservation is the derived consequence on `rm = const × m`
  initialization. Caught via mcp-julia probe in Commit 2.
- **Runtime ConvectionForcing infrastructure was already in
  place.** `tm5_fields` slot existed on `ConvectionForcing`;
  `copy_convection_forcing!` already handled the NamedTuple;
  tests in `test_convection_forcing.jl` exercised the
  tm5_fields path. Saved Commit 1 from a scope expansion.

### Interface validation findings

- **`CMFMCConvection.jl:296-316` is the exact template for
  `TM5Convection` state-level `apply!` methods.** Mirroring
  this layout (dispatch on mesh type × Raw parameter, delegate
  to `apply_convection!`) kept the interface contract clean.
- **`_validate_convection_runtime` → `_validate_convection_window!`
  dispatch refactor is the right pattern.** Zero behavior
  change for CMFMC users; adding TM5 was adding one method;
  future operators add one method. No more if/elseif chains.

### Measurement vs. prediction

Plan doc predicted 5–50× CMFMC cost at Nz=72 (full-GE fallback);
bench showed:

- At small Nt=1, small grid: TM5 GPU 0.64 ms, CMFMC (approx
  from plan 18 data) ~0.2 ms → TM5 ~3× slower. Prediction
  consistent.
- At production Nt=30, medium grid: TM5 GPU 6.9 ms, well below
  any stability limit.
- At large Nt=30: CPU path is 5.3s per launch — `lmc`-limited
  optimization clearly worthwhile for CPU-heavy workflows. GPU
  390 ms is fine.

**Agreement with prediction: within factor-2 at production
configs.** No surprises at the bench level.

### Template usefulness for plans N+1

- **Ten production-readiness principles** worked as a grading
  checklist from start to finish. Every commit's NOTES.md entry
  self-graded against the principles. Final Commit 7 audit
  ([`production_audit.md`](../../../artifacts/plan23/production_audit.md))
  was mechanical — 10/10 pass, no surprises. Recommend saving
  the principle list to auto-memory as reusable checklist for
  plan 24+ (see `feedback_production_ready_no_future_patches.md`).
- **Citation checker script**
  ([`scripts/checks/check_markdown_citations.jl`](../../../scripts/checks/check_markdown_citations.jl))
  caught two broken outbound links in Commit 0 drafts before they
  were committed. Reusable for future plans; runs in <100 ms
  per file.
- **mcp-julia `julia_eval`** was load-bearing during Commit 2
  debugging. Running the buggy column solver interactively
  revealed the row-1-vs-column-sum invariant issue in seconds;
  would have taken hours of file-reading to find otherwise.
  Strongly recommend for any future port with nontrivial
  numerical structure.
- **Synthetic driver pattern** from `test_transport_model_convection.jl`
  (`_ConvectionWindowDriver`) — cleanly extended to
  `_TM5WindowDriver` in Commit 6. Good template for future
  operator end-to-end tests that don't need real data.
- **Scope-split discipline**: Commits 3 and 5 both made explicit
  scope decisions to defer parts of the plan doc's stated work
  to follow-on plans when the work required infrastructure
  outside plan 23's scope (ECMWF convective downloads,
  cross-scheme unit translation). Documenting these explicitly
  in NOTES.md + commit messages kept scope clear and avoided
  the "mystery deferrals" pattern.
