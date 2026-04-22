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

## Retrospective sections (filled during execution)

### Decisions beyond the plan

*(Filled as they happen.)*

### Surprises

*(Filled as they happen.)*

### Interface validation findings

*(Filled as they happen.)*

### Measurement vs. prediction

*(Filled at Commit 7.)*

### Template usefulness for plans N+1

*(Filled at Commit 7.)*
