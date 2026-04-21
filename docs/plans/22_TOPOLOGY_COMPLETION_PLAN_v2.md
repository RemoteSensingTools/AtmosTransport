# Plan 22 v2 — Topology Completion Roadmap

**Status:** proposed replacement for the current monolithic
`22_TOPOLOGY_COMPLETION_PLAN.md`

**Audience:** hand-off plan for Claude or any other execution agent

**Primary goal:** finish operator support across LatLon, ReducedGaussian
(RG), and CubedSphere (CS) without forcing one fake "topology-agnostic"
implementation strategy onto code paths that currently have very
different runtime maturity.

## 1. Executive summary

The previous plan treats RG and CS as the same class of problem. They
are not.

- **ReducedGaussian** is mostly a **column-local operator completion**
  problem. The grid, state, face-indexed fluxes, met-driver path, and
  `TransportModel` / `DrivenSimulation` runtime already exist.
- **CubedSphere** is still a **runtime enablement** problem. Advection
  kernels exist, but the standard model / driver path still rejects CS
  and the live CS runner uses a separate panel-oriented workflow.

So this should not be executed as one branch or one commit stack. The
correct shape is:

1. **Plan 22A:** RG column-local operators
2. **Plan 22B:** CS runtime enablement
3. **Plan 22C:** CS column-local operators

This document is the roadmap tying those three plans together.

## 2. Current ground truth in `src`

These are the load-bearing facts the plan must respect:

- RG advection already exists through the face-indexed path in
  `src/Operators/Advection/StrangSplitting.jl`.
- RG transport windows already load through
  `src/MetDrivers/TransportBinaryDriver.jl`.
- `CellState` and `FaceIndexedFluxState` already support RG-style
  `(ncells, Nz, Nt)` and `(nfaces, Nz)` storage in
  `src/State/CellState.jl` and `src/State/FaceFluxState.jl`.
- `ImplicitVerticalDiffusion` is still structured-only in practice:
  rank-3 Kz contract, rank-4 tracer buffer contract, and structured
  workspace assumptions in `src/Operators/Diffusion/`.
- `SurfaceFluxOperator` is still structured-only in practice at the
  array-level entry point in `src/Operators/SurfaceFlux/operators.jl`,
  even though `SurfaceFluxSource` helpers already understand
  face-indexed surface shapes in `src/Operators/SurfaceFlux/sources.jl`.
- `CMFMCConvection` is currently structured-only and explicitly rejects
  face-indexed state in `src/Operators/Convection/CMFMCConvection.jl`.
- `TransportModel.step!` still does **not** execute a convection block.
  Convection topology work is therefore gated on the remaining plan-18
  structured runtime wiring landing first.
- CS advection exists as a panel-oriented path:
  `strang_split_cs!` in `src/Operators/Advection/CubedSphereStrang.jl`
  and the standalone runner in `scripts/run_cs_transport.jl`.
- The standard runtime still rejects CS:
  `TransportModel` and structured advection throw metadata-only errors
  for `CubedSphereMesh`.
- CS forcing I/O is not in `TransportBinaryDriver`; it lives in the
  separate `src/MetDrivers/CubedSphereBinaryReader.jl`.

## 3. Hard decisions

### Decision 1: keep the public operator API uniform, but allow
specialized kernels underneath

The target is a common high-level contract:

```julia
apply!(state, meteo_or_forcing, grid, op, dt; workspace)
```

That does **not** imply one generic kernel for every topology.

- LatLon should keep its existing structured fast paths.
- RG can use dedicated face-indexed kernels.
- CS can use per-panel wrappers around structured kernels.

### Decision 2: do not regress the structured fast path in pursuit of
"elegance"

Structured LatLon is the mature performance path. Plan 22 must not
replace proven structured kernels with a generic abstraction unless
benchmarks show the result is flat within noise.

### Decision 3: RG comes before CS

RG already has the runtime. CS does not. Do the tractable operator
completion first.

### Decision 4: CS must preserve the existing panel-oriented advection
boundary

Do **not** rewrite CS advection as part of topology completion. The
existing panel-oriented `strang_split_cs!` path is the production
behavior that runtime work must adapt to.

If the eventual model-level storage is packed, it must expose zero-copy
per-panel views to advection. Do not force a packed-5D advection rewrite
into this effort.

### Decision 5: convection is a gated track

Do not promise RG or CS convection end-to-end until all of the following
are true on structured LatLon:

- convection is wired into `TransportModel.step!`
- `DrivenSimulation` refreshes model forcing correctly
- the operator to be extended actually exists in `src`

Today that means:

- `CMFMCConvection` is potentially in scope after the structured runtime
  wiring lands
- `TM5Convection` is **not** a blocking prerequisite for 22A/22B/22C if
  it has not landed yet

### Decision 6: public dispatch is topology/type-based, not array-rank-based

Plan 20 should describe the **entrypoint dispatch**, not every storage
detail inside lower-level kernels.

- `apply!` / `TransportModel` / `DrivenSimulation` should dispatch on
  mesh and runtime container types.
- Internal array-level kernels may still be storage-specialized where
  that keeps the implementation honest and fast.
- Do **not** describe "rank-4 = LatLon, rank-3 = ReducedGaussian" as the
  architectural story. That is an implementation detail of some
  existing kernels, not the runtime contract 22B should extend.

## 4. Plan 22A — ReducedGaussian column-local operators

**Goal:** make RG support the same column-local operator suite that
LatLon already has, using the existing face-indexed runtime.

**Scope:**

- face-indexed vertical diffusion
- face-indexed surface flux
- face-indexed CMFMC convection, but only if the structured convection
  runtime is already live when this plan starts

**Non-goals:**

- new RG advection algorithm work
- CS anything
- TM5 work if `TM5Convection` is not already in `src`

### Commit 0: baseline and gating check

- Confirm the RG runtime is green with current tests:
  - `test/test_basis_explicit_core.jl`
  - `test/test_transport_binary_reader.jl`
  - `test/test_diffusion_operator.jl`
  - `test/test_surface_flux_operator.jl`
- Record whether structured convection is actually live. If not, mark
  RG convection as deferred and continue with diffusion + surface flux.
- Create `artifacts/plan22/perf/` and capture baseline LatLon overhead
  numbers with:
  - `scripts/benchmarks/bench_diffusion_overhead.jl`
  - `scripts/benchmarks/bench_emissions_overhead.jl`

### Commit 1: face-indexed diffusion

- Extend the diffusion operator contract so a face-indexed Kz field is
  legal. The type/constructor contract must no longer hardcode
  structured rank-3 only.
- Add a face-indexed array-level path operating on:
  - `q_raw :: (ncells, Nz, Nt)`
  - `dz_scratch :: (ncells, Nz)`
  - `w_scratch :: (ncells, Nz)`
- Use a dedicated RG kernel or method if that is simpler and faster than
  a generic rank-polymorphic kernel.
- Add tests for:
  - `ConstantField` / profile / precomputed Kz on face-indexed state
  - mass conservation
  - structured-vs-unrolled equivalence on a toy grid
  - `TransportModel` / `DrivenSimulation` RG diffusion path

### Commit 2: face-indexed surface flux

- Extend the array-level surface-flux path to accept
  `q_raw :: (ncells, Nz, Nt)`.
- Reuse the existing surface-shape conventions already present in
  `src/Operators/SurfaceFlux/sources.jl`.
- Keep the per-emitter launch pattern unless profiling proves it is a
  bottleneck.
- Add tests for:
  - single-emitter RG surface addition
  - multiple tracers / skipped tracers
  - structured-vs-unrolled equivalence
  - RG palindrome integration through `TransportModel.step!`

### Commit 3: RG end-to-end transport block

- Verify the face-indexed advection path composes cleanly with the new
  diffusion and emissions operators through `TransportModel.step!`.
- Add a real RG smoke test using `TransportBinaryDriver` and
  `DrivenSimulation` with the new operators enabled.

### Commit 4: RG convection, if and only if the structured runtime is live

- Extend `CMFMCConvection` from structured `(Nx, Ny, Nz, Nt)` to
  face-indexed `(ncells, Nz, Nt)`.
- Keep the same numerical behavior:
  - same basis contract
  - same optional-`dtrain` semantics
  - same mandatory CFL subcycling
- Use the existing face-indexed rejection tests in
  `test/test_cmfmc_convection.jl` as the scaffold for the new positive
  path.
- If structured convection is still not live when this commit is
  reached, stop and split RG convection into a follow-up plan instead of
  inventing a parallel runtime.

### Commit 5: RG perf and docs

- Extend the existing diffusion and emissions overhead benches with a
  `topology = :latlon | :rg` switch rather than creating brand-new
  scripts.
- Only add a convection overhead bench if RG convection actually lands.
- Write `artifacts/plan22/perf/SUMMARY_RG.md`.

### 22A acceptance criteria

- RG `TransportModel` runs with advection + diffusion + surface flux.
- RG `DrivenSimulation` runs with real transport windows and the same
  operators.
- Structured LatLon tests remain green.
- Structured LatLon overhead remains within about `±5%` on the existing
  benchmarks, or any larger change is explained and justified.

## 5. Plan 22B — CubedSphere runtime enablement

**Goal:** make CS a real runtime path through model + driver without
changing CS advection mathematics.

**Scope:**

- CS state / flux / window / driver architecture
- `TransportModel` and `DrivenSimulation` support for CS advection
- replacing the current metadata-only API contract

**Non-goals:**

- diffusion / surface flux / convection kernels
- changing `strang_split_cs!` kernel math

### Commit 0: pick the runtime representation

Before writing production code, make the representation choice explicit
in the roadmap. Do **not** leave this as an open design question.

Chosen rule:

- the **operator boundary** stays panel-oriented because that is what
  current CS advection already consumes
- the runtime storage is **panel-native and haloed**
- `air_mass` is `NTuple{6}` of per-panel `(Nc + 2Hp, Nc + 2Hp, Nz)`
  arrays
- `tracers_raw` is `NTuple{6}` of per-panel
  `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)` arrays
- `am`, `bm`, `cm` are `NTuple{6}` of per-panel flux arrays matching
  the existing `strang_split_cs!` boundary
- halo exchange remains explicit at the operator boundary

That is the representation 22B should implement. If a future refactor
wants wrapper arrays or a more generic state hierarchy, it must preserve
this panel-native boundary rather than erase it.

### Commit 1: introduce CS runtime types

Implement the chosen representation with a dedicated CS runtime family:

- dedicated CS state type carrying panel-native air mass + tracer storage
- dedicated CS flux type carrying panel-native `am` / `bm` / `cm`
- dedicated CS transport window / driver types

Do **not** pretend that the current `CellState` plus
`StructuredFaceFluxState` constructors already solve CS. They do not.

### Commit 2: CS met-driver path

- Add a real CS transport-window / driver path backed by
  `CubedSphereBinaryReader`.
- Prefer a dedicated CS driver over forcing `TransportBinaryDriver` to
  carry semantics it does not currently model.
- Preserve the separation between reader-owned timing and model-owned
  state.

### Commit 3: CS `TransportModel` / `DrivenSimulation` support

- Add model constructors and `apply!` dispatch that route CS advection
  through `strang_split_cs!`.
- Replace the current metadata-only throws only after the real path is
  green.

### Commit 4: replace the metadata-only test contract

- Remove or rewrite the current "Honest metadata-only CubedSphere API"
  expectations in `test/test_basis_explicit_core.jl`.
- Replace them with:
  - CS model construction smoke tests
  - CS step conservation tests
  - CS driver/window loading tests

### Commit 5: CS advection runtime perf check

- Compare the new runtime path against the existing standalone
  `scripts/run_cs_transport.jl` shape.
- The goal is not zero overhead, but no obviously bad architecture:
  no host allocations in the step loop, no unnecessary repacking, no
  panel copies just to satisfy the model API.

### 22B acceptance criteria

- CS no longer throws metadata-only errors in the main runtime.
- CS advection works through the same model / simulation entry points as
  the other grids.
- The runtime does not require a rewrite of `strang_split_cs!`.

## 6. Plan 22C — CubedSphere column-local operators

**Goal:** add diffusion, surface flux, and convection on top of the new
CS runtime.

**Scope:** per-panel column-local operator support only after 22B lands.

### Commit 1: CS diffusion

- Reuse the structured diffusion implementation per panel.
- Do not force a single 4D/5D generic kernel if per-panel structured
  dispatch is simpler and keeps performance predictable.
- Add conservation and structured-vs-panel-equivalence tests.

### Commit 2: CS surface flux

- Reuse the structured surface-flux implementation per panel.
- Keep emitter-to-tracer resolution on the host just as LatLon does.

### Commit 3: CS convection

- Only after structured convection is live and 22B has established a CS
  forcing path.
- Implement `CMFMCConvection` first.
- Do not block 22C on `TM5Convection` if TM5 is still absent from the
  tree.

### Commit 4: CS end-to-end tests and perf

- Add CS `TransportModel` / `DrivenSimulation` tests with the new
  operators active.
- Measure overhead against the 22B advection baseline.

### 22C acceptance criteria

- CS supports the same column-local operator suite that has shipped on
  LatLon, modulo any operator that still does not exist in `src`.
- The operator boundary remains compatible with panel-oriented CS
  advection.

## 7. Performance plan

This roadmap needs explicit perf discipline.

### 7.1 Rules

- **Measurement is the gate.** Do not estimate from launch-count math.
- **Protect structured LatLon first.** That is the current fast path.
- **No host allocations in hot loops.**
- **Prefer extending existing bench scripts** over creating a new harness
  for every operator/topology combination.

### 7.2 What to benchmark

- `scripts/benchmarks/bench_strang_sweep.jl` for structured advection
  baseline
- `scripts/benchmarks/bench_diffusion_overhead.jl`
- `scripts/benchmarks/bench_emissions_overhead.jl`
- add a convection bench only when convection is actually live end-to-end

### 7.3 Concrete kernel guidance

- **Diffusion:** keep one thread per column-tracer and loop vertically
  inside the kernel; do not launch one kernel per level.
- **Surface flux:** one launch per emitting tracer is acceptable unless
  profiling proves otherwise.
- **RG:** avoid reshape/scatter/gather glue in every step. Operate on the
  natural face-indexed layout directly.
- **CS:** the biggest risk is accidental repacking or six sequential
  host-side copies, not the existence of six panel launches by itself.

### 7.4 Soft guardrails

- Structured LatLon regression target: within about `±5%` on the
  existing overhead benches.
- CS runtime target: no obvious architectural regression relative to the
  current standalone CS runner.
- New RG / CS operator costs should be documented, not hand-waved.

## 8. What Claude should not do

- Do not execute 22A, 22B, and 22C in one branch.
- Do not promise CS by "just making kernels rank-agnostic."
- Do not block RG diffusion and surface flux on unfinished CS runtime
  work.
- Do not block topology completion on `TM5Convection` if it is still not
  present.
- Do not wire a second, plan-local convection runtime just to make 22A
  look complete before plan 18 is actually finished.

## 9. Recommended execution order

If I were driving:

1. Finish the remaining structured convection runtime wiring from plan 18
   or explicitly defer topology-convection work.
2. Execute **22A** through RG diffusion + RG surface flux.
3. Add RG convection only if the structured convection runtime is truly
   ready.
4. Execute **22B** as a standalone CS runtime plan.
5. Execute **22C** only after 22B is stable.

That is the lowest-risk path that actually matches the state of the
current tree.
