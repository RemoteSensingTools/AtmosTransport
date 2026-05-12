# Plan 26 — TM5-style footprint + inversion scaffold

Branch: `convection`
Owner: claude (with codex review at every commit)
Started: 2026-05-11
Status: planning
Parent plan: `/home/cfranken/.claude/plans/how-does-tm5-handle-joyful-marble.md`

## Goal

Extend the existing AtmosTransport.jl footprint + 4D-Var prototype toward
TM5-cy3-4DVar architectural parity. Concretely close three gaps:

1. **In-memory tape** → add a NetCDF-backed tape storage policy + sliding
   window checkpoint scheduler so 2-week C180 inversions become feasible
   (peak RAM bounded by one replay window, not the full tape).
2. **Diagonal-only background** → replace with B = (per-cell σ) ×
   (horizontal correlation) × (temporal correlation), and evaluate
   cost/gradient in preconditioned space χ ≡ B^(−1/2)(x − x_b).
3. **No quasi-Newton optimizer** → add L-BFGS via `Optim.jl`, retain the
   current gradient-descent as a fallback.

Loose-compat with TM5 NetCDF conventions: same concepts (per-tracer groups,
`date_components`, `mixing_ratio` + sigma) and group/variable layout, but
Julia-native variable names and a versioned `cs_observations_v1.toml`
schema doc.

## Pre-context

- Plan 25 Commit 6 (`5150d50`, on `convection`) integrates LinRood into
  `cs_surface_emission_footprint` with cross-panel halo adjoint and
  `_CSLinRoodHorizRecord`. Plan 26 builds on that.
- `src/Adjoints/Adjoints.jl` is currently ~3700 lines and mixes adjoint
  kernels, tape policies, the footprint API, and the 4D-Var driver. Phase 0
  splits it into focused folders before adding new functionality.

## Folder restructure (Phase 0)

```
src/
├── Adjoints/    ← reverse-mode kernels only (per-physics adjoints + halo + objective seeding)
├── Tape/        ← NEW — storage policies, tape records, checkpointing
├── Footprint/   ← NEW — cs_surface_emission_footprint API + reverse-loop driver
└── Inversion/   ← NEW — observations, covariance, preconditioner, optimizer, 4D-Var driver
```

Module dependency order: **Adjoints → Tape → Footprint → Inversion**.

Phase 0 is pure code motion (no semantic change). Existing 95 adjoint
tests must stay green every commit.

## Staged commit plan

Each commit gets `codex review --uncommitted` before landing.

### Phase 0 — refactor

- **P0.0** — this NOTES file (plan-only commit, codex-reviewed).
- **P0.1** — extract `src/Tape/` with `TapeStorage.jl` + `TapeRecords.jl`.
- **P0.2** — split `src/Adjoints/` into `ObjectiveSeeding.jl`,
  `AdvectionAdjoint.jl`, `DiffusionAdjoint.jl`, `ConvectionAdjoint.jl`,
  `HaloAdjoint.jl`. `LinRoodTape.jl` already focused.
- **P0.3** — create `src/Footprint/` with `FootprintResult.jl`,
  `TapeRecording.jl`, `ReverseLoop.jl`, `FootprintAPI.jl`.
- **P0.4** — create `src/Inversion/` with `Observations.jl`, `Jacobian.jl`,
  `CostGradient.jl`, `Optimizer.jl` (gradient-descent shim).

### Phase A — on-disk tape (lands in `src/Tape/`)

- **A1** — `NetCDFCSTapeStorage <: AbstractCSTapeStorage` +
  `test_cs_tape_netcdf_roundtrip.jl`.
- **A2** — `CSCheckpointSchedule` + per-window replay driver +
  FD-identity tests parametrised over `(tape_storage, checkpoint)`.
- **A3** — public-API kwargs (`tape_storage=:netcdf, tape_path,
  checkpoint`) on `cs_surface_emission_footprint`. C48 14-day stretch
  run.

### Phase D — observation IO (lands in `src/Inversion/`)

- **D1** — `ObservationsIO.jl` + `schemas/cs_observations_v1.toml` +
  round-trip test.
- **D2** — `bind_to_mesh` + 4D-Var equivalence test (literal vector vs
  `CSObservationSet`).
- **D3** — `write_departures` / `read_departures`.

### Phase B — background covariance + preconditioner (lands in `src/Inversion/`)

- **B1** — `Covariance.jl` types + `apply_B_half!` + spectrum test.
- **B2** — `Preconditioning.jl` + adjoint-identity test + log-normal
  optim_type bijection.
- **B3** — wire into `cs_surface_flux_4dvar` (`preconditioned=true`)
  + FD-identity test with non-trivial B.

### Phase C — L-BFGS + driver (lands in `src/Inversion/`)

- **C1** — extract `CSOptimizer` polymorphic API; existing tests green.
- **C2** — `CSLBFGS` via `Optim.jl` + L-BFGS-vs-GD test.
- **C3** — `CSIterationLog` + integration test.
- **C4** — `scripts/inversions/cs_4dvar.jl` + `example_c48.toml` +
  end-to-end inversion smoke test.

## Loose-compat NetCDF schema (D1)

Single file. Dim `obs` (unlimited). Vars per the parent plan:
`id (i64)`, `date_components (i16[6,obs])`, `lat (f32)`, `lon (f32)`,
`alt (f32)`, `value (f64)`, `value_sigma (f64)`,
`instrument_type (string)`, `tracer (string)`. Root attrs:
`cs_observations_schema = "v1"`, `time_origin`.

## Definition of done

- All Phase 0 + A-D commits land with green tests.
- Stretch: C48 14-day forward+reverse with on-disk tape, peak RSS < 50 GB.
- Stretch: C24 synthetic inversion via `scripts/inversions/cs_4dvar.jl`
  recovers prior emissions within 2σ in < 15 L-BFGS iterations.
- `docs/src/theory/adjoint_status.md` updated to reflect the new scaffold.
- No regression in existing Plan-25 tests (95 total: 71 PPM + 23 LinRood
  kernel + 1 LinRood integration).

## Open questions / risks

- **Phase 0 hidden semantic change**: pure code motion can still alter
  public API export order, include order, or method-extension visibility
  in subtle ways. Mitigation: "all 95 adjoint tests stay green every
  P0.* commit" hard gate, codex review for "no semantic change" on
  each move commit.
- **Phase A — disk-IO bandwidth at C180**: tape stage/replay throughput
  is the bottleneck. Mitigation: chunk + compression benchmarking before
  locking in defaults.
- **Phase A — reverse-replay locality**: a correct NetCDF schema can
  still perform poorly if each reverse step triggers scattered reads.
  Mitigation: chunk records along the time axis so a single chunk holds
  one full substep's tape entries (panels + face arrays); LRU cache
  pre-fetches the upcoming window's chunks.
- **Phase B — cross-panel correlation**: v1 ships panel-local FFT only;
  edge artefacts may need a Schur complement or small eigendecomp in v2.
- **Phase B/C — preconditioner masks optimizer comparison**: L-BFGS-vs-GD
  benchmarking must use PHYSICAL-space cost / ‖g_phys‖ for the stopping
  criterion comparison, not χ-space metrics — otherwise the comparison
  trivially favours whichever optimizer has better preconditioner
  interaction.
- **Phase C — Optim.jl allocations** at high resolution: in-place `g!`
  API needed; fallback to `LBFGSB.jl` or hand-rolled L-BFGS if profile
  shows the issue.

## Reusable references

- TM5-cy3-4DVar (`deps/tm5-cy3-4dvar/`): architecture inspiration only,
  no direct code reuse. Key files studied:
  `base/py/main/RunTM5_base.py` (orchestration),
  `base/py/main/Optimizer_base.py` (M1QN3 + conGrad),
  `base/py/main/Precon_base.py` (B^(−1/2) preconditioner),
  `base/py/main/PointObs_base.py` (observation NetCDF schema),
  `base/src/adj_*.F90` (modular physics adjoints).

- Plan 25 NOTES at `docs/plans/25_LINROOD_ADJOINT/NOTES.md` — same
  commit-cadence template and FD-identity testing strategy.
