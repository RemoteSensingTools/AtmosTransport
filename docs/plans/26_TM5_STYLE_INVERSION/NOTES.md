# Plan 26 — TM5-style footprint + inversion scaffold

Branch: `convection`
Owner: claude (with codex review at every commit)
Started: 2026-05-11
Status: planning
Parent plan: `/home/cfranken/.claude/plans/how-does-tm5-handle-joyful-marble.md`

## Goal

Extend the existing AtmosTransport.jl footprint + 4D-Var prototype toward
TM5-cy3-4DVar architectural parity. Concretely close three gaps:

1. **In-memory tape** → add an on-disk tape storage policy + sliding
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

Storage-format decision (2026-05-11, after Zarr/NetCDF/mmap pros-cons):
**raw binary + mmap is the production default**, NetCDF stays as an
optional archival format, Zarr deferred. Rationale: tape access is
sequential-append on forward + LIFO sequential on reverse — exactly
the workload mmap was designed for. Hot-path serialization + compression
CPU in NetCDF/Zarr competes with the GPU forward step, and we already
have precedent for custom binary files in this repo (the transport
binaries used as preprocessor output). The
`AbstractCSTapeStorage` abstraction (P0.1) means a future
`NetCDFCSTapeStorage` or `ZarrCSTapeStorage` is a drop-in swap if
storage budget or cloud deployment ever forces it.

Concrete layout:

```
tape/
├── manifest.toml      # record table: [step, op, offset, shape, dtype, panel_idx]
├── records.bin        # raw appended record data (preallocated, sparse-OK)
└── (optional) checksums.bin
```

- **A1** — `MmapCSTapeStorage <: AbstractCSTapeStorage` with manifest
  + preallocated `records.bin`; `_allocate_tape_slot` reserves a
  manifest entry + bumps cursor, `stage_panels!` pwrites 6 panels at
  the slot's offset, `_tape_panels` mmap-views the slot and copies
  into a device-side LRU cache. New test
  `test_cs_tape_mmap_roundtrip.jl` (stage → mmap-evict → read-back →
  bit-exact equality). **SHIPPED 2026-05-11 (commits `ba18260` src +
  CUDA-ext, `2aca9e3` tests).** CPU `:device`/`:mmap` and GPU
  `:pinned_host`/`:mmap` parity bit-exact across upwind, linear PPM,
  monotone PPM, and ImplicitVerticalDiffusion. 111 dedicated tests +
  4 new GPU parity assertions in `test_cs_ppm_adjoint_footprint.jl`.
  Public verb is `finalize_tape!(storage)` (manifest emission only
  fires on explicit call — GC finalizer just closes the IOStream).
  Single `device_cache` per storage, reallocated on shape switch;
  keyed LRU deferred to A2.
- **A2** — mmap follow-ups before checkpointing lands. **A.2a / A.2b /
  codex-fixes / A.2c SHIPPED.**
  * **A.2a (commit `c168653`).** Shape-keyed device cache
    (`device_caches::Dict`) so heterogeneous-shape reads reuse one
    per-shape cache; addresses A.1 reviewer Finding 3.
  * **A.2b (commit `0570514`).** `cs_tape_byte_estimate` docs +
    realised-size cross-check tests.
  * **Codex-review pass (commit `ecc15a2`, message mislabels it
    "P0.A.2c").** LinRood non-`:device` rejection, CSSurfaceFluxControl
    panel-shape validation, `total_records` op-count correctness,
    DiffusionAdjoint trailing-blank.
  * **A.2c — manifest-driven resume API.** `load_mmap_tape(dir;
    readonly=true)` reopens a finalised tape directory, parses
    `manifest.toml`, validates version + endianness + finalised +
    `records.bin` size, and rebuilds the slot table.
    `get_record(storage, record_id)` materialises a
    `MmapCSTapeSlot` for `_tape_panels` to mmap-view. Readonly mode
    blocks `_bump_cursor!` / `_allocate_tape_slot` / `stage_panels!`
    with `ArgumentError`. 57 new test assertions in
    `test_cs_tape_mmap_roundtrip.jl` (bit-exact reload, multi-slot
    heterogeneous-shape, readonly enforcement, manifest validation
    errors, `get_record` bounds + closed-storage). Manifest schema is
    unchanged from A.1; restart-for-append (richer manifest with op
    semantics) remains Phase B work.
- **A3 (was A2 in original numbering)** — `CSCheckpointSchedule`
  (`:full`, `:stride`, `:revolve`) + per-window replay driver +
  FD-identity tests parametrised over `(tape_storage, checkpoint)`.
- **A3** — public-API kwargs (`tape_storage=:mmap, tape_path,
  checkpoint`) on `cs_surface_emission_footprint`. C48 14-day stretch
  run; goal: peak RSS < 50 GB, tape disk < 1.5 TB.
- **A4** — bench compress/decompress throughput on real C180 tape;
  if NVMe seq-write or scratch-quota is the bottleneck, add
  `NetCDFCSTapeStorage` (DEFLATE-1 fallback) or `ZarrCSTapeStorage`
  (zstd-3, cloud-ready) as a parallel option behind the same
  `AbstractCSTapeStorage` interface — not a replacement.

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
  is the bottleneck. Mitigation: mmap + raw binary (A1) avoids
  serialization CPU on the hot path; A4 benches it against a
  NetCDF/Zarr fallback before locking the default.
- **Phase A — reverse-replay locality**: for mmap the kernel's
  readahead handles LIFO sequential access. `posix_fadvise(POSIX_FADV_
  SEQUENTIAL)` on the tape file at open; `posix_fadvise(POSIX_FADV_
  DONTNEED)` after reverse pass to release page-cache.
- **Phase A — mmap on networked FS**: mmap is unreliable on NFS and
  has caveats on Lustre (mmap+O_DIRECT, page coherence). Local NVMe
  scratch is fine. If a user points `tape_path` at NFS, A1 should
  detect and refuse with a clear error pointing to the
  NetCDF fallback (A4).
- **Phase A — manifest crash safety**: if a forward run crashes
  mid-record, the manifest TOML may not reflect the partial write.
  Mitigation: write manifest entries atomically (rename-on-close)
  and skip the trailing torn record on resume.
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

- Existing custom-binary precedent in this repo: the transport binary
  format (`src/Preprocessing/transport_binary/`,
  `docs/reference/BINARY_FORMAT_V5.md`) — preprocessor output uses a
  documented offset + manifest layout we should mirror for tape
  records (versioned magic header, host endianness check, sidecar
  manifest).
