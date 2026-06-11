# Backlog: Known Issues, Consolidation Targets, and Potential Work

The committed, living list of potential work. Unlike `docs/plans/`
(gitignored working notes), this page is part of the repo: when an item
ships, move it to the "Done" log at the bottom with the commit hash, so the
page doubles as a record of debt paid. Sources: the 2026-06-11 repo-wide
audit (structure, Julia-style, docs) plus the open items from
[MASS_BALANCE.md](MASS_BALANCE.md).

Conventions: **[S]** small (≤1 day) · **[M]** medium (days) · **[L]** large
(re-architecture). Each item lists the payoff and the risk.

## P1 — correctness-adjacent / user-facing

1. **[S] Conservative fillz** — if GCHP-faithful positivity AND exact
   conservation are ever both required: Kahan-compensate the fillz borrow
   round-trip. Verify with `ATMOSTR_FILLZ_MASS_DIAG=1` (net must be 0).
   Until then `[advection] fillz = false` is the conservation answer
   (ADVECTION_SCHEMES.md §fillz).
2. **[S] Pre-existing test failures** — `test/orphan/test_output_snapshots.jl`
   has stale `output_field_spec` imports. (The suite-aborting
   `test_cs_writer_contract_guard.jl` fixture is fixed — see Done log.)
3. **[M] Per-level reference `q_ref[k]`** (plan 45 deferred extension) —
   removes the mean vertical profile too (~100× background reduction for
   CO₂-class tracers vs ~34× for the global scalar). Requires a
   reference-aware vertical donor-cell sweep + matching adjoint term.
4. **[S] SF6-class budgets below 0.1 %** — per-tracer F64 runs (curry/A100
   only; L40S has no FP64 units). Measured floor: F64 +0.06 % vs F32 +0.21 %.

## P2 — consolidation: code (from the structure + style audits)

Ranked by payoff × (1/risk). Estimated total: ~18 kLOC of structural
redundancy; the items below are the tractable core.

8. **[M] Per-topology NetCDF writers** — three ~90-line
   `_write_snapshot_payload!` methods sharing the skeleton
   (`src/Output/netcdf_writer.jl`). Shared body + topology shims
   (~150 LOC). Medium risk: the CS `_cs_stack3` and RG raster paths are
   subtle — gate with an output-bytes regression test.
9. **[L] LL vs CS runner unification** — ~900 overlapping lines in
   `src/Models/DrivenRunner.jl`. The runners diverged for physics reasons
   (CS per-binary flux reallocation, moist guard, reference gates), so do
   the LOW-RISK PARTIAL first: extract snapshot accumulation, output
   flushing, and the mass-logging epilog into shared helpers; do NOT force
   a unified loop.
10. **[L] Preprocessing pipeline unification** — `era5_n320_regrid.jl` /
    `latlon_spectral.jl` / `merra2_latlon_regrid.jl` are ~60 % parallel
    `process_day` pipelines (~1.9 kLOC). A parameterized
    `process_day_generic!` + `MetSourceAdapter` types also fixes the
    long-standing N320 substep-contract divergence (KEY_PARADIGMS §A5).
11. **[M] Topology binary-contract gates** — `*_contracts.jl` triplicate the
    replay/positivity gate logic (~2 kLOC, ~65 % overlap). Shared gate
    framework + section-access callbacks.

**Do-NOT-refactor list** (verbosity that is load-bearing — from the style
audit): the `reconstruction.jl` x/y/z face-flux families (genuinely
different boundary conditions per axis + documentation value); the diffusion
kernels' `cref` asymmetry; the CS per-binary `fluxes_d` reallocation; the
existing `sweep_x/y/z!` `@eval` loops (already idiomatic); the
`RunProgressTimer` helper chain.

11a. **[S] Strict unknown-basis handling end-to-end** — `State.mass_basis_type`
    now throws on unknown basis symbols, but the LL/RG header read path
    (`transport_binary/header.jl` `_transport_basis_symbol(::Symbol)` and the
    header parser) still silently coerces any non-`"dry"` value to `:moist`.
    Tighten to throw; audit old binaries first (read-path behavior change).

## P3 — hygiene: tests, scripts, configs

12. **[M] Orphan test triage** — `test/orphan/` is 7.5 kLOC, CI-excluded,
    60–70 % dead or subsumed by core. Promote / merge / delete per file.
13. **[S] Shared test fixtures** — mini CS-state and synthetic-binary
    builders are copy-pasted across ≥5 core test files. One
    `test/core/test_fixtures.jl` module.
14. **[M] Diagnostics scripts consolidation** — `scripts/diagnostics/` has
    91 files, ~70 % one-shot or overlapping: 5 mass-balance checkers → one
    parametric tool (built on the F64 `<tracer>_total_mass` variable +
    logged rates, per MASS_BALANCE.md rules); 6 profile plotters → one
    templated script; archive the 20+ `proto_*`/`fingerfix_*` experiment
    artifacts to `scripts/archive/`.
15. **[S] Config archival policy** — 260 active + 80 `likely_legacy` run
    TOMLs. Create `config/archived/`, move legacy dirs there, add a
    `config/runs/README.md` with a maintained/reference/experimental/archived
    classification, and (optionally) a CI check that maintained configs'
    data paths resolve.

## P4 — docs

16. **[S] CAVEATS.md refresh** — add the new-knob caveats
    (`air_mass_reset_mode` choice, referencing constraints, `fillz=false`
    negatives) and prune items fixed by the 2026-06 conservation work.
17. **[S] Archived-test provenance** — `test/archived/` (560 lines of v2/v4
    era tests) lacks headers explaining what era each documents.

## Done log

- 2026-06-11 — consolidation batch 2: Thomas-solver dedup (890→579 lines in
  `diffusion_kernels.jl`): 8 `@kernel` bodies → two `@inline` column cores
  (`_thomas_geometric_column!` / `_thomas_massflux_column!`) over a zero-cost
  `_ColumnView`/`_KzColumn` accessor; CS `cref` anomaly path preserved via
  `cref::Union{Nothing,FT}` compile-time dispatch (plain path adds NO float
  ops). Verified: old-vs-new CPU harness bit-identical (9/9, all layouts +
  negative-anomaly cref), 1-day C180 GPU run bit-identical, codex APPROVE.
  Also fixed suite-aborter #2: stale `test_era5_n320_reader` "unknown source"
  case used `MERRA-2`, which became a real source — `Pkg.test()` now runs the
  full alphabet.
- 2026-06-11 — consolidation batch 1 (bit-identical gated): suite-aborting
  writer-guard fixture fixed (full `Pkg.test()` restored as CI gate); twin
  `_cs_section_elements` → one canonical superset + header-dispatch shim;
  `_x/_y/_z_subcycling_pass_count` → one `Val{dim}` core; `_sweep_*_pp_subcycled!`
  → `@eval` loop; `midpoint!` closure → `_build_midpoint_closure`;
  `_copy_optional_*` → one higher-order `_copy_optional!`; basis `Symbol` ↔ tag
  maps canonicalized as `State.mass_basis_type` / `State.mass_basis_symbol`
  (replacing 4 ternaries — one of which silently defaulted unknown bases to
  moist — and 4 duplicate reverse maps).
- 2026-06-11 — `air_mass_reset_mode`, `fillz`, and
  `[tracers.X.transport] reference` documented in the canonical TOML schema;
  `GEOS_PREPROCESSING_MASS_BALANCE.md` §5/§6 modernized (the Policy A/B fork
  dissolution shipped as `preserve_tracer_mass`); campaign archive annotated.
- 2026-06-11 — [MASS_BALANCE.md](MASS_BALANCE.md): the F32 conservation
  story (verification methodology, 7-issue catalog with commits, measured
  floors, F64 guidance).
- 2026-06 — the conservation campaign itself: Kahan emissions (`4fe0b61e`),
  diffusion anomaly fix (`9ddf3d68`), day-1 flux-replay clock fix
  (`1c1c7005`), reference-state anomaly transport (plan 45,
  `35204257`…`9e584089`), reference-aware `preserve_tracer_mass` reset
  (`55de815b`), F64 `<tracer>_total_mass` output (`aa403559`), `fillz` knob
  (`79132f78`).
