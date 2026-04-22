# Plan 24 — TM5 Convection Preprocessor Integration

## Baseline (Commit 0, 2026-04-22)

**Parent commit:** `9f63e68` (plan 23 Commit 7, "production audit,
bench, retrospective, ship").

**Branch:** `convection` (tracking `origin/convection`, plan 23
tip not yet pushed).

**Scope:** port TM5 convection preprocessing — missing piece from
plan 23. Add `ec2tm_from_rates!` + hydrostatic `dz` computation,
NC→BIN converter with mmap-friendly binary reader, `process_day`
integration, and end-to-end CATRINE-style test. Plan doc at
`/home/cfranken/.claude/plans/bring-last-session-into-lively-scroll.md`.

## Commit 0 artifacts

- [`artifacts/plan24/survey/`](../../../artifacts/plan24/survey/)
  — main-branch scripts captured for reference during port:
  - `main_preprocess_era5_tm5_convection.jl` — main's standalone
    preprocessor (will be split into `src/Preprocessing/`
    library + CLI wrapper).
  - `main_merge_era5_cmfmc_to_massflux.jl` — main's CMFMC merge
    into massflux NC (deprecated path; plan 24 writes TM5
    sections directly into transport binary).
  - `main_download_era5_physics.py` — main's unified downloader
    for UDMF/DDMF/UDRF/DDRF + T+Q.
  - `main_download_era5_cmfmc.py` — older CMFMC-only downloader
    superseded by `download_era5_physics.py`.
  - `tm5_phys_convec_ec2tm.F90` — TM5-4DVAR authoritative
    reference (`ECconv_to_TMconv` subroutine at lines 87–237).

## Data already available on this host

**`~/data/AtmosTransport/met/era5/0.5x0.5/physics/`** — 5 days of
pre-downloaded ERA5 physics data, Dec 2021:

- `era5_convection_20211201.nc` through `era5_convection_20211205.nc`
  (~636 MB each, zlib-compressed) — UDMF + DDMF + UDRF + DDRF
  on all 137 model levels, hourly forecast fields.
- `era5_thermo_ml_20211201.nc` through `era5_thermo_ml_20211205.nc`
  (~2.6 GB each) — T + Q analyses, hourly, 137 model levels.

Path deviates from plan doc — plan doc said
`met/era5/physics/raw/YYYY/`, actual data sits at
`met/era5/0.5x0.5/physics/` (grid in path per DATA_LAYOUT.md
canonical convention). **Plan 24 adapts to this layout**:

```
met/era5/0.5x0.5/physics/                  # existing, leave as-is
├── era5_convection_YYYYMMDD.nc
└── era5_thermo_ml_YYYYMMDD.nc

met/era5/0.5x0.5/physics_bin/YYYY/         # new, Commit 2 output
└── era5_physics_YYYYMMDD.bin
```

The year subdir in `physics_bin/` keeps multi-year archives
navigable. The NC dir stays flat (existing convention). **No data
move required.**

## TM5 Fortran reference stability

`deps/tm5/base/src/phys_convec_ec2tm.F90` is a vendored copy (not
a git submodule — `deps/tm5/` contains plain directories with 158
`.F90` files under `base/src/`). Line numbers for
`ECconv_to_TMconv` are stable:

- `subroutine ECconv_to_TMconv` at line 87.
- `end subroutine` at line 237.
- Algorithm body is lines 87–237 (small-value clipping → dz
  integration → uptop/dotop search → mass-budget closure →
  negative redistribution).

Commit 1 cites these line ranges in the Julia port.

## Deviations from plan doc

### Commit 0

- **Path correction** above (`0.5x0.5/physics/` not
  `physics/raw/YYYY/`). NOTES deviation only; plan stays
  architecturally correct.
- **Data window**: plan says 7 days (from cmfmc/ download log).
  Actual download produced **5 days** (Dec 1–5); Dec 6–7 were
  in the log but the NCs aren't present. Commit 6 runs against
  5 days, not 7.

### Commit 1

- No deviations from plan. `ec2tm_from_rates!` ports TM5 F90
  `ECconv_to_TMconv` (deps/tm5/base/src/phys_convec_ec2tm.F90:
  87-237) line-for-line. `dz_hydrostatic_virtual!` uses the
  virtual-temperature correction per decision 6; `dz_hydrostatic_constT!`
  ships as the fallback per the plan's risk-register entry.
- `TM5CleanupStats` tracks 11 counters (`columns_processed`,
  `no_updraft`, `no_downdraft`, four clipping counts, four
  negative-redistribution counts). Bumps are conditional on
  `stats !== nothing` (zero overhead when off).
- 48 tests: byte-for-byte F90 parity (via an independent Julia
  reference implementation), zero-forcing, bad-data cleanup,
  negative redistribution with mass-budget-in-active-window
  invariant, dz virtual-T vs const-T tropospheric divergence,
  shape guards, and a plan-23 `ec2tm!` regression check.
- 0 regressions on existing plan-23 tests
  (`test_tm5_preprocessing.jl` 43 pass, README gate 74 pass).

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
