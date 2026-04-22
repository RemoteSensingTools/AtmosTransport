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

### Commit 2

- **NC→BIN converter + mmap reader** in
  [`src/Preprocessing/era5_physics_binary.jl`](../../../src/Preprocessing/era5_physics_binary.jl)
  — single file (~530 LOC) containing format constants, header
  struct, writer (`convert_era5_physics_nc_to_bin`), reader
  (`ERA5PhysicsBinaryReader` + `open_*` / `get_era5_physics_field`
  / `close_*`).
- **Calendar-day splicing** — convection NCs are forecast-based
  (07:00 day D through 06:00 day D+1) while thermo is calendar-
  day aligned. The writer builds a calendar-day BIN by pulling
  hours 00–06 from the PREV-day convection NC and hours 07–23
  from the target-day convection NC. Documented in the module
  docstring + `_splice_calendar_day` helper. Tests assert the
  splice picks the right values.
- **Binary format** — 4 KB JSON header (magic `"ERA5PHYS"`,
  format_version=1, per-var offsets + nelems, lat range + convention,
  provenance: source NC paths, timestamp, git sha) followed by
  flat Float32 payload. Six variables:
  `udmf, ddmf, udrf_rate, ddrf_rate, t, q`. Shape
  `(Nlon, Nlat, Nlev, 24)` per var. Latitude flipped N→S to
  S→N at write time (AtmosTransport orientation). Hybrid level
  already matches (k=1=TOA).
- **CLI wrapper** at
  [`scripts/preprocessing/convert_era5_physics_nc_to_bin.jl`](../../../scripts/preprocessing/convert_era5_physics_nc_to_bin.jl)
  — thin arg-parser → delegates to `convert_era5_physics_nc_to_bin`.
  Used as-is on real ERA5 data during the live smoke below.
- **Smoke test on real data**: converted Dec 2 2021 from
  `~/data/AtmosTransport/met/era5/0.5x0.5/physics/` (needing Dec 1
  + Dec 2 convection NCs + Dec 2 thermo NC) to
  `/temp1/era5_bin_smoke/2021/era5_physics_20211202.bin`
  (20.51 GB, ~15 minutes). Reader path end-to-end confirmed:
  shape (720, 361, 137, 24); UDMF range [0, 0.84] kg/m²/s;
  T range [174, 316] K; Q range [-3.5e-6, 0.023]; latitude
  correctly S→N (-90 → 90). Smoke data cleaned up after verification
  (not part of the commit).

- **Surprise: short-write silent corruption.** Initial smoke run
  targeted `/tmp` (10 GB tmpfs). `write(io, arr)` returned fewer
  bytes than requested when `/tmp` filled; my writer ignored the
  return value and produced a half-full BIN whose header advertised
  the full layout. Reader then tried to mmap past EOF and triggered
  a file-grow error on a read-only stream.
  **Fix**: `_write_payload!` now asserts `written == sizeof(arr)`
  and errors cleanly with a disk-full hint. Applies to all six
  variables' writes.

- **Tests** (37 core testsets in
  [`test/test_era5_physics_binary.jl`](../../../test/test_era5_physics_binary.jl),
  registered in `core_tests`):
  - BIN roundtrip on synthetic NCs (20 — header fields, payload
    byte-exactness, hour-splice correctness).
  - Latitude flip N→S to S→N (3).
  - Missing-file errors name the fix (5 — both "missing today
    NC" and "missing prev-day NC" cases).
  - `open_era5_physics_binary` on missing BIN errors clearly (3).
  - Idempotent write (4 — skip-if-exists vs `force_rewrite=true`).
  - Zero-allocation getter (1 — `@allocated get_era5_physics_field`).
  - Unknown-var symbol errors cleanly (1).
  - Plus a `--all`-gated real-data test that reads one day from
    the archive.

- **0 regressions** on existing plan 23 / plan 24 Commit 1 tests;
  README gate 74 pass.

### Commit 3

- **Grid-level pipeline hook**
  [`tm5_native_fields_for_hour!`](../../../src/Preprocessing/tm5_convection_conversion.jl)
  — for one hour of the 2D `(Nlon, Nlat)` grid, compute dz from
  `(T, Q, ps)` per column, call `ec2tm_from_rates!`, fill the 3D
  `(Nlon, Nlat, Nz_native)` output. Per-column scratch vectors
  are reused via an optional `scratch` kwarg; `TM5CleanupStats`
  counters aggregate across all columns.
- **Native → merged remap**
  [`merge_tm5_field_3d!`](../../../src/Preprocessing/tm5_convection_conversion.jl)
  — accumulate native-L137 TM5 fluxes into merged-Nz buckets
  via the existing `merge_map`. Semantics match
  `merge_cell_field!` from mass_support.jl for mass-flux
  fields: sum native layers that map to the same merged layer.
  Preserves column-integrated flux (mass-budget).
- **Half-level interface packing** — the writer's binary stores
  `udmf[i, j, k]` as ERA5's layer-top flux (k=1..137). The
  Commit 3 column loop repacks into the `(Nz+1)`-length interface
  vector expected by `ec2tm_from_rates!`: `udmf_col[1]=0` (TOA),
  `udmf_col[k+1] = udmf[i, j, k]` for `k=1..Nz_native`. Documented
  in the function body.
- **Tests** (133 new testsets in
  [`test/test_tm5_vertical_remap.jl`](../../../test/test_tm5_vertical_remap.jl),
  registered in `core_tests`):
  - `merge_tm5_field_3d!` column-sum preservation (62).
  - Zero-out of unused buckets (5).
  - Shape guards (2).
  - `tm5_native_fields_for_hour!` grid loop correctness:
    all-columns-uniform-in → uniform-out, non-negative outputs,
    column-processed counter (15).
  - Output shape guard (1).
  - **End-to-end native → merged column-sum invariant** (48):
    nonzero updraft profile + hydrostatic dz → native ec2tm →
    merge, and the column sum of each TM5 field at merged Nz
    equals the column sum at native 137L (rtol 1e-12). This
    is the plan 24 Commit 3 headline invariant.
- **0 regressions** on plan 23 / plan 24 prior tests; README gate 74.

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
