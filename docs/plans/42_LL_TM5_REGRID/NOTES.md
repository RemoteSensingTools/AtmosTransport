# Plan 42 -- Lat-Lon TM5 Down-Resolution Contract

## Status

Primary spectral+physics path implemented, 2026-05-19. This plan captures the
gap found while benchmarking TM5 convection: TM5 preprocessing could generate
native LL720x361 binaries, but could not produce lower-resolution LL binaries
carrying TM5 sections from the same ERA physics source.

## Problem

The spectral LL preprocessor currently requires the TM5 physics BIN horizontal
shape to match the transport target grid exactly. For ERA5 physics BINs this
means `720x361` only. Trying to build, for example, `ll72x37` with
`[tm5_convection] enable=true` fails with:

```text
Plan 24 Commit 4 requires LL target == physics BIN shape.
BIN is (720, 361), target is (72, 37).
```

There are two related missing contracts:

1. Spectral ERA5 + native ERA physics BIN -> lower-resolution LL transport
   binary with TM5 sections.
2. Fine LL transport binary -> lower-resolution LL transport binary, preserving
   the transport/runtime binary contract.

This plan prioritized item 1 because it unblocks campaign-style LL
`advdiffconv` runs. Item 2 should share the same target-grid validation and
metadata decisions, but remains a follow-up.

## Non-goals

- Do not change TM5 column math or runtime `TM5Convection`.
- Do not use bilinear or nearest-neighbor regridding for TM5 mass-flux fields.
- Do not modify the existing LL->CS regrid contract except to reuse helpers.
- Do not optimize convection performance here; this is a preprocessing contract
  fix.

## Contract Decisions

### Source and target grids

When TM5 source and target LL grids differ, the binary header must record:

- `tm5_source_grid_type = "latlon"`
- `tm5_source_nlon`, `tm5_source_nlat`
- `tm5_target_nlon`, `tm5_target_nlat`
- `tm5_regrid_method = "conservative"`
- `tm5_ps_source = "spectral_source_grid"` or another explicit value

Runtime should still consume only target-grid `entu/detu/entd/detd`; the new
metadata is for provenance, validation, and future stale-binary checks.

### Surface pressure for TM5 conversion

The ERA physics BIN does not carry PS. TM5 conversion needs PS on the physics
source grid to compute layer thicknesses. For source != target, do not reuse
target-grid `transform.sp`.

Required behavior:

- If TM5 source grid == target grid, keep the current fast path.
- If source grid != target grid, synthesize PS on the TM5 source LL grid from
  the same spectral hour before running `compute_tm5_merged_hour_on_source!`.
- Keep the source-grid PS synthesis deterministic and use the same vertical
  setup / mass-basis assumptions as the target transport path.

### TM5 field regridding

`entu`, `detu`, `entd`, and `detd` are produced on the ERA source LL grid after
native-level conversion and vertical merging. They must be conservatively
regridded to the target LL mesh before being stored in the transport binary.

The existing `TM5PreprocessingWorkspace.regridder` and
`tm5_copy_or_regrid_ll!` are the intended hooks, but they are incomplete:

- The spectral LL path never constructs a source->target LL regridder.
- `_store_window_tm5_fields!` currently allocates output arrays using source
  dimensions, not target dimensions.
- `apply_regridder!` expects flattened horizontal leading dimension, so
  `(Nx, Ny, Nz)` TM5 arrays need a reshape/adapter helper before calling it.

### Transport payload shape

Every optional TM5 section in the output binary must have target-grid shape:

```text
(target_nlon, target_nlat, nlevel)
```

Header counts (`n_entu`, `n_detu`, `n_entd`, `n_detd`) already derive from the
target `LatLonTargetGeometry`; the storage path must match that contract.

## Implementation Plan

### Commit 1 -- Contract test and current failure preservation

- Add a focused test that attempts `ll72x37` TM5 preprocessing from a synthetic
  or tiny physics BIN whose source shape differs from target.
- Initially assert the current failure message so the gap is explicit.
- Add a second unit test around `tm5_copy_or_regrid_ll!` documenting expected
  target shape and conservation behavior.

### Commit 2 -- LL source-grid PS support

- Add a small source-grid spectral workspace for TM5 PS synthesis when
  `physics_reader.header.(Nlon,Nlat) != target.(Nx,Ny)`.
- Reuse existing spectral input and vertical setup; do not read PS from a new
  external file unless a later contract explicitly adds that source.
- Keep the identity path unchanged for native LL720x361.

### Commit 3 -- LL TM5 source->target conservative regrid

- Build or reuse a conservative LL->LL regridder:
  `build_regridder(source_ll_mesh, target_ll_mesh; normalize=false)` or the
  same normalization convention used by existing mass-flux regrid paths.
- Wire that regridder into `allocate_tm5_workspace(...; regridder=...)`.
- Replace the raw `(Nx, Ny, Nz)` call to `apply_regridder!` with a helper that
  flattens horizontal dimensions to `(Nx*Ny, Nz)`, applies the regridder, and
  reshapes back.
- Allocate TM5 destination arrays using target dimensions before storage.

### Commit 4 -- Header/provenance metadata and stale-binary checks

- Add the TM5 source/target/regrid metadata listed above to LL headers when
  `include_tm5conv=true`.
- Update `inspect_binary` output or capability metadata if useful.
- Ensure stale-binary checks distinguish native TM5 binaries from regridded
  TM5 binaries without requiring runtime changes.

### Commit 5 -- Real-data integration smoke

Generate a one-day `ll72x37` F32 TM5 binary from ERA5 Dec 2, 2021:

```text
source physics BIN: /home/cfranken/data/AtmosTransport/met/era5/0.5x0.5/physics_bin
target grid: ll72x37
date: 2021-12-02
```

Validate:

- `inspect_binary` reports `tm5_convection = true`.
- Payload sections include `entu`, `detu`, `entd`, `detd`.
- TM5 section counts match `72 * 37 * nlevel`.
- One-window runtime smoke with `[convection] kind = "tm5"` succeeds.
- Area-integrated TM5 source vs target totals are close enough to justify the
  conservative regrid tolerance.

### Commit 6 -- Campaign config enablement

- Re-enable or generate LL `advdiffconv` configs for `ll72` / `ll144` once the
  binary contract is proven.
- Update the stale comments in `scripts/preprocessing/generate_campaign5d_configs.jl`
  that currently say LL TM5 is intentionally skipped.

## Follow-up: Fine LL Binary -> Coarse LL Binary

This is related but should not block the spectral+physics path above.

Needed design:

- Read fine LL binary windows with current-format header.
- Recover or aggregate target-grid transport fields under the same
  substep/window contract.
- Reconstruct target LL face fluxes and rerun replay/positivity gates.
- Regrid optional TM5 sections using the same LL TM5 regrid helper.
- Preserve or explicitly rewrite schedule metadata; do not silently change
  `flux_kind = substep_mass_amount` semantics.

This should probably be a second entry point analogous to
`regrid_ll_binary_to_cs`, e.g. `regrid_ll_binary_to_ll`, after the TM5
source-grid path is working.

## Validation Checklist

- Unit: source!=target TM5 shape handling.
- Unit: LL TM5 regrid helper uses flattened-horizontal contract correctly.
- Integration: synthetic small source->target binary write/read.
- Real data: one-day `ll72x37` F32 TM5 binary.
- Runtime: one-window TM5 convection smoke on the generated binary.
- Regression: native `ll720x361` TM5 path stays byte-shape compatible.
- Search cleanup: no stale comments claiming LL TM5 is impossible once shipped.

## Resume Pointer

Start in:

- `src/Preprocessing/transport_binary/latlon_spectral.jl`
- `src/Preprocessing/transport_binary/latlon_workspaces.jl`
- `src/Preprocessing/tm5_convection_pipeline.jl`
- `src/Regridding/weights_io.jl`
- `scripts/preprocessing/generate_campaign5d_configs.jl`

The immediate blocker is the shape guard in `latlon_spectral.jl`; remove it
only after source-grid PS synthesis and target-shaped TM5 storage are in place.
