# Mass-Flux Binary Format — v5 Specification (StructuredDirectional / LatLon)

**Status**: Proposed (v5). Current production is v4. v1–v4 remain supported
for reading; new writers should target v5.

**Parent spec**: See `BINARY_FORMAT.md` for the topology-generic binary
family spec. This document is the StructuredDirectional specialization
for rectangular lat-lon grids (ERA5).

## Design goals

1. **Minimal advection contract**: a reader that only needs to run transport
   should parse exactly: m, am, bm, cm, ps, vertical metadata, and optionally qv.
2. **Self-describing**: the file carries all geometry, vertical coordinates,
   and time metadata needed to interpret the payload. No sidecar files required.
3. **Precision-generic**: on-disk data can be Float32 or Float64, declared in
   the header. The reader promotes to the caller's working precision at load time.
4. **Backward-compatible**: a v5 reader MUST accept v1–v4 files. A v4 reader
   MAY reject v5 files (new keys are additive, no existing keys are removed).
5. **Extensible without contract creep**: physics extensions (convection, surface,
   temperature) are explicit optional blocks that the advection reader ignores.

---

## File layout

```
┌─────────────────────────────────────────────────────────┐
│  JSON header (null-padded to `header_bytes`)            │
├─────────────────────────────────────────────────────────┤
│  Window 1:  core block  [| optional blocks in order ]   │
├─────────────────────────────────────────────────────────┤
│  Window 2:  ...                                         │
├─────────────────────────────────────────────────────────┤
│  ...                                                    │
├─────────────────────────────────────────────────────────┤
│  Window Nt: ...                                         │
└─────────────────────────────────────────────────────────┘
```

- Header is UTF-8 JSON followed by null bytes (`0x00`) to fill `header_bytes`.
- Payload is a flat array of `float_type` values (`Float32` or `Float64`).
- Windows are contiguous, each exactly `elems_per_window` elements.

---

## Header keys

### Required (core advection contract)

| Key | Type | Description |
|-----|------|-------------|
| `magic` | string | `"MFLX"` — identifies file type |
| `version` | int | Format version (5 for this spec) |
| `header_bytes` | int | Total header size in bytes (≥16384) |
| `float_type` | string | `"Float32"` or `"Float64"` — on-disk element type |
| `float_bytes` | int | 4 or 8 — redundant with `float_type`, kept for simplicity |
| `Nx` | int | Number of longitude cells |
| `Ny` | int | Number of latitude cells |
| `Nz` | int | Number of vertical levels (merged) |
| `Nt` | int | Number of time windows in the file |
| `n_m` | int | Elements per window for cell mass: `Nx × Ny × Nz` |
| `n_am` | int | Elements for x-face flux: `(Nx+1) × Ny × Nz` |
| `n_bm` | int | Elements for y-face flux: `Nx × (Ny+1) × Nz` |
| `n_cm` | int | Elements for z-face flux: `Nx × Ny × (Nz+1)` |
| `n_ps` | int | Elements for surface pressure: `Nx × Ny` |
| `dt_seconds` | float | Full met-window duration [s] |
| `half_dt_seconds` | float | Half-timestep duration [s] — the time basis for mass fluxes |
| `steps_per_met_window` | int | Number of advection substeps per window |
| `A_ifc` | float[Nz+1] | Hybrid A-coefficients at level interfaces [Pa] |
| `B_ifc` | float[Nz+1] | Hybrid B-coefficients at level interfaces [–] |
| `mass_basis` | string | **NEW in v5**: `"moist"` or `"dry"` — what basis m/am/bm/cm are on |

### Required geometry

| Key | Type | Description |
|-----|------|-------------|
| `lons` | float[Nx] | Cell-center longitudes [degrees] |
| `lats` | float[Ny] | Cell-center latitudes [degrees] |
| `grid_convention` | string | `"TM5"` (S→N latitude, periodic longitude) |

### Optional: specific humidity

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `include_qv` | bool | false | Whether QV block is present |
| `n_qv` | int | 0 | Elements: `Nx × Ny × Nz` |

When `mass_basis = "moist"` and `include_qv = true`, the reader can convert
to dry basis via `m_dry = m × (1 - qv)`.

### Optional: flux deltas (substep interpolation)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `include_flux_delta` | bool | false | Whether delta blocks are present |
| `n_dam` | int | 0 | Elements: `(Nx+1) × Ny × Nz` |
| `n_dbm` | int | 0 | Elements: `Nx × (Ny+1) × Nz` |
| `n_dm` | int | 0 | Elements: `Nx × Ny × Nz` |
| `n_dcm` | int | 0 | Elements: `Nx × Ny × (Nz+1)` |

Flux deltas encode `field_next - field_curr` for TM5-style linear temporal
interpolation within a window. The advection core does NOT require them;
they improve accuracy when the driver interpolates fluxes across substeps.

### Optional: physics extensions (NOT part of advection contract)

These blocks are carried for convection/diffusion operators but are explicitly
outside the advection contract. An advection-only reader skips them.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `include_cmfmc` | bool | false | Convective mass flux (Nx × Ny × (Nz+1)) |
| `include_surface` | bool | false | Surface fields: pblh, t2m, ustar, hflux (4 × Nx × Ny each) |
| `include_tm5conv` | bool | false | TM5 convection: entu, detu, entd, detd (4 × Nx × Ny × Nz each) |
| `include_temperature` | bool | false | Model-level temperature (Nx × Ny × Nz) |

### Optional: provenance and preprocessing metadata

Not part of the data contract. Used for staleness checks and reproducibility.

| Key | Type | Description |
|-----|------|-------------|
| `date` | string | Date of the meteorological data (ISO 8601) |
| `creation_time` | string | When the binary was written |
| `script_path` | string | Absolute path to the preprocessor script |
| `script_mtime_unix` | float | Modification time of the script at write time |
| `git_commit` | string | Git commit hash of the source tree |
| `git_dirty` | bool | Whether the tree had uncommitted changes |
| `merge_map` | int[Nz_native] | Native→merged level mapping (for QV remapping) |
| `Nz_native` | int | Number of native levels before merging |
| `level_top` | int | Top native level index |
| `level_bot` | int | Bottom native level index |
| `merge_min_thickness_Pa` | float | Minimum layer thickness used in merging |
| `mass_fix_enabled` | bool | Whether global mean ps was pinned |
| `mass_fix_target_ps_dry_pa` | float | Dry-air ps target [Pa] |
| `mass_fix_qv_global_climatology` | float | Climatological global mean QV |
| `ps_offsets_pa_per_window` | float[Nt] | Per-window ps offset applied [Pa] |
| `var_names` | string[] | Ordered list of payload variable names |

---

## Payload order (per window)

The payload is a flat contiguous array. Blocks appear in this fixed order;
absent optional blocks are simply not present (their element count is 0).

### Tier 1: Core advection

| Block | Shape | Elements | Always present |
|-------|-------|----------|----------------|
| `m` | (Nx, Ny, Nz) | n_m | yes |
| `am` | (Nx+1, Ny, Nz) | n_am | yes |
| `bm` | (Nx, Ny+1, Nz) | n_bm | yes |
| `cm` | (Nx, Ny, Nz+1) | n_cm | yes |
| `ps` | (Nx, Ny) | n_ps | yes |

### Tier 1b: Optional advection support

| Block | Shape | Condition |
|-------|-------|-----------|
| `qv` | (Nx, Ny, Nz) | `include_qv = true` |

### Tier 2: Physics extensions

| Block | Shape | Condition |
|-------|-------|-----------|
| `cmfmc` | (Nx, Ny, Nz+1) | `include_cmfmc = true` |
| `pblh` | (Nx, Ny) | `include_surface = true` |
| `t2m` | (Nx, Ny) | `include_surface = true` |
| `ustar` | (Nx, Ny) | `include_surface = true` |
| `hflux` | (Nx, Ny) | `include_surface = true` |
| `entu` | (Nx, Ny, Nz) | `include_tm5conv = true` |
| `detu` | (Nx, Ny, Nz) | `include_tm5conv = true` |
| `entd` | (Nx, Ny, Nz) | `include_tm5conv = true` |
| `detd` | (Nx, Ny, Nz) | `include_tm5conv = true` |
| `T` | (Nx, Ny, Nz) | `include_temperature = true` |

### Tier 1c: Optional temporal interpolation

| Block | Shape | Condition |
|-------|-------|-----------|
| `dam` | (Nx+1, Ny, Nz) | `include_flux_delta = true` |
| `dbm` | (Nx, Ny+1, Nz) | `include_flux_delta = true` |
| `dm` | (Nx, Ny, Nz) | `include_flux_delta = true` |
| `dcm` | (Nx, Ny, Nz+1) | `include_flux_delta = true` |

**elems_per_window** = sum of all present blocks.

---

## Physical semantics of `m`

`m` is the **total (moist) air mass per grid cell** [kg], defined as:

    m[i,j,k] = dp[k](i,j) × A[j] / g

where:
- `dp[k]` = `(A_ifc[k+1] - A_ifc[k]) + (B_ifc[k+1] - B_ifc[k]) × ps[i,j]`
  is the total pressure thickness of layer k
- `A[j]` = cell area at latitude j [m²]
- `g` = gravitational acceleration [m/s²]

This is total air mass (dry air + water vapor). When `mass_basis = "moist"`,
all of m, am, bm, cm are on a moist (total-air) basis.

To obtain dry air mass: `m_dry[i,j,k] = m[i,j,k] × (1 - qv[i,j,k])`.

**Important**: `m` is NOT the mass evolved by advection. It is the
*reference mass* at the start of the window. The fluxes `am`, `bm`, `cm`
are scaled to `half_dt_seconds` and represent the mass transported per
half-timestep across each face.

---

## Mass flux conventions

| Field | Meaning | Positive direction |
|-------|---------|-------------------|
| `am[i,j,k]` | Mass flux across x-face at (i,j,k) | Eastward (increasing i) |
| `bm[i,j,k]` | Mass flux across y-face at (i,j,k) | Northward (increasing j) |
| `cm[i,j,k]` | Mass flux across z-face at (i,j,k) | Downward (increasing k, toward surface) |

Boundary conditions:
- `am` is periodic in i: `am[1,:,:] = am[Nx+1,:,:]`
- `bm[:,1,:] = 0`, `bm[:,Ny+1,:] = 0` (pole faces)
- `cm[:,:,1] = 0` (TOA), `cm[:,:,Nz+1] = 0` (surface)
- `am` is zeroed at polar rows: `am[:,1,:] = 0`, `am[:,Ny,:] = 0`

---

## Float type handling

The `float_type` header key declares the on-disk precision:
- `"Float32"`: each element is 4 bytes, IEEE 754 single-precision
- `"Float64"`: each element is 8 bytes, IEEE 754 double-precision

The reader promotes elements to its working precision `FT` at load time via
`copyto!`. Writing a Float64 binary doubles the file size but eliminates
the F32→F64 promotion noise (relevant for debugging and validation).

Existing v1–v4 binaries that lack `float_type` are assumed `"Float32"`.

---

## v5 changes from v4

| Change | Rationale |
|--------|-----------|
| `mass_basis` key added (required) | Makes moist/dry basis explicit in the file, not just in code comments |
| `float_type` formalized | Already present in v4 headers but not in the spec; now required |
| Tier 1/2 split documented | Advection readers can safely skip Tier 2 blocks |
| `m` semantics documented | Eliminates ambiguity about what `m` represents |
| Payload order locked | v1–v4 de facto order is now the normative spec |

No on-disk format changes. A v4 file with `mass_basis` added to the header IS
a valid v5 file. This is an additive specification, not a breaking change.

---

## Reader contract (code-level)

### Tier 1 reader (advection only)

```julia
reader = MassFluxBinaryReader(path; FT=Float64)
m, ps, fluxes = load_window!(reader, win)
# fluxes :: StructuredFaceFluxState{MoistMassFluxBasis}  (if mass_basis="moist")
# fluxes :: StructuredFaceFluxState{DryMassFluxBasis}    (if mass_basis="dry")
```

The reader MUST:
- Parse the header and validate required keys
- Compute `elems_per_window` from present blocks
- Load only the 5 core arrays for `load_window!`
- Tag the returned flux state with the correct basis from `mass_basis`
- Provide `load_qv_window!` if `include_qv = true`

The reader MUST NOT:
- Require Tier 2 blocks to be present
- Fail if unknown header keys are present (forward compatibility)

### Tier 2 reader (physics)

```julia
# Optional loaders — return nothing if block is absent
load_qv_window!(reader, win)
load_flux_delta_window!(reader, win)
load_cmfmc_window!(reader, win)    # Tier 2
load_tm5conv_window!(reader, win)  # Tier 2
```

---

## Backward compatibility matrix

| File version | v5 reader | v4 reader |
|-------------|-----------|-----------|
| v1 | reads (mass_basis defaults to "moist") | reads |
| v2 | reads | reads |
| v3 | reads | reads |
| v4 | reads | reads |
| v5 | reads | reads if no new required keys used |
