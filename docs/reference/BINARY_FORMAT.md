# Mass-Flux Binary Format Family

**Status**: Proposed. Supersedes `BINARY_FORMAT_V5.md` (which remains valid
as the StructuredDirectional specialization for lat-lon ERA5 files).

---

## 1. Design principle

The binary format family homogenizes at the level of the **record model**,
not at the level of array shapes.

All files in the family represent the same mathematical objects:

- cell-centered layer mass
- horizontal face mass fluxes
- vertical interface mass fluxes
- surface pressure
- optional humidity, thermo, convection fields
- enough geometry metadata to reconstruct the mesh

But the on-disk encoding of the horizontal part is allowed to differ by
topology class. Rectangular lat-lon does not pretend to be face-indexed.
Reduced Gaussian does not pretend to be rectangular. Both are valid
members of the same file family.

---

## 2. File layout (all topologies)

```
┌──────────────────────────────────────────────────────────┐
│  JSON header (null-padded to header_bytes)                │
├──────────────────────────────────────────────────────────┤
│  [Geometry section — optional, topology-dependent]        │
├──────────────────────────────────────────────────────────┤
│  Window 1: state + flux sections                         │
├──────────────────────────────────────────────────────────┤
│  Window 2: ...                                           │
├──────────────────────────────────────────────────────────┤
│  ...                                                     │
├──────────────────────────────────────────────────────────┤
│  Window Nt: ...                                          │
└──────────────────────────────────────────────────────────┘
```

- Header is UTF-8 JSON followed by null bytes to fill `header_bytes`.
- Payload is a flat array of `float_type` elements.
- Windows are contiguous, each exactly `elems_per_window` elements.
- The geometry section (connectivity, metrics) is stored once, after the
  header, before the first window. It is present only for `FaceIndexed`
  topology. Structured topologies reconstruct geometry from header metadata.

---

## 3. Common header (all topologies)

Every file in the family has these keys. A reader first parses the common
header to learn what kind of file it is, then dispatches on `grid_type`
and `horizontal_topology` for the rest.

### 3.1 Format metadata

| Key | Type | Description |
|-----|------|-------------|
| `magic` | string | `"MFLX"` — identifies file type |
| `format_version` | int | Version of this spec (1) |
| `header_bytes` | int | Total header size in bytes (≥ 16384) |
| `float_type` | string | `"Float32"` or `"Float64"` |
| `float_bytes` | int | 4 or 8 |

### 3.2 Grid identification

| Key | Type | Description |
|-----|------|-------------|
| `grid_type` | string | `"latlon"`, `"cubed_sphere"`, `"reduced_gaussian"` |
| `horizontal_topology` | string | `"StructuredDirectional"` or `"FaceIndexed"` |

**Allowed combinations**:

| `grid_type` | `horizontal_topology` | Notes |
|-------------|----------------------|-------|
| `"latlon"` | `"StructuredDirectional"` | The natural choice |
| `"cubed_sphere"` | `"StructuredDirectional"` | Per-panel am/bm (production path) |
| `"cubed_sphere"` | `"FaceIndexed"` | Flattened face storage (future) |
| `"reduced_gaussian"` | `"FaceIndexed"` | The only valid choice |

### 3.3 Dimensions

| Key | Type | Description |
|-----|------|-------------|
| `ncell` | int | Total horizontal cells |
| `nface_h` | int | Total horizontal faces |
| `nlevel` | int | Number of vertical levels |
| `nwindow` | int | Number of time windows in the file |

### 3.4 Vertical coordinate

| Key | Type | Description |
|-----|------|-------------|
| `vertical_coordinate_type` | string | `"hybrid_sigma_pressure"` |
| `A_ifc` | float[nlevel+1] | Hybrid A at interfaces [Pa] |
| `B_ifc` | float[nlevel+1] | Hybrid B at interfaces [–] |

### 3.5 Time and transport

| Key | Type | Description |
|-----|------|-------------|
| `dt_met_seconds` | float | Met-window duration [s] |
| `half_dt_seconds` | float | Legacy timing metadata retained for interoperability; `src` kernels do not read this field directly |
| `steps_per_window` | int | Advection substeps per window |
| `source_flux_sampling` | string | Raw met-flux provenance recorded by the writer: `"window_start_endpoint"`, `"window_end_endpoint"`, `"window_mean"`, or `"interval_integrated"` (`"unknown"` is reader-only legacy compatibility) |
| `air_mass_sampling` | string | Stored air-mass timing semantics; current `src` runtime expects `"window_start_endpoint"` |
| `flux_sampling` | string | Stored horizontal/vertical flux timing semantics; current runtime support is driver-specific, and the ERA5 lat-lon reference path uses `"window_constant"` |
| `flux_kind` | string | Stored flux value contract; current runtime expects `"substep_mass_amount"` |
| `humidity_sampling` | string | `"window_endpoints"`, `"single_field"`, or `"none"` |
| `delta_semantics` | string | `"forward_window_endpoint_difference"` or `"none"` |
| `poisson_balance_target_scale` | float | Preprocessing continuity target scale applied to `(m_next - m_curr)` before Poisson balancing horizontal fluxes |
| `poisson_balance_target_semantics` | string | Human-readable description of the Poisson target normalization |

The crucial distinction is between raw met provenance and stored runtime semantics.
A preprocessor may ingest interval-mean or interval-integrated source products,
but the binary should normalize them into the runtime contract expected by
`src`. The current `DrivenSimulation` path assumes a validated stored
contract chosen by the driver. For the ERA5 lat-lon reference path that means:

- `air_mass_sampling = "window_start_endpoint"`
- `flux_sampling = "window_constant"`
- `flux_kind = "substep_mass_amount"`
- `delta_semantics = "forward_window_endpoint_difference"` when deltas are present

### 3.6 Mass basis

| Key | Type | Description |
|-----|------|-------------|
| `mass_basis` | string | `"moist"` or `"dry"` |

Declares whether `m`, horizontal fluxes, and `cm` are on a total-air
(moist) or dry-air basis. Readers tag the loaded flux state accordingly.

### 3.7 Payload manifest

| Key | Type | Description |
|-----|------|-------------|
| `payload_sections` | string[] | Ordered list of sections in each window |
| `elems_per_window` | int | Total elements per window |

`payload_sections` declares what is present. Example for a minimal lat-lon file:

```json
["m", "hflux", "cm", "ps"]
```

For a file with humidity and physics extensions:

```json
["m", "hflux", "cm", "ps", "qv_start", "qv_end", "cmfmc", "temperature"]
```

For moist-mass transport files that may later need dry-air diagnostics or
dry-VMR output, `qv_start` + `qv_end` is the recommended contract. A
single `qv` field remains valid as a legacy fallback, but it does not
carry endpoint semantics.

If time interpolation is part of the contract, the preferred transport-delta
payload is the full set of mass and flux tendencies:

```json
["dam", "dbm", "dcm", "dm"]
```

This keeps vertical-flux variation explicit in the binary and avoids runtime
closure inside advection kernels.

The reader uses this manifest to compute offsets and skip sections it
does not need.

### 3.8 Optional fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `include_qv` | bool | false | Single specific-humidity field `qv` (legacy / no explicit endpoint semantics) |
| `include_qv_endpoints` | bool | false | Endpoint humidity fields `qv_start` and `qv_end`; preferred for moist transport with dry diagnostics |
| `include_cmfmc` | bool | false | Convective mass flux |
| `include_surface` | bool | false | Surface fields |
| `include_convection` | bool | false | Convection scheme fields |
| `include_temperature` | bool | false | Model-level temperature |
| `include_flux_delta` | bool | false | Temporal interpolation deltas such as `dam`, `dbm`, `dcm`, `dm`, or `dhflux` |

---

## 4. Topology-specific header extensions

### 4.A StructuredDirectional

#### 4.A.1 Rectangular lat-lon

| Key | Type | Description |
|-----|------|-------------|
| `Nx` | int | Longitude cells |
| `Ny` | int | Latitude cells |
| `lons` | float[Nx] | Cell-center longitudes [deg] |
| `lats` | float[Ny] | Cell-center latitudes [deg] |
| `grid_convention` | string | `"TM5"` (S→N, periodic lon) |

Consistency: `ncell = Nx × Ny`, `nface_h = (Nx+1)×Ny + Nx×(Ny+1)`.

Element counts per window (derivable from Nx/Ny/Nz but stored for
fast offset computation):

| Key | Elements |
|-----|----------|
| `n_m` | Nx × Ny × Nz |
| `n_am` | (Nx+1) × Ny × Nz |
| `n_bm` | Nx × (Ny+1) × Nz |
| `n_cm` | Nx × Ny × (Nz+1) |
| `n_ps` | Nx × Ny |

#### 4.A.2 Cubed-sphere (per-panel structured)

| Key | Type | Description |
|-----|------|-------------|
| `Nc` | int | Cells per panel edge |
| `npanel` | int | 6 |
| `Hp` | int | Halo width for panel arrays |
| `panel_convention` | string | `"GEOSFP_file"` or `"gnomonic"` |

Consistency: `ncell = 6 × Nc × Nc`.

Element counts per panel (with halos):

| Key | Elements |
|-----|----------|
| `n_delp_panel` | (Nc + 2Hp) × (Nc + 2Hp) × Nz |
| `n_am_panel` | (Nc + 2Hp + 1) × (Nc + 2Hp) × Nz |
| `n_bm_panel` | (Nc + 2Hp) × (Nc + 2Hp + 1) × Nz |

### 4.B FaceIndexed

#### 4.B.1 Reduced Gaussian

| Key | Type | Description |
|-----|------|-------------|
| `nlat` | int | Number of latitude rings |
| `nlon_per_ring` | int[nlat] | Cells per ring |
| `latitudes` | float[nlat] | Ring center latitudes [deg] |

Geometry section (stored once, after header, before window data):

| Array | Shape | Description |
|-------|-------|-------------|
| `cell_area` | float[ncell] | Cell areas [m²] |
| `face_length` | float[nface_h] | Face lengths [m] |
| `face_left` | int32[nface_h] | Left cell of each face |
| `face_right` | int32[nface_h] | Right cell of each face |
| `face_normal_lon` | float[nface_h] | Normal east component |
| `face_normal_lat` | float[nface_h] | Normal north component |

Optional (for direct divergence kernels):

| Array | Shape | Description |
|-------|-------|-------------|
| `cell_face_ptr` | int32[ncell+1] | CSR pointer into face list |
| `cell_face_idx` | int32[sum(faces_per_cell)] | Face indices per cell |

The geometry section element count is recorded in `n_geometry_elems`.

#### 4.B.2 Cubed-sphere (flattened, future)

Same face-indexed structure but with `grid_type = "cubed_sphere"` and
panel metadata in the header. Not specified here — defer to Phase 2+.

---

## 5. Per-window payload

Each window is a contiguous block of `elems_per_window` values in
`float_type` precision. The sections within a window appear in the order
given by `payload_sections`.

### 5.1 State section

| Field | StructuredDirectional (LL) | StructuredDirectional (CS) | FaceIndexed |
|-------|---------------------------|---------------------------|-------------|
| `m` | (Nx, Ny, Nz) | 6 × (Nc+2Hp, Nc+2Hp, Nz) | (ncell, Nz) |
| `ps` | (Nx, Ny) | 6 × (Nc, Nc) | (ncell,) |

### 5.2 Horizontal flux section

| Field | StructuredDirectional (LL) | StructuredDirectional (CS) | FaceIndexed |
|-------|---------------------------|---------------------------|-------------|
| `hflux` | `am` (Nx+1,Ny,Nz) + `bm` (Nx,Ny+1,Nz) | 6×am_panel + 6×bm_panel | (nface_h, Nz) |

For StructuredDirectional, horizontal flux is split into directional
components (`am`/`bm`). For FaceIndexed, it is a single face-indexed
array with one flux value per face per level.

### 5.3 Vertical flux section

| Field | All topologies |
|-------|---------------|
| `cm` | (ncell, Nz+1) — or structured equivalent |

For StructuredDirectional LL: `(Nx, Ny, Nz+1)`.
For StructuredDirectional CS: `6 × (Nc, Nc, Nz+1)` (if stored — may be
diagnosed from horizontal convergence).

### 5.4 Optional sections

Appear in `payload_sections` order when present:

| Section | Shape (LL) | Shape (FaceIndexed) |
|---------|-----------|-------------------|
| `qv` | (Nx, Ny, Nz) | (ncell, Nz) |
| `qv_start` | (Nx, Ny, Nz) | (ncell, Nz) |
| `qv_end` | (Nx, Ny, Nz) | (ncell, Nz) |
| `cmfmc` | (Nx, Ny, Nz+1) | (ncell, Nz+1) |
| `temperature` | (Nx, Ny, Nz) | (ncell, Nz) |
| surface fields | per-field (Nx, Ny) | per-field (ncell,) |
| flux deltas | matching hflux+cm shapes | matching shapes |

---

## 6. Physical semantics

### 6.1 Cell mass `m`

`m[c, k]` is the layer air mass per cell [kg]:

    m = dp × A_cell / g

When `mass_basis = "moist"`, this is total air mass (dry + vapor).
When `mass_basis = "dry"`, this is dry air mass only.

`m` is the **reference mass at the start of the window**, not the mass
evolved by advection.

### 6.2 Horizontal fluxes

The horizontal flux is the prepared mass transported across
a horizontal face [kg for the prepared transport substep].

**StructuredDirectional** stores separate directional arrays:
- `am[i,j,k]` — positive eastward (increasing i)
- `bm[i,j,k]` — positive northward (increasing j)

**FaceIndexed** stores one value per face:
- `hflux[f,k]` — positive from `face_left[f]` toward `face_right[f]`

These are different on-disk encodings of the same mathematical object:
the signed mass flux across each horizontal face.

### 6.3 Vertical flux `cm`

`cm[c, k]` is the prepared mass transported across the vertical
interface `k` [kg for the prepared transport substep].

Positive = downward (increasing k, toward surface).

Boundary conditions: `cm[:, 1] = 0` (TOA), `cm[:, Nz+1] = 0` (surface).

### 6.4 Surface pressure `ps`

`ps[c]` is the surface pressure [Pa] at the start of the window.

### 6.5 Humidity endpoint semantics

`qv` is a single specific-humidity field with file-specific timing. It is
kept for compatibility, but it should not be used as the default contract
for new moist-basis binaries when downstream dry-air diagnostics matter.

`qv_start` and `qv_end` are the preferred humidity sections for new files:

- `qv_start[c,k]` = specific humidity at the start of the transport window
- `qv_end[c,k]` = specific humidity at the end of the transport window

For a moist-mass transport run that later writes dry VMR, the expected
end-of-window conversion is:

- use the advected moist air mass `m_end` produced by the runtime
- use `qv_end` for the corresponding endpoint humidity
- compute dry-air diagnostics from the matched end state, not from a
  stale window-start humidity field

If an adapter wants to reconstruct endpoint dry reference mass from
pressure fields instead of the transported `m`, it may also need a
`ps_end` surface-field extension. That is optional adapter metadata, not a
requirement of the core transport contract.

When `include_flux_delta = true`, the reference `src` runtime samples the
interpolated forcing at the transport-substep midpoint. For substep `s` in a
window with `steps_per_window = N`, the interpolation fraction is
`λ = (s - 0.5) / N`. This convention applies only to binaries whose stored
semantics declare:

- `air_mass_sampling = "window_start_endpoint"`
- `flux_sampling = "window_start_endpoint"`
- `flux_kind = "substep_mass_amount"`
- `delta_semantics = "forward_window_endpoint_difference"`

The advection kernels do not perform this interpolation themselves. They consume
the fully prepared instantaneous/substep forcing state produced by the driver.

---

## 7. Reader contract

### 7.1 Universal reader entry point

```julia
reader = TransportBinaryReader(path; FT=Float64)

# Properties available for all topologies:
reader.grid_type          # :latlon, :cubed_sphere, :reduced_gaussian
reader.horizontal_topology # :StructuredDirectional, :FaceIndexed
reader.ncell
reader.nface_h
reader.nlevel
reader.nwindow
reader.mass_basis         # :moist or :dry
```

### 7.2 Loading (topology-dispatched)

```julia
m, ps, fluxes = load_window!(reader, win)
```

The return type of `fluxes` depends on topology:

| Topology | Return type |
|----------|-------------|
| StructuredDirectional | `StructuredFaceFluxState{Basis}` with `.am`, `.bm`, `.cm` |
| FaceIndexed | `FaceIndexedFluxState{Basis}` with `.horizontal_flux`, `.cm` |

Both are subtypes of `AbstractFaceFluxState`. The basis parameter
(`MoistMassFluxBasis` or `DryMassFluxBasis`) is set from `mass_basis`.

### 7.3 Optional section loaders

```julia
load_qv_window!(reader, win)          # → Array or nothing
load_qv_pair_window!(reader, win)      # → (; qv_start, qv_end) or nothing
load_flux_delta_window!(reader, win)   # → NamedTuple or nothing
load_cmfmc_window!(reader, win)        # → Array or nothing
load_temperature_window!(reader, win)  # → Array or nothing
```

All return `nothing` when the section is absent.

---

## 8. How existing formats map into this family

### ERA5 lat-lon (v1–v5)

```
grid_type            = "latlon"
horizontal_topology  = "StructuredDirectional"
```

The existing v4 binary header already contains all required common keys
(under slightly different names). A compatibility shim maps:
- `Nt` → `nwindow`
- `Nx × Ny` → `ncell`
- `version` → `format_version` (via offset)
- `mass_basis` defaults to `"moist"` if absent

v1–v5 files remain fully supported via this mapping.

### GEOS-FP / GEOS-IT cubed-sphere (v1–v4)

```
grid_type            = "cubed_sphere"
horizontal_topology  = "StructuredDirectional"
```

Maps:
- `Nc` → panel edge cells; `ncell = 6 × Nc²`
- `n_delp_panel` → per-panel cell mass (with halos)
- `n_am_panel`, `n_bm_panel` → per-panel directional fluxes
- No `cm` on disk — diagnosed from horizontal convergence

### Future: reduced Gaussian

```
grid_type            = "reduced_gaussian"
horizontal_topology  = "FaceIndexed"
```

New format, no backward compatibility needed. Geometry section stores
connectivity and metrics. Per-window payload uses flat cell/face indexing.

---

## 9. What this does NOT homogenize

- **Array shapes**: lat-lon stays `(Nx, Ny, Nz)`, CS stays per-panel,
  reduced Gaussian uses flat `(ncell, Nz)`. No fake reshaping.

- **Payload order within topology**: each topology defines its own
  section ordering. The manifest (`payload_sections`) makes it explicit.

- **Structured kernel internals**: structured grids keep their `am`/`bm`
  fast-path arrays. The reader delivers them directly, not via a generic
  face-indexed wrapper.

---

## 10. What this DOES homogenize

- **Header schema**: every file starts with the same common keys
- **Physical variable meanings**: `m` is always cell layer mass, fluxes
  are always prepared substep mass transport, `ps` is always surface
  pressure, `cm` is always vertical interface flux
- **Basis tagging**: every file declares moist vs dry
- **Reader output contract**: every reader produces
  `(m, ps, AbstractFaceFluxState)` — the transport core never sees
  topology-specific details
- **Time/window semantics**: same `dt_met_seconds`, `steps_per_window`,
  and explicit sampling semantics everywhere
- **Optional section protocol**: same `include_*` / `payload_sections`
  mechanism for all topologies
