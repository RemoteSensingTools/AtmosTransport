# Transport binary format v4

This is the only supported transport-binary format. Readers reject every
other version; obsolete files must be regenerated.

## Physical contract

- Vertical index `k = 1` is the top of atmosphere and `k = nlevel` is the
  surface layer.
- `m` is cell air mass [kg], not mass per unit area.
- `am`, `bm`, and reduced-Gaussian `hflux` are horizontal face mass amounts
  [kg] with the time interpretation declared by `flux_kind`.
- `cm` is vertical-interface mass amount [kg] with the same time
  interpretation and positive downward.
- Production files use `mass_basis = "dry"`. Exact TM5 diffusion is `dkg`,
  the dry-air exchange rate [kg s⁻¹] between layers `k` and `k + 1`; its last
  level is the zero-flux surface boundary. Layer-centre `kz` is not a v4
  payload section.

The header must state timing rather than relying on a filename or reader
default. Lat-lon and reduced-Gaussian files require
`flux_kind = "substep_mass_amount"`. Cubed-sphere files may also use
`full_window_mass_amount`; the cubed-sphere runtime scales that value by its
declared substep schedule.

## File layout

The file consists of:

1. a UTF-8 JSON header, null-terminated and padded to `header_bytes`;
2. optional topology geometry; and
3. `nwindow` fixed-stride payload windows.

Payload values use the declared `float_type` (`Float32` or `Float64`). The
file size must equal the header region, geometry region, and declared payload
exactly. Readers memory-map the payload and do not accept trailing data.

## Required header fields

Every topology declares:

- identity: `magic = "MFLX"`, `format_version = 4`, `header_bytes`,
  `float_type`, and `float_bytes`;
- geometry: `grid_type`, `horizontal_topology`, `ncell`, `nface_h`,
  `nlevel`, `A_ifc`, and `B_ifc`;
- schedule: `nwindow`, `dt_met_seconds`, `steps_per_window`,
  `steps_per_window_by_window`, and
  `poisson_balance_target_scale_by_window`;
- semantics: `mass_basis`, `source_flux_sampling`, `air_mass_sampling`,
  `flux_sampling`, `flux_kind`, `humidity_sampling`, and `delta_semantics`;
- layout: ordered `payload_sections`, `n_geometry_elems`, and
  `elems_per_window`.

`steps_per_window` is the maximum of `steps_per_window_by_window`. It is a
display summary, not an alternative schedule.

Topology metadata is explicit:

- lat-lon structured: `Nx`, `Ny`, `lons`, and `lats`;
- reduced Gaussian face-indexed: `nlat`, `latitudes`, and `nlon_per_ring`;
- cubed sphere structured: `Nc`, `npanel = 6`, `panel_convention`,
  `cs_definition`, `cs_coordinate_law`, `cs_center_law`, and
  `longitude_offset_deg`.

## Canonical payload sections

| Section | Placement and shape | Units / meaning |
|---|---|---|
| `m` | cell centres, one value per layer | air mass [kg] |
| `am`, `bm` | structured horizontal faces | mass amount [kg] |
| `hflux` | face-indexed horizontal faces | mass amount [kg] |
| `cm` | vertical interfaces, `nlevel + 1` | mass amount [kg] |
| `ps` | cell centres, surface | pressure [Pa] |
| `dm`, `dam`, `dbm`, `dcm`, `dhflux` | same location as base field | forward-window difference |
| `qv_start`, `qv_end` | cell centres | specific humidity [kg kg⁻¹] |
| `entu`, `detu`, `entd`, `detd` | cell centres | TM5 convection rates [s⁻¹] |
| `cmfmc` | vertical interfaces | convective mass flux |
| `dtrain` | cell centres | convective detraining-cloud fraction |
| `pblh`, `ustar`, `pbl_hflux`, `t2m` | surface cell centres | PBL forcing in SI units |
| `vdiff_u`, `vdiff_v`, `vdiff_t`, `vdiff_qv` | cell centres | GEOS VDIFF state in SI units |
| `dkg` | layer interfaces encoded at layer index `k` | exact TM5 dry-air exchange [kg s⁻¹] |

Optional groups are atomic. A file may contain all four TM5 convection
sections or none; all four PBL surface sections or none; all four VDIFF
sections or none; and both humidity endpoints or neither.

## Topology layouts

Structured lat-lon arrays are Julia column-major arrays with shapes
`m(Nx, Ny, Nz)`, `am(Nx+1, Ny, Nz)`, `bm(Nx, Ny+1, Nz)`,
`cm(Nx, Ny, Nz+1)`, and `ps(Nx, Ny)`.

Cubed-sphere sections store panels sequentially, panel 1 through panel 6.
Each panel has the analogous `Nc`-based structured shape. On-disk arrays do
not include runtime halos.

Reduced-Gaussian arrays use `m(ncell, Nz)`, `hflux(nface_h, Nz)`,
`cm(ncell, Nz+1)`, and `ps(ncell)`. Connectivity is reconstructed
deterministically from `latitudes` and `nlon_per_ring`; v4 stores no separate
reduced-Gaussian geometry payload.

## Validation

`validate_transport_contract!` is the executable authority. It rejects a
wrong version, incomplete optional groups, incompatible topology/section
sets, invalid schedules, invalid byte counts, `dkg` on a non-dry basis, and
obsolete `kz` payloads. `inspect_binary` uses the same parser and validation
path as runtime construction.
