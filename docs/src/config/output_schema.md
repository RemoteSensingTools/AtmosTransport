# Output schema

The runtime writes a single **NetCDF4** file per run, declared by
`[output] snapshot_file` in the run config. This page documents the
exact variable layout, dimensions, units, and per-topology
conventions, so a downstream tool (Python / Julia / NCO / CDO) can
consume the output without having to look up the writer source.

The writer entry point is `src/Output/netcdf_writer.jl`
(`write_snapshot_netcdf` at line 81) which dispatches on the runtime
mesh type into one of three per-topology writers.

## Global attributes

Every snapshot file carries a CF-style global header (declared in
`src/Output/netcdf_schema.jl:19-29`):

| Attribute | Value |
|---|---|
| `Conventions` | `"CF-1.8"` |
| `title` | `"AtmosTransport runtime snapshot"` |
| `source` | `"AtmosTransport.jl"` |
| `grid` | `summary(mesh)` string (e.g. `"72×37 LatLonMesh{Float32}"`) |
| `grid_type` | `"latlon"` / `"reduced_gaussian"` / `"cubed_sphere"` |
| `mass_basis` | `"dry"` or `"moist"` (matches `state.air_mass`) |
| `output_contract` | version tag for the schema |
| `history` | `@sprintf("written by AtmosTransport.Output with %d frame(s)", length(frames))` |

## Lat-lon snapshot

Dimensions:

| Dim | Length |
|---|---|
| `lon` | `Nx` (cell centers) |
| `lat` | `Ny` |
| `lev` | `Nz` (`positive = "down"` — `lev[1]` is TOA, `lev[end]` is surface) |
| `time` | one entry per `snapshot_hours` value that actually fired |

Coordinate variables:

| Variable | Shape | Units (writer string) |
|---|---|---|
| `lon` | `(lon,)` | `degrees_east` |
| `lat` | `(lat,)` | `degrees_north` |
| `lon_bounds` | `(lon, nv)` (`nv = 2`) | `degrees_east` |
| `lat_bounds` | `(lat, nv)` (`nv = 2`) | `degrees_north` |
| `cell_area` | `(lon, lat)` | `m2` |
| `time` | `(time,)` | `hours since 2000-01-01 00:00:00` |
| `lev` | `(lev,)` | `1` (dimensionless level index; `positive = "down"`) |

Per-topology mass diagnostics (always written):

| Variable | Shape | Units (writer string) | Meaning |
|---|---|---|---|
| `air_mass` | `(lon, lat, lev, time)` | `kg` | per-cell air mass on `mass_basis` |
| `air_mass_per_area` | `(lon, lat, lev, time)` | `kg m-2` | layer mass divided by `cell_area` |
| `column_air_mass_per_area` | `(lon, lat, time)` | `kg m-2` | column total divided by `cell_area` |

Per-tracer fields (one set per `[tracers.<name>]` block). The
`units` string written into the NetCDF reflects the runtime basis:

| Variable | Shape | Units (DryBasis writer string) | Units (MoistBasis writer string) |
|---|---|---|---|
| `<tracer>` | `(lon, lat, lev, time)` | `mol mol-1 dry` | `mol mol-1` |
| `<tracer>_column_mean` | `(lon, lat, time)` | `mol mol-1 dry` | `mol mol-1` |
| `<tracer>_column_mass_per_area` | `(lon, lat, time)` | `kg m-2` | `kg m-2` |

The per-tracer full-3D field `<tracer>` is the **mixing ratio**, not
the mass; for mass × area use `<tracer>_column_mass_per_area`.

## Reduced-Gaussian snapshot

Dimensions:

| Dim | Length |
|---|---|
| `cell` | `ncells` (flat ring-by-ring; ring `j` starts at `ring_offsets[j]`) |
| `lev`, `time` | as for LL |
| `lon`, `lat` | rasterized regular LL diagnostic grid (for plotting) |

All horizontal fields are written in **native face-indexed** form
(dimension `cell`). For plot tools that don't understand reduced
Gaussian, a single rasterized variant — the **per-tracer column
mean** — is also written on a regular LL grid (`(lon, lat)`) via
**nearest-neighbor lookup**. The native fields remain authoritative
for any quantitative analysis.

| Variable | Native shape | Rasterized? |
|---|---|---|
| `air_mass` | `(cell, lev, time)` | no |
| `air_mass_per_area` | `(cell, lev, time)` | no |
| `column_air_mass_per_area` | `(cell, time)` | no |
| `cell_area` | `(cell,)` | no |
| `<tracer>` | `(cell, lev, time)` | no |
| `<tracer>_column_mean_native` | `(cell, time)` | — |
| `<tracer>_column_mean` | — | `(lon, lat, time)` (rasterized via nearest-neighbor — diagnostic only) |
| `<tracer>_column_mass_per_area` | `(cell, time)` | no |

The native fields are authoritative; the rasterized ones are for
visualization.

## Cubed-sphere snapshot

Dimensions:

| Dim | Length |
|---|---|
| `Xdim` | `Nc` (per-panel cell-x index) |
| `Ydim` | `Nc` (per-panel cell-y index) |
| `nf` | `6` (panel face index, ordered by the active `panel_convention`) |
| `lev`, `time` | as for LL |

The per-panel arrays are stacked into the `nf` dimension at write
time (`_cs_stack3` / `_cs_stack2` in `netcdf_writer.jl:36-52`).

Per-topology fields:

| Variable | Shape | Units (writer string) |
|---|---|---|
| `air_mass` | `(Xdim, Ydim, nf, lev, time)` | `kg` |
| `air_mass_per_area` | `(Xdim, Ydim, nf, lev, time)` | `kg m-2` |
| `column_air_mass_per_area` | `(Xdim, Ydim, nf, time)` | `kg m-2` |
| `cell_area` | `(Xdim, Ydim, nf)` | `m2` |
| `<tracer>` | `(Xdim, Ydim, nf, lev, time)` | `mol mol-1 dry` (or `mol mol-1` on moist basis) |
| `<tracer>_column_mean` | `(Xdim, Ydim, nf, time)` | same as `<tracer>` |
| `<tracer>_column_mass_per_area` | `(Xdim, Ydim, nf, time)` | `kg m-2` |

A `grid_mapping = "cubed_sphere"` attribute is set on the
horizontally-resolved variables; the active CS definition, coordinate law,
center law, panel convention (`gnomonic` / `geos_native`), and longitude offset
are in the global header so consumers can reconstruct the panel layout if needed (see
[Cubed-sphere](@ref Grids)).

## Reading the snapshot

### `ncdump`

```bash
ncdump -h ~/data/.../my_run.nc | head -40
```

### Python (NetCDF4)

```python
import netCDF4 as nc

ds = nc.Dataset("~/data/.../my_run.nc")
print(ds.dimensions)
print(list(ds.variables.keys()))

# LL example
co2_cm = ds["co2_bl_column_mean"][:]   # shape (time, lat, lon)
print(co2_cm.shape, co2_cm.min(), co2_cm.max(), co2_cm.mean())

# CS example
ds_cs = nc.Dataset("~/data/.../my_cs_run.nc")
co2_cs = ds_cs["co2_bl_column_mean"][:]   # shape (time, nf, Ydim, Xdim)
panel = co2_cs[-1, 0, :, :]               # last frame, panel 1
```

### Julia (NCDatasets.jl)

```julia
using NCDatasets

ds = NCDataset("~/data/.../my_run.nc")
@show keys(ds.variables)

co2_cm = ds["co2_bl_column_mean"][:, :, end]   # last frame, (lon, lat) for LL
co2_air = ds["air_mass"][:, :, :, end]         # full 3D, (lon, lat, lev) for LL
```

## Compression and packing

| Option | Default | Effect |
|---|---|---|
| `[output] deflate_level` | `0` (no compression) | NetCDF4 zlib level 0..9 |
| `[output] shuffle` | `true` | shuffle filter (only effective when `deflate_level > 0`) |

For long production runs, `deflate_level = 4, shuffle = true` cuts
file size ~3-4× with negligible compute overhead. Higher levels
(`6+`) hit diminishing returns and slow the writer noticeably.

`float_type` is determined by the runtime's
`[numerics].float_type` — F32 runs write F32 NetCDF, F64 runs write
F64.

## Where to read next

- [TOML schema](@ref) — the full `[output]` block reference.
- [Inspecting output](@ref) — diagnostic CLI tools and quick Python
  recipes.
- [Data sources](@ref) — where the raw met data comes from.
