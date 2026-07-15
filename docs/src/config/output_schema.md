# Output schema

The default runtime output is a **NetCDF4** snapshot file declared by
`[output] path`. `split = "single"` writes one file per run; `split = "daily"`
writes one file per daily binary. This page documents the exact NetCDF
variables, dimensions, units, and per-topology conventions.

For long runs, `format = "binary_mmap"` writes ATMSNAP files with a Float32
spatial payload and compensated Float64 tracer totals; convert them to this
same NetCDF schema with
`scripts/postprocess/binary_to_netcdf.jl`. See [TOML schema](@ref) for that
throughput-oriented workflow.

The `write_snapshot_netcdf` entry point in `src/Output/netcdf_writer.jl`
dispatches on the runtime mesh type into one of three per-topology writers.

The variable list is controlled by `[output.fields]`. By default every field
below is written. Setting `layers = "none"` suppresses per-level tracer VMR
variables; setting `layers = "selected"` writes the same variable names on the
`lev_selected` dimension. `tracers = [...]` restricts all tracer diagnostics to
that subset, with optional `[output.fields.per_tracer.<name>]` overrides.

Every selected tracer also has a topology-independent
`<tracer>_total_mass(time)` variable. It is the compensated Float64 global sum
of the conservative `mixing_ratio × carrier_air_mass` storage captured from the
model state. Negative values are valid, and `kg` refers to the carrier-mass
storage unit rather than physical kilograms of tracer species. This is the
authoritative snapshot conservation series; do not reconstruct it by summing a
Float32 spatial payload when signed components can cancel.

## Global attributes

Every snapshot file carries a CF-style global header set by
`_define_common_attributes!` in `src/Output/netcdf_schema.jl`:

| Attribute | Value |
| --- | --- |
| `Conventions` | `"CF-1.8"` |
| `title` | `"AtmosTransport runtime snapshot"` |
| `source` | `"AtmosTransport.jl"` |
| `institution` | `ENV["ATMOSTR_INSTITUTION"]` if set, else `"Caltech / Frankenberg group"` |
| `grid` | `summary(mesh)` string (e.g. `"72×37 LatLonMesh{Float32}"`) |
| `grid_type` | `"latlon"` / `"reduced_gaussian"` / `"cubed_sphere"` |
| `mass_basis` | `"dry"` or `"moist"` (matches `state.air_mass`) |
| `output_contract` | version tag for the schema |
| `creation_date` | ISO-8601 UTC timestamp of the run |
| `framework` | `"AtmosTransport.jl"` |
| `framework_commit` | git SHA of the source tree at run time (or `"unknown"`) |
| `framework_dirty` | `"clean"` or `"dirty"` (uncommitted changes flag) |
| `runtime` | Julia version plus machine, kernel, and operating-system summary |
| `hostname` | `Base.Libc.gethostname()` at run start |
| `user` | `$USER` (or `$USERNAME` on Windows; `"unknown"` if neither is set) |
| `output_options` | `float_type=…, deflate_level=…, shuffle=…` (only present when writer options are passed) |
| `history` | CF-canonical chain; the writer prepends `"<creation_date>: written by AtmosTransport.Output (commit <sha>[+dirty]) with N frame(s)"` |

Every provenance value is best-effort: non-git checkouts get
`framework_commit = "unknown"`; environments without a `USER` variable get
`user = "unknown"`. `output_options` is present when writer options are passed;
the other listed attributes are written by the current NetCDF writer.

## Lat-lon snapshot

Dimensions:

| Dim | Length |
|---|---|
| `lon` | `Nx` (cell centers) |
| `lat` | `Ny` |
| `lev` | `Nz` (`positive = "down"` — `lev[1]` is TOA, `lev[end]` is surface) |
| `time` | one entry per configured output time that actually fired |
| `lev_selected` | only present when `[output.fields] layers = "selected"` or `air_mass_layers = "selected"` |

Coordinate variables:

| Variable | Shape | Units (writer string) |
|---|---|---|
| `lon` | `(lon,)` | `degrees_east` |
| `lat` | `(lat,)` | `degrees_north` |
| `lon_bounds` | `(lon, nv)` (`nv = 2`) | `degrees_east` |
| `lat_bounds` | `(lat, nv)` (`nv = 2`) | `degrees_north` |
| `cell_area` | `(lon, lat)` | `m2` |
| `time` | `(time,)` | `hours` since the configured simulation start |
| `lev` | `(lev,)` | `1` (dimensionless level index; `positive = "down"`) |

Per-topology mass diagnostics (enabled by default):

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
| `<tracer>_column_mass_per_area` | `(lon, lat, time)` | `kg m-2` model storage | `kg m-2` model storage |
| `<tracer>_total_mass` | `(time,)`, Float64 | `kg` model storage | `kg` model storage |

The per-tracer full-3D field `<tracer>` is the mixing ratio. The
`<tracer>_column_mass_per_area` diagnostic is the sum of the model's
`χ × carrier-air-mass` storage divided by area; no molecular-weight conversion
to physical kg species is applied.

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
| `<tracer>_total_mass` | `(time,)`, Float64 | no |

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

The per-panel arrays are stacked into the `nf` dimension at write time by
`_cs_stack3` / `_cs_stack2` in `src/Output/netcdf_writer.jl`.

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
| `<tracer>_total_mass` | `(time,)`, Float64 | `kg` model storage |

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
from os.path import expanduser
import netCDF4 as nc

with nc.Dataset(expanduser("~/data/.../my_run.nc")) as ds:
    print(ds.dimensions)
    print(list(ds.variables.keys()))
    co2_cm = ds["co2_bl_column_mean"][:]  # (time, lat, lon) in netCDF4
    print(co2_cm.shape, co2_cm.min(), co2_cm.max(), co2_cm.mean())

with nc.Dataset(expanduser("~/data/.../my_cs_run.nc")) as ds:
    co2_cs = ds["co2_bl_column_mean"][:]  # (time, nf, Ydim, Xdim)
    panel = co2_cs[-1, 0, :, :]            # last frame, panel 1
```

### Julia (NCDatasets.jl)

```julia
using NCDatasets

NCDataset(expanduser("~/data/.../my_run.nc")) do ds
    @show keys(ds.variables)
    co2_cm = ds["co2_bl_column_mean"][:, :, end] # last frame, (lon, lat)
    co2_air = ds["air_mass"][:, :, :, end]       # (lon, lat, lev)
end
```

## Fill value

Every payload variable is defined with `_FillValue = 1.0e15` (and
`missing_value = 1.0e15` for older tools); this matches the GEOS-Chem
convention (`Met_AD._FillValue == 1.0e15`) so Panoply / ncview / IDV
mask the same out-of-range cells with the same value. Float32 outputs
truncate to `Float32(1e15)`, which sits comfortably below
`floatmax(Float32) ≈ 3.4e38` and outside any physical mass /
mixing-ratio range. The sentinel is written via NetCDF4's storage
default so uninitialised cells are masked even if the writer never
reaches them.

## Compression and packing

| Option | Default | Effect |
| --- | --- | --- |
| `[output] deflate_level` | `0` (no compression) | NetCDF4 zlib level 0..9 |
| `[output] shuffle` | `true` | shuffle filter (only effective when `deflate_level > 0`) |

Compression trades writer time for disk space and depends strongly on the
field. Benchmark `deflate_level = 1` through `4` on representative output
before choosing a production setting; high levels usually have diminishing
returns.

`float_type` is determined by the runtime's
`[numerics].float_type` — F32 runs write F32 NetCDF, F64 runs write
F64.

## Where to read next

- [TOML schema](@ref) — the full `[output]` block reference.
- [Inspecting output](@ref) — diagnostic CLI tools and quick Python
  recipes.
- [Data sources](@ref) — where the raw met data comes from.
