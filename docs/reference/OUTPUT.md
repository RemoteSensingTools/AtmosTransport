# Diagnostic NetCDF Output

`AtmosTransport.Output` owns runtime snapshot capture, cadence parsing, file
partitioning, and NetCDF layout. Runners should call the typed output contract
instead of defining topology-specific schedules or paths.

## Contract

- Snapshot frames store full topology-native air mass and tracer mass fields.
- The writer derives per-layer VMR, air-mass-weighted column mean VMR, layer air mass per area, column air mass per area, and tracer column mass per area.
- `mass_basis` is written globally and must match every `SnapshotFrame`.
- LL, RG, and CS dispatch through one writer API.
- `[output]` is parsed into a `RuntimeOutputSpec`, whose schedule and
  partition are typed (`ExplicitSnapshotSchedule` /
  `IntervalSnapshotSchedule`, `SingleOutputFile` / `DailyOutputFiles`).

## Topology Layout

| Topology | Native layout | Panoply/debug support |
|----------|---------------|-----------------------|
| LL | `(lon, lat, lev, time)` | CF lon/lat coordinates, bounds, cell area |
| RG | `(cell, lev, time)` plus native `cell_lon/cell_lat` and `cell_lon_bounds/cell_lat_bounds` | Legacy `*_column_mean(lon,lat,time)` raster for quick maps |
| CS | `(Xdim, Ydim, nf, lev, time)` | `lons`, `lats`, `corner_lons`, `corner_lats`, `cell_area`, `cubed_sphere` mapping |

CS coordinates use the same `CubedSphereMesh` definition as regridding. For
native GEOS output this means the GEOS-FP/GEOS-IT panel order, native
orientation, GMAO equal-distance coordinate law, four-corner center law, and
global `-10°` longitude offset.

RG bounds are written as flattened native-cell quadrilaterals with vertex
order SW, SE, NE, NW. They expose the same per-ring longitude partition and
latitude-face contract used by `ReducedGaussianMesh`. This is CF-style bounds
metadata, not a full UGRID topology; UGRID export would add a `mesh_topology`
variable, node coordinate arrays, and face-node connectivity.

## API

```julia
frame = capture_snapshot(model; time_hours = 24.0, halo_width = 0)

write_snapshot_netcdf("snapshot.nc", [frame], model.grid;
                      mass_basis = :dry,
                      options = SnapshotWriteOptions(float_type = Float32,
                                                     deflate_level = 1))
```

`halo_width` is only needed for panel-native CS runtime state; it strips halos
before writing panel interiors.

From TOML, prefer these `[output]` keys:

```toml
[output]
path = "~/data/AtmosTransport/output/run.nc"
cadence_hours = 3
split = "single"       # "single" or "daily"
deflate_level = 1
shuffle = true
```

For exact output times, replace `cadence_hours` with `hours = [0, 6, 12, 24]`.
For second-based cadence, use `cadence_seconds`. `start_hour` and `stop_hour`
bound interval schedules when the run should not use the default generous
coverage.

`split = "single"` writes one NetCDF at `path` after the run. `split = "daily"`
writes one complete NetCDF per daily binary. If `path` contains `{date}`,
`{YYYYMMDD}`, or `{day}`, those tokens are replaced; otherwise the date suffix
is inserted before `.nc`, e.g. `run.nc` becomes `run_20211201.nc`.

Field selection is explicit and independent of cadence:

```toml
[output.fields]
tracers = ["co2_natural", "co2_fossil"]  # omit or use "all" for every tracer
layers = "selected"                      # "full" | "selected" | "none"
levels = [1, 32, 64]                     # 1-based model levels for selected
column_mean = true
column_mass_per_area = false
air_mass_layers = "none"                 # same choices as layers
air_mass = false
air_mass_per_area = false
column_air_mass_per_area = true

[output.fields.per_tracer.sf6]
layers = "none"
column_mean = true
column_mass_per_area = false
```

Defaults preserve the historical full diagnostic file: all tracers, all
levels, tracer column means, tracer column mass per area, stored air mass,
layer air mass per area, and column air mass per area. Production smoke runs
should usually set `layers = "none"` and keep `column_mean = true` to avoid
writing multi-GB per-level files when only maps/timeseries are needed.

Legacy keys remain accepted while configs migrate:

```toml
[output]
snapshot_hours = [0, 6, 12, 24]
snapshot_file = "~/data/AtmosTransport/output/run.nc"
snapshot_interval_hours = 3
```

## Storage Options

`SnapshotWriteOptions` controls the on-disk representation of heavy payload
variables (`air_mass`, per-layer VMR, column means, and mass-per-area fields):

| Option | Default | Meaning |
|--------|---------|---------|
| `float_type` | `Float32` | NetCDF type for heavy data variables |
| `deflate_level` | `0` | Compression level, where `0` is off and `1:9` enables NetCDF deflate |
| `shuffle` | `true` | Enables the NetCDF shuffle filter when compression is active |

Coordinate variables and small metadata are left uncompressed. Use
`deflate_level = 1` for a good production default; keep `0` for fastest local
debug writes.

## Extension Points

New topologies should add methods in `src/Output/netcdf_schema.jl` and
`src/Output/netcdf_writer.jl` for their geometry and payload layout. Runners
should continue to call `capture_snapshot` and `write_snapshot_netcdf`; they
should not define topology-specific NetCDF schemas.
