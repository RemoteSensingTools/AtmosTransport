# Diagnostic NetCDF Output

`AtmosTransport.Output` owns runtime snapshot capture and NetCDF layout. Runners
should only decide when to sample; they should call `capture_snapshot` and
`write_snapshot_netcdf` rather than defining topology-specific NetCDF schemas.

## Contract

- Snapshot frames store full topology-native air mass and tracer mass fields.
- The writer derives per-layer VMR, air-mass-weighted column mean VMR, layer air mass per area, column air mass per area, and tracer column mass per area.
- `mass_basis` is written globally and must match every `SnapshotFrame`.
- LL, RG, and CS dispatch through one writer API.

## Topology Layout

| Topology | Native layout | Panoply/debug support |
|----------|---------------|-----------------------|
| LL | `(lon, lat, lev, time)` | CF lon/lat coordinates, bounds, cell area |
| RG | `(cell, lev, time)` plus native `cell_lon/cell_lat` | Legacy `*_column_mean(lon,lat,time)` raster for quick maps |
| CS | `(Xdim, Ydim, nf, lev, time)` | `lons`, `lats`, `corner_lons`, `corner_lats`, `cell_area`, `cubed_sphere` mapping |

CS coordinates use the same `CubedSphereMesh` panel convention as regridding.
`GEOSNativePanelConvention` includes the GEOS-FP/GEOS-IT panel order, native
orientation, and global `-10°` longitude offset.

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

From TOML, runners read these `[output]` keys:

```toml
[output]
snapshot_hours = [0, 6, 12, 24]
snapshot_file = "~/data/AtmosTransport/output/run.nc"
deflate_level = 1
shuffle = true
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
