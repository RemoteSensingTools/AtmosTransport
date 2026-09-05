# Output

`Output` owns the runtime snapshot contract and NetCDF file schema.

Runners decide when to sample state. `capture_snapshot(model; time_hours,
halo_width, fields)` captures the requested tracers, levels, and column
diagnostics. Column reductions run on the compute backend before transfer to
host memory. Omitting `fields` retains the full snapshot API.

Single-file NetCDF runs append each frame through `NetCDFSnapshotStream`,
keeping memory independent of snapshot count. Daily output retains one day's
frames and allows one background host-side write while transport continues.
The enclosing run owns and drains that task on success or failure, so a run
cannot finish while its daily output is still being written. If transport and
output both fail, both exceptions are reported. Binary snapshot output retains
its existing batch behavior.

## Files

- `snapshots.jl` defines `SnapshotFrame`, `SnapshotWriteOptions`, and model-state capture.
- `selected_snapshots.jl` captures compact fields and backend column sums.
- `runtime_output.jl` parses field selection, snapshot schedules, and partitioning.
- `diagnostics.jl` derives VMR, column means, and mass-per-area fields.
- `netcdf_schema.jl` defines topology-specific dimensions, coordinates, and metadata.
- `netcdf_writer.jl` writes topology-specific payload variables through one public API.
- `netcdf_stream.jl` appends and flushes records without retaining earlier frames.
- `binary_writer.jl` writes full frames in the binary snapshot format.

The batch API remains `write_snapshot_netcdf(path, frames, grid; mass_basis,
options, fields)`. Schema and payload methods are shared with the stream.

## Topology Contract

- LL writes CF lon/lat coordinates, bounds, cell areas, selected per-level fields, and column diagnostics.
- RG writes native `cell` variables, quadrilateral cell bounds, plus a legacy lon/lat raster for quick plots.
- CS writes native `(Xdim, Ydim, nf, lev, time)` fields with `lons`, `lats`, corners, cell area, and a `cubed_sphere` mapping variable.

To add a topology, implement schema and payload methods for the new mesh type.
Do not special-case the runner.
