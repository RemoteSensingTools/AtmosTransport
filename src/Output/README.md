# Output

`Output` owns the runtime snapshot contract and NetCDF file schema.

Runners should only decide when to sample state. They should call
`capture_snapshot(model; time_hours, halo_width)` and
`write_snapshot_netcdf(path, frames, grid; mass_basis, options)` instead of
writing topology-specific NetCDF files directly.

## Files

- `snapshots.jl` defines `SnapshotFrame`, `SnapshotWriteOptions`, and model-state capture.
  Capture reads the PHYSICAL tracer mass via `get_tracer_full` — for
  reference-state (anomaly-transport) tracers this reconstructs
  `q_anom·m + q_ref·m`, so output files are always reference-agnostic and
  downstream diagnostics never need to know `q_ref` (plan 45).
- `diagnostics.jl` derives VMR, column means, and mass-per-area fields.
- `netcdf_schema.jl` defines topology-specific dimensions, coordinates, and metadata.
- `netcdf_writer.jl` writes topology-specific payload variables through one public API,
  plus a topology-independent `<tracer>_total_mass` (Float64, `time`) — the EXACT
  per-tracer total mass from `total_mass_full` at capture. **Use this, not an
  integral of the spatial field, for mass budgets**: for a reference-state
  (anomaly) tracer the spatial field is the F32 full-field reconstruction and its
  integral is polluted at the background-rounding scale.

## Topology Contract

- LL writes CF lon/lat coordinates, bounds, cell areas, full per-level fields, and column diagnostics.
- RG writes native `cell` variables, quadrilateral cell bounds, plus a legacy lon/lat raster for quick plots.
- CS writes native `(Xdim, Ydim, nf, lev, time)` fields with `lons`, `lats`, corners, cell area, and a `cubed_sphere` mapping variable.

To add a topology, implement schema and payload methods for the new mesh type.
Do not special-case the runner.
