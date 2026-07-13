# Inspecting output

A successful run leaves you with two kinds of artifacts: the
**transport binary** consumed by the runtime (input side) and the
**snapshot files** written by the runtime (output side). This page
covers the tooling for both.

## Inspect a transport binary

`scripts/diagnostics/inspect_transport_binary.jl` is a thin CLI wrapper
over the `AtmosTransport.inspect_binary` function. It auto-detects
LL / RG vs CS binaries by peeking the JSON header, runs load-time
gates, and prints a capability summary.

```bash
julia --project=. scripts/diagnostics/inspect_transport_binary.jl <path.bin>
```

Typical output:

```text
TransportBinaryReader
├── path:          /temp2/.../era5_transport_20211201_merged1000Pa_float32.bin
├── geometry:      latlon 72×37×34, ncell=2664, nface_h=…
├── storage:       Float32 on disk, load as Float32
├── basis:         dry
├── timing:        dt=3600.0 s, steps/window=8 (constant)
├── payload:       m, am, bm, cm, ps, dam, dbm, dcm, dm, qv_start, qv_end
├── humidity:      qv_start/qv_end
├── semantics:     flux_sampling=window_constant, flux_kind=substep_mass_amount
├── poisson:       scale=1.0, target=continuity
└── windows:       24
Capabilities:
  ✓ advection         (m, am, bm, cm)
  ✓ replay gate       (dam, dbm, dcm, dm)
  ✗ TM5 convection    (entu, detu, entd, detd)
  ✗ CMFMC convection  (cmfmc)
  ✗ PBL diffusion     (pblh, ustar, pbl_hflux, t2m)
  ✓ surface pressure  (ps)
  ✓ humidity          (qv_start, qv_end)
  mass_basis       = :dry
  grid_type        = :latlon
```

The capability rows tell the runtime which operators are eligible. A
config that requests `convection.kind = "cmfmc"` against a binary
without the `:cmfmc` payload section is rejected at load time —
no silent capability mismatch. (`:dtrain` is optional even when
`:cmfmc` is present.)

Transport binaries must be `format_version = 4`. Every other version is rejected by
the same reader path used by runtime drivers; regenerate them with the current
preprocessor rather than loading them with compatibility defaults.

## Inspect a snapshot NetCDF

The runtime's `[output]` block writes NetCDF snapshots at `path`, with cadence
from `hours`, `cadence_hours`, or `cadence_seconds`. `split = "single"` writes
one file for the run; `split = "daily"` writes one file per daily binary. The
variables and their dimensions depend on the target topology:

`[output.fields]` can further restrict output to selected tracers, column means
only, or selected model levels. If a variable is missing, first check whether
the run config intentionally disabled that product.

For a lat-lon snapshot the actual variable list looks like:

```
lev, time, lon, lat, lon_bounds, lat_bounds, cell_area,
air_mass, air_mass_per_area, column_air_mass_per_area,
co2_bl, co2_bl_column_mean, co2_bl_column_mass_per_area
```

Dimension order for LL payload variables follows
`netcdf_writer.jl::_def_payload_var`: 3D variables (column-mean,
column-mass-per-area) are `(lon, lat, time)`; 4D per-layer variables
are `(lon, lat, lev, time)`. Note this is the column-major Julia
storage order; readers that report `(time, lev, lat, lon)` (e.g.
`xarray`, `ncdump`) print the dimensions in row-major order — same
data, opposite-ordered axes.

Per topology, the per-tracer set is:

| Topology | Full-3D | Column mean (2D) | Column mass / area (2D) |
|---|---|---|---|
| Lat-lon | `<tracer>(time, lev, lat, lon)` | `<tracer>_column_mean(time, lat, lon)` | `<tracer>_column_mass_per_area(time, lat, lon)` |
| Reduced Gaussian | `<tracer>` (per-cell native dim) | `<tracer>_column_mean(time, lat, lon)` (rasterized) plus `<tracer>_column_mean_native(time, cell)` | `<tracer>_column_mass_per_area` |
| Cubed-sphere | `<tracer>(time, lev, nf, Ydim, Xdim)` | `<tracer>_column_mean(time, nf, Ydim, Xdim)` | `<tracer>_column_mass_per_area(time, nf, Ydim, Xdim)` |

Each frame also writes the matching `air_mass`,
`air_mass_per_area`, `column_air_mass_per_area`, and `cell_area`
fields so you can recompute mass-weighted means without re-loading the
binary. Tracer names come straight from the `[tracers.<name>]` block
in the run config — for the quickstart configs that's `co2_bl`.

### From the shell

The simplest verification is `ncdump -h`:

```bash
ncdump -h ~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc | head -40
```

For a full Python inspection:

```python
import netCDF4 as nc
ds = nc.Dataset("~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc")
print(list(ds.variables.keys()))
cm = ds["co2_bl_column_mean"][:]   # (time, lat, lon) for LL
print(cm.shape, cm.min(), cm.max(), cm.mean())
```

!!! note "Existing helper scripts"
    The repository ships `scripts/diagnostics/verify_snapshot_netcdf.py`
    and `scripts/diagnostics/quick_viz.py`, but both were written
    against an older snapshot schema (`co2_surface`, `co2_column_mean`,
    `time_hours`) and have not been updated for the current variable
    names (`<tracer>_column_mean`, `<tracer>_column_mass_per_area`,
    `time`). They will need a small refresh before recommending.

### From Julia

For programmatic access without leaving Julia:

```julia
using AtmosTransport
caps = AtmosTransport.inspect_binary("/path/to/transport.bin")
@show caps   # NamedTuple of capability booleans

using NCDatasets
ds = NCDataset("~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc")
@show keys(ds.variables)
cm = ds["co2_bl_column_mean"][:, :, end]   # last frame, (lon, lat)
```

## Common gotchas

| Symptom | First check |
|---|---|
| Transport about 8× too slow | `mass_flux_dt = 450` (FV3 dynamics step), not 1. |
| Extreme CFL or NaNs | Vertical level ordering (preprocessor auto-detects but check the binary header), or stale binary. |
| ~10 % mass loss per step | In-place sweep update bug; sweeps must ping-pong source/destination arrays. |
| Surface emissions invisible in column means | Diffusion likely disabled. |
| Uniform-tracer jump near the surface | Inconsistent pressure/mass geometry in preprocessing; verify layer pressure edges and stored vertical mass flux. |
| Day-boundary continuity warnings | Regenerate the binary with the current preprocessor (the contract evolves). |

The repository README carries the maintained Fast-Failure-Triage table.

## What's next

- [Concepts](../concepts/grids.md) — the model architecture.
- [Theory & Verification](#) — mass conservation, advection schemes,
  validation results (Phase 6).
