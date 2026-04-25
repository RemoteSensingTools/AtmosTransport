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
  path:           /temp2/.../geos_transport_20211201_float32.bin
  grid_type:      :cubed_sphere
  Nc:             180
  Nz:             72
  windows:        24
  steps_per_met:  8
  mass_basis:     :dry
  panel_convention: :geos_native
  payload_sections: [:m, :am, :bm, :cm, :ps, :dm]
  flux_kind:      :substep
  source_flux_sampling: :window_constant

Capabilities:
  ✓ supports_advection
  ✓ supports_diagnostic_replay
  ✓ supports_dry_basis_runtime
  ✗ supports_convection         (cmfmc / dtrain not present)
  ✗ supports_diffusion          (Kz not present)
  ✓ supports_surface_pressure
  ✓ supports_flux_delta
```

The capability rows tell the runtime which operators are eligible. A
config that requests `convection.kind = "cmfmc"` against a binary
without `:cmfmc` / `:dtrain` payload sections is rejected at load time —
no silent capability mismatch.

For pre-plan-39 binaries that lack the 8 self-describing header fields,
pass `--allow-legacy` to demote the contract violation to a warning
(runtime behavior is then **not** trusted).

## Inspect a snapshot NetCDF

The runtime's `[output]` block writes a single NetCDF at `snapshot_file`,
one frame per entry in `snapshot_hours`. The variables and their
dimensions depend on the target topology:

For a lat-lon snapshot the actual variable list looks like (verified
against `config/runs/quickstart/ll72x37_advonly.toml`):

```
lev, time, lon, lat, lon_bounds, lat_bounds, cell_area,
air_mass, air_mass_per_area, column_air_mass_per_area,
co2_bl, co2_bl_column_mean, co2_bl_column_mass_per_area
```

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
    `time`). They will need a small refresh before recommending — track
    in the documentation overhaul follow-ups.

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
| Uniform-tracer jump from 400 → ~535 ppm near the surface | Hybrid PE bug in vertical remap; should be direct `cumsum` PE. |
| Day-boundary continuity warnings | Regenerate the binary with the current preprocessor (the contract evolves). |

Refer to [CLAUDE.md](https://github.com/RemoteSensingTools/AtmosTransport/blob/main/CLAUDE.md)
in the repository root for the full Fast-Failure-Triage table — it
encodes hard-won debugging knowledge.

## What's next

- [Concepts](#) — the model architecture (Phase 3).
- [Theory & Verification](#) — mass conservation, advection schemes,
  validation results (Phase 6).
