# Quickstart with example data

This is the fastest path from a fresh clone to a real simulation, a
NetCDF output file, and a plot. We host a small bundle of preprocessed
ERA5 transport binaries (3 days, December 2021) so you can skip the
preprocessor and jump straight to running.

The bundle covers two grid topologies at two resolutions each:

| Bundle entry | Grid | Resolution | Points |
|---|---|---|---|
| `era5_ll72x37_dec2021_f32` | regular lat-lon | 5° | 72 × 37 |
| `era5_ll144x73_dec2021_f32` | regular lat-lon | 2.5° | 144 × 73 |
| `era5_cs_c24_dec2021_f32` | cubed-sphere | C24 | 6 × 24² |
| `era5_cs_c90_dec2021_f32` | cubed-sphere | C90 (~1°) | 6 × 90² |

All four are **F32**, dry-basis, level-merged to ~34 tropospheric
layers, written by the canonical
`scripts/preprocessing/preprocess_transport_binary.jl` from raw ERA5
spectral GRIB. They give you something concrete to run, modify, and
benchmark — without depending on a multi-TB ERA5 archive.

## 1. Download the bundle

!!! note "Bundle URL"
    The bundle is hosted on Dropbox.
    **TODO: paste the public Dropbox link and SHA-256 checksum here once
    the upload is in place.** The convenience script
    `scripts/download_quickstart_data.sh` will be updated to pull from
    that URL by default.

```bash
bash scripts/download_quickstart_data.sh
```

The script downloads the tarball (~250 MB compressed), verifies its
SHA-256, and extracts it under
`~/data/AtmosTransport_quickstart/met/`. After extraction:

```
~/data/AtmosTransport_quickstart/
└── met/
    ├── era5_ll72x37_dec2021_f32/
    │   ├── era5_transport_20211201_merged1000Pa_float32.bin
    │   ├── era5_transport_20211202_merged1000Pa_float32.bin
    │   └── era5_transport_20211203_merged1000Pa_float32.bin
    ├── era5_ll144x73_dec2021_f32/   (3 binaries)
    ├── era5_cs_c24_dec2021_f32/     (3 binaries)
    └── era5_cs_c90_dec2021_f32/     (3 binaries)
```

## 2. Run the simulation

Four ready-to-run example configs ship in `config/runs/quickstart/`,
one per topology / resolution combination:

```bash
# Lat-lon, 5° (smallest, fastest)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml

# Lat-lon, 2.5° (still fast, more spatial detail)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll144x73_advonly.toml

# Cubed-sphere C24 (smallest CS — see panel-edge behavior on a small grid)
julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c24_advonly.toml

# Cubed-sphere C90 (~1°, the highest-resolution entry — best for demos)
julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c90_advonly.toml
```

Each config is a 3-day advection-only run (no diffusion, no convection,
no chemistry) with a single passive tracer named `co2_bl`. Output is
13 frames every 6 hours (t=0…72h), written as a single NetCDF per run
under `~/data/AtmosTransport_quickstart/output/`. The output variables
are `<tracer>_column_mean(…)`, `<tracer>_column_mass_per_area(…)`, and
`air_mass_column(…)` — see [Inspecting output](@ref) for the precise
dimensions per topology.

The two LL configs use a `bl_enhanced` initial condition (uniform 400
ppm plus +100 ppm in the lowest 3 model layers — boundary-layer
injection at `t = 0`). The two CS configs use a uniform 400 ppm IC
because `bl_enhanced` is currently LL-only. Both are designed so the
3-day output shows clear transport signatures.

The configs default to `use_gpu = true`. For CPU execution edit
`[architecture] use_gpu = false` in the chosen config — every example
in the bundle runs comfortably on a recent CPU at these resolutions.

## 3. Inspect the output

A successful run writes
`~/data/AtmosTransport_quickstart/output/<config-name>.nc`. The
cheapest way to confirm the run produced sensible numbers:

```bash
ncdump -h ~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc | head -30
```

You should see variables like `co2_bl(time, lev, lat, lon)` (full-3D
tracer), `co2_bl_column_mean(time, lat, lon)`,
`co2_bl_column_mass_per_area(time, lat, lon)`,
`column_air_mass_per_area(time, lat, lon)`, plus the coordinate
variables `time`, `lev`, `lon`, `lat`. CS snapshots have a
`(time, lev, nf, Ydim, Xdim)` layout per panel — see
[Inspecting output](@ref) for the full schema.

For a quick numeric sanity check from Python:

```python
import netCDF4 as nc
ds = nc.Dataset("~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc")
cm = ds["co2_bl_column_mean"][:]
print(cm.shape, "min", cm.min(), "max", cm.max(), "mean", cm.mean())
# Expect: shape (13, 37, 72)  min ~3.99e-4  max ~5e-4  mean ~4.01e-4
```

## 4. Modify and re-run

The four bundled configs are deliberately minimal so you can use them
as starting points:

| To try… | Edit |
|---|---|
| A different IC | `[tracers.co2_bl.init]` block. `kind = "uniform"` with `background = 4.0e-4` is the simplest; `kind = "bl_enhanced"` (LL only) with `background`, `enhancement`, `n_layers`; `kind = "gaussian_blob"` (LL only); `kind = "file"` / `"netcdf"` to load from disk (see `config/runs/catrine_*.toml` for file-init examples). |
| A second tracer | Add `[tracers.<name>]` and `[tracers.<name>.init]` blocks; the runtime advects all tracers in lockstep. |
| Different snapshot times | Edit `[output] snapshot_hours = […]`. |
| F64 instead of F32 | Re-preprocess from raw ERA5 to F64; use `config/preprocessing/era5_ll72x37_advresln_dec2021.toml` (the F64 sibling) as the template. |
| A different advection scheme | `[run] scheme = "ppm"`. PPM also accepts `ppm_order = 4..7`. |
| Different grid topology | Pick the matching bundle config; the runtime auto-dispatches on the binary's `grid_type` header. |

## What this quickstart does *not* cover

- **Preprocessing**. The bundle ships preprocessed binaries; the
  ERA5-spectral preprocessor itself is documented in Phase 5 of the
  documentation overhaul.
- **GEOS native CS**. The bundle is ERA5-only. GEOS-IT C180 native
  preprocessing is documented separately (Phase 5).
- **Convection / diffusion**. The bundled configs are advection-only.
  See `config/runs/c180_uniform_*.toml` for full-physics templates
  (these need the larger GEOS-IT dataset — out of scope for a
  newcomer's first run).

## What's next

- [Inspecting output](@ref) — deeper coverage of the diagnostic tools.
- [Concepts](#) — how the model is organized internally (Phase 3).
- [Tutorials](#) — Literate-driven worked examples per topology
  (Phase 4).
