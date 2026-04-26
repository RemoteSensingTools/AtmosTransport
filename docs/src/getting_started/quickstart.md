# Quickstart with example data

This is the fastest path from a fresh clone to a real simulation, a
NetCDF output file, and a plot. We host a small bundle of preprocessed
ERA5 transport binaries (3 days, December 2021) so you can skip the
preprocessor and jump straight to running.

The bundle covers two grid topologies at two resolutions each, split
into a small **lat-lon tarball** and a larger **cubed-sphere tarball**
so newcomers can grab just LL on a slow connection:

| Bundle | Grid | Resolution | Points | Approx raw size |
|---|---|---|---|---|
| `quickstart_ll_dec2021_v1.tar.gz` | regular lat-lon | 5°    | 72 × 37   | ~260 MB |
| `quickstart_ll_dec2021_v1.tar.gz` | regular lat-lon | 2.5°  | 144 × 73  | ~1.0 GB |
| `quickstart_cs_dec2021_v1.tar.gz` | cubed-sphere    | C24   | 6 × 24²   | ~175 MB |
| `quickstart_cs_dec2021_v1.tar.gz` | cubed-sphere    | C90 (~1°) | 6 × 90² | ~2.4 GB |

All four are **F32**, dry-basis, level-merged to ~34 tropospheric
layers, written by the canonical
`scripts/preprocessing/preprocess_transport_binary.jl` from raw ERA5
spectral GRIB. They give you something concrete to run, modify, and
benchmark — without depending on a multi-TB ERA5 archive.

## 1. Download the bundle

Both tarballs are hosted as assets on the
[`data-quickstart-v1`](https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-quickstart-v1)
GitHub Release.

| Tarball | SHA-256 | Approx compressed |
|---|---|---|
| `quickstart_ll_dec2021_v1.tar.gz` | `1d9928c3f43084f8397af14399f8c438a6c4bfeadabe37f0000fad3fa1ef76d7` | ~1.0 GB |
| `quickstart_cs_dec2021_v1.tar.gz` | `ada76e875cf2852d23f544f9aeb41456e6f13c502d4d6227fac676dcca554b94` | ~1.6 GB |

```bash
# Just LL (newcomer path; small download)
bash scripts/download_quickstart_data.sh ll

# Just CS
bash scripts/download_quickstart_data.sh cs

# Both (default; everything ready for the four example configs)
bash scripts/download_quickstart_data.sh
```

The script downloads, verifies SHA-256, validates the tar contents
against absolute / parent-traversing paths, and extracts under
`~/data/AtmosTransport_quickstart/met/`. After extraction (with
`all`):

```
~/data/AtmosTransport_quickstart/
└── met/
    ├── era5_ll72x37_dec2021_f32/      (3 binaries, ~88 MB each)
    ├── era5_ll144x73_dec2021_f32/     (3 binaries, ~352 MB each)
    ├── era5_cs_c24_dec2021_f32/       (3 binaries, ~58 MB each)
    └── era5_cs_c90_dec2021_f32/       (3 binaries, ~806 MB each)
```

## 2. Run the simulation

Four ready-to-run example configs ship in `config/runs/quickstart/`,
one per topology / resolution combination:

```bash
# Lat-lon, 5° (smallest, fastest)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml

# Lat-lon, 2.5° (still fast, more spatial detail)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll144x73_advonly.toml

# Cubed-sphere C24 (smallest CS — see panel-edge transport on a coarse grid)
julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c24_advonly.toml

# Cubed-sphere C90 (~1°, the highest-resolution entry — best demo)
julia --project=. scripts/run_transport.jl config/runs/quickstart/cs_c90_advonly.toml
```

The four configs all run a **3-day advection-only** simulation (no
diffusion, no convection, no chemistry) with a single passive tracer
named `co2_bl`. The two **lat-lon** configs use a `bl_enhanced` IC
(uniform 400 ppm + 100 ppm in the lowest 3 model layers — a BL
injection at `t = 0`); the two **cubed-sphere** configs use a
`uniform` 400 ppm IC (`bl_enhanced` is currently LL-only). Output is
13 frames every 6 hours (t=0…72h), written as a single NetCDF per
run under `~/data/AtmosTransport_quickstart/output/`. Output
variables are `<tracer>_column_mean(…)`,
`<tracer>_column_mass_per_area(…)`, and
`column_air_mass_per_area(…)` — see [Inspecting output](@ref) for
the full per-topology schema.

The configs default to `use_gpu = true`. For CPU execution edit
`[architecture] use_gpu = false` in the chosen config — every
example in the bundle runs comfortably on a recent CPU at these
resolutions, with the C90 run being the slowest at a few minutes
per day.

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

The four bundled configs are deliberately minimal so you can use
them as starting points:

| To try… | Edit |
|---|---|
| A different IC | `[tracers.co2_bl.init]` block. `kind = "uniform"` with `background = 4.0e-4` is the simplest; `kind = "bl_enhanced"` (LL only) with `background`, `enhancement`, `n_layers`; `kind = "gaussian_blob"` (LL only); `kind = "file"` / `"netcdf"` to load from disk (see `config/runs/catrine_*.toml` for file-init examples). |
| A second tracer | Add `[tracers.<name>]` and `[tracers.<name>.init]` blocks; the runtime advects all tracers in lockstep. |
| Different snapshot times | Edit `[output] snapshot_hours = […]`. |
| F64 instead of F32 | Re-preprocess from raw ERA5 to F64; use `config/preprocessing/era5_ll72x37_advresln_dec2021.toml` (the F64 sibling) as the template. |
| A different advection scheme | `[run] scheme = "ppm"` for Putman-Lin PPM, or `scheme = "linrood"` (CS only) which also accepts `ppm_order = 5` or `7`. The plain `ppm` path has no `order` parameter. |
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
