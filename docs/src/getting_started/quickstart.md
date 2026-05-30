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
| `quickstart_ll_dec2021_v2.tar.gz` | regular lat-lon | 5°    | 72 × 37   | ~260 MB |
| `quickstart_ll_dec2021_v2.tar.gz` | regular lat-lon | 2.5°  | 144 × 73  | ~1.0 GB |
| `quickstart_cs_dec2021_v2.tar.gz` | cubed-sphere    | C24   | 6 × 24²   | ~175 MB |
| `quickstart_cs_dec2021_v2.tar.gz` | cubed-sphere    | C90 (~1°) | 6 × 90² | ~2.4 GB |

All four are **F32**, dry-basis, level-merged to ~34 tropospheric
layers, written by the canonical
`scripts/preprocessing/preprocess_transport_binary.jl` from raw ERA5
spectral GRIB. They give you something concrete to run, modify, and
benchmark — without depending on a multi-TB ERA5 archive.

## 1. Download the bundle

Both tarballs are hosted as assets on the
[`data-quickstart-v2`](https://github.com/RemoteSensingTools/AtmosTransport.jl/releases/tag/data-quickstart-v2)
GitHub Release.

| Tarball | SHA-256 | Approx compressed |
|---|---|---|
| `quickstart_ll_dec2021_v2.tar.gz` | `656fde3472014adc3b60ebc3fd1666a2ffe4e46761ec363ec9e61c98300fd735` | ~1.0 GB |
| `quickstart_cs_dec2021_v2.tar.gz` | `ab554354a4d6a4289e5b68f9d53f3cdf97d00eb3568ada381334f49d5b59eec6` | ~1.9 GB |

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

For a different storage location, set the quickstart data root before
downloading and running:

```bash
export ATMOSTRANSPORT_DATA_ROOT_quickstart=/scratch/$USER/AtmosTransport_quickstart
bash scripts/download_quickstart_data.sh ll
```

The quickstart run configs use
`$ATMOSTRANSPORT_DATA_ROOT_quickstart/...` for both input and output paths. If
the variable is unset, AtmosTransport resolves it to
`~/data/AtmosTransport_quickstart`.

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
named `co2_bl`. They start from a localized Gaussian anomaly:
a 400 ppm background plus an 80 ppm plume centered over the northern
midlatitudes. Output is 13 frames every 6 hours (t=0…72h), written as
a single NetCDF per run under `~/data/AtmosTransport_quickstart/output/`.
Output variables are `<tracer>_column_mean(…)`,
`<tracer>_column_mass_per_area(…)`, and
`column_air_mass_per_area(…)` — see [Inspecting output](@ref) for
the full per-topology schema.

The configs default to `use_gpu = true` with automatic backend
detection: CUDA on NVIDIA hosts and Metal on Apple Silicon. The runtime
does not silently fall back to CPU if no usable GPU backend is available;
it fails so the execution path is explicit. For CPU execution edit
`[architecture] use_gpu = false` in the chosen config — every example
in the bundle runs comfortably on a recent CPU at these resolutions, with
the C90 run being the slowest at a few minutes per day.

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

The C90 cubed-sphere quickstart produces a filled column-mean heatmap
like this after 72 hours:

```@raw html
<video src="../assets/quickstart/cs_c90_advonly.mp4"
       controls="controls"
       loop="loop"
       muted="muted"
       playsinline="playsinline"
       poster="../assets/quickstart/cs_c90_advonly_t72.png"
       style="width: 100%; border: 1px solid var(--vp-c-divider); border-radius: 8px;"></video>
```

The movie was generated from the NetCDF snapshot with the topology-aware
visualization CLI:

```bash
julia --project=. scripts/visualization/atmos_viz.jl \
    --input ~/data/AtmosTransport_quickstart/output/cs_c90_advonly.nc \
    --tracer co2_bl \
    --kind movie \
    --transform column_mean \
    --ppm \
    --fps 4 \
    --out ~/data/AtmosTransport_quickstart/output/cs_c90_advonly.mp4
```

For a quick numeric sanity check from Python:

```python
import os
import netCDF4 as nc
ds = nc.Dataset(os.path.expanduser("~/data/AtmosTransport_quickstart/output/cs_c90_advonly.nc"))
cm = ds["co2_bl_column_mean"][:]
print(cm.shape, "min", cm.min(), "max", cm.max(), "mean", cm.mean())
# Expect: shape (90, 90, 6, 13)  min ~4.00e-4  max ~4.80e-4  mean ~4.06e-4
```

## 4. Modify and re-run

The four bundled configs are deliberately minimal so you can use
them as starting points:

| To try… | Edit |
|---|---|
| A different IC | `[tracers.co2_bl.init]` block. `kind = "uniform"` with `background = 4.0e-4` is the simplest; `kind = "latitude_step"` with `south_value`, `north_value`, `split_lat_deg` works on LL/RG/CS; `kind = "gaussian_blob"` with `background`, `amplitude`, `lon0_deg`, `lat0_deg`, `sigma_lon_deg`, `sigma_lat_deg`; `kind = "bl_enhanced"` (LL only) with `background`, `enhancement`, `n_layers`; `kind = "file"` / `"netcdf"` to load from disk (see `config/runs/catrine_*.toml` for file-init examples). |
| A second tracer | Add `[tracers.<name>]` and `[tracers.<name>.init]` blocks; the runtime advects all tracers in lockstep. |
| Different snapshot times | Edit `[output] snapshot_hours = […]`. |
| F64 instead of F32 | Re-preprocess from raw ERA5 to F64; use `config/preprocessing/era5_ll72x37_advresln_dec2021.toml` (the F64 sibling) as the template. |
| A different advection scheme | `[run] scheme = "ppm"` for Putman-Lin PPM, or `scheme = "linrood"` (CS only) which also accepts `ppm_order = 5` or `7`. The plain `ppm` path has no `order` parameter. |
| Different grid topology | Pick the matching bundle config; the runtime auto-dispatches on the binary's `grid_type` header. |

## What this quickstart does *not* cover

- **Preprocessing**. The bundle ships preprocessed binaries; the
  ERA5-spectral preprocessor itself is documented under
  [ERA5 spectral path](@ref).
- **GEOS native CS**. The bundle is ERA5-only. GEOS-IT C180 native
  preprocessing is covered under
  [GEOS native cubed-sphere](@ref).
- **Convection / diffusion**. The bundled configs are advection-only.
  See `config/runs/c180_uniform_*.toml` for full-physics templates
  (these need the larger GEOS-IT dataset — out of scope for a
  newcomer's first run).

## What's next

- [Inspecting output](@ref) — deeper coverage of the diagnostic tools.
- [Concepts](#) — how the model is organized internally (Phase 3).
- [Tutorials](#) — Literate-driven worked examples per topology
  (Phase 4).
