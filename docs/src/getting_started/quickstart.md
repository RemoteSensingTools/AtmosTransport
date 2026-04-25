# Quickstart with example data

This is the fastest path from a fresh clone to a real simulation, a
NetCDF output file, and a plot. We host a small bundle of preprocessed
ERA5 transport binaries (3 days, December 2021) so you can skip the
preprocessor and jump straight to running.

The bundle covers regular lat-lon at two resolutions:

| Bundle entry | Grid | Resolution | Points |
|---|---|---|---|
| `era5_ll72x37_dec2021_f32` | regular lat-lon | 5° | 72 × 37 |
| `era5_ll144x73_dec2021_f32` | regular lat-lon | 2.5° | 144 × 73 |

Both are **F32**, dry-basis, level-merged to ~34 tropospheric layers,
written by the canonical
`scripts/preprocessing/preprocess_transport_binary.jl` from raw ERA5
spectral GRIB. They give you something concrete to run, modify, and
benchmark — without depending on a multi-TB ERA5 archive.

!!! note "Cubed-sphere binaries"
    Cubed-sphere quickstart binaries (e.g. C24, C90) are not included
    in this first bundle so the download stays focused on the smallest
    runnable examples. F32 CS preprocessing is available through
    `config/preprocessing/era5_cs_c24_transport_binary_f32.toml` and
    `config/preprocessing/era5_cs_c90_transport_binary_f32.toml` when
    you have the ERA5 spectral input locally. F64 CS siblings are also
    available as `era5_cs_c24_transport_binary.toml` and
    `era5_cs_c90_transport_binary.toml`.

## 1. Download the bundle

!!! note "Bundle hosting"
    The bundle is **`atmos_transport_quickstart_v1.tar.gz`**
    (≈ **1.0 GB** compressed — the LL 144×73 binaries dominate;
    the LL 72×37-only subset is just ~80 MB raw if you only want the
    smaller one).
    SHA-256: `42c63d300c5da7e776de9b25cc00884c28e3c37abf9d421df9151793a4c85f88`.
    **TODO: paste the public Dropbox download URL into
    `scripts/download_quickstart_data.sh` once the upload is in place
    (or set `ATMOSTR_QUICKSTART_URL` in your environment).**

```bash
bash scripts/download_quickstart_data.sh
```

The script downloads the tarball, verifies its SHA-256, and extracts
it under `~/data/AtmosTransport_quickstart/met/`. After extraction:

```
~/data/AtmosTransport_quickstart/
└── met/
    ├── era5_ll72x37_dec2021_f32/
    │   ├── era5_transport_20211201_merged1000Pa_float32.bin
    │   ├── era5_transport_20211202_merged1000Pa_float32.bin
    │   └── era5_transport_20211203_merged1000Pa_float32.bin
    └── era5_ll144x73_dec2021_f32/   (3 binaries)
```

## 2. Run the simulation

Two ready-to-run example configs ship in `config/runs/quickstart/`,
one per LL resolution:

```bash
# Lat-lon, 5° (smallest, fastest)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml

# Lat-lon, 2.5° (still fast, more spatial detail)
julia --project=. scripts/run_transport.jl config/runs/quickstart/ll144x73_advonly.toml
```

Both configs run a 3-day advection-only simulation (no diffusion, no
convection, no chemistry) with a single passive tracer named `co2_bl`
initialized as a `bl_enhanced` field — uniform 400 ppm background
plus +100 ppm in the lowest 3 model layers (boundary-layer injection
at `t = 0`). Output is 13 frames every 6 hours (t=0…72h), written as
a single NetCDF per run under
`~/data/AtmosTransport_quickstart/output/`. Output variables are
`<tracer>_column_mean(…)`, `<tracer>_column_mass_per_area(…)`, and
`column_air_mass_per_area(…)` — see [Inspecting output](@ref) for the
full schema.

The configs default to `use_gpu = true`. For CPU execution edit
`[architecture] use_gpu = false` in the chosen config — both examples
in the bundle run comfortably on a recent CPU at these resolutions.

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

The two bundled configs are deliberately minimal so you can use them
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
