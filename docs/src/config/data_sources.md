# Data sources

This page covers where to obtain the raw meteorological input the
preprocessor needs, how to authenticate against each source, and
the recommended local layout under `~/data/AtmosTransport/met/`.

If you only want to run the model and don't care about the
preprocessing pipeline, **skip to [the quickstart bundle](#The-quickstart-bundle)**
at the end — that's preprocessed binaries already, downloadable
from a GitHub Release.

## ERA5 (ECMWF Reanalysis 5)

ERA5 is the primary spectral source for the LL / CS spectral
preprocessing path. Authoritative source: the
[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).

### What you need per day

| File | Variable | Format |
|---|---|---|
| `era5_spectral_YYYYMMDD_lnsp.gb` | log surface pressure (LNSP) spectral coefficients | GRIB |
| `era5_spectral_YYYYMMDD_vo_d.gb` | vorticity + divergence (VO + D) spectral coefficients | GRIB |
| `era5_thermo_ml_YYYYMMDD.nc` | model-level specific humidity (`q`), temperature, etc. | NetCDF |

The thermo file is mandatory for `mass_basis = "dry"` (the runtime
default). All three files are model-level (137 levels, hybrid σ-p).

### Credentials

Get a free CDS account, then drop your key in `~/.cdsapirc`:

```text
url: https://cds.climate.copernicus.eu/api
key: <UID>:<API-KEY>
```

(Get the `<UID>:<API-KEY>` pair from the bottom-right of your CDS
account profile page once you're logged in.) The `base_url` in
`config/met_sources/era5.toml` matches this. Tools that read the
CDS API will pick up the `~/.cdsapirc` file automatically.

### Datasets

| CDS dataset name | Use |
|---|---|
| `reanalysis-era5-complete` | Model-level spectral fields (VO, D, LNSP) — the AtmosTransport spectral preprocessor input |
| `reanalysis-era5-single-levels` | Surface fields (PS, 2T, 10U, 10V, …) |
| `reanalysis-era5-pressure-levels` | Pressure-level diagnostics (not used by the preprocessor) |

A reference download script lives outside the repository (the
historical practice is per-user `cdsapi` calls); see
`config/met_sources/era5.toml` for the canonical descriptor.

### Recommended local layout

```
~/data/AtmosTransport/met/era5/
└── 0.5x0.5/
    ├── spectral_hourly/                    # CDS reanalysis-era5-complete output
    │   ├── era5_spectral_20211201_lnsp.gb
    │   ├── era5_spectral_20211201_vo_d.gb
    │   └── …
    └── physics/
        └── era5_thermo_ml_20211201.nc      # CDS reanalysis-era5-complete with q
```

The `spectral_dir` and `thermo_dir` keys in
[TOML schema](@ref) point at these.

## GEOS-IT (NASA GMAO Integrated Tropospheric)

GEOS-IT is the primary native cubed-sphere source. C180 (~50 km) is
the production/debug resolution. GEOS-FP native C720 hourly CTM files
are wired through the same source contract, with optional 0.25°
surface/convection fallback files attached into the preprocessed
binary.

### Per-day file set

For each `YYYYMMDD`:

| File | Cadence | Variables |
|---|---|---|
| `GEOSIT.YYYYMMDD.CTM_A1.C180.nc` | hourly (window-averaged) | `MFXC`, `MFYC`, `DELP` |
| `GEOSIT.YYYYMMDD.CTM_I1.C180.nc` | hourly (instantaneous) | `PS`, `QV` |
| `GEOSIT.YYYYMMDD.A1.C180.nc` | hourly | `PBLH`, `USTAR`, `HFLUX`, `T2M` *(only with `include_surface`)* |
| `GEOSIT.YYYYMMDD.A3mstE.C180.nc` | 3-hourly | `CMFMC` *(only with `include_convection`)* |
| `GEOSIT.YYYYMMDD.A3dyn.C180.nc` | 3-hourly | `DTRAIN` *(only with `include_convection`)* |

The preprocessor needs **next-day hour 0** for the last window's
forward-flux endpoint; download `[start, end+1]` for production
runs.

### Access

| Source | URL pattern | Auth |
|---|---|---|
| **AWS S3 (primary)** — `s3://geos-chem/GEOS_C180/GEOS_IT/...` | public bucket; requester-pays NOT required | none — use `aws s3 cp --no-sign-request` |
| WashU HTTP archive (fallback) | `http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT/...` | none |

The canonical descriptor is `config/met_sources/geosit.toml` (line
50-60), including the bucket name and the WashU base URL.

### Recommended local layout

The downloader's canonical layout puts each day's collections under
a per-day subdirectory:

```
~/data/AtmosTransport/met/geosit/
└── C180/
    └── daily/
        └── raw/
            └── 20211201/
                ├── GEOSIT.20211201.CTM_A1.C180.nc
                ├── GEOSIT.20211201.CTM_I1.C180.nc
                ├── GEOSIT.20211201.A3mstE.C180.nc       # if convection
                └── GEOSIT.20211201.A3dyn.C180.nc        # if convection
```

The `[source].root_dir` key in the GEOS preprocessing TOML points at
the parent directory containing the `YYYYMMDD/` per-day folders
(`~/data/AtmosTransport/met/geosit/C180/daily/raw` in this layout).
The preprocessor's file resolver also accepts a flat directory of
all NetCDFs (no per-day subdir) — that's the `raw_catrine` layout
the project's own configs use historically.

## GEOS-FP, MERRA-2 (status)

**GEOS-FP (C720).** The active descriptor is native cubed-sphere:

- `config/met_sources/geosfp.toml` — the **native C720 hourly CTM**
  product (`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.*.nc4`).
- `config/downloads/geosfp_c720.toml` — the **native C720**
  cubed-sphere download descriptor; the **`src/Downloads/sources/geosfp.jl`**
  downloader pulls from the WashU HTTP archive (NOT the
  GEOS-IT-style AWS S3 path) into a local directory.

`GEOSSettings{:geosfp}` opens 24 hourly native CTM files plus the
next-day 00Z endpoint. The WashU archive names the hourly averaged
files with `HH30` timestamps; the resolver also accepts legacy `HH00`
fixtures. When `[source] include_surface = true` or
`include_convection = true`, set `[source] physics_dir` to a directory
containing `GEOSFP.YYYYMMDD.{A1,A3mstE,A3dyn}.025x03125.nc` files (or
pre-regridded CS equivalents) and the preprocessor embeds `PBLH`,
`USTAR`, `HFLUX`, `T2M`, `CMFMC`, and `DTRAIN` in the transport binary.

**MERRA-2.** Descriptor stub at `config/met_sources/merra2.toml`,
but MERRA-2 is **not** an `AbstractGEOSSettings` preprocessing
path — it has no native MFXC/MFYC, so the GEOS shortcut doesn't
apply. A MERRA-2 preprocessing path would need its own settings
type that derives mass fluxes from U/V/DELP at LL cell centers
(closer in shape to the spectral path); not implemented today.
NASA Earthdata Login auth would be required for actual data
access.

## The quickstart bundle

If you don't want to manage raw met data — for a smoke test, a tutorial,
or a short benchmark — there's a **preprocessed bundle** on the
[`data-quickstart-v1` GitHub Release](https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-quickstart-v1):

| Tarball | Contents |
|---|---|
| `quickstart_ll_dec2021_v1.tar.gz` (~1 GB) | LL 72×37 + LL 144×73 v4 transport binaries, F32, Dec 1-3 2021 |
| `quickstart_cs_dec2021_v1.tar.gz` (~1.6 GB) | CS C24 + CS C90 v4 transport binaries, F32, Dec 1-3 2021 |

Both built from raw ERA5 spectral via the preprocessor described in
[ERA5 spectral path](@ref). Use:

```bash
bash scripts/download_quickstart_data.sh ll      # newcomer path
bash scripts/download_quickstart_data.sh cs
bash scripts/download_quickstart_data.sh         # both
```

See [Quickstart with example data](@ref) for the runnable
walkthrough.

## A note on disk space

For reference, a 30-day production preprocessing job needs:

| Source × target | Raw input | Binary output (F32) |
|---|---|---|
| ERA5 spectral → LL 720×361 (full ERA5 res) | ~6 GB / day raw GRIB+NetCDF | ~5 GB / day binary |
| ERA5 spectral → CS C90 | same raw | ~700 MB / day binary |
| GEOS-IT C180 → CS C180 (passthrough) | ~3 GB / day raw NetCDF | ~3 GB / day binary |

Plan storage accordingly; production run datasets are tens to
hundreds of GB.

## Where to read next

- [Quickstart with example data](@ref) — runnable bundle walkthrough.
- [TOML schema](@ref) — `[input]` / `[source]` / `[grid]` reference.
- [Preprocessing overview](@ref) — the unified `process_day` dispatch.
