# Data sources

This page covers where to obtain the raw meteorological input the
preprocessor needs, how to authenticate against each source, and
the recommended local layout under `~/data/AtmosTransport/met/`.

If you only want to learn the runtime first, start with the
[Quickstart](@ref). It generates small current-format forcing locally and does
not require a data account or a large download.

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

Get a free CDS account, then drop your Personal Access Token (PAT)
in `~/.cdsapirc`:

```text
url: https://cds.climate.copernicus.eu/api
key: <YOUR-PAT>
```

CDS migrated to single PAT-style keys in September 2024; the older
`<UID>:<API-KEY>` format is no longer accepted. Get your PAT from your
CDS account profile page once you're logged in. The `base_url` in
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
| `GEOSIT.YYYYMMDD.A3dyn.C180.nc` | 3-hourly | `DTRAIN` *(only with `include_convection`)*; `U`, `V` *(with `include_vdiff_fields`)* |
| `GEOSIT.YYYYMMDD.I3.C180.nc` | 3-hourly | `T` *(only with `include_vdiff_fields`)* |

The GCHP Holtslag-Boville VDIFF data contract additionally requires
`include_vdiff_fields = true` in the preprocessing TOML and the
`config/downloads/geosit_c180_gchp_vdiff.toml` download recipe.

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
                ├── GEOSIT.20211201.A3dyn.C180.nc        # if convection or VDIFF
                └── GEOSIT.20211201.I3.C180.nc           # if GCHP VDIFF
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
files with `HH30` timestamps; test fixtures may also use `HH00`. When
`[source] include_surface = true` or
`include_convection = true`, set `[source] physics_dir` to a directory
containing `GEOSFP.YYYYMMDD.{A1,A3mstE,A3dyn}.025x03125.nc` files (or
pre-regridded CS equivalents) and the preprocessor embeds `PBLH`,
`USTAR`, `HFLUX`, `T2M`, `CMFMC`, and `DTRAIN` in the transport binary.

**MERRA-2.** `MERRA2Settings` and the wind-derived CS writer are implemented.
They read native 0.5° × 0.625° PS/QV/U/V fields, derive mass fluxes, and write
CS transport binaries through the canonical preprocessing CLI; see
`config/preprocessing/merra2_c180_dec2021_f32.toml`. MERRA-2 has no native
MFXC/MFYC, so this is deliberately separate from `AbstractGEOSSettings`.
The unified `OPeNDAPProtocol.execute!` downloader is still unavailable, so
raw files must currently be staged separately with NASA Earthdata credentials.

## Try the runtime without external data

The maintained quickstart creates a small, deterministic version-4 transport
binary in the repository's ignored `data/quickstart/` directory:

```bash
julia --project=. examples/generate_synthetic_quickstart.jl
julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml
```

This is the supported smoke test and tutorial path.

## A note on disk space

Binary size scales with grid cells, vertical levels, windows, precision, and
optional physics sections. Generate one representative day, inspect its
`payload_sections`, and size campaign storage from that file rather than from
a different source or physics configuration.

## Where to read next

- [Quickstart](@ref) — a zero-download, runnable walkthrough.
- [TOML schema](@ref) — `[input]` / `[source]` / `[grid]` reference.
- [Preprocessing overview](@ref) — the unified `process_day` dispatch.
