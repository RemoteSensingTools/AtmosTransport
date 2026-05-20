# Preprocessing Configs

Preprocessing TOMLs are consumed by:

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl <config.toml> --day YYYY-MM-DD
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl <config.toml> --start YYYY-MM-DD --end YYYY-MM-DD
```

The preprocessor writes one transport binary per day. Runtime configs under
`config/runs/` then point `[input].folder` at that output directory.

| Group | Purpose | Start Here |
|---|---|---|
| `era5_ll72x37_quickstart_v2.toml` and siblings | Rebuild the downloadable quickstart binaries from ERA5 spectral input. | Pick the grid/resolution matching `config/runs/quickstart/`. |
| `era5_*transport_binary*.toml` | ERA5 spectral to lat-lon, reduced-Gaussian, or cubed-sphere transport binaries. | Match the target grid in the filename. |
| `geosit_*` | GEOS-IT native cubed-sphere preprocessing. | `geosit_c180_native_dec2021_f32.toml` |
| `geosfp_*` | GEOS-FP native cubed-sphere preprocessing. | `geosfp_c720_native_to_cs180.toml` |
| `catrine5d/` | CATRINE campaign preprocessing matrix. | Match the runtime config under `config/runs/catrine5d/`. |
| `likely_legacy/` | Older descriptors preserved for provenance. | Avoid for new work. |

## Important Knobs

| Section | Key | Notes |
|---|---|---|
| `[input]` or `[source]` | source directories | Accept `~/...`, `$ATMOSTRANSPORT_DATA_ROOT/...`, and other set env vars. |
| `[output]` | `directory` | Runtime configs point `[input].folder` here. |
| `[preprocessing]` | `mass_flux_dt_seconds` | Must be between 100 and 3600; GEOS defaults to 450 s. |
| `[preprocessing]` | `include_surface` | Required for PBL surface fields. |
| `[preprocessing]` | `include_convection` | Required for TM5/CMFMC convection payloads. |
| `[preprocessing]` | `include_vdiff_fields` | Required for GCHP VDIFF payloads. |

## Data Roots

Production preprocessing configs normally use `$ATMOSTRANSPORT_DATA_ROOT`,
which defaults to `~/data/AtmosTransport`. Set it once instead of editing many
TOMLs:

```bash
export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/AtmosTransport
```
