# Download Scripts

## Unified entry point

All downloads are now driven by TOML configs via a single script:

```bash
julia --project=. scripts/downloads/download_data.jl config/downloads/<recipe>.toml \
    [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]
```

Download recipe TOMLs are in `config/downloads/`. They reference met source
definitions in `config/met_sources/`. Output paths follow the canonical
Data Layout hierarchy (`docs/reference/DATA_LAYOUT.md`).

### Available recipes

| Recipe | Source | Chunk | Description |
|--------|--------|-------|-------------|
| `era5_native_monthly.toml` | ERA5 | Monthly | All fields: core (VO/D/T/Q/LNSP), convection, surface |
| `geosfp_c720.toml` | GEOS-FP | Per-file | C720 cubed-sphere CTM mass fluxes from WashU |
| `geosit_c180.toml` | GEOS-IT | Per-file | C180 cubed-sphere from AWS S3 |
| `merra2.toml` | MERRA-2 | Per-day | OPeNDAP download (not yet implemented) |

### Examples

```bash
# Preview what would be downloaded (no network calls)
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/geosit_c180.toml --dry-run

# Download one month of ERA5
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/era5_native_monthly.toml \
    --start 2021-12-01 --end 2021-12-31

# Check existing files for completeness
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/geosit_c180.toml --verify
```

## Legacy scripts

Individual download scripts have been moved to `legacy/` and are retained
for reference only. They will eventually be removed.
