# Download Setup

How to obtain meteorological data for AtmosTransport simulations.

## ERA5 (ECMWF Reanalysis v5)

**Resolution**: T1279 spectral (~9 km), 137 model levels, hourly
**Coverage**: 1940-present
**Access**: Copernicus Climate Data Store (CDS)

### Prerequisites

1. Create a CDS account at https://cds.climate.copernicus.eu
2. Accept the ERA5 licence terms
3. Install `cdsapi`:
   ```bash
   pip install cdsapi
   ```
4. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-uid>:<your-api-key>
   ```

### Download commands

```bash
# Spectral fields (vorticity, divergence, LNSP) — needed for mass-flux preprocessing
julia --project=. scripts/downloads/download_era5_spectral.jl \
    config/downloads/era5_spectral_dec2021.toml --day 2021-12-01

# Gridpoint fields (u, v, T, q) — for diffusion and convection
julia --project=. scripts/downloads/download_era5_gridpoint.jl \
    config/downloads/era5_gridpoint_dec2021.toml --day 2021-12-01
```

### What gets downloaded

| Field | Type | Format | Size/day |
|-------|------|--------|----------|
| VO (vorticity) | Spectral | GRIB | ~2 GB |
| D (divergence) | Spectral | GRIB | ~2 GB |
| LNSP (log surface pressure) | Spectral | GRIB | ~50 MB |
| U, V (winds) | Gridpoint | GRIB | ~4 GB |
| T (temperature) | Gridpoint | GRIB | ~2 GB |
| Q (specific humidity) | Gridpoint | GRIB | ~2 GB |

## GEOS-FP (NASA Forward Processing)

**Resolution**: C720 cubed-sphere (~12.5 km), 72 levels, 3-hourly
**Coverage**: near-real-time (1-2 day latency)
**Access**: OPeNDAP (no credentials needed)

### Download

```bash
julia --project=. scripts/downloads/download_geosfp.jl \
    config/downloads/geosfp_c720_dec2021.toml --day 2021-12-01
```

Key parameters: `mass_flux_dt = 450` (dynamics timestep in seconds).

## GEOS-IT (NASA Instrument Team)

**Resolution**: C180 cubed-sphere (~50 km), 72 levels, 3-hourly
**Coverage**: 1980-present (reanalysis)
**Access**: OPeNDAP or HTTP from WashU archive

### Download

```bash
julia --project=. scripts/downloads/download_geosit.jl \
    config/downloads/geosit_c180_dec2021.toml --day 2021-12-01
```

## Emissions and initial conditions

### GridFED fossil CO2

Download manually from https://zenodo.org/records/11048957 (GCP-GridFEDv2024.0).
Place in `~/data/AtmosTransport/catrine/Emissions/gridfed/`.

### Catrine initial conditions

The CATRINE intercomparison provides CO2 initial conditions:
`~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc`

Contact the CATRINE coordination team for access.

## Data organization

The standard layout is:
```
~/data/AtmosTransport/
├── met/
│   ├── era5/
│   │   ├── spectral/           # Raw GRIB downloads
│   │   ├── 0.5x0.5/            # Preprocessed LatLon binaries
│   │   ├── N320/               # Preprocessed Reduced Gaussian
│   │   └── C90/                # Preprocessed Cubed-Sphere
│   ├── geosfp/                 # GEOS-FP downloads
│   └── geosit/                 # GEOS-IT downloads
├── catrine/
│   ├── InitialConditions/      # CO2 ICs
│   └── Emissions/              # GridFED, EDGAR, etc.
└── output/                     # Simulation output
```

See [DATA_LAYOUT.md](DATA_LAYOUT.md) for the full specification.

## Troubleshooting

- **CDS rate limiting**: requests queue behind other users. Check status at
  https://cds.climate.copernicus.eu/requests
- **MARS vs CDS**: this project uses CDS only (no MARS access required)
- **Large downloads**: ERA5 spectral T1279 is ~12 GB/day. Use `--verify`
  mode to check existing files before re-downloading
