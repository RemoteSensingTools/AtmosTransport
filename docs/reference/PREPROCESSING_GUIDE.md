# Preprocessing Guide

Decision tree for choosing the right preprocessor for your data and grid.

## Which preprocessor do I need?

```
What met data source?
├── ERA5
│   ├── What target grid?
│   │   ├── LatLon (0.5° or custom) → preprocess_transport_binary.jl
│   │   ├── Reduced Gaussian (N24-N320) → preprocess_era5_reduced_gaussian_transport_binary_v2.jl
│   │   └── Cubed Sphere (C24-C720) → preprocess_transport_binary.jl (LL first)
│   │                                  then regrid_ll_transport_binary_to_cs.jl
│   └── What input type?
│       ├── Spectral (VO, D, LNSP) → recommended (exact mass conservation)
│       └── Gridpoint (u, v, sp) → legacy only (~0.9% mass drift)
│
├── GEOS-FP C720 → preprocess_geosfp_cs.jl (optional binary staging)
├── GEOS-IT C180 → preprocess_geosit_cs.jl (optional binary staging)
└── MERRA-2 → not yet implemented
```

## ERA5 LatLon (recommended starting point)

The spectral preprocessor is the most validated path:

```bash
julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/era5_latlon_transport_binary_v2.toml --day 2021-12-01
```

Key config options:
```toml
[grid]
Nx = 720          # longitude cells (0.5° at 720)
Ny = 361          # latitude cells

[vertical]
n_target = 34     # number of output levels (34 = tropospheric subset)
merge_above = 1000  # merge levels above this pressure [Pa]

[mass_fix]
enable = true     # pin global mean ps (eliminates ERA5 mass drift)
```

**Output**: `era5_transport_v2_YYYYMMDD_merged1000Pa_float64.bin`
containing 12 windows × (am, bm, cm, m, qv) per day.

## ERA5 Reduced Gaussian

For ERA5-native resolution without interpolation artifacts:

```bash
julia -t8 --project=. scripts/preprocessing/preprocess_era5_reduced_gaussian_transport_binary_v2.jl \
    config/preprocessing/era5_reduced_gaussian_transport_binary_v2.toml --day 2021-12-01
```

Uses a streaming writer (low memory) and compressed-Laplacian CG for Poisson balance.

## ERA5 Cubed Sphere

Two-step process:
1. Preprocess to LatLon binary (see above)
2. Conservative regridding to cubed-sphere:

```bash
julia --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
    --input ~/data/.../era5_transport_v2_20211201_float64.bin \
    --output ~/data/.../era5_transport_v2_cs90_20211201_float64.bin \
    --Nc 90
```

Or use the spectral→CS direct path (most accurate but slower):

```bash
julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    config/preprocessing/era5_cs_c90_transport_binary.toml --day 2021-12-01
```

## What the preprocessor does

1. **Read** ERA5 spectral/gridpoint data
2. **Synthesize** winds on the target grid (spectral → grid via Legendre + FFT)
3. **Compute mass fluxes** (am, bm) from winds and surface pressure
4. **Poisson balance** the horizontal fluxes (ensures div(flux) is solvable)
5. **Diagnose cm** from continuity: cm[k+1] = cm[k] + dm - div_h
6. **Pin global ps** to fixed dry-air mass target (TM5 `Match('area-aver')` equivalent)
7. **Write** binary with v4/v5 header (provenance, checksums, ps offsets)

## Implementation layout

The transport-binary preprocessor is organized by contract first, then by
topology:

| File | Responsibility |
|------|----------------|
| `src/Preprocessing/binary_pipeline.jl` | Include index and topology contract overview |
| `src/Preprocessing/transport_binary/core.jl` | Shared payload sizing, headers, provenance, raw writes |
| `src/Preprocessing/transport_binary/latlon_workspaces.jl` | LL spectral staging, humidity, dry-basis, mass-fix, vertical merge state |
| `src/Preprocessing/transport_binary/latlon_contracts.jl` | LL replay checks, Poisson closure, v4 window writer |
| `src/Preprocessing/transport_binary/latlon_spectral.jl` | ERA5 spectral → structured LL daily workflow |
| `src/Preprocessing/transport_binary/cubed_sphere_contracts.jl` | CS endpoint-delta and write-time replay helpers |
| `src/Preprocessing/transport_binary/cubed_sphere_spectral.jl` | ERA5 spectral → CS daily workflow |
| `src/Preprocessing/transport_binary/cubed_sphere_regrid.jl` | LL binary → CS binary regrid workflow |
| `src/Preprocessing/reduced_transport_helpers.jl` | ERA5 spectral → reduced-Gaussian daily workflow |

New topologies should add a new `AbstractTargetGeometry` subtype and a
dedicated `process_day(date, grid::NewTopology, settings, vertical; ...)`
method. The new method must use explicit forward endpoint mass targets, declare
payload semantics, run write-time replay validation, and provide load-time
replay coverage in the matching driver.

## Validation

After preprocessing, check:
- `max(|cm|/m)` should be < 10^-13 (machine precision for balanced binaries)
- `ps_offsets_pa_per_window` in the header should be < 1 Pa
- Runtime should report no "STALE BINARY WARNING" or "cm-continuity check FAILED"

## Config reference

All preprocessing configs live in `config/preprocessing/`. See the TOML
files for all available options with inline documentation.
