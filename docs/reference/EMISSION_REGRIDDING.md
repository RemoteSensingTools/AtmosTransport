# Emission Regridding Tutorial

How to preprocess gridded emission inventories for use with AtmosTransport
on cubed-sphere grids. Covers the TOML-driven pipeline, conservative
regridding algorithm, and validation against GEOS-Chem reference data.

## Overview

AtmosTransport ingests surface emission fluxes as **kg/m²/s** on the model
grid. For cubed-sphere (CS) simulations, lat-lon emission inventories must
be regridded to CS panels. The pipeline:

```
Source NetCDF (lat-lon)  →  conservative regrid  →  CS binary (.bin)  →  model
```

The regridding is mass-conserving by construction: total emission mass on
the source grid equals total mass on the target grid, regardless of
resolution mismatch.

## Quick Start

Regrid a single emission source to cubed-sphere binary:

```bash
julia --project=. scripts/preprocessing/regrid_emissions.jl \
    config/emissions/edgar_sf6.toml
```

This reads the TOML config, loads the source NetCDF, builds the
conservative regrid map, writes a compact binary, and prints mass
conservation diagnostics.

## Supported Sources

| Source | Resolution | Units | Config |
|--------|-----------|-------|--------|
| EDGAR v8.0 SF6 | 0.1° | tonnes/cell/year | `config/emissions/edgar_sf6.toml` |
| GridFED fossil CO2 | 0.1° | various | `config/emissions/gridfed_fossil_co2.toml` |
| Zhang Rn222 | 0.5° | kg/m²/s | `config/emissions/zhang_rn222.toml` |
| LMDZ/CAMS CO2 | ~1.9° | kgC/m²/s | `config/emissions/lmdz_co2.toml` |

Any regular lat-lon NetCDF source can be added by writing a new TOML config.

## TOML Config Format

Each emission source has a TOML file in `config/emissions/`:

```toml
[source]
type = "netcdf_latlon"
path = "~/data/emissions/edgar_v8/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc"
variable = "auto"           # auto-detect first non-coordinate variable
lon_coord = "lon"
lat_coord = "lat"

[units]
input = "tonnes/cell/year"
output = "kg/m2/s"
conversions = ["tonnes_per_cell_to_kgm2s"]

[target]
grid = "cubed_sphere"
Nc = 180
coord_file = "~/data/geosit_c180_catrine/20211201/GEOSIT.20211201.CTM_A1.C180.nc"
# gridspec_file = "data/grids/cs_c180_gridspec.nc"  # optional; auto-generated if omitted

[output]
path = "~/data/preprocessed/edgar_sf6_cs_c180_float32.bin"
species = "sf6"
float_type = "Float32"

[validation]
expected_total = 10.0       # kt/yr
expected_unit = "kt/yr"
tolerance = 0.02            # 2% tolerance
```

### Key fields

- **`[source].path`**: Path to the source NetCDF (supports `~` and globs)
- **`[target].gridspec_file`** (optional): Path to CS grid specification file with
  exact cell corners and areas (see below). If omitted, GMAO coordinates and
  areas are auto-generated via `generate_cs_gridspec(Nc)` for any resolution
- **`[units].conversions`**: Unit conversion chain applied before regridding.
  Available: `tonnes_per_cell_to_kgm2s`, `multiply_by_KGC_TO_KGCO2`
- **`[validation]`**: Optional mass budget check after regridding

## Grid Specification

The regridder needs GMAO cubed-sphere cell centers and areas. These can come
from two sources:

### Auto-generation (default)

When no `gridspec_file` is specified, `build_conservative_cs_map` automatically
calls `generate_cs_gridspec(Nc)` to compute exact GMAO coordinates for any
resolution. This is a pure-Julia port of gcpy's `CSGrid` + `csgrid_gmao`
(gnomonic projection with GMAO face orientation). Generated areas match
GEOS-Chem's `Met_AREAM2` to <0.00001% per cell.

No external files or downloads required — works for C24, C180, C720, or any Nc.

### Pre-computed gridspec file (optional)

For repeated preprocessing, a pre-computed `gridspec_file` avoids regeneration:

`data/grids/cs_c180_gridspec.nc` is a compact (7.8 MB) NetCDF containing:

| Variable | Shape | Description |
|----------|-------|-------------|
| `corner_lons` | (181, 181, 6) | Cell corner longitudes |
| `corner_lats` | (181, 181, 6) | Cell corner latitudes |
| `lons` | (180, 180, 6) | Cell center longitudes |
| `lats` | (180, 180, 6) | Cell center latitudes |
| `areas` | (180, 180, 6) | Exact cell areas [m²] |

This file was extracted from a GEOS-Chem diagnostic using
`scripts/preprocessing/extract_cs_gridspec.jl`. To generate a gridspec for
other resolutions, use `generate_cs_gridspec(Nc)` directly.

## Conservative Regridding Algorithm

The regridding uses **sub-cell sampling** to compute overlap weights
between source (lat-lon) and target (cubed-sphere) cells:

### Step 1: Build the regrid map

For each source cell:
1. Place N_sub x N_sub uniformly-spaced sample points within the cell
2. For each sample, find the nearest CS cell center via spatial binning
3. The fraction of samples landing in each CS cell gives the overlap weight

N_sub scales with the resolution ratio:
- Fine-to-coarse (0.1° EDGAR on C180 ~0.5°): N_sub = 5 (25 samples)
- Comparable (0.5° Rn222 on C180): N_sub = 8 (64 samples)
- Coarse-to-fine (1° LMDZ on C180): N_sub = 20 (400 samples)

The map is stored as a compressed sparse row (CSR) structure and reused
for all time steps of the same source grid.

### Step 2: Distribute mass

For each source cell with flux `f` [kg/m²/s]:
```
mass_rate = f × native_cell_area        [kg/s]
mass_acc[cs_cell] += mass_rate × weight  [kg/s]
```

Weights sum to 1.0 per source cell, so total mass is conserved exactly.

### Step 3: Convert to flux density

```
flux_out[cs_cell] = mass_acc[cs_cell] / eff_area[cs_cell]   [kg/m²/s]
```

where `eff_area` is the effective area from sampling (sum of weighted
source areas contributing to each CS cell). This guarantees that a
uniform input field produces a uniform output field.

### Mass conservation

Mass is conserved by construction:
- `total_src = sum(flux[i,j] * native_area[j])` for all source cells
- `total_tgt = sum(mass_acc)` = `total_src` (weights sum to 1)

The exact CS cell areas (from gridspec file or auto-generated) are used for
mass budget diagnostics but not for the density conversion itself.

## Comparison with ESMF/GCHP

GEOS-Chem uses ESMF's `ESMF_RegridWeightGen` which computes exact
polygon-polygon intersections on the sphere. Our sub-cell sampling
trades per-cell accuracy for simplicity and zero external dependencies:

| Aspect | ESMF | Sub-cell sampling |
|--------|------|-------------------|
| Overlap | Exact polygon intersection | N_sub² sample points |
| Per-cell accuracy | Machine precision | ~0.1-1% median |
| Mass conservation | Exact | Exact |
| Dependencies | ESMF (Fortran) | Pure Julia |

For atmospheric transport, where other model uncertainties (meteorology,
boundary layer mixing, emission inventories) are typically 5-20%, the
sub-cell sampling accuracy is more than sufficient.

## Lat-Lon to Lat-Lon Regridding

For lat-lon model grids (e.g., ERA5), the function `conservative_regrid_ll`
in `src/Sources/regrid_utils.jl` computes **exact** spherical overlap
fractions between source and target cells using the sin(phi) latitude
formula. No sampling approximation is needed because rectangles on
rectangles have analytically computable intersections.

This handles arbitrary non-integer resolution ratios (e.g., 0.1° to 0.25°).

## Validation

### Running the validation suite

```bash
julia --project=. scripts/validation/validate_conservative_regrid.jl
```

This runs three tests using the C180 grid:

1. **Uniform field**: 1° constant flux regridded to C180. All CS cells
   should be exactly 1.0 (verifies eff_area consistency).

2. **EDGAR SF6**: 0.1° source regridded and compared per-cell against
   GEOS-Chem's `EmisSF6` on the same C180 grid.

3. **Zhang Rn222**: 0.5° source regridded and compared against
   GEOS-Chem's `EmisRn_Soil`.

### Validated accuracy

| Test | Total mass error | Significant cells (50th pct) |
|------|-----------------|------------------------------|
| Uniform 1° → C180 | 0% (exact) | 0% (exact) |
| EDGAR SF6 0.1° → C180 | 1.1% | 7.4% |
| Zhang Rn222 0.5° → C180 | 0.002% | 0.14% |

The SF6 total mass difference reflects unit conversion differences with
GEOS-Chem (different radius, seconds-per-year), not regridding error.
Rn222 (already in kg/m²/s, no conversion needed) shows 0.002% total error.

## Adding a New Emission Source

1. **Create a TOML config** in `config/emissions/`:
   ```toml
   [source]
   type = "netcdf_latlon"
   path = "~/data/my_source.nc"
   variable = "emissions"
   lon_coord = "auto"
   lat_coord = "auto"

   [units]
   input = "kg/m2/s"
   output = "kg/m2/s"
   conversions = []

   [target]
   grid = "cubed_sphere"
   Nc = 180
   coord_file = "~/data/geosit_c180_catrine/20211201/GEOSIT.20211201.CTM_A1.C180.nc"
   gridspec_file = "data/grids/cs_c180_gridspec.nc"

   [output]
   path = "~/data/preprocessed/my_source_cs_c180_float32.bin"
   species = "my_tracer"
   float_type = "Float32"
   ```

2. **Run the preprocessor**:
   ```bash
   julia --project=. scripts/preprocessing/regrid_emissions.jl \
       config/emissions/my_source.toml
   ```

3. **Reference the binary in your run config**:
   The model auto-discovers preprocessed binaries by species name and grid
   size. Place the binary in the same directory as other preprocessed
   emissions, or specify the path explicitly in the tracer config.

## Binary Format

The output `.bin` files have a compact layout:

- **Header** (4096 bytes): JSON metadata including grid size, float type,
  species name, number of time steps, and time offsets
- **Data**: For each time step, 6 panels of `Nc × Nc` values in the
  specified float type (default Float32)

Total size for C180 single-timestep: `6 × 180 × 180 × 4 bytes ≈ 1.5 MB`.

## Key Source Files

| File | Purpose |
|------|---------|
| `src/Sources/regrid_utils.jl` | Core algorithms: `build_conservative_cs_map`, `regrid_latlon_to_cs`, `conservative_regrid_ll` |
| `scripts/preprocessing/regrid_emissions.jl` | TOML-driven preprocessor CLI |
| `scripts/preprocessing/extract_cs_gridspec.jl` | One-time grid spec extraction |
| `scripts/validation/validate_conservative_regrid.jl` | Validation suite |
| `config/emissions/*.toml` | Per-source configuration files |
| `data/grids/cs_c180_gridspec.nc` | CS grid spec (corners, centers, areas) |
