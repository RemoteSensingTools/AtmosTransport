# Meteorological Data: Preprocessing and Data Sources

This document explains how meteorological data flows into both TM5 and our Julia
model, what preprocessing each requires, and why TM5's approach differs
fundamentally from a simple gridpoint wind-based method.

## Overview

| Aspect | TM5 | Julia AtmosTransportModel |
|--------|-----|--------------------------|
| Advection input | Mass fluxes (kg/s) through cell faces | Wind velocities (m/s) at cell centers |
| Mass conservation | Guaranteed by spectral integration (Bregman et al. 2003) | Depends on advection scheme discretization |
| Vertical coordinate | Hybrid sigma-pressure (A/B coefficients) | Hybrid sigma-pressure or pressure levels |
| Convective fluxes | Read from ECMWF archive (eu, ed, du, dd) | Read from met data if available |
| Met format | Preprocessed NetCDF/HDF with specific naming | Standard NetCDF (ERA5, GEOS-FP, MERRA-2) |

## TM5 Meteo Preprocessing

### Why TM5 needs its own preprocessor

TM5 uses **mass fluxes** (kg/s through cell faces), not wind velocities, for
tracer advection. Computing mass fluxes from ECMWF wind fields requires working
in **spherical harmonic (spectral) space** to guarantee that the resulting mass
fluxes satisfy the continuity equation exactly. A naive gridpoint computation
(multiply u by dp/g and cell width) introduces mass imbalances that grow over
time.

The foundational paper is:

> Bregman, B., Segers, A., Krol, M., Meijer, E., and van Velthoven, P.:
> "On the use of mass-conserving wind fields in chemistry-transport models",
> Atmos. Chem. Phys., 3, 447-457, 2003.
> https://acp.copernicus.org/articles/3/447/2003/

### TM5 is its own preprocessor

The preprocessing code lives inside TM5 itself, in the TMM (Transport Model
Meteo) module:

- **`base/src/tmm.F90`**: Core mass flux computation
  - `tmm_Read_MFUV` (~line 1944): horizontal mass fluxes from spectral U/V
  - `tmm_Read_MFW` (~line 2234): vertical mass flux from spectral divergence
- **`base/src/tmm_mf_ecmwf_tmpp.F90`**: Reads raw ECMWF GRIB files
- **`base/src/file_grib.F90`**: GRIB I/O using GRIBEX library
- **`base/src/grid_interpol.F90`**: Spectral-to-gridpoint transforms

The mass flux formulas (computed in spectral space):

```
mfu = (R/g) * integral[ U * (dA + dB * exp(LNSP)) / cos(lat) * dlat ]
mfv = (R/g) * integral[ V * (dA + dB * exp(LNSP)) * dlon ]
mfw = derived from spectral divergence + surface pressure tendency
```

where `A`, `B` are hybrid level coefficients, `LNSP` is log surface pressure,
`R` is Earth's radius, and `g` is gravitational acceleration. The integrals are
evaluated analytically in spectral space.

### Preprocessing mode

When TM5 is configured with `tmm.output : T`, it:

1. Reads raw ECMWF GRIB data (spectral vorticity, divergence, LNSP)
2. Computes mass-conserving mass fluxes via spectral integration
3. Writes preprocessed files in TM5's NetCDF format
4. Subsequent runs read from this archive with `tmm.output : F`

### Required ECMWF fields (spectral GRIB)

TM5's mass flux computation requires these ERA5 fields **in spectral form**
(spherical harmonic coefficients), not gridpoint:

| Field | GRIB param | Description | Form |
|-------|-----------|-------------|------|
| VO | 138 | Vorticity | Spectral |
| D | 155 | Divergence | Spectral |
| LNSP | 152 | Log surface pressure | Spectral |
| T | 130 | Temperature | Spectral or gridpoint |
| Q | 133 | Specific humidity | Gridpoint |
| CLWC | 246 | Cloud liquid water | Gridpoint |
| CIWC | 247 | Cloud ice water | Gridpoint |
| CC | 248 | Cloud cover | Gridpoint |

Convective fields (gridpoint, from ECMWF forecast):

| Field | GRIB param | Description |
|-------|-----------|-------------|
| MUMF | 20 (table 162) | Updraft mass flux |
| MDMF | 21 (table 162) | Downdraft mass flux |
| EU | 44 | Updraft entrainment |
| ED | 45 | Updraft detrainment |
| DU | 46 | Downdraft entrainment |
| DD | 47 | Downdraft detrainment |

Surface fields (gridpoint):

| Field | GRIB param | Description |
|-------|-----------|-------------|
| SP | 134 | Surface pressure |
| Z | 129 | Orography (geopotential) |
| LSM | 172 | Land-sea mask |
| BLH | 159 | Boundary layer height |
| 10U | 165 | 10m U-wind |
| 10V | 166 | 10m V-wind |
| 2T | 167 | 2m temperature |
| 2D | 168 | 2m dewpoint |
| SSHF | 146 | Surface sensible heat flux |
| SLHF | 147 | Surface latent heat flux |

### TM5 preprocessed file format

After preprocessing, TM5 stores files as NetCDF with:

- **Directory structure**: `ec-ea-fc012up2tr3-ml137-glb100x100/YYYY/MM/`
- **File naming**: `<paramkey>_YYYYMMDD_<tres>.nc` (e.g. `mfuv_20250201_00p03.nc`)
- **Variable groups**: `sp`, `mfuv`, `mfw`, `t`, `q`, `cld`, `convec`
- **Dimensions**: `lon`, `lat`, `lev`, `time`
- **Hybrid coefficients**: `ap`, `b` (and bounds) for vertical coordinate
- **Surface pressure**: `ps` included in every 3D file
- **Time**: 3-hourly (`_00p03` = 00:00, 03:00, 06:00, ...)
- **CF conventions**: CF-1.4 with standard names

The I/O module is `base/src/tmm_mf_tm5_nc.F90` (and the ERA5-specific
variant in `proj/era5_met/src/tmm_mf_tm5_nc.F90`).

### Prerequisites for running TM5 preprocessing

1. **ecCodes** (or legacy GRIBEX) with Fortran bindings, compiled with `ifx`
2. TM5 built with `with_tmm_ecmwf` and `with_grib` macros enabled
3. ERA5 data downloaded as GRIB with spectral coefficients (not gridpoint NetCDF)
4. CDS API access configured (`~/.cdsapirc`)

See [TM5_LOCAL_SETUP.md](TM5_LOCAL_SETUP.md) for build instructions.

## Julia Model Meteorological Input

### Design Principle: Universal Hybrid Grid

**All met data sources run on the same hybrid sigma-pressure vertical
coordinate.** The transport model never operates on raw pressure levels —
every source provides A/B coefficients that define terrain-following levels:

```
p(k) = A[k] + B[k] × p_surface
```

This is a fundamental design decision:

- **ERA5** (137 levels): spectral-harmonic native grid → distributed on
  regular lat-lon. A/B coefficients from `config/era5_L137_coefficients.toml`.
- **GEOS-FP** (72 levels): cubed-sphere internally → output on regular
  lat-lon. A/B from `config/geos_L72_coefficients.toml`.
- **MERRA-2** (72 levels): same vertical coordinate as GEOS-FP, different
  horizontal resolution. Same `geos_L72_coefficients.toml`.
- **Future sources**: any met data that provides horizontal winds, surface
  pressure, and hybrid A/B coefficients can be added by writing a new TOML
  config and (if needed) a coefficient file. No Julia code changes required.

The `HybridSigmaPressure` type provides the universal abstraction. A
vertical coordinate is constructed from any met source config with:

```julia
config = default_met_config("geosfp")
vc = build_vertical_coordinate(config)  # 72-level HybridSigmaPressure
```

This means the transport model sees the same type regardless of whether
the underlying data came from ECMWF spectral harmonics, NASA's cubed-sphere
model, or any future gridded dataset. The physics code (advection,
convection, diffusion) only depends on the `HybridSigmaPressure` interface.

### Supported data sources

| Source | Config file | Resolution | Levels | Coefficients | Access |
|--------|------------|------------|--------|-------------|--------|
| ERA5 | `config/met_sources/era5.toml` | 0.25-2 deg | 137 model | `era5_L137_coefficients.toml` | CDS API (`~/.cdsapirc`) |
| GEOS-FP | `config/met_sources/geosfp.toml` | 0.25 deg | 72 hybrid | `geos_L72_coefficients.toml` | OPeNDAP (no auth) |
| MERRA-2 | `config/met_sources/merra2.toml` | 0.5x0.625 deg | 72 hybrid | `geos_L72_coefficients.toml` | NASA Earthdata (`~/.netrc`) |

### Processing pipeline

1. **Read** (`src/IO/met_reader.jl`): Load NetCDF variables, apply unit conversions
2. **Build vertical coordinate**: `build_vertical_coordinate(config)` loads
   A/B coefficients and constructs `HybridSigmaPressure`
3. **Stagger** (`src/IO/met_fields_bridge.jl`): Interpolate cell-center winds to faces
   - `u` -> x-faces (Nx+1, Ny, Nz)
   - `v` -> y-faces (Nx, Ny+1, Nz)
   - `omega` -> w at z-interfaces (Nx, Ny, Nz+1), sign flip (omega>0 downward, w>0 upward)
4. **Convection**: Read `conv_mass_flux_up/down` if available, else convection is a no-op
5. **Diffusivity**: Read `kh` (eddy diffusivity) if available

### Download scripts

| Script | Source | Output location |
|--------|--------|----------------|
| `scripts/download_era5_week.jl` | ERA5 (CDS API) | `~/data/metDrivers/era5/` |
| `scripts/download_geosfp_week.jl` | GEOS-FP (OPeNDAP) | `~/data/metDrivers/geosfp/` |
| `scripts/download_test_data.jl` | All three | `~/data/metDrivers/*/test/` |

### Current status

- ERA5 model-level data (L137) now supported via `scripts/download_era5_model_levels.jl`
- A/B coefficients available for both ERA5 (137 levels) and GEOS/MERRA (72 levels)
- Universal `build_vertical_coordinate()` works for all met sources
- No convective mass flux variables in the ERA5 config yet
- No spectral processing — uses gridpoint winds directly (TM5 comparison ongoing)
- Regridding module exists but is not yet implemented

## Validation Strategy: TM5 vs Julia Model

### Using the same ERA5 data

For a fair comparison, both models must process the same underlying ERA5
meteorology. The pipeline is:

```
ERA5 (ECMWF archive)
    |
    +--> Spectral GRIB -----> TM5 preprocessing (tmm.F90) -----> TM5 forward run
    |                              mass-conserving mass fluxes
    |
    +--> Gridpoint NetCDF ---> Julia model (met_reader.jl) -----> Julia forward run
                                   direct wind-based advection
```

Both runs use the same ERA5 fields for the same time period. The comparison then
tests whether the Julia model's gridpoint approach gives comparable tracer
transport to TM5's spectral mass-flux approach.

### What the comparison tests

- **Advection accuracy**: Gridpoint winds vs spectral mass fluxes
- **Mass conservation**: TM5 guarantees it; Julia model should be close
- **Convection**: Both use ECMWF-derived convective mass fluxes (once enabled in Julia)
- **Overall transport**: Column-integrated tracers, zonal means, time series

### GEOS-FP validation (separate)

TM5 has no GEOS-FP reader. For GEOS-FP validation, compare the Julia model
against:
- GEOS-Chem (which natively uses GEOS-FP)
- Observations (surface stations, satellite retrievals)
- The Julia model already has a working GEOS-FP reader

### Mass fluxes vs. winds — lessons from GCHP v13

Martin et al. (2022, GMD 15, 8731-8748) demonstrate that **directly using air
mass fluxes** instead of inferring them from archived winds significantly reduces
transport errors in offline models. Key findings relevant to our model:

1. **Surface pressure tendency error**: Using winds gives 15× larger error than
   using mass fluxes directly (MAE 15 Pa vs 1 Pa for a 5-min timestep).

2. **Restaggering error**: Even on the native cubed-sphere grid, converting
   between A-grid (cell-center winds) and C-grid (face mass fluxes) weakens
   vertical advection by ~8% (regression slope = 0.92). Our `stagger_velocities`
   function performs exactly this operation.

3. **Regridding + restaggering**: The traditional GEOS-FP pipeline
   (C720 cubed-sphere → 0.25° lat-lon winds → regrid to model grid → restagger)
   introduces compounding errors that dominate vertical transport, especially in
   the stratosphere where vertical motion is weak.

4. **Native cubed-sphere mass fluxes**: Since March 2021, GEOS-FP operationally
   archives hourly C720 cubed-sphere mass fluxes. GEOS-IT provides C180 archives
   for 1998-present. These avoid all regridding/restaggering errors.

**Implications for our model:**
- Our current wind-based approach (cell-center → face staggering) introduces a
  known ~1% mass conservation error, consistent with our observed 0.9% drift
- Supporting direct mass flux ingestion from native cubed-sphere GEOS-FP/GEOS-IT
  archives would eliminate this error source entirely
- The `CubedSphereGrid` implementation uses the same gnomonic projection as FV3,
  enabling direct use of native GEOS-FP mass flux archives in the future

## Data on disk

Current meteorological data available at `~/data/metDrivers/`:

| Dataset | Location | Size | Details |
|---------|----------|------|---------|
| ERA5 test (1 day) | `era5/test/` | 5.9 MB | 2024-03-01, 2 deg, 8 pressure levels |
| ERA5 week | `era5/` | 178 MB | 2025-02-01 to 07, 2 deg, 20 pressure levels |
| GEOS-FP week | `geosfp/` | 654 MB | 2025-02-01 to 07, ~4x5 deg |
| TM5 meteo archive | (not yet created) | -- | Requires preprocessing step above |

## References

- Bregman et al. (2003): Mass-conserving wind fields. ACP 3, 447-457.
- Krol et al. (2005): TM5 algorithm. ACP 5, 417-432.
- Williams et al. (2017): TM5-MP description. GMD 10, 721.
- Huijnen et al. (2010): TM5 tropospheric chemistry. GMD 3, 445.
- Martin et al. (2022): GCHP v13 — improved advection, native CS archives. GMD 15, 8731-8748.
  https://doi.org/10.5194/gmd-15-8731-2022
- Putman & Lin (2007): FV3 cubed-sphere advection. JCP 227, 55-78.
- TM5 Wiki: https://sourceforge.net/p/tm5/wiki/Meteo/
