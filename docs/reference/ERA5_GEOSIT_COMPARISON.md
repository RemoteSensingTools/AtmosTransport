# ERA5 vs GEOS-IT Comparison Run: Data Download and Preprocessing

Step-by-step guide for running the June 2023 ERA5 (0.5° lat-lon) vs GEOS-IT C180
(cubed-sphere) intercomparison. Both runs use identical physics (PPM ORD=7 advection,
Tiedtke convection, PBL diffusion) so that the only differences are the meteorological
forcing and grid.

---

## Overview

| Property | ERA5 | GEOS-IT C180 |
|----------|------|-------------|
| Grid | 720 × 361 × 137 lat-lon (0.5°) | C180 cubed-sphere (~55 km) |
| Advection | PPM ORD=7 (Putman & Lin 2007) | PPM ORD=7 |
| Mass fluxes | Spectral-derived (TM5-style) | Native model MFXC/MFYC |
| Convection | Tiedtke (from ERA5 CMFMC) | Tiedtke (from GEOS CMFMC) |
| Diffusion | PBL (from ERA5 BLH/SSHF/U10) | PBL (from GEOS PBLH/USTAR/HFLUX) |
| Config | `config/runs/era5_spectral_june2023.toml` | `config/runs/geosit_c180_june2023.toml` |

**Total data footprint** (after preprocessing, June 2023):

| Dataset | Raw | Preprocessed |
|---------|-----|-------------|
| GEOS-IT C180 NetCDF | ~15 GB | — (read on-the-fly) |
| ERA5 spectral GRIB | ~4 GB | ~7 GB NetCDF |
| ERA5 surface fields | ~2 GB | embedded in NetCDF |
| ERA5 CMFMC GRIB | ~25 GB | embedded in NetCDF |

---

## Prerequisites

### Software
- Julia 1.12+ via [juliaup](https://github.com/JuliaLang/juliaup)
- Python 3 with `cdsapi`: `pip install cdsapi`
- `eccodes` / `GRIB.jl` (already a project dependency)
- `unzip` (system package, for ERA5 surface field extraction)

### Credentials
- **CDS API** (for ERA5 downloads): register at https://cds.climate.copernicus.eu/,
  create `~/.cdsapirc`:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: <uid>:<api-key>
  ```
  MARS access (for convective mass flux) requires additionally accepting the
  ERA5 licence for "Pressure level" and "Model level" data. Standard CDS
  accounts include MARS access via `reanalysis-era5-complete`.

- **GEOS-IT data**: publicly available, no credentials needed. Archive at:
  `http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT/`

---

## Part 1: GEOS-IT C180

### 1.1 Download

GEOS-IT C180 data is read on-the-fly (NetCDF mode, no binary preprocessing
required). Download the daily NetCDF files:

```bash
# Set date range
START=2023-06-01
END=2023-06-30
OUTDIR=~/data/geosit_c180

mkdir -p "$OUTDIR"

# Download A1 (1-hourly instantaneous: PBLH, USTAR, HFLUX, T2M, ...)
# Download A3dyn (3-hourly: U, V, OMEGA for diagnostics)
# Download A3mstE (3-hourly: CMFMC convective mass flux at edges)
# Download CTM_A1 (1-hourly CTM fields: PBLH, USTAR, HFLUX, T2M)
for date in $(seq -f "%g" $(date -d "$START" +%s) 86400 $(date -d "$END" +%s) | xargs -I{} date -d @{} +%Y%m%d); do
    BASE_URL="http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT"
    YEAR=${date:0:4}
    MONTH=${date:0:6}

    mkdir -p "$OUTDIR/$date"

    for product in A1 A3dyn A3mstE CTM_A1; do
        FILE="GEOSIT.${date}.${product}.C180.nc"
        URL="${BASE_URL}/${YEAR}/${MONTH}/${FILE}"
        [ -f "$OUTDIR/$date/$FILE" ] || wget -q -P "$OUTDIR/$date/" "$URL"
    done
    echo "  $date done"
done
```

**Files per day** (4 files, ~500 MB/day total):

| File | Frequency | Contents |
|------|-----------|---------|
| `GEOSIT.YYYYMMDD.A1.C180.nc` | 1-hourly | Dynamics: DELP, PS, MFXC, MFYC |
| `GEOSIT.YYYYMMDD.A3dyn.C180.nc` | 3-hourly | U, V, OMEGA |
| `GEOSIT.YYYYMMDD.A3mstE.C180.nc` | 3-hourly | CMFMC (convective mass flux at edges) |
| `GEOSIT.YYYYMMDD.CTM_A1.C180.nc` | 1-hourly | CTM surface: PBLH, USTAR, HFLUX, T2M |

**Total for June 2023**: ~15 GB.

> **Note on vertical level ordering**: GEOS-IT stores levels bottom-to-top
> (k=1=surface, k=72=TOA), opposite to GEOS-FP (k=1=TOA). The driver
> auto-detects and flips by comparing DELP at level 1 vs Nz.

> **Note on mass flux accumulation**: GEOS-IT MFXC/MFYC are accumulated over
> the dynamics timestep (~450 s), not the full met interval. Set `mass_flux_dt = 450`
> in the config (already set in `geosit_c180_june2023.toml`).

### 1.2 Configuration

The run config is already complete at
[`config/runs/geosit_c180_june2023.toml`](../config/runs/geosit_c180_june2023.toml).
Verify `netcdf_dir` points to your data directory:

```toml
[met_data]
driver       = "geosfp_cs"
product      = "geosit_c180"
netcdf_dir   = "~/data/geosit_c180"    # ← adjust if different
start_date   = "2023-06-01"
end_date     = "2023-06-30"
mass_flux_dt = 450
```

### 1.3 Run

```bash
julia --threads=2 --project=. scripts/run.jl config/runs/geosit_c180_june2023.toml
```

Expected performance (A100 GPU, SingleBuffer): ~0.11 s/window GPU, ~1 s/window IO,
~790 s total for 30 days.

---

## Part 2: ERA5 Spectral Mass Fluxes

ERA5 requires three separate download steps plus two preprocessing steps.

### 2.1 Download Spectral GRIB Files (Mass Fluxes)

ERA5 spectral coefficients for vorticity (VO), divergence (D), and log-surface
pressure (LNSP) are used to derive mass fluxes via the TM5 spectral method
(Bregman et al. 2003). This gives near-exact mass conservation (drift < 0.01%/day
vs ~0.9%/day from gridpoint winds).

```bash
python3 scripts/download_era5_grib_tm5.py \
    --start 2023-06-01 --end 2023-06-30 \
    --outdir ~/data/metDrivers/era5/spectral_june2023
```

This downloads per-day GRIB files (~130 MB/day):
```
~/data/metDrivers/era5/spectral_june2023/
  era5_spectral_20230601_vo_d.gb     # VO + D on 137 model levels, 4 times/day
  era5_spectral_20230601_lnsp.gb     # LNSP (log surface pressure), 4 times/day
  era5_spectral_20230602_vo_d.gb
  ...  (60 files total = 30 days × 2)
```

**Total**: ~4 GB. Download time: 30–60 minutes depending on CDS queue.

> **Source**: `reanalysis-era5-complete`, model-level spectral coefficients,
> stream=oper, type=an, truncation T639 (spectral), 6-hourly (00/06/12/18 UTC).

### 2.2 Download Surface Physics Fields

Surface fields for PBL diffusion (boundary layer height, sensible heat flux,
2m temperature, friction velocity from 10m winds):

```bash
python3 scripts/download_era5_surface_fields.py \
    --start 2023-06-01 --end 2023-06-30 \
    --outdir ~/data/metDrivers/era5/surface_fields_june2023
```

This downloads per-day files from the CDS API (~60 MB/day each):
```
~/data/metDrivers/era5/surface_fields_june2023/
  era5_surface_20230601.nc     # ZIP containing BLH, SSHF, T2M, U10, V10 (hourly)
  era5_surface_20230602.nc
  ...  (30 files)
```

> **CDS API format note**: The new CDS API (v2, 2024+) returns `.nc` files that
> are actually ZIP archives containing two inner NetCDF files:
> - `data_stream-oper_stepType-instant.nc`: BLH, T2M, U10, V10
> - `data_stream-oper_stepType-accum.nc`: SSHF (accumulated J/m²)
>
> The postprocessor handles this automatically.

**Total**: ~2 GB. The script skips files that already exist (safe to re-run).

### 2.3 Download Convective Mass Flux (MARS)

Parameterized convective mass flux on model levels (param 77.128, shortName `mflx`,
TM5 convention). Requires MARS access via `reanalysis-era5-complete`:

```bash
python3 scripts/download_era5_mars_cmfmc.py \
    --start 2023-06-01 --end 2023-06-30 \
    --outdir ~/data/metDrivers/era5/cmfmc_june2023
```

Per-day GRIB files (~800 MB/day at 0.5°, all 137 model levels):
```
~/data/metDrivers/era5/cmfmc_june2023/
  era5_cmfmc_20230601.grib    # 3-hourly: steps 3/6/9/12 from 06/18 UTC
  era5_cmfmc_20230602.grib
  ...  (30 files)
```

**Total**: ~25 GB. MARS jobs are queued server-side; allow 2–6 hours.

> **Temporal sampling**: The files contain 8 values per day (steps 3/6/9/12
> from base times 06 and 18 UTC), giving valid times 09/12/15/18/21/00/03/06 UTC.
> The postprocessor interpolates these to 6-hourly met windows (00/06/12/18 UTC).

### 2.4 Preprocess Spectral GRIB → NetCDF

Convert VO/D/LNSP spectral coefficients to gridpoint mass fluxes on the 0.5° grid:

```bash
julia --project=. scripts/preprocess_spectral_massflux.jl \
    config/preprocessing/spectral_june2023.toml
```

**Processing steps** (per 6-hourly window):
1. Read spectral VO, D coefficients (T639 truncation)
2. Truncate to T359 (Nyquist for 0.5° grid)
3. VO/D → U/V via Helmholtz decomposition (`sh_vod2uv`)
4. Inverse spherical harmonic transform → 0.5° gridpoint U, V
5. Stagger U/V to cell faces (periodic in x, zero-flux at poles)
6. LNSP → surface pressure (exp transform)
7. Compute layer thickness Δp = |dA + dB·ps|
8. Compute air mass m = Δp·A/g
9. Compute mass fluxes am, bm (horizontal) and cm (vertical from continuity)
10. Write to NetCDF

**Output**:
```
~/data/metDrivers/era5/spectral_05deg_june2023/
  massflux_era5_spectral_202306_float32.nc   # ~7 GB
```

**Timing**: ~30–60 seconds per day × 30 days ≈ 15–30 minutes total on a
modern CPU (dominated by the inverse SHT at T359).

The config file [`config/preprocessing/spectral_june2023.toml`](../config/preprocessing/spectral_june2023.toml)
specifies all parameters:
```toml
[input]
spectral_dir = "~/data/metDrivers/era5/spectral_june2023"
coefficients = "config/era5_L137_coefficients.toml"

[output]
directory = "~/data/metDrivers/era5/spectral_05deg_june2023"

[grid]
nlon      = 720    # 0.5° longitude
nlat      = 361    # 0.5° latitude (includes poles)
level_top = 1
level_bot = 137

[numerics]
float_type   = "Float32"
dt           = 900.0       # advection sub-step [s]
met_interval = 21600.0     # 6-hourly met windows
```

### 2.5 Postprocess: Add Surface Physics to NetCDF

Merge the surface fields and CMFMC GRIB into the preprocessed NetCDF. This
appends `pblh`, `ustar`, `hflux`, `t2m`, and `conv_mass_flux` variables:

```bash
julia --project=. scripts/postprocess_era5_surface_physics.jl \
    config/preprocessing/spectral_june2023.toml
```

**Derived quantities**:

| Output variable | Source | Derivation |
|----------------|--------|-----------|
| `pblh` [m] | `blh` from ERA5 surface | Direct (rename) |
| `t2m` [K] | `t2m` from ERA5 surface | Direct (rename) |
| `ustar` [m/s] | `u10`, `v10` from ERA5 surface | `ustar = √(Cd) × |U₁₀|`, Cd = 1.2×10⁻³ |
| `hflux` [W/m²] | `sshf` from ERA5 surface | Deaccumulate: `hflux = −SSHF/3600` (upward positive) |
| `conv_mass_flux` [kg/m²/s] | param 77.128 from MARS | Nearest 3-hourly value at each 6-hourly window |

After postprocessing, the single NetCDF file contains all fields needed for
both transport (m, am, bm, cm) and physics (pblh, ustar, hflux, t2m,
conv_mass_flux).

> **SSHF sign convention**: ERA5 SSHF (146.128) is negative when heat flows
> upward from surface to atmosphere. `hflux = −SSHF/3600` converts to
> upward-positive W/m², consistent with the PBL diffusion kernel.

> **Optional**: if CMFMC GRIB is not yet downloaded, the postprocessor skips
> `conv_mass_flux` and the run continues without convection (a warning is
> logged per window). The PBL diffusion will still be applied.

### 2.6 Verify Preprocessed File

```bash
julia --project=. -e '
using NCDatasets
ds = Dataset(expanduser(
    "~/data/metDrivers/era5/spectral_05deg_june2023/massflux_era5_spectral_202306_float32.nc"))
println("Variables: ", keys(ds))
println("Time steps: ", length(ds["time"][:]))   # should be 120 (30 days × 4)
println("Grid: ", length(ds["lon"][:]), " × ", length(ds["lat"][:]),
        " × ", length(ds["lev"][:]))              # 720 × 361 × 137
close(ds)
'
```

Expected output:
```
Variables: ["lon", "lat", "time", "A_coeff", "B_coeff", "m", "am", "bm", "cm", "ps",
            "pblh", "ustar", "hflux", "t2m", "conv_mass_flux"]
Time steps: 120
Grid: 720 × 361 × 137
```

### 2.7 Configuration

The run config is at
[`config/runs/era5_spectral_june2023.toml`](../config/runs/era5_spectral_june2023.toml).
Verify `directory` matches your preprocessed output:

```toml
[met_data]
driver    = "preprocessed_latlon"
directory = "~/data/metDrivers/era5/spectral_05deg_june2023"   # ← adjust if different
dt        = 900
```

### 2.8 Run

```bash
julia --threads=2 --project=. scripts/run.jl config/runs/era5_spectral_june2023.toml
```

---

## Part 3: Initial Conditions

Both runs use uniform initial CO₂ = 420 ppm (global column mean for June 2023).
To use a realistic initial condition from CarbonTracker or a previous run,
add to the tracer config:

```toml
[tracers.co2]
emission      = "edgar"
initial_field = "~/data/initial_conditions/ct2023_june01.nc"  # optional
```

---

## Part 4: Running the Comparison

Once preprocessing is complete, run both models and compare:

```bash
# Terminal 1: GEOS-IT (data already available)
julia --threads=2 --project=. scripts/run.jl config/runs/geosit_c180_june2023.toml

# Terminal 2 (after ERA5 preprocessing completes):
julia --threads=2 --project=. scripts/run.jl config/runs/era5_spectral_june2023.toml
```

### Expected Outputs

```
~/data/output/
  geosit_c180_june2023_05deg.nc    # 30 days, 2-hourly, column-mean + surface + 800hPa
  era5_spectral_june2023_05deg.nc  # same format
```

Both files are on the same 0.5° lat-lon grid (`Nlon=720, Nlat=361`), making
direct comparison straightforward.

### Quick Visualization

```bash
# Animate 4-panel comparison (requires GLMakie/CairoMakie)
julia --project=. scripts/animate_4panel_comparison.jl \
    ~/data/output/geosit_c180_june2023_05deg.nc \
    ~/data/output/era5_spectral_june2023_05deg.nc
```

---

## Troubleshooting

### GEOS-IT: "extreme CFL" or NaN in first window
The vertical level ordering flip detection may have failed. Check:
```julia
# In Julia REPL
using NCDatasets
ds = Dataset("~/data/geosit_c180/20230601/GEOSIT.20230601.A1.C180.nc")
delp = ds["DELP"][:,:,:,:,1]
println("DELP level 1: ", mean(delp[:,:,1,:]))    # should be small (near TOA ~hPa)
println("DELP level 72: ", mean(delp[:,:,72,:]))  # should be large (surface ~100 hPa)
```
If level 1 is large, the auto-flip is broken — check `geosfp_cubed_sphere_reader.jl`.

### ERA5 spectral preprocessor: "No spectral GRIB files found"
The GRIB filename pattern must match `era5_spectral_YYYYMMDD_lnsp.gb`.
Check that the download script completed and files are in the right directory.

### ERA5 postprocessor: "Surface file not found"
Run `download_era5_surface_fields.py` first and verify all 30 days downloaded.

### CMFMC download: "MarsNoDataError"
Check that you are using `param=77.128` (not 78.162) and `data_format=grib`
(not the deprecated `format`). Both fixes are in the current script.

### ERA5 run: "conv_mass_flux not found in file" (warning, non-fatal)
The CMFMC postprocessor has not been run yet (or MARS download incomplete).
The run will continue without convection. Re-run the postprocessor after the
MARS download completes to add the field.

---

## Caveats and Known Differences

1. **Convective mass flux**: ERA5 uses `mflx` (param 77.128, moist convective
   mass flux) from 3-hourly forecast fields. GEOS-IT uses `CMFMC` from 3-hourly
   archive fields. These represent the same physical process but differ in
   parameterization and temporal sampling.

2. **ustar estimation**: ERA5 ustar is estimated from neutral-drag bulk formula
   (`ustar = √Cd × |U₁₀|`) rather than diagnosed from the IFS surface layer
   scheme. GEOS-IT provides a direct model-diagnosed ustar. The bulk estimate
   may underestimate ustar in convective conditions.

3. **Vertical grid**: ERA5 uses 137 hybrid sigma-pressure levels (L137);
   GEOS-IT uses 72 levels. ERA5 has much finer vertical resolution in the
   boundary layer.

4. **Horizontal resolution**: ERA5 is on a regular 0.5° grid; GEOS-IT C180
   is a cubed-sphere with ~55 km resolution (~0.5° equivalent). For output,
   both are regridded to the same 0.5° lat-lon grid.

5. **Mass flux accumulation period**: GEOS-IT MFXC/MFYC are per-dynamics-step
   (~450 s); ERA5 mass fluxes are instantaneous at 6-hourly analysis times.
   The ERA5 spectral preprocessor uses the instantaneous fields at 00/06/12/18
   UTC to represent each 6-hour window.

---

## Reference

- Putman & Lin (2007): PPM advection scheme used for horizontal transport
- Bregman et al. (2003): Spectral mass flux derivation for TM5
- Tiedtke (1989): Convective mass flux scheme
- Louis (1979): Surface layer formulation for PBL diffusion
