# Preprocessing overview

The preprocessor turns **raw meteorological input** (ERA5 spectral GRIB,
GEOS-IT/FP native NetCDF, …) into the **v4 transport binary** the
runtime consumes. It runs offline, once per day per (source, target)
combination; the runtime then memory-maps the result.

!!! warning "Preprocessing is time-intensive — and that's the point"
    A single day of GEOS-IT C180 → CS C180 preprocessing takes a few
    minutes on a current workstation; a global ERA5 spectral day at
    LL 720×361 takes longer. **This is intentional.** The
    preprocessor does the expensive work once: spectral synthesis,
    vertical level merging, conservative regridding, mass-fix,
    Poisson balance, write-time replay-gate verification. The
    resulting binary is a flat, self-describing, memory-map-friendly
    file with **fixed-stride per-window I/O** so the runtime can
    stream a window in microseconds. Plan to spend hours
    preprocessing a multi-day production dataset; expect to spend
    fractions of a second per window during the actual simulation.

## Two dispatch axes, one entry point

The whole preprocessor sits behind one orchestrator method that
dispatches on **source** (which raw met data) × **target** (which
output topology):

```
process_day(date, grid::AbstractTargetGeometry,
                  settings::AbstractMetSettings,
                  vertical; …)
```

- **`settings`** — the source. `GEOSITSettings` /
  `GEOSFPSettings` for the native GEOS family (subtypes of
  `AbstractMetSettings` / `AbstractGEOSSettings`); the spectral-ERA5
  path predates the trait and uses an untyped settings NamedTuple
  built directly from the preprocessing TOML.
- **`grid::AbstractTargetGeometry`** — the target. `LatLonTargetGeometry`,
  `ReducedGaussianGeometry`, `CubedSphereTargetGeometry`.

Each (source, target) pair routes to a specialized orchestrator file
under `src/Preprocessing/transport_binary/`:

| Source ↓  Target → | LatLon | Reduced Gaussian | CubedSphere |
|---|---|---|---|
| **Spectral ERA5** | `latlon_spectral.jl` | `reduced_transport_helpers.jl` (RG path lives here, not in a `_spectral.jl` peer) | `cubed_sphere_spectral.jl` |
| **GEOS-IT** native | (planned, deferred) | (planned, deferred) | `cubed_sphere_geos.jl` |
| **GEOS-FP** native | (planned, deferred) | (planned, deferred) | hourly C720 CTM reader; identity or nested C720→C180 block coarsening with optional physics fallback |
| **MERRA-2** (future) | (planned) | (planned) | (planned) |

```mermaid
flowchart LR
    RAW[Raw meteo<br/>GRIB / NetCDF]
    READER[Source reader<br/>fills RawWindow]
    REGRID[Regrid /<br/>identity]
    BAL[Mass-fix +<br/>Poisson / cm]
    GATE[Write-time<br/>replay gate]
    BIN[v4 transport<br/>binary]
    RAW --> READER
    READER --> REGRID
    REGRID --> BAL
    BAL --> GATE
    GATE --> BIN
```

The native-GEOS path uses `RawWindow{FT, A2, A3}` (declared in
`src/Preprocessing/met_sources.jl`) as the in-flight intermediate
between source readers and target orchestrators. The legacy spectral
path threads its own intermediate buffers through
`SpectralTransformWorkspace` rather than `RawWindow`; both
ultimately call into the same writer machinery.

## Run from the CLI

The canonical entry point:

```bash
julia --project=. -t8 scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --day 2021-12-01

# Or a date range (native GEOS-source path only):
julia --project=. -t8 scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --start 2021-12-01 --end 2021-12-03
```

The CLI accepts:

- **`--day YYYY-MM-DD`** on every source path (spectral and native).
- **`--start YYYY-MM-DD --end YYYY-MM-DD`** on the **native-source
  paths only** (GEOS-IT today). The spectral path either takes
  `--day` or, if neither flag is given, processes every day for
  which spectral input is on disk.

`-t8` enables 8 Julia threads — the spectral synthesis path
parallelizes naturally per latitude row, so threads pay off. The
script reads the TOML, looks at the `[grid]` block to pick the
target geometry, looks at the `[input]` or `[source]` block to pick
the source, and dispatches to the right `process_day` method.

## What a preprocessing config contains

A typical config has six blocks:

```toml
# WHERE the raw met data lives
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"

# WHERE the output binary goes + on what basis
[output]
directory  = "~/data/AtmosTransport/met/era5/ll72x37_advresln/transport_binary_v2_tropo34_dec2021_f32"
mass_basis = "dry"          # binary header: mass_basis = :dry

# THE TARGET TOPOLOGY (drives target-axis dispatch)
[grid]
type    = "latlon"          # or "cubed_sphere", "synthetic_reduced_gaussian"
nlon    = 72
nlat    = 37
echlevs = "ml137_tropo34"   # tropospheric vertical-level merging mode

# NUMERICS
[numerics]
float_type   = "Float32"     # binary's on_disk_float_type
dt           = 900.0         # advection sub-step (s) used for replay gate
met_interval = 3600.0        # window cadence (s); 1 hour for ERA5/GEOS-IT

# OPTIONAL: pin global-mean dry surface pressure (recommended for ERA5)
[mass_fix]
enable                = true
target_ps_dry_pa      = 98726.0
qv_global_climatology = 0.00247
```

For GEOS-native preprocessing the source axis is selected by
including a `[source]` block (e.g. `name = "geosit"`) instead of the
spectral `[input]` paths. See [GEOS native cubed-sphere](@ref) for
the full schema.

## What the orchestrator does, conceptually

For every date in the requested range:

1. **Open the day's raw data** — open all the necessary GRIB / NetCDF
   handles. For GEOS-IT this means `CTM_A1`, `CTM_I1`, optionally
   `A3mstE`, `A3dyn`. For ERA5 spectral it's `_lnsp.gb`, `_vo_d.gb`,
   `_thermo_ml_*.nc`.
2. **Build the regridder** (if source mesh ≠ target mesh) — conservative
   weights, cached as JLD2 under `~/.cache/AtmosTransport/cr_regridding/`.
   Same-mesh paths use `IdentityRegrid` (zero-overhead passthrough).
3. **Per window (typically 24 hourly windows per day)**:
    - Read the source data into `RawWindow` — endpoints at `t_n` and
      `t_{n+1}`, plus window-integrated mass fluxes.
    - Apply the source-specific physics conventions (level orientation,
      dry-basis conversion, `mass_flux_dt = 450 s` for FV3).
    - Regrid (or pass through identically) to the target mesh.
    - Compute the vertical mass flux `cm` — by **Poisson balance** for
      synthesized winds (spectral path), by the **FV3 pressure-fixer**
      formula for native FV3 mass fluxes (GEOS path).
    - Run the **write-time replay gate**:
      `‖m_evolved − m_stored_{n+1}‖ / ‖m_stored_{n+1}‖ ≤ tol`.
      Failures abort the run rather than producing a known-bad file.
    - Stream the window to the binary.
4. **Cross-day chaining** — the day's final mass endpoint is threaded
   into the next day's `seed_m` so window boundaries close.

## Where to read next

- [ERA5 spectral path](@ref) — the LL / RG / CS spectral pipeline
  (Holton synthesis, vertical merging, mass-fix, Poisson balance).
- [GEOS native cubed-sphere](@ref) — the GEOS-IT / GEOS-FP CS
  passthrough (FV3 pressure-fixer cm, dry-basis conversion, GCHP
  convection wiring, `mass_flux_dt = 450`).
- [Regridding](@ref) — conservative weights, IdentityRegrid, cache
  key, mass-consistency correction.
- [Conventions cheat sheet](@ref) — units, replay tolerances, level
  orientation, panel conventions — the one-page reference for
  debugging an unexpected binary.
