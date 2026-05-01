# TOML schema

This page is the **canonical reference** for the TOML configs that
drive both the runtime (`scripts/run_transport.jl`) and the
preprocessor (`scripts/preprocessing/preprocess_transport_binary.jl`).
Per-block, per-key, with type, default, and what it does.

The two config families (run vs. preprocessing) live in different
directories and are consumed by different code paths; this page
keeps them separate.

## Run config (`config/runs/*.toml`)

Consumed by `scripts/run_transport.jl` (line 74) which parses the
TOML and forwards the resulting Julia `Dict` to
`run_driven_simulation(cfg)` in `src/Models/DrivenRunner.jl`.

The runtime infers the **target topology** from the binary header's
`grid_type` field at load time (`DrivenRunner.jl:293-298`); a
`[grid]` block in the run config is therefore unnecessary and
ignored.

### `[input]` — which transport binaries to load

Two valid shapes (mutually exclusive):

```toml
# Shape A — explicit list
[input]
binary_paths = [
    "~/data/.../era5_transport_20211201.bin",
    "~/data/.../era5_transport_20211202.bin",
]

# Shape B — folder + date range
[input]
folder       = "~/data/.../era5_ll72x37_dec2021_f32/"
start_date   = "2021-12-01"
end_date     = "2021-12-03"
file_pattern = "{YYYYMMDD}"   # optional; default scans for date stamp
```

Path expansion + continuity validation lives in
`src/Models/BinaryPathExpander.jl::expand_binary_paths` (line 62).
Shape B asserts that the resolved binaries form a contiguous date
sequence; gaps fail at expansion time, not at first window-load.

### `[architecture]` — backend selection

```toml
[architecture]
use_gpu = true                # default: false
backend = "auto"              # default: "auto" if use_gpu else "cpu"
```

| `backend` | Effect |
|---|---|
| `"cpu"` | CPU only. **Conflicts with `use_gpu = true` and errors at config-load time** (see `src/Architectures.jl:120-121`). |
| `"cuda"` | NVIDIA CUDA via `CUDA.jl` (must be installed). |
| `"metal"` | Apple Silicon Metal via `Metal.jl`. F32 only. |
| `"auto"` | Auto-detects an available GPU backend (CUDA → Metal); errors if none is available. |
| (omitted) | If `backend` is absent, the runtime picks `CPUBackend` when `use_gpu = false` and auto-detects a GPU backend when `use_gpu = true`. |

GPU preload happens at TOML-parse time
(`scripts/run_transport.jl:37-63`) so the CUDA / Metal extension
loads before AtmosTransport — this avoids a Julia world-age issue
that bit early users.

### `[numerics]` — precision

```toml
[numerics]
float_type = "Float32"        # default: "Float64"; one of "Float32" / "Float64"
```

Float32 is the recommended default for L40S / consumer-GPU
production runs; F64 needs an A100-class card or CPU. Mixing
with the binary's `on_disk_float_type` is allowed (the runtime
casts on load).

### `[run]` — runtime knobs

```toml
[run]
start_window = 1              # default: 1 — first window to process
stop_window  = 24             # default: nothing — uses the binary's full range
scheme       = "slopes"       # advection scheme; one of "upwind" / "slopes" / "ppm" / "linrood"
```

`stop_window` is the inclusive last window; setting it lets you
run a partial day for smoke tests.

### `[tracers.<name>]` — per-tracer setup

Each tracer gets its own block. The name is what shows up in the
output NetCDF; `[tracers.co2_bl]` writes `co2_bl`,
`co2_bl_column_mean`, etc.

```toml
[tracers.co2_bl]
species = "co2"               # optional; controls molar-mass conversions for diagnostics

[tracers.co2_bl.init]
kind        = "bl_enhanced"   # initial-condition kind; see table below
background  = 4.0e-4          # uniform background dry VMR (mol/mol)
enhancement = 1.0e-4          # extra dry VMR in lowest n_layers (LL only)
n_layers    = 3
```

Initial-condition kinds (declared in `src/Models/InitialConditionIO.jl`):

| Kind | LL | RG | CS | Required keys |
|---|---|---|---|---|
| `"uniform"` | yes | yes | yes | `background` |
| `"bl_enhanced"` | yes | **no** | **no** | `background`, `enhancement`, `n_layers` (LL-only; RG/CS path errors at IC build) |
| `"gaussian_blob"` | yes | yes | **no** | `background`, `lon0_deg`, `lat0_deg`, `sigma_lon_deg`, `sigma_lat_deg`, `amplitude` |
| `"file"` / `"netcdf"` | yes | yes | yes | `file`, `variable`, optional `time_index` |
| `"file_field"` | yes | yes | yes | `file`, `variable` |
| `"catrine_co2"` | yes | yes | yes | `file`, `variable`, optional `time_index` |

Surface-flux emission is configured under the same tracer block via
`surface_flux_*` keys (read at `DrivenRunner.jl:109-114`):

```toml
[tracers.co2_bl]
surface_flux_kind     = "edgar_co2"
surface_flux_file     = "~/data/.../EDGAR_v8.0_CO2.nc"
surface_flux_variable = "emi_co2"
surface_flux_time_index = 1
surface_flux_month    = 12        # for inventories indexed by month
surface_flux_scale    = 1.0
```

### `[advection]`, `[diffusion]`, `[convection]`

Each operator has a `kind` selector + per-kind kwargs. See
[Operators](@ref) and [Advection schemes](@ref) for what each
selector means; relevant config keys:

```toml
[advection]
scheme    = "ppm"
ppm_order = 7                  # cubed-sphere LinRoodPPM only; ∈ {5, 7}

[diffusion]
kind  = "constant"             # "none" | "constant"
value = 1.0                    # m²/s — broadcast Kz when kind="constant"

[convection]
kind = "cmfmc"                 # "none" | "cmfmc" | "tm5"

# TM5-only — per-topology budget for the column-tile workspace,
# in binary GiB. Default 1.0 (fits production through C720/L137 on
# H100). Set lower on memory-tight GPUs; setting it higher beyond
# the topology's total cells is a no-op.
tile_workspace_gib = 1.0
```

The runtime **rejects** at load time any operator selection that
the binary doesn't support (e.g. `convection.kind = "cmfmc"`
against a binary lacking `:cmfmc` payload). See
[Binary format](@ref Binary-format) for the capability surface.

### `[output]` — snapshot NetCDF

```toml
[output]
snapshot_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
snapshot_file  = "~/data/.../my_run.nc"
deflate_level  = 0             # NetCDF4 deflate (0..9); 0 = no compression
shuffle        = true          # shuffle filter; only effective when deflate>0
```

A frame lands when `|t_simulation_hours − snapshot_hours[k]| < 0.5`
(`DrivenRunner.jl:398`); the float-tolerance window absorbs
sub-step rounding. See [Output schema](@ref) for the per-topology
variable list the file actually contains.

### Multi-threaded execution

```bash
julia --threads=2 --project=. scripts/run_transport.jl <cfg.toml>
```

Some preprocessing kernels (spectral synthesis, regridding) and
some host-side workspace operations parallelize across threads.
The runtime window-load itself is **synchronous** today
(`DrivenSimulation.jl::step!` blocks on each `read_window!` rather
than overlapping with compute on window `N − 1`); there is no
`[buffering]` TOML block.

## Preprocessing config (`config/preprocessing/*.toml`)

Consumed by `scripts/preprocessing/preprocess_transport_binary.jl`
which calls `process_day(cfg::Dict)` in
`src/Preprocessing/transport_binary/entrypoint.jl` (line 204).

The preprocessing config has a different shape from the run config:
the **target topology IS specified here** because that's the act of
producing a binary for that topology.

### `[input]` (legacy spectral) or `[source]` (native)

The preprocessor source-axis dispatch reads either:

```toml
# ERA5 spectral (legacy path)
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"
```

```toml
# GEOS-IT native (current path)
[source]
toml     = "config/met_sources/geosit.toml"     # source descriptor
root_dir = "~/data/AtmosTransport/met/geosit/C180/raw_catrine"
```

For the GEOS path, `[source].toml` points to the **source
descriptor** (a separate TOML in `config/met_sources/`) that
declares collection mappings, `mass_flux_dt_seconds`, and
`level_orientation`. See [GEOS native cubed-sphere](@ref) for the
descriptor schema details.

### `[output]`

```toml
[output]
directory  = "~/data/AtmosTransport/met/.../preprocessed/"
mass_basis = "dry"             # default: "dry"; "moist" supported but not recommended for the runtime
include_qv = false             # LL spectral path only — controls writing the :qv payload.
                               # Native GEOS and CS/RG writers ignore this key today.
```

### `[grid]` — target topology

```toml
# Lat-lon
[grid]
type   = "latlon"
nlon   = 144
nlat   = 73
echlevs              = "ml137_tropo34"
level_top            = 1
level_bot            = 137
merge_min_thickness_Pa = 1000.0

# Cubed sphere
[grid]
type                = "cubed_sphere"
Nc                  = 180
panel_convention    = "geos_native"             # or "gnomonic"
definition          = "gmao"                    # optional; inferred from convention if omitted
regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"

# Reduced Gaussian (synthetic — picks a standard ECMWF reduced-Gaussian grid)
[grid]
type            = "synthetic_reduced_gaussian"
gaussian_number = 90
nlon_mode       = "octahedral"                  # ECMWF O-grid distribution
```

### `[vertical]`

```toml
[vertical]
coefficients = "config/geos_L72_coefficients.toml"
```

Per-source defaults are baked into the source-descriptor TOML; this
key is the per-run override.

### `[numerics]`

The numerics block has **different keys** on the spectral and native
paths:

```toml
# Spectral (legacy ERA5) preprocessing
[numerics]
float_type   = "Float32"     # "Float32" or "Float64"
dt           = 900.0         # advection sub-step (s)
met_interval = 3600.0        # window cadence (s); 1 hour for ERA5
cs_balance_tol = 1e-14       # CS Poisson balance tolerance
cs_balance_project_every = 50 # CS PCG mean-zero projection cadence; 1 = legacy
```

```toml
# Native (GEOS-IT) preprocessing
[numerics]
float_type     = "Float32"
dt_met_seconds = 3600.0      # window cadence (s); 1 hour for GEOS-IT
```

`mass_flux_dt` for the GEOS path lives in the **source descriptor's**
`[preprocessing].mass_flux_dt_seconds` (`config/met_sources/geosit.toml`,
default `450.0` — the FV3 dynamics step); there is **no per-run
`[numerics].mass_flux_dt` override** today.

### `[mass_fix]` — global PS pinning (spectral path only)

```toml
[mass_fix]
enable                = true
target_ps_dry_pa      = 98726.0
qv_global_climatology = 0.00247
```

The GEOS native CS path doesn't apply mass fix (the FV3 dynamical
core's mass flux is already conservative). LL spectral runs without
it drift by tens of Pa per window.

## Where to read next

- [Output schema](@ref) — what the snapshot NetCDF actually contains
  per topology.
- [Data sources](@ref) — ERA5 / GEOS access, credentials, recommended
  local layout, and the quickstart bundle.
- [First run](@ref) — the smallest end-to-end invocation pattern.
