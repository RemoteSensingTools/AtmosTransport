# First run

This page covers the **general** runtime invocation pattern — the TOML
config schema, where met data comes from, and how runs flow through
`scripts/run_transport.jl`.

!!! tip "Just want to try it?"
    For the fastest path from a fresh clone to a real simulation +
    NetCDF + plot, see [Quickstart with example data](@ref). It ships a
    250 MB downloadable bundle of preprocessed transport binaries and
    four ready-to-run configs (LL at two resolutions + CS at two
    resolutions, all F32, 3 days of December 2021 ERA5).

A general run needs (1) a TOML config, (2) preprocessed met binaries
(produced once by the preprocessor), and optionally (3) emissions /
initial-condition files. The quickstart bundle gives you (1) and (2);
this page documents the pattern so you can swap in your own.

## The runtime invocation

There is one runtime entrypoint:

```bash
julia --project=. scripts/run_transport.jl <config.toml>
```

The script reads the TOML, opens the first transport binary referenced by
`[input]`, dispatches on the binary's `grid_type` header
(`:latlon`, `:reduced_gaussian`, `:cubed_sphere`), and runs the simulation
loop. All topology-specific behavior — initial-condition pipeline,
surface fluxes, snapshot output, GPU-residency assertions, capability
validation — lives in `src/Models/DrivenRunner.jl` and dispatches on the
mesh type via Julia's multiple dispatch.

For double-buffered I/O overlap on GPU runs:

```bash
julia --threads=2 --project=. scripts/run_transport.jl <config.toml>
```

## A worked TOML walkthrough

`config/runs/quickstart/ll72x37_advonly.toml` (the smallest of the four
quickstart configs) is a concise 3-day advection-only run on a 5°
lat-lon grid with a boundary-layer-enhanced CO₂ initial condition. The
schema is the canonical one consumed by `scripts/run_transport.jl` via
`expand_binary_paths`:

```toml
[input]
folder     = "~/data/AtmosTransport_quickstart/met/era5_ll72x37_dec2021_f32/"
start_date = "2021-12-01"
end_date   = "2021-12-03"
# Alternative: explicit list
#   binary_paths = ["~/.../day1.bin", "~/.../day2.bin", …]

[architecture]
use_gpu = true               # false for CPU

[numerics]
float_type = "Float32"       # "Float64" on CUDA / CPU debug

[run]
scheme = "slopes"            # "slopes" (Russell-Lerner) or "ppm" (Putman-Lin)

[tracers.co2_bl]
[tracers.co2_bl.init]
kind        = "bl_enhanced"  # uniform | bl_enhanced | gaussian_blob | file | netcdf
background  = 4.0e-4         # ~400 ppm dry VMR (uniform background)
enhancement = 1.0e-4         # +100 ppm in lowest n_layers (LL only)
n_layers    = 3              # bottom-3-layer enhancement

[output]
snapshot_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
snapshot_file  = "~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc"
```

Things to know:

- **`[input]`** accepts either a `folder` + `start_date` / `end_date` pair
  (the runtime expands to one binary per day under that folder) or an
  explicit `binary_paths = […]` list. See `BinaryPathExpander.jl`.
- **Topology** is auto-detected from the binary's `grid_type` header
  field. The runtime then dispatches `DrivenRunner` on the mesh type;
  no explicit `[grid]` block is needed for a run.
- **`[run] scheme`** picks the advection scheme. For PPM, add
  `ppm_order = 7` (orders 4-7).
- **`[tracers.<name>.init]`** declares the initial condition. Cubed-sphere
  supports `uniform | file | netcdf | file_field | catrine_co2`. Lat-lon
  additionally supports `bl_enhanced | gaussian_blob`.
- **`[output]`** writes a single NetCDF at `snapshot_file` containing per-
  tracer `<name>_column_mean` and `<name>_column_mass_per_area` plus the
  `air_mass_column` (dimensions vary with topology — see
  [Inspecting output](@ref)).

## Where the met data comes from

The runtime consumes **v4 transport binaries** — a self-describing flat
format produced by the preprocessor. There are two preprocessing paths
today:

1. **ERA5 spectral**. Reads vorticity, divergence, and log-PS GRIB files;
   synthesizes mass fluxes via Holton's continuity-consistent approach;
   pins global mean PS for mass closure. Configs in
   `config/preprocessing/era5_*.toml`.

2. **GEOS native cubed-sphere**. Reads GEOS-IT C180 NetCDF
   (CTM_A1 hourly, CTM_I1 instantaneous, optionally A3mstE / A3dyn for
   convection); applies dry-basis conversion, FV3 pressure-fixer cm, and
   chained cross-day mass continuity. Configs in
   `config/preprocessing/geosit_*.toml`.

CLI:

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --day YYYY-MM-DD
# or
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --start YYYY-MM-DD --end YYYY-MM-DD
```

The preprocessor writes one binary per day to the path declared in the
config's `[output] directory`. A subsequent run config points at that
directory via `[met_data] preprocessed_dir`.

The detailed preprocessing guide lands in Phase 5 of the documentation
overhaul.

## Synthetic-fixture route (no external data)

The following test files build complete v4 binaries from synthetic fixtures
and exercise the runtime end-to-end. They are the most accurate "minimal
working example" today:

| File | What it covers |
|---|---|
| `test/test_geos_cs_passthrough.jl` | GEOS native C8 fixture → CS passthrough preprocessor → write-time replay gate (3467 cases). |
| `test/test_geos_convection.jl` | The same fixture extended with synthetic A3mstE/A3dyn → convection forcing → binary payload sections. |
| `test/test_driven_simulation.jl` | Synthetic LL binary loaded by `DrivenRunner` and stepped forward. |

To run just one of them without the full suite:

```bash
julia --project=. test/test_driven_simulation.jl
```

Phase 4 of the documentation overhaul will package one of these synthetic
flows as a Literate.jl tutorial that produces both a runnable script and
a rendered HTML page in the docs.

## What you should see

A successful run prints, roughly (output trimmed from a real
`config/runs/quickstart/ll72x37_advonly.toml` invocation):

```text
[ Info: Preloading CUDA (GPU backend)
[ Info: [gpu verified] backend=cuda backing=CuArray device=NVIDIA L40S
[ Info: Backend: GPU (CUDA, NVIDIA L40S)
[ Info: Physics: advection=SlopesScheme diffusion=NoDiffusion convection=NoConvection
[ Info: Snapshot 1 at t=0h
[ Info: Running era5_transport_20211201_merged1000Pa_float32.bin with SlopesScheme on 72×37 LatLonMesh{Float32} (24 windows)
…
[ Info: Saved snapshots: ~/data/AtmosTransport_quickstart/output/ll72x37_advonly.nc (13 frame(s), 72×37 LatLonMesh{Float32}, mass_basis=dry)
[ Info: Final air-mass change vs initial state:  -1.071e-07
[ Info: Final tracer-mass drift for co2_bl:         0.000e+00
```

Two diagnostic lines worth watching for:

- `[gpu verified]` — the runtime asserts that `state.air_mass` lives on the
  selected GPU backend when `use_gpu = true`. If you see CPU residency
  despite the flag, the dispatch chain is mis-wired.
- The closing `Final tracer-mass drift` line should be exactly `0.000e+00`
  for advection-only runs (the mass-fixer is on by default and any drift
  beyond F32 noise indicates a regression).

## What's next

- [Inspecting output](@ref) — verify the snapshot NetCDF and the input
  binary.
- [Configuration & Runtime](#) — the full TOML schema (Phase 7).
- [Tutorials](#) — Literate-driven topology examples (Phase 4).
