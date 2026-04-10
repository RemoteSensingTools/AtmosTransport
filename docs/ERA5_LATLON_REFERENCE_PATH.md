# ERA5 Lat-Lon Reference Path for `src_v2`

Date: 2026-04-09

This note defines the first polished `src_v2` reference path.

## Scope

This path is intentionally narrow:

- grid: ERA5-derived structured lat-lon
- runtime: `src_v2` only
- operator: `UpwindAdvection`
- physics: pure advection only
- basis: explicit dry or moist, with moist as the default reference path

This is the path that future reduced-Gaussian and cubed-sphere adapters should
match at the interface level.

## Core Runtime Contract

The runtime advances:

- prognostic air mass `m`
- prognostic tracer masses `rm`
- face mass fluxes supplied by the met driver

The runtime does not diagnose vertical closure inside advection kernels.
If time interpolation is used, the binary or driver contract should carry the
full forcing, including vertical-flux variation.

## Binary Sections

The structured ERA5 lat-lon transport binary uses the topology-generic binary
family with:

- `grid_type = "latlon"`
- `horizontal_topology = "StructuredDirectional"`
- `mass_basis = "moist"` or `"dry"`

Required per-window sections:

- `m`
- `am`
- `bm`
- `cm`
- `ps`

Recommended humidity sections for moist transport that may later write dry VMR:

- `qv_start`
- `qv_end`

Recommended delta sections for time-interpolated forcing:

- `dam`
- `dbm`
- `dcm`
- `dm`

The design intent is explicit: if the runtime interpolates fluxes through the
window, `dcm` should be present so vertical forcing is not reconstructed by a
closure routine during stepping. The stored runtime contract for this ERA5 path
is:

- `air_mass_sampling = "window_start_endpoint"`
- `flux_sampling = "window_start_endpoint"`
- `flux_kind = "substep_mass_amount"`
- `delta_semantics = "forward_window_endpoint_difference"`

If the raw met source uses interval means or interval-integrated fluxes, the
preprocessor must normalize them into this stored contract before writing the
binary.

## Timing Semantics

For the ERA5 spectral transport preprocessor:

- `source_flux_sampling` must be written explicitly as one of `window_start_endpoint`, `window_end_endpoint`, `window_mean`, or `interval_integrated`
- `qv_start` is the humidity field aligned with the current window
- `qv_end` is the next humidity field in time
- for the last window of the day, `qv_end` comes from next-day 00 UTC when
  available

For dry-air diagnostics from moist transport:

- use runtime end-state moist mass together with `qv_end`
- do not reuse stale `qv_start` for output conversion

For runtime interpolation of delta payloads in `DrivenSimulation`:

- each met window contains `steps_per_window` transport substeps
- forcing is sampled at the substep midpoint
- the interpolation fraction is `λ = (s - 0.5) / steps_per_window`
- the same convention is applied to `am`, `bm`, `cm`, `m`, and optional
  humidity diagnostics carried in the current window state

## `src_v2` Driver Path

The clean met-driver seam is now:

1. `TransportBinaryReader`
   - parses the generic binary family
   - reconstructs the grid
   - loads per-window state/flux payloads
2. `TransportBinaryDriver`
   - owns file/window timing
   - returns typed transport windows
   - exposes endpoint humidity and optional flux deltas
3. `DrivenSimulation`
   - owns window/substep orchestration
   - interpolates forcing into the model flux workspace
   - keeps tracer and air-mass prognostics inside the model
4. `TransportModel`
   - owns prognostic air mass and tracer masses
   - advances one substep via `step!(model, Δt)`

The driver does not own tracer state. It owns forcing and timing.
The model owns prognostic transport state.

## What Is Implemented

Implemented now:

- basis-explicit `CellState` and `FluxState`
- generic transport-binary reader/writer for structured lat-lon
- typed `TransportBinaryDriver`
- typed `StructuredTransportWindow`
- endpoint humidity loading
- optional full flux-delta loading, including `dcm`
- standalone `src_v2` upwind stepping on structured lat-lon
- standalone `DrivenSimulation` multi-window runtime with midpoint forcing
  interpolation

## What Is Not Implemented Yet

Not implemented yet:

- a polished CLI for real-data lat-lon runs
- higher-order structured advection on this path as a production-ready option
- reduced-Gaussian preprocessing through the same native path
- cubed-sphere runtime support in `src_v2`
- threaded payload packing/writing in the preprocessor

## Minimal Usage Sketch

Preprocess one day:

```bash
julia --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
  config/preprocessing/era5_spectral_v4.toml --day 2021-12-01
```

Run two windows through the standalone `src_v2` runtime:

```julia
include("src_v2/AtmosTransportV2.jl")
using .AtmosTransportV2

path = "path/to/era5_transport_20211201_merged33Pa_float64.bin"
driver = TransportBinaryDriver(path; FT=Float64)
grid = driver_grid(driver)

window1 = load_transport_window(driver, 1)
state = CellState(MoistBasis, copy(window1.air_mass); CO2 = copy(window1.air_mass) .* 400e-6)
fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid); FT=Float64, basis=MoistBasis)
model = TransportModel(state, fluxes, grid, UpwindAdvection())
sim = DrivenSimulation(model, driver; start_window=1, stop_window=2)

run!(sim)
```

Current status: the binary, driver, and multi-window runtime layers are now in
place for this path. The next step is a documented real-data Dec 2021 smoke run
through the same contract.
