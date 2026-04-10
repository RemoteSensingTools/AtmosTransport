# ERA5 Lat-Lon Reference Path for `src_v2`

Date: 2026-04-09

This note defines the first polished `src_v2` reference path.

## Scope

This path is intentionally narrow:

- grid: ERA5-derived structured lat-lon
- runtime: `src_v2` only
- operators: `UpwindScheme`, `SlopesScheme`
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

Recommended delta sections for explicit forward-window provenance:

- `dam`
- `dbm`
- `dcm`
- `dm`

The stored runtime contract for the current ERA5 path is:

- `air_mass_sampling = "window_start_endpoint"`
- `flux_sampling = "window_constant"`
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

The lat-lon preprocessor also records the Poisson-balance normalization used to make the stored horizontal fluxes consistent with the Strang-split mass path:

- `poisson_balance_target_scale = 1 / (2 * steps_per_window)`
- this means the balance target is `(m_next - m_curr) / (2 * steps_per_window)`
- the factor of `2` comes from the repeated horizontal sweeps inside `X Y Z Z Y X`

For the current reference path:

- each met window contains `steps_per_window` transport substeps
- `am`, `bm`, and `cm` stay constant within the window
- endpoint humidity diagnostics may still use `qv_start/qv_end`
- `dam`, `dbm`, `dcm`, and `dm` remain in the binary as explicit forward-window
  deltas and provenance, but the current lat-lon runtime does not interpolate
  fluxes with them

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
   - prepares forcing into the model flux workspace
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
- standalone `src_v2` upwind and slopes stepping on structured lat-lon
- standalone `DrivenSimulation` multi-window runtime with driver-controlled
  forcing semantics
- verified 2-day real-data Dec 2021 pure-advection runs for both `UpwindScheme`
  and `SlopesScheme` on the current reference binaries

## What Is Not Implemented Yet

Not implemented yet:

- a polished CLI for real-data lat-lon runs
- higher-order structured advection on this path as a production-ready option
- reduced-Gaussian preprocessing through the same native path
- cubed-sphere runtime support in `src_v2`

## Minimal Usage Sketch

Preprocess one day:

```bash
julia -t24 --project=. scripts/preprocessing/preprocess_era5_latlon_transport_binary_v2.jl \
  config/preprocessing/era5_latlon_transport_binary_v2.toml --day 2021-12-01
```

Run the standalone `src_v2` reference path:

```bash
julia --project=. scripts/run_transport_binary_v2.jl path/to/run_transport_v2_upwind.toml
```

Current status: the binary, driver, and multi-window runtime layers are now in
place for this path and the first 2-day Dec 2021 reference runs are complete.
The next step is to carry the same validated forcing contract into the native
ReducedGaussian path.
