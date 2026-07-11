```@meta
CurrentModule = AtmosTransport
```

# AtmosTransport.jl

![AtmosTransport.jl](assets/brand/AtmosTransport_banner.png)

A Julia-based, GPU-portable atmospheric tracer transport model for offline
chemistry / chemical-transport applications. Designed for **mass-conserving**
advection, convection, and boundary-layer diffusion on **lat-lon, reduced
Gaussian, and cubed-sphere** grids, driven by **ERA5** or **GEOS** met data,
with a clean separation between offline preprocessing and runtime stepping.

## At a glance

- **Multi-grid**: regular lat-lon, reduced Gaussian, cubed-sphere (gnomonic
  and GEOS-native panel conventions).
- **Multi-source**: ERA5 spectral (vorticity / divergence / log-PS GRIB),
  GEOS-IT C180 native NetCDF, GEOS-FP C720 native hourly NetCDF, and a
  preview MERRA-2 wind-derived cubed-sphere preprocessor.
- **GPU-portable**: single codebase for CPU, NVIDIA CUDA, and Apple Silicon
  Metal via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
  Metal is restricted to `Float32` runtime numerics.
- **Mass-conserving**: dry-basis air-mass bookkeeping, with **write-time
  replay gates** in the preprocessor and **opt-in load-time replay
  validation** at runtime.
- **Operator-modular**: every physics operator is behind an abstract type
  with a `No<Operator>` no-op default; swap schemes via type dispatch.
- **TM5-faithful core**: Russell-Lerner slopes advection with Strang
  splitting; CMFMC convection (GCHP-style for GEOS sources) and TM5
  convection (entrainment / detrainment for ERA5 sources) sharing one
  runtime carrier.

## When to use AtmosTransport

- You have offline meteorological fields (winds, mass fluxes, surface pressure,
  optionally moist physics) and want to integrate one or more passive or
  reactive trace gases at coarse-to-medium resolution.
- You need GPU performance with bit-reproducible CPU fallback.
- You want a model where the mass-conservation contract is explicit at every
  layer (preprocessor output ↔ runtime state ↔ snapshot output).

If you need a fully online dynamical core (LES, GCM), look elsewhere —
AtmosTransport assumes a precomputed mass-flux time series.

## Where to start

The recommended reading order is:

1. [Installation](getting_started/installation.md) and
   [Quickstart](getting_started/quickstart.md).
2. [Concepts](concepts/grids.md) — grids, state, operators, and binaries.
3. [Preprocessing](preprocessing/overview.md) — raw meteorology to transport binaries.
4. [Theory and verification](theory/mass_conservation.md).
5. [API reference](api/index.md).

The most useful repository entry points are:

- `scripts/run_transport.jl` — runtime driver script.
- `scripts/preprocessing/preprocess_transport_binary.jl` — preprocessing CLI.
- `scripts/diagnostics/inspect_transport_binary.jl` — inspect a transport binary.
- `config/runs/` — example run configurations (TOML).
- The repository `README.md` and `docs/reference/` pages carry the current
  status, invariants, and project map.
