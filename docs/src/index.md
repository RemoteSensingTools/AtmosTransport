```@meta
CurrentModule = AtmosTransport
```

# AtmosTransport.jl

A Julia-based, GPU-portable atmospheric tracer transport model for offline
chemistry / chemical-transport applications. Designed for **mass-conserving**
advection, convection, and boundary-layer diffusion on **lat-lon, reduced
Gaussian, and cubed-sphere** grids, driven by **ERA5** or **GEOS** met data,
with a clean separation between offline preprocessing and runtime stepping.

!!! warning "Work in Progress"
    This project is under rapid active development. APIs, file formats, and
    physics implementations may change without notice. The documentation is
    being overhauled; see [About these docs](@ref) for the current status.

## At a glance

- **Multi-grid**: regular lat-lon, reduced Gaussian, cubed-sphere (gnomonic
  and GEOS-native panel conventions).
- **Multi-source**: ERA5 spectral (vorticity / divergence / log-PS GRIB)
  and GEOS-IT C180 native NetCDF. (GEOS-FP native is a planned follow-up;
  the source-axis abstraction is in place.)
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

Once the rest of these docs land, the recommended reading order is:

1. **[Getting Started](#)** — install, run a tiny example, look at output.
2. **[Concepts](#)** — grids, state, operators, what the binary contains.
3. **[Tutorials](#)** — end-to-end runnable examples per grid topology.
4. **[Theory & Verification](#)** — mass conservation derivation, advection
   schemes, validation results.
5. **[Preprocessing](#)** — turning raw met data into a v4 transport binary.
6. **[API Reference](#)** — full function/type index.

In the meantime, the most useful entry points in the repository are:

- `scripts/run_transport.jl` — runtime driver script.
- `scripts/preprocessing/preprocess_transport_binary.jl` — preprocessing CLI.
- `scripts/diagnostics/inspect_transport_binary.jl` — inspect a v4 binary.
- `config/runs/` — example run configurations (TOML).
- `CLAUDE.md` (root) — high-signal invariants and project map.
