# Coming from TM5

Quick reference for TM5 users switching to AtmosTransport.jl. This covers
concept mapping, workflow differences, and where to find familiar operations.

## Concept mapping

| TM5 (Fortran) | AtmosTransport.jl | Notes |
|---|---|---|
| `advectx.F90` / `advecty.F90` | `src/Operators/Advection/StrangSplitting.jl` | Same Strang X-Y-Z-Z-Y-X split |
| Russell-Lerner slopes | `SlopesScheme(MonotoneLimiter())` | Identical algorithm, 3-arg minmod |
| `advectm_cfl.F90` / `nloop` | `_x_subcycling_pass_count` | CFL evolving-mass pilot |
| `dynam0` / `dynamw_1d` | `diagnose_cm!` / vertical remap | cm from continuity equation |
| `grid_type_ll.F90` Poisson | `balance_mass_fluxes!` / `LLPoissonWorkspace` | Identical FFT algorithm |
| `sp = exp(lnsp)` | `spectral_synthesis.jl` | Spectral → gridpoint via Legendre + FFT |
| `Match('area-aver', ...)` | `pin_global_mean_ps!` | Global mean ps mass fix |
| `echlev` / level merging | `select_levels_echlevs` / `merge_thin_levels` | Same interface-index scheme |
| `tm5_massflux.bin` | Transport binary (v4) | Different format (JSON header + flat F32/F64) |
| `mk.F90` (mass update) | `strang_split!` tracer loop | Air mass co-evolved with tracers |
| Tiedtke convection | `TiedtkeConvection` | Same algorithm, explicit upwind |
| PBL diffusion | `BoundaryLayerDiffusion` | Implicit tridiagonal solve |

## Key architectural differences

### No global arrays
TM5 uses global `m(im,jm,lm)` and `rm(im,jm,lm,ntracerp)`. AtmosTransport
passes explicit `CellState` + `FaceFluxState` objects through dispatch.

### Config-driven, no recompilation
TM5 builds use `#ifdef` and Makefile options. AtmosTransport uses TOML configs:
```bash
julia --project=. scripts/run.jl config/runs/era5_f64_debug.toml
```

### Grid dispatch (not if/else)
TM5 has separate code paths for each grid. AtmosTransport dispatches on
`AbstractGrid` subtypes: `LatLonMesh`, `ReducedGaussianMesh`, `CubedSphereMesh`.
The same `strang_split!` interface works on all.

### GPU-portable kernels
All advection kernels use KernelAbstractions.jl — the same code runs on CPU and
GPU. No separate GPU code to maintain.

## Preprocessing workflow

### TM5 workflow
```
1. Download ERA5 spectral from MARS
2. Run mk_massflux to build transport binary
3. Run TM5 with the binary
```

### AtmosTransport workflow
```
1. Download ERA5 spectral (CDS or MARS): scripts/downloads/download_data.jl
2. Preprocess:
   julia scripts/preprocessing/preprocess_transport_binary.jl config.toml --day 2021-12-01
3. Run transport:
   julia scripts/run.jl config/runs/my_config.toml
```

### Preprocessing targets
| Target | Config `[grid] type` | TM5 equivalent |
|--------|---------------------|----------------|
| Regular lat-lon | `"latlon"` | TM5 standard grid |
| Reduced Gaussian | `"reduced_gaussian"` | Not in TM5 |
| Cubed sphere | `"cubed_sphere"` | Not in TM5 (GCHP) |
| LL binary to CS | `regrid_ll_binary_to_cs()` | Not in TM5 |

## Advection schemes

| Scheme | Config `scheme =` | TM5 equivalent | Accuracy |
|--------|------------------|----------------|----------|
| `"upwind"` | `UpwindScheme()` | `slopes=false` | 1st order |
| `"slopes"` | `SlopesScheme()` | Default TM5 (Russell-Lerner) | 2nd order |
| `"ppm"` | `PPMScheme()` | Not in standard TM5 | 3rd order |

## File locations

| What | TM5 | AtmosTransport |
|------|-----|----------------|
| Advection | `advectx.F90`, `advecty.F90`, `advectz.F90` | `src/Operators/Advection/` |
| Spectral synthesis | `spectral.F90` | `src/Preprocessing/spectral_synthesis.jl` |
| Poisson balance | `grid_type_ll.F90:2536-2653` | `src/Preprocessing/mass_support.jl` |
| Level merging | `echlev.F90` | `src/Preprocessing/vertical_coordinates.jl` |
| Transport binary | `mk_massflux.F90` | `src/Preprocessing/binary_pipeline.jl` |
| Convection | `tiedtke.F90` | `src/Operators/Convection/` |
| Diffusion | `diffusion.F90` | `src/Operators/Diffusion/` |
| Main run loop | `mainloop.F90` | `src/Models/run_loop.jl` |

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for end-to-end setup.

For TM5-equivalent ERA5 runs:
1. Use `config/preprocessing/era5_ll_96x48_transport_binary.toml` (3°×2° like TM5 standard)
2. Or `config/preprocessing/era5_ll_360x181_transport_binary.toml` (1°×1°)
3. Set `scheme = "slopes"` for TM5-equivalent advection
4. Enable `[diffusion]` and `[convection]` for realistic column behavior
