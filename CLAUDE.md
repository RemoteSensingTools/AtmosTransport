# AtmosTransport — Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Solves tracer
continuity on latitude-longitude (ERA5) and cubed-sphere (GEOS-FP C720, GEOS-IT C180)
grids with mass-conserving advection, Tiedtke convection, and boundary-layer diffusion.
Oceananigans-inspired modular architecture; KernelAbstractions.jl for GPU portability.

## Running simulations

```bash
julia --project=. scripts/run.jl config/runs/<config>.toml
```

All simulation parameters live in TOML configs under `config/runs/`. Preprocessing
configs are in `config/preprocessing/`. See `docs/QUICKSTART.md` for full reference.

## Source layout

`src/AtmosTransport.jl` includes modules in strict dependency order:
Architectures → Grids → Fields → Advection → Convection → Diffusion → Chemistry →
TimeSteppers → Adjoint → Callbacks → Regridding → Diagnostics → Sources → IO →
Visualization → Models

Key directories:
- `scripts/` — download, preprocess, run, and diagnostic scripts
- `config/runs/` — simulation configs; `config/preprocessing/` — preprocessor configs
- `config/met_sources/` — per-source variable mappings to canonical names
- `config/canonical_variables.toml` — the contract between met sources and physics code

## Code philosophy

- **TOML-driven**: all parameters in config files, no hardcoded paths in source
- **Canonical variables**: physics code uses abstract names (e.g. `surface_pressure`);
  met source TOMLs map native variable names to these canonicals
- **GPU via extension**: `using CUDA` or `using Metal` before `using AtmosTransport` loads
  the appropriate extension (`AtmosTransportCUDAExt` or `AtmosTransportMetalExt`).
  `scripts/run.jl` auto-detects: Metal on macOS, CUDA on Linux/Windows.
- **KernelAbstractions**: all GPU kernels use `@kernel`/`@index` for CUDA/Metal/CPU portability.
  CUDA and Metal are supported. AMDGPU is a future goal.
  One atomic operation in `regridding_diagnostics.jl` (`@atomic` scatter) may need a CPU
  fallback on Metal if Float32 atomics are unsupported.
- **Mass conservation first**: advection via mass fluxes (not winds); vertical flux from
  continuity equation; Strang splitting X→Y→Z→Z→Y→X

## Met data pipeline

download (`scripts/download_*.jl`) → preprocess (`scripts/preprocess_*.jl`, optional) → run

- **ERA5**: spectral VO/D/LNSP → `preprocess_spectral_massflux.jl` → NetCDF mass fluxes (default);
  gridpoint u/v/lnsp → `preprocess_mass_fluxes.jl` (stopgap, ~0.9% mass drift)
- **GEOS-FP C720**: NetCDF mass fluxes from WashU → optional binary staging via `preprocess_geosfp_cs.jl`
- **GEOS-IT C180**: same pipeline as GEOS-FP; data from WashU archive

## Advection schemes

- `scheme = "slopes"` (default): van Leer slopes, 2nd-order
- `scheme = "ppm"` with `ppm_order ∈ {4,5,6,7}`: Putman & Lin (2007) PPM

## Critical invariants

These are hard-won correctness constraints. Violating any causes silent wrong results:

1. **CS panel convention**: GEOS-FP file panels (nf=1..6) differ from gnomonic numbering.
   Code uses FILE convention. `panel_connectivity.jl` and `cgrid_to_staggered_panels`
   handle rotated boundary fluxes (MFXC↔MFYC at panel edges).

2. **Vertical level ordering**: GEOS-IT is bottom-to-top (k=1=surface); GEOS-FP is
   top-to-bottom (k=1=TOA). Auto-detected in reader; wrong ordering → extreme CFL, NaN.

3. **GEOS mass_flux_dt** (GEOS-IT AND GEOS-FP): MFXC accumulated over dynamics timestep
   (~450s), NOT the 1-hour met interval. Config: `mass_flux_dt = 450` in `[met_data]`.
   Without this, transport is 8x too slow. Verified for both GEOS-IT C180 (via A3dyn
   wind comparison) and GEOS-FP C720 (via surface wind speed analysis).

4. **Z-advection double buffering**: kernel reads from `ws.rm_buf`/`ws.m_buf` (copies of
   originals), writes to `rm`/`m`. In-place update breaks flux telescoping (~10% mass loss).

5. **Tiedtke convection**: explicit upwind, conditionally stable. Adaptive subcycling in
   `_max_conv_cfl_cs` keeps CFL < 0.9. No positivity clamp needed.

6. **Spectral SHT**: FFTW `bfft` is unnormalized backward FFT = direct Fourier synthesis.
   Do NOT divide by N. Must fill negative frequencies (`fft_buf[N-m+1] = conj(fft_buf[m+1])`)
   for real-valued fields.

7. **Diffusion for column-mean output**: without vertical mixing, surface emissions stay
   in bottom level. Column-mean dilutes by ~72x → looks like zero transport. Always enable
   `[diffusion]` for realistic visualization.
