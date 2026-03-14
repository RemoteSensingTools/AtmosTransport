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
- `linrood = true`: Lin-Rood cross-term advection (FV3's `fv_tp_2d` algorithm).
  Averages X-first and Y-first PPM from original field. Eliminates directional
  splitting bias at CS panel boundaries. File: `src/Advection/cubed_sphere_fvtp2d.jl`.
- `vertical_remap = true`: Replaces Strang-split Z-advection with FV3-style
  conservative PPM remap (`map1_q2` + `ppm_profile`). Horizontal-only Lin-Rood
  transport on Lagrangian surfaces, then single vertical remap per window.
  File: `src/Advection/vertical_remap.jl`.

## Vertical remap — GCHP comparison

The vertical remap path is designed to match GCHP's `offline_tracer_advection`:
- **Identical**: Lin-Rood `fv_tp_2d` per level, dry basis, PPM remap.
- **PE computation** (fixed 2026-03-13): Source PE from air mass cumsum (inside
  remap kernel), target PE from `cumsum(next_delp × (1-qv))`. Both use direct
  cumsum — NO hybrid formula (`ak + bk × PS`). The hybrid approach was tested
  but caused massive vertical pumping (uniform 400 ppm → 535 ppm surface, 44 ppm
  std after 2 days) because GEOS DELP deviates from the hybrid formula by 0.1-1%
  per level (up to 250 Pa). GCHP compensates with `calcScalingFactor`; we avoid
  the issue entirely by using direct cumsum. See `fv_tracer2d.F90:988-1070`.
- **GCHP reference** (`offline_tracer_advection`): source PE = cumsum of
  post-horizontal dpA, target PE = `ak + bk × PS_from_dpA` with bottom PE clamped
  to source (`pe2(npz+1) = pe1(npz+1)`). The clamping preserves column mass. GCHP
  then applies `calcScalingFactor` to correct remap-vs-met-PE mismatch. Our direct
  cumsum approach avoids the hybrid-vs-actual DELP discrepancy entirely.
- **Validated** (2026-03-13): 48-window (2-day) uniform tracer test: surface std
  2.0 ppm (vs 93 ppm with hybrid PE), mass drift 0.02% (vs 0.75%), no panel
  boundary artifacts (edge/interior ratio 0.93).
- 7-day CATRINE run (LR + vremap + diffusion): 4 min, all tracers Δ < 5e-5%.
- See `docs/CLAUDE_VERTICAL_REMAP_PATH_FORWARD.md` for validation protocol.

## Run loop architecture

Unified `_run_loop!` via multiple dispatch (replaced 4 duplicated methods, ~1940 lines).
Files in `src/Models/`:
- `run_loop.jl` — single entry point `run!(model)` + `_run_loop!`
- `physics_phases.jl` — grid-dispatched phase functions (IO, compute, advection, output)
- `simulation_state.jl` — allocation factories for tracers, air mass, workspaces
- `io_scheduler.jl` — `IOScheduler{B}` abstracts single/double buffering
- `mass_diagnostics.jl` — `MassDiagnostics` + global mass fixer
- `run_helpers.jl` — physics helpers, advection dispatch, emission wrappers

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

8. **Column-major loop ordering**: Julia is column-major (like Fortran). In nested loops
   over arrays, the **leftmost/innermost index must be the innermost loop**. For a 3D
   array `A[i, j, k]`, loop as `for k, for j, for i` (i innermost). Wrong order causes
   catastrophic cache misses — e.g. a CPU loop over `(Nc, Nc, Nz)` panels went from
   3.8 s to 0.8 s just by fixing loop order. Always verify loop nesting matches memory
   layout in any new CPU-side code.

9. **Moist vs dry mass fluxes in GEOS met data**: FV3's dynamical core operates on dry
   air mass. The exported variables have DIFFERENT moisture conventions:

   | Variable | Basis | Notes |
   |----------|-------|-------|
   | MFXC, MFYC | **DRY** | Horizontal mass fluxes from FV3 dynamics (dry air transport) |
   | DELP | **MOIST** (total) | Exported as total pressure thickness, NOT dry |
   | CMFMC | **MOIST** (total) | Convective mass flux — total air including vapor |
   | DTRAIN | **MOIST** (total) | Detraining mass flux — total air |
   | QV | — | Specific humidity; convert wet→dry: `x_dry = x_wet × (1 - qv)` |

   **Transport AND diffusion run on DRY basis; convection on MOIST basis.**
   GeosChem uses dry air mass (`AD = State_Met%AD`) for both PBL mixing
   (`pbl_mix_mod.F90`) and convection (`BMASS = DELP_DRY` in `convection_mod.F90`).
   `compute_air_mass_phase!` computes both:
   - `air.m` = `DELP × (1 - QV) × area / g` (DRY) — for advection, diffusion,
     and vertical remap (consistent with DRY horizontal mass fluxes am/bm).
   - `air.m_wet` = `DELP × area / g` (MOIST) — for convection only
     (CMFMC/DTRAIN are MOIST fluxes; TODO: may also need dry basis).
   `gpu.delp` stays MOIST — convection, diffusion, PS, and emissions all expect
   moist DELP. For vertical remap, `compute_target_pressure_from_dry_delp_direct!`
   builds target PE from `DELP × (1 - QV)` to match dry-basis source PE.
   The `apply_dry_*_panel!` kernels remain for backward compatibility but are no
   longer called in the main transport path.
   Configs without QV fall back to moist air mass (unchanged behavior).
