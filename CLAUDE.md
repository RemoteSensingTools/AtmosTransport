# AtmosTransport.jl — Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Solves tracer
continuity on latitude-longitude (ERA5) and cubed-sphere (GEOS-FP C720, GEOS-IT C180)
grids with mass-conserving advection, Tiedtke/RAS convection, and boundary-layer diffusion.
Oceananigans-inspired modular architecture; KernelAbstractions.jl for GPU portability.

# Rule 1: Never speculate without evidence, check your own intuition, it was often wrong

# Rule 1b: Assume bugs first. If a result looks unphysical, it IS a bug until proven
# otherwise with quantitative evidence. Do NOT rationalize wrong numbers. CO2 does not
# double at the surface in 6 hours. Wind speeds are 5-10 m/s, not 50. Mass doesn't
# appear from nowhere. When you see a suspicious result, say "this looks wrong, let me
# find the bug" — not "this could be explained by X."

# Rule 2: Debugging protocol for transport and physics issues

When investigating ANY transport bug, numerical instability, or physics mismatch:

1. **Read the reference code FIRST.** The TM5 F90 source (`deps/tm5/base/src/`) and
   the old working Julia code (git history, legacy scripts) are the ground truth.
   Read them LINE BY LINE before forming any hypothesis.

2. **Diff old vs new.** If something used to work and now doesn't, the answer is in
   the diff. Find the EXACT semantic change — not an approximate summary.

3. **Red team / blue team for every significant change.** Before implementing any
   fix to advection, preprocessing, or flux scaling:
   - Launch a PROPOSE agent to design the change with TM5 F90 evidence
   - Launch an EVALUATE agent to challenge it, find risks, verify against reference
   - Only implement after both agents agree (or disagreement is explicitly resolved)
   - NEVER skip this for "obvious" fixes — the ERA5 LL NaN bug (2026-04-03) was a
     one-line change (m_dev reset inside vs outside substep loop) that was missed
     for 12+ hours because this protocol was not followed.

4. **One change at a time.** Make one change, test, verify. Never stack multiple
   untested hypotheses. Each change must have evidence (code reference, math derivation,
   or diagnostic output) BEFORE implementation.

5. **Never guess at scale factors.** If a flux needs scaling, derive the factor from
   first principles with the exact TM5 formula, verify the units, and confirm with
   a diagnostic. The 4× scaling attempt (2026-04-02) wasted hours because the math
   was done without reading how TM5 actually applies the fluxes.

## Quick start

```bash
julia --project=. scripts/run.jl config/runs/<config>.toml
```

All simulation parameters live in TOML configs under `config/runs/`. Preprocessing
configs are in `config/preprocessing/`. See `docs/QUICKSTART.md` for full reference.

---

## Architecture overview

### Module dependency chain

```
Architectures → Parameters → Communications
       ↓
Grids → Fields
       ↓
Advection → Convection → Diffusion → Chemistry
       ↓
TimeSteppers → Adjoint → Callbacks
       ↓
Regridding → Diagnostics → Sources
       ↓
IO → Visualization → Models
```

`src/AtmosTransport.jl` includes all modules in this strict order. Each module
may import from earlier modules but never from later ones.

### Type hierarchy (multiple dispatch)

All physics and infrastructure is organized via abstract type hierarchies.
Physics code dispatches on concrete types — no if/else on grid or scheme.

```
AbstractArchitecture        → CPU, GPU
AbstractGrid{FT, Arch}      → LatitudeLongitudeGrid, CubedSphereGrid
AbstractAdvectionScheme      → SlopesAdvection, PPMAdvection{ORD}
AbstractConvection           → NoConvection, TiedtkeConvection, RASConvection
AbstractDiffusion            → NoDiffusion, BoundaryLayerDiffusion, PBLDiffusion, NonLocalPBLDiffusion
AbstractChemistry            → NoChemistry, RadioactiveDecay, CompositeChemistry
AbstractMetDriver{FT}        → ERA5MetDriver, PreprocessedLatLonMetDriver, GEOSFPCubedSphereMetDriver
AbstractSurfaceFlux          → SurfaceFlux, TimeVaryingSurfaceFlux, EdgarSource, ...
AbstractBufferingStrategy    → SingleBuffer, DoubleBuffer
AbstractOutputWriter         → BinaryOutputWriter, NetCDFOutputWriter
```

### Key design patterns

- **TOML-driven**: all parameters in config files, no hardcoded paths in source
- **Canonical variables**: physics code uses abstract names (e.g. `surface_pressure`);
  met source TOMLs map native variable names to these canonicals
- **GPU via extension**: `using CUDA` or `using Metal` before `using AtmosTransport` loads
  the appropriate extension (`AtmosTransportCUDAExt` or `AtmosTransportMetalExt`).
  `scripts/run.jl` auto-detects: Metal on macOS, CUDA on Linux/Windows.
- **KernelAbstractions**: all GPU kernels use `@kernel`/`@index` for CUDA/Metal/CPU portability
- **Mass conservation first**: advection via mass fluxes (not winds); vertical flux from
  continuity equation; Strang splitting X→Y→Z→Z→Y→X
- **TransportPolicy**: single source of truth for transport configuration (vertical method,
  pressure basis, mass balance). See `src/Models/transport_policy.jl`.

### Source layout

Key directories:
- `src/` — all Julia source code, organized by module
- `scripts/` — download, preprocess, run, diagnostic, and visualization scripts
- `config/runs/` — simulation configs
- `config/preprocessing/` — preprocessor configs
- `config/met_sources/` — per-source variable mappings to canonical names
- `config/canonical_variables.toml` — the contract between met sources and physics code
- `test/` — test suite (15 files, 4570+ tests)
- `docs/` — Documenter.jl + Literate.jl documentation
- `ext/` — GPU extensions (CUDA, Metal)

---

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
- **Validated** (2026-03-13): 48-window (2-day) uniform tracer test: surface std
  2.0 ppm (vs 93 ppm with hybrid PE), mass drift 0.02% (vs 0.75%), no panel
  boundary artifacts (edge/interior ratio 0.93).

## Run loop architecture

Unified `_run_loop!` via multiple dispatch (replaced 4 duplicated methods, ~1940 lines).
Files in `src/Models/`:
- `run_loop.jl` — single entry point `run!(model)` + `_run_loop!`
- `physics_phases.jl` — grid-dispatched phase functions (IO, compute, advection, output)
- `simulation_state.jl` — allocation factories for tracers, air mass, workspaces
- `io_scheduler.jl` — `IOScheduler{B}` abstracts single/double buffering
- `mass_diagnostics.jl` — `MassDiagnostics` + global mass fixer
- `run_helpers.jl` — physics helpers, advection dispatch, emission wrappers
- `transport_policy.jl` — `TransportPolicy` type for transport configuration

---

## Critical invariants

These are hard-won correctness constraints. Violating any causes silent wrong results.

### If you see... check this

| Symptom | Likely cause | Where to look |
|---------|-------------|---------------|
| Transport 8x too slow | Missing `mass_flux_dt = 450` | Invariant 3 below |
| Extreme CFL, NaN | Wrong vertical level ordering | Invariant 2 |
| ~10% mass loss per step | In-place Z-advection (no double buffer) | Invariant 4 |
| Panel boundary waves (CS) | Wrong flux rotation or missing `linrood` | Invariant 1 |
| Surface emissions invisible in column mean | Missing `[diffusion]` section | Invariant 7 |
| 5x slower CPU loops | Wrong loop nesting order | Invariant 8 |
| CO2 surface 535 ppm from 400 ppm uniform | Hybrid PE in vertical remap | Use direct cumsum PE |

### Invariant details

1. **CS panel convention**: GEOS-FP file panels (nf=1..6) differ from gnomonic numbering.
   Code uses FILE convention. `panel_connectivity.jl` and `cgrid_to_staggered_panels`
   handle rotated boundary fluxes (MFXC↔MFYC at panel edges).

2. **Vertical level ordering**: GEOS-IT is bottom-to-top (k=1=surface); GEOS-FP is
   top-to-bottom (k=1=TOA). Auto-detected in reader; wrong ordering → extreme CFL, NaN.

3. **GEOS mass_flux_dt** (GEOS-IT AND GEOS-FP): MFXC accumulated over dynamics timestep
   (~450s), NOT the 1-hour met interval. Config: `mass_flux_dt = 450` in `[met_data]`.
   Without this, transport is 8x too slow. Verified for both GEOS-IT C180 and GEOS-FP C720.

4. **Z-advection double buffering**: kernel reads from `ws.rm_buf`/`ws.m_buf` (copies of
   originals), writes to `rm`/`m`. In-place update breaks flux telescoping (~10% mass loss).

5. **Tiedtke convection**: explicit upwind, conditionally stable. Adaptive subcycling in
   `_max_conv_cfl_cs` keeps CFL < 0.9. No positivity clamp needed.

6. **Spectral SHT**: FFTW `bfft` is unnormalized backward FFT = direct Fourier synthesis.
   Do NOT divide by N. Must fill negative frequencies for real-valued fields.

7. **Diffusion for column-mean output**: without vertical mixing, surface emissions stay
   in bottom level. Column-mean dilutes by ~72x → looks like zero transport. Always enable
   `[diffusion]` for realistic visualization.

8. **Column-major loop ordering**: Julia is column-major (like Fortran). In nested loops
   over arrays, the **leftmost/innermost index must be the innermost loop**. For a 3D
   array `A[i, j, k]`, loop as `for k, for j, for i` (i innermost). Wrong order causes
   catastrophic cache misses (5x slowdown).

9. **Moist vs dry mass fluxes in GEOS met data**: FV3's dynamical core operates on dry
   air mass. The exported variables have DIFFERENT moisture conventions:

   | Variable | Basis | Notes |
   |----------|-------|-------|
   | MFXC, MFYC | **DRY** | Horizontal mass fluxes from FV3 dynamics |
   | DELP | **MOIST** (total) | Exported as total pressure thickness |
   | CMFMC | **MOIST** (total) | Convective mass flux |
   | DTRAIN | **MOIST** (total) | Detraining mass flux |
   | QV | — | Specific humidity; convert wet→dry: `x_dry = x_wet × (1 - qv)` |

   **GCHP path (`gchp=true`)**: Runs on MOIST basis (matches GCHP `Use_Total_Air_Pressure > 0`).
   MFXC converted to moist: `MFX = MFXC / (1-QV)` (GCHPctmEnv:1029). Tracers converted
   dry→wet before advection: `q_wet = q × (1-QV)` (AdvCore:1070). dp = moist DELP.
   Back-conversion wet→dry is implicit via `rm / m_dry`. Required because ak/bk in the
   vertical remap target PE expect moist surface pressure.

   **Non-GCHP paths**: Transport and diffusion run on DRY basis; convection on MOIST basis.

---

## Workflow: Adding a new advection scheme

1. **Define the type** in `src/Advection/`:
   ```julia
   struct MyScheme <: AbstractAdvectionScheme
       param1::Float64
   end
   ```

2. **Implement the interface** (see `src/Advection/abstract_advection.jl` for contract):
   ```julia
   advect!(tracers, vel, grid, scheme::MyScheme, Δt) = ...
   adjoint_advect!(adj_tracers, vel, grid, scheme::MyScheme, Δt) = ...
   ```
   For operator splitting, also implement `advect_x!`, `advect_y!`, `advect_z!`.

3. **Wire into config** (`src/IO/configuration.jl`):
   Add a branch in `_build_advection_scheme(config)` to parse your TOML params.

4. **Export** from `src/Advection/Advection.jl`.

5. **Add tests** in `test/test_advection.jl`:
   - Uniform field invariance
   - Mass conservation
   - Adjoint identity (dot-product test)

## Workflow: Adding a new met driver

1. **Subtype `AbstractMetDriver{FT}`** in `src/IO/`:
   ```julia
   struct MyDriver{FT} <: AbstractMetDriver{FT}
       ...
   end
   ```

2. **Implement required methods** (see `src/IO/abstract_met_driver.jl`):
   - `total_windows(driver)` → Int
   - `window_dt(driver)` → seconds per window
   - `steps_per_window(driver)` → sub-steps per window
   - `load_met_window!(cpu_buf, driver, grid, win_index)` → fill buffer with am, bm, cm, delp

3. **Add TOML variable mapping** in `config/met_sources/your_source.toml`.

4. **Register** in `src/IO/configuration.jl` driver construction.

## Workflow: Adding a new physics operator

Follow the pattern used by existing operators (convection, diffusion, chemistry):

1. **Define abstract + concrete types** with keyword constructor and validation
2. **Implement forward + adjoint** methods dispatching on your type
3. **Add GPU kernels** using `@kernel`/`@index` from KernelAbstractions
4. **Wire into `OperatorSplittingTimeStepper`** if needed
5. **Add tests** for mass conservation and adjoint identity

See `src/Diffusion/Diffusion.jl` for a clean example with 4 implementations.

## Workflow: Running an ERA5 vs GEOS comparison

1. Create matched configs with identical physics settings
2. Use `TransportPolicy` to ensure same transport configuration
3. Run both: `julia --project=. scripts/run.jl config/runs/comparison/<era5>.toml`
4. Convert output: `julia --project=. scripts/postprocessing/convert_output_to_netcdf.jl`
5. Compare: use `scripts/visualization/animate_comparison.jl` or custom analysis

---

## Performance tips

- **GPU vs CPU**: GPU kernels via KernelAbstractions. Same code runs on both.
  All array allocation dispatches on `array_type(arch)` → `Array` (CPU) or `CuArray` (GPU).
- **IO dominates for NetCDF**: preprocessed binary < 1 s/win vs 15+ s/win for on-the-fly NetCDF
- **Double buffering**: overlaps IO with GPU compute. Use `buffering = "double"` in config.
- **Column-major loops**: see invariant 8. Always verify loop nesting matches memory layout.
- **Avoid GC pressure**: large intermediate allocations trigger GC (was 31 s/win, fixed to 0.9 s/win
  by reading directly into output panels with a single buffer).
- **Kernel launch overhead**: fuse small sequential kernels where memory traffic dominates.
- **Coalesced access**: ensure fastest-varying thread dimension matches innermost array index.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

15 test files, 4570+ tests. Key test categories:
- Per-module physics tests (advection, diffusion, convection)
- Mass conservation (total tracer mass preserved)
- Adjoint identity (dot-product test: machine precision for linear operators)
- TM5 reference validation
- Integration tests (full run loop)

---

## Reference: GEOS-Chem comparison notes

See `docs/TRANSPORT_COMPARISON.md` for detailed comparison of
AtmosTransport vs TM5 vs GEOS-Chem/GCHP algorithms, including the phased
implementation blueprint for unified ERA5/GEOS transport. Parts might be obsolete now...

## Data Layout
Follow the data guide in `docs/DATA_LAYOUT.md`
