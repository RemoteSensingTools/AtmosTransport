# AtmosTransport.jl — Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Solves tracer
continuity on latitude-longitude (ERA5) and cubed-sphere (GEOS-FP C720, GEOS-IT C180)
grids with mass-conserving advection, Tiedtke/RAS convection, and boundary-layer diffusion.
Oceananigans-inspired modular architecture; KernelAbstractions.jl for GPU portability.

## Project discipline (read before editing)

**Rule 0 — Probe before building.** Before writing any function that transforms
data (spectral synthesis, vertical remap, Poisson balance, IC loading, flux
scaling), print shape/dtype and a representative value at a known physical
location (e.g. ps at Tibet ≈ 55 kPa; ps at equatorial ocean ≈ 101 kPa) and
verify it matches reality. Compare against an independent reference when one
exists (RG vs LL at the same point; Catrine source vs model-loaded IC; ERA5
GRIB vs spectral-synthesized). Probe cost: 5 seconds. Not-probing cost: hours.

**Rule 1 — Evidence, not speculation.** Never guess. Check your intuition; it
is often wrong. If a result looks unphysical, it IS a bug until quantitative
evidence proves otherwise. Don't rationalize wrong numbers — CO₂ doesn't
double at the surface in 6 hours, wind speeds are 5–10 m/s not 50, mass
doesn't appear from nowhere. Say "this looks wrong, let me find the bug",
not "this could be explained by X".

**Rule 1c — Never write hypotheses to persistent docs as fact.** MEMORY.md,
plan docs, and design memos get read by future agents who treat them as
ground truth. If pattern-matching, say "consistent with X" or "candidate: X";
never "confirmed X". Claims that an unexplained drift is "real / physical /
expected" require (a) a quantitative back-of-envelope that matches AND (b) a
citation. Without both, the entry is "drift −X% over Y, source unknown".

**Rule 2 — Transport-bug debugging protocol.** For any transport bug,
numerical instability, or physics mismatch:

1. Read the reference code line-by-line first (TM5 F90 in `deps/tm5/base/src/`,
   old Julia in git history) before forming any hypothesis.
2. If something used to work and now doesn't, find the exact semantic change
   in the diff — not an approximate summary.
3. Red team / blue team significant changes to advection, preprocessing, or
   flux scaling: propose, challenge with reference evidence, agree, then
   implement. Skipping this cost 12+ hours on the ERA5 LL NaN bug (2026-04-03).
4. One change at a time. Each change needs evidence (code reference, math
   derivation, or diagnostic output) before implementation.
5. Never guess scale factors. Derive from first principles with the exact
   reference formula, verify units, confirm with a diagnostic. The 4× scaling
   attempt (2026-04-02) wasted hours because the math was done without
   reading how TM5 actually applies the fluxes.

**Rule 3 — Production-level codebase.** Prefer clean, mathematically correct,
future-proof implementations over quick hacks. No ad-hoc limiters/clamps
without a first-principles derivation and a WHY comment. No "temporary"
workarounds that silently become permanent — mark real shortcuts with
`# HACK:` + TODO pointing at the clean replacement. Code should be
self-documenting via clear docstrings, convention comments, and unit
annotations. A newcomer reading any function should understand the math,
sign conventions, and index layout without external docs.

## Quick start

```bash
julia --project=. scripts/run.jl config/runs/<config>.toml
```

All simulation parameters live in TOML configs under `config/runs/`. Preprocessing
configs are in `config/preprocessing/`. See `docs/reference/QUICKSTART.md` for full reference.

---

## Architecture overview

### Module dependency chain

```
Architectures → Parameters → Grids
       ↓
State (CellState, FluxState, Fields/AbstractTimeVaryingField, MetState)
       ↓
MetDrivers (ERA5, GEOS binaries, current_time accessor)
       ↓
Operators (Diffusion → SurfaceFlux → Advection → Chemistry)
       ↓
Kernels (CellKernels, FaceKernels, ColumnKernels)
       ↓
Models (TransportModel, DrivenSimulation, Simulation)
       ↓
Regridding → Preprocessing → Downloads
```

`src/AtmosTransport.jl` includes all modules in this strict order. Each module
may import from earlier modules but never from later ones. Inside
`src/Operators/`, `Operators.jl` pulls Diffusion before Advection (so the
palindrome's `V` call can dispatch) and SurfaceFlux before Advection in
plan 17 (so the palindrome's `S` call can dispatch).

### Type hierarchy (multiple dispatch)

All physics and infrastructure is organized via abstract type hierarchies.
Physics code dispatches on concrete types — no if/else on grid or scheme.

```
AbstractArchitecture             → CPU, GPU
AbstractGrid{FT, Arch}           → AtmosGrid{<:LatLonMesh}, AtmosGrid{<:CubedSphereMesh}
AbstractAdvectionScheme          → UpwindScheme, SlopesScheme{L}, PPMScheme{L, ORD}
AbstractDiffusionOperator        → NoDiffusion, ImplicitVerticalDiffusion{FT, KzF}        (plan 16b, src/Operators/Diffusion/)
AbstractChemistryOperator        → NoChemistry, ExponentialDecay{FT, N, R}, CompositeChemistry  (plan 15)
AbstractSurfaceFluxOperator      → NoSurfaceFlux, SurfaceFluxOperator{FT, M}              (plan 17, src/Operators/SurfaceFlux/)
AbstractTimeVaryingField{FT, N}  → ConstantField{FT, N}, ProfileKzField{FT, V},
                                   PreComputedKzField{FT, A}, DerivedKzField{FT, ...},
                                   StepwiseField{FT, N, A, B, W}                         (plan 16a/16b/17, src/State/Fields/)
AbstractMetDriver                → PreprocessedERA5Driver, TransportBinaryDriver, CubedSphereBinaryReader, ...
AbstractMassBasis                → DryBasis, MoistBasis
```

Legacy src_legacy/ hierarchies (parked, not loaded): `AbstractConvection` (NoConvection / TiedtkeConvection / RASConvection), `AbstractDiffusion` (BoundaryLayerDiffusion / PBLDiffusion / NonLocalPBLDiffusion), and the pre-restructure `AbstractSurfaceFlux` hierarchy.

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
- `src/` — active Julia source (basis-explicit transport core, promoted from src_v2)
- `test/` — active test suite (promoted from test_v2)
- `scripts/` — download, preprocess, run, diagnostic, and visualization scripts
- `config/runs/` — simulation configs
- `config/preprocessing/` — preprocessor configs
- `config/met_sources/` — per-source variable mappings to canonical names
- `config/canonical_variables.toml` — the contract between met sources and physics code
- `docs/` — active documentation (contracts, reference, memos)
- `src_legacy/` — original TM5-aligned module (parked, not loaded by Project.toml)
- `test_legacy/` — tests for src_legacy
- `ext_legacy/` — GPU extensions for src_legacy (CUDA, Metal)
- `docs_legacy/` — Documenter.jl site + session logs for src_legacy
- `scripts_legacy/` — v1-only run/diagnostic/validation scripts

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
| ~10% mass loss per step | In-place sweep update (kernel wrote dst == src) | Invariant 4 |
| Panel boundary waves (CS) | Wrong flux rotation or missing `linrood` | Invariant 1 |
| Surface emissions invisible in column mean | Missing `[diffusion]` section | Invariant 7 |
| 5x slower CPU loops | Wrong loop nesting order | Invariant 8 |
| CO2 surface 535 ppm from 400 ppm uniform | Hybrid PE in vertical remap | Use direct cumsum PE |
| Polar mass drainage / Y nloop max_nloop hit | Stale binary from old preprocessor OR mass_fixer=false | Invariant 10, 11 |
| `STALE BINARY WARNING` at startup | Binary written by older preprocessor than current source | Invariant 10 |
| `cm-continuity check FAILED` at startup | Binary's cm doesn't match its am/bm divergence — regenerate | Invariant 10 |
| Tracer mass drifts ~10⁻⁴/day with uniform IC (ERA5 LL) | Mass fix disabled in preprocessor; raw ERA5 ⟨ps⟩ drift not absorbed | Invariant 12 |
| CS Poisson balance absorbs moisture signal | Moist-basis preprocessing: Poisson on moist fluxes | Invariant 14 |
| `mass_basis field missing` warning at startup | Old binary without mass_basis in header — regenerate on dry basis | Invariant 14 |
| Uniform-IC tracer deviates ±1-5 ppm at day boundary, clean within a day | Dry-basis binary written with legacy Δb×pit cm closure — regenerate | Invariant 10 |
| `Write-time replay gate FAILED: rel=… > tol=…` during preprocessing | Preprocessor path bypassing the explicit-dm closure | Invariant 10 |

### Invariant details

1. **CS panel convention**: GEOS-FP file panels (nf=1..6) differ from gnomonic numbering.
   Code uses FILE convention. `panel_connectivity.jl` and `cgrid_to_staggered_panels`
   handle rotated boundary fluxes (MFXC↔MFYC at panel edges).

2. **Vertical level ordering**: GEOS-IT is bottom-to-top (k=1=surface); GEOS-FP is
   top-to-bottom (k=1=TOA). Auto-detected in reader; wrong ordering → extreme CFL, NaN.

3. **GEOS mass_flux_dt** (GEOS-IT AND GEOS-FP): MFXC accumulated over dynamics timestep
   (~450s), NOT the 1-hour met interval. Config: `mass_flux_dt = 450` in `[met_data]`.
   Without this, transport is 8x too slow. Verified for both GEOS-IT C180 and GEOS-FP C720.

4. **Ping-pong double buffering** (all directional sweeps, not just Z): every sweep kernel
   writes to a DIFFERENT array than it reads from. `AdvectionWorkspace` carries two buffer
   pairs (`rm_A`/`m_A` + `rm_B`/`m_B`), and `strang_split!` / `strang_split_mt!` alternate
   source ↔ destination across the six-sweep palindrome, tracking parity so the final
   result lands back in the caller's arrays with zero inter-sweep `copyto!` in the common
   `n_sub == 1` case. In-place (src == dst) kernel update still breaks flux telescoping
   (~10% mass loss). Plan 11 shipped with `rm_buf`/`m_buf`/`rm_4d_buf` preserved as
   `getproperty` aliases; plan 13 Commit 4 (`12560be`) dropped the shim and renamed all
   callers (LinRood, CubedSphereStrang, tests) to the final `rm_A`/`m_A`/`rm_4d_A` names.
   See commit sequence on `restructure/dry-flux-interface` (`fb55852`, `1a7e2ba`, `12560be`).

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

10. **Transport binaries MUST satisfy the explicit-dm cm closure contract
    at write time, not the hybrid Δb×pit closure.** The unified v2
    preprocessor (`scripts/preprocessing/preprocess_transport_binary.jl`)
    builds per-window `(am, bm, cm)` so that

        m[k] − 2·steps·((am[i+1]−am[i]) + (bm[j+1]−bm[j]) + (cm[k+1]−cm[k])) = m[k+1]

    holds per cell to Poisson-balance tolerance (~F64 ULP). The legacy
    Δb×pit closure (LL: `recompute_cm_from_divergence!`, RG:
    `recompute_faceindexed_cm_from_divergence!`) only satisfies this
    under moist hybrid coordinates where `dm[k] = dB[k] × Σ_k dm[k]`. On
    dry-basis binaries `qv[k]` varies with level, breaking the identity
    by up to 27%, which produced the 0.75% day-boundary air_mass jump
    found in the plan-39 F64 probe (2026-04-22). Fixed by using the
    explicit-dm closure (`recompute_cm_from_dm_target!`,
    `recompute_faceindexed_cm_from_dm_target!`) with CS-style residual
    redistribution at `cm[Nz+1]`.

    **Three gates enforce this invariant:**
    - **Write-time gate (Commit E)**: `verify_storage_continuity_ll!` +
      `verify_storage_continuity_rg!` run inside `apply_poisson_balance!`
      / `apply_reduced_poisson_balance!` / `balance_window!`. Errors
      loudly if per-cell rel err > `1e-10` (F64) / `1e-4` (F32). Bypass
      with `ENV["ATMOSTR_NO_WRITE_REPLAY_CHECK"]="1"`.
    - **Load-time gate (Commit F)**: LL-only, opt-in via
      `TransportBinaryDriver(path; validate_replay=true)` or
      `ENV["ATMOSTR_REPLAY_CHECK"]="1"`. For suspect binaries (manual
      imports, older preprocessor versions). Bypass an enabled gate with
      `ENV["ATMOSTR_NO_REPLAY_CHECK"]="1"`.
    - **Regression tests**: `test/test_replay_consistency.jl` (18 tests).

    Symptoms of a binary produced with the legacy Δb×pit closure on dry
    basis: ~0.75% day-boundary `max|m_runtime_end − m_stored_next|/max|m|`
    jump; Day N+1 t=0 tracer VMR deviates ±1% from Day N end; mass drift
    itself remains ~0% because total mass is conserved (just redistributed
    across the level-qv-weighted mismatch).

    Also applies to legacy `MassFluxBinaryReader` binaries: it runs a
    `_verify_cm_continuity` pass on window 1 at driver construction
    (added 2026-04-06). Disable with `ENV["ATMOSTR_NO_CM_CHECK"]="1"`.
    Provenance fields (script_path, script_mtime, git_commit, git_dirty)
    in v4 headers also trigger a stale-binary warning if the source has
    moved on. Disable with `ENV["ATMOSTR_NO_STALE_CHECK"]="1"`.

11. **`mass_fixer = true` required for ERA5 LL F64 stability.** Pole-adjacent
    stratospheric cells (j ∈ {1, 2, Ny-1, Ny}) have |bm|/m ≈ 0.30 per face
    from the spectral preprocessor. Without the runtime mass fixer, cumulative
    drainage over one Strang palindrome exceeds cell mass and the local
    nloop hits its max. Regression test:
    `config/runs/era5_f64_debug_moist_v4_nofix.toml`. This is local cell
    stability, independent of the global drift fix (invariant 12). Whether
    TM5 r1112 needs the same fixer is NOT verified — earlier drafts claimed
    it did; that was never checked against actual TM5 sources.

12. **Global mean ps pinned in the v4 spectral preprocessor** (TM5
    `Match('area-aver')` equivalent). ERA5's 4DVar analysis is not
    mass-conserving; raw ⟨ps⟩ drifts ~10⁻⁴/day →
    `Σm = const + (A_Earth/g)·⟨sp⟩_area` drifts in lockstep.
    `preprocess_spectral_v4_binary.jl` applies a uniform additive shift to
    `sp` so that `⟨sp⟩_area` tracks a fixed dry-air target
    (Trenberth & Smith 2005: M_dry = 5.1352e18 kg → ⟨ps_dry⟩ ≈ 98726 Pa).
    Mirrors TM5 cy3-4dvar `meteo.F90:1361-1374`. Verified 2026-04-07: Σm
    drift drops from -8.31e-04% to +5.88e-09% (F32 quantization floor).
    Disable via `[mass_fix] enable=false` for diagnostic runs. Does NOT
    fix local polar drainage (invariant 11) — uniform shift only changes
    `m` by `b_k · Δps` ≈ 0.4% at surface.

13. **Every binary preprocessor that diagnoses `cm` from continuity MUST
    Poisson-balance horizontal fluxes first.** Raw spectral-synthesis
    fluxes have divergence residuals ~10¹² kg/cell at ERA5 T47; these
    integrate through the continuity diagnosis into unphysical `cm` and
    either trip the face-indexed CFL pilot (`max_n_sub=4096`) or drive
    silent instability on limiter-equipped schemes. Three implementations:

    - **LL**: [`mass_support.jl`](src/Preprocessing/mass_support.jl)
      `balance_mass_fluxes!` — 2D FFT on the circulant Laplacian.
    - **RG**: [`reduced_transport_helpers.jl`](src/Preprocessing/reduced_transport_helpers.jl)
      `balance_reduced_horizontal_fluxes!` — JPCG on the graph Laplacian
      with interior-only face degrees; rhs/r/p kept mean-zero because
      Laplacian is singular (constant null space).
    - **CS**: [`cs_poisson_balance.jl`](src/Preprocessing/cs_poisson_balance.jl)
      `balance_cs_global_mass_fluxes!` — JPCG on the global 6-panel
      Laplacian, 12Nc² faces. Cross-panel faces use canonical/mirror
      same-sign pairs. Replaces the broken per-panel FFT.

    Validation target: post-balance `max(|cm|/m) ~ 1e-14` (F64 floor).
    When adding a new grid-type preprocessor, verify this probe before
    declaring success. Two JPCG gotchas: (a) count only INTERIOR faces
    in the diagonal — boundary-stub pole faces produce ghost residuals;
    (b) project `r` to mean-zero only when the Laplacian is singular.
    Getting (b) wrong gives false-positive L2 convergence with 13-
    orders-of-magnitude max-residual error.

14. **Dry-basis is the default and required contract for transport
    binaries.** Runtime never performs moist→dry conversion; all carrier-
    mass conversion and continuity closure happen in preprocessing:

    ```
    spectral/gridpoint synthesis (moist)
      → load native QV, convert m/am/bm to dry: field *= (1 - qv)
      → merge to transport levels → Poisson-balance dry fluxes
      → diagnose dry cm → write binary (header: mass_basis = :dry)
      → runtime reads dry fields directly; rm₀ = vmr × m_dry
    ```

    All three grid paths support dry preprocessing:
    [`apply_dry_basis_native!`](src/Preprocessing/mass_support.jl) for
    LL/CS; [`apply_dry_basis_reduced!`](src/Preprocessing/reduced_transport_helpers.jl)
    for RG (with bilinear QV interpolation from LL thermo grid).

    [`DryFluxBuilder`](src/MetDrivers/ERA5/DryFluxBuilder.jl) is the
    runtime converter retained for moist-basis binaries only; binaries
    without `mass_basis` default to `:moist` with a warning. Preprocessing
    TOMLs must set `mass_basis = "dry"` and `thermo_dir`.

---

## Workflow: Adding a new advection scheme

1. **Define the type** in `src/Operators/Advection/schemes.jl`:
   ```julia
   struct MyScheme <: AbstractAdvectionScheme
       param1::Float64
   end
   ```

2. **Implement the interface** (see `src/Operators/Advection/Advection.jl` for contract).
   For operator-split structured grids, wire into `strang_split!` /
   `strang_split_mt!` via the sweep kernels in `structured_kernels.jl`.

3. **Export** from `src/Operators/Advection/Advection.jl`.

4. **Add tests** in `test/test_advection_kernels.jl` / `test/test_cubed_sphere_advection.jl`:
   - Uniform field invariance (constant → constant)
   - Mass conservation (total tracer mass preserved to machine precision F64 / ULP F32)
   - Adjoint identity (dot-product test)
   - CPU/GPU agreement to ≤4 ULP (1-step), ≤16 ULP (4-step)

## Workflow: Adding a new met driver

1. **Subtype `AbstractMetDriver`** in `src/MetDrivers/`:
   ```julia
   struct MyDriver{FT} <: AbstractMetDriver
       ...
   end
   ```

2. **Implement required methods** (see `src/MetDrivers/AbstractMetDriver.jl`):
   - `total_windows(driver)` → Int
   - `window_dt(driver)` → seconds per window
   - `steps_per_window(driver)` → sub-steps per window
   - `load_transport_window(driver, win_idx)` → transport window with am, bm, cm, air_mass
   - `current_time(driver)` → seconds since run start, for `update_field!` of `StepwiseField` / `DerivedKzField` (plan 16b stub, threaded through in plan 17)

3. **Add TOML variable mapping** in `config/met_sources/your_source.toml`.

4. **Register** in scripts/config that construct drivers (e.g. `scripts/run_transport_binary.jl`).

## Workflow: Adding a new physics operator

Pattern validated across chemistry (plan 15), diffusion (plan 16b), and surface
emissions (plan 17). All operators conform to a stable `apply!` signature:

```julia
apply!(state::CellState, meteo, grid::AtmosGrid, op, dt::Real; workspace) -> state
```

Steps:

1. **Define abstract + concrete types** in `src/Operators/<YourOperator>/`:
   ```julia
   abstract type AbstractYourOperator end
   struct NoYourOperator <: AbstractYourOperator end
   struct YourOperator{FT, F} <: AbstractYourOperator
       field::F     # e.g. an AbstractTimeVaryingField or a parametric input
   end
   ```

2. **Ship a `No<Operator>` dead branch** as the default. Dispatch should compile
   away to zero floating-point work so the default path preserves pre-refactor
   behaviour bit-exact:
   ```julia
   apply!(state, meteo, grid, ::NoYourOperator, dt; workspace) = state
   ```
   If the operator is called from inside the palindrome (plan 16b/17 style),
   ship an **array-level entry point** too — the palindrome operates on
   whichever ping-pong buffer currently holds the tracer state, not always
   `state.tracers_raw`:
   ```julia
   apply_your_operator!(q_raw::AbstractArray{FT, 4}, ::NoYourOperator, ws, dt) = nothing
   ```

3. **Preserve coefficient structures for future adjoints**. If the operator's
   solver has identifiable coefficients (e.g. tridiagonal `(a, b, c)` for
   backward-Euler diffusion), keep them as named locals per index — do NOT
   fuse into prefactored forms like `w[k]` / `inv_denom[k]`. The transposition
   rule (e.g. `a_T[k] = c[k-1]; b_T[k] = b[k]; c_T[k] = a[k+1]`) becomes a
   mechanical adjoint port. Post-hoc un-fusing is not mechanical.

4. **GPU kernels via KernelAbstractions**. Use `@kernel` / `@index`; launch
   with `get_backend(state.tracers_raw)`; `synchronize(backend)` once at the
   end. Fields backing device arrays need `Adapt.adapt_structure` methods so
   `samples::Vector → CuArray{FT, 1}` transparently (see `ProfileKzField` / 
   `StepwiseField`).

5. **Wire through TransportModel**. Add a field to `TransportModel`, ship a
   `with_<operator>(model, op)` helper, and thread the operator into the
   appropriate block of `step!(model, dt)`.

6. **Add tests** in `test/test_<your_operator>*.jl`:
   - `No<Operator>` default path is **bit-exact** `==` to the explicit-no-op
     path (not just `≈`; any FP work means callers without the kwarg see
     subtly different numerics)
   - Mass conservation (if applicable)
   - CPU/GPU consistency
   - Tests observe post-operator state via `get_tracer(state, name)` or
     `state.tracers.name` — NEVER through input arrays passed at construction
     (plan 14's 4D tracers_raw separated storage; cached input arrays go stale)

See `src/Operators/Diffusion/` (clean plan-16b example) or
`src/Operators/Chemistry/` (simplest plan-15 example).

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

- **CFL pilot dominates GPU runtime on unstructured grids**. The face-indexed
  (RG) CFL subcycling pilot transfers arrays GPU→CPU every substep. With
  Poisson-balanced binaries (`max(|flux|/m) ≈ 1e-14`), the pilot always returns
  `n_sub=1`, so it's pure overhead. Pass `cfl_limit=Inf` to `apply!` to bypass
  the pilot entirely. Measured impact (2026-04-12, NVIDIA L40S, F32, 24h):

  | Grid     | Type        | CPU (16t)  | GPU (pilot) | GPU (bypass) | Speedup |
  |----------|-------------|------------|-------------|--------------|---------|
  | N24      | RG face-idx | 860 ms     | 862 ms      | **8 ms**     | **103×** |
  | N90      | RG face-idx | 13.7 s     | 12.5 s      | **23 ms**    | **588×** |
  | N160     | RG face-idx | 45.9 s     | 45.5 s      | **161 ms**   | **285×** |
  | 720×361  | LL struct   | 19.6 s     | —           | **432 ms**   | 46×     |

  The face-indexed `@atomic` scatter kernel is actually faster than structured
  LL at matched cell counts because it has fewer sweep directions (H-V-V-H
  vs X-Y-Z-Z-Y-X) and the L40S handles atomic accumulation efficiently at
  high occupancy.

- **Higher-order schemes (Slopes, PPM) are free on GPU**: they cost 2× on CPU
  (wider stencil) but the GPU's arithmetic throughput absorbs the extra
  computation while memory access time stays constant. At 720×361×34:
  LL Upwind 432 ms, LL Slopes 429 ms, LL PPM 445 ms — all within noise.

- **IO dominates for NetCDF**: preprocessed binary < 1 s/win vs 15+ s/win for on-the-fly NetCDF
- **Double buffering**: overlaps IO with GPU compute. Use `buffering = "double"` in config.
- **Column-major loops**: see invariant 8. Always verify loop nesting matches memory layout.
- **Avoid GC pressure**: large intermediate allocations trigger GC (was 31 s/win, fixed to 0.9 s/win
  by reading directly into output panels with a single buffer).
- **Kernel launch overhead**: fuse small sequential kernels where memory traffic dominates.
- **Coalesced access**: ensure fastest-varying thread dimension matches innermost array index.
- **Vertical diffusion cost scales with Nz**, not with Kz-field complexity.
  Plan 16b Commit 6 benchmarked `ImplicitVerticalDiffusion` at wurst L40S F32,
  cfl=0.4. `ConstantField`, `ProfileKzField` (`CuArray{FT, 1}` via `Adapt`),
  and `PreComputedKzField` (`CuArray{FT, 3}`) all cost essentially the same
  per step — the 2–6% spread between them is memory traffic, not
  `field_value` dispatch. What moves the needle is the column-serial
  Thomas solve: ~5% overhead at Nz=4, ~20% at Nz=32, ~75% at Nz=72 for
  Upwind Nt=10. Slopes advection is more expensive per sweep, so the
  same ~8 ms large-grid Thomas fraction is a smaller percentage (~40%
  at Nz=72). The 30% target in the plan is soft; large-grid use can
  still pay the cost. Future optimizations (documented in
  [artifacts/plan16/perf/SUMMARY.md](artifacts/plan16/perf/SUMMARY.md)):
  multi-tracer fusion inside the kernel (build coefficients once,
  back-substitute Nt times), shared-memory `w[k]`, persistent
  `w_scratch` reuse when Kz is stationary.

- **Measure, don't subtract (sync cost in particular)**. Plan 13 hypothesized
  that removing `synchronize(backend)` from Strang sweeps would give 20–40%
  GPU speedup. Direct CUDA-event measurement (`CUDA.@elapsed` wrapped inside
  `gpu_event_time` in [bench_strang_sweep.jl](scripts/benchmarks/bench_strang_sweep.jl)
  with `--events`) showed Δ host − cuda is only **~10–12 μs per
  `strang_split!`** on L40S F32, regardless of problem size (medium ~3 ms/step,
  large ~47 ms/step). The ~3 ms / ~47 ms is kernel arithmetic, not sync.
  When budgeting performance from removing an operation, time the operation
  directly with CUDA events or nsys rather than inferring from subtraction.
  Full write-up: [artifacts/plan13/perf/sync_thesis_report.md](artifacts/plan13/perf/sync_thesis_report.md).

---

## Plan execution rhythm

Accumulated pattern across plans 13-17. Every plan follows the same commit-0-plus-sequence shape.

### Commit 0: NOTES + baseline, no source changes

Every plan's Commit 0 does the following, with **no source changes**:

1. Create `docs/plans/<PLAN>/NOTES.md` with baseline section (parent commit,
   branch name, pre-existing failure count).
2. Run precondition verification (§4.1 of the plan doc: grep survey for
   existing infrastructure in the plan's problem domain, test suite sanity).
3. Capture baseline test summary to `artifacts/<plan>/baseline_test_summary.log`
   covering the test files plan work will touch.
4. Record baseline git hash to `artifacts/<plan>/baseline_commit.txt`.
5. Run existing-infrastructure survey; log results to
   `artifacts/<plan>/existing_*_survey.txt`. Flag scope revisions in NOTES if
   the survey reveals the greenfield assumption is wrong.
6. Memory compaction: update `MEMORY.md` "Current State" with the previous
   plan's completion note; add a `planNN_start.md` memo for this plan.

This is the **last clean commit** for rollback.

### Commit sequence is draft, not contract

Plan docs list expected commit sequences. Reality compresses, splits, reorders.
Plan 15: 8→6. Plan 16b: 8→9 (Commit 1 split into 1a/1b/1c). Plan 17 (2026-04):
Commit 2 compressed by migrating existing `SurfaceFluxSource` rather than
greenfield. Every NOTES.md must have a **§"Deviations from plan doc §4.4"**
section capturing every commit merge, split, or skip with reason. The plan
doc stays as original design intent; NOTES is the source of truth for what
shipped.

### Rollback point discipline

Commits must be individually revertable. If Commit N breaks, `git revert N`
should restore Commit-(N-1) state cleanly with tests passing. Anti-pattern:
a Commit 4 that depends on a fix in Commit 5 — reverting 4 alone leaves the
repo broken. Ship the fix in 4 or restructure commits.

### Retrospective sections are cumulative

Fill in NOTES.md's "Decisions beyond the plan", "Surprises", "Interface
validation findings", "Template usefulness for plans N+1" sections as
execution proceeds, **not all at once at the final commit**. Future plans
reading the NOTES want the narrative of discovery, not a polished summary.
Minor language editing at plan completion is fine.

### Measurement is the decision gate

Every refactor plan since 13 has been wrong about performance predictions in
one direction or another:

- Plan 13 predicted sync-removal would give a measurable win. Measurement
  showed sync was 0.02-0.4% of step time — prediction was an order of
  magnitude too optimistic.
- Plan 14 v3 predicted 0-10% GPU speedup from multi-tracer fusion. Measurement
  showed 10-270% at production Nt=30 — prediction was 5-10× too conservative.
- Plan 16b predicted ≤30% diffusion overhead at large grids. Measurement
  showed 65-76% at C180 Nt=10 — soft target exceeded, plan shipped anyway
  after re-framing the value proposition.

**Rule:** Commit 0's baseline measurements are the decision gate for whether
the plan's motivation is real, not a confirmation of a pre-registered
prediction. If measurement surprises, re-frame the plan's value proposition
based on what's actually true. Specifically:

- Launch overhead is <1% of production GPU time. Fusing to reduce launch
  count is the wrong optimization target.
- Bandwidth is ~95% of production GPU time. Fusing to reduce memory traffic
  IS the target. A 2-3× speedup in bandwidth-dominated kernels is plausible
  when sequential operators read/write the same data.

### Survey before greenfield

Plans 15, 16b, and 17 all assumed significant greenfield work and found
substantial existing infrastructure:

- Plan 15 expected "minimal ad-hoc decay" and found a 78-line
  `src/Operators/Chemistry/Chemistry.jl` with `AbstractChemistry`,
  `RadioactiveDecay`, `CompositeChemistry`, wired into `DrivenSimulation`.
- Plan 16b inherited `src_legacy/Diffusion/` (four implementations, ~800 lines)
  as starting point, not reference.
- Plan 17 found `SurfaceFluxSource` already live in `src/Models/DrivenSimulation.jl`
  with `_apply_surface_source!` applied inline post-transport — triggered a
  scope revision from greenfield to migration.

**Rule:** §4.1 of every plan includes a grep survey for existing infrastructure
in the plan's problem domain. Log results to
`artifacts/<plan>/existing_<domain>_survey.txt`. Revise scope before Commit 1
if the survey reveals the greenfield assumption is wrong.

Typical patterns for survey queries:

```bash
# Adjust keywords per plan domain
grep -rn "decay\|half_life\|exponential" src/ --include="*.jl"
grep -rn -i "diffus\|kz\|thomas\|tridiag" src/ --include="*.jl"
grep -rn -i "emission\|flux\|surface_flux\|deposition" src/ --include="*.jl"
grep -rn -i "convection\|tiedtke\|mass_flux\|entrainment" src/ --include="*.jl"
```

---

## Branch hygiene

### Stack plan branches, don't parallel them

Plans 14 → 15 → 16a → 16b → 17 each forked from the previous plan's tip.
This creates a linear chain:

```
main ← advection-unification ← slow-chemistry ← time-varying-fields ← vertical-diffusion ← surface-emissions
```

Each plan branch is self-contained and reviewable as a PR against the prior
plan's tip. Merge to `main` in order when review catches up.

**Rule:** Don't branch from `main` for a plan N when plans before N haven't
merged. Chain off the latest plan's tip instead. Keep history linear.

### Merge staging branches to `main` when the abstraction stabilizes

Don't merge every plan to `main` immediately — intermediate states can be
internally inconsistent. But don't wait for the entire refactor suite either —
`main` falls behind and misses real production value.

Guideline: merge when a natural grouping of plans reaches a stable API
boundary. Plans 11-14 (advection refactor) = one logical unit. Plans 15-16b
(operator abstractions) = another. Plan 17 (emissions) + palindrome updates
+ plan 18 (convection) = another.

---

## Julia / language gotchas

Patterns that have cost real debugging time here. Canonical examples live in
the cited source files.

- **No `Real → FT` outer constructor.** Julia's synthesized inner constructor
  already calls `convert(FT, x)`. Adding `Foo{FT}(x::Real) where FT = ...` as
  an outer only causes ambiguity.
- **Parametric kwarg defaults evaluate at module scope.**
  `f(; params = Foo{FT}()) where FT` fails with `UndefVarError: FT`. Use
  `params = nothing` + `something(params, Foo{FT}())` inside the body — see
  [`DerivedKzField`](src/State/Fields/DerivedKzField.jl).
- **Array-holding fields must be parametric on the array type** so
  `Adapt.adapt_structure` can swap backing at launch. `V <: AbstractVector{FT}`
  not `::Vector{FT}` — see [`ProfileKzField`](src/State/Fields/ProfileKzField.jl).
- **`Ref{Int}` breaks GPU kernels.** Host-side pointers error inside a KA
  kernel. Use a 1-element `Vector{Int}` that `Adapt` converts to a 1-element
  device array — see
  [`StepwiseField.current_window`](src/State/Fields/StepwiseField.jl).
- **`Adapt.adapt_structure` cannot run validating inner constructors on
  device arrays.** Ship a `Val(:unchecked)` inner-constructor path for the
  Adapt round-trip — see [`StepwiseField`](src/State/Fields/StepwiseField.jl).
- **CPU/GPU dispatch on `parent(arr) isa Array`, not `arr isa Array`.**
  Post-plan-14 tracer views (`selectdim(tracers_raw, 4, i)`) are `SubArray`s
  that fail `isa Array` and misroute to GPU. Use
  `parent(arr) isa Array` or `KernelAbstractions.get_backend(arr)`.

---

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

20+ test files, 5000+ tests. Key test categories:
- Per-module physics tests (advection, diffusion, chemistry, surface flux)
- Mass conservation (total tracer mass preserved to machine precision F64 /
  ULP F32)
- Adjoint identity (dot-product test: machine precision for linear operators)
- TM5 reference validation
- Integration tests (full DrivenSimulation run loop)

### Testing discipline (accumulated across plans 14-17)

**Test contract: observe through the accessor API.** Tests observe
post-operator state via `get_tracer(state, name)` or `state.tracers.name`,
never through input arrays passed at construction:

```julia
# GOOD
state = CellState(air_mass; CO2 = zeros(FT, Nx, Ny, Nz))
apply!(state, meteo, grid, op, dt; workspace)
@test get_tracer(state, :CO2) ≈ expected   # accessor API

# BAD (stale after plan 14's 4D tracers_raw refactor)
rm_CO2 = zeros(FT, Nx, Ny, Nz)
state  = CellState(air_mass; CO2 = rm_CO2)
apply!(state, meteo, grid, op, dt; workspace)
@test rm_CO2 ≈ expected     # rm_CO2 storage is separate from state.tracers.CO2
```

**Default kwargs must be bit-exact to the explicit-no-op path.** When adding
a kwarg with a `No<Something>()` default to a hot-path function (e.g.
`apply!(...; diffusion_op = NoDiffusion())`, `strang_split_mt!(...; emissions_op = NoSurfaceFlux())`),
ship an explicit `==` regression test comparing the default path to the
explicit-no-op path. Not `≈` — `==`. If the default branch does any
floating-point work, some caller without the kwarg sees subtly different
numerics. Plan 16b Commit 4's first testset is the pattern.

**Baseline failure count is invariant.** 77 pre-existing test failures across
three files, stable since plan 12:

- `test_basis_explicit_core.jl`: 2 (metadata-only CubedSphere API)
- `test_structured_mesh_metadata.jl`: 3 (panel conventions)
- `test_poisson_balance.jl`: 72 (mirror_sign inflow/outflow)

Every plan preserves this count. New failures → STOP, revert, investigate.
Plans that would fix any of these need a separate scope commitment. Baseline
is captured in `artifacts/<plan>/baseline_test_summary.log` at Commit 0 and
compared against after every subsequent commit.

## CI enforcement (added plan 21 Phase 6)

Three static/semantic gates run as part of the core test suite
(`julia --project=. test/runtests.jl`). All three are **hard gates** —
failures block merge.

### 1. `test/test_aqua.jl` — package health

Aqua.jl checks: ambiguities, unbound args, undefined exports,
project extras, stale deps, deps_compat, type piracies. All seven
must pass; `persistent_tasks` is disabled because GPU-extension
async precompile tasks trip false positives.

**When Aqua fails:** investigate. If it surfaces a real bug, fix it.
If it flags a well-understood false positive for a documented
reason, narrow the individual check rather than globally disabling.
Do not silence the test entirely.

### 2. `test/test_jet.jl` — type-inference snapshot

JET.jl runs on hot-path modules (`Operators`, `State`, `Models`,
`Grids`) with `target_modules` filter. The report count must be
≤ a documented baseline (`JET_HOT_PATH_BASELINE`, currently 117).

**Two categories of expected reports** — see `test_jet.jl` docstring
and `artifacts/plan21/jet_baseline.txt`:

- `KernelAbstractions.Kernel` `kwcall` dispatch (~116 reports) — a
  well-known JET ↔ KA false positive from `kernel(args...; ndrange=...)`.
- Parametric `@kwdef` zero-arg constructor (1 report).

**When JET fails:** the test prints the new reports. If they're
genuine bugs, fix. If they're additional KA kwcall noise or other
documented patterns, update `JET_HOT_PATH_BASELINE` with a comment
citing the reason. Escape hatch: set `ATMOSTRANSPORT_JET_ADVISORY=1`
to demote breaches to warnings during local dev on intermediate
refactors.

### 3. `test/test_readme_current.jl` — README freshness

For each directory listed in `README_DIRS`, every `.jl` file must
appear as a substring in that directory's `README.md`. Runs in
~100 ms.

**When a new .jl file is added to a tracked directory:**
update the corresponding `README.md` File Map section. Or, if the
file has a documented reason not to appear (compatibility shim,
generated code), add it to `EXCLUDE_FILES` in
`test/test_readme_current.jl` with a comment.

### What triggers each gate

| Change | Aqua | JET | README |
|---|---|---|---|
| New `.jl` file in tracked dir | possible | possible | **YES** |
| New method in hot-path module | no | possible | no |
| New runtime dep | **YES** | no | no |
| New test-only dep | no (if [extras]) | no | no |
| New qualified-name export / ambiguous method | **YES** | no | no |

---

## What NOT to do (accumulated anti-patterns)

- **Do NOT** estimate GPU perf from launch-overhead math. Bandwidth dominates.
- **Do NOT** add `Real → FT` coercing outer constructors for parametric structs — Julia already handles it.
- **Do NOT** write `params = SomeParametric{FT}()` as a kwarg default in a `where FT` method — evaluates at module scope.
- **Do NOT** cache caller arrays in test helpers expecting `===` identity after operator — 4D storage separates input.
- **Do NOT** dispatch backend on `arr isa Array` when views are possible.
- **Do NOT** pre-factor tridiagonal coefficients if adjoint port is future work.
- **Do NOT** assume greenfield — survey the repo first (§4.1).
- **Do NOT** merge plan branches to `main` before the logical grouping is stable.
- **Do NOT** skip Commit 0 measurement because the plan seems low-risk.
- **Do NOT** write retrospective sections only at plan completion — fill in during.
- **Do NOT** use `Base.RefValue` for mutable scalars in kernel-facing structs — use 1-elem arrays.
- **Do NOT** call validating inner constructors from `Adapt.adapt_structure` — ship a `Val(:unchecked)` path.

---

## Reference: GEOS-Chem comparison notes

See `docs/reference/TRANSPORT_COMPARISON.md` for detailed comparison of
AtmosTransport vs TM5 vs GEOS-Chem/GCHP algorithms, including the phased
implementation blueprint for unified ERA5/GEOS transport. Parts might be obsolete now...

## Data Layout
Follow the data guide in `docs/reference/DATA_LAYOUT.md`
