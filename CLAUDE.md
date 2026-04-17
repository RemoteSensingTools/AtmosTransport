# AtmosTransport.jl — Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Solves tracer
continuity on latitude-longitude (ERA5) and cubed-sphere (GEOS-FP C720, GEOS-IT C180)
grids with mass-conserving advection, Tiedtke/RAS convection, and boundary-layer diffusion.
Oceananigans-inspired modular architecture; KernelAbstractions.jl for GPU portability.

# Rule 0: TEST EVERY ASSUMPTION WITH A PROBE BEFORE BUILDING ON IT.
#
# The recurring failure mode in this project is: correct high-level reasoning
# followed by sloppy implementation that introduces trivial bugs (wrong index,
# wrong sign, wrong dimension order, wrong convention). These bugs then cascade
# into hours of debugging or silently wrong results.
#
# BEFORE writing any function that transforms data (spectral synthesis, vertical
# remap, Poisson balance, IC loading, flux scaling), validate the INPUTS and
# OUTPUTS with a short MCP probe:
#
#   1. Print the shape, dtype, and a few representative values of every array
#      BEFORE the function processes them.
#   2. After the function runs, print the output at a known physical location
#      (e.g. ps at Tibet should be ~55 kPa, ps at equatorial ocean ~101 kPa)
#      and verify it matches physical reality.
#   3. Compare against an independent reference (RG vs LL at the same point,
#      Catrine source vs model-loaded IC, ERA5 GRIB vs spectral-synthesized).
#
# Examples of bugs this rule would have caught in < 30 seconds:
#   - LL 180° longitude shift (spectral_synthesis.jl center_shift=dlon/2
#     instead of deg2rad(λᶜ[1])): probe `ps[57, 25]` at (1.875°, 31.875°)
#     would show 101232 Pa instead of expected ~88500 Pa. Caught by the user
#     pushing back on "LL vs RG IC looks different", not by me.
#   - RG Poisson balance false convergence: probe `max(|L*psi - rhs|)` after
#     the CG solve, not just the CG's internal residual metric. Three
#     iterations of bugs (wrong degree, wrong projection, mean-zero on
#     non-singular L) before it worked.
#   - NetCDF dim-order confusion: `var[:, :, 0]` on a `(lon, lat, time)`
#     file gives `(lon, lat)` at time=0, but on `(time, lat, lon)` gives
#     `(time, lat)` at lon=0. One `print(var.dimensions, var.shape)` would
#     have shown the real layout.
#   - Catrine surface level: `co2[k=0]` could be TOA or surface depending
#     on the ap/bp convention. One `print(ap[1], bp[1])` reveals p[0]=ps
#     → surface.
#
# The cost of a probe is 5 seconds and 2 lines of code. The cost of NOT
# probing is hours of debugging, false conclusions written to MEMORY.md,
# and user trust erosion. When in doubt, probe first, code second.

# Rule 1: Never speculate without evidence, check your own intuition, it was often wrong

# Rule 1b: Assume bugs first. If a result looks unphysical, it IS a bug until proven
# otherwise with quantitative evidence. Do NOT rationalize wrong numbers. CO2 does not
# double at the surface in 6 hours. Wind speeds are 5-10 m/s, not 50. Mass doesn't
# appear from nowhere. When you see a suspicious result, say "this looks wrong, let me
# find the bug" — not "this could be explained by X."

# Rule 1c: NEVER write a hypothesis to MEMORY.md (or any persistent doc) labeled as
# fact. Persistent memory is read by future agents who treat it as ground truth.
# If you are pattern-matching, say "consistent with X" or "candidate: X" — never
# "real X", "confirmed X", or "TM5 would also do X". Especially never claim that
# an unexplained drift, error, or artifact is "real / physical / expected" without
# (a) a back-of-envelope number that quantitatively matches, AND (b) a citation
# (paper, code line, or independent run). Pattern-matching the timing of two spikes
# to "looks like a 12h cycle" is NOT evidence — it's a guess. If you do not have
# both (a) and (b), the right entry is "drift -X% over Y, source unknown". Saving
# a wrong "explanation" is worse than saving "I don't know" because it ends the
# investigation prematurely.

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

# Rule 3: This is a production-level codebase — prefer long-term solutions
#
# Every piece of code added here will be maintained, extended, and relied upon
# for years. Prefer clean, mathematically correct, future-proof implementations
# over quick hacks that "kind of work" but are neither rigorous nor elegant.
# Specifically:
#   - If two approaches exist and one is more accurate / more general / cleaner
#     but requires more work, choose it. The extra effort pays for itself in
#     reduced debugging and easier extension later.
#   - No ad-hoc limiters, clamps, or fudge factors without a first-principles
#     derivation and a comment explaining WHY the limiter is needed.
#   - No "temporary" workarounds that silently become permanent. If a shortcut
#     is truly needed for iteration speed, mark it with `# HACK:` and a TODO
#     with the clean replacement path.
#   - Code should be self-documenting with clear docstrings, inline convention
#     comments, and unit annotations on physical quantities. A newcomer reading
#     any function should understand the math, the sign conventions, and the
#     index layout without consulting external docs.

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
AbstractAdvectionScheme      → UpwindScheme, SlopesScheme{L}, PPMScheme{L}
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

10. **ERA5 LL binaries MUST be made by `preprocess_spectral_v4_binary.jl` or
    `preprocess_era5_daily.jl`** — both call `recompute_cm_from_divergence!`
    to rebuild cm from continuity using the merged am/bm. The OBSOLETE
    `convert_merged_massflux_to_binary.jl` PICKS native cm at merged
    interfaces and smears the residual via `correct_cm_residual!`, producing
    a cm that does NOT satisfy local continuity (89.7% of cells violated
    in the broken Dec 2021 binary). Symptoms: polar Y nloop hits max_nloop,
    polar cells go negative, cumulative drainage. **Foolproof check (added
    2026-04-06)**: `MassFluxBinaryReader` runs a `_verify_cm_continuity` pass
    on window 1 at driver construction time. Errors LOUDLY if the binary's
    cm is inconsistent. Disable with `ENV["ATMOSTR_NO_CM_CHECK"]="1"`.
    Provenance fields (script_path, script_mtime, git_commit, git_dirty)
    in v4 headers also trigger a stale-binary warning if the source has
    moved on. Disable with `ENV["ATMOSTR_NO_STALE_CHECK"]="1"`.

11. **`mass_fixer = true` is required for the ERA5 LL F64 debug test to
    run to completion.** Pole-adjacent stratospheric cells
    (j ∈ {1, 2, Ny-1, Ny}) have |bm|/m ≈ 0.30 per face from the spectral
    preprocessor. With `mass_fixer = false`, cumulative drainage over a
    window of 4 substeps × 6 sweeps (X-Y-Z-Z-Y-X) exceeds cell mass at
    those cells; the local nloop refinement hits its max and the run
    aborts (regression test:
    `config/runs/era5_f64_debug_moist_v4_nofix.toml`). This is a LOCAL
    cell-stability issue and is independent of the global mass drift fix
    in invariant 12 below — both are needed for ERA5 LL.

    **OPEN: whether this matches TM5 r1112 behavior is NOT verified.**
    A previous CLAUDE.md draft asserted "TM5 r1112 effectively does
    mass-fixing via m = (at + bt × ps) × area / g each substep". That
    was a hypothesis, not a checked claim — do not propagate as fact.
    Verifying it requires reading TM5 r1112's actual m-evolution path
    (advect_tools.F90 / dynam0 / Setup_MassFlow) and possibly running
    TM5 on the same data. Until that's done, the honest position is
    "we need mass_fixer=true to avoid polar drainage; whether TM5 needs
    it too is unknown".

12. **Global mean ps is pinned in the v4 spectral preprocessor (TM5
    `Match('area-aver')` equivalent).** ERA5's 4DVar analysis is not
    mass-conserving; raw ⟨ps⟩ drifts ~10⁻⁴/day, which translates
    directly into a Σm drift in the binary because
    `Σm = const + (A_Earth/g)·⟨sp⟩_area` (with `Σ_k b_k = 1`). To
    eliminate this, `preprocess_spectral_v4_binary.jl` applies a
    uniform additive shift to the gridded `sp` immediately after
    `sp = exp(LNSP_grid)` so that `⟨sp⟩_area` corresponds to a fixed
    dry-air mass target (Trenberth & Smith 2005:
    `M_dry = 5.1352e18 kg → ⟨ps_dry⟩ ≈ 98726 Pa`, converted to total
    via `target_total = target_dry / (1 - ⟨qv⟩_global ≈ 0.00247)`).
    The implementation mirrors TM5 cy3-4dvar `meteo.F90:1361-1374`
    which calls `Match('area-aver', sp_region0=p_global=98500 Pa)`
    via `grid_type_ll.F90:1147-1155`.

    **Verified 2026-04-07**: 24h F64 LL test with non-uniform ps drift
    in raw ERA5 → after fix, Σm drift drops from -8.31e-04% to
    +5.88e-09% (~140,000× improvement, F32 quantization noise floor).
    Per-window offsets logged in the binary header
    (`ps_offsets_pa_per_window`). Disable via `[mass_fix] enable=false`
    in the preprocessor TOML for diagnostic purposes.

    **What this does NOT fix**: local polar drainage (invariant 11).
    The shift is uniform globally and only changes local `m` by
    `b_k · Δps` (~0.4% at the surface, less aloft). It does not
    change `|bm|/m` enough at the troubled polar cells to remove the
    need for `mass_fixer = true` at runtime.

13. **Every binary preprocessor that diagnoses `cm` from continuity
    MUST apply a Poisson mass-flux balance to the horizontal fluxes
    first.** Without balance, the raw spectral-synthesis fluxes have
    divergence residuals of ~10¹² kg per cell (for ERA5 at T47),
    which then integrate through the continuity diagnosis into
    unphysically large vertical mass fluxes `cm`. Symptoms: the
    face-indexed runtime CFL pilot hits `max_n_sub=4096` in window 1,
    or (on older schemes with limiters) silently unstable transport
    that drifts over time.

    **Reference implementations**:
    * **LL path**: `mass_support.jl:balance_mass_fluxes!` — 2D FFT on
      the circulant Laplacian (`fac = 2*(cos(2π(i-1)/Nx) + cos(2π(j-1)/Ny) - 2)`),
      called from `binary_pipeline.jl:apply_poisson_balance!`. Exact
      to machine precision up to roundoff.
    * **RG path**: `reduced_transport_helpers.jl:balance_reduced_horizontal_fluxes!`
      — Jacobi-Preconditioned Conjugate Gradient on the graph Laplacian
      with **interior-only face degrees** (boundary-stub pole-cap faces
      NOT counted, because the correction operator only touches interior
      fluxes). The graph Laplacian is singular (constant null space), so
      `rhs`, `r`, `p` are kept in range(L) by subtracting their means
      every CG iteration. Called from
      `preprocess_era5_reduced_gaussian_transport_binary_v2.jl` via
      `apply_reduced_poisson_balance!`.

    **Validation** (2026-04-11): at matched 4608 cells,
    post-balance worst `|cm|/m` should be ~1e-14 (F64 machine
    precision). LL 96×48 reference: 5.3e-15. Synthetic RG N24:
    1.3e-14. RG N320 production binaries (pre-fix): 0.77 — these
    were quietly broken and silently unstable.

    * **CS path**: `cs_global_poisson_balance.jl:balance_cs_global_mass_fluxes!`
      — JPCG on the global 6-panel graph Laplacian with 12Nc² faces
      (interior + cross-panel). All cells have degree 4 (no boundary
      stubs on a closed sphere). Cross-panel faces use canonical/mirror
      pairs with same-sign convention (outgoing edge ↔ incoming edge).
      Replaces the per-panel FFT that wrongly treated panels as
      doubly-periodic. Called from
      `preprocess_era5_cs_conservative_v2.jl`. Validated: C24×34 in
      0.13s, 224 CG iterations, projected residual 8.5e-14.

    **When adding a new grid-type preprocessor**:
    verify that the post-balance `max(|cm|/m)` probe matches the LL
    reference before declaring the binary working. Never trust
    "the binary compiles and exit-code-0 runs" without this check.

    Two CG-solver gotchas discovered during the fix:
    a. `cell_face_degree` must count ONLY interior faces. Counting
       pole-cap boundary stubs in the diagonal produces a
       `n_stubs * psi[c]` residual at pole cells that the solver
       can't see.
    b. Do NOT project `r` to mean-zero inside CG if the Laplacian is
       non-singular (e.g., Dirichlet BCs). DO project if it is
       singular (the range-of-L condition). Getting this wrong gives
       a false-positive convergence where CG reports tight L2
       residuals but the actual max|L*psi - rhs| is 13 orders of
       magnitude larger.

14. **Dry-basis is the default and required contract for all transport
    binaries (Invariant 14).** Runtime transport never performs moist-to-dry
    conversion. All carrier-mass conversion and continuity closure are
    completed during preprocessing. The canonical chain is:

    ```
    native spectral/gridpoint synthesis (moist)
      → load native QV from thermo NetCDF
      → convert m, am, bm to dry basis: field *= (1 - qv)
      → merge native levels to transport levels
      → Poisson-balance on dry fluxes
      → diagnose dry vertical fluxes (cm)
      → write dry-basis binary (header: mass_basis = :dry)
      → runtime transport reads dry fields directly
      → tracer mass initialized as rm = vmr × m_dry
    ```

    All three grid paths (LL, CS, RG) support dry preprocessing:
    - **LL/CS**: `apply_dry_basis_native!` in `mass_support.jl` on 3D arrays
    - **RG**: `apply_dry_basis_reduced!` in `reduced_transport_helpers.jl`
      with bilinear QV interpolation from LL thermo grid to RG cells

    The `DryFluxBuilder` runtime converter (`src/MetDrivers/ERA5/DryFluxBuilder.jl`)
    is retained for backward compatibility with old moist-basis binaries only.
    Binary headers without `mass_basis` default to `:moist` with a warning.

    **Config**: all preprocessing TOMLs must set `mass_basis = "dry"` and
    `thermo_dir` pointing to ERA5 thermo NetCDF files with hourly `q`.
    The default in `resolve_mass_basis` is `:dry`.

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

See `docs/reference/TRANSPORT_COMPARISON.md` for detailed comparison of
AtmosTransport vs TM5 vs GEOS-Chem/GCHP algorithms, including the phased
implementation blueprint for unified ERA5/GEOS transport. Parts might be obsolete now...

## Data Layout
Follow the data guide in `docs/reference/DATA_LAYOUT.md`
