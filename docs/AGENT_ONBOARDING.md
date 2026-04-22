# Agent Onboarding Guide

Quick-start orientation for AI agents working on this codebase.
Read this AFTER `CLAUDE.md` (project rules) and BEFORE diving into code.

## Reading Order

1. `CLAUDE.md` at repo root — verification discipline, project rules, critical invariants
2. **This file** — directory map, type hierarchy, healthy reference values
3. [`../src/README.md`](../src/README.md) — runtime entry points
4. [`../src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md) — canonical operator × topology matrix
5. [`20_RUNTIME_FLOW.md`](20_RUNTIME_FLOW.md) — end-to-end `step!` walkthrough
6. [`plans/PLAN_HISTORY.md`](plans/PLAN_HISTORY.md) — what shipped, what's in flight

## Directory Map

| Directory | What it is |
|-----------|------------|
| `src/` | Basis-explicit transport core (the active module, `AtmosTransport.jl`) |
| `docs/` | Documentation for `src/` |
| `scripts/` | Download, preprocessing, run, diagnostics, visualization |
| `test/` | Tests for `src/` |
| `config/` | TOML configs for runs + preprocessing |
| `deps/tm5*/` | TM5 Fortran sources (read-only ground truth for verification) |
| `docs/resources/developer_notes/legacy_adjoint_templates/` | Archival adjoint templates for plan 19 |

All `*_legacy/` directories were removed in plan 21. Use `git show <commit>:<path>` to consult legacy content; the last commit on `convection` where `src_legacy/` / `scripts_legacy/` / `test_legacy/` existed is `ec2d2c0`.

## Key Architectural Concepts (src/)

### Three-topology runtime

Every operator dispatches on grid mesh type:

- `AtmosGrid{<:LatLonMesh}` — structured lat-lon, rank-4 `tracers_raw::Array{FT, 4}`
- `AtmosGrid{<:ReducedGaussianMesh}` — face-indexed RG, rank-3 `tracers_raw::Array{FT, 3}` (ncells, Nz, Nt)
- `AtmosGrid{<:CubedSphereMesh}` — panel-native CS, `tracers::NTuple{6, Array{FT, 4}}`

### State types

- `CellState` — LatLon or RG. Tracer storage is rank-4 (LatLon) or rank-3 face-indexed (RG); `TracerAccessor` wraps it so `state.tracers.CO2` works in either case
- `CubedSphereState` — CS. Tracer storage is `NTuple{6}` of per-panel rank-4 arrays with halos

### Type hierarchy (multiple dispatch)

```
AbstractAdvectionScheme
  UpwindScheme | SlopesScheme{L} | PPMScheme{L, ORD}

AbstractMassBasis
  DryBasis | MoistBasis

AbstractFaceFluxState{Basis}
  StructuredFaceFluxState{Basis}       -- LatLon (am, bm, cm arrays)
  FaceIndexedFluxState{Basis}          -- ReducedGaussian (hflux, cm)
  CubedSphereFaceFluxState{Basis}      -- CubedSphere (panel-native)

AbstractDiffusionOperator
  NoDiffusion | ImplicitVerticalDiffusion{FT, KzF}

AbstractSurfaceFluxOperator
  NoSurfaceFlux | SurfaceFluxOperator{FT, M}

AbstractConvectionOperator
  NoConvection | CMFMCConvection{...}

AbstractChemistryOperator
  NoChemistry | ExponentialDecay{...} | CompositeChemistry{...}

AbstractTimeVaryingField{FT, N}
  ConstantField | ProfileKzField | PreComputedKzField | DerivedKzField | StepwiseField
```

### Preprocessing pipelines

All three grid topologies have dedicated preprocessors under `scripts/preprocessing/`:

- **LatLon**: ERA5 spectral (VO/D/LNSP) → `preprocess_spectral_v4_binary.jl` → transport binary
- **Reduced Gaussian**: ERA5 spectral → `preprocess_era5_reduced_gaussian_transport_binary_v2.jl`
- **Cubed Sphere**: LL ERA5 fluxes → `preprocess_era5_cs_conservative_v2.jl` (regrid to panels)

All three apply a Poisson mass-flux balance before diagnosing `cm` from continuity (Invariant 13). All three default to dry-basis output (Invariant 14).

### Strang splitting per topology

- **LatLon**: X → Y → Z → V(dt) → Z → Y → X palindrome; double-buffered (Invariant 4). `strang_split!` / `strang_split_mt!` dispatch on rank-4 `tracers_raw`.
- **Reduced Gaussian**: face-indexed H-V-V-H (no separate Y sweep); `@atomic` scatter kernel.
- **Cubed Sphere**: panel-oriented `strang_split_cs!` with halo exchanges between panels (FV3-style).

See [`../src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md) for the authoritative per-operator dispatch map.

### Operator composition

`TransportModel.step!` executes three blocks in order:

1. **Transport block**: advection, with diffusion and surface flux embedded at the Strang midpoint
2. **Convection block**: `CMFMCConvection` on all three topologies (plan 22D)
3. **Chemistry block**: `ExponentialDecay` / `CompositeChemistry` on `CellState`; CS chemistry is the one documented gap

`NoDiffusion` / `NoSurfaceFlux` / `NoConvection` / `NoChemistry` are compile-time dead branches — default paths are bit-exact to their absent-operator equivalents.

## What does "healthy" look like?

Reference values to sanity-check results:

| Metric | Healthy | Broken |
|--------|---------|--------|
| `max(\|cm\|/m)` post-balance | < 1e-12 | > 0.01 (missing Poisson balance) |
| Global mass drift (24h, F64) | < 1e-8 % | > 1e-4 % (missing ps pin) |
| X/Y CFL (ERA5 LL 0.5°) | 15–40 | > 100 (wrong `mass_flux_dt`) |
| Z CFL (Poisson-balanced) | < 1 | > 10 (unbalanced cm) |
| Surface CO2 range (Catrine IC) | 380–500 ppm | > 600 or < 300 (transport bug) |
| Column mean CO2 | ~411 ppm (uniform) | deviates > 1 ppm (mass issue) |
| Tracer mass drift (uniform IC) | < 1e-14 (F64) | > 1e-10 (kernel bug) |

## Common agent pitfalls

1. **Don't claim "TM5-faithful"** without reading the F90 line-by-line
2. **Don't rationalize wrong numbers** — if it looks unphysical, it IS a bug
3. **Probe before building** — 5-second print check saves hours of debugging (Rule 0 in `CLAUDE.md`)
4. **Don't write hypotheses to MEMORY.md as fact** — use "consistent with X", never "confirmed X"
5. **One change at a time** — test each change before stacking another
6. **Read the reference first** — `deps/tm5*/base/src/` is ground truth

## Documentation layers

1. `CLAUDE.md` — project rules, invariants, workflows. Always loaded.
2. `docs/AGENT_ONBOARDING.md` (this file) — quick orientation, reference values.
3. `docs/0x_*.md` — core contracts, runtime flow, binary format, quality gates.
4. `docs/plans/PLAN_HISTORY.md` — canonical per-plan manifest.
5. `src/Operators/TOPOLOGY_SUPPORT.md` — canonical per-operator dispatch map.
6. `docs/reference/` — shared reference docs (architecture, data layout, formats).
7. `docs/memos/` — design memos and debugging analyses (read on-demand).
8. `docs/resources/` — archival material (bug archive, developer notes, adjoint templates).

## Test commands

```bash
# Core test suite (no external data)
julia --project=. test/runtests.jl

# Including real-data tests (requires preprocessed binaries)
julia --project=. test/runtests.jl --all

# Targeted single-file tests (no test-only deps)
julia --project=. test/test_basis_explicit_core.jl
julia --project=. test/test_advection_kernels.jl
julia --project=. test/test_driven_simulation.jl
julia --project=. test/test_transport_model_convection.jl
julia --project=. test/test_cubed_sphere_runtime.jl

# Targeted tests that need test-only deps (Aqua / JET):
# first-time setup instantiates test/Project.toml
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test test/test_aqua.jl
julia --project=test test/test_jet.jl
julia --project=test test/test_readme_current.jl

# Preprocess ERA5 binary (Dec 1, 2021)
julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
    config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml --day 2021-12-01

# Run 24h F64 stress test
ATMOSTRANSPORT_DEBUG_SWEEPS=1 julia --threads=2 --project=. \
    scripts/run.jl config/runs/era5_f64_debug_moist_v4_24h.toml
```

## Server environment

| Server | GPUs | F64? | Use for |
|--------|------|------|---------|
| wurst | 2× L40S | No (F32 only) | CPU runs + GPU F32 |
| curry (ssh curry) | 2× A100-40GB | Yes | GPU F64, heavy compute |

Rule: **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`) on both servers.
Home/data shared via NFS. Only `/tmp` is local.
