# Agent Onboarding Guide

Quick-start orientation for AI agents working on this codebase.
Read this AFTER `CLAUDE.md` (project rules) and BEFORE diving into code.

## Directory Map

| Directory | Status | What it is |
|-----------|--------|------------|
| `src/` | **Active** | Basis-explicit transport core (promoted from `src_v2/`) |
| `src_legacy/` | Legacy | Original TM5-aligned transport |
| `docs/` | **Active** | Documentation for `src/` |
| `docs_legacy/` | Legacy | Documenter.jl site + session logs for `src_legacy/` |
| `scripts/` | Shared | Download, preprocessing, run, diagnostics |
| `test/` | **Active** | Tests for `src/` (promoted from `test_v2/`) |
| `config/` | Shared | TOML configs for runs + preprocessing |
| `deps/tm5*/` | Reference | TM5 Fortran sources (read-only ground truth) |

## Key Architectural Concepts (src/)

### Type hierarchy (multiple dispatch, no if/else)
```
AbstractAdvectionScheme
  AbstractConstantScheme  -> UpwindScheme
  AbstractLinearScheme    -> SlopesScheme{L <: AbstractLimiter}
  AbstractQuadraticScheme -> PPMScheme{L <: AbstractLimiter}

AbstractMassBasis
  DryBasis    -- dry-air mass fluxes
  MoistBasis  -- total-air mass fluxes

AbstractFaceFluxState{Basis}
  StructuredFaceFluxState{Basis}     -- LatLon (am, bm, cm arrays)
  FaceIndexedFluxState{Basis}        -- ReducedGaussian (hflux, cm)
```

### Preprocessing pipeline
```
ERA5 GRIB (spectral) --> preprocess script --> transport binary (.bin)
                                                    |
transport binary --> TransportBinaryReader --> TransportBinaryDriver
                                                    |
                                        build_dry_fluxes!() --> runtime
```

### Strang splitting
X -> Y -> Z -> Z -> Y -> X per substep. Double-buffered (Invariant 4).
CFL subcycling per direction. Mass conservation is exact (telescoping sum).

## What Does "Healthy" Look Like?

Use these reference values to sanity-check results:

| Metric | Healthy | Broken |
|--------|---------|--------|
| `max(\|cm\|/m)` post-balance | < 1e-12 | > 0.01 (missing Poisson balance) |
| Global mass drift (24h, F64) | < 1e-8 % | > 1e-4 % (missing ps pin) |
| X/Y CFL (ERA5 LL 0.5deg) | 15-40 | > 100 (wrong mass_flux_dt) |
| Z CFL (Poisson-balanced) | < 1 | > 10 (unbalanced cm) |
| Surface CO2 range (Catrine IC) | 380-500 ppm | > 600 or < 300 (transport bug) |
| Column mean CO2 | ~411 ppm (uniform) | deviates > 1 ppm (mass issue) |
| Tracer mass drift (uniform IC) | < 1e-14 (F64) | > 1e-10 (kernel bug) |

## File Ownership (Multi-Agent Sessions)

When Claude and Codex work in parallel, ownership is tracked in `AGENT_CHAT.md`:

- **Claude** owns: CS files (CubedSphereMesh, HaloExchange, CubedSphereStrang),
  preprocessing scripts, visualization
- **Codex** owns: Runtime files (DrivenSimulation, StrangSplitting,
  run_transport_binary, TransportBinaryDriver), test/ files
- **Shared** (coordinate before editing): AtmosTransport.jl, Grids.jl,
  Operators.jl, FaceFluxState.jl

## Common Agent Pitfalls

1. **Don't claim "TM5-faithful"** without reading the F90 line-by-line
2. **Don't rationalize wrong numbers** -- if it looks unphysical, it IS a bug
3. **Probe before building** -- 5-second print check saves hours of debugging
4. **Don't write hypotheses to MEMORY.md as fact** -- use "consistent with X", never "confirmed X"
5. **One change at a time** -- test each change before stacking another
6. **Read the reference first** -- `deps/tm5*/base/src/` is ground truth

## Documentation Layers

1. **CLAUDE.md** -- Project rules, invariants, workflows. Always loaded.
2. **docs/AGENT_ONBOARDING.md** (this file) -- Quick orientation, reference values.
3. **docs/0x_*.md** -- Core contracts, runtime flow, binary format, quality gates.
4. **docs/reference/** -- Shared reference docs (architecture, data layout, formats).
5. **docs/memos/** -- Design memos and debugging analyses (read on-demand).
6. **docs_legacy/** -- Old Documenter.jl site and session logs for `src_legacy/`.

## Test Commands

```bash
# Run active test suite (src/)
julia --project=. test/test_basis_explicit_core.jl
julia --project=. test/test_advection_kernels.jl
julia --project=. test/test_driven_simulation.jl

# Preprocess ERA5 binary (Dec 1, 2021)
julia -t8 --project=. scripts/preprocessing/preprocess_spectral_v4_binary.jl \
    config/preprocessing/era5_spectral_v4_tropo34_dec2021.toml --day 2021-12-01

# Run 24h F64 stress test
ATMOSTRANSPORT_DEBUG_SWEEPS=1 julia --threads=2 --project=. \
    scripts/run.jl config/runs/era5_f64_debug_moist_v4_24h.toml
```

## Server Environment

| Server | GPUs | F64? | Use for |
|--------|------|------|---------|
| wurst | 2x L40S | No (F32 only) | CPU runs + GPU F32 |
| curry (ssh curry) | 2x A100-40GB | Yes | GPU F64, heavy compute |

Rule: **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`) on both servers.
Home/data shared via NFS. Only `/tmp` is local.
