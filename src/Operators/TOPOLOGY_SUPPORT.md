# Topology Support Matrix

Canonical source of truth for which operators support which
topologies. Module READMEs reference this file rather than
duplicating coverage claims.

**Updated:** 2026-09-05. This table describes runtime dispatch support;
device-specific numerical and performance evidence is recorded separately in
the tests and benchmark reports.

## Topologies

| Symbol | State type | Storage | Flux state |
|--------|-----------|---------|------------|
| **LatLon** | `CellState` | rank-4 packed tracers | `StructuredFaceFluxState{Basis}` |
| **RG** (reduced Gaussian) | `CellState` | rank-3 face-indexed packed tracers | `FaceIndexedFluxState{Basis}` |
| **CS** (cubed sphere) | `CubedSphereState` | six rank-4 halo-padded tracer panels | `CubedSphereFaceFluxState{Basis}` |

Storage adapts to the backend (`Array`, `CuArray`, or another supported array
type); shape and basis contracts remain the same.

## Matrix

| Operator | LatLon | RG | CS |
|----------|:------:|:--:|:--:|
| `UpwindScheme` | ✅ | ✅ | ✅ |
| `SlopesScheme` / `PPMScheme` | ✅ | ❌ | ✅ |
| `LinRoodPPMScheme` (ORD=5/7) | ❌ | ❌ | ✅ |
| `NoAdvection` | ✅ | ✅ | ✅ |
| `ImplicitVerticalDiffusion` | ✅ | ✅ | ✅ |
| `SurfaceFluxOperator` | ✅ | ✅ | ✅ |
| `CMFMCConvection` | ✅ | ✅ | ✅ |
| `TM5Convection` | ✅ | ✅ | ✅ |
| `CMFMCMatrixConvection` | ✅ | ✅ | ✅ |
| `ExponentialDecay` / `CompositeChemistry` | ✅ | ✅ | ✅ |

✅ = dedicated `apply!` or `apply_*!` dispatch exists, tested and live through `TransportModel.step!`.
❌ = unsupported combination, rejected during configuration or dispatch.

`NoAdvection` supports diffusion-only runs; combining it with surface emissions
is currently rejected. Matrix convection's collaborative path requires Float32
GPU storage and an effective depth of at most 85 levels. Its six-slot tracer
buffer is reused across batches and does not limit the total tracer count.

## Evidence anchors

For each ✅ combination, the authoritative dispatch method:

### Advection

- **LatLon** — rank-4 Strang palindrome `X→Y→Z→V(dt)→Z→Y→X` in
  [`Advection/StrangSplitting.jl`](Advection/StrangSplitting.jl)
- **RG** — face-indexed `H→V(dt)→H` in the same file
- **CS** — panel-oriented packed-tracer
  [`strang_split_cs_mt!`](Advection/CubedSphereStrang.jl) for split-sweep
  schemes, with [`strang_split_cs!`](Advection/CubedSphereStrang.jl) retained
  as the scalar reference and Lin-Rood support path

### Diffusion

Four valid `apply_vertical_diffusion!` dispatches + three error-
branch methods in [`Diffusion/operators.jl`](Diffusion/operators.jl):

- rank-4 (LatLon)
- rank-3 (RG face-indexed, multi-tracer)
- rank-2 (RG face-indexed, single-tracer)
- `NTuple{6, Array{FT, 4}}` (CS packed production path)
- `NTuple{6, Array{FT, 3}}` (CS scalar compatibility/reference path)

### Surface flux

- [`SurfaceFlux/operators.jl`](SurfaceFlux/operators.jl) —
  topology-dispatched `apply!` for all three flux-state types
- Kernels live in [`SurfaceFlux/surface_flux_kernels.jl`](SurfaceFlux/surface_flux_kernels.jl)

### Convection (CMFMC)

Three valid `apply!` methods + one rejection in
[`Convection/CMFMCConvection.jl`](Convection/CMFMCConvection.jl):

- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:LatLonMesh}, ::CMFMCConvection, dt)`
- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:ReducedGaussianMesh}, ::CMFMCConvection, dt)`
- `apply!(::CubedSphereState, ::ConvectionForcing, ::AtmosGrid{<:CubedSphereMesh}, ::CMFMCConvection, dt)`
- A fourth dispatch rejects face-indexed state on non-RG grids to
  catch configuration mistakes.

### Matrix convection (TM5 and CMFMC-derived)

- [`Convection/TM5Convection.jl`](Convection/TM5Convection.jl) provides the
  LL/RG/CS array and state dispatches for both legacy and collaborative solves.
- [`Convection/CMFMCMatrixConvection.jl`](Convection/CMFMCMatrixConvection.jl)
  derives cached rates from CMFMC/DTRAIN and delegates to those same solvers.
- [`Convection/tm5_kernels.jl`](Convection/tm5_kernels.jl) contains the three
  collaborative kernels; [`Convection/README.md`](Convection/README.md)
  describes their depth, tracer, and validation contracts.

### Chemistry

Three valid `apply!` dispatches in
[`Chemistry/Chemistry.jl`](Chemistry/Chemistry.jl) per state type:

- `apply!(::CellState, ..., ::NoChemistry, dt)`
- `apply!(::CellState, ..., ::ExponentialDecay, dt)`
- `apply!(::CellState, ..., ::CompositeChemistry, dt)`

And three corresponding `CubedSphereState` dispatches (plan 21
follow-up). CS chemistry loops over the six panels and launches
the same rank-agnostic decay kernel per panel.

## Known gaps

None at present. Plan 21's topology completion work has no remaining
documented operator × topology gaps.

## How to update this file

When topology support changes:

1. Update the matrix above.
2. Update the "Evidence anchors" list to point at the new `apply!`
   method.
3. Update the "Last verified" date.
4. In the corresponding submodule README, reference this file
   rather than restating coverage.

A CI test (plan 21 Phase 6 — not yet landed) will validate that
every ✅ claim maps to an actual `apply!` method and every ❌ (gap)
maps to either no method or a `throw(ArgumentError(...))`
rejection.
