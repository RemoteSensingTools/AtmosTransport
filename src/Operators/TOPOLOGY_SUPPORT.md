# Topology Support Matrix

Canonical source of truth for which operators support which
topologies. Module READMEs reference this file rather than
duplicating coverage claims.

**Last verified:** 2026-04-21 (plan 21 Phase 5 + CS chemistry follow-up)

## Topologies

| Symbol | State type | Storage | Flux state |
|--------|-----------|---------|------------|
| **LatLon** | `CellState` | rank-4 `tracers_raw::Array{FT, 4}` | `StructuredFaceFluxState{Basis}` |
| **RG** (reduced Gaussian) | `CellState` | rank-3 face-indexed `tracers_raw::Array{FT, 3}` | `FaceIndexedFluxState{Basis}` |
| **CS** (cubed sphere) | `CubedSphereState` | `NTuple{6, Array{FT, 4}}` panel-native | `CubedSphereFaceFluxState{Basis}` |

## Matrix

| Operator | LatLon | RG | CS |
|----------|:------:|:--:|:--:|
| `UpwindScheme` / `SlopesScheme` / `PPMScheme` | ✅ | ✅ | ✅ |
| `ImplicitVerticalDiffusion` | ✅ | ✅ | ✅ |
| `SurfaceFluxOperator` | ✅ | ✅ | ✅ |
| `CMFMCConvection` | ✅ | ✅ | ✅ |
| `ExponentialDecay` / `CompositeChemistry` | ✅ | ✅ | ✅ |

✅ = dedicated `apply!` or `apply_*!` dispatch exists, tested and live through `TransportModel.step!`.
❌ = no dispatch; operator rejects or (for CS chemistry) is not yet wired.

## Evidence anchors

For each ✅ combination, the authoritative dispatch method:

### Advection

- **LatLon** — rank-4 Strang palindrome `X→Y→Z→V(dt)→Z→Y→X` in
  [`Advection/StrangSplitting.jl`](Advection/StrangSplitting.jl)
- **RG** — face-indexed `H→V(dt)→H` in the same file
- **CS** — panel-oriented
  [`strang_split_cs!`](Advection/CubedSphereStrang.jl) with halo
  exchanges between panels

### Diffusion

Four valid `apply_vertical_diffusion!` dispatches + three error-
branch methods in [`Diffusion/operators.jl`](Diffusion/operators.jl):

- rank-4 (LatLon)
- rank-3 (RG face-indexed, multi-tracer)
- rank-2 (RG face-indexed, single-tracer)
- `NTuple{6, Array{FT, 3}}` (CS)

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
