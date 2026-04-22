# Scope And Status

## What `docs/` is

Technical documentation for `src/`, the basis-explicit transport core.
The documentation style stays:

- explicit
- technical
- reviewable by non-Julia users
- close to the actual code paths

## What `src/` is

Offline atmospheric transport runtime with:

- **Multi-topology**: structured LatLon, face-indexed reduced Gaussian,
  panel-native cubed sphere (all three runtime-live after plan 22A–D)
- **Multi-backend**: CPU and GPU via KernelAbstractions.jl
- **Operator suite**: advection, vertical diffusion, surface flux,
  convection, chemistry — composed via symmetric Strang splitting
- **Basis-explicit mass-flux advection**: `DryBasis` / `MoistBasis`
  tags in the type domain; advection kernels do not decide dry vs
  moist at runtime
- **Extensible by multiple dispatch**: new operators, grids, and data
  sources plug in without modifying existing code paths

## What ships today

- **Advection**: live on LatLon, RG, CS (see `Operators/Advection/`)
- **Diffusion**: live on LatLon, RG, CS (see `Operators/Diffusion/`)
- **Surface flux**: live on LatLon, RG, CS (see `Operators/SurfaceFlux/`)
- **Convection (CMFMC)**: live on LatLon, RG, CS (plan 22D;
  `Operators/Convection/CMFMCConvection.jl`)
- **Chemistry** (`ExponentialDecay`, `CompositeChemistry`): live on
  `CellState` (LatLon, RG) and `CubedSphereState` topologies
  (CS dispatch shipped in commit `bcd4fea`)
- **Met drivers**: ERA5 spectral, GEOS-FP C720, GEOS-IT C180,
  cubed-sphere binary
- **Adjoint**: forward operators ported; hand-coded discrete adjoint
  planned as plan 19. Archival templates under
  [`resources/developer_notes/legacy_adjoint_templates/`](resources/developer_notes/legacy_adjoint_templates/)

Canonical operator × topology matrix:
[`src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md).

## What is intentionally deferred

- Plan 19 adjoint operator suite
- Plan 20 user-facing documentation overhaul (Documenter + Literate)
- Observation operators for 4D-Var

## Current phase

Topology completion (plans 22A–D) and stabilization (plan 21)
landed April 2026. Plan 23 (TM5 convection, in progress on branch
`convection`) adds the `TM5Convection` operator for ERA5, to ship
alongside the existing `CMFMCConvection` for GEOS-FP. After plan
23, focus shifts to plan 19 (adjoint) and plan 20 (user-facing
docs).

See [`plans/PLAN_HISTORY.md`](plans/PLAN_HISTORY.md) for the canonical
per-plan status.

---

*Last verified against `src/` on 2026-04-21 (plan 21 Phase 3a).*
