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
  panel-native cubed sphere
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
- **Convection (CMFMC)**: live on LatLon, RG, CS
  (`Operators/Convection/CMFMCConvection.jl`)
- **Chemistry** (`ExponentialDecay`, `CompositeChemistry`): live on
  `CellState` (LatLon, RG) and `CubedSphereState` topologies
  (CS dispatch shipped in commit `bcd4fea`)
- **Met drivers**: ERA5 spectral, GEOS-FP C720, GEOS-IT C180, MERRA-2
  wind-derived CS, and cubed-sphere binary
- **Adjoint/inversion**: the cubed-sphere discrete-adjoint and surface-flux
  4D-Var stack ship for a documented operator matrix, including checkpointing,
  observation binding, covariance preconditioning, and optimization drivers.
  Archival templates live under
  [`resources/developer_notes/legacy_adjoint_templates/`](resources/developer_notes/legacy_adjoint_templates/)

Canonical operator × topology matrix:
[`src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md).

## What is intentionally deferred

- Remaining adjoint coverage for physics operators
- Real-data user tutorials and campaign-scale validation reports
- Additional observation operators and campaign-scale 4D-Var validation

## Current phase

The active runtime supports LatLon, reduced Gaussian, and cubed-sphere
topologies with advection, diffusion, surface fluxes, convection, chemistry,
transport-binary I/O, and selected adjoint/inversion tooling. The top-level
[`README.md`](../README.md) is the canonical capability/status table.

---

*Last verified against `src/` on 2026-07-13.*
