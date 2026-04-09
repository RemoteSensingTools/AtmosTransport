# Core Contracts

## Core state contract

The transport core advances:

- cell air mass `m`
- tracer masses `rm`

The core does not advance concentrations directly.

## Basis contract

Mass basis is explicit in the type domain:

- `CellState{DryBasis}` or `CellState{MoistBasis}`
- `FluxState{DryBasis}` or `FluxState{MoistBasis}`

Rules:

- state and flux basis must match
- basis conversion belongs in drivers/adapters
- advection kernels should not decide dry vs moist semantics on the fly

## Kernel contract

Advection kernels should consume a fully prepared forcing state:

- current cell mass
- current tracer masses
- current horizontal mass fluxes
- current vertical mass fluxes

Kernels should not:

- interpret raw met timing semantics
- reconstruct missing vertical closure
- own humidity-to-dry conversion logic

## Reconstruction contract

Prognostic transport state remains:

- `m`
- `rm`

Reconstruction is diagnostic and workspace-backed.

Planned public reconstruction families:

- `AbstractConstantReconstruction`
- `AbstractLinearReconstruction`
- `AbstractQuadraticReconstruction`

Current runnable scheme:

- `UpwindAdvection <: AbstractConstantReconstruction`

## Topology contract

Two first-class horizontal representations are expected:

- `StructuredDirectional`
- `FaceIndexed`

The common numerical model is still conservative tracer-mass transport, but the
storage and kernel pathways are allowed to differ by topology.
