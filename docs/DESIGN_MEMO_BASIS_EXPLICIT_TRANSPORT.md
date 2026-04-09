# Design Memo: Basis-Explicit Transport Core for `src_v2`

Date: 2026-04-08

This memo supersedes the dry-only framing in
[DESIGN_MEMO_DRY_MASS_TRANSPORT.md](/home/cfranken/code/gitHub/AtmosTransportModel/docs/DESIGN_MEMO_DRY_MASS_TRANSPORT.md)
for new `src_v2` runtime work.

## Core Invariant

`src_v2` is being built around one explicit contract:

- conservative transport of tracer mass
- on an abstract mesh
- using explicitly tagged air-mass basis

The transport core operates on:

- `CellState{Basis}`
- `FluxState{Basis}`
- `Grid`
- `AbstractAdvection`

`Basis` is part of the type contract:

- `DryBasis`
- `MoistBasis`

Operators require matching basis at dispatch time. Basis conversion belongs in
drivers and adapters, never inside advection kernels.

## What This Changes

The old design memo treated dry face mass fluxes as the central architecture.
That was too narrow for the new direction.

The new rule is:

- the core is basis-explicit, not dry-only
- dry and moist are both first-class
- adapter paths may choose one basis or convert between them
- parity claims against TM5 or GCHP belong to adapter/config combinations, not
  to the core itself

## Layering

The intended `src_v2` layering is:

1. `Grids`
- mesh topology and geometry
- vertical coordinates
- `PlanetParameters`

2. `State`
- `CellState{Basis}`
- `StructuredFaceFluxState{Basis}`
- `FaceIndexedFluxState{Basis}`

3. `Operators`
- advection and later physics operators
- kernels take raw arrays plus concrete config

4. `Drivers` / `Adapters`
- read native meteorology or preprocessed products
- construct basis-matching state and flux objects
- own humid/dry conversion and closure policy

5. `Models`
- standalone runtime objects such as `TransportModel` and `Simulation`

## Topology Model

Two storage topologies are first-class:

- `StructuredTopology`
  - regular lat-lon today
  - cubed-sphere fast paths later

- `FaceConnectedTopology`
  - reduced Gaussian
  - any mesh with explicit face connectivity

The public API must support both without forcing one storage model to emulate
the other inside hot kernels.

## Immediate Runtime Priorities

Phase ordering for `src_v2` should be:

1. generic correctness
2. one simple runnable structured path
3. one simple runnable face-connected path
4. higher-order operators
5. adapter maturity
6. model-specific parity work

This means:

- first runnable operator: `FirstOrderUpwindAdvection`
- first runnable path: structured lat-lon
- immediate second runnable path: reduced-Gaussian / face-connected

High-order transport, TM5-faithful behavior, and GCHP-faithful behavior come
later as adapter-validation efforts.

## Binary Contract Implication

A basis-explicit core also means the binary contract must be explicit about
what is needed to move between moist and dry diagnostics.

For new moist-basis transport binaries that may later write dry-air output:

- `mass_basis = "moist"` must be recorded in the header
- `qv_start` and `qv_end` should be carried per window
- dry-VMR output should use the runtime end state plus `qv_end`, not a stale
  window-start humidity field

A single `qv` field is still acceptable as a compatibility payload, but it is
not the preferred contract for new files because it lacks explicit endpoint
semantics.

If an adapter reconstructs dry reference mass from pressure endpoints instead
of the transported moist mass, then endpoint surface pressure such as
`ps_end` belongs in adapter-specific metadata. That is not a core transport
requirement.

## Compatibility Boundary

The dry-binary handoff documented in
[MEMO_DRY_BINARY_RUNTIME_HANDOFF_2026-04-08.md](/home/cfranken/code/gitHub/AtmosTransportModel/docs/MEMO_DRY_BINARY_RUNTIME_HANDOFF_2026-04-08.md)
remains useful, but only as a compatibility side-track.

Its purpose is:

- preserving a regression bridge from `src_v2` preprocessing into legacy `src`
- validating metadata and basis-tag handling
- avoiding loss of existing test coverage during the rewrite

It must not define the architecture of the new runtime.

## Engineering Rules

`src_v2` should follow the existing Oceanigans-style direction already visible
in the grid layer:

- validated constructors
- small concrete immutable structs
- explicit geometry APIs
- type-driven dispatch outside kernels
- no abstractly-typed fields in hot objects
- no allocations in stepping wrappers after workspace creation
- honest stubs or explicit errors for unimplemented paths

## Current Implementation Status

As of this memo:

- `PlanetParameters` exists in `src_v2`
- basis-explicit `CellState` and `FluxState` exist
- a standalone `TransportModel` and `Simulation` exist
- `FirstOrderUpwindAdvection` runs on:
  - structured lat-lon
  - face-connected reduced-Gaussian meshes
- unsupported cubed-sphere geometry is now explicitly metadata-only and should
  fail honestly until implemented

## Non-Goals For The Core

The core should not:

- assume dry mass everywhere
- assume regular index directions are universal
- embed TM5 or GCHP semantics directly into operators
- depend permanently on the legacy `src` runtime

## Short Decision Rule

When adding new code to `src_v2`, prefer the answer that keeps this statement
true:

> The transport core knows only geometry, basis-tagged state, basis-tagged face
> fluxes, and numerical operators.

