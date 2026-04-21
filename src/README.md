# src

Primary source tree for AtmosTransportModel.

This is the runtime code, not just utilities. If you are trying to
understand how a transport step is assembled end to end, start here.

## Main Runtime Flow

At a high level, the live path is:

1. [`MetDrivers/README.md`](MetDrivers/README.md) loads or prepares a
   transport window and its grid
2. [`State/README.md`](State/README.md) holds air mass, tracer mass, and
   face-flux storage in topology-appropriate containers
3. [`Models/README.md`](Models/README.md) composes state, grid, drivers,
   and operators into a runnable model/simulation
4. [`Operators/README.md`](Operators/README.md) applies transport and
   post-transport physics

Geometry and topology decisions come from
[`Grids/README.md`](Grids/README.md).

## Folder Map

- [`Grids/`](Grids/README.md) — mesh types, topology tags, vertical
  coordinates, geometry accessors
- [`State/`](State/README.md) — prognostic state, flux state, tracer
  accessors, time-varying operator fields
- [`MetDrivers/`](MetDrivers/README.md) — transport-window readers,
  runtime drivers, mass closure, convection forcing, ERA5-specific paths
- [`Operators/`](Operators/README.md) — advection, diffusion, surface
  flux, convection, chemistry
- [`Models/`](Models/README.md) — `TransportModel`, fixed-step runtime,
  window-driven runtime
- `Parameters/` — physical constants and planetary parameters
- `Architectures.jl` — CPU/GPU adaptation helpers and backend utilities
- `Kernels/` — lower-level kernel helpers shared outside the operator tree
- `Regridding/` — conservative regridding and weight application
- `Preprocessing/` — transport-binary and met-data preparation
- `Downloads/` — data acquisition helpers

## Read This First For Common Goals

- "I need to understand one model step":
  start with [`Models/README.md`](Models/README.md), then
  [`Operators/README.md`](Operators/README.md)
- "I need to understand why a topology dispatches differently":
  start with [`Grids/README.md`](Grids/README.md), then
  [`State/README.md`](State/README.md), then the relevant operator README
- "I need to trace data from disk to runtime":
  start with [`MetDrivers/README.md`](MetDrivers/README.md)
- "I need to add a new operator input field":
  start with [`State/Fields/README.md`](State/Fields/README.md)

## Related Docs

- Runtime walkthrough:
  [`../docs/20_RUNTIME_FLOW.md`](../docs/20_RUNTIME_FLOW.md)
- Binary and driver overview:
  [`../docs/30_BINARY_AND_DRIVERS.md`](../docs/30_BINARY_AND_DRIVERS.md)
- Quality gates:
  [`../docs/40_QUALITY_GATES.md`](../docs/40_QUALITY_GATES.md)
- Operator block ordering:
  [`../docs/plans/OPERATOR_COMPOSITION.md`](../docs/plans/OPERATOR_COMPOSITION.md)
