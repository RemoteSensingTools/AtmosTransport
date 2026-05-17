# Transport Runtime And Preprocessing Paradigm

This document describes the production target for binary preprocessing and
runtime transport. The goal is one contract and one mental model across
meteorology sources, grid topologies, and advection schemes.

## Core Contract

Preprocessing owns all expensive, source-specific, and topology-specific
normalization:

- Read raw meteorology through a typed reader.
- Apply a typed vertical transform.
- Regrid or map onto the runtime topology.
- Verify continuity, positivity, replay, and substep schedule requirements.
- Write a binary whose header and per-window metadata fully define the runtime
  contract.

Runtime owns only repeated prognostic work:

- Load one verified window.
- Refresh the model state and forcing views.
- Advance the configured operators with the schedule stored in the binary.
- Refuse obsolete or incomplete binary contracts.

Configuration files choose inputs, dates, physics, and output. They should not
duplicate facts that are already in the binary, such as the level count or
adaptive substep schedule.

## Topology Storage

Every topology stores all tracers in one packed raw buffer:

| Topology | State storage | Runtime transport target |
| --- | --- | --- |
| LatLon | `Array{FT,4}`: `(Nx, Ny, Nz, Nt)` | packed multi-tracer sweeps |
| ReducedGaussian | `Array{FT,3}`: `(ncells, Nz, Nt)` | face-indexed sweeps, currently per tracer |
| CubedSphere | `NTuple{6}` of `(N+2Hp, N+2Hp, Nz, Nt)` | packed panel multi-tracer sweeps |

The packed buffer is the source of truth. `get_tracer` and `eachtracer` are
views for compatibility and diagnostics, not the preferred production transport
interface.

## Cubed-Sphere Production Path

Cubed-sphere split-sweep schemes now follow the same packed-tracer paradigm as
the structured path:

1. Fill halos once for `state.air_mass` and once for packed
   `state.tracers_raw`.
2. Run the palindrome `X -> Y -> Z -> midpoint -> Z -> Y -> X`.
3. Update air mass once per sweep.
4. Update every tracer inside the same panel sweep kernel.
5. Apply diffusion and surface fluxes at the palindrome midpoint to the packed
   CS tracer storage.

`LinRoodPPMScheme` remains a separate research path because its horizontal
update and adjoint tape are algorithmically different. Production full-physics
GEOS-CS runs should use `PPMScheme`.

## Binary Schedule Ownership

For current CS binaries, the preprocessor writes `steps_per_window` per
meteorological window. The runtime honors that schedule directly and does not
run a second adaptive pilot unless `ATMOSTR_ASSERT_CS_BINARY_CFL=1` is set for
diagnostics.

The preprocessor schedule gate uses the same static palindrome CFL budget as
the runtime assertion:

```text
required_steps = ceil(2 * (out_x + out_y + out_z) / (m * cfl_target))
```

This is a conservative proxy for the six half-sweep palindrome. Strict replay
verification remains the contract that proves a written window is acceptable.

## Physics Placement

Runtime composition is:

```text
transport_block(dt) -> convection_block(dt) -> chemistry_block(dt)
```

The transport block is:

```text
X -> Y -> Z -> V(dt) -> Z -> Y -> X
```

or, when surface fluxes are active:

```text
X -> Y -> Z -> V(dt/2) -> S(dt) -> V(dt/2) -> Z -> Y -> X
```

For binary-scheduled driven runs, the binary schedule is an advection substep
contract, not a physics cadence contract. The runtime applies the transport
block at each stored substep, resets to the verified window endpoint, then
applies convection and chemistry once at the end of the meteorological window
with `window_dt`. This should be treated as an explicit model choice and
validated for campaign configurations with strong convective forcing.

## Adding Or Changing A Topology

A production topology must provide:

- A typed binary contract and replay verifier.
- A workspace allocator with all steady scratch buffers.
- A packed tracer transport entry point.
- A no-allocation midpoint path for diffusion and surface fluxes.
- A schedule trait that makes binary-owned adaptive substeps explicit.
- Tests for constant and non-uniform per-window schedules.
- Tests comparing packed multi-tracer transport against the scalar reference
  for at least one low-order and one production-order scheme.

New topology code should not add runner-specific branches for source readers,
vertical transforms, positivity gates, or binary schedule interpretation. Those
belong in the preprocessing contract layer.
