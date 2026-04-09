# Scope And Status

## What `docs_v2` is

`docs_v2` is a breadcrumb layer for the new transport architecture in `src_v2`.
It exists to make the implementation easier to follow while the system is still
being designed carefully.

The documentation style should stay:

- explicit
- technical
- reviewable by non-Julia users
- close to the actual code paths

## What `src_v2` is trying to become

`src_v2` is the future standalone runtime target. It is being designed around a
generic transport core rather than around TM5-, GEOS-, or GCHP-specific control
flow.

The key architectural direction is:

- explicit cell mass and tracer-mass transport
- explicit mass-basis tags (`DryBasis`, `MoistBasis`)
- topology-aware grids and flux states
- clean met-driver boundaries
- no closure logic inside advection kernels

## Current first polished path

The first polished reference path is:

- grid: ERA5-derived structured lat-lon
- runtime: `src_v2` only
- operator: `UpwindAdvection`
- basis: moist or dry, with moist as the default reference path
- physics scope: pure advection only

## What is intentionally deferred

- production-ready higher-order schemes
- cubed-sphere runtime support
- observation operators
- chemistry and physics packages
- public Documenter site structure

## Current implementation phase

At the moment the codebase has:

- basis-explicit state and flux types
- a generic transport-binary family
- a typed met-driver interface
- a standalone driven runtime
- structured lat-lon and reduced-Gaussian smoke paths
- explicit binary timing semantics and driver-side validation

The next phases should refine the real-data paths, not expand surface area
blindly.
