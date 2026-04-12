# Runtime Flow

## High-level flow

The clean `src` runtime split is:

1. reader parses a transport binary
2. driver exposes typed per-window forcing
3. model owns prognostic transport state
4. runtime prepares substep forcing
5. advection operator advances the model one substep

## Ownership rules

The met driver owns:

- file I/O
- window timing
- forcing payloads
- humidity endpoint metadata
- interpolation metadata

The model owns:

- air mass
- tracer masses
- advection workspace

This separation is important because tracer state should not be entangled with
I/O policy.

## Time interpolation

Time interpolation belongs upstream of the kernels.

The current `DrivenSimulation` path:

- loads one transport window
- computes a substep interpolation fraction
- prepares the instantaneous/substep forcing state
- calls `step!(model, Δt)`

The kernels themselves are time-agnostic.

## Closure policy

The preferred `src` direction is:

- write all required mass fluxes into the binary
- do closure in preprocessing or in a driver-prep layer
- keep closure out of the advection kernels

Temporary compatibility logic may still exist elsewhere, but it should not be
the architectural target.
