"""
    Preprocessing transport-binary pipeline

Include index for the transport-binary preprocessor implementation.

Topology-specific workflows live under `src/Preprocessing/transport_binary/`:

- `latlon_spectral.jl` writes structured lat-lon binaries from ERA5 spectral input.
- `cubed_sphere_spectral.jl` writes cubed-sphere binaries from ERA5 spectral input.
- `cubed_sphere_regrid.jl` rewrites structured lat-lon binaries onto cubed-sphere panels.
- `entrypoint.jl` contains the TOML-driven CLI entry point.

Shared contracts and support code are kept separate from topology workflows:

- `core.jl` handles binary metadata, payload sizing, provenance, and raw writes.
- `latlon_workspaces.jl` stages spectral, humidity, dry-basis, and vertical merge state.
- `latlon_contracts.jl` implements structured replay, Poisson closure, and v4 writes.
- `cubed_sphere_contracts.jl` implements CS replay and endpoint-delta helpers.
- `topology_dispatch.jl` documents the `process_day(date, grid, settings, vertical)` extension point.

New topologies should add a dedicated workflow file and dispatch on a new
`AbstractTargetGeometry` subtype. The workflow must write binaries that satisfy
the same transport contract: explicit forward endpoint mass deltas, write-time
replay validation, declared payload semantics, and load-time replay coverage.
"""

include("transport_binary/core.jl")
include("transport_binary/latlon_workspaces.jl")
include("transport_binary/latlon_contracts.jl")
include("transport_binary/cubed_sphere_contracts.jl")
include("transport_binary/topology_dispatch.jl")
include("transport_binary/cubed_sphere_spectral.jl")
include("transport_binary/cubed_sphere_regrid.jl")
include("transport_binary/latlon_spectral.jl")
include("transport_binary/entrypoint.jl")
