# Multiple-dispatch extension point for topology-specific daily preprocessors.

"""
    process_day(date::Date, grid::AbstractTargetGeometry, settings, vertical; next_day_hour0=nothing)

Topology-specific daily transport-binary preprocessor extension point.

Concrete target geometries implement this method with ordinary Julia multiple
dispatch:

- `LatLonTargetGeometry` writes structured directional LL binaries.
- `ReducedGaussianTargetGeometry` writes face-indexed RG binaries.
- `CubedSphereTargetGeometry` writes panel-local CS binaries.

Every implementation must satisfy the same transport contract:

- use explicit forward endpoint mass targets for every window, including the
  final cross-day window when `next_day_hour0` is available;
- write declared payload semantics, including `delta_semantics`;
- run a write-time replay check unless explicitly disabled for diagnostics;
- produce binaries that the runtime driver can load with replay validation.

This fallback rejects unsupported target geometries after config parsing has
already produced an `AbstractTargetGeometry`.
"""
function process_day(date::Date,
                     grid::AbstractTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    ensure_supported_target(grid)
    return nothing
end
