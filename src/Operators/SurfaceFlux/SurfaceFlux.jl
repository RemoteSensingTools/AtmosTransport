"""
    SurfaceFlux

Surface emission operators (plan 17).

Ships the data types and helpers needed to apply per-tracer surface
sources to a `CellState`:

- [`SurfaceFluxSource{RateT}`](@ref) — single-tracer source + rate
  array (kg/s per cell). Migrated from `src/Models/DrivenSimulation.jl`
  in plan 17 Commit 2. The name remains re-exported from the top-level
  `AtmosTransport` module for backward compat.
- [`PerTracerFluxMap{S}`](@ref) — NTuple-backed map of
  `SurfaceFluxSource`s, keyed by `tracer_name`. Ships with the
  `flux_for(map, :name)` lookup helper. Storage-bits-stable on GPU.

Plan 17 Commit 3 will add the `AbstractSurfaceFluxOperator` hierarchy
(`NoSurfaceFlux`, `SurfaceFluxOperator`), the `_surface_flux_kernel!`
KA kernel, and the `apply!` / `apply_surface_flux!` entry points. Until
then, `DrivenSimulation` continues to call the legacy
`_apply_surface_source!` helpers at sim level; Commit 2 only reorganises
where those helpers live.

# Surface layer convention

All kernels here assume `k = Nz` is the surface (plan 17 Decision 2).
This matches `src/Models/DrivenSimulation.jl` (pre-17) and the LatLon
grid storage layout. A future `AbstractLayerOrdering{TopDown, BottomUp}`
refactor can generalise this; out of scope for plan 17.
"""
module SurfaceFlux

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState, get_tracer, tracer_index, eachtracer
import ..apply!

export SurfaceFluxSource, PerTracerFluxMap, flux_for
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!, emitting_tracer_indices

include("sources.jl")
include("PerTracerFluxMap.jl")
include("surface_flux_kernels.jl")
include("operators.jl")

end # module SurfaceFlux
