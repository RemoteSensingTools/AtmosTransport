"""
    State (v2)

Prognostic and diagnostic state containers for the basis-explicit transport architecture.

Provides:
- `CellState` — cell-centered air mass + tracer masses (prognostic)
- `AbstractFaceFluxState` hierarchy — face mass fluxes
  - `AbstractStructuredFaceFluxState` → `StructuredFaceFluxState` (am, bm, cm)
  - `AbstractUnstructuredFaceFluxState` → `FaceIndexedFluxState` (Phase 2+)
- `MetState` — upstream meteorological fields (consumed by flux builders, not transport)
- Tracer allocation and mixing-ratio utilities
"""
module State

using ..Grids: AbstractHorizontalMesh, AbstractStructuredMesh,
    StructuredFluxTopology, FaceIndexedFluxTopology,
    flux_topology, ncells, nfaces, nx, ny

include("Basis.jl")
include("CellState.jl")
include("FaceFluxState.jl")
include("MetState.jl")
include("Tracers.jl")

end # module State
