"""
    State (v2)

Prognostic and diagnostic state containers for the dry-mass transport architecture.

Provides:
- `CellState` — cell-centered dry air mass + tracer masses (prognostic)
- `AbstractFaceFluxState` hierarchy — face dry mass fluxes
  - `AbstractStructuredFaceFluxState` → `StructuredFaceFluxState` (am, bm, cm)
  - `AbstractUnstructuredFaceFluxState` → `FaceIndexedFluxState` (Phase 2+)
- `MetState` — upstream meteorological fields (consumed by flux builders, not transport)
- Tracer allocation and mixing-ratio utilities
"""
module State

using ..Grids: StructuredFluxTopology

include("CellState.jl")
include("FaceFluxState.jl")
include("MetState.jl")
include("Tracers.jl")

end # module State
