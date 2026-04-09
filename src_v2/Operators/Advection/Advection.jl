"""
    Advection (v2)

Advection operators for the basis-explicit transport architecture.

Provides:
- `UpwindAdvection` — simple conservative reference operator
- `RussellLernerAdvection` — linear reconstruction with van Leer-type slopes
- `PPMAdvection` — quadratic reconstruction family stub (Phase 4 kernels)
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- `diagnose_cm!` — compatibility shim for vertical-flux diagnosis outside the core stepping path
- CFL utilities for subcycling decisions
"""
module Advection

using DocStringExtensions

import ..AbstractAdvection, ..AbstractConstantReconstruction, ..AbstractLinearReconstruction, ..AbstractQuadraticReconstruction, ..AbstractOperator, ..apply!
using ...State: CellState, AbstractStructuredFaceFluxState, AbstractFaceFluxState,
    StructuredFaceFluxState, AbstractUnstructuredFaceFluxState,
    DryMassFluxBasis, DryStructuredFluxState, AbstractMassBasis,
    FaceIndexedFluxState
using ...Grids: AtmosGrid, AbstractHorizontalMesh, AbstractStructuredMesh,
    face_cells, nfaces
using ...MetDrivers: diagnose_cm_from_continuity!

include("FaceReconstruction.jl")
include("MassCFLPilot.jl")
include("Upwind.jl")
include("RussellLerner.jl")
include("PPM.jl")
include("Divergence.jl")
include("StrangSplitting.jl")

end # module Advection
