"""
    Advection (v2)

Advection operators for the dry-mass transport architecture.

Provides:
- `FirstOrderUpwindAdvection` — simple conservative reference operator
- `RussellLernerAdvection` — TM5-faithful van Leer slopes (fully implemented)
- `PPMAdvection` — Putman & Lin PPM (type + stub, Phase 2 kernels)
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- `diagnose_cm!` — vertical flux from horizontal continuity
- CFL utilities for subcycling decisions
"""
module Advection

using DocStringExtensions

import ..AbstractAdvection, ..AbstractOperator, ..apply!
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
