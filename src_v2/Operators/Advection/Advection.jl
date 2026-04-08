"""
    Advection (v2)

Advection operators for the dry-mass transport architecture.

Provides:
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
    StructuredFaceFluxState, DryMassFluxBasis, DryStructuredFluxState
using ...Grids: AtmosGrid, AbstractStructuredMesh

include("FaceReconstruction.jl")
include("MassCFLPilot.jl")
include("RussellLerner.jl")
include("PPM.jl")
include("Divergence.jl")
include("StrangSplitting.jl")

end # module Advection
