"""
    Advection (v2)

Advection operators for the basis-explicit transport architecture.

Provides:

**New hierarchy** (preferred):
- `UpwindScheme <: AbstractConstantScheme` — first-order upwind via generic kernels
- `SlopesScheme <: AbstractLinearScheme`   — van Leer slopes (limiter-dispatched)
- `PPMScheme <: AbstractQuadraticScheme`   — structured-grid PPM (not yet an official real-data reference path)
- `AbstractLimiter` subtypes: `NoLimiter`, `MonotoneLimiter`, `PositivityLimiter`

**Multi-tracer optimization**:
- `TracerView` — zero-cost 3D slice adapter for 4D tracer arrays
- Multi-tracer kernel shells fuse the N-tracer loop into GPU kernels,
  reducing launches from 6N to 6 per Strang split

**Legacy types** (kept for backward compatibility during transition):
- `UpwindAdvection`, `RussellLernerAdvection`, `PPMAdvection`

**Infrastructure**:
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- `diagnose_cm!` — vertical-flux diagnosis shim
- CFL utilities for subcycling decisions
"""
module Advection

using Adapt
using DocStringExtensions

import ..AbstractAdvection, ..AbstractConstantReconstruction, ..AbstractLinearReconstruction, ..AbstractQuadraticReconstruction, ..AbstractOperator, ..apply!
using ...State: CellState, AbstractStructuredFaceFluxState, AbstractFaceFluxState,
    StructuredFaceFluxState, AbstractUnstructuredFaceFluxState,
    DryMassFluxBasis, DryStructuredFluxState, AbstractMassBasis,
    FaceIndexedFluxState
using ...Grids: AtmosGrid, AbstractHorizontalMesh, AbstractStructuredMesh,
    LatLonMesh, CubedSphereMesh, face_cells, nfaces
using ...MetDrivers: diagnose_cm_from_continuity!

# New scheme hierarchy (include before anything that references these types)
include("schemes.jl")
include("limiters.jl")
include("reconstruction.jl")
include("structured_kernels.jl")
include("multitracer_kernels.jl")

# Legacy scheme files
include("FaceReconstruction.jl")
include("MassCFLPilot.jl")
include("Upwind.jl")
include("RussellLerner.jl")
include("PPM.jl")
include("Divergence.jl")
include("StrangSplitting.jl")

end # module Advection
