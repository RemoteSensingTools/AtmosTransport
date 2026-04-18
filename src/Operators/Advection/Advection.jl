"""
    Advection

Advection operators for the basis-explicit transport architecture.

Provides:

**Scheme hierarchy**:
- `UpwindScheme <: AbstractConstantScheme` — first-order upwind via generic kernels
- `SlopesScheme <: AbstractLinearScheme`   — van Leer slopes (limiter-dispatched)
- `PPMScheme <: AbstractQuadraticScheme`   — structured-grid PPM (not yet an official real-data reference path)
- `AbstractLimiter` subtypes: `NoLimiter`, `MonotoneLimiter`, `PositivityLimiter`

**Multi-tracer optimization**:
- `TracerView` — zero-cost 3D slice adapter for 4D tracer arrays
- Multi-tracer kernel shells fuse the N-tracer loop into GPU kernels,
  reducing launches from 6N to 6 per Strang split

**Infrastructure**:
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- `diagnose_cm!` — vertical-flux diagnosis shim
- CFL utilities for subcycling decisions
"""
module Advection

using Adapt
using DocStringExtensions

import ..AbstractOperator, ..apply!
using ...State: CellState, AbstractStructuredFaceFluxState, AbstractFaceFluxState,
    StructuredFaceFluxState, AbstractUnstructuredFaceFluxState,
    DryMassFluxBasis, DryStructuredFluxState, AbstractMassBasis,
    FaceIndexedFluxState,
    ntracers, tracer_index, tracer_name, get_tracer, eachtracer
using ...Grids: AtmosGrid, AbstractHorizontalMesh, AbstractStructuredMesh,
    LatLonMesh, CubedSphereMesh, face_cells, nfaces,
    PanelConnectivity, reciprocal_edge,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ...MetDrivers: diagnose_cm_from_continuity!
using ...Architectures: _kahan_add

# New scheme hierarchy (include before anything that references these types)
include("schemes.jl")
include("limiters.jl")
include("reconstruction.jl")
include("structured_kernels.jl")
include("multitracer_kernels.jl")

# Cubed-sphere halo exchange and Strang splitting
include("HaloExchange.jl")
include("CubedSphereStrang.jl")

# PPM subgrid distributions (shared by CS PPM kernels and LinRood)
include("ppm_subgrid_distributions.jl")

# Lin-Rood cross-term advection for cubed-sphere grids (FV3 fv_tp_2d)
include("LinRood.jl")

# Vertical remap (FV3-style conservative PPM, per-column)
include("VerticalRemap.jl")

include("Divergence.jl")
include("StrangSplitting.jl")

end # module Advection
