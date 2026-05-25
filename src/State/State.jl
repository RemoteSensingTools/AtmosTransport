"""
    State

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

using Adapt
using ..Grids: AbstractHorizontalMesh, AbstractStructuredMesh, CubedSphereMesh,
    StructuredFluxTopology, FaceIndexedFluxTopology,
    flux_topology, ncells, nfaces, nx, ny

include("Basis.jl")
include("CellState.jl")
include("CubedSphereState.jl")
include("FaceFluxState.jl")
include("MetState.jl")
include("Tracers.jl")
include("Fields/Fields.jl")

using .Fields: AbstractTimeVaryingField, AbstractCubedSphereField,
               ConstantField, ProfileKzField, PreComputedKzField,
               CubedSphereField, DerivedKzField, WindowPBLKzField,
               LocalHoltslagBovilleKzField,
               GCHPHoltslagBovilleKzField,                  # deprecated alias
               PBLPhysicsParameters, StepwiseField,
               field_value, update_field!, refresh_pbl_kz_cache!,
               refresh_local_holtslag_boville_kz_cache!,
               refresh_gchp_holtslag_boville_kz_cache!,     # deprecated alias
               integral_between, panel_field
export AbstractTimeVaryingField, AbstractCubedSphereField,
       ConstantField, ProfileKzField, PreComputedKzField,
       CubedSphereField, DerivedKzField, WindowPBLKzField,
       LocalHoltslagBovilleKzField,
       GCHPHoltslagBovilleKzField,                          # deprecated alias
       PBLPhysicsParameters, StepwiseField,
       field_value, update_field!, refresh_pbl_kz_cache!,
       refresh_local_holtslag_boville_kz_cache!,
       refresh_gchp_holtslag_boville_kz_cache!,             # deprecated alias
       integral_between, panel_field

end # module State
