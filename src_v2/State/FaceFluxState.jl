# ---------------------------------------------------------------------------
# FaceFluxState — face-centered dry mass fluxes for transport
#
# Abstraction lives at the operator contract level, not the storage level:
#
#   abstract type AbstractFaceFluxState end
#       ├── AbstractStructuredFaceFluxState   (am, bm, cm on logically rectangular meshes)
#       └── AbstractUnstructuredFaceFluxState  (face-indexed connectivity, Phase 2+)
#
# Structured meshes keep the proven directional storage (am, bm, cm) and
# cell-loop kernels.  Unstructured meshes will get face-indexed storage and
# face-loop kernels.  The operator dispatch selects the right realization:
#
#   apply!(state, fluxes::AbstractStructuredFaceFluxState,   grid, scheme, dt)
#   apply!(state, fluxes::AbstractUnstructuredFaceFluxState, grid, scheme, dt)
#
# Same math, same high-level API, different low-level memory layout,
# different kernels where justified.
# ---------------------------------------------------------------------------

"""
    AbstractFaceFluxState

Root type for all face-centered dry mass flux representations.

The transport operator contract is written in terms of this abstract type.
Concrete subtypes differ in storage layout to match the mesh's natural
`flux_topology`.
"""
abstract type AbstractFaceFluxState end

"""
    AbstractStructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as separate directional arrays on a logically rectangular
mesh.  Concrete subtypes expose `am` (x-face), `bm` (y-face), `cm` (z-face).

Structured cell-loop kernels access these arrays directly for performance.
"""
abstract type AbstractStructuredFaceFluxState <: AbstractFaceFluxState end

"""
    AbstractUnstructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as a single face-indexed array with explicit connectivity.
Natural for reduced Gaussian and other unstructured meshes (Phase 2+).
"""
abstract type AbstractUnstructuredFaceFluxState <: AbstractFaceFluxState end

# ---------------------------------------------------------------------------
# Concrete structured type — keeps am, bm, cm for the fast path
# ---------------------------------------------------------------------------

"""
    StructuredFaceFluxState{AX, AY, AZ} <: AbstractStructuredFaceFluxState

Face-centered dry mass fluxes for structured grids.

# Fields
- `am :: AX` — x-face (longitude) dry mass flux [kg per half-timestep].
  Layout: `(Nx+1, Ny, Nz)` for LatLon, `(Nc+1, Nc, Nz)` per panel for CS.
- `bm :: AY` — y-face (latitude) dry mass flux [kg per half-timestep].
  Layout: `(Nx, Ny+1, Nz)` for LatLon, `(Nc, Nc+1, Nz)` per panel for CS.
- `cm :: AZ` — z-face (vertical) dry mass flux [kg per half-timestep].
  Layout: `(Nx, Ny, Nz+1)` for LatLon.

# Convention
- Positive `am` = eastward mass transport
- Positive `bm` = northward mass transport
- Positive `cm` = downward (increasing k / pressure) mass transport
"""
struct StructuredFaceFluxState{AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState
    am :: AX
    bm :: AY
    cm :: AZ
end

# ---------------------------------------------------------------------------
# Concrete unstructured type — Phase 2+
# ---------------------------------------------------------------------------

"""
    FaceIndexedFluxState{A, AZ} <: AbstractUnstructuredFaceFluxState

Face-centered dry mass fluxes for unstructured meshes (Phase 2+).

# Fields
- `horizontal_flux :: A` — dry mass flux per horizontal face [kg per half-timestep].
  Layout: `(nfaces, Nz)`. Positive = flow in face-normal direction.
- `cm :: AZ` — vertical flux, same convention as structured.

# Vertical storage convention
The `cm` field assumes vertical fluxes are columnar (one column per horizontal
cell, same for all mesh types). This is a convenience that holds for every
atmospheric grid we currently target (ERA5, GEOS-FP, GEOS-IT, reduced Gaussian).
If a future mesh requires non-columnar vertical connectivity, define a new
concrete subtype of `AbstractUnstructuredFaceFluxState` with different storage —
the abstract hierarchy supports this without breaking existing code.
"""
struct FaceIndexedFluxState{A <: AbstractArray, AZ <: AbstractArray} <: AbstractUnstructuredFaceFluxState
    horizontal_flux :: A
    cm              :: AZ
end

# ---------------------------------------------------------------------------
# Scoped accessor functions
#
# These are available for generic code and validation but the structured
# fast-path kernels bypass them and use am/bm/cm arrays directly.
# ---------------------------------------------------------------------------

@inline face_flux_x(s::AbstractStructuredFaceFluxState, i, j, k) = s.am[i, j, k]
@inline face_flux_y(s::AbstractStructuredFaceFluxState, i, j, k) = s.bm[i, j, k]
@inline face_flux_z(s::AbstractFaceFluxState, i, j, k)           = s.cm[i, j, k]
@inline face_flux(s::AbstractUnstructuredFaceFluxState, f, k)     = s.horizontal_flux[f, k]

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    allocate_face_fluxes(::StructuredFluxTopology, Nx, Ny, Nz; FT=Float64, ArrayType=Array)

Allocate zeroed face flux arrays for a structured mesh.
"""
function allocate_face_fluxes(::StructuredFluxTopology,
                              Nx::Int, Ny::Int, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array)
    am = ArrayType(zeros(FT, Nx + 1, Ny,     Nz))
    bm = ArrayType(zeros(FT, Nx,     Ny + 1, Nz))
    cm = ArrayType(zeros(FT, Nx,     Ny,     Nz + 1))
    return StructuredFaceFluxState(am, bm, cm)
end

export AbstractFaceFluxState
export AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export allocate_face_fluxes
