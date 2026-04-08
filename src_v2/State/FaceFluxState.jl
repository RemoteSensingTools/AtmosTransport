# ---------------------------------------------------------------------------
# FaceFluxState — face-centered mass fluxes for transport
#
# Abstraction lives at the operator contract level, not the storage level:
#
#   abstract type AbstractFaceFluxState end
#       ├── AbstractStructuredFaceFluxState   (am, bm, cm on logically rectangular meshes)
#       └── AbstractUnstructuredFaceFluxState  (face-indexed connectivity, Phase 2+)
#
# Each concrete flux state carries a `Basis <: AbstractMassFluxBasis` type
# parameter that records whether the stored fluxes are on a moist or dry
# basis.  This prevents accidentally passing moist fluxes to a dry-only
# transport operator at the type level.
#
# Structured meshes keep the proven directional storage (am, bm, cm) and
# cell-loop kernels.  Unstructured meshes will get face-indexed storage and
# face-loop kernels.  The operator dispatch selects the right realization:
#
#   apply!(state, fluxes::StructuredFaceFluxState{DryMassFluxBasis}, grid, scheme, dt)
#
# Same math, same high-level API, different low-level memory layout,
# different kernels where justified.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mass flux basis — moist vs dry tagging at the type level
# ---------------------------------------------------------------------------

"""
    AbstractMassFluxBasis

Supertype for mass flux basis tags.  Carried as the first type parameter
on concrete `AbstractFaceFluxState` subtypes to distinguish moist from dry
fluxes at compile time.
"""
abstract type AbstractMassFluxBasis end

"""
    MoistMassFluxBasis <: AbstractMassFluxBasis

Tag indicating that fluxes are on a **moist** (total air) basis.
Binary readers produce flux states with this basis.
"""
struct MoistMassFluxBasis <: AbstractMassFluxBasis end

"""
    DryMassFluxBasis <: AbstractMassFluxBasis

Tag indicating that fluxes are on a **dry** air basis.
Transport operators require flux states with this basis.
"""
struct DryMassFluxBasis <: AbstractMassFluxBasis end

# ---------------------------------------------------------------------------
# Abstract hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractFaceFluxState

Root type for all face-centered mass flux representations.

The transport operator contract is written in terms of this abstract type.
Concrete subtypes differ in storage layout to match the mesh's natural
`flux_topology`, and carry a `Basis <: AbstractMassFluxBasis` type parameter
to enforce moist/dry safety.
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
    StructuredFaceFluxState{Basis, AX, AY, AZ} <: AbstractStructuredFaceFluxState

Face-centered mass fluxes for structured grids, tagged with `Basis` to
indicate whether the stored values are moist or dry.

# Type parameters
- `Basis <: AbstractMassFluxBasis` — `MoistMassFluxBasis` or `DryMassFluxBasis`

# Fields
- `am :: AX` — x-face (longitude) mass flux [kg per half-timestep].
  Layout: `(Nx+1, Ny, Nz)` for LatLon, `(Nc+1, Nc, Nz)` per panel for CS.
- `bm :: AY` — y-face (latitude) mass flux [kg per half-timestep].
  Layout: `(Nx, Ny+1, Nz)` for LatLon, `(Nc, Nc+1, Nz)` per panel for CS.
- `cm :: AZ` — z-face (vertical) mass flux [kg per half-timestep].
  Layout: `(Nx, Ny, Nz+1)` for LatLon.

# Convention
- Positive `am` = eastward mass transport
- Positive `bm` = northward mass transport
- Positive `cm` = downward (increasing k / pressure) mass transport

# Examples
```jldoctest
julia> am = zeros(11, 8, 4); bm = zeros(10, 9, 4); cm = zeros(10, 8, 5);

julia> dry = StructuredFaceFluxState{DryMassFluxBasis}(am, bm, cm);

julia> flux_basis(dry)
DryMassFluxBasis()

julia> moist = StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm);

julia> flux_basis(moist)
MoistMassFluxBasis()
```
"""
struct StructuredFaceFluxState{Basis <: AbstractMassFluxBasis,
                                AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState
    am :: AX
    bm :: AY
    cm :: AZ
end

function StructuredFaceFluxState{B}(am::AX, bm::AY, cm::AZ) where {B <: AbstractMassFluxBasis,
                                                                       AX <: AbstractArray,
                                                                       AY <: AbstractArray,
                                                                       AZ <: AbstractArray}
    StructuredFaceFluxState{B, AX, AY, AZ}(am, bm, cm)
end

StructuredFaceFluxState(am, bm, cm) = StructuredFaceFluxState{DryMassFluxBasis}(am, bm, cm)

# ---------------------------------------------------------------------------
# Concrete unstructured type — Phase 2+
# ---------------------------------------------------------------------------

"""
    FaceIndexedFluxState{Basis, A, AZ} <: AbstractUnstructuredFaceFluxState

Face-centered mass fluxes for unstructured meshes (Phase 2+), tagged with
`Basis` for moist/dry safety.

# Type parameters
- `Basis <: AbstractMassFluxBasis` — `MoistMassFluxBasis` or `DryMassFluxBasis`

# Fields
- `horizontal_flux :: A` — mass flux per horizontal face [kg per half-timestep].
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
struct FaceIndexedFluxState{Basis <: AbstractMassFluxBasis,
                             A <: AbstractArray,
                             AZ <: AbstractArray} <: AbstractUnstructuredFaceFluxState
    horizontal_flux :: A
    cm              :: AZ
end

function FaceIndexedFluxState{B}(hflux::A, cm::AZ) where {B <: AbstractMassFluxBasis,
                                                            A <: AbstractArray,
                                                            AZ <: AbstractArray}
    FaceIndexedFluxState{B, A, AZ}(hflux, cm)
end

FaceIndexedFluxState(hflux, cm) = FaceIndexedFluxState{DryMassFluxBasis}(hflux, cm)

# ---------------------------------------------------------------------------
# Basis accessor
# ---------------------------------------------------------------------------

"""
    flux_basis(state) → AbstractMassFluxBasis

Return the mass flux basis tag for the given flux state.
"""
@inline flux_basis(::StructuredFaceFluxState{B}) where {B} = B()
@inline flux_basis(::FaceIndexedFluxState{B}) where {B} = B()

# ---------------------------------------------------------------------------
# Type alias for convenience
# ---------------------------------------------------------------------------

const DryStructuredFluxState = StructuredFaceFluxState{DryMassFluxBasis}
const MoistStructuredFluxState = StructuredFaceFluxState{MoistMassFluxBasis}

# ---------------------------------------------------------------------------
# Scoped accessor functions
#
# These are available for generic code and validation but the structured
# fast-path kernels bypass them and use am/bm/cm arrays directly.
# Accessors are basis-agnostic — they don't care whether fluxes are moist
# or dry.
# ---------------------------------------------------------------------------

@inline face_flux_x(s::AbstractStructuredFaceFluxState, i, j, k) = s.am[i, j, k]
@inline face_flux_y(s::AbstractStructuredFaceFluxState, i, j, k) = s.bm[i, j, k]
@inline face_flux_z(s::AbstractFaceFluxState, i, j, k)           = s.cm[i, j, k]
@inline face_flux(s::AbstractUnstructuredFaceFluxState, f, k)     = s.horizontal_flux[f, k]

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    allocate_face_fluxes(::StructuredFluxTopology, Nx, Ny, Nz;
                         FT=Float64, ArrayType=Array,
                         basis::Type{<:AbstractMassFluxBasis}=DryMassFluxBasis)

Allocate zeroed face flux arrays for a structured mesh.
The returned `StructuredFaceFluxState` is tagged with the specified `basis`.
"""
function allocate_face_fluxes(::StructuredFluxTopology,
                              Nx::Int, Ny::Int, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{<:AbstractMassFluxBasis} = DryMassFluxBasis)
    am = ArrayType(zeros(FT, Nx + 1, Ny,     Nz))
    bm = ArrayType(zeros(FT, Nx,     Ny + 1, Nz))
    cm = ArrayType(zeros(FT, Nx,     Ny,     Nz + 1))
    return StructuredFaceFluxState{basis}(am, bm, cm)
end

export AbstractMassFluxBasis, MoistMassFluxBasis, DryMassFluxBasis
export flux_basis
export DryStructuredFluxState, MoistStructuredFluxState
export AbstractFaceFluxState
export AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export allocate_face_fluxes
