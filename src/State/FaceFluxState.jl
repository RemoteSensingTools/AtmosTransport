# ---------------------------------------------------------------------------
# FaceFluxState — face-centered mass fluxes for transport
#
# Abstraction lives at the operator contract level, not the storage level:
#
#   abstract type AbstractFaceFluxState end
#       ├── AbstractStructuredFaceFluxState   (am, bm, cm on logically rectangular meshes)
#       └── AbstractUnstructuredFaceFluxState  (face-indexed connectivity, Phase 2+)
#
# Each concrete flux state carries a `Basis <: AbstractMassBasis` type
# parameter that records whether the stored fluxes are on a moist or dry
# basis. This prevents accidentally mixing state/flux basis at dispatch time.
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
abstract type AbstractFaceFluxState{Basis <: AbstractMassBasis} end

"""
    AbstractStructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as separate directional arrays on a logically rectangular
mesh.  Concrete subtypes expose `am` (x-face), `bm` (y-face), `cm` (z-face).

Structured cell-loop kernels access these arrays directly for performance.
"""
abstract type AbstractStructuredFaceFluxState{Basis <: AbstractMassBasis} <: AbstractFaceFluxState{Basis} end

"""
    AbstractUnstructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as a single face-indexed array with explicit connectivity.
Natural for reduced Gaussian and other unstructured meshes (Phase 2+).
"""
abstract type AbstractUnstructuredFaceFluxState{Basis <: AbstractMassBasis} <: AbstractFaceFluxState{Basis} end

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
- `am :: AX` — x-face (longitude) prepared substep mass transport [kg for the active transport substep].
  Layout: `(Nx+1, Ny, Nz)` for LatLon, `(Nc+1, Nc, Nz)` per panel for CS.
- `bm :: AY` — y-face (latitude) prepared substep mass transport [kg for the active transport substep].
  Layout: `(Nx, Ny+1, Nz)` for LatLon, `(Nc, Nc+1, Nz)` per panel for CS.
- `cm :: AZ` — z-face (vertical) prepared substep mass transport [kg for the active transport substep].
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
DryBasis()

julia> moist = StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm);

julia> flux_basis(moist)
MoistBasis()
```
"""
struct StructuredFaceFluxState{Basis <: AbstractMassBasis,
                                AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState{Basis}
    am :: AX
    bm :: AY
    cm :: AZ
end

function StructuredFaceFluxState{B}(am::AX, bm::AY, cm::AZ) where {B <: AbstractMassBasis,
                                                                       AX <: AbstractArray,
                                                                       AY <: AbstractArray,
                                                                       AZ <: AbstractArray}
    StructuredFaceFluxState{B, AX, AY, AZ}(am, bm, cm)
end

StructuredFaceFluxState(am, bm, cm) = StructuredFaceFluxState{DryMassFluxBasis}(am, bm, cm)

function Adapt.adapt_structure(to, fluxes::StructuredFaceFluxState{B}) where {B <: AbstractMassBasis}
    am = Adapt.adapt(to, fluxes.am)
    bm = Adapt.adapt(to, fluxes.bm)
    cm = Adapt.adapt(to, fluxes.cm)
    return StructuredFaceFluxState{B, typeof(am), typeof(bm), typeof(cm)}(am, bm, cm)
end

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
- `horizontal_flux :: A` — prepared substep mass transport per horizontal face [kg for the active transport substep].
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
struct FaceIndexedFluxState{Basis <: AbstractMassBasis,
                             A <: AbstractArray,
                             AZ <: AbstractArray} <: AbstractUnstructuredFaceFluxState{Basis}
    horizontal_flux :: A
    cm              :: AZ
end

"""
    CubedSphereFaceFluxState{Basis, AX, AY, AZ} <: AbstractStructuredFaceFluxState

Panel-native structured-directional flux storage for cubed-sphere transport.
Each field is an `NTuple{6}` of halo-padded panel arrays matching the
`strang_split_cs!` contract.
"""
struct CubedSphereFaceFluxState{Basis <: AbstractMassBasis,
                                AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState{Basis}
    am :: NTuple{6, AX}
    bm :: NTuple{6, AY}
    cm :: NTuple{6, AZ}
end

function CubedSphereFaceFluxState{B}(am::NTuple{6}, bm::NTuple{6}, cm::NTuple{6}) where {B <: AbstractMassBasis}
    return CubedSphereFaceFluxState{B, typeof(am[1]), typeof(bm[1]), typeof(cm[1])}(am, bm, cm)
end

function FaceIndexedFluxState{B}(hflux::A, cm::AZ) where {B <: AbstractMassBasis,
                                                            A <: AbstractArray,
                                                            AZ <: AbstractArray}
    FaceIndexedFluxState{B, A, AZ}(hflux, cm)
end

FaceIndexedFluxState(hflux, cm) = FaceIndexedFluxState{DryMassFluxBasis}(hflux, cm)

function Adapt.adapt_structure(to, fluxes::FaceIndexedFluxState{B}) where {B <: AbstractMassBasis}
    hflux = Adapt.adapt(to, fluxes.horizontal_flux)
    cm = Adapt.adapt(to, fluxes.cm)
    return FaceIndexedFluxState{B, typeof(hflux), typeof(cm)}(hflux, cm)
end

function Adapt.adapt_structure(to, fluxes::CubedSphereFaceFluxState{B}) where {B <: AbstractMassBasis}
    am = Adapt.adapt(to, fluxes.am)
    bm = Adapt.adapt(to, fluxes.bm)
    cm = Adapt.adapt(to, fluxes.cm)
    return CubedSphereFaceFluxState{B, typeof(am[1]), typeof(bm[1]), typeof(cm[1])}(am, bm, cm)
end

# ---------------------------------------------------------------------------
# Basis accessor
# ---------------------------------------------------------------------------

"""
    flux_basis(state) → AbstractMassFluxBasis

Return the mass flux basis tag for the given flux state.
"""
@inline flux_basis(::StructuredFaceFluxState{B}) where {B} = B()
@inline flux_basis(::FaceIndexedFluxState{B}) where {B} = B()
@inline flux_basis(::CubedSphereFaceFluxState{B}) where {B} = B()
@inline mass_basis(::StructuredFaceFluxState{B}) where {B} = B()
@inline mass_basis(::FaceIndexedFluxState{B}) where {B} = B()
@inline mass_basis(::CubedSphereFaceFluxState{B}) where {B} = B()

# ---------------------------------------------------------------------------
# Type alias for convenience
# ---------------------------------------------------------------------------

const DryStructuredFluxState = StructuredFaceFluxState{DryMassFluxBasis}
const MoistStructuredFluxState = StructuredFaceFluxState{MoistMassFluxBasis}
const DryCubedSphereFluxState = CubedSphereFaceFluxState{DryMassFluxBasis}
const MoistCubedSphereFluxState = CubedSphereFaceFluxState{MoistMassFluxBasis}
const FluxState = AbstractFaceFluxState

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
                              basis::Type{B} = DryMassFluxBasis) where {B <: AbstractMassBasis}
    am = ArrayType(zeros(FT, Nx + 1, Ny,     Nz))
    bm = ArrayType(zeros(FT, Nx,     Ny + 1, Nz))
    cm = ArrayType(zeros(FT, Nx,     Ny,     Nz + 1))
    return StructuredFaceFluxState{B}(am, bm, cm)
end

"""
    allocate_face_fluxes(::FaceIndexedFluxTopology, nfaces, ncells, Nz;
                         FT=Float64, ArrayType=Array,
                         basis::Type{<:AbstractMassBasis}=DryMassFluxBasis)

Allocate zeroed face-indexed flux arrays for a connected-face mesh.
"""
function allocate_face_fluxes(::FaceIndexedFluxTopology,
                              nfaces::Int, ncells::Int, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryMassFluxBasis) where {B <: AbstractMassBasis}
    hflux = ArrayType(zeros(FT, nfaces, Nz))
    cm = ArrayType(zeros(FT, ncells, Nz + 1))
    return FaceIndexedFluxState{B}(hflux, cm)
end

"""
    allocate_face_fluxes(mesh::AbstractStructuredMesh, Nz; kwargs...)

Allocate a flux container using the mesh's natural structured topology.
"""
function allocate_face_fluxes(mesh::AbstractStructuredMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryMassFluxBasis) where {B <: AbstractMassBasis}
    return allocate_face_fluxes(StructuredFluxTopology(), nx(mesh), ny(mesh), Nz;
                                FT=FT, ArrayType=ArrayType, basis=B)
end

function allocate_face_fluxes(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryMassFluxBasis) where {B <: AbstractMassBasis}
    N = mesh.Nc + 2 * mesh.Hp
    am = ntuple(_ -> ArrayType(zeros(FT, N + 1, N, Nz)), 6)
    bm = ntuple(_ -> ArrayType(zeros(FT, N, N + 1, Nz)), 6)
    cm = ntuple(_ -> ArrayType(zeros(FT, N, N, Nz + 1)), 6)
    return CubedSphereFaceFluxState{B}(am, bm, cm)
end

"""
    allocate_face_fluxes(mesh::AbstractHorizontalMesh, Nz; kwargs...)

Allocate a flux container using the mesh's natural face-connected topology.
"""
function allocate_face_fluxes(mesh::AbstractHorizontalMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryMassFluxBasis) where {B <: AbstractMassBasis}
    return allocate_face_fluxes(FaceIndexedFluxTopology(), nfaces(mesh), ncells(mesh), Nz;
                                FT=FT, ArrayType=ArrayType, basis=B)
end

export flux_basis
export DryStructuredFluxState, MoistStructuredFluxState
export DryCubedSphereFluxState, MoistCubedSphereFluxState
export AbstractFaceFluxState, FluxState
export AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState, CubedSphereFaceFluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export allocate_face_fluxes
