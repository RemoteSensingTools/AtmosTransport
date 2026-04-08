# Design Memo: Dry-Mass Transport Architecture for AtmosTransport.jl

> **Origin**: This memo was drafted in April 2026 as the guiding architectural
> document for restructuring AtmosTransport.jl toward a dry-mass finite-volume
> transport engine with host-model-specific meteorological adapters. It is the
> single source of truth for the `src_v2/` layered architecture.
>
> **Status**: Phase 1 (formalize dry-flux interface) is implemented on the
> `restructure/dry-flux-interface` branch. Phases 2–5 are future work.

---

## 1. Purpose

AtmosTransport should evolve into a dry-mass finite-volume transport engine with
host-model-specific meteorological adapters layered around it. The central idea
is that advection is universal, while diffusion, convection, and some aspects of
vertical closure depend on what the driving meteorological product actually
provides. This is already close to the current repo philosophy: the public README
describes a Julia atmospheric transport model inspired by TM5, using
Oceananigans-style multiple dispatch, with multi-grid and multi-backend support,
and with physics operators separated behind abstract types.

The key architectural move proposed here is to make **dry face mass fluxes the
formal interface between meteorology and transport**. The transport core should
never need to know whether fluxes were built from ERA5 spectral winds, GEOS-FP
gridded winds, or future host-model products. It should only need geometry, dry
cell mass, dry face fluxes, and tracer state.

## 2. Scope and Motivation

The current codebase already supports:

- latitude-longitude and cubed-sphere grids,
- CPU and GPU execution through KernelAbstractions,
- multiple met-data families including ERA5, GEOS-FP, and GEOS-IT,
- more than one advection scheme,
- and hand-coded discrete adjoints.

It also states explicitly that, in the ERA5 example, horizontal mass fluxes are
built from hybrid-level winds and vertical fluxes are diagnosed from horizontal
convergence to ensure column mass conservation, with the full transport loop
running on GPU.

That is the right foundation. What is still needed is a cleaner separation
between:

- generic transport operators,
- grid geometry,
- backend execution,
- and met-specific flux reconstruction and closure logic.

This becomes especially important if reduced Gaussian grids are added as a
first-class grid type. On reduced Gaussian grids, there is no globally aligned
"x-face" and "y-face" structure in the regular tensor-product sense; the natural
transport object is the face-normal flux on an unstructured control-volume mesh,
not a pair of structured directional flux arrays. That geometric reality should
shape the whole abstraction. The ECMWF finite-volume discussion for reduced
Gaussian grids points in that same direction by treating the grid through control
volumes and face fluxes rather than as a simple lat-lon C-grid.

## 3. Core Design Principle

The transport core should consume:

- cell-centered dry air mass,
- cell-centered tracer mass or mixing ratio,
- face-centered dry mass flux,
- geometry operators for cells and faces,
- and time integration / splitting logic.

It should **not** consume raw u, v, ω, vo, d, lnsp, q, or host-model-specific
hybrid metadata directly. Those belong in the met-adapter layer.

In compact form:

```
meteorology → dry flux builder → generic transport core
```

**This is the single most important interface boundary in the design.**

## 4. Proposed Layered Architecture

### 4.1 Geometry Layer

The geometry layer defines only the mesh and vertical coordinate. It knows
nothing about tracers, nothing about host-model meteorology, and nothing about
transport schemes beyond geometry queries.

```julia
abstract type AbstractHorizontalMesh end
abstract type AbstractVerticalCoordinate end
abstract type AbstractAtmosGrid end

struct AtmosGrid{H<:AbstractHorizontalMesh, V<:AbstractVerticalCoordinate, Arch, FT} <: AbstractAtmosGrid
    horizontal :: H
    vertical   :: V
    arch       :: Arch
end
```

Planned horizontal mesh types:

```julia
struct LatLonMesh{FT} <: AbstractHorizontalMesh ... end
struct CubedSphereMesh{FT} <: AbstractHorizontalMesh ... end
struct ReducedGaussianMesh{FT} <: AbstractHorizontalMesh ... end
```

The geometry API should be **face/cell oriented**, not index-direction oriented:

```julia
ncells(mesh)
nfaces(mesh)
nlevels(grid)

cell_area(mesh, c)
cell_volume(grid, c, k, state)

face_area(mesh, f)          # horizontal face length or full metric area
face_normal(mesh, f)        # local unit normal
face_cells(mesh, f)         # (left, right)
cell_faces(mesh, c)         # compressed adjacency list
```

For regular lat-lon, these functions can be implemented with simple structured
indexing. For reduced Gaussian, they expose the same interface but are backed by
explicit connectivity and metrics. That keeps the operator code grid-agnostic.

### 4.2 State Layer

The state layer separates prognostic cell fields from diagnostic face fields and
auxiliary meteorology.

```julia
struct CellState{A, Tr}
    air_dry_mass :: A          # (cell, level)
    tracers      :: Tr         # NamedTuple or custom tracer container
end

struct FaceFluxState{A}
    dry_mass_flux :: A         # (face, level)
end

struct MetState{A, M}
    ps        :: A
    q         :: A
    metvars   :: M             # winds, omega, vo/d, etadot, diffusivities, etc.
end
```

The advection operator should require only:

- `CellState`
- `FaceFluxState`
- `AtmosGrid`

Everything else is upstream preprocessing.

### 4.3 Met-Driver Layer

This layer is host-model aware.

```julia
abstract type AbstractMetDriver end

struct ERA5Driver    <: AbstractMetDriver end
struct GEOSFPDriver  <: AbstractMetDriver end
struct GEOSITDriver  <: AbstractMetDriver end
```

Responsibilities:

- read native meteorological fields,
- reconstruct pressures and layer masses,
- compute dry mass fluxes,
- diagnose or ingest vertical fluxes,
- enforce continuity / closure,
- map to the transport grid if needed.

This layer should own the messy details of data availability. The repo already
signals this need by advertising support for ERA5, GEOS-FP C720, and GEOS-IT
C180 with automatic regridding, which is exactly the kind of host-specific logic
that should not leak into core transport operators.

### 4.4 Operator Layer

The operator layer contains the transport and physics schemes.

```julia
abstract type AbstractOperator end
abstract type AbstractAdvection    <: AbstractOperator end
abstract type AbstractDiffusion    <: AbstractOperator end
abstract type AbstractConvection   <: AbstractOperator end
abstract type AbstractSourceSink   <: AbstractOperator end
```

Examples:

```julia
struct RussellLernerAdvection <: AbstractAdvection end
struct PPMAdvection{Order}    <: AbstractAdvection end

struct ImplicitBLDiffusion    <: AbstractDiffusion end
struct NoDiffusion            <: AbstractDiffusion end

struct HostConvection         <: AbstractConvection end
struct NoConvection           <: AbstractConvection end
```

These should remain pure numerical operators. The repo already advertises
Russell-Lerner slopes and Putman & Lin PPM, as well as operator splitting and
hand-coded adjoints, so this operator-oriented decomposition is already
consistent with current direction.

### 4.5 Backend Layer

Backend abstractions should remain orthogonal to the rest of the model. The
current code already uses KernelAbstractions for CPU and GPU execution, which is
exactly the right basis for this.

Kernel types should be organized by dependency pattern:

- **cell kernels** — source injection, chemistry, cell-mass updates, diagnostics
- **face kernels** — advection reconstruction, face flux evaluation
- **column kernels** — vertical diffusion, convection, tridiagonal solves

That is more stable than organizing them by physical process name alone.

## 5. Reduced Gaussian Grid Strategy

This is the most important geometry design issue.

On a reduced Gaussian grid, latitude rings have variable longitude count.
North-south cell boundaries do not line up one-to-one across adjacent rings.
Because of that, there is no clean global `yflux[i,j,k]` concept that remains
valid across the mesh.

So the transport core must be built around face-normal fluxes on a
control-volume mesh.

### 5.1 Internal Representation

Reduced Gaussian geometry should be stored in an explicitly connected mesh form,
even if preprocessing and I/O remain ring-based.

```julia
struct ReducedGaussianMesh{FT} <: AbstractHorizontalMesh
    latitudes      :: Vector{FT}
    nlon_per_ring  :: Vector{Int}
    lon_offsets    :: Vector{Int}

    face_cells     :: Vector{NTuple{2, Int}}
    cell_face_ptr  :: Vector{Int}          # CSR pointer
    cell_face_idx  :: Vector{Int}          # CSR indices

    face_normals   :: Matrix{FT}           # 3 × nfaces or local tangent
    face_lengths   :: Vector{FT}
    cell_areas     :: Vector{FT}
    cell_centers   :: Matrix{FT}
end
```

The ring metadata is useful for decoding native reduced Gaussian data, building
connectivity, and debugging. But the transport core should see only: cell areas,
face metrics, face normals, and connectivity.

### 5.2 Transport Implications

All horizontal advection schemes should operate on:

- cell states,
- face-normal dry mass fluxes,
- face reconstruction stencils.

They should **never** assume that flux directions correspond to globally aligned
longitude and latitude axes.

For reduced Gaussian, a thread should ideally process one face at a time:

1. reconstruct left and right face states,
2. compute upwinded tracer state,
3. multiply by dry mass flux,
4. store face tracer flux,
5. reduce face fluxes back into cell divergence.

This is the natural pattern for both CPU and GPU.

## 6. Dry-Mass Formulation

The transport model should be built around dry mass consistently.

### 6.1 Cell Dry Mass

For a hybrid pressure layer,

$$\Delta p_k = p_{k+1/2} - p_{k-1/2}$$

Total moist mass per area is

$$m_k = \frac{\Delta p_k}{g}$$

Dry mass per area is

$$m_k^{\mathrm{dry}} = \frac{\Delta p_k}{g}(1 - q_k)$$

where $q$ is specific humidity defined relative to total moist mass, as in ECMWF
products. The dry correction is therefore straightforward once $q$ is known.
ERA5's native model-level state uses hybrid vertical coordinates and provides
lnsp and $q$, so this formulation is aligned with the available meteorology.

### 6.2 Face Dry Mass Flux

The universal object passed into advection should be:

$$F_f^{\mathrm{dry}} = (\rho_d \mathbf{u}) \cdot \hat{n}_f \, A_f$$

or, in discrete mass-flux form, the dry mass crossing face $f$ per time per
level.

Transport of a tracer mass mixing ratio $\chi$ then becomes a conservative flux:

$$\Phi_f = F_f^{\mathrm{dry}} \, \chi_f^{\mathrm{upwind}}$$

This is independent of whether the host meteorology came from ERA5, GEOS-FP, or
something else.

### 6.3 Vertical Dry Continuity

A dedicated continuity module should diagnose or correct vertical dry fluxes so
that column dry mass closes exactly.

```julia
abstract type AbstractMassClosure end

struct DiagnoseVerticalFromHorizontal <: AbstractMassClosure end
struct PressureTendencyClosure        <: AbstractMassClosure end
struct NativeVerticalFluxClosure      <: AbstractMassClosure end
```

This is important because the current public README already describes diagnosing
vertical fluxes from horizontal convergence in the ERA5 example to guarantee
column mass conservation. That logic should become a formal interchangeable
module rather than remaining implicit inside one driver.

## 7. Advection Interface

Advection should accept only dry mass fluxes plus geometry and tracer state.

Proposed interface:

```julia
apply!(state_new, state_old, fluxes, grid, op::AbstractAdvection, Δt)
```

**Not:**

```julia
apply!(state_new, state_old, u, v, w, ps, q, grid, ...)
```

The second form hard-wires meteorology assumptions into the advection scheme. The
first form keeps advection universal.

## 8. Capability-Based Host-Model Plugins

Because diffusion and convection depend on what the driving meteorology actually
provides, these should be capability-based rather than assumed.

```julia
supports_diffusion(::AbstractMetDriver) = false
supports_convection(::AbstractMetDriver) = false
supports_native_vertical_flux(::AbstractMetDriver) = false
supports_dry_mass_closure(::AbstractMetDriver) = true
```

Then:

```julia
supports_diffusion(::GEOSFPDriver) = true
supports_convection(::GEOSFPDriver) = true
```

This avoids contaminating the core with met-specific if/else logic and fits the
repo's stated goal of extensible physics operators behind abstract types.

## 9. CPU/GPU Kernel Structure

The backend-neutral kernel design should be organized around three kernel motifs.

### 9.1 Cell Kernels

Use for: source injection, simple local chemistry hooks, cell-mass updates,
diagnostics, boundary conditions. One thread per cell-level.

### 9.2 Face Kernels

Use for: horizontal advection reconstruction, face flux evaluation, face metric
application. One thread per face-level.

**This is the most important kernel family for reduced Gaussian support.**

### 9.3 Column Kernels

Use for: vertical diffusion, convection closures, pressure-thickness
reconstruction, dry mass closure, any tridiagonal solve. One thread or
cooperative workgroup per column.

This also aligns with the existing README note that vertical diffusion is handled
with an implicit Thomas solver and that the full loop runs on GPU.

## 10. Data Layout Recommendations

### 10.1 Prognostic State

For GPU efficiency, keep the fast-varying dimension contiguous and avoid deep
nesting of small Julia structs in hot loops.

Preferred:

- flat arrays or strided arrays for cell-level fields,
- compressed sparse row style adjacency for mesh connectivity,
- separate arrays for geometry metrics.

```julia
air_dry_mass[cell, level]
tracer_mass[cell, level, tracer]
dry_mass_flux[face, level]
```

### 10.2 Connectivity

For unstructured or semi-structured meshes:

```julia
face_left[face]
face_right[face]

cell_face_ptr[cell]
cell_face_idx[idx]
```

This is preferable to vectors of vectors and maps well to CPU cache and GPU
global memory access.

### 10.3 Geometry Caches

Precompute and store: cell areas, face lengths, face normals, interpolation or
reconstruction metadata, ring metadata for reduced Gaussian I/O only.

Geometry should be immutable for a given grid instance.

## 11. Multiple Dispatch Structure

The architecture should use multiple dispatch for four orthogonal axes:

- grid geometry,
- backend,
- met driver,
- numerical operator.

For example:

```julia
build_dry_fluxes!(fluxes, met, grid::AtmosGrid{<:ReducedGaussianMesh}, driver::ERA5Driver, closure)
build_dry_fluxes!(fluxes, met, grid::AtmosGrid{<:LatLonMesh},          driver::ERA5Driver, closure)

apply!(state_new, state_old, fluxes, grid::AtmosGrid{<:ReducedGaussianMesh}, op::RussellLernerAdvection, Δt)
apply!(state_new, state_old, fluxes, grid::AtmosGrid{<:CubedSphereMesh},     op::PPMAdvection{7}, Δt)
```

That is much closer to the Oceananigans spirit of separating grid, architecture,
and numerical operators than pushing everything through one giant monolithic
TransportModel implementation.

## 12. Suggested Module Layout

```
AtmosTransport/
  Grids/
    AbstractMeshes.jl
    LatLonMesh.jl
    CubedSphereMesh.jl
    ReducedGaussianMesh.jl
    VerticalCoordinates.jl
    GeometryOps.jl

  State/
    CellState.jl
    FaceFluxState.jl
    MetState.jl
    Tracers.jl

  MetDrivers/
    AbstractMetDriver.jl
    ERA5/
      Reader.jl
      SpectralReconstruction.jl
      DryFluxBuilder.jl
      Closure.jl
    GEOSFP/
      Reader.jl
      DryFluxBuilder.jl
      Closure.jl
    GEOSIT/
      Reader.jl
      DryFluxBuilder.jl
      Closure.jl

  Operators/
    AbstractOperators.jl
    Advection/
      RussellLerner.jl
      PPM.jl
      FaceReconstruction.jl
      Divergence.jl
    Diffusion/
      ImplicitBLDiffusion.jl
    Convection/
      HostConvection.jl
    Sources/
      SurfaceFluxes.jl

  Kernels/
    CellKernels.jl
    FaceKernels.jl
    ColumnKernels.jl

  Solvers/
    StrangSplitting.jl
    TimeStepping.jl
    Diagnostics.jl

  Adjoint/
    Checkpointing.jl
    AdjointOperators.jl
```

This is not a call to rewrite the repo from scratch. It is a target structure for
making the current design more explicit.

## 13. Execution Flow

Recommended timestep flow:

```julia
read_met!(met, driver, t)

build_geometry_dependent_fields!(geomcache, grid, met)
build_dry_fluxes!(fluxes, met, grid, driver, closure)

apply!(state, fluxes, grid, advection, Δt)

if supports_convection(driver)
    apply!(state, met, grid, convection, Δt)
end

if supports_diffusion(driver)
    apply!(state, met, grid, diffusion, Δt)
end

apply!(state, sources, grid, emissions, Δt)

diagnostics!(diag, state, grid)
```

Only the met-read and dry-flux-build steps are host-model aware. Everything else
is generic.

## 14. Adjoint Design Implications

The project already advertises a hand-coded discrete adjoint with Revolve
checkpointing.

The proposed separation helps a lot here:

- the transport adjoint should depend only on fluxes, geometry, and scheme
  stencils,
- the met-preprocessing adjoint should be separate and optional.

That avoids entangling the adjoint of tracer transport with the adjoint of ERA5
spectral reconstruction, humidity correction, or continuity closure. Those are
different problems and should stay different in code.

A clean split would be:

- `AdjointTransportOperators`
- `AdjointMetFluxBuilders`

The former is essential. The latter can come later if needed for inversion
against meteorological controls.

## 15. Development Phases

### Phase 1: Formalize Dry-Flux Interface

Refactor advection so all schemes consume dry face fluxes and geometry only.

**Deliverable:** universal advection API, no direct dependence on raw winds in
transport kernels.

### Phase 2: Reduced Gaussian Geometry Package

Implement reduced Gaussian mesh generation, connectivity, and geometry tests.

**Deliverable:** `ReducedGaussianMesh`, conservative face-cell divergence
identities, mesh metrics verified on the sphere.

### Phase 3: ERA5 Dry-Flux Builder on Reduced Gaussian

Implement native reduced-Gaussian dry flux generation from ERA5 fields.

**Deliverable:** pressure reconstruction, dry layer mass, horizontal dry face
fluxes, vertical dry closure.

### Phase 4: Backend-Optimized Face Kernels

Develop face-loop transport kernels for CPU and GPU.

**Deliverable:** generalized face-based advection engine, performance benchmark
against lat-lon and cubed-sphere structured kernels.

### Phase 5: Capability-Driven Diffusion and Convection

Move host-specific physics support behind capability traits.

**Deliverable:** advection universal, diffusion and convection optional and
modular.

## 16. Recommended Immediate Coding Decisions

The most important immediate choices are these.

1. Define `FaceFluxState` and make it the one true transport input.
2. Add a face/cell geometry API that every mesh must satisfy, even if lat-lon and
   cubed sphere use optimized structured implementations underneath.
3. Treat reduced Gaussian as a mesh with connectivity, not as a quirky structured
   grid.
4. Isolate all host-met logic in dedicated dry-flux builders.
5. Keep adjoint transport separate from met preprocessing.

## 17. Bottom Line

AtmosTransport should be centered on one invariant:

> **Conservative transport of tracer mass using dry face mass fluxes on an
> abstract mesh.**

Everything else is a plugin:

- which mesh,
- which met source,
- which closure,
- which advection scheme,
- which backend.

---

## Addendum: Phase 1 Implementation Notes (April 2026)

The following design refinements were made during Phase 1 implementation based on
detailed review:

### Flux Topology Trait

Rather than forcing a single `FaceFluxState{A}` with
`dry_mass_flux[face, level]` for all meshes, the implementation uses a **flux
topology trait** that lets each mesh advertise its natural flux storage:

```julia
abstract type AbstractFluxTopology end
struct StructuredFluxTopology  <: AbstractFluxTopology end
struct FaceIndexedFluxTopology <: AbstractFluxTopology end

flux_topology(::AbstractStructuredMesh)  = StructuredFluxTopology()
flux_topology(::AbstractHorizontalMesh)  = FaceIndexedFluxTopology()
```

### Abstract Face-Flux Hierarchy

The face flux state uses a two-level abstraction: operator contract at the
mathematical level, storage freedom at the implementation level.

```julia
abstract type AbstractFaceFluxState end
abstract type AbstractStructuredFaceFluxState   <: AbstractFaceFluxState end
abstract type AbstractUnstructuredFaceFluxState  <: AbstractFaceFluxState end

struct StructuredFaceFluxState{AX, AY, AZ} <: AbstractStructuredFaceFluxState
    am :: AX   # x-face dry mass flux
    bm :: AY   # y-face dry mass flux
    cm :: AZ   # z-face dry mass flux
end
```

**Rationale**: Do not unify at the storage level. Unify at the
mathematical/operator level. Structured grids keep `am`/`bm`/`cm` as the fast
path. Unstructured meshes get face-indexed storage. The operator dispatch selects
the right realization for each mesh/flux topology pair.

### Kernel Strategy

Phase 1 retains proven cell-loop kernels for structured grids. Face-loop kernels
will be added in Phase 2+ when reduced Gaussian arrives. The abstract transport
interface is face-oriented from day one, but the structured fast-path bypasses
generic accessors for performance.
