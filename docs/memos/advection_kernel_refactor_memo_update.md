# Memo update: advection-kernel refactor plan for AtmosTransport

## Purpose

This memo updates the advection-kernel refactor plan after comparing two design directions and grounding them against the actual target use case:

- offline mass-flux atmospheric transport
- TM5 / FV3 / GCHP-style advection logic
- ERA5 / GEOS-family preprocessed binary inputs
- GPU-capable Julia implementation
- adjoint-friendly kernels and dataflow
- support for multiple grids, especially:
  - rectangular lat-lon
  - cubed sphere
  - reduced Gaussian

The main conclusion is:

**the right foundation is a mass-flux transport core with tracer-mass prognostics, diagnostic reconstruction workspaces, topology-specialized kernel shells, and preprocessing-heavy binaries.**

This memo is intended as a planning note for Claude and Codex and as a design checkpoint for the ongoing `src` refactor.

---

## Executive summary

There is broad agreement on the high-level architecture:

- unified outer API
- type-driven dispatch
- explicit topology traits
- separate kernel families by topology
- SoA layout for GPU
- aggressive preprocessing into binary files
- explicit limiter types
- basis attached to the flux object
- adjoint-friendliness as a first-class concern

The important update is that we should **not** organize the advection core around prognostic subcell moments by default. That is elegant, but it matches a Prather-style second-order-moments framework more than the actual target models you care about.

Instead, the core should be organized like TM5 / FV3 / GCHP mass-flux transport:

- **prognostic variable**: tracer mass `rm`
- **air mass**: `m`
- **face mass fluxes**: `am, bm, cm` for structured grids, or `hflux, vflux` for face-indexed grids
- **subcell reconstruction**: diagnostic, recomputed from `rm / m` during each sweep
- **reconstruction family**:
  - constant / upwind
  - linear / slopes
  - quadratic / PPM
- **kernel shells**:
  - structured directional sweeps
  - face-indexed horizontal/vertical kernels
- **basis enforcement**: at the API boundary, not in kernel dispatch

The main design principle is:

**same operator contract, different specialized concrete paths underneath.**

---

## What is settled

The following points are now effectively settled and should not be reopened unless there is a strong reason:

### 1. Unified outer API

There should be one high-level advection call, conceptually of the form:

```julia
advect!(rm_new, rm_old, m, fluxes, grid, scheme, Δt)
```

or equivalent, where dispatch determines the correct specialization.

### 2. Topology-trait dispatch

The mesh should advertise a topology trait, e.g.

```julia
flux_topology(mesh)
```

with at least:

- `StructuredDirectionalTopology`
- `FaceIndexedTopology`

This chooses the kernel family and geometry access pattern.

### 3. Separate topology-specific kernel motifs

Do not force all grids into one low-level kernel pattern.

Use:

- structured directional sweeps for rectangular lat-lon, and probably initially for cubed sphere
- face-indexed kernels for reduced Gaussian and later genuinely unstructured meshes

### 4. Structure-of-arrays for GPU

Hot-loop data should be stored as separate arrays, not nested object graphs.

Good:
- `rm[cell, lev]`
- `m[cell, lev]`
- `am[i, j, k]`
- `bm[i, j, k]`
- `cm[i, j, k]`
- `slope_x[cell, lev]`

Bad:
- per-cell nested structs with many fields inside kernels

### 5. Aggressive preprocessing

Anything that does not depend on the instantaneous tracer state should be pushed into preprocessing as much as possible:

- geometry
- topology
- connectivity
- merge maps
- face metrics
- panel transforms
- donor-independent reconstruction geometry
- vertical metadata
- interpolation weights if stable and reusable

### 6. Explicit limiter types

Limiters should be explicit policy objects, not hidden inside reconstruction code.

### 7. Basis attached to flux object

Mass basis belongs on the flux object, not just on the producer/driver.

But see below: basis should **not** generally enter kernel dispatch unless the math actually changes.

---

## Main architectural conclusion

The correct base architecture is **mass-flux transport with diagnostic reconstruction**, not prognostic subcell moments.

That means:

- the transported prognostic quantity is tracer mass `rm`
- the transported air mass is `m`
- face mass fluxes come from the met / preprocessing path
- the reconstruction for upwind, slopes, or PPM is computed on demand from the current tracer state
- the reconstruction workspace is temporary / diagnostic, not the main prognostic state

This matches:
- TM5
- FV3
- GCHP
- your existing v1 logic and reference expectations

This does **not** mean prognostic moments are wrong in general. It means they are the wrong default foundation for this codebase.

---

## Key divergences that were resolved

## Divergence 1: should basis enter kernel dispatch?

### Rejected default
Dispatching kernels directly on `DryBasis` vs `MoistBasis`.

### Chosen approach
Basis is enforced at the API boundary and on the flux-state type, but the kernel math is basis-agnostic unless the actual kernel equations differ.

Reason:
If the kernel is doing something like

```julia
rm_new = rm_old + flux_divergence
```

then the update math is identical whether the flux object carries dry or moist basis. Requiring separate GPU kernels for `DryBasis` and `MoistBasis` would duplicate code without scientific benefit.

### Implication
Use basis tagging for:
- validation
- API checks
- conversion correctness
- documentation
- file semantics

Do not use basis to split kernels unless the numerical formula itself changes.

---

## Divergence 2: what is prognostic?

### Rejected default
Making slopes or curvature part of the main tracer state by default.

### Chosen approach
Tracer mass `rm` is the prognostic variable.
Subcell reconstruction data are diagnostic workspace.

Reason:
This matches TM5 / FV3 / GCHP logic, avoids large memory blow-up, and is easier to keep adjoint-friendly.

### Consequences

For all standard mass-flux schemes:
- store `rm`
- compute `q = rm / m`
- compute reconstruction workspace from `q`
- compute face flux
- update `rm`

Do **not** evolve slopes or curvatures as the default state.

### Note
If you later want a true Prather-style second-order moments option, it should be added as an **additional scheme family**, not as the organizing principle for the whole codebase.

---

## Divergence 3: geometric Taylor evaluation vs Courant-number flux-form reconstruction

### Rejected default
Treating face values as pointwise polynomial evaluations at a geometric offset.

### Chosen approach
Use mass-flux finite-volume reconstruction based on the **Courant fraction** of the donor cell.

Reason:
In TM5 / FV3 / GCHP-style transport, the face quantity is not the tracer value at a geometric point. It is the average tracer content of the donor-cell fraction swept through the face during the timestep.

So reconstruction should depend on something like

```julia
alpha = |F| / m_donor
```

where `alpha` is the fraction of the donor cell mass leaving through that face during the substep.

This is the correct finite-volume mass-flux view.

### Implication
For linear or PPM reconstruction, the face contribution should be computed from the swept donor fraction, not from a pointwise polynomial evaluation at a face center.

---

## Divergence 4: what gets passed into kernels?

### Rejected default
Passing rich composite objects like full `grid`, `scheme`, or complex tracer-state objects deep into GPU kernels.

### Chosen approach
Use dispatch outside the kernel to choose a concrete path, then pass only the arrays and small concrete arguments the kernel truly needs.

Reason:
This is safer for GPU execution and easier to reason about for adjoints.

Good kernel arguments:
- `@Const(rm)`
- `@Const(m)`
- `@Const(am)`
- `@Const(bm)`
- `@Const(cm)`
- reconstruction workspace arrays
- small integer dimensions
- compact geometry arrays if needed

Avoid passing:
- large mesh objects with incidental metadata
- generic scheme objects if avoidable
- anything non-concrete or not GPU-compatible

---

## Divergence 5: deep file tree vs shared shells with dispatch

### Rejected default
A large cross-product tree like:
- `Structured/Upwind...`
- `Structured/Slopes...`
- `Structured/PPM...`
- `FaceIndexed/Upwind...`
- `FaceIndexed/Slopes...`
- `FaceIndexed/PPM...`

if most of the shell logic is duplicated.

### Chosen approach
Use:
- shared kernel shells by topology family
- shared reconstruction helpers by scheme
- dispatch and inlining to specialize behavior

Reason:
This avoids combinatorial code duplication while preserving specialization.

### Practical rule
Split files when the algorithmic shell genuinely differs.
Do not split files just because the scheme label differs.

---

## The correct organizing principle

The core abstraction should be:

### 1. Flux basis
Carried by the flux object for validation and API correctness.

### 2. Topology
Determines kernel family and geometry access pattern.

### 3. Reconstruction family
Determines how reconstruction workspace is computed and how donor-face values are evaluated.

### 4. Prognostic transport state
Tracer mass and air mass.

---

## Recommended design

## Type axes

### Mass basis

```julia
abstract type AbstractMassBasis end
struct MoistBasis <: AbstractMassBasis end
struct DryBasis   <: AbstractMassBasis end
```

### Topology

```julia
abstract type AbstractFluxTopology end
struct StructuredDirectionalTopology <: AbstractFluxTopology end
struct FaceIndexedTopology           <: AbstractFluxTopology end
```

### Reconstruction family

```julia
abstract type AbstractReconstruction end
struct ConstantReconstruction  <: AbstractReconstruction end
struct LinearReconstruction    <: AbstractReconstruction end
struct QuadraticReconstruction <: AbstractReconstruction end
```

### Scheme types

```julia
abstract type AbstractAdvectionScheme{R<:AbstractReconstruction} end

struct UpwindScheme <: AbstractAdvectionScheme{ConstantReconstruction} end

struct SlopesScheme{L} <: AbstractAdvectionScheme{LinearReconstruction}
    limiter :: L
end

struct PPMScheme{L} <: AbstractAdvectionScheme{QuadraticReconstruction}
    limiter :: L
end
```

### Limiters

```julia
abstract type AbstractLimiter end
struct NoLimiter <: AbstractLimiter end
struct MonotoneLimiter <: AbstractLimiter end
struct PositivityLimiter <: AbstractLimiter end
```

---

## Flux-state types

The basis belongs here.

```julia
abstract type AbstractFaceFluxState{B<:AbstractMassBasis} end

struct StructuredFaceFluxState{B, AX, AY, AZ} <: AbstractFaceFluxState{B}
    am :: AX
    bm :: AY
    cm :: AZ
end

struct FaceIndexedFluxState{B, AH, AV} <: AbstractFaceFluxState{B}
    hflux :: AH
    vflux :: AV
end
```

This allows:
- `StructuredFaceFluxState{MoistBasis}`
- `StructuredFaceFluxState{DryBasis}`
- `FaceIndexedFluxState{DryBasis}`

But again: basis should not normally split GPU kernels.

---

## Prognostic state

Keep this minimal.

```julia
struct TracerMassState{A}
    rm :: A
end

struct AirMassState{A}
    m :: A
end
```

or equivalent.

The key is:
- `rm` is prognostic
- `m` is carried from met/preprocessing or updated consistently as required
- slopes/PPM edges are not the default prognostic state

---

## Reconstruction workspace

This is the most important idea to adopt as a clean dispatch point.

### New recommended stage

```julia
compute_reconstruction!(ws, rm, m, grid, scheme)
```

This should be an explicit dispatch point.

For example:
- upwind: no real workspace, maybe just `q = rm / m`
- slopes: donor-cell linear slopes
- PPM: left/right edge states and curvature-like coefficients

This is where the family-specific logic should live.

### Why this matters
It cleanly separates:
- prognostic state
- reconstruction data
- flux shell
- divergence update

This is both:
- GPU-friendly
- adjoint-friendly
- easy to benchmark
- easy to compare across schemes

---

## Recommended operator decomposition

The runtime transport path should be decomposed into explicit phases.

## Phase 1: reconstruction

```julia
compute_reconstruction!(ws, rm, m, grid, scheme)
```

Outputs scheme-specific reconstruction workspace.

Examples:
- constant: maybe `q`
- slopes: limited slopes
- PPM: `ql`, `qr`, and `q6` or equivalent

## Phase 2: face flux

Structured:
```julia
xface_flux_kernel!(...)
yface_flux_kernel!(...)
zface_flux_kernel!(...)
```

Face-indexed:
```julia
hface_flux_kernel!(...)
vface_flux_kernel!(...)
```

These compute tracer mass fluxes from:
- mass fluxes
- reconstruction workspace
- donor selection
- scheme-specific face evaluation

## Phase 3: divergence update

```julia
divergence_update_kernel!(...)
```

This updates tracer mass from face flux divergence.

This explicit 3-step decomposition is recommended for:
- clarity
- adjoints
- debugging
- testing
- parity checking

---

## Correct face evaluation philosophy

This is critical.

For mass-flux transport, the face quantity should be interpreted as the tracer content of the **fraction of the donor cell swept through the face during the timestep**.

So the relevant parameter is something like:

```julia
alpha = abs(F) / m_donor
```

not a geometric offset from cell center to face center.

### Therefore:
- upwind uses donor-cell mean
- slopes uses a linear donor reconstruction integrated over the swept donor fraction
- PPM uses the parabolic donor reconstruction integrated over the swept donor fraction

This is the correct TM5 / FV3 / GCHP-style finite-volume view.

---

## Kernel structure by topology

## StructuredDirectionalTopology

This is the correct family for:
- rectangular lat-lon
- likely initially cubed sphere per tile/panel

Recommended kernel shells:
- `xface_flux_kernel!`
- `yface_flux_kernel!`
- `zface_flux_kernel!`
- `divergence_update_kernel!`

These kernels should:
- be simple
- read flat arrays
- use precomputed workspace
- use inlined scheme-specific helper functions

### Example conceptual shell

```julia
@kernel function xface_flux_kernel!(
    fx,                 # tracer face flux output
    rm, m, am,          # state + mass flux
    recon_ws,           # scheme-specific reconstruction workspace
    Nx, Ny, Nz
)
    i, j, k = @index(Global, NTuple)

    F = am[i, j, k]
    donor = ifelse(F >= 0, i-1, i)
    alpha = abs(F) / m[donor, j, k]

    qface = face_value_x(recon_ws, rm, m, donor, i, j, k, alpha)
    fx[i, j, k] = F * qface
end
```

The helper `face_value_x(...)` should dispatch or inline-specialize on the scheme/reconstruction family.

---

## FaceIndexedTopology

This is the correct family for:
- reduced Gaussian
- later general face-connected meshes

Recommended kernel shells:
- `hface_flux_kernel!`
- `vface_flux_kernel!`
- `divergence_update_kernel!`

### Example conceptual shell

```julia
@kernel function hface_flux_kernel!(
    fh,
    rm, m, hflux,
    face_left, face_right,
    recon_ws
)
    f, k = @index(Global, NTuple)

    F = hflux[f, k]
    left  = face_left[f]
    right = face_right[f]
    donor = ifelse(F >= 0, left, right)

    alpha = abs(F) / m[donor, k]
    qface = face_value_h(recon_ws, rm, m, donor, f, k, alpha)
    fh[f, k] = F * qface
end
```

Again:
- flat arrays
- explicit connectivity
- minimal branching
- scheme logic isolated in reconstruction helpers

---

## What should not be prognostic by default

Do **not** default to storing these as the main tracer state:
- `sx`
- `sy`
- `sz`
- `sxx`
- `syy`
- `szz`
- cross moments

unless you deliberately add a Prather-style moment scheme later.

Why:
- high memory cost
- more update kernels
- more adjoint complexity
- not needed for TM5/FV3/GCHP-style mass-flux transport
- not what your current reference models do

---

## Where Prather-style moments fit

A true moment-transport scheme can still be added later.

If that ever happens, it should be an explicit additional scheme family, not the default backbone.

For example:

```julia
struct SecondMomentScheme{L} <: AbstractAdvectionScheme{QuadraticReconstruction}
    limiter :: L
end
```

paired with its own prognostic state and its own update kernels.

That is a future extension, not the foundation.

---

## Adjoint-friendly design rules

This refactor must keep adjoints in mind from the start.

The advection path should be written so that reverse logic is as local and explicit as possible.

### Recommended adjoint-friendly principles

1. **Minimal hidden state**
   - avoid implicit updates buried inside helper functions
   - prefer explicit workspace and explicit outputs

2. **Simple dataflow**
   - reconstruct
   - face flux
   - divergence update
   Each phase should be inspectable and replayable.

3. **No unnecessary runtime branching**
   - scheme dispatch outside kernels
   - topology dispatch outside kernels
   - kernels should be as static as possible

4. **Flat array arguments**
   - easier to reason about in reverse
   - easier for hand-written adjoints
   - safer for GPU

5. **Precompute donor-independent geometry**
   - do not reconstruct mesh geometry inside kernels

6. **Keep temporary workspace explicit**
   - makes checkpointing and reverse accumulation cleaner

7. **Preserve clear mass-flow semantics**
   - donor selection
   - Courant fraction
   - reconstructed donor fraction
   - flux divergence update

### Recommended conceptual adjoint structure

Forward:
- `compute_reconstruction!`
- `face_flux_kernel!`
- `divergence_update_kernel!`

Reverse:
- adjoint of divergence update
- adjoint of face flux
- adjoint of reconstruction

This is much easier than reversing a monolithic fused kernel.

---

## Preprocessing-first philosophy

As much as possible should be done in preprocessing of the binary files.

### Precompute and store when possible

For structured grids:
- grid metrics
- vertical metadata
- merge maps
- CFL-relevant normalization factors if stable
- panel-edge mapping data for cubed sphere

For face-indexed grids:
- face connectivity
- `face_left`, `face_right`
- `cell_face_ptr`, `cell_face_idx`
- face lengths / areas
- face normals
- cell areas
- face-local tangent data if needed
- any donor-independent geometric coefficients for reconstruction

For vertical handling:
- interface metadata
- level maps
- merge maps
- precomputed reduced-level aggregation information

### Principle
Push complexity upstream whenever it does **not** depend on instantaneous tracer state.

This helps:
- performance
- reproducibility
- adjoint clarity
- simpler runtime kernels
- consistency across CPU/GPU
- consistency across grids

---

## Binary-file implications

The binary format should support this design cleanly.

### For structured directional grids
Core advection payload:
- `m`
- `am`
- `bm`
- `cm`
- `ps`
- vertical metadata
- optional humidity if needed for basis conversion or dry construction

Optional extended blocks:
- temperature
- convection
- diagnostics
- delta fields
- surface terms

### For face-indexed grids
Core advection payload:
- `m[cell, lev]`
- `hflux[face, lev]`
- `vflux[cell, interface]`
- `ps[cell]`
- geometry / connectivity block
- vertical metadata
- optional humidity

### Top-level rule
Homogenize the binary family at the **semantic envelope** level, not by forcing all grids into the same raw storage shape.

---

## Recommended module structure

Avoid a deep combinatorial tree if most of the shell is shared.

A better structure is something like:

```text
Advection/
  AbstractSchemes.jl
  Basis.jl
  Reconstruction.jl
  Limiters.jl

  KernelShells/
    StructuredKernels.jl
    FaceIndexedKernels.jl
    DivergenceKernels.jl

  ReconstructionFamilies/
    Upwind.jl
    Slopes.jl
    PPM.jl

  Workspace/
    ReconstructionWorkspace.jl

  Adjoint/
    AdjointStructured.jl
    AdjointFaceIndexed.jl
    AdjointReconstruction.jl
```

The idea is:
- topology-specific shell code in one place
- reconstruction-family code in one place
- no unnecessary scheme × topology file explosion

---

## Recommended outer dispatch pattern

The outer API should select topology and validate basis, then dispatch to specialized implementations.

Conceptually:

```julia
function advect!(rm_new, rm_old, m, fluxes, grid, scheme, Δt)
    validate_basis(fluxes, scheme, grid)
    _advect!(rm_new, rm_old, m, fluxes, grid, flux_topology(grid), scheme, Δt)
end
```

Then:

```julia
function _advect!(..., ::StructuredDirectionalTopology, scheme, Δt)
    compute_reconstruction!(...)
    xface_flux_kernel!(...)
    yface_flux_kernel!(...)
    zface_flux_kernel!(...)
    divergence_update_kernel!(...)
end

function _advect!(..., ::FaceIndexedTopology, scheme, Δt)
    compute_reconstruction!(...)
    hface_flux_kernel!(...)
    vface_flux_kernel!(...)
    divergence_update_kernel!(...)
end
```

---

## Immediate implementation priorities

## 1. Formalize `compute_reconstruction!`

This should be added explicitly now.

That is the most useful design improvement to take from the broader abstraction discussion.

### Goal
Make reconstruction a clear dispatch point:
- upwind
- slopes
- PPM

without changing the kernel shells.

---

## 2. Keep kernel arguments flat

Refactor or preserve the shell so kernels receive:
- arrays
- dimensions
- connectivity arrays
- small geometry arrays if necessary

not rich composite objects.

---

## 3. Keep basis checks outside kernels

Continue to carry basis on the flux object, but enforce it outside the GPU kernels unless the numerical formula truly differs.

---

## 4. Keep reconstruction diagnostic

Do not let the refactor drift toward prognostic moments unless deliberately implementing a different scheme family.

---

## 5. Keep preprocessing aggressive

Especially for:
- reduced Gaussian connectivity
- face geometry
- cubed-sphere panel boundary transforms
- any donor-independent reconstruction geometry

---

## 6. Keep adjoint phases explicit

Do not fuse everything into one opaque kernel unless there is a compelling performance reason and a clear reverse strategy.

---

## Recommended validation plan

The architecture and real-data proof are already in good shape. The next validation work should focus on scientifically meaningful refinements, especially:

### 1. Exact moist-to-dry face correction
Replace approximate
- `F_dry ≈ F_moist`

with exact or more consistent
- `F_dry = F_moist * (1 - q_face)`

using a clearly defined face-humidity policy.

### 2. Compare reconstruction families
Once the new shell is stable:
- constant / upwind
- linear / slopes
- quadratic / PPM

on the same real-data path.

### 3. Keep diagnostics explicit
Measure:
- total tracer mass drift
- dry air mass drift if relevant
- per-column closure residual
- field differences relative to reference path
- cost and memory scaling

---

## Final design statement

The refactor should be built around this principle:

**Use a mass-flux transport core with tracer-mass prognostics, scheme-dispatched diagnostic reconstruction workspaces, topology-specialized kernel shells, and preprocessing-heavy binaries.**

That means:

- basis is attached to flux objects, but kernel math stays basis-agnostic unless physically necessary
- tracer mass is prognostic
- reconstruction is diagnostic
- face evaluation uses Courant-fraction finite-volume logic
- structured and face-indexed topologies each get their own kernel shells
- preprocessing carries as much geometry and donor-independent complexity as possible
- runtime kernels remain simple, explicit, and adjoint-friendly

That is the correct foundation for:
- TM5-parity
- FV3 / GCHP-style PPM
- reduced Gaussian support
- cube-sphere support
- efficient GPU execution in Julia
- future adjoints and inversions

---

## Implementation status (2026-04-09)

### Completed: Phase A — scaffolding, Phase B — SlopesScheme

Files added/modified in `src/Operators/Advection/`:

| File | Role |
|------|------|
| `schemes.jl` | Type hierarchy: `AbstractConstantScheme`, `AbstractLinearScheme`, `AbstractQuadraticScheme` with concrete `UpwindScheme`, `SlopesScheme{L}`, `PPMScheme{L}`. Full docstrings with equations and citations. |
| `limiters.jl` | `@inline` branchless limiters: `_minmod3`, `_limited_slope`, `_limited_moment` dispatching on `NoLimiter`, `MonotoneLimiter`, `PositivityLimiter`. Documented with CW84/van Leer/Sweby equations. |
| `reconstruction.jl` | `@inline` face flux functions for `UpwindScheme`, `SlopesScheme`, and `PPMScheme`. Periodic wrap via `mod1`, clamped boundaries for y/z. Full derivation of Courant-fraction flux formula. |
| `structured_kernels.jl` | Universal `@kernel` shells: `_xsweep_kernel!`, `_ysweep_kernel!`, `_zsweep_kernel!` — thin 5-line indexing shells calling `@inline` face flux. Documented mass conservation proof. |
| `multitracer_kernels.jl` | `TracerView` wrapper + multi-tracer kernel shells `_xsweep_mt_kernel!`, `_ysweep_mt_kernel!`, `_zsweep_mt_kernel!` — fuses N-tracer loop into GPU kernel. |
| `StrangSplitting.jl` | Refactored with `@eval` loops: single/multi-tracer sweeps, face-indexed helpers, `apply!` entry points. `strang_split!` and `strang_split_mt!`. |

**Equivalence validated** (0 ULP in all cases):
- `UpwindScheme` ↔ `UpwindAdvection`: bit-identical, F32 + F64
- `SlopesScheme(MonotoneLimiter())` ↔ `RussellLernerAdvection(use_limiter=true)`: bit-identical
- `SlopesScheme(NoLimiter())` ↔ `RussellLernerAdvection(use_limiter=false)`: bit-identical

**Bug found and fixed**: `_wrap_periodic` used single-step `ifelse` wrapping that failed for SlopesScheme's 4-point stencil (`face_i - 2` can be -1). Fixed with `mod1` — also future-proofs for PPM's wider stencils.

### Completed: Phase C — multi-tracer kernel fusion

Reduces kernel launches from 6N to 6 per Strang split for N tracers.

**Design**: A lightweight `TracerView{FT, A}` struct wraps a 4D array `(Nx, Ny, Nz, Nt)` and presents it as a 3D array with a fixed tracer index. The multi-tracer kernel shells create `TracerView(rm_4d, t)` inside the inner loop and call the SAME `_xface_tracer_flux` / `_yface_tracer_flux` / `_zface_tracer_flux` functions as the single-tracer kernels. Julia inlines and compiles away the wrapper entirely.

```julia
struct TracerView{FT, A <: AbstractArray{FT, 4}}
    data::A
    t::Int32
end
Base.@propagate_inbounds Base.getindex(v::TracerView, i, j, k) = v.data[i, j, k, v.t]

@kernel function _xsweep_mt_kernel!(rm_new_4d, @Const(rm_4d), m_new, @Const(m),
                                     @Const(am), scheme, Nx, Nt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        m_new[i,j,k] = m[i,j,k] + am[i,j,k] - am[i+1,j,k]  # once
        for t in Int32(1):Int32(Nt)
            rm_t = TracerView(rm_4d, t)
            flux_L = _xface_tracer_flux(Int32(i), j, k, rm_t, m, am[i,j,k], scheme, Nx)
            flux_R = _xface_tracer_flux(Int32(i)+Int32(1), j, k, rm_t, m, am[i+1,j,k], scheme, Nx)
            rm_new_4d[i,j,k,t] = rm_4d[i,j,k,t] + flux_L - flux_R
        end
    end
end
```

**Key advantage**: zero code duplication — `TracerView` lets multi-tracer kernels reuse all existing face flux functions. Single-tracer and multi-tracer paths produce BIT-IDENTICAL results (verified for all 3 schemes × 2 precisions).

### Completed: Phase D — PPMScheme (quadratic reconstruction)

`PPMScheme{L} <: AbstractQuadraticScheme` — full face flux implementation for all 3 directions.

**Edge interpolation**: 4th-order CW84 formula (eq. 1.6):

    e_{i+1/2} = (7/12)(c_i + c_{i+1}) - (1/12)(c_{i-1} + c_{i+2})

**Profile limiting**: CW84 monotonicity constraints (eqs. 1.10a–c) dispatched on `MonotoneLimiter`, `NoLimiter`, `PositivityLimiter`.

**Flux formula**: Same Courant-fraction formula as SlopesScheme (via `_slopes_face_flux`), but with PPM edge-offset moments:
- For F > 0 (left donor): sx = m_L · (q_R_L - c̄_L)
- For F < 0 (right donor): sx = m_R · (c̄_R - q_L_R)

This matches the production PPM code in `src/Advection/latlon_mass_flux_ppm.jl` and Putman & Lin (2007).

**Stencil**: 6 cells per face (3 per side). Periodic via `mod1` (x), clamped + fallback-to-upwind at boundaries (y, z).

### Validation summary (164 tests, all pass)

| Test category | Count | Result |
|---|---|---|
| UpwindScheme: {CPU,GPU} × {F32,F64} | 40 | ✓ |
| SlopesScheme: {CPU,GPU} × {F32,F64} | 20 | ✓ |
| PPMScheme: {CPU,GPU} × {F32,F64} | 20 | ✓ |
| Legacy ↔ new scheme equivalence | varied | ✓ bit-identical |
| Multi-tracer ≡ per-tracer (3 schemes × 2 precisions) | 36 | ✓ bit-identical |
| Multi-tracer mass conservation (5 tracers) | 36 | ✓ machine-eps |
| Multi-tracer CPU-GPU agreement | 12 | ✓ < 16 ULP |
| **Total** | **164** | **All pass** |

**Cross-backend/precision** (all paths):
- CPU-GPU agreement: ≤ 0.25 ULP (F64), ≤ 0.25 ULP (F32)
- F32 vs F64 bias: < 1.5e-6 relative, < 2.2e-8 global mass bias
