# AtmosTransport architecture

AtmosTransport follows an Oceananigans-style composition model: grids,
states, forcing drivers, and physics operators are concrete values, while
Julia multiple dispatch selects topology- and backend-specific behavior.

## Dependency direction

```text
Architectures, Parameters
          ↓
Grids ── State
   \      /
    Operators
       ↓
MetDrivers ── Preprocessing
       \       /
         Models
           ↓
Output, Footprint, Inversion, Visualization
```

The runtime core never reads raw meteorology. Preprocessing converts source
data to the canonical v4 transport-binary contract; met drivers decode one
forcing window at a time into basis-explicit state and face-flux containers.

## Core composition

- `AtmosGrid(horizontal, vertical, architecture)` combines mesh geometry,
  hybrid pressure coordinates, and execution architecture.
- `CellState{Basis}` stores packed tracer mass on lat-lon or face-indexed
  reduced-Gaussian grids.
- `CubedSphereState{Basis}` stores six halo-padded panels.
- Face-flux state types mirror each topology and carry the same `DryBasis` or
  `MoistBasis` tag as the prognostic state.
- `TransportModel` owns concrete advection, diffusion, surface-flux,
  convection, and chemistry operators plus independent workspaces.
- `DrivenSimulation` owns the clock and the active met window; it refreshes
  forcing and delegates numerical work to `TransportModel`.

## Operator hierarchy and step order

All physics-family roots subtype `AbstractOperator` and use the shared
`apply!` protocol. Concrete `No*` operators represent inactive slots.

```text
transport palindrome
  X → Y → Z → diffusion/surface center → Z → Y → X
          ↓
convection
          ↓
chemistry
```

Advection, diffusion, and convection own separate workspaces. Direct operator
calls validate workspace topology, shape, element type, backend, and timestep
before mutating state. Model constructors allocate the correct workspaces.

## Supported topology/storage pairs

| Topology | State | Face fluxes | Advection |
|---|---|---|---|
| Lat-lon | `CellState` | `StructuredFaceFluxState` | upwind, slopes, PPM |
| Reduced Gaussian | `CellState` | `FaceIndexedFluxState` | upwind |
| Cubed sphere | `CubedSphereState` | `CubedSphereFaceFluxState` | split-sweep or Lin-Rood PPM |

Panel halos and topology-specific kernel layouts remain explicit because they
encode real numerical ownership differences. The shared orchestration lives in
the model/operator interfaces rather than flattening these representations.

## Canonical binary boundary

Only format version 4 is supported. A header fully declares geometry,
mass basis, sampling semantics, timestep schedule, payload sections, and
floating-point storage. Contradictory or obsolete geometry is rejected rather
than normalized through compatibility defaults.

The preprocessing boundary owns physical conversions such as dry-air mass,
convection entrainment/detrainment, exact TM5 `dkg`, and continuity closure.
Runtime operators consume these prepared quantities without source-specific
branches.

## Extension points

| Add | Subtype or extend | Required interface |
|---|---|---|
| Horizontal mesh | `AbstractHorizontalMesh` | geometry, connectivity, flux topology |
| Advection scheme | `AbstractAdvectionScheme` | topology-specific `apply!`/sweep methods |
| Diffusion | `AbstractDiffusion` | `apply!` and workspace construction |
| Surface flux | `AbstractSurfaceFluxOperator` | `apply!` / raw-buffer application |
| Convection | `AbstractConvection` | forcing validation, workspace, `apply!` |
| Chemistry | `AbstractChemistryOperator` | `apply!` |
| Met driver | `AbstractMetDriver` | timing, grid, basis, window loading |

Keep configuration parsing outside hot kernels. Runtime physics specs should
materialize concrete operator values before model construction.

## GPU execution

Numerical kernels use KernelAbstractions and dispatch from the storage array's
backend. CUDA and Metal support are loaded through package extensions. Backend
adaptation moves state, forcing, and each operator workspace together; scalar
GPU access is disabled in production runs.
