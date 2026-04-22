"""
    Convection

Convective transport operators (plan 18).

This submodule ships the convection-operator hierarchy:

- [`AbstractConvectionOperator`](@ref) — supertype for all convective
  transport operators.
- [`NoConvection`](@ref) — identity no-op; default for configurations
  without active convection. Dispatch is a compile-time dead branch
  in `TransportModel.step!` so the presence of the convection block
  has zero cost for users who don't opt in.

History (plan 18 + plan 23):

- Plan 18 Commit 1: type hierarchy + `NoConvection` no-op.
- Plan 18 Commit 2: `ConvectionForcing` in `..MetDrivers` with
  `copy_convection_forcing!`, `allocate_convection_forcing_like`,
  and window-struct integration.
- Plan 18 Commit 3: `CMFMCConvection` (GCHP path, CMFMC+DTRAIN
  kernel with mandatory CFL sub-cycling and well-mixed sub-cloud).
- Plan 22D: `step!`-level runtime block wiring across all
  three topologies (LL / RG / CS).
- Plan 23 Commit 1: `TM5Convection` struct + `TM5Workspace` +
  dispatch stubs. Commit 2 ships the column solver; Commit 4
  replaces stubs with real KA kernels on all three topologies.

## `apply!` contract (per plan 18 v5.1 §2.14 Decision 3)

    apply!(state::CellState{B},
           forcing::ConvectionForcing,
           grid::AtmosGrid,
           op::AbstractConvectionOperator,
           dt::Real;
           workspace) where {B <: AbstractMassBasis}

The operator takes `ConvectionForcing` directly (not a transport
window or driver). `_refresh_forcing!` populates
`model.convection_forcing` each substep by copying from
`sim.window.convection`. `TransportModel.step!` executes the convection
block between transport and chemistry. No `meteo` kwarg — the forcing
arrays are the time information; the operator does not call
`current_time`.

## Face-indexed scope

`CMFMCConvection` now supports structured LatLon, face-indexed
ReducedGaussian, and panel-native CubedSphere state layouts. The CS
path keeps forcing panel-native too: the driver loads `cmfmc` /
`dtrain` as per-panel tuples and the operator applies the same
column-local logic on each panel interior.

`TM5Convection` ships the same three-topology scope from the
first kernel commit (plan 23 Commit 4); Commit 1 lands the type,
workspace, and dispatch stubs so downstream wiring compiles
without any kernel work.

The no-op `NoConvection` path accepts any state shape — it's a pure
dead branch.
"""
module Convection

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState
using ...Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cell_areas_by_latitude
using ...MetDrivers: ConvectionForcing
import ..apply!

export AbstractConvectionOperator, NoConvection
export CMFMCConvection                          # plan 18 Commit 3
export CMFMCWorkspace, invalidate_cmfmc_cache!  # plan 18 Commit 3
export TM5Convection                            # plan 23 Commit 1
export TM5Workspace                             # plan 23 Commit 1
export apply_convection!

include("operators.jl")
include("convection_workspace.jl")   # CMFMCWorkspace (Commit 3) + TM5Workspace (plan 23)
include("cmfmc_kernels.jl")          # kernels + inline helpers (Commit 3)
include("CMFMCConvection.jl")        # struct + apply! methods (Commit 3)
include("tm5_column_solve.jl")       # backend-agnostic column solver (plan 23 Commit 2)
include("tm5_kernels.jl")            # @kernel wrappers per topology (plan 23 Commit 4)
include("TM5Convection.jl")          # struct + apply! methods (plan 23 Commit 1 + Commit 4)

end # module Convection
