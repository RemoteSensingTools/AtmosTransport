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

Commit 1 ships only the type hierarchy and no-op. Future commits:

- Commit 3: `CMFMCConvection` (GCHP path, CMFMC+DTRAIN kernel with
  mandatory CFL sub-cycling and well-mixed sub-cloud treatment).
- Commit 4: `TM5Convection` (TM5 path, four-field matrix scheme with
  in-kernel LU solve).
- Commit 2: `ConvectionForcing` already lives in `..MetDrivers` —
  Commit 2 extends it with validating constructors, `copy_convection_forcing!`,
  `allocate_convection_forcing_like`, and window-struct integration.

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
`sim.window.convection`. `TransportModel.step!` does not consume that
forcing yet, so the end-to-end structured convection runtime remains a
gated follow-up. No `meteo` kwarg — the forcing arrays are the time
information; the operator does not call `current_time`.

## Face-indexed scope

Plan 18 is structured-only (v5.1 §2.19 Decision 25). Face-indexed
state (`Raw <: AbstractArray{_, 3}` tracers) raises `ArgumentError`
from the concrete operator `apply!` methods shipping in Commits 3
and 4, pointing at Plan 18b as the follow-up.

The no-op `NoConvection` path in Commit 1 accepts any state shape —
it's a pure dead branch.
"""
module Convection

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState
using ...Grids: AtmosGrid, LatLonMesh, cell_areas_by_latitude
using ...MetDrivers: ConvectionForcing
import ..apply!

export AbstractConvectionOperator, NoConvection
export CMFMCConvection                          # plan 18 Commit 3
export CMFMCWorkspace, invalidate_cmfmc_cache!  # plan 18 Commit 3
export apply_convection!

include("operators.jl")
include("convection_workspace.jl")   # CMFMCWorkspace (Commit 3)
include("cmfmc_kernels.jl")          # kernels + inline helpers (Commit 3)
include("CMFMCConvection.jl")        # struct + apply! methods (Commit 3)

end # module Convection
