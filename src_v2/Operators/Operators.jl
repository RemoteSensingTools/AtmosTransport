"""
    Operators (v2)

Physics operators for the dry-mass transport architecture.

Provides:
- Abstract operator types (`AbstractAdvection`, `AbstractDiffusion`, etc.)
- Advection schemes: `FirstOrderUpwindAdvection`, `RussellLernerAdvection`, `PPMAdvection`
- Strang splitting orchestrator: `strang_split!`, `apply!`
- Vertical flux diagnosis: `diagnose_cm!`
"""
module Operators

# Re-export State and Grids into Operators scope for sub-submodules
using ..State
using ..Grids
using ..MetDrivers

include("AbstractOperators.jl")
include("Advection/Advection.jl")
using .Advection

# Re-export advection types and functions
export FirstOrderUpwindAdvection, RussellLernerAdvection, PPMAdvection, ppm_order
export AdvectionWorkspace, strang_split!
export diagnose_cm!

end # module Operators
