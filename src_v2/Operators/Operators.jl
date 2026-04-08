"""
    Operators (v2)

Physics operators for the dry-mass transport architecture.

Provides:
- Abstract operator types (`AbstractAdvection`, `AbstractDiffusion`, etc.)
- Advection schemes: `RussellLernerAdvection`, `PPMAdvection`
- Strang splitting orchestrator: `strang_split!`, `apply!`
- Vertical flux diagnosis: `diagnose_cm!`
"""
module Operators

# Re-export State and Grids into Operators scope for sub-submodules
using ..State
using ..Grids

include("AbstractOperators.jl")
include("Advection/Advection.jl")
using .Advection

# Re-export advection types and functions
export RussellLernerAdvection, PPMAdvection, ppm_order
export AdvectionWorkspace, strang_split!, sweep_x!, sweep_y!, sweep_z!
export minmod, van_leer_slope
export max_cfl_x, max_cfl_y, max_cfl_z
export diagnose_cm!

end # module Operators
