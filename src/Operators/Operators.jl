"""
    Operators

Physics operators for the basis-explicit transport architecture.

Provides:
- Abstract operator types (`AbstractAdvection`, `AbstractDiffusion`, etc.)
- New advection hierarchy: `AbstractConstantScheme`, `AbstractLinearScheme`,
  `AbstractQuadraticScheme` with concrete `UpwindScheme`, `SlopesScheme`, and structured-grid `PPMScheme`
- Legacy advection types: `UpwindAdvection`, `RussellLernerAdvection`, `PPMAdvection`
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

include("Chemistry/Chemistry.jl")
using .Chemistry

# Re-export legacy advection types
export AbstractConstantReconstruction, AbstractLinearReconstruction, AbstractQuadraticReconstruction
export UpwindAdvection, FirstOrderUpwindAdvection, RussellLernerAdvection, PPMAdvection, ppm_order
export AdvectionWorkspace, strang_split!, strang_split_mt!
export TracerView
export diagnose_cm!

# Re-export new advection hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter
export UpwindScheme, SlopesScheme, PPMScheme
export reconstruction_order

# Chemistry
export AbstractChemistry, NoChemistry, RadioactiveDecay, CompositeChemistry
export apply_chemistry!

# Cubed-sphere advection
export fill_panel_halos!, strang_split_cs!, CSAdvectionWorkspace

end # module Operators
