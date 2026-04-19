"""
    Operators

Physics operators for the basis-explicit transport architecture.

Provides:
- Abstract operator types (`AbstractOperator`, `AbstractDiffusion`, etc.)
- Advection hierarchy: `AbstractAdvectionScheme` → `AbstractConstantScheme`,
  `AbstractLinearScheme`, `AbstractQuadraticScheme` with concrete
  `UpwindScheme`, `SlopesScheme`, and structured-grid `PPMScheme`
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

include("Diffusion/Diffusion.jl")
using .Diffusion

export AdvectionWorkspace, strang_split!, strang_split_mt!
export TracerView
export diagnose_cm!

# Advection scheme hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter
export UpwindScheme, SlopesScheme, PPMScheme
export reconstruction_order

# Chemistry
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export chemistry_block!

# Diffusion solver infrastructure (plan 16b Commit 2)
export solve_tridiagonal!, build_diffusion_coefficients

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs!, CSAdvectionWorkspace

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export fillz_q!, apply_divergence_damping_cs!

end # module Operators
