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

# Diffusion is included BEFORE Advection so `strang_split_mt!`
# (plan 16b Commit 4 palindrome integration) can import
# `NoDiffusion`, `AbstractDiffusionOperator`, and
# `apply_vertical_diffusion!`. Diffusion has no dependency on
# Advection; reordering preserves correctness.
include("Diffusion/Diffusion.jl")
using .Diffusion

# SurfaceFlux is included BEFORE Advection so `strang_split_mt!`
# (plan 17 Commit 5 palindrome integration) can import
# `NoSurfaceFlux`, `AbstractSurfaceFluxOperator`, and
# `apply_surface_flux!`. Commit 2 ships only the data types
# (`SurfaceFluxSource`, `PerTracerFluxMap`); the operator types
# and kernel land in Commit 3.
include("SurfaceFlux/SurfaceFlux.jl")
using .SurfaceFlux

include("Advection/Advection.jl")
using .Advection

include("Chemistry/Chemistry.jl")
using .Chemistry

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

# Diffusion solver infrastructure + operator types (plan 16b Commits 2-4)
export solve_tridiagonal!, build_diffusion_coefficients
export AbstractDiffusionOperator, NoDiffusion, ImplicitVerticalDiffusion
export apply_vertical_diffusion!

# SurfaceFlux data types (plan 17 Commit 2); operator hierarchy lands in Commit 3.
export SurfaceFluxSource, PerTracerFluxMap, flux_for

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs!, CSAdvectionWorkspace

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export fillz_q!, apply_divergence_damping_cs!

end # module Operators
