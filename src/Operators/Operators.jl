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
using ..SectionTimer
using ..State
using ..Grids
using ..MetDrivers

include("AbstractOperators.jl")

# Diffusion is included BEFORE Advection so `strang_split_mt!`
# (plan 16b Commit 4 palindrome integration) can import
# `NoDiffusion`, `AbstractDiffusion`, and
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

# Convection is included before Advection. Plan 18 v5.1 Decision 1
# runs convection as a SEPARATE block in `TransportModel.step!`
# (between the transport palindrome and the chemistry block), so
# `strang_split_mt!` doesn't need the convection types. The include
# order still puts Convection alongside Diffusion/SurfaceFlux (both
# column/point-local physics) for consistency.
include("Convection/Convection.jl")
using .Convection

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
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme
export reconstruction_order, required_halo_width

# Chemistry
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export chemistry_block!

# Diffusion solver infrastructure + operator types (plan 16b Commits 2-4)
export solve_tridiagonal!, build_diffusion_coefficients
export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion
export apply_vertical_diffusion!

# SurfaceFlux data types + operator hierarchy (plan 17 Commits 2-3)
export SurfaceFluxSource, PerTracerFluxMap, flux_for
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!

# Convection operator hierarchy (plan 18 + plan 23).
# NoConvection, CMFMCConvection live since plan 18; TM5Convection
# lands via plan 23 (Commit 1: types + dispatch stubs; Commit 4:
# real kernels on all three topologies).
export AbstractConvection, NoConvection
export CMFMCConvection                          # plan 18 Commit 3
export CMFMCWorkspace, invalidate_cmfmc_cache!  # plan 18 Commit 3
export TM5Convection                            # plan 23 Commit 1
export TM5Workspace                             # plan 23 Commit 1
export apply_convection!

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs!, CSAdvectionWorkspace

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, CSLinRoodAdvectionWorkspace
export fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export fillz_q!, apply_divergence_damping_cs!

end # module Operators
