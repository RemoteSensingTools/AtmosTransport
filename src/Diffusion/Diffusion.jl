"""
    Diffusion

Vertical diffusion parameterizations with paired discrete adjoints.

Vertical diffusion in atmospheric transport is typically an implicit solve
(tridiagonal system). The adjoint is the transpose of the tridiagonal matrix,
solved with a transposed Thomas algorithm.

# Interface contract

    diffuse!(tracers, met, grid, diff::AbstractDiffusion, Δt)
    adjoint_diffuse!(adj_tracers, met, grid, diff::AbstractDiffusion, Δt)
"""
module Diffusion

using DocStringExtensions

using ..Grids: AbstractGrid
using ..Fields: AbstractField

export AbstractDiffusion, BoundaryLayerDiffusion, NoDiffusion
export diffuse!, adjoint_diffuse!
export DiffusionWorkspace, diffuse_gpu!, build_diffusion_coefficients

"""
$(TYPEDEF)

Supertype for vertical diffusion parameterizations.
"""
abstract type AbstractDiffusion end

"""
$(TYPEDEF)

No diffusion (pass-through). Adjoint is also a no-op.
"""
struct NoDiffusion <: AbstractDiffusion end

diffuse!(tracers, met, grid, ::NoDiffusion, Δt) = nothing
adjoint_diffuse!(adj_tracers, met, grid, ::NoDiffusion, Δt) = nothing

"""
$(TYPEDEF)

Boundary-layer vertical diffusion parameterization.
Uses diffusivity coefficients from met data (Kz).

Forward: implicit tridiagonal solve.
Adjoint: transposed tridiagonal solve (transposed Thomas algorithm).

$(FIELDS)
"""
struct BoundaryLayerDiffusion{FT} <: AbstractDiffusion
    "maximum allowed diffusivity [m²/s]"
    Kz_max :: FT
end

BoundaryLayerDiffusion(; Kz_max = 100.0) = BoundaryLayerDiffusion(Kz_max)

function diffuse!(tracers, met, grid, ::BoundaryLayerDiffusion, Δt)
    error("BoundaryLayerDiffusion forward not yet implemented")
end

function adjoint_diffuse!(adj_tracers, met, grid, ::BoundaryLayerDiffusion, Δt)
    error("BoundaryLayerDiffusion adjoint not yet implemented")
end

include("boundary_layer_diffusion.jl")
include("boundary_layer_diffusion_adjoint.jl")

end # module Diffusion
