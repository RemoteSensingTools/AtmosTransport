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
export DiffusionWorkspace, diffuse_gpu!, diffuse_cs_panels!, build_diffusion_coefficients

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
Parametric exponential Kz profile (largest near surface, decaying upward).

Forward: implicit tridiagonal solve (Thomas algorithm).
Adjoint: transposed tridiagonal solve (transposed Thomas algorithm).

$(FIELDS)
"""
struct BoundaryLayerDiffusion{FT} <: AbstractDiffusion
    "maximum diffusivity [Pa²/s in pressure coords]"
    Kz_max      :: FT
    "e-folding scale height in levels from surface"
    H_scale     :: FT
end

BoundaryLayerDiffusion(; Kz_max = 100.0, H_scale = 8.0) =
    BoundaryLayerDiffusion(Float64(Kz_max), Float64(H_scale))

include("boundary_layer_diffusion.jl")
include("boundary_layer_diffusion_adjoint.jl")

end # module Diffusion
