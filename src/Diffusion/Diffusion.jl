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

export AbstractDiffusion, BoundaryLayerDiffusion, PBLDiffusion, NoDiffusion
export diffuse!, adjoint_diffuse!
export DiffusionWorkspace, diffuse_gpu!, diffuse_cs_panels!, build_diffusion_coefficients
export diffuse_pbl!

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

"""
$(TYPEDEF)

Met-data-driven PBL diffusion following TM5's revised LTG scheme
(Beljaars & Viterbo 1998). Computes Kz profiles from PBLH, u*, and
sensible heat flux using Monin-Obukhov similarity theory.

Unlike `BoundaryLayerDiffusion` (static exponential Kz), this scheme
produces spatially and temporally varying Kz that depends on stability.

$(FIELDS)
"""
struct PBLDiffusion{FT} <: AbstractDiffusion
    "Businger-Dyer heat parameter (TM5 default: 15.0)"
    β_h    :: FT
    "background Kz above PBL [m²/s] (TM5 default: 0.1)"
    Kz_bg  :: FT
    "minimum Kz in PBL [m²/s]"
    Kz_min :: FT
    "maximum allowed Kz [m²/s] (safety clamp)"
    Kz_max :: FT
end

PBLDiffusion(; β_h = 15.0, Kz_bg = 0.1, Kz_min = 0.01, Kz_max = 500.0) =
    PBLDiffusion(Float64(β_h), Float64(Kz_bg), Float64(Kz_min), Float64(Kz_max))

include("boundary_layer_diffusion.jl")
include("boundary_layer_diffusion_adjoint.jl")
include("pbl_diffusion.jl")

end # module Diffusion
