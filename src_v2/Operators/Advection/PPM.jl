# ---------------------------------------------------------------------------
# PPM (Piecewise Parabolic Method) advection — v2 interface stub
#
# Phase 1 provides the type and dispatch entry point. The actual PPM
# kernels from src/Advection/ppm_advection.jl and
# src/Advection/ppm_subgrid_distributions.jl will be ported in Phase 2.
# ---------------------------------------------------------------------------

"""
    PPMAdvection{ORD} <: AbstractQuadraticReconstruction

Piecewise Parabolic Method advection following Putman & Lin (2007).
`ORD` is the reconstruction order (4, 5, 6, or 7).

# Fields
- `use_limiter :: Bool` — enable monotonicity constraint
"""
struct PPMAdvection{ORD} <: AbstractQuadraticReconstruction
    use_limiter :: Bool
end
PPMAdvection(; order::Int = 4, use_limiter::Bool = true) =
    PPMAdvection{order}(use_limiter)

ppm_order(::PPMAdvection{ORD}) where ORD = ORD

function apply!(state::CellState{B},
                fluxes::StructuredFaceFluxState{B},
                grid::AtmosGrid{<:AbstractStructuredMesh},
                ::PPMAdvection, dt;
                kwargs...) where {B <: AbstractMassBasis}
    throw(ArgumentError("PPMAdvection is Phase 4+: kernels are not implemented in src_v2 yet"))
end

export PPMAdvection, ppm_order
