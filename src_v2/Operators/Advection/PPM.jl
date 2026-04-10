# ---------------------------------------------------------------------------
# PPM (Piecewise Parabolic Method) advection — v2 interface stub
#
# This file keeps the legacy `PPMAdvection` wrapper as an explicit stub while
# the new `PPMScheme` path matures. Structured generic kernels live in
# `reconstruction.jl` / `structured_kernels.jl`; the legacy wrapper remains
# unsupported until a clean migration/replacement is chosen.
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
    throw(ArgumentError("PPMAdvection remains a legacy stub in src_v2; use PPMScheme for the structured generic kernel path"))
end

export PPMAdvection, ppm_order
