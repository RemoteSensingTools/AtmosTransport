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

using ..Architectures: for_panels, for_panels_nosync, _kahan_add
using ..Grids: AbstractGrid
using ..Fields: AbstractField

export AbstractDiffusion, BoundaryLayerDiffusion, PBLDiffusion, NonLocalPBLDiffusion, NoDiffusion
export diffuse!, adjoint_diffuse!
export DiffusionWorkspace, diffuse_gpu!, diffuse_cs_panels!, build_diffusion_coefficients
export diffuse_pbl!, diffuse_nonlocal_pbl!

"""
$(TYPEDEF)

Supertype for vertical diffusion parameterizations.

# Interface contract

Subtypes must implement:

    diffuse!(tracers, met, grid, diff::YourDiffusion, Δt)
    adjoint_diffuse!(adj_tracers, met, grid, diff::YourDiffusion, Δt)

For GPU support, implement KernelAbstractions kernels that dispatch on
`get_backend(array)`. See `BoundaryLayerDiffusion` for a minimal example
and `NonLocalPBLDiffusion` for a full-featured implementation with
counter-gradient transport.

# Available implementations

- `NoDiffusion` — no-op pass-through
- `BoundaryLayerDiffusion` — static exponential Kz profile
- `PBLDiffusion` — met-data-driven Kz from PBLH, u*, HFLUX (TM5/Beljaars & Viterbo 1998)
- `NonLocalPBLDiffusion` — local + counter-gradient (Holtslag & Boville 1993 / GEOS-Chem VDIFF)
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

function BoundaryLayerDiffusion(; Kz_max = 100.0, H_scale = 8.0)
    Kz_max > 0 || throw(ArgumentError("Kz_max must be positive, got $Kz_max"))
    H_scale > 0 || throw(ArgumentError("H_scale must be positive, got $H_scale"))
    BoundaryLayerDiffusion(Float64(Kz_max), Float64(H_scale))
end

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

function PBLDiffusion(; β_h = 15.0, Kz_bg = 0.1, Kz_min = 0.01, Kz_max = 500.0)
    Kz_max > 0  || throw(ArgumentError("Kz_max must be positive, got $Kz_max"))
    Kz_bg  >= 0 || throw(ArgumentError("Kz_bg must be non-negative, got $Kz_bg"))
    Kz_min >= 0 || throw(ArgumentError("Kz_min must be non-negative, got $Kz_min"))
    Kz_min <= Kz_max || throw(ArgumentError("Kz_min ($Kz_min) must be ≤ Kz_max ($Kz_max)"))
    PBLDiffusion(Float64(β_h), Float64(Kz_bg), Float64(Kz_min), Float64(Kz_max))
end

"""
$(TYPEDEF)

Non-local PBL diffusion with counter-gradient transport following
Holtslag & Boville (1993) / GEOS-Chem VDIFF.

Extends `PBLDiffusion` with a counter-gradient term γ_c that represents
non-local transport by organized thermals in convective boundary layers.
The tridiagonal matrix (LHS) is identical to local K-diffusion; only the
RHS gets an additive source term from γ_c.

$(FIELDS)
"""
struct NonLocalPBLDiffusion{FT} <: AbstractDiffusion
    "Businger-Dyer heat parameter (TM5 default: 15.0)"
    β_h    :: FT
    "background Kz above PBL [m²/s]"
    Kz_bg  :: FT
    "minimum Kz in PBL [m²/s]"
    Kz_min :: FT
    "maximum allowed Kz [m²/s]"
    Kz_max :: FT
    "counter-gradient tuning constant (GEOS-Chem/Holtslag-Boville: 8.5)"
    fak    :: FT
    "surface layer fraction of PBL (default: 0.1)"
    sffrac :: FT
end

function NonLocalPBLDiffusion(; β_h = 15.0, Kz_bg = 0.1, Kz_min = 0.01, Kz_max = 500.0,
                               fak = 8.5, sffrac = 0.1)
    Kz_max > 0  || throw(ArgumentError("Kz_max must be positive, got $Kz_max"))
    Kz_bg  >= 0 || throw(ArgumentError("Kz_bg must be non-negative, got $Kz_bg"))
    Kz_min >= 0 || throw(ArgumentError("Kz_min must be non-negative, got $Kz_min"))
    Kz_min <= Kz_max || throw(ArgumentError("Kz_min ($Kz_min) must be ≤ Kz_max ($Kz_max)"))
    0 < sffrac < 1   || throw(ArgumentError("sffrac must be in (0,1), got $sffrac"))
    fak > 0          || throw(ArgumentError("fak must be positive, got $fak"))
    NonLocalPBLDiffusion(Float64(β_h), Float64(Kz_bg), Float64(Kz_min), Float64(Kz_max),
                         Float64(fak), Float64(sffrac))
end

include("boundary_layer_diffusion.jl")
include("boundary_layer_diffusion_adjoint.jl")
include("pbl_diffusion.jl")
include("nonlocal_pbl_diffusion.jl")

end # module Diffusion
