"""
    Chemistry

Atmospheric chemistry and loss-rate framework.

Provides a hierarchy of chemistry types for applying species-specific
transformations (decay, photolysis, deposition, reactions) to tracer fields.

# Type hierarchy

    AbstractChemistry
    ├── NoChemistry               — inert tracers (no-op)
    ├── AbstractFirstOrderLoss    — first-order loss processes
    │   └── RadioactiveDecay      — uniform constant decay (e.g. ²²²Rn)
    └── CompositeChemistry        — combine multiple schemes for multi-tracer runs

The framework is designed for extension: future implementations can add
spatially varying loss rates `k(x,y,z)`, time-varying rates `k(t)`,
or full chemical mechanisms by subtyping `AbstractChemistry`.

# Interface contract

    apply_chemistry!(tracers, grid, chem::AbstractChemistry, Δt)
    adjoint_chemistry!(adj_tracers, grid, chem::AbstractChemistry, Δt)
"""
module Chemistry

using DocStringExtensions

export AbstractChemistry, NoChemistry
export AbstractFirstOrderLoss, RadioactiveDecay
export CompositeChemistry
export apply_chemistry!, adjoint_chemistry!

# =====================================================================
# Abstract types
# =====================================================================

"""
$(TYPEDEF)

Supertype for chemistry schemes. Subtypes implement forward + adjoint methods.
"""
abstract type AbstractChemistry end

"""
$(TYPEDEF)

Supertype for first-order loss processes of the form:

    dc/dt = -k * c

where `k` is the loss rate (s⁻¹). Subtypes differ in how `k` is specified:
- `RadioactiveDecay`: uniform constant `k = ln(2)/t_half`
- Future: `SpatiallyVaryingLoss{FT}` with `k(x,y,z)` field
- Future: `TimeVaryingLoss{FT}` with `k(x,y,z,t)` callback
"""
abstract type AbstractFirstOrderLoss <: AbstractChemistry end

# =====================================================================
# NoChemistry — inert tracers
# =====================================================================

"""
$(TYPEDEF)

No chemistry (inert tracers). Both forward and adjoint are no-ops.
"""
struct NoChemistry <: AbstractChemistry end

apply_chemistry!(tracers, grid, ::NoChemistry, Δt) = nothing
adjoint_chemistry!(adj_tracers, grid, ::NoChemistry, Δt) = nothing

# =====================================================================
# RadioactiveDecay — uniform constant first-order loss
# =====================================================================

"""
$(TYPEDEF)

First-order radioactive decay with a constant, spatially uniform rate.

The tracer is multiplied by `exp(-λ Δt)` each time step, where
`λ = ln(2) / t_half`. This is exact for constant `λ` and any `Δt`.

Works on both CPU `Array` and GPU `CuArray` via broadcasting.

# Common isotopes
- ²²²Rn: `half_life = 330_350.4` s (3.8235 days)
- ⁸⁵Kr:  `half_life = 3.394e8` s (10.76 years)
- ¹⁴C:   `half_life = 1.808e11` s (5730 years)

$(FIELDS)
"""
struct RadioactiveDecay{FT} <: AbstractFirstOrderLoss
    "target tracer name (e.g. :rn222)"
    species   :: Symbol
    "radioactive half-life [s]"
    half_life :: FT
    "decay constant λ = ln(2)/half_life [s⁻¹]"
    lambda    :: FT
end

"""
    RadioactiveDecay(; species, half_life, FT=Float64)

Construct a radioactive decay scheme. Precomputes `λ = ln(2)/half_life`.

# Example
```julia
rn_decay = RadioactiveDecay(; species=:rn222, half_life=330_350.4, FT=Float32)
```
"""
function RadioactiveDecay(; species::Symbol, half_life::Real, FT::Type{<:AbstractFloat}=Float64)
    lambda = FT(log(2) / half_life)
    RadioactiveDecay{FT}(species, FT(half_life), lambda)
end

"""
    apply_chemistry!(tracers, grid, chem::RadioactiveDecay, Δt)

Apply radioactive decay: `c .*= exp(-λ Δt)` for the target species.
Handles both regular arrays and NTuple{6} of panel arrays (cubed-sphere).
"""
function apply_chemistry!(tracers, grid, chem::RadioactiveDecay{FT}, Δt) where FT
    haskey(tracers, chem.species) || return nothing
    c = tracers[chem.species]
    decay_factor = exp(-chem.lambda * FT(Δt))
    _apply_decay!(c, decay_factor)
    return nothing
end

# Dispatch: regular array
_apply_decay!(c::AbstractArray, f) = (c .*= f)

# Dispatch: NTuple of arrays (cubed-sphere panels)
function _apply_decay!(panels::NTuple{N, <:AbstractArray}, f) where N
    for p in 1:N
        panels[p] .*= f
    end
end

"""
    adjoint_chemistry!(adj_tracers, grid, chem::RadioactiveDecay, Δt)

Adjoint of radioactive decay. Since `c_new = c_old * f` where `f = exp(-λΔt)`,
the adjoint is `adj_c_old += f * adj_c_new`, which for in-place update is
`adj_c .*= f` — identical to the forward operation (self-adjoint).
"""
function adjoint_chemistry!(adj_tracers, grid, chem::RadioactiveDecay{FT}, Δt) where FT
    haskey(adj_tracers, chem.species) || return nothing
    adj_c = adj_tracers[chem.species]
    decay_factor = exp(-chem.lambda * FT(Δt))
    _apply_decay!(adj_c, decay_factor)
    return nothing
end

# =====================================================================
# CompositeChemistry — multiple schemes for multi-tracer runs
# =====================================================================

"""
$(TYPEDEF)

Applies multiple chemistry schemes sequentially. Use this when different
tracers have different chemistry (e.g. ²²²Rn decays while CO₂ is inert).

$(FIELDS)

# Example
```julia
chem = CompositeChemistry([
    RadioactiveDecay(; species=:rn222, half_life=330_350.4, FT=Float32),
    # future: PhotolysisLoss(; species=:o3, ...)
])
```
"""
struct CompositeChemistry{S} <: AbstractChemistry
    "vector of chemistry schemes to apply in order"
    schemes :: S
end

CompositeChemistry(schemes::Vector{<:AbstractChemistry}) =
    CompositeChemistry{typeof(schemes)}(schemes)

CompositeChemistry(schemes::AbstractChemistry...) =
    CompositeChemistry(collect(AbstractChemistry, schemes))

function apply_chemistry!(tracers, grid, chem::CompositeChemistry, Δt)
    for scheme in chem.schemes
        apply_chemistry!(tracers, grid, scheme, Δt)
    end
    return nothing
end

function adjoint_chemistry!(adj_tracers, grid, chem::CompositeChemistry, Δt)
    # Adjoint is applied in reverse order
    for scheme in reverse(chem.schemes)
        adjoint_chemistry!(adj_tracers, grid, scheme, Δt)
    end
    return nothing
end

end # module Chemistry
