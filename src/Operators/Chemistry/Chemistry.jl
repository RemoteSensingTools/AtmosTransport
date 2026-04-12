"""
    Chemistry

Source/sink operators for tracer transformations (decay, photolysis, etc.).

Type hierarchy:

    AbstractChemistry
    ├── NoChemistry               — inert tracers (no-op)
    ├── RadioactiveDecay{FT}      — constant first-order decay (e.g. ²²²Rn)
    └── CompositeChemistry        — combine schemes for multi-tracer runs

Interface: `apply_chemistry!(tracers, chem, Δt)`
"""
module Chemistry

export AbstractChemistry, NoChemistry, RadioactiveDecay, CompositeChemistry
export apply_chemistry!

abstract type AbstractChemistry end

struct NoChemistry <: AbstractChemistry end
apply_chemistry!(tracers, ::NoChemistry, Δt) = nothing

"""
    RadioactiveDecay{FT}(; species, half_life, FT=Float64)

First-order radioactive decay: `rm .*= exp(-λ·Δt)` where `λ = ln(2)/half_life`.

Exact for constant λ and any Δt. Works on CPU Array and GPU CuArray via broadcasting.

Common isotopes:
- ²²²Rn: half_life = 330_350.4 s (3.8235 days)
- ⁸⁵Kr:  half_life = 3.394e8 s (10.76 years)
"""
struct RadioactiveDecay{FT} <: AbstractChemistry
    species   :: Symbol
    half_life :: FT
    lambda    :: FT
end

function RadioactiveDecay(; species::Symbol, half_life::Real, FT::Type{<:AbstractFloat}=Float64)
    RadioactiveDecay{FT}(species, FT(half_life), FT(log(2) / half_life))
end

function apply_chemistry!(tracers, chem::RadioactiveDecay{FT}, Δt) where FT
    haskey(tracers, chem.species) || return nothing
    c = getfield(tracers, chem.species)
    c .*= exp(-chem.lambda * FT(Δt))
    return nothing
end

"""
    CompositeChemistry(schemes...)

Apply multiple chemistry schemes sequentially. For multi-tracer runs where
different species have different transformations.

    chem = CompositeChemistry(
        RadioactiveDecay(; species=:rn222, half_life=330_350.4)
    )
"""
struct CompositeChemistry{S} <: AbstractChemistry
    schemes :: S
end

CompositeChemistry(schemes::AbstractChemistry...) =
    CompositeChemistry(collect(AbstractChemistry, schemes))

function apply_chemistry!(tracers, chem::CompositeChemistry, Δt)
    for scheme in chem.schemes
        apply_chemistry!(tracers, scheme, Δt)
    end
    return nothing
end

end # module Chemistry
