"""
    AbstractCubedSphereField{FT}

Panel-native field contract for cubed-sphere operators.

Unlike [`AbstractTimeVaryingField`](@ref), which describes a single
rank-`N` spatial field, cubed-sphere runtime paths keep the panel
boundary explicit. A concrete cubed-sphere field therefore owns one
structured rank-3 field per panel and exposes them through
[`panel_field`](@ref).
"""
abstract type AbstractCubedSphereField{FT} end

"""
    CubedSphereField{FT, F}(panels::NTuple{6, F})

Panel-native cubed-sphere field wrapper.

Each panel field must satisfy the standard structured field contract:

- `panel_field(f, p) isa AbstractTimeVaryingField{FT, 3}`
- `field_value(panel_field(f, p), (i, j, k)) -> FT`
- `update_field!(panel_field(f, p), t)`

This keeps the cubed-sphere operator boundary honest without forcing a
fake global 4D/5D field abstraction onto code that already runs panel by
panel.
"""
struct CubedSphereField{FT, F <: AbstractTimeVaryingField{FT, 3}} <: AbstractCubedSphereField{FT}
    panels :: NTuple{6, F}
end

function CubedSphereField(panels::NTuple{6, F}) where {FT <: AbstractFloat,
                                                       F <: AbstractTimeVaryingField{FT, 3}}
    return CubedSphereField{FT, F}(panels)
end

function CubedSphereField(panels::NTuple{6, A}) where {FT <: AbstractFloat,
                                                       A <: AbstractArray{FT, 3}}
    wrapped = ntuple(p -> PreComputedKzField(panels[p]), 6)
    return CubedSphereField(wrapped)
end

function CubedSphereField(field::F) where {FT <: AbstractFloat, F <: AbstractTimeVaryingField{FT, 3}}
    return CubedSphereField(ntuple(_ -> field, 6))
end

@inline panel_field(f::CubedSphereField, p::Integer) = f.panels[Int(p)]

function update_field!(f::CubedSphereField, t::Real)
    @inbounds for p in 1:6
        update_field!(f.panels[p], t)
    end
    return f
end

function Adapt.adapt_structure(to, f::CubedSphereField)
    return CubedSphereField(Adapt.adapt(to, f.panels))
end

export AbstractCubedSphereField, CubedSphereField, panel_field
