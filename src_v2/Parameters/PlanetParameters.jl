"""
    PlanetParameters{FT}

Physical constants associated with the modeled planet.
"""
struct PlanetParameters{FT <: AbstractFloat}
    radius             :: FT
    gravity            :: FT
    reference_pressure :: FT
end

function PlanetParameters(; FT::Type{<:AbstractFloat} = Float64,
                          radius = FT(6.371e6),
                          gravity = FT(9.80665),
                          reference_pressure = FT(101325.0))
    return PlanetParameters{FT}(FT(radius), FT(gravity), FT(reference_pressure))
end

earth_parameters(; FT::Type{<:AbstractFloat} = Float64) = PlanetParameters(; FT=FT)

export PlanetParameters, earth_parameters
