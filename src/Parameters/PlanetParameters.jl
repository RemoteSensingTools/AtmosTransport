"""
    PlanetParameters{FT}

Physical constants associated with the modeled planet.
"""
struct PlanetParameters{FT <: AbstractFloat}
    radius             :: FT
    gravity            :: FT
    reference_pressure :: FT

    function PlanetParameters{FT}(radius::FT, gravity::FT,
                                  reference_pressure::FT) where FT <: AbstractFloat
        isfinite(radius) && radius > 0 ||
            throw(ArgumentError("planet radius must be finite and positive; got $(radius)"))
        isfinite(gravity) && gravity > 0 ||
            throw(ArgumentError("planet gravity must be finite and positive; got $(gravity)"))
        isfinite(reference_pressure) && reference_pressure > 0 ||
            throw(ArgumentError("planet reference_pressure must be finite and positive; got $(reference_pressure)"))
        return new{FT}(radius, gravity, reference_pressure)
    end
end

function PlanetParameters(radius::Real, gravity::Real, reference_pressure::Real)
    FT = promote_type(typeof(float(radius)), typeof(float(gravity)),
                      typeof(float(reference_pressure)))
    return PlanetParameters{FT}(FT(radius), FT(gravity), FT(reference_pressure))
end

function PlanetParameters(; FT::Type{<:AbstractFloat} = Float64,
                          radius = FT(6.371e6),
                          gravity = FT(9.80665),
                          reference_pressure = FT(101325.0))
    return PlanetParameters{FT}(FT(radius), FT(gravity), FT(reference_pressure))
end

earth_parameters(; FT::Type{<:AbstractFloat} = Float64) = PlanetParameters(; FT=FT)

export PlanetParameters, earth_parameters
