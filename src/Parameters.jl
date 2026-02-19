"""
    Parameters

Type-stable parameter management from TOML configuration files.

Inspired by CliMA/ClimaParams.jl: parameters are read from TOML once, converted
to the chosen float type `FT`, and stored in concrete structs.  Downstream code
receives these structs (not dictionaries), so all access is type-stable and
zero-cost after construction.

# Usage

```julia
params = load_parameters(Float64)                          # defaults only
params = load_parameters(Float32; override="experiment.toml")  # with overrides
params.planet.radius       # ::Float32, type-stable
params.numerics.cfl_limit  # ::Float32
```
"""
module Parameters

using TOML

export PlanetParameters, NumericsParameters, SimulationParameters
export ModelParameters, load_parameters

# ---------------------------------------------------------------------------
# Parameter structs — all fields concretely typed via FT
# ---------------------------------------------------------------------------

"""
    PlanetParameters{FT}

Physical constants of the planet (Earth by default).
"""
struct PlanetParameters{FT}
    radius                     :: FT
    gravity                    :: FT
    reference_surface_pressure :: FT
end

"""
    NumericsParameters{FT}

Numerical tuning knobs.
"""
struct NumericsParameters{FT}
    polar_epsilon :: FT
    cfl_limit     :: FT
end

"""
    SimulationParameters{FT}

Run-time simulation settings.
"""
struct SimulationParameters{FT}
    dt      :: FT
    n_hours :: Int
end

"""
    ModelParameters{FT}

Top-level container holding all parameter groups.
Passed to grid constructors, advection kernels, etc.
"""
struct ModelParameters{FT, P <: PlanetParameters{FT},
                           N <: NumericsParameters{FT},
                           S <: SimulationParameters{FT}}
    planet     :: P
    numerics   :: N
    simulation :: S
end

float_type(::ModelParameters{FT}) where {FT} = FT

# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

const DEFAULT_TOML = joinpath(@__DIR__, "..", "config", "defaults.toml")

"""
    load_parameters(::Type{FT}; override=nothing, defaults=DEFAULT_TOML)

Read `defaults` TOML, merge with optional `override` TOML, and return a
fully type-stable `ModelParameters{FT}`.

`override` can be a file path (String) or an already-parsed Dict.
"""
function load_parameters(::Type{FT};
                         override = nothing,
                         defaults::String = DEFAULT_TOML) where {FT <: AbstractFloat}
    d = TOML.parsefile(defaults)

    if override !== nothing
        od = override isa AbstractString ? TOML.parsefile(override) : override
        _deep_merge!(d, od)
    end

    planet = PlanetParameters{FT}(
        FT(d["planet"]["radius"]),
        FT(d["planet"]["gravity"]),
        FT(d["planet"]["reference_surface_pressure"]),
    )

    numerics = NumericsParameters{FT}(
        FT(d["numerics"]["polar_epsilon"]),
        FT(d["numerics"]["cfl_limit"]),
    )

    sim = SimulationParameters{FT}(
        FT(d["simulation"]["dt"]),
        Int(d["simulation"]["n_hours"]),
    )

    return ModelParameters(planet, numerics, sim)
end

"""
    _deep_merge!(base, override)

Recursively merge `override` into `base`, overwriting leaf values.
"""
function _deep_merge!(base::Dict, override::Dict)
    for (k, v) in override
        if haskey(base, k) && base[k] isa Dict && v isa Dict
            _deep_merge!(base[k], v)
        else
            base[k] = v
        end
    end
    return base
end

end # module Parameters
