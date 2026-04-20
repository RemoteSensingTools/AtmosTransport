"""
    PerTracerFluxMap{S <: Tuple}

An ordered tuple of `SurfaceFluxSource`s, keyed by the `tracer_name`
field on each entry. Supplies the surface flux operator (Commit 3)
with per-tracer 2D flux data and ensures efficient tuple-splatting
at kernel-launch time.

The map is **NTuple-backed** rather than Dict-backed. Rationale:

- Existing `DrivenSimulation.surface_sources::Tuple` storage is already
  a tuple; the map type is a thin re-export with a lookup helper.
- Tuples are bits-stable on GPU (captured by kernels without
  boxing / hashing); Dicts would require special Adapt handling and
  per-launch lookup cost.
- Small-N (typically 1-5 emitting tracers) makes linear scan cheaper
  than hashing anyway.

Tracers absent from the map have **zero surface flux**. The consumer
operator iterates the map and applies each source; it does NOT walk
the state's full `tracer_names` and look up each by name.

# Construction

```julia
co2_rate  = fill(2.0e-7, Nx, Ny)    # kg/s per cell for :CO2
sf6_rate  = fill(1.5e-9, Nx, Ny)    # kg/s per cell for :SF6
rn222_rate = fill(3.0e-11, Nx, Ny)  # kg/s per cell for :Rn222

map = PerTracerFluxMap(
    SurfaceFluxSource(:CO2,   co2_rate),
    SurfaceFluxSource(:SF6,   sf6_rate),
    SurfaceFluxSource(:Rn222, rn222_rate),
)

length(map)                 # 3
flux_for(map, :CO2) === ... # the :CO2 SurfaceFluxSource
flux_for(map, :CH4)         # nothing (not emitting)
```

# Adapt / GPU

`Adapt.adapt(CuArray, map)` converts each `cell_mass_rate` to a
`CuArray` transparently via `SurfaceFluxSource`'s `adapt_structure`;
tuples adapt element-wise.

# Fields
- `sources :: S` — `NTuple{N, SurfaceFluxSource{<:AbstractArray}}` for
  some `N`. Each entry carries a tracer name and the rate array.
"""
struct PerTracerFluxMap{S <: Tuple}
    sources :: S

    function PerTracerFluxMap(sources::S) where {S <: Tuple}
        # Validate that every entry is a SurfaceFluxSource.
        for (k, src) in enumerate(sources)
            src isa SurfaceFluxSource ||
                throw(ArgumentError("PerTracerFluxMap entry $k is a $(typeof(src)), expected SurfaceFluxSource"))
        end
        # Validate tracer names are unique — duplicate entries would
        # double-apply the emission for the same tracer.
        names = map(s -> s.tracer_name, sources)
        length(unique(names)) == length(names) ||
            throw(ArgumentError("PerTracerFluxMap: duplicate tracer names $(names)"))
        new{S}(sources)
    end
end

"""
    PerTracerFluxMap(sources::SurfaceFluxSource...)

Variadic constructor: `PerTracerFluxMap(src1, src2, src3)` wraps the
three sources into an NTuple-backed map.
"""
PerTracerFluxMap(sources::SurfaceFluxSource...) = PerTracerFluxMap(sources)

"""
    PerTracerFluxMap(sources::AbstractVector{<:SurfaceFluxSource})
    PerTracerFluxMap(sources::Tuple)

Generic collection constructor. The input is frozen into an NTuple.
"""
PerTracerFluxMap(sources::AbstractVector{<:SurfaceFluxSource}) =
    PerTracerFluxMap(Tuple(sources))

"""
    flux_for(map, tracer_name::Symbol) -> SurfaceFluxSource | nothing

Return the `SurfaceFluxSource` for the named tracer, or `nothing` if the
tracer has no surface source in this map. O(N) linear scan, which is
fine for typical N ≤ 10.
"""
@inline function flux_for(map::PerTracerFluxMap, tracer_name::Symbol)
    for src in map.sources
        src.tracer_name === tracer_name && return src
    end
    return nothing
end

# =========================================================================
# Iteration + collection interface
# =========================================================================

Base.length(m::PerTracerFluxMap) = length(m.sources)
Base.eltype(::Type{PerTracerFluxMap{S}}) where {S} = eltype(S)
Base.iterate(m::PerTracerFluxMap, args...) = iterate(m.sources, args...)
Base.firstindex(m::PerTracerFluxMap) = firstindex(m.sources)
Base.lastindex(m::PerTracerFluxMap) = lastindex(m.sources)
Base.getindex(m::PerTracerFluxMap, i) = m.sources[i]

"""
    tracer_names(map::PerTracerFluxMap) -> NTuple{N, Symbol}

Ordered tuple of tracer names in the map, matching the storage order of
`map.sources`.
"""
tracer_names(m::PerTracerFluxMap) = map(s -> s.tracer_name, m.sources)

# =========================================================================
# Adapt — GPU dispatch
# =========================================================================

# Tuples adapt element-wise; each SurfaceFluxSource's adapt_structure
# carries its cell_mass_rate to the device backing. The map stays a
# thin wrapper.
Adapt.adapt_structure(to, m::PerTracerFluxMap) =
    PerTracerFluxMap(map(s -> Adapt.adapt(to, s), m.sources))
