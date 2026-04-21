# ---------------------------------------------------------------------------
# CubedSphereState — panel-native prognostic state for cubed-sphere runtime
# ---------------------------------------------------------------------------

"""
    CubedSphereState{Basis, A3, Raw4, Names}

Panel-native prognostic state for cubed-sphere transport.

# Fields
- `air_mass :: NTuple{6, A3}` — halo-padded panel air mass arrays with shape
  `(Nc + 2Hp, Nc + 2Hp, Nz)`.
- `tracers_raw :: NTuple{6, Raw4}` — packed per-panel tracer storage with shape
  `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`.
- `tracer_names :: Names` — names of the tracer axis in storage order.
- `halo_width :: Int` — halo width `Hp` needed to recover the physical interior.
"""
struct CubedSphereState{Basis <: AbstractMassBasis,
                        A3 <: AbstractArray,
                        Raw4 <: AbstractArray,
                        Names <: Tuple}
    air_mass     :: NTuple{6, A3}
    tracers_raw  :: NTuple{6, Raw4}
    tracer_names :: Names
    halo_width   :: Int
end

@inline halo_width(state::CubedSphereState) = state.halo_width

@inline function _pack_cs_tracers_raw(panels_m::NTuple{6, A3}, tr::NamedTuple) where {A3 <: AbstractArray}
    Nt = length(tr)
    return ntuple(6) do p
        panel_m = panels_m[p]
        raw = similar(panel_m, eltype(panel_m), size(panel_m)..., Nt)
        if Nt == 0
            return raw
        end
        @inbounds for (i, tracer_panels) in enumerate(values(tr))
            selectdim(raw, ndims(raw), i) .= tracer_panels[p]
        end
        raw
    end
end

function CubedSphereState(::Type{B}, mesh::CubedSphereMesh,
                          air_mass::NTuple{6}; tracers...) where {B <: AbstractMassBasis}
    return CubedSphereState(B, air_mass; halo_width = mesh.Hp, tracers...)
end

function CubedSphereState(::Type{B},
                          air_mass::NTuple{6};
                          halo_width::Integer,
                          tracers...) where {B <: AbstractMassBasis}
    tr = NamedTuple(tracers)
    raw = _pack_cs_tracers_raw(air_mass, tr)
    names = keys(tr)
    return CubedSphereState{B, typeof(air_mass[1]), typeof(raw[1]), typeof(names)}(
        air_mass, raw, names, Int(halo_width))
end

function CubedSphereState(::Type{B},
                          air_mass::NTuple{6},
                          tracers_raw::NTuple{6},
                          tracer_names::Tuple;
                          halo_width::Integer) where {B <: AbstractMassBasis}
    return CubedSphereState{B, typeof(air_mass[1]), typeof(tracers_raw[1]), typeof(tracer_names)}(
        air_mass, tracers_raw, tracer_names, Int(halo_width))
end

mass_basis(::CubedSphereState{B}) where {B <: AbstractMassBasis} = B()

function Adapt.adapt_structure(to, state::CubedSphereState{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, state.air_mass)
    tracers_raw = Adapt.adapt(to, state.tracers_raw)
    names = state.tracer_names
    return CubedSphereState{B, typeof(air_mass[1]), typeof(tracers_raw[1]), typeof(names)}(
        air_mass, tracers_raw, names, state.halo_width)
end

function Base.getproperty(state::CubedSphereState, name::Symbol)
    if name === :air_dry_mass
        return getfield(state, :air_mass)
    elseif name === :tracers
        return TracerAccessor(state)
    else
        return getfield(state, name)
    end
end

@inline function _panel_interior(panel::AbstractArray, Hp::Int)
    nx_panel = size(panel, 1)
    ny_panel = size(panel, 2)
    return @view panel[Hp + 1:nx_panel - Hp, Hp + 1:ny_panel - Hp, :]
end

@inline function _panel_interior(panel::AbstractArray, Hp::Int, tracer_idx::Int)
    nx_panel = size(panel, 1)
    ny_panel = size(panel, 2)
    return @view panel[Hp + 1:nx_panel - Hp, Hp + 1:ny_panel - Hp, :, tracer_idx]
end

mixing_ratio(state::CubedSphereState, name::Symbol) =
    ntuple(6) do p
        get_tracer(state, name)[p] ./ state.air_mass[p]
    end

function total_mass(state::CubedSphereState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    Hp = halo_width(state)
    total = zero(eltype(state.air_mass[1]))
    @inbounds for p in 1:6
        total += sum(_panel_interior(state.tracers_raw[p], Hp, idx))
    end
    return total
end

function total_air_mass(state::CubedSphereState)
    Hp = halo_width(state)
    total = zero(eltype(state.air_mass[1]))
    @inbounds for p in 1:6
        total += sum(_panel_interior(state.air_mass[p], Hp))
    end
    return total
end

tracer_names(state::CubedSphereState) = getfield(state, :tracer_names)

const DryCubedSphereState = CubedSphereState{DryBasis}
const MoistCubedSphereState = CubedSphereState{MoistBasis}

export CubedSphereState, DryCubedSphereState, MoistCubedSphereState
export halo_width
