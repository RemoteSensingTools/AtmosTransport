# ---------------------------------------------------------------------------
# Tracer utilities and storage-agnostic accessor API
#
# The accessor functions below are the PRIMARY interface for non-kernel
# code that needs to touch `state.tracers`. They read directly from
# `state.tracers_raw` + `state.tracer_names` (see CellState.jl).
#
# Kernels (multi-tracer sweeps, vertical remap) bypass this API and
# dispatch on raw storage directly — that is intentional and fast.
# ---------------------------------------------------------------------------

"""
    allocate_tracers(names::NTuple{N, Symbol}, Nx, Ny, Nz;
                     FT=Float64, ArrayType=Array, fill_value=zero(FT))

Allocate a NamedTuple of 3D tracer mass arrays. Still useful for test
fixtures that want to pre-build per-tracer 3D arrays for a
`CellState(m; tracers...)` keyword-form constructor call.
"""
function allocate_tracers(names::NTuple{N, Symbol}, Nx::Int, Ny::Int, Nz::Int;
                          FT::Type{<:AbstractFloat} = Float64,
                          ArrayType = Array,
                          fill_value = zero(FT)) where N
    arrays = ntuple(N) do _
        arr = ArrayType(zeros(FT, Nx, Ny, Nz))
        fill_value != zero(FT) && fill!(arr, fill_value)
        arr
    end
    return NamedTuple{names}(arrays)
end

# =========================================================================
# Accessor API
# =========================================================================

"""
    ntracers(state::CellState) -> Int

Number of tracers carried by `state`.
"""
ntracers(state::CellState) = length(state.tracer_names)
ntracers(state::CubedSphereState) = length(state.tracer_names)

"""
    tracer_index(state::CellState, name::Symbol) -> Union{Int, Nothing}

Index of the tracer named `name` in `state.tracer_names`, or `nothing`
if absent.
"""
function tracer_index(state::CellState, name::Symbol)
    return findfirst(==(name), state.tracer_names)
end

function tracer_index(state::CubedSphereState, name::Symbol)
    return findfirst(==(name), state.tracer_names)
end

"""
    tracer_name(state::CellState, idx::Integer) -> Symbol

Name of the tracer at index `idx`. Throws `BoundsError` if out of
range.
"""
tracer_name(state::CellState, idx::Integer) = state.tracer_names[Int(idx)]
tracer_name(state::CubedSphereState, idx::Integer) = state.tracer_names[Int(idx)]

"""
    get_tracer(state::CellState, name::Symbol)
    get_tracer(state::CellState, idx::Integer)

Return a view of the tracer mass slice. For a structured grid with
`state.tracers_raw :: Array{FT, 4}`, this is
`selectdim(state.tracers_raw, 4, idx)`, a contiguous
`SubArray{FT, 3}` (because Julia is column-major and the tracer axis
is the slowest-varying). Mutations through the returned view are
reflected in `state.tracers_raw`.

Throws `KeyError(name)` if `name` is not a tracer in `state`.
"""
function get_tracer(state::CellState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    return get_tracer(state, idx)
end

function get_tracer(state::CubedSphereState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    return get_tracer(state, idx)
end

function get_tracer(state::CellState, idx::Integer)
    raw = state.tracers_raw
    return selectdim(raw, ndims(raw), Int(idx))
end

function get_tracer(state::CubedSphereState, idx::Integer)
    tracer_idx = Int(idx)
    return ntuple(6) do p
        raw = state.tracers_raw[p]
        selectdim(raw, ndims(raw), tracer_idx)
    end
end

"""
    eachtracer(state::CellState)

Iterate `name => tracer_slice` pairs for every tracer in `state`, in
storage order. The yielded shape matches the previous
`pairs(::NamedTuple)` contract so callers that destructure
`for (name, rm) in eachtracer(state)` continue to work.
"""
function eachtracer(state::CellState)
    return (n => get_tracer(state, i) for (i, n) in enumerate(state.tracer_names))
end

function eachtracer(state::CubedSphereState)
    return (n => get_tracer(state, i) for (i, n) in enumerate(state.tracer_names))
end

# =========================================================================
# Raw vs full tracer access (reference-state / anomaly transport, plan 45)
#
# For a tracer with an active reference (kind != REF_NONE) the stored array
# holds ANOMALY mass `q_anom·m`; the physical field is `q_anom·m + q_ref·m`.
# The split is enforced, not conventional:
#
# - Operators/kernels mutate the STORED array in place → `get_tracer_raw`
#   (exactly `get_tracer`; the bare name keeps raw semantics).
# - Output, diagnostics, and budget readers want the PHYSICAL field →
#   `get_tracer_full` / `total_mass_full` / `mixing_ratio_full`.
#
# A test gate (test/core/test_tracer_references.jl) fails on bare accessor
# names in output/diagnostic namespaces so a physical-field caller cannot
# silently read anomaly mass. CellState carries no references (referencing is
# cubed-sphere-only); its `_full` variants reduce to the raw ones.
# =========================================================================

@inline tracer_reference_kind(::CellState, ::Integer) = REF_NONE
@inline tracer_reference_value(::CellState, ::Integer) = nothing

"""
    get_tracer_raw(state, name_or_idx)

The mutable STORED tracer-mass array — anomaly mass for referenced tracers,
full mass otherwise. Identical to `get_tracer`; the explicit name marks
operator/kernel call sites that must keep mutating storage in place.
"""
@inline get_tracer_raw(state::CellState, x::Union{Symbol, Integer}) = get_tracer(state, x)
@inline get_tracer_raw(state::CubedSphereState, x::Union{Symbol, Integer}) = get_tracer(state, x)

"""
    get_tracer_full(state, name_or_idx)

The PHYSICAL tracer-mass field: the stored array for unreferenced tracers
(zero-copy view), or a materialized `q_anom·m + q_ref·m` for referenced ones.
Read-only by contract — mutations to a materialized copy are lost.
"""
function get_tracer_full(state::CellState, idx::Integer)
    raw = get_tracer(state, idx)
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return raw
    FT = eltype(state.air_mass)
    return raw .+ FT(q_ref) .* state.air_mass
end

function get_tracer_full(state::CubedSphereState, idx::Integer)
    raw = get_tracer(state, idx)
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return raw
    FT = eltype(state.air_mass[1])
    qr = FT(q_ref)
    return ntuple(p -> raw[p] .+ qr .* state.air_mass[p], 6)
end

function get_tracer_full(state::Union{CellState, CubedSphereState}, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    return get_tracer_full(state, idx)
end

"""
    total_mass_full(state, name) -> Float64

Physical tracer burden `Σ_interior(q_anom·m) + q_ref·Σ_interior(m)`.
Accumulates in `Float64` regardless of storage `FT` — this is the budget
diagnostic the reference bookkeeping must close against, so it must not
round in `FT`. (`total_mass`/raw accumulates in storage eltype.)
"""
function total_mass_full(state::CellState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    raw_sum = sum(Float64, get_tracer(state, idx))
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return raw_sum
    return raw_sum + q_ref * sum(Float64, state.air_mass)
end

function total_mass_full(state::CubedSphereState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    Hp = halo_width(state)
    raw_sum = 0.0
    @inbounds for p in 1:6
        raw_sum += sum(Float64, _panel_interior(state.tracers_raw[p], Hp, idx))
    end
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return raw_sum
    air_sum = 0.0
    @inbounds for p in 1:6
        air_sum += sum(Float64, _panel_interior(state.air_mass[p], Hp))
    end
    return raw_sum + q_ref * air_sum
end

"""
    mass_weighted_global_mean_vmr(rm_panels, m_panels, Hp) -> Float64

Mass-weighted global-mean dry VMR `Σ_interior(rm) / Σ_interior(m)`, both sums
in `Float64` over EXACTLY the interior cells `total_mass_full` uses — that
shared cell set is what makes the reference seed/burden bookkeeping close
exactly. This is the `q_ref` definition for `reference = "global_mean"`
(plan 45): NOT an area-weighted or column-averaged mean.
"""
function mass_weighted_global_mean_vmr(rm_panels::NTuple{6}, m_panels::NTuple{6},
                                       Hp::Integer)
    rm_sum = 0.0
    m_sum  = 0.0
    @inbounds for p in 1:6
        rm_sum += sum(Float64, _panel_interior(rm_panels[p], Int(Hp)))
        m_sum  += sum(Float64, _panel_interior(m_panels[p], Int(Hp)))
    end
    m_sum > 0.0 || throw(ArgumentError(
        "mass_weighted_global_mean_vmr: total interior air mass is not positive"))
    return rm_sum / m_sum
end

"""
    mixing_ratio_full(state, name)

Physical dry VMR `q_anom + q_ref` (= `get_tracer_full ./ air_mass`, computed
as `raw ./ m .+ q_ref` to avoid the `q_ref·m` round-trip). Reduces to
`mixing_ratio` for unreferenced tracers.
"""
function mixing_ratio_full(state::CellState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return mixing_ratio(state, name)
    FT = eltype(state.air_mass)
    return get_tracer(state, idx) ./ state.air_mass .+ FT(q_ref)
end

function mixing_ratio_full(state::CubedSphereState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    q_ref = tracer_reference_value(state, idx)
    q_ref === nothing && return mixing_ratio(state, name)
    raw = get_tracer(state, idx)
    FT = eltype(state.air_mass[1])
    qr = FT(q_ref)
    return ntuple(p -> raw[p] ./ state.air_mass[p] .+ qr, 6)
end

# =========================================================================
# Mutating utilities
# =========================================================================

"""
    set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)

Set tracer `name` to uniform mixing ratio χ: tracer_mass = χ × air_dry_mass.
"""
function set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)
    rm = get_tracer(state, name)
    rm .= χ .* state.air_mass
    return nothing
end

function set_uniform_mixing_ratio!(state::CubedSphereState, name::Symbol, χ)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    # Full-field semantic write: `rm = χ·m` assumes the stored array holds full
    # mass. For a referenced tracer that would silently corrupt the anomaly
    # store (the correct write would be `(χ - q_ref)·m`) — reject until a
    # reference-aware variant is actually needed.
    tracer_reference_kind(state, idx) == REF_NONE || throw(ArgumentError(
        "set_uniform_mixing_ratio! writes full-field mass and is not " *
        "reference-aware; tracer $(name) carries an active reference"))
    rm_panels = get_tracer(state, idx)
    @inbounds for p in 1:6
        rm_panels[p] .= χ .* state.air_mass[p]
    end
    return nothing
end

export allocate_tracers, set_uniform_mixing_ratio!
export ntracers, tracer_index, tracer_name, get_tracer, eachtracer
export get_tracer_raw, get_tracer_full, total_mass_full, mixing_ratio_full
export mass_weighted_global_mean_vmr
