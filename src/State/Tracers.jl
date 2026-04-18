# ---------------------------------------------------------------------------
# Tracer utilities and storage-agnostic accessor API
#
# The accessor functions below (`ntracers`, `tracer_index`, `tracer_name`,
# `get_tracer`, `eachtracer`) are the PRIMARY interface for non-kernel code
# that needs to reach into `state.tracers`. They dispatch through `_*_impl`
# helpers on the concrete storage type, so the public API stays stable
# across a future storage change from `NamedTuple` to `Array{FT, 4}`.
#
# Kernel code (see `StrangSplitting.jl`, multi-tracer sweeps) bypasses the
# API and reads the raw storage directly — that is intentional and fast.
# ---------------------------------------------------------------------------

"""
    allocate_tracers(names::NTuple{N, Symbol}, Nx, Ny, Nz;
                     FT=Float64, ArrayType=Array, fill=zero(FT))

Allocate a NamedTuple of 3D tracer mass arrays for structured grids.
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
# Storage-agnostic accessor API
# =========================================================================

"""
    ntracers(state::CellState) -> Int

Number of tracers carried by `state`.
"""
ntracers(state::CellState) = _ntracers_impl(state.tracers)
_ntracers_impl(t::NamedTuple) = length(t)

"""
    tracer_index(state::CellState, name::Symbol) -> Union{Int, Nothing}

Index of the tracer named `name`, or `nothing` if it is not present.
"""
tracer_index(state::CellState, name::Symbol) = _tracer_index_impl(state.tracers, name)
_tracer_index_impl(t::NamedTuple, name::Symbol) = findfirst(==(name), keys(t))

"""
    tracer_name(state::CellState, idx::Integer) -> Symbol

Name of the tracer at index `idx`. Throws `BoundsError` if out of range.
"""
tracer_name(state::CellState, idx::Integer) = _tracer_name_impl(state.tracers, Int(idx))
_tracer_name_impl(t::NamedTuple, idx::Int) = keys(t)[idx]

"""
    get_tracer(state::CellState, name::Symbol)
    get_tracer(state::CellState, idx::Integer)

Return the tracer-mass array for the given name or index. Returns the
underlying storage slice (a view into `tracers_raw` after the storage
flip in plan 14's Commit 4); callers MUST treat mutations as reflected
in `state`.
"""
get_tracer(state::CellState, name::Symbol)    = _get_tracer_impl(state.tracers, name)
get_tracer(state::CellState, idx::Integer)    = _get_tracer_impl(state.tracers, Int(idx))
_get_tracer_impl(t::NamedTuple, name::Symbol) = getfield(t, name)
_get_tracer_impl(t::NamedTuple, idx::Int)     = t[idx]

"""
    eachtracer(state::CellState)

Iterate `(name::Symbol, tracer_mass) ` pairs for every tracer in `state`.
Yields the same `name => array` shape regardless of the underlying
storage, so destructuring `for (name, rm) in eachtracer(state)` works
both for the current NamedTuple storage and for the post-Commit-4 4D
array storage.
"""
eachtracer(state::CellState) = _eachtracer_impl(state.tracers)
_eachtracer_impl(t::NamedTuple) = pairs(t)

# =========================================================================
# Mutating utilities
# =========================================================================

"""
    set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)

Set tracer `name` to uniform mixing ratio χ: tracer_mass = χ × air_dry_mass.
"""
function set_uniform_mixing_ratio!(state, name::Symbol, χ)
    rm = get_tracer(state, name)
    rm .= χ .* state.air_mass
    return nothing
end

export allocate_tracers, set_uniform_mixing_ratio!
export ntracers, tracer_index, tracer_name, get_tracer, eachtracer
