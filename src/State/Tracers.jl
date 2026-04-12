# ---------------------------------------------------------------------------
# Tracer utilities
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

"""
    set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)

Set tracer `name` to uniform mixing ratio χ: tracer_mass = χ × air_dry_mass.
"""
function set_uniform_mixing_ratio!(state, name::Symbol, χ)
    rm = getfield(state.tracers, name)
    rm .= χ .* state.air_mass
    return nothing
end

export allocate_tracers, set_uniform_mixing_ratio!
