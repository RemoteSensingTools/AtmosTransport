"""
    Architectures

Standalone architecture layer for `src`, following the same Oceananigans-style
pattern as the production runtime without depending on `src/AtmosTransport.jl`.

Only the small, generic CPU/GPU contract is defined here. Multi-GPU panel
helpers stay in `src/` until cubed-sphere support is ported for real.
"""
module Architectures

using DocStringExtensions
using KernelAbstractions: KernelAbstractions as KA

export AbstractArchitecture, CPU, GPU
export array_type, device, architecture, _kahan_add

abstract type AbstractArchitecture end

"""
$(TYPEDEF)

Host CPU execution architecture.
"""
struct CPU <: AbstractArchitecture end

"""
$(TYPEDEF)

GPU execution architecture placeholder. Concrete array/device support is added
by adapter code or future extensions.
"""
struct GPU <: AbstractArchitecture end

array_type(::CPU) = Array
device(::CPU) = KA.CPU()

function array_type(::GPU)
    throw(ArgumentError("GPU array_type is not wired in src yet"))
end

function device(::GPU)
    throw(ArgumentError("GPU device is not wired in src yet"))
end

function architecture end

@inline function _kahan_add(s::T, c::T, x::T) where {T <: Union{Float16, Float32}}
    y = x - c
    t = s + y
    c_new = (t - s) - y
    return (t, c_new)
end

@inline _kahan_add(s::T, c::T, x::T) where {T <: Float64} = (s + x, zero(T))

end # module Architectures
