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
export AbstractRuntimeBackend, AbstractGPURuntimeBackend
export CPUBackend, CUDAGPUBackend, MetalGPUBackend
export runtime_backend_from_config, autodetect_gpu_backend, is_gpu_backend
export ensure_backend_runtime!, backend_array_adapter, backend_label
export backend_device_name, backend_name, synchronize_backend!
export array_adapter_for, assert_backend_residency!, assert_backend_float_type!

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

# GPU methods are supplied by the AtmosTransportCUDAExt / AtmosTransportMetalExt
# extensions. Calling `array_type(GPU())` or `device(GPU())` without either
# extension loaded throws a MethodError, which is the intended behavior.

function architecture end

# ---------------------------------------------------------------------------
# Runtime backend selection
#
# The architecture marker above is intentionally small. The driven runtime also
# needs to load optional GPU packages, pick an array adapter, verify residency,
# and synchronize before/after timing. Keep those operations in one place so
# adding another KernelAbstractions backend is a matter of adding a backend type
# plus a few methods here instead of threading `if CUDA ...` checks through the
# model runner.
# ---------------------------------------------------------------------------

abstract type AbstractRuntimeBackend end
abstract type AbstractGPURuntimeBackend <: AbstractRuntimeBackend end

struct CPUBackend <: AbstractRuntimeBackend end
struct CUDAGPUBackend <: AbstractGPURuntimeBackend end
struct MetalGPUBackend <: AbstractGPURuntimeBackend end

is_gpu_backend(::AbstractRuntimeBackend) = false
is_gpu_backend(::AbstractGPURuntimeBackend) = true

backend_name(::CPUBackend) = :cpu
backend_name(::CUDAGPUBackend) = :cuda
backend_name(::MetalGPUBackend) = :metal

_runtime_module(name::Symbol) = getfield(Main, name)

function _load_runtime_package!(name::Symbol)
    if !isdefined(Main, name)
        if name === :CUDA
            Core.eval(Main, :(using CUDA))
        elseif name === :Metal
            Core.eval(Main, :(using Metal))
        else
            throw(ArgumentError("unsupported runtime package $(name)"))
        end
    end
    return _runtime_module(name)
end

_backend_from_symbol(::Val{:cpu}) = CPUBackend()
_backend_from_symbol(::Val{:cuda}) = CUDAGPUBackend()
_backend_from_symbol(::Val{:metal}) = MetalGPUBackend()

function _backend_symbol(raw)
    s = lowercase(String(raw))
    s = replace(s, '-' => '_', ' ' => '_')
    if s in ("cpu", "host")
        return :cpu
    elseif s in ("cuda", "nvidia")
        return :cuda
    elseif s in ("metal", "apple", "apple_metal")
        return :metal
    elseif s in ("auto", "gpu")
        return :auto
    end
    throw(ArgumentError(
        "unknown architecture.backend = \"$(raw)\"; supported values are " *
        "\"cpu\", \"cuda\", \"metal\", and \"auto\"."))
end

function runtime_backend_from_config(arch_cfg)
    use_gpu = Bool(get(arch_cfg, "use_gpu", false))
    raw_backend = get(arch_cfg, "backend", nothing)

    if raw_backend === nothing
        return use_gpu ? autodetect_gpu_backend() : CPUBackend()
    end

    backend = _backend_symbol(raw_backend)
    backend === :cpu && use_gpu &&
        throw(ArgumentError("[architecture] use_gpu = true conflicts with backend = \"cpu\""))
    backend === :auto && return autodetect_gpu_backend()
    return _backend_from_symbol(Val(backend))
end

function _try_backend!(backend::AbstractGPURuntimeBackend)
    try
        ensure_backend_runtime!(backend)
        return true, nothing
    catch err
        return false, err
    end
end

function autodetect_gpu_backend()
    candidates = Sys.isapple() ?
        (MetalGPUBackend(), CUDAGPUBackend()) :
        (CUDAGPUBackend(), MetalGPUBackend())

    failures = String[]
    for backend in candidates
        if backend isa MetalGPUBackend && !Sys.isapple() && !isdefined(Main, :Metal)
            continue
        end
        ok, err = _try_backend!(backend)
        ok && return backend
        push!(failures, "$(backend_name(backend)): $(sprint(showerror, err))")
    end

    detail = isempty(failures) ? "No candidate backend was attempted." :
             "Tried " * join(failures, "; ")
    throw(ArgumentError(
        "[architecture] requested GPU backend auto-detection, but no supported " *
        "GPU backend is usable on this host. $(detail)"))
end

ensure_backend_runtime!(::CPUBackend) = true

function ensure_backend_runtime!(::CUDAGPUBackend)
    CUDA = _load_runtime_package!(:CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        throw(ArgumentError("CUDA runtime is not functional on this host"))
    isdefined(CUDA, :allowscalar) &&
        Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

function ensure_backend_runtime!(::MetalGPUBackend)
    Sys.isapple() ||
        throw(ArgumentError("Metal backend requires macOS on Apple Silicon"))
    Metal = _load_runtime_package!(:Metal)
    if isdefined(Metal, :functional)
        Base.invokelatest(getproperty(Metal, :functional)) ||
            throw(ArgumentError("Metal runtime is not functional on this host"))
    end
    # `device()` is the lightweight availability probe Metal.jl exposes.
    isdefined(Metal, :device) && Base.invokelatest(getproperty(Metal, :device))
    isdefined(Metal, :allowscalar) &&
        Base.invokelatest(getproperty(Metal, :allowscalar), false)
    return true
end

backend_array_adapter(::CPUBackend) = Array

function backend_array_adapter(backend::CUDAGPUBackend)
    ensure_backend_runtime!(backend)
    return getproperty(_runtime_module(:CUDA), :CuArray)
end

function backend_array_adapter(backend::MetalGPUBackend)
    ensure_backend_runtime!(backend)
    return getproperty(_runtime_module(:Metal), :MtlArray)
end

backend_device_name(::CPUBackend) = "CPU"

function backend_device_name(backend::CUDAGPUBackend)
    ensure_backend_runtime!(backend)
    CUDA = _runtime_module(:CUDA)
    return string(Base.invokelatest(getproperty(CUDA, :name),
                                    Base.invokelatest(getproperty(CUDA, :device))))
end

function backend_device_name(backend::MetalGPUBackend)
    ensure_backend_runtime!(backend)
    Metal = _runtime_module(:Metal)
    dev = isdefined(Metal, :device) ?
          Base.invokelatest(getproperty(Metal, :device)) :
          nothing
    dev === nothing && return "Metal device"
    return hasproperty(dev, :name) ? string(getproperty(dev, :name)) : string(dev)
end

backend_label(::CPUBackend) = "CPU"
backend_label(backend::CUDAGPUBackend) = "GPU (CUDA, $(backend_device_name(backend)))"
backend_label(backend::MetalGPUBackend) = "GPU (Metal, $(backend_device_name(backend)))"

synchronize_backend!(::CPUBackend) = nothing

function synchronize_backend!(backend::CUDAGPUBackend)
    ensure_backend_runtime!(backend)
    Base.invokelatest(getproperty(_runtime_module(:CUDA), :synchronize))
    return nothing
end

function synchronize_backend!(backend::MetalGPUBackend)
    ensure_backend_runtime!(backend)
    Metal = _runtime_module(:Metal)
    if isdefined(Metal, :synchronize)
        Base.invokelatest(getproperty(Metal, :synchronize))
    else
        KA.synchronize(getproperty(Metal, :MetalBackend)())
    end
    return nothing
end

function array_adapter_for(reference_array)
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    if isdefined(Main, :CUDA)
        CUDA = _runtime_module(:CUDA)
        if isdefined(CUDA, :AbstractGPUArray)
            AbstractGPUArray = getproperty(CUDA, :AbstractGPUArray)
            ref isa AbstractGPUArray && return getproperty(CUDA, :CuArray)
        end
    end
    if isdefined(Main, :Metal)
        Metal = _runtime_module(:Metal)
        MtlArray = getproperty(Metal, :MtlArray)
        ref isa MtlArray && return MtlArray
    end
    return Array
end

_is_backend_array(::CPUBackend, backing) = backing isa Array

function _is_backend_array(backend::CUDAGPUBackend, backing)
    CuArray = backend_array_adapter(backend)
    return backing isa CuArray
end

function _is_backend_array(backend::MetalGPUBackend, backing)
    MtlArray = backend_array_adapter(backend)
    return backing isa MtlArray
end

function assert_backend_residency!(storage, backend::AbstractRuntimeBackend;
                                   label::AbstractString = "storage")
    is_gpu_backend(backend) ||
        return storage isa Tuple ? parent(storage[1]) : parent(storage)
    backing = storage isa Tuple ? parent(storage[1]) : parent(storage)
    _is_backend_array(backend, backing) || throw(ErrorException(
        "[gpu residency check] expected $(label) to live on $(backend_name(backend)) " *
        "but found $(typeof(backing)). CPU fallback aborted."))
    return backing
end

assert_backend_float_type!(::AbstractRuntimeBackend, ::Type{<:AbstractFloat}) = nothing

function assert_backend_float_type!(::MetalGPUBackend, ::Type{FT}) where {FT <: AbstractFloat}
    FT === Float32 || throw(ArgumentError(
        "Metal backend requires [numerics] float_type = \"Float32\"; got $(FT). " *
        "Apple Metal does not support Float64 kernels for this runtime."))
    return nothing
end

@inline function _kahan_add(s::T, c::T, x::T) where {T <: Union{Float16, Float32}}
    y = x - c
    t = s + y
    c_new = (t - s) - y
    return (t, c_new)
end

@inline _kahan_add(s::T, c::T, x::T) where {T <: Float64} = (s + x, zero(T))

end # module Architectures
