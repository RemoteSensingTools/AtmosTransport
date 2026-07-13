"""
    Architectures

Execution architectures and their optional runtime integration.

An architecture is part of a grid's type-level model contract: `CPU()` selects
host execution, while `GPU(:cuda)` and `GPU(:metal)` select a concrete GPU
runtime. The same object controls array adaptation, device synchronization, and
runtime validation, so grid metadata cannot diverge from model storage.
"""
module Architectures

using DocStringExtensions
using KernelAbstractions: KernelAbstractions as KA

export AbstractArchitecture, CPU, GPU
export array_type, device, architecture, architecture_from_config
export autodetect_gpu_architecture, is_gpu, ensure_runtime!, array_adapter
export architecture_label, device_name, backend_name, synchronize_architecture!
export array_adapter_for, assert_residency!, assert_float_type!
export reclaim_backend_pool!, _kahan_add

abstract type AbstractArchitecture end

"""
$(TYPEDEF)

Host CPU execution architecture.
"""
struct CPU <: AbstractArchitecture end

"""
$(TYPEDEF)

GPU execution architecture for backend `B`.

Construct with `GPU(:cuda)` or `GPU(:metal)`. Making the backend explicit keeps
CUDA and Metal methods unambiguous when both optional packages are loaded.
"""
struct GPU{B} <: AbstractArchitecture end

function GPU(backend::Symbol)
    backend in (:cuda, :metal) || throw(ArgumentError(
        "unsupported GPU backend $(repr(backend)); expected :cuda or :metal"))
    return GPU{backend}()
end

GPU(backend::AbstractString) = GPU(_architecture_symbol(backend))

array_type(::CPU) = Array
device(::CPU) = KA.CPU()

# GPU array and KernelAbstractions device methods live in the CUDA and Metal
# package extensions. Without the corresponding optional package, calling
# `array_type` or `device` for that architecture intentionally has no method.

function architecture end

is_gpu(::CPU) = false
is_gpu(::GPU) = true

backend_name(::CPU) = :cpu
backend_name(::GPU{B}) where {B} = B

const _RUNTIME_PACKAGE_IDS = (
    CUDA = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"),
    Metal = Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"),
)

function _load_runtime_package!(name::Symbol)
    hasproperty(_RUNTIME_PACKAGE_IDS, name) ||
        throw(ArgumentError("unsupported runtime package $(name)"))
    pkgid = getproperty(_RUNTIME_PACKAGE_IDS, name)
    try
        return Base.require(pkgid)
    catch err
        throw(ArgumentError(
            "$(name) backend requested, but $(name).jl could not be loaded from " *
            "the active environment: $(sprint(showerror, err))"))
    end
end

function _architecture_symbol(raw)
    name = replace(lowercase(String(raw)), '-' => '_', ' ' => '_')
    name in ("cpu", "host") && return :cpu
    name in ("cuda", "nvidia") && return :cuda
    name in ("metal", "apple", "apple_metal") && return :metal
    name in ("auto", "gpu") && return :auto
    throw(ArgumentError(
        "unknown architecture.backend = \"$(raw)\"; supported values are " *
        "\"cpu\", \"cuda\", \"metal\", and \"auto\"."))
end

_architecture(::Val{:cpu}) = CPU()
_architecture(::Val{:cuda}) = GPU(:cuda)
_architecture(::Val{:metal}) = GPU(:metal)

"""
    architecture_from_config(config) -> AbstractArchitecture

Resolve an `[architecture]` configuration table to one concrete execution
architecture. An omitted backend selects `CPU()` unless `use_gpu = true`, in
which case a usable GPU runtime is detected.
"""
function architecture_from_config(config)
    use_gpu = get(config, "use_gpu", false)
    use_gpu isa Bool || throw(ArgumentError(
        "[architecture].use_gpu must be true or false; got $(repr(use_gpu))"))
    raw_backend = get(config, "backend", nothing)

    raw_backend === nothing && return use_gpu ? autodetect_gpu_architecture() : CPU()

    backend = _architecture_symbol(raw_backend)
    backend === :cpu && use_gpu && throw(ArgumentError(
        "[architecture] use_gpu = true conflicts with backend = \"cpu\""))
    backend === :auto && return autodetect_gpu_architecture()
    return _architecture(Val(backend))
end

function _try_architecture!(arch::GPU)
    try
        ensure_runtime!(arch)
        return true, nothing
    catch err
        return false, err
    end
end

"""
    autodetect_gpu_architecture() -> GPU

Return the first functional supported GPU architecture on this host.
"""
function autodetect_gpu_architecture()
    candidates = Sys.isapple() ?
        (GPU(:metal), GPU(:cuda)) :
        (GPU(:cuda), GPU(:metal))

    failures = String[]
    for arch in candidates
        arch isa GPU{:metal} && !Sys.isapple() && continue
        ok, err = _try_architecture!(arch)
        ok && return arch
        push!(failures, "$(backend_name(arch)): $(sprint(showerror, err))")
    end

    detail = isempty(failures) ? "No candidate backend was attempted." :
             "Tried " * join(failures, "; ")
    throw(ArgumentError(
        "[architecture] requested GPU backend auto-detection, but no supported " *
        "GPU backend is usable on this host. $(detail)"))
end

ensure_runtime!(::CPU) = true

function ensure_runtime!(::GPU{:cuda})
    CUDA = _load_runtime_package!(:CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        throw(ArgumentError("CUDA runtime is not functional on this host"))
    isdefined(CUDA, :allowscalar) &&
        Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

function ensure_runtime!(::GPU{:metal})
    Sys.isapple() ||
        throw(ArgumentError("Metal backend requires macOS on Apple Silicon"))
    Metal = _load_runtime_package!(:Metal)
    if isdefined(Metal, :functional)
        Base.invokelatest(getproperty(Metal, :functional)) ||
            throw(ArgumentError("Metal runtime is not functional on this host"))
    end
    isdefined(Metal, :device) && Base.invokelatest(getproperty(Metal, :device))
    isdefined(Metal, :allowscalar) &&
        Base.invokelatest(getproperty(Metal, :allowscalar), false)
    return true
end

array_adapter(::CPU) = Array

function array_adapter(arch::GPU{:cuda})
    ensure_runtime!(arch)
    return getproperty(_load_runtime_package!(:CUDA), :CuArray)
end

function array_adapter(arch::GPU{:metal})
    ensure_runtime!(arch)
    return getproperty(_load_runtime_package!(:Metal), :MtlArray)
end

device_name(::CPU) = "CPU"

function device_name(arch::GPU{:cuda})
    ensure_runtime!(arch)
    CUDA = _load_runtime_package!(:CUDA)
    return string(Base.invokelatest(getproperty(CUDA, :name),
                                    Base.invokelatest(getproperty(CUDA, :device))))
end

function device_name(arch::GPU{:metal})
    ensure_runtime!(arch)
    Metal = _load_runtime_package!(:Metal)
    dev = isdefined(Metal, :device) ?
          Base.invokelatest(getproperty(Metal, :device)) :
          nothing
    dev === nothing && return "Metal device"
    return hasproperty(dev, :name) ? string(getproperty(dev, :name)) : string(dev)
end

architecture_label(::CPU) = "CPU"
architecture_label(arch::GPU{:cuda}) = "GPU (CUDA, $(device_name(arch)))"
architecture_label(arch::GPU{:metal}) = "GPU (Metal, $(device_name(arch)))"

synchronize_architecture!(::CPU) = nothing

function synchronize_architecture!(arch::GPU{:cuda})
    ensure_runtime!(arch)
    Base.invokelatest(getproperty(_load_runtime_package!(:CUDA), :synchronize))
    return nothing
end

function synchronize_architecture!(arch::GPU{:metal})
    ensure_runtime!(arch)
    Metal = _load_runtime_package!(:Metal)
    if isdefined(Metal, :synchronize)
        Base.invokelatest(getproperty(Metal, :synchronize))
    else
        KA.synchronize(getproperty(Metal, :MetalBackend)())
    end
    return nothing
end

function array_adapter_for(reference_array)
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    # Optional-package extensions may have loaded after this caller was
    # compiled, so cross the world-age boundary at this single startup lookup.
    return Base.invokelatest(_array_adapter_for, ref)
end

_array_adapter_for(::Any) = Array

"""
    reclaim_backend_pool!(reference_array)

Release device allocator caches associated with `reference_array` after
startup transients become unreachable. CPU arrays are a no-op.
"""
function reclaim_backend_pool!(reference_array)
    ref = reference_array isa Tuple ? reference_array[1] : reference_array
    return Base.invokelatest(_reclaim_backend_pool!, ref)
end

_reclaim_backend_pool!(::Any) = nothing

_is_architecture_array(::CPU, backing) = backing isa Array

function _is_architecture_array(arch::GPU{:cuda}, backing)
    return backing isa array_adapter(arch)
end

function _is_architecture_array(arch::GPU{:metal}, backing)
    return backing isa array_adapter(arch)
end

"""
    assert_residency!(storage, architecture; label="storage")

Verify that an array or tuple of arrays is resident on `architecture`. CPU
storage is returned directly; a GPU mismatch aborts rather than falling back
silently to host execution.
"""
function assert_residency!(storage, arch::AbstractArchitecture;
                           label::AbstractString = "storage")
    backing = storage isa Tuple ? parent(storage[1]) : parent(storage)
    is_gpu(arch) || return backing
    _is_architecture_array(arch, backing) || throw(ErrorException(
        "[gpu residency check] expected $(label) to live on $(backend_name(arch)) " *
        "but found $(typeof(backing)). CPU fallback aborted."))
    return backing
end

assert_float_type!(::AbstractArchitecture, ::Type{<:AbstractFloat}) = nothing

function assert_float_type!(::GPU{:metal}, ::Type{FT}) where {FT <: AbstractFloat}
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
