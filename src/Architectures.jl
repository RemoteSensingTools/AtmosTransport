"""
    Architectures

CPU and GPU backend abstraction following the Oceananigans.jl pattern.

All architecture-dependent behavior (array allocation, kernel launch, device selection)
dispatches on `AbstractArchitecture` subtypes so that a single codebase runs on both
CPU and GPU without branching.

# Interface contract

Any new architecture subtype must implement:
- `array_type(arch)` → the array constructor (e.g. `Array`, `CuArray`)
- `device(arch)`     → the KernelAbstractions device

GPU-specific methods are defined in `ext/AtmosTransportModelCUDAExt.jl` (weak dependency).
"""
module Architectures

export AbstractArchitecture, CPU, GPU
export array_type, device, architecture

using KernelAbstractions: KernelAbstractions as KA

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

"""
    AbstractArchitecture

Supertype for all compute backends. Subtypes determine where arrays live
and how kernels are launched.
"""
abstract type AbstractArchitecture end

# ---------------------------------------------------------------------------
# Concrete architectures
# ---------------------------------------------------------------------------

"""
    CPU()

Run on the host CPU. Arrays are standard `Array`s.
"""
struct CPU <: AbstractArchitecture end

"""
    GPU()

Run on a GPU device. The concrete array type (e.g. `CuArray`) and device are
provided by the CUDA extension (`ext/AtmosTransportModelCUDAExt.jl`).
"""
struct GPU <: AbstractArchitecture end

# ---------------------------------------------------------------------------
# Interface: array_type
# ---------------------------------------------------------------------------

"""
    array_type(arch::AbstractArchitecture)

Return the array constructor for the given architecture.
CPU returns `Array`; GPU is defined by the CUDA extension.
"""
array_type(::CPU) = Array

# ---------------------------------------------------------------------------
# Interface: device
# ---------------------------------------------------------------------------

"""
    device(arch::AbstractArchitecture)

Return the KernelAbstractions device for kernel launches.
CPU returns `KA.CPU()`; GPU is defined by the CUDA extension.
"""
device(::CPU) = KA.CPU()

# ---------------------------------------------------------------------------
# Utility: extract architecture from objects
# ---------------------------------------------------------------------------

"""
    architecture(x)

Return the `AbstractArchitecture` associated with `x`.
Concrete types (grids, fields, models) should specialize this.
"""
function architecture end

end # module Architectures
