"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides `array_type` and `device` overloads so `GPU(:cuda)` selects
`CuArray` and `CUDABackend()`.
"""
module AtmosTransportCUDAExt

import AtmosTransport
import CUDA
import AtmosTransport.Adjoints:
    PinnedHostCSTapeStorage,
    PinnedHostCSTapeSlot,
    MmapCSTapeStorage,
    _allocate_tape_slot,
    _ensure_tape_read_cache!,
    _mmap_prepare_for_panels!,
    stage_panels!
using AtmosTransport.Architectures: GPU
using CUDA: CuArray, CUDABackend

AtmosTransport.Architectures.array_type(::GPU{:cuda}) = CuArray
AtmosTransport.Architectures.device(::GPU{:cuda})     = CUDABackend()
AtmosTransport.Architectures.architecture(::CUDA.AbstractGPUArray) = GPU(:cuda)
AtmosTransport.Architectures._array_adapter_for(::CUDA.AbstractGPUArray) = CuArray

function AtmosTransport.Architectures._reclaim_backend_pool!(::CUDA.AbstractGPUArray)
    CUDA.synchronize()
    CUDA.reclaim()
    return nothing
end

_pinned_host_panel(a::CuArray{T,N}) where {T,N} =
    CUDA.pin(Array{T,N}(undef, size(a)))

function _allocate_tape_slot(storage::PinnedHostCSTapeStorage,
                             panels::NTuple{6,<:CuArray})
    storage.synchronize = CUDA.synchronize
    _ensure_tape_read_cache!(storage, panels)
    host_panels = ntuple(p -> _pinned_host_panel(panels[p]), 6)
    return PinnedHostCSTapeSlot(storage, host_panels)
end

function stage_panels!(slot::PinnedHostCSTapeSlot,
                       src::NTuple{6,<:CuArray})
    @inbounds for p in 1:6
        copyto!(slot.host_panels[p], src[p])
    end
    CUDA.synchronize()
    return slot
end

# Plan 26 Phase A.1 — Mmap tape policy on GPU. `_mmap_prepare_for_panels!`
# is called from `_allocate_tape_slot(::MmapCSTapeStorage, panels)` BEFORE
# disk reservation; it installs a shared CuArray read cache and the
# CUDA.synchronize hook so subsequent `copyto!(::Array, ::CuArray)` and
# `copyto!(::CuArray, ::Array)` operations against the mmap views are
# wrapped by a device sync. No specialised `_allocate_tape_slot` or
# `stage_panels!` is needed: both default to mmap-write / mmap-read,
# and CUDA.jl provides synchronous `copyto!` between `Array` and
# `CuArray` of matching element type and shape.
function _mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                                   panels::NTuple{6,<:CuArray})
    storage.synchronize = CUDA.synchronize
    _ensure_tape_read_cache!(storage, panels)
    return storage
end

end # module AtmosTransportCUDAExt
