"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides `array_type` and `device` overloads so `GPU()` selects `CuArray`
and `CUDABackend()`.
"""
module AtmosTransportCUDAExt

import AtmosTransport
import CUDA
import AtmosTransport.Adjoints:
    PinnedHostCSTapeStorage,
    PinnedHostCSTapeSlot,
    _allocate_tape_slot,
    _ensure_tape_read_cache!,
    stage_panels!
using AtmosTransport.Architectures: GPU
using CUDA: CuArray, CUDABackend

AtmosTransport.Architectures.array_type(::GPU) = CuArray
AtmosTransport.Architectures.device(::GPU)     = CUDABackend()

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

end # module AtmosTransportCUDAExt
