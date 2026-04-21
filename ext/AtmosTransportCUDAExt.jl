"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides `array_type` and `device` overloads so `GPU()` selects `CuArray`
and `CUDABackend()`.
"""
module AtmosTransportCUDAExt

import AtmosTransport
using AtmosTransport.Architectures: GPU
using CUDA: CuArray, CUDABackend

AtmosTransport.Architectures.array_type(::GPU) = CuArray
AtmosTransport.Architectures.device(::GPU)     = CUDABackend()

end # module AtmosTransportCUDAExt
