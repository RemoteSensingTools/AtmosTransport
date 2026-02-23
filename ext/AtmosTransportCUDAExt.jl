"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides GPU array types and KernelAbstractions device for the GPU architecture.
"""
module AtmosTransportCUDAExt

import AtmosTransport
using AtmosTransport.Architectures: GPU
using CUDA: CuArray, CUDABackend

AtmosTransport.Architectures.array_type(::GPU) = CuArray
AtmosTransport.Architectures.device(::GPU)     = CUDABackend()

end # module AtmosTransportCUDAExt
