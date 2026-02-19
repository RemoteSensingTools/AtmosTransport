"""
CUDA extension for AtmosTransportModel.

Loaded automatically when `using CUDA` is called alongside AtmosTransportModel.
Provides GPU array types and KernelAbstractions device for the GPU architecture.
"""
module AtmosTransportModelCUDAExt

import AtmosTransportModel
using AtmosTransportModel.Architectures: GPU
using CUDA: CuArray, CUDABackend

AtmosTransportModel.Architectures.array_type(::GPU) = CuArray
AtmosTransportModel.Architectures.device(::GPU)     = CUDABackend()

end # module AtmosTransportModelCUDAExt
