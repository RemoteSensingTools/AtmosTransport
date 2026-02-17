"""
CUDA extension for AtmosTransportModel.

Loaded automatically when `using CUDA` is called alongside AtmosTransportModel.
Provides GPU array types and KernelAbstractions device for the GPU architecture.
"""
module AtmosTransportModelCUDAExt

using AtmosTransportModel.Architectures: GPU, array_type, device
using CUDA: CuArray, CUDABackend
using KernelAbstractions: KernelAbstractions as KA

# Override the error-throwing stubs in Architectures.jl
AtmosTransportModel.Architectures.array_type(::GPU) = CuArray
AtmosTransportModel.Architectures.device(::GPU) = CUDABackend()

end # module AtmosTransportModelCUDAExt
