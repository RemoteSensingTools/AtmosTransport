"""
Metal extension for AtmosTransport.

Loaded automatically when `using Metal` is called alongside AtmosTransport.
Provides GPU array types and KernelAbstractions device for Apple Silicon GPUs.
"""
module AtmosTransportMetalExt

import AtmosTransport
using AtmosTransport.Architectures: GPU
using Metal: MtlArray, MetalBackend

AtmosTransport.Architectures.array_type(::GPU) = MtlArray
AtmosTransport.Architectures.device(::GPU)     = MetalBackend()

end # module AtmosTransportMetalExt
