"""
    Architectures (v2)

Re-exports from the existing Architectures module for backward compatibility.
The v2 architecture layer is identical to v1 — CPU/GPU abstraction, array_type,
device, KernelAbstractions backend.  We re-export rather than duplicate so that
both `src/` and `src_v2/` share the same concrete types (`CPU`, `GPU`).
"""
module Architectures

# Pull everything from the production Architectures module.
# This file is the v2 entry point; the actual code lives in src/.
using ...AtmosTransport.Architectures: AbstractArchitecture, CPU, GPU,
    array_type, device, architecture, _kahan_add,
    AbstractPanelMap, SingleGPUMap, PanelGPUMap

export AbstractArchitecture, CPU, GPU
export array_type, device, architecture, _kahan_add

end # module
