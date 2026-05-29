"""
NVTX extension for AtmosTransport.

Loaded automatically when `using NVTX` is called alongside AtmosTransport.
Overrides `SectionTimer._nvtx_start` / `_nvtx_end` so that, with
`ATMOSTR_NVTX=1`, every `@section` and `time_section` region emits a
matching NVTX range. Without this extension loaded, those stubs are
no-ops and SectionTimer falls back to host-side timing only.

To activate, the user adds NVTX to their environment alongside
AtmosTransport:

    julia> using Pkg; Pkg.add("NVTX")
    julia> using NVTX
    julia> using AtmosTransport   # extension auto-loads

Then launch under `nsys profile --trace=cuda,nvtx ...` with
`ATMOSTR_NVTX=1` to see labeled regions on the Nsight Systems timeline.
"""
module AtmosTransportNVTXExt

import AtmosTransport
import NVTX

const ST = AtmosTransport.SectionTimer

ST._nvtx_start(label::AbstractString) = NVTX.range_start(; message = label)
ST._nvtx_end(handle::NVTX.RangeId) = NVTX.range_end(handle)

end # module AtmosTransportNVTXExt
