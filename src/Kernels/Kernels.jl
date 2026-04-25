"""
    Kernels

KernelAbstractions kernel patterns organized by dependency structure.

Three kernel families:
- **Cell kernels** — one thread per (cell, level): sources, mass updates, diagnostics
- **Face kernels** — one thread per (face, level): reconstruction, flux evaluation
- **Column kernels** — one thread per column: tridiagonal solves, cm diagnosis

This organization is more stable than grouping by physics process and maps
naturally to both structured and unstructured grid execution patterns.
"""
module Kernels

using ..Architectures: _kahan_add

include("CellKernels.jl")
include("FaceKernels.jl")
include("ColumnKernels.jl")

end # module Kernels
