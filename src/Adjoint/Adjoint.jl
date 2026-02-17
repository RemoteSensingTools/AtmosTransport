"""
    Adjoint

Infrastructure for the hand-coded discrete adjoint:
- Revolve-style checkpointing for memory-bounded adjoint runs
- 4DVar cost function structure
- Gradient test utility (adjoint vs finite-difference verification)
"""
module Adjoint

using ..Grids
using ..TimeSteppers

export AbstractCheckpointer, StoreAllCheckpointer, RevolveCheckpointer
export AbstractCostFunction, CostFunction4DVar
export AbstractObservationOperator
export run_adjoint!, gradient_test

include("checkpointing.jl")
include("cost_functions.jl")
include("gradient_test.jl")

end # module Adjoint
