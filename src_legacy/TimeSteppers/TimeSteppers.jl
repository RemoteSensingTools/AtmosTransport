"""
    TimeSteppers

Time integration for the atmospheric transport model.

The primary scheme is symmetric Strang operator splitting (TM5-style):
advect_x, advect_y, advect_z, convect, diffuse, sources, advect_z, advect_y, advect_x.

The adjoint time step reverses the temporal order of operators and
calls the adjoint of each operator.

# Interface contract

    time_step!(model, Δt)
    adjoint_time_step!(model, Δt)
"""
module TimeSteppers

using DocStringExtensions

export AbstractTimeStepper, OperatorSplittingTimeStepper
export Clock
export time_step!, adjoint_time_step!, tick!, tick_backward!

include("clock.jl")
include("operator_splitting.jl")
include("operator_splitting_adjoint.jl")

end # module TimeSteppers
