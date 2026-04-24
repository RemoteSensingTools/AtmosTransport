"""
    Models

Minimal standalone runtime layer for `src`.
"""
module Models

using Adapt
using ..State
using ..Grids
using ..Operators
using ..MetDrivers

include("TransportModel.jl")
include("CSPhysicsRecipe.jl")
include("InitialConditionIO.jl")  # plan 40 Commit 1b: LL/RG hoisted; CS in 1c
using .InitialConditionIO: FileInitialConditionSource,
                           build_initial_mixing_ratio,
                           pack_initial_tracer_mass
export FileInitialConditionSource, build_initial_mixing_ratio, pack_initial_tracer_mass
include("Simulation.jl")
include("DrivenSimulation.jl")

end # module Models
