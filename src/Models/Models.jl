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
include("InitialConditionIO.jl")  # plan 40 Commit 1a: scaffold; 1b/1c fill
include("Simulation.jl")
include("DrivenSimulation.jl")

end # module Models
