"""
    Models (v2)

Minimal standalone runtime layer for `src_v2`.
"""
module Models

using Adapt
using ..State
using ..Grids
using ..Operators
using ..MetDrivers

include("TransportModel.jl")
include("Simulation.jl")
include("DrivenSimulation.jl")

end # module Models
