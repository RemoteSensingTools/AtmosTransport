"""
    Models (v2)

Minimal standalone runtime layer for `src_v2`.
"""
module Models

using ..State
using ..Grids
using ..Operators

include("TransportModel.jl")
include("Simulation.jl")

end # module Models
