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
# plan 40 Commit 1c — bring Regridding + Preprocessing into Models' namespace
# so the nested `InitialConditionIO` submodule can `using ..Regridding` etc.
# Regridding + Preprocessing are loaded before Models in AtmosTransport.jl.
using ..Regridding
using ..Preprocessing

include("TransportModel.jl")
include("CSPhysicsRecipe.jl")
include("InitialConditionIO.jl")  # plan 40 Commit 1b: LL/RG hoisted; CS in 1c
using .InitialConditionIO: FileInitialConditionSource,
                           build_initial_mixing_ratio,
                           pack_initial_tracer_mass,
                           FileSurfaceFluxField,
                           build_surface_flux_source,
                           build_surface_flux_sources
export FileInitialConditionSource, build_initial_mixing_ratio, pack_initial_tracer_mass
export FileSurfaceFluxField, build_surface_flux_source, build_surface_flux_sources
include("BinaryPathExpander.jl")  # plan 40 Commit 4: `[input]` folder+date-range
using .BinaryPathExpander: expand_binary_paths
export expand_binary_paths
include("Simulation.jl")
include("DrivenSimulation.jl")
include("DrivenRunner.jl")        # plan 40 Commit 6a: library-level driven runner
using .DrivenRunner: run_driven_simulation, TransportTracerSpec
export run_driven_simulation, TransportTracerSpec

end # module Models
