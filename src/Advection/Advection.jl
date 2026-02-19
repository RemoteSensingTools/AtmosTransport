"""
    Advection

Advection schemes for tracer transport, each with a paired discrete adjoint.

Every advection scheme dispatches on both the scheme type AND the grid type,
so that grid-specific stencils (e.g. cubed-sphere panel boundaries) are
handled transparently.

# Interface contract for new schemes

    advect!(tracers, velocities, grid, scheme::YourScheme, Δt)
    adjoint_advect!(adj_tracers, velocities, grid, scheme::YourScheme, Δt)

Optional (has defaults):

    advection_cfl(grid, velocities, scheme::YourScheme)
"""
module Advection

using DocStringExtensions

using ..Grids: AbstractGrid, AbstractStructuredGrid, LatitudeLongitudeGrid, Δx, Δy, Δz, floattype
using ..Grids: ReducedGridSpec, reduce_row!, expand_row!, reduce_velocity_row!
using ..Fields: AbstractField
using ..Architectures: architecture, device, array_type, CPU, GPU

export AbstractAdvectionScheme
export SlopesAdvection, UpwindAdvection
export advect!, adjoint_advect!
export advect_x!, advect_y!, advect_z!
export adjoint_advect_x!, adjoint_advect_y!, adjoint_advect_z!
export advection_cfl
export max_cfl_x, max_cfl_y, max_cfl_z, subcycling_counts
export advect_x_subcycled!, advect_y_subcycled!, advect_z_subcycled!

include("abstract_advection.jl")
include("slopes_advection_kernels.jl")
include("slopes_advection.jl")
include("slopes_advection_adjoint.jl")
include("upwind_advection.jl")
include("upwind_advection_adjoint.jl")
include("subcycling.jl")

end # module Advection
