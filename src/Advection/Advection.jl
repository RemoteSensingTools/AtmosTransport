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

using ..Grids: AbstractGrid, AbstractStructuredGrid, LatitudeLongitudeGrid, Δx, Δy, Δz
using ..Fields: AbstractField

export AbstractAdvectionScheme
export SlopesAdvection, UpwindAdvection
export advect!, adjoint_advect!
export advect_x!, advect_y!, advect_z!
export adjoint_advect_x!, adjoint_advect_y!, adjoint_advect_z!
export advection_cfl

include("abstract_advection.jl")
include("slopes_advection.jl")
include("slopes_advection_adjoint.jl")
include("upwind_advection.jl")
include("upwind_advection_adjoint.jl")

end # module Advection
