# ---------------------------------------------------------------------------
# Operator-splitting time stepper — adjoint
#
# For Strang splitting the adjoint reverses temporal order and transposes
# each operator. Due to the symmetry of Strang splitting, the operator
# ORDER within one step is the same, but we run BACKWARD in time.
#
# Adjoint step (one Δt, running backward):
#   1. adjoint_advect_x  (Δt/2)
#   2. adjoint_advect_y  (Δt/2)
#   3. adjoint_advect_z  (Δt/2)
#   4. adjoint_chemistry (Δt)
#   5. adjoint_diffuse   (Δt)
#   6. adjoint_convect   (Δt)
#   7. adjoint_advect_z  (Δt/2)
#   8. adjoint_advect_y  (Δt/2)
#   9. adjoint_advect_x  (Δt/2)
# ---------------------------------------------------------------------------

using ..Advection: adjoint_advect_x!, adjoint_advect_y!, adjoint_advect_z!
using ..Convection: adjoint_convect!
using ..Diffusion: adjoint_diffuse!
using ..Chemistry: adjoint_chemistry!

"""
$(SIGNATURES)

Perform one adjoint time step (backward in time).
`model` must have fields: `adj_tracers`, `met_data`, `grid`, `timestepper`, `clock`.
"""
function adjoint_time_step!(model, Δt)
    ts   = model.timestepper
    atr  = model.adj_tracers
    met  = model.met_data
    g    = model.grid
    half = Δt / 2

    vel = _extract_velocities(met)

    adjoint_advect_x!(atr, vel, g, ts.advection, half)
    adjoint_advect_y!(atr, vel, g, ts.advection, half)
    adjoint_advect_z!(atr, vel, g, ts.advection, half)

    adjoint_chemistry!(atr, g, ts.chemistry, Δt)
    adjoint_diffuse!(atr, met, g, ts.diffusion, Δt)
    adjoint_convect!(atr, met, g, ts.convection, Δt)

    adjoint_advect_z!(atr, vel, g, ts.advection, half)
    adjoint_advect_y!(atr, vel, g, ts.advection, half)
    adjoint_advect_x!(atr, vel, g, ts.advection, half)

    tick_backward!(model.clock, Δt)
    return nothing
end
