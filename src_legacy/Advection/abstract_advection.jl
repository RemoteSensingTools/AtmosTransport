# ---------------------------------------------------------------------------
# Abstract advection scheme and interface contract
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Supertype for all advection schemes. Each subtype must implement both
forward and adjoint methods.

# Required methods

    advect!(tracers, velocities, grid, scheme, Δt)
    adjoint_advect!(adj_tracers, velocities, grid, scheme, Δt)

These may further dispatch on grid type for grid-specific boundary handling.

# Directional variants

For operator splitting, directional methods are also needed:

    advect_x!(tracers, velocities, grid, scheme, Δt)
    advect_y!(tracers, velocities, grid, scheme, Δt)
    advect_z!(tracers, velocities, grid, scheme, Δt)

and their adjoints:

    adjoint_advect_x!(adj_tracers, velocities, grid, scheme, Δt)
    adjoint_advect_y!(adj_tracers, velocities, grid, scheme, Δt)
    adjoint_advect_z!(adj_tracers, velocities, grid, scheme, Δt)
"""
abstract type AbstractAdvectionScheme end

# ---------------------------------------------------------------------------
# Default error-throwing methods (enforces the contract)
# ---------------------------------------------------------------------------

function advect!(tracers, vel, grid, scheme::AbstractAdvectionScheme, Δt)
    error("advect! not implemented for scheme=$(typeof(scheme)), grid=$(typeof(grid))")
end

function adjoint_advect!(adj_tracers, vel, grid, scheme::AbstractAdvectionScheme, Δt)
    error("adjoint_advect! not implemented for scheme=$(typeof(scheme)), grid=$(typeof(grid))")
end

for dir in (:x, :y, :z)
    fwd = Symbol(:advect_, dir, :!)
    adj = Symbol(:adjoint_advect_, dir, :!)
    @eval function $fwd(tracers, vel, grid, scheme::AbstractAdvectionScheme, Δt)
        error(string($fwd) * " not implemented for scheme=$(typeof(scheme)), grid=$(typeof(grid))")
    end
    @eval function $adj(adj_tracers, vel, grid, scheme::AbstractAdvectionScheme, Δt)
        error(string($adj) * " not implemented for scheme=$(typeof(scheme)), grid=$(typeof(grid))")
    end
end

# ---------------------------------------------------------------------------
# Optional: CFL calculation (has a default)
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Compute the advective CFL number. Default is a generic estimate;
schemes may override for tighter bounds.
"""
function advection_cfl(grid, velocities, ::AbstractAdvectionScheme)
    return NaN  # generic: override for specific schemes
end
