# ---------------------------------------------------------------------------
# CFL-adaptive subcycling for advection
#
# At fine resolutions (e.g. 0.25°), the outer time step may violate CFL
# in one or more directions. These routines automatically subcycle the
# advection to keep CFL below a safe limit.
# ---------------------------------------------------------------------------

"""
    max_cfl_x(velocities, grid::LatitudeLongitudeGrid, dt)

Maximum CFL number for x-advection, accounting for the reduced grid.
"""
function max_cfl_x(velocities, grid::LatitudeLongitudeGrid, dt)
    u = Array(velocities.u)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    rg = grid.reduced_grid
    cfl_max = zero(Float64)
    @inbounds for k in 1:Nz, j in 1:Ny
        cluster = rg === nothing ? 1 : rg.cluster_sizes[j]
        dx_eff = Float64(Δx(1, j, grid)) * cluster
        for i in 1:(Nx + 1)
            cfl_max = max(cfl_max, abs(Float64(u[i, j, k])) * abs(dt) / dx_eff)
        end
    end
    return cfl_max
end

"""
    max_cfl_y(velocities, grid::LatitudeLongitudeGrid, dt)

Maximum CFL number for y-advection.
"""
function max_cfl_y(velocities, grid::LatitudeLongitudeGrid, dt)
    v = Array(velocities.v)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dy = Float64(Δy(1, 1, grid))
    cfl_max = zero(Float64)
    @inbounds for k in 1:Nz, j in 1:(Ny + 1), i in 1:Nx
        cfl_max = max(cfl_max, abs(Float64(v[i, j, k])) * abs(dt) / dy)
    end
    return cfl_max
end

"""
    max_cfl_z(velocities, grid::LatitudeLongitudeGrid, dt)

Maximum CFL number for z-advection.
"""
function max_cfl_z(velocities, grid::LatitudeLongitudeGrid, dt)
    w = Array(velocities.w)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    cfl_max = zero(Float64)
    @inbounds for k in 1:(Nz + 1), j in 1:Ny, i in 1:Nx
        kk = clamp(k, 1, Nz)
        dz = Float64(Δz(kk, grid))
        cfl_max = max(cfl_max, abs(Float64(w[i, j, k])) * abs(dt) / dz)
    end
    return cfl_max
end

"""
    subcycling_counts(velocities, grid, dt; cfl_limit=0.95)

Compute the number of sub-steps needed for each advection direction.
Returns `(nx, ny, nz)` where each is ≥ 1.
"""
function subcycling_counts(velocities, grid::LatitudeLongitudeGrid, dt;
                           cfl_limit=0.95)
    cx = max_cfl_x(velocities, grid, dt)
    cy = max_cfl_y(velocities, grid, dt)
    cz = max_cfl_z(velocities, grid, dt)

    nx = max(1, ceil(Int, cx / cfl_limit))
    ny = max(1, ceil(Int, cy / cfl_limit))
    nz = max(1, ceil(Int, cz / cfl_limit))

    return (; nx, ny, nz, cfl_x=cx, cfl_y=cy, cfl_z=cz)
end

"""
    advect_x_subcycled!(tracers, vel, grid, scheme, dt; n_sub=nothing, cfl_limit=0.95)

Subcycled x-advection. If `n_sub` is provided, uses that many sub-steps;
otherwise automatically determines from CFL.
"""
function advect_x_subcycled!(tracers, vel, grid, scheme, dt;
                              n_sub=nothing, cfl_limit=0.95)
    if n_sub === nothing
        cfl = max_cfl_x(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_x!(tracers, vel, grid, scheme, dt_sub)
    end
    return n_sub
end

"""
    advect_y_subcycled!(tracers, vel, grid, scheme, dt; n_sub=nothing, cfl_limit=0.95)

Subcycled y-advection.
"""
function advect_y_subcycled!(tracers, vel, grid, scheme, dt;
                              n_sub=nothing, cfl_limit=0.95)
    if n_sub === nothing
        cfl = max_cfl_y(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_y!(tracers, vel, grid, scheme, dt_sub)
    end
    return n_sub
end

"""
    advect_z_subcycled!(tracers, vel, grid, scheme, dt; n_sub=nothing, cfl_limit=0.95)

Subcycled z-advection.
"""
function advect_z_subcycled!(tracers, vel, grid, scheme, dt;
                              n_sub=nothing, cfl_limit=0.95)
    if n_sub === nothing
        cfl = max_cfl_z(vel, grid, dt)
        n_sub = max(1, ceil(Int, cfl / cfl_limit))
    end
    dt_sub = dt / n_sub
    for _ in 1:n_sub
        advect_z!(tracers, vel, grid, scheme, dt_sub)
    end
    return n_sub
end
