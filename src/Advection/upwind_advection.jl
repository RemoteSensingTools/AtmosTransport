# ---------------------------------------------------------------------------
# First-order upwind advection (reference scheme)
#
# Simple, diffusive, but useful for testing and as a baseline.
# Linear in tracer concentration, so adjoint is exact.
#
# Tracers and velocities are NamedTuples of 3D arrays (not Field objects).
# u: (Nx+1, Ny, Nz), v: (Nx, Ny+1, Nz), w: (Nx, Ny, Nz+1)
# ---------------------------------------------------------------------------

"""
    UpwindAdvection <: AbstractAdvectionScheme

First-order upwind advection. Simple reference scheme for testing.
"""
struct UpwindAdvection <: AbstractAdvectionScheme end

# ---------------------------------------------------------------------------
# Forward: advect_x!, advect_y!, advect_z!
# ---------------------------------------------------------------------------

"""
    advect_x!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

First-order upwind advection in x (longitude). Periodic boundaries.
"""
function advect_x!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δx_ij = Δx(i, j, grid)
            i_prev = i == 1 ? Nx : i - 1
            i_next = i == Nx ? 1 : i + 1
            u_right = u[i + 1, j, k]
            u_left = u[i, j, k]
            flux_right = u_right > 0 ? u_right * c[i, j, k] : u_right * c[i_next, j, k]
            flux_left = u_left > 0 ? u_left * c[i_prev, j, k] : u_left * c[i, j, k]
            c_new[i, j, k] = c[i, j, k] - Δt / Δx_ij * (flux_right - flux_left)
        end
        copyto!(c, c_new)
    end
    return nothing
end

"""
    advect_y!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

First-order upwind advection in y (latitude). Bounded boundaries with zero flux.
"""
function advect_y!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δy_ij = Δy(i, j, grid)
            j_prev = j - 1
            j_next = j + 1
            v_right = v[i, j + 1, k]
            v_left = v[i, j, k]
            flux_right = if j < Ny
                v_right > 0 ? v_right * c[i, j, k] : v_right * c[i, j_next, k]
            else
                v_right > 0 ? v_right * c[i, j, k] : zero(eltype(c))
            end
            flux_left = if j > 1
                v_left > 0 ? v_left * c[i, j_prev, k] : v_left * c[i, j, k]
            else
                v_left <= 0 ? v_left * c[i, j, k] : zero(eltype(c))
            end
            c_new[i, j, k] = c[i, j, k] - Δt / Δy_ij * (flux_right - flux_left)
        end
        copyto!(c, c_new)
    end
    return nothing
end

"""
    advect_z!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

First-order upwind advection in z (vertical). Bounded boundaries with zero flux.
"""
function advect_z!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δz_k = Δz(k, grid)
            k_prev = k - 1
            k_next = k + 1
            w_top = w[i, j, k]
            w_bot = w[i, j, k + 1]
            flux_top = if k > 1
                w_top > 0 ? w_top * c[i, j, k_prev] : w_top * c[i, j, k]
            else
                w_top <= 0 ? w_top * c[i, j, k] : zero(eltype(c))
            end
            flux_bot = if k < Nz
                w_bot > 0 ? w_bot * c[i, j, k] : w_bot * c[i, j, k_next]
            else
                w_bot > 0 ? w_bot * c[i, j, k] : zero(eltype(c))
            end
            c_new[i, j, k] = c[i, j, k] - Δt / Δz_k * (flux_bot - flux_top)
        end
        copyto!(c, c_new)
    end
    return nothing
end
