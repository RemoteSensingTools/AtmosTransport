# ---------------------------------------------------------------------------
# Slopes advection scheme (Russell & Lerner, 1981)
#
# TM5's default advection method. Predicts mean concentration and its spatial
# gradients (slopes) as prognostic quantities. Second-order accurate.
#
# The scheme is linear in tracer concentration when the flux limiter is off,
# enabling an exact discrete adjoint.
#
# Forward implementation for LatitudeLongitudeGrid.
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Russell-Lerner slopes advection scheme.

$(FIELDS)
"""
struct SlopesAdvection{L} <: AbstractAdvectionScheme
    "enable/disable flux limiter. When off: forward is linear, adjoint is exact (machine precision). When on: monotone but adjoint is approximate (continuous adjoint)."
    use_limiter :: L
    "use TM5-style prognostic slopes (rxm/rym/rzm evolved with pf-term). Requires more GPU memory but enables mass_fixer=false."
    prognostic_slopes :: Bool
end

SlopesAdvection(; use_limiter::Bool = true, prognostic_slopes::Bool = false) =
    SlopesAdvection(use_limiter, prognostic_slopes)

"""
$(SIGNATURES)

Minmod limiter: returns the value with smallest magnitude if all have the same sign,
otherwise zero.
"""
function minmod(a, b, c)
    if a > 0 && b > 0 && c > 0
        return min(a, b, c)
    elseif a < 0 && b < 0 && c < 0
        return max(a, b, c)
    else
        return zero(a)
    end
end

"""
$(SIGNATURES)

Russell-Lerner slopes advection in x (longitude). Periodic boundaries.
When the grid has a reduced grid specification, high-latitude rows are
advected on a coarser zonal grid (TM5-style) to avoid polar CFL violations.
"""
function advect_x!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    FT = floattype(grid)
    arch = architecture(grid)

    # KA kernel — works on both CPU (Array) and GPU (CuArray) via get_backend
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        backend = get_backend(c)
        Δx_j = array_type(arch)(FT[Δx(1, j, grid) for j in 1:Ny])
        kernel! = advect_x_kernel(backend, 256)
        kernel!(c_new, c, u, Δx_j, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
        synchronize(backend)
        copyto!(c, c_new)
    end
    return nothing
end

"""Advect one (j,k) row on the full uniform grid (no reduction)."""
function _advect_x_uniform_row!(c, c_new, u, grid, j, k, Δt, use_limiter)
    Nx = grid.Nx
    @inbounds for i in 1:Nx
        Δx_ij = Δx(i, j, grid)
        i_prev = i == 1 ? Nx : i - 1
        i_next = i == Nx ? 1 : i + 1

        s_i = (c[i_next, j, k] - c[i_prev, j, k]) / 2
        if use_limiter
            s_i = minmod(s_i, 2 * (c[i_next, j, k] - c[i, j, k]), 2 * (c[i, j, k] - c[i_prev, j, k]))
        end

        i_next_next = i_next == Nx ? 1 : i_next + 1
        s_i_next = (c[i_next_next, j, k] - c[i, j, k]) / 2
        if use_limiter
            s_i_next = minmod(s_i_next, 2 * (c[i_next_next, j, k] - c[i_next, j, k]), 2 * (c[i_next, j, k] - c[i, j, k]))
        end

        i_prev_prev = i_prev == 1 ? Nx : i_prev - 1
        s_i_prev = (c[i, j, k] - c[i_prev_prev, j, k]) / 2
        if use_limiter
            s_i_prev = minmod(s_i_prev, 2 * (c[i, j, k] - c[i_prev, j, k]), 2 * (c[i_prev, j, k] - c[i_prev_prev, j, k]))
        end

        u_right = u[i + 1, j, k]
        u_left = u[i, j, k]
        Δx_next = Δx(i_next, j, grid)
        Δx_prev = Δx(i_prev, j, grid)

        flux_right = if u_right > 0
            u_right * (c[i, j, k] + (1 - u_right * Δt / Δx_ij) * s_i / 2)
        else
            u_right * (c[i_next, j, k] - (1 + u_right * Δt / Δx_next) * s_i_next / 2)
        end

        flux_left = if u_left > 0
            u_left * (c[i_prev, j, k] + (1 - u_left * Δt / Δx_prev) * s_i_prev / 2)
        else
            u_left * (c[i, j, k] - (1 + u_left * Δt / Δx_ij) * s_i / 2)
        end

        c_new[i, j, k] = c[i, j, k] - Δt / Δx_ij * (flux_right - flux_left)
    end
    return nothing
end

"""
Advect one (j,k) row using the reduced grid: average fine cells → advect
on coarser row → distribute the change back to fine cells.
"""
function _advect_x_reduced_row!(c, c_new, u, grid, j, k, cluster, Δt, use_limiter)
    Nx = grid.Nx
    Nx_red = Nx ÷ cluster
    FT = floattype(grid)
    Δx_red = Δx(1, j, grid) * cluster   # effective spacing on reduced grid

    c_red     = Vector{FT}(undef, Nx_red)
    c_red_old = Vector{FT}(undef, Nx_red)
    c_red_new = Vector{FT}(undef, Nx_red)
    u_red     = Vector{FT}(undef, Nx_red + 1)

    reduce_row!(c_red, c, j, k, cluster, Nx)
    copyto!(c_red_old, c_red)
    reduce_velocity_row!(u_red, u, j, k, cluster, Nx)

    @inbounds for i in 1:Nx_red
        i_prev = i == 1 ? Nx_red : i - 1
        i_next = i == Nx_red ? 1 : i + 1

        s_i = (c_red[i_next] - c_red[i_prev]) / 2
        if use_limiter
            s_i = minmod(s_i, 2 * (c_red[i_next] - c_red[i]), 2 * (c_red[i] - c_red[i_prev]))
        end

        i_next_next = i_next == Nx_red ? 1 : i_next + 1
        s_i_next = (c_red[i_next_next] - c_red[i]) / 2
        if use_limiter
            s_i_next = minmod(s_i_next, 2 * (c_red[i_next_next] - c_red[i_next]), 2 * (c_red[i_next] - c_red[i]))
        end

        i_prev_prev = i_prev == 1 ? Nx_red : i_prev - 1
        s_i_prev = (c_red[i] - c_red[i_prev_prev]) / 2
        if use_limiter
            s_i_prev = minmod(s_i_prev, 2 * (c_red[i] - c_red[i_prev]), 2 * (c_red[i_prev] - c_red[i_prev_prev]))
        end

        ur = u_red[i + 1]
        ul = u_red[i]

        flux_right = if ur > 0
            ur * (c_red[i] + (1 - ur * Δt / Δx_red) * s_i / 2)
        else
            ur * (c_red[i_next] - (1 + ur * Δt / Δx_red) * s_i_next / 2)
        end

        flux_left = if ul > 0
            ul * (c_red[i_prev] + (1 - ul * Δt / Δx_red) * s_i_prev / 2)
        else
            ul * (c_red[i] - (1 + ul * Δt / Δx_red) * s_i / 2)
        end

        c_red_new[i] = c_red[i] - Δt / Δx_red * (flux_right - flux_left)
    end

    # Initialize c_new from c, then add reduced-grid deltas
    @inbounds for i in 1:Nx
        c_new[i, j, k] = c[i, j, k]
    end
    expand_row!(c_new, c_red_new, c_red_old, j, k, cluster, Nx)
    return nothing
end

"""
$(SIGNATURES)

Russell-Lerner slopes advection in y (latitude). Bounded boundaries with zero flux.
"""
function advect_y!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    FT = floattype(grid)
    Δy_val = Δy(1, 1, grid)

    # KA kernel — works on both CPU (Array) and GPU (CuArray) via get_backend
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        backend = get_backend(c)
        kernel! = advect_y_kernel(backend, 256)
        kernel!(c_new, c, v, Δy_val, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
        synchronize(backend)
        copyto!(c, c_new)
    end
    return nothing
end

"""
$(SIGNATURES)

Russell-Lerner slopes advection in z (vertical). Bounded boundaries with zero flux
at model top (k=1) and surface (k=Nz+1).

Sign convention: `w > 0` means **downward** — flow from level k-1 toward level k
(increasing k index, toward the surface). This matches the ERA5/ECMWF omega
convention where `omega > 0` is downward (increasing pressure).
"""
function advect_z!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    FT = floattype(grid)
    arch = architecture(grid)
    ps = _get_p_surface(velocities)
    Δz_3d = _build_Δz_3d(grid, ps)

    # Upload Δz to the correct device (GPU or CPU)
    Δz_dev = array_type(arch)(Δz_3d)

    # KA kernel — works on both CPU (Array) and GPU (CuArray) via get_backend
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        backend = get_backend(c)
        kernel! = advect_z_kernel(backend, 256)
        kernel!(c_new, c, w, Δz_dev, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
        synchronize(backend)
        copyto!(c, c_new)
    end
    return nothing
end
