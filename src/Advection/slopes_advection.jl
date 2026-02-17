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
    SlopesAdvection{L} <: AbstractAdvectionScheme

Russell-Lerner slopes advection scheme.

# Fields
- `use_limiter :: Bool` — enable/disable flux limiter.
  When off: forward is linear, adjoint is exact (machine precision).
  When on: monotone but adjoint is approximate (continuous adjoint).
"""
struct SlopesAdvection{L} <: AbstractAdvectionScheme
    use_limiter :: L
end

SlopesAdvection(; use_limiter::Bool = true) = SlopesAdvection(use_limiter)

"""
    minmod(a, b, c)

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
    advect_x!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)

Russell-Lerner slopes advection in x (longitude). Periodic boundaries.
"""
function advect_x!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δx_ij = Δx(i, j, grid)
                i_prev = i == 1 ? Nx : i - 1
                i_next = i == Nx ? 1 : i + 1

                # Slope at cell i: centered difference
                s_i = (c[i_next, j, k] - c[i_prev, j, k]) / 2
                if use_limiter
                    s_i = minmod(s_i, 2 * (c[i_next, j, k] - c[i, j, k]), 2 * (c[i, j, k] - c[i_prev, j, k]))
                end

                # Slope at cell i+1 (for flux when u < 0)
                i_next_next = i_next == Nx ? 1 : i_next + 1
                s_i_next = (c[i_next_next, j, k] - c[i, j, k]) / 2
                if use_limiter
                    s_i_next = minmod(s_i_next, 2 * (c[i_next_next, j, k] - c[i_next, j, k]), 2 * (c[i_next, j, k] - c[i, j, k]))
                end

                # Slope at cell i-1 (for flux_left when u > 0)
                i_prev_prev = i_prev == 1 ? Nx : i_prev - 1
                s_i_prev = (c[i, j, k] - c[i_prev_prev, j, k]) / 2
                if use_limiter
                    s_i_prev = minmod(s_i_prev, 2 * (c[i, j, k] - c[i_prev, j, k]), 2 * (c[i_prev, j, k] - c[i_prev_prev, j, k]))
                end

                u_right = u[i + 1, j, k]
                u_left = u[i, j, k]
                Δx_next = Δx(i_next, j, grid)
                Δx_prev = Δx(i_prev, j, grid)

                # Flux at face i+1/2
                if u_right > 0
                    flux_right = u_right * (c[i, j, k] + (1 - u_right * Δt / Δx_ij) * s_i / 2)
                else
                    flux_right = u_right * (c[i_next, j, k] - (1 + u_right * Δt / Δx_next) * s_i_next / 2)
                end

                # Flux at face i-1/2
                if u_left > 0
                    flux_left = u_left * (c[i_prev, j, k] + (1 - u_left * Δt / Δx_prev) * s_i_prev / 2)
                else
                    flux_left = u_left * (c[i, j, k] - (1 + u_left * Δt / Δx_ij) * s_i / 2)
                end

                c_new[i, j, k] = c[i, j, k] - Δt / Δx_ij * (flux_right - flux_left)
            end
        end
        copyto!(c, c_new)
    end
    return nothing
end

"""
    advect_y!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)

Russell-Lerner slopes advection in y (latitude). Bounded boundaries with zero flux.
"""
function advect_y!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δy_ij = Δy(i, j, grid)
                j_prev = j - 1
                j_next = j + 1

                # Slopes: zero at boundaries (no neighbor)
                s_j = if j > 1 && j < Ny
                    s_raw = (c[i, j_next, k] - c[i, j_prev, k]) / 2
                    if use_limiter
                        minmod(s_raw, 2 * (c[i, j_next, k] - c[i, j, k]), 2 * (c[i, j, k] - c[i, j_prev, k]))
                    else
                        s_raw
                    end
                else
                    zero(eltype(c))
                end

                s_j_next = if j_next <= Ny && j_next > 1
                    j_next_next = j_next + 1
                    if j_next_next <= Ny
                        s_raw = (c[i, j_next_next, k] - c[i, j, k]) / 2
                        if use_limiter
                            minmod(s_raw, 2 * (c[i, j_next_next, k] - c[i, j_next, k]), 2 * (c[i, j_next, k] - c[i, j, k]))
                        else
                            s_raw
                        end
                    else
                        zero(eltype(c))
                    end
                else
                    zero(eltype(c))
                end

                s_j_prev = if j_prev >= 1 && j_prev < Ny
                    j_prev_prev = j_prev - 1
                    if j_prev_prev >= 1
                        s_raw = (c[i, j, k] - c[i, j_prev_prev, k]) / 2
                        if use_limiter
                            minmod(s_raw, 2 * (c[i, j, k] - c[i, j_prev, k]), 2 * (c[i, j_prev, k] - c[i, j_prev_prev, k]))
                        else
                            s_raw
                        end
                    else
                        zero(eltype(c))
                    end
                else
                    zero(eltype(c))
                end

                v_right = v[i, j + 1, k]
                v_left = v[i, j, k]
                Δy_next = j_next <= Ny ? Δy(i, j_next, grid) : Δy_ij
                Δy_prev = j_prev >= 1 ? Δy(i, j_prev, grid) : Δy_ij

                # Flux at face j+1/2 (right)
                flux_right = if j < Ny
                    if v_right > 0
                        v_right * (c[i, j, k] + (1 - v_right * Δt / Δy_ij) * s_j / 2)
                    else
                        v_right * (c[i, j_next, k] - (1 + v_right * Δt / Δy_next) * s_j_next / 2)
                    end
                else
                    v_right > 0 ? v_right * c[i, j, k] : zero(eltype(c))
                end

                # Flux at face j-1/2 (left)
                flux_left = if j > 1
                    if v_left > 0
                        v_left * (c[i, j_prev, k] + (1 - v_left * Δt / Δy_prev) * s_j_prev / 2)
                    else
                        v_left * (c[i, j, k] - (1 + v_left * Δt / Δy_ij) * s_j / 2)
                    end
                else
                    v_left <= 0 ? v_left * c[i, j, k] : zero(eltype(c))
                end

                c_new[i, j, k] = c[i, j, k] - Δt / Δy_ij * (flux_right - flux_left)
            end
        end
        copyto!(c, c_new)
    end
    return nothing
end

"""
    advect_z!(tracers, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)

Russell-Lerner slopes advection in z (vertical). Bounded boundaries with zero flux.
"""
function advect_z!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δz_k = Δz(k, grid)
                k_prev = k - 1
                k_next = k + 1

                # Slopes: zero at boundaries
                s_k = if k > 1 && k < Nz
                    s_raw = (c[i, j, k_next] - c[i, j, k_prev]) / 2
                    if use_limiter
                        minmod(s_raw, 2 * (c[i, j, k_next] - c[i, j, k]), 2 * (c[i, j, k] - c[i, j, k_prev]))
                    else
                        s_raw
                    end
                else
                    zero(eltype(c))
                end

                s_k_next = if k_next <= Nz && k_next > 1
                    k_next_next = k_next + 1
                    if k_next_next <= Nz
                        s_raw = (c[i, j, k_next_next] - c[i, j, k]) / 2
                        if use_limiter
                            minmod(s_raw, 2 * (c[i, j, k_next_next] - c[i, j, k_next]), 2 * (c[i, j, k_next] - c[i, j, k]))
                        else
                            s_raw
                        end
                    else
                        zero(eltype(c))
                    end
                else
                    zero(eltype(c))
                end

                s_k_prev = if k_prev >= 1 && k_prev < Nz
                    k_prev_prev = k_prev - 1
                    if k_prev_prev >= 1
                        s_raw = (c[i, j, k] - c[i, j, k_prev_prev]) / 2
                        if use_limiter
                            minmod(s_raw, 2 * (c[i, j, k] - c[i, j, k_prev]), 2 * (c[i, j, k_prev] - c[i, j, k_prev_prev]))
                        else
                            s_raw
                        end
                    else
                        zero(eltype(c))
                    end
                else
                    zero(eltype(c))
                end

                w_top = w[i, j, k]
                w_bot = w[i, j, k + 1]
                Δz_next = k_next <= Nz ? Δz(k_next, grid) : Δz_k
                Δz_prev = k_prev >= 1 ? Δz(k_prev, grid) : Δz_k

                # Flux at top face (k)
                flux_top = if k > 1
                    if w_top > 0
                        w_top * (c[i, j, k_prev] + (1 - w_top * Δt / Δz_prev) * s_k_prev / 2)
                    else
                        w_top * (c[i, j, k] - (1 + w_top * Δt / Δz_k) * s_k / 2)
                    end
                else
                    w_top <= 0 ? w_top * c[i, j, k] : zero(eltype(c))
                end

                # Flux at bottom face (k+1)
                flux_bot = if k < Nz
                    if w_bot > 0
                        w_bot * (c[i, j, k] + (1 - w_bot * Δt / Δz_k) * s_k / 2)
                    else
                        w_bot * (c[i, j, k_next] - (1 + w_bot * Δt / Δz_next) * s_k_next / 2)
                    end
                else
                    w_bot > 0 ? w_bot * c[i, j, k] : zero(eltype(c))
                end

                c_new[i, j, k] = c[i, j, k] - Δt / Δz_k * (flux_bot - flux_top)
            end
        end
        copyto!(c, c_new)
    end
    return nothing
end
