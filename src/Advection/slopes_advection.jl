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
end

SlopesAdvection(; use_limiter::Bool = true) = SlopesAdvection(use_limiter)

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
    arch = architecture(grid)
    rg = grid.reduced_grid

    if arch isa GPU
        FT = floattype(grid)
        for (name, c) in pairs(tracers)
            c_new = similar(c)
            Δx_j = array_type(arch)([Δx(1, j, grid) for j in 1:Ny])
            kernel = advect_x_kernel(device(arch), 256)
            event = kernel(c_new, c, u, Δx_j, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
            wait(device(arch), event)
            copyto!(c, c_new)
        end
        return nothing
    end

    for (name, c) in pairs(tracers)
        c_new = similar(c)
        for k in 1:Nz, j in 1:Ny
            cluster = rg === nothing ? 1 : rg.cluster_sizes[j]

            if cluster > 1
                _advect_x_reduced_row!(c, c_new, u, grid, j, k, cluster, Δt, use_limiter)
            else
                _advect_x_uniform_row!(c, c_new, u, grid, j, k, Δt, use_limiter)
            end
        end
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
    arch = architecture(grid)

    if arch isa GPU
        FT = floattype(grid)
        Δy_val = Δy(1, 1, grid)
        for (name, c) in pairs(tracers)
            c_new = similar(c)
            kernel = advect_y_kernel(device(arch), 256)
            event = kernel(c_new, c, v, Δy_val, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
            wait(device(arch), event)
            copyto!(c, c_new)
        end
        return nothing
    end

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
$(SIGNATURES)

Russell-Lerner slopes advection in z (vertical). Bounded boundaries with zero flux.
"""
function advect_z!(tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::SlopesAdvection, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    use_limiter = scheme.use_limiter
    arch = architecture(grid)

    if arch isa GPU
        FT = floattype(grid)
        Δz_arr = array_type(arch)(FT[Δz(k, grid) for k in 1:Nz])
        for (name, c) in pairs(tracers)
            c_new = similar(c)
            kernel = advect_z_kernel(device(arch), 256)
            event = kernel(c_new, c, w, Δz_arr, Nx, Ny, Nz, FT(Δt), use_limiter; ndrange=(Nx, Ny, Nz))
            wait(device(arch), event)
            copyto!(c, c_new)
        end
        return nothing
    end

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
