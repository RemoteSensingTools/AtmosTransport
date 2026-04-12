# ---------------------------------------------------------------------------
# SlopesAdvection on CubedSphereGrid
#
# Each panel is advected as a regular 2D grid. Panel boundaries use halo
# data filled by fill_panel_halos!() before each 1D sweep.
#
# Data layout:
#   tracers: NamedTuple of NTuple{6, Array{FT, 3}}
#     Each panel is (Nc + 2*Hp) × (Nc + 2*Hp) × Nz
#     Interior at [Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
#   velocities: NamedTuple (u, v, w) of NTuple{6, Array{FT, 3}}
#     u per panel: (Nc + 1, Nc, Nz) — x-face velocities in local coords
#     v per panel: (Nc, Nc + 1, Nz) — y-face velocities in local coords
#     w per panel: (Nc, Nc, Nz + 1) — z-interface velocities
#
# References:
#   Russell & Lerner (1981) — slopes advection scheme
#   Putman & Lin (2007) — cubed-sphere advection (FV3, different scheme)
# ---------------------------------------------------------------------------

using ..Grids: CubedSphereGrid, fill_panel_halos!, allocate_cubed_sphere_field

"""
$(SIGNATURES)

Russell-Lerner slopes advection in x (panel-local) on `CubedSphereGrid`.
Fills panel halos, then advects each panel independently.
"""
function advect_x!(tracers, velocities, grid::CubedSphereGrid,
                   scheme::SlopesAdvection, Δt)
    Nc = grid.Nc
    Hp = grid.Hp
    use_limiter = scheme.use_limiter

    for (name, c_panels) in pairs(tracers)
        fill_panel_halos!(c_panels, grid)

        for p in 1:6
            c = c_panels[p]
            u = velocities.u[p]
            Nz = size(c, 3)
            c_new = copy(c)

            @inbounds for k in 1:Nz, j in 1:Nc
                jj = Hp + j  # array index
                for i in 1:Nc
                    ii = Hp + i  # array index

                    dx_ij = Δx(i, j, grid; panel=p)

                    # Neighbor concentrations (reach into halo if needed)
                    c_im1 = c[ii - 1, jj, k]
                    c_ip1 = c[ii + 1, jj, k]
                    c_i   = c[ii, jj, k]

                    # Slope at cell i
                    s_i = (c_ip1 - c_im1) / 2
                    if use_limiter
                        s_i = minmod(s_i,
                            2 * (c_ip1 - c_i),
                            2 * (c_i - c_im1))
                    end

                    # Slope at cell i+1
                    c_ip2 = c[ii + 2, jj, k]
                    s_ip1 = (c_ip2 - c_i) / 2
                    if use_limiter
                        s_ip1 = minmod(s_ip1,
                            2 * (c_ip2 - c_ip1),
                            2 * (c_ip1 - c_i))
                    end

                    # Slope at cell i-1
                    c_im2 = c[ii - 2, jj, k]
                    s_im1 = (c_i - c_im2) / 2
                    if use_limiter
                        s_im1 = minmod(s_im1,
                            2 * (c_i - c_im1),
                            2 * (c_im1 - c_im2))
                    end

                    dx_ip1 = Δx(min(i + 1, Nc), j, grid; panel=p)
                    dx_im1 = Δx(max(i - 1, 1), j, grid; panel=p)

                    u_right = u[i + 1, j, k]
                    u_left  = u[i, j, k]

                    flux_right = if u_right > 0
                        u_right * (c_i + (1 - u_right * Δt / dx_ij) * s_i / 2)
                    else
                        u_right * (c_ip1 - (1 + u_right * Δt / dx_ip1) * s_ip1 / 2)
                    end

                    flux_left = if u_left > 0
                        u_left * (c_im1 + (1 - u_left * Δt / dx_im1) * s_im1 / 2)
                    else
                        u_left * (c_i - (1 + u_left * Δt / dx_ij) * s_i / 2)
                    end

                    c_new[ii, jj, k] = c_i - Δt / dx_ij * (flux_right - flux_left)
                end
            end

            # Copy updated interior back
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                c[Hp + i, Hp + j, k] = c_new[Hp + i, Hp + j, k]
            end
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Russell-Lerner slopes advection in y (panel-local) on `CubedSphereGrid`.
"""
function advect_y!(tracers, velocities, grid::CubedSphereGrid,
                   scheme::SlopesAdvection, Δt)
    Nc = grid.Nc
    Hp = grid.Hp
    use_limiter = scheme.use_limiter

    for (name, c_panels) in pairs(tracers)
        fill_panel_halos!(c_panels, grid)

        for p in 1:6
            c = c_panels[p]
            v = velocities.v[p]
            Nz = size(c, 3)
            c_new = copy(c)

            @inbounds for k in 1:Nz, i in 1:Nc
                ii = Hp + i
                for j in 1:Nc
                    jj = Hp + j

                    dy_ij = Δy(i, j, grid; panel=p)

                    c_jm1 = c[ii, jj - 1, k]
                    c_jp1 = c[ii, jj + 1, k]
                    c_j   = c[ii, jj, k]

                    s_j = (c_jp1 - c_jm1) / 2
                    if use_limiter
                        s_j = minmod(s_j,
                            2 * (c_jp1 - c_j),
                            2 * (c_j - c_jm1))
                    end

                    c_jp2 = c[ii, jj + 2, k]
                    s_jp1 = (c_jp2 - c_j) / 2
                    if use_limiter
                        s_jp1 = minmod(s_jp1,
                            2 * (c_jp2 - c_jp1),
                            2 * (c_jp1 - c_j))
                    end

                    c_jm2 = c[ii, jj - 2, k]
                    s_jm1 = (c_j - c_jm2) / 2
                    if use_limiter
                        s_jm1 = minmod(s_jm1,
                            2 * (c_j - c_jm1),
                            2 * (c_jm1 - c_jm2))
                    end

                    dy_jp1 = Δy(i, min(j + 1, Nc), grid; panel=p)
                    dy_jm1 = Δy(i, max(j - 1, 1), grid; panel=p)

                    v_top  = v[i, j + 1, k]
                    v_bot  = v[i, j, k]

                    flux_top = if v_top > 0
                        v_top * (c_j + (1 - v_top * Δt / dy_ij) * s_j / 2)
                    else
                        v_top * (c_jp1 - (1 + v_top * Δt / dy_jp1) * s_jp1 / 2)
                    end

                    flux_bot = if v_bot > 0
                        v_bot * (c_jm1 + (1 - v_bot * Δt / dy_jm1) * s_jm1 / 2)
                    else
                        v_bot * (c_j - (1 + v_bot * Δt / dy_ij) * s_j / 2)
                    end

                    c_new[ii, jj, k] = c_j - Δt / dy_ij * (flux_top - flux_bot)
                end
            end

            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                c[Hp + i, Hp + j, k] = c_new[Hp + i, Hp + j, k]
            end
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Russell-Lerner slopes advection in z (vertical) on `CubedSphereGrid`.
No panel halo exchange needed — vertical is independent.
"""
function advect_z!(tracers, velocities, grid::CubedSphereGrid,
                   scheme::SlopesAdvection, Δt)
    Nc = grid.Nc
    Hp = grid.Hp
    use_limiter = scheme.use_limiter

    for (name, c_panels) in pairs(tracers)
        for p in 1:6
            c = c_panels[p]
            w = velocities.w[p]
            Nz = size(c, 3)
            c_new = copy(c)

            @inbounds for j in 1:Nc, i in 1:Nc
                ii, jj = Hp + i, Hp + j
                for k in 1:Nz
                    dz_k = Δz(k, grid)
                    c_k = c[ii, jj, k]

                    # Slopes
                    s_k = if k > 1 && k < Nz
                        s = (c[ii, jj, k + 1] - c[ii, jj, k - 1]) / 2
                        use_limiter ? minmod(s,
                            2 * (c[ii, jj, k + 1] - c_k),
                            2 * (c_k - c[ii, jj, k - 1])) : s
                    else
                        zero(eltype(c))
                    end

                    s_kp1 = if k < Nz - 1
                        s = (c[ii, jj, k + 2] - c_k) / 2
                        use_limiter ? minmod(s,
                            2 * (c[ii, jj, k + 2] - c[ii, jj, k + 1]),
                            2 * (c[ii, jj, k + 1] - c_k)) : s
                    elseif k < Nz
                        zero(eltype(c))
                    else
                        zero(eltype(c))
                    end

                    s_km1 = if k > 2
                        s = (c_k - c[ii, jj, k - 2]) / 2
                        use_limiter ? minmod(s,
                            2 * (c_k - c[ii, jj, k - 1]),
                            2 * (c[ii, jj, k - 1] - c[ii, jj, k - 2])) : s
                    elseif k > 1
                        zero(eltype(c))
                    else
                        zero(eltype(c))
                    end

                    dz_kp1 = k < Nz ? Δz(k + 1, grid) : dz_k
                    dz_km1 = k > 1 ? Δz(k - 1, grid) : dz_k

                    w_top = w[i, j, k]
                    w_bot = w[i, j, k + 1]

                    flux_top = if k > 1
                        if w_top > 0
                            w_top * (c[ii, jj, k - 1] + (1 - w_top * Δt / dz_km1) * s_km1 / 2)
                        else
                            w_top * (c_k - (1 + w_top * Δt / dz_k) * s_k / 2)
                        end
                    else
                        w_top <= 0 ? w_top * c_k : zero(eltype(c))
                    end

                    flux_bot = if k < Nz
                        if w_bot > 0
                            w_bot * (c_k + (1 - w_bot * Δt / dz_k) * s_k / 2)
                        else
                            w_bot * (c[ii, jj, k + 1] - (1 + w_bot * Δt / dz_kp1) * s_kp1 / 2)
                        end
                    else
                        w_bot > 0 ? w_bot * c_k : zero(eltype(c))
                    end

                    c_new[ii, jj, k] = c_k - Δt / dz_k * (flux_bot - flux_top)
                end
            end

            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                c[Hp + i, Hp + j, k] = c_new[Hp + i, Hp + j, k]
            end
        end
    end
    return nothing
end
