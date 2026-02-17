# ---------------------------------------------------------------------------
# Adjoint of the slopes advection scheme (Russell & Lerner, 1981)
#
# Two modes, following TM5/NICAM-TM practice:
#
# 1. DISCRETE ADJOINT (use_limiter = false):
#    The forward operator without the flux limiter is fully linear:
#      c_new = L * c_old
#    The adjoint is the exact matrix transpose:
#      λ_old = L^T * λ_new
#    This gives machine-precision dot-product identity ⟨L^T λ, δc⟩ = ⟨λ, L δc⟩
#    but may produce oscillatory (negative) adjoint sensitivities.
#
# 2. CONTINUOUS ADJOINT (use_limiter = true):
#    The continuous adjoint of the conservation-form advection equation
#      ∂(ρq)/∂t + ∇·(ρvq) = 0
#    is (Niwa et al. 2017, Eq. 7–8):
#      ∂(ρq̃*)/∂t + ∇·(ρvq̃*) = 0
#    which has the same mathematical form as the forward equation. Therefore,
#    the adjoint step **reuses the forward advection code with negated wind
#    velocities**. The flux limiter is applied to the adjoint field, ensuring
#    monotonicity (non-oscillatory) sensitivities. The adjoint relationship is
#    imperfect but reasonable for 4D-Var optimization.
#
# The discrete adjoint approach is preferred when exact gradients are required
# (e.g., gradient testing). The continuous adjoint is preferred for production
# inversions where monotone sensitivities matter.
#
# References:
#   - Niwa et al. (2017), GMD 10, 1157–1174, doi:10.5194/gmd-10-1157-2017
#   - Meirink et al. (2008), ACP 8, 6341–6353, doi:10.5194/acp-8-6341-2008
#   - Thuburn & Haine (2001), J. Comput. Phys. 171, 616–631
# ---------------------------------------------------------------------------

using ..Grids: LatitudeLongitudeGrid, Δx, Δy, Δz

# ═══════════════════════════════════════════════════════════════════════════════
# Main adjoint interface — dispatches on use_limiter
# ═══════════════════════════════════════════════════════════════════════════════

"""
    adjoint_advect_x!(adj_tracers, velocities, grid::LatitudeLongitudeGrid,
                       scheme::SlopesAdvection, Δt)

Adjoint of `advect_x!` for the Russell-Lerner slopes scheme.

When `use_limiter = false`: exact discrete adjoint (matrix transpose, machine
precision). When `use_limiter = true`: continuous adjoint — reuses forward code
with negated wind following TM5/NICAM-TM (Niwa et al., 2017).
"""
function adjoint_advect_x!(adj_tracers::NamedTuple, velocities,
                            grid::LatitudeLongitudeGrid,
                            scheme::SlopesAdvection, Δt)
    if scheme.use_limiter
        neg_vel = (; u = .-velocities.u)
        advect_x!(adj_tracers, neg_vel, grid, scheme, Δt)
    else
        _discrete_adjoint_advect_x!(adj_tracers, velocities, grid, Δt)
    end
    return nothing
end

"""
    adjoint_advect_y!(adj_tracers, velocities, grid::LatitudeLongitudeGrid,
                       scheme::SlopesAdvection, Δt)

Adjoint of `advect_y!`. See `adjoint_advect_x!` for the two-mode description.
"""
function adjoint_advect_y!(adj_tracers::NamedTuple, velocities,
                            grid::LatitudeLongitudeGrid,
                            scheme::SlopesAdvection, Δt)
    if scheme.use_limiter
        neg_vel = (; v = .-velocities.v)
        advect_y!(adj_tracers, neg_vel, grid, scheme, Δt)
    else
        _discrete_adjoint_advect_y!(adj_tracers, velocities, grid, Δt)
    end
    return nothing
end

"""
    adjoint_advect_z!(adj_tracers, velocities, grid::LatitudeLongitudeGrid,
                       scheme::SlopesAdvection, Δt)

Adjoint of `advect_z!`. See `adjoint_advect_x!` for the two-mode description.
"""
function adjoint_advect_z!(adj_tracers::NamedTuple, velocities,
                            grid::LatitudeLongitudeGrid,
                            scheme::SlopesAdvection, Δt)
    if scheme.use_limiter
        neg_vel = (; w = .-velocities.w)
        advect_z!(adj_tracers, neg_vel, grid, scheme, Δt)
    else
        _discrete_adjoint_advect_z!(adj_tracers, velocities, grid, Δt)
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Exact discrete adjoint (transpose) — limiter OFF only
#
# When the limiter is off, the slope is s = (c_R - c_L)/2, which is linear
# with constant partial derivatives: ∂s/∂c_L = -1/2, ∂s/∂c_C = 0, ∂s/∂c_R = 1/2.
# The forward operator is c_new = L * c_old and we compute λ_old = L^T * λ_new
# using a scatter pattern.
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# _discrete_adjoint_advect_x! — periodic boundaries
# ---------------------------------------------------------------------------

function _discrete_adjoint_advect_x!(adj_tracers::NamedTuple, velocities,
                                      grid::LatitudeLongitudeGrid, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    for (name, λ) in pairs(adj_tracers)
        λ_old = zeros(eltype(λ), size(λ))

        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δx_i = Δx(i, j, grid)
                ip  = i == Nx ? 1 : i + 1
                im  = i == 1  ? Nx : i - 1
                ipp = ip == Nx ? 1 : ip + 1
                imm = im == 1  ? Nx : im - 1

                ur = u[i + 1, j, k]
                ul = u[i, j, k]
                fac = Δt / Δx_i
                λv = λ[i, j, k]

                # Identity contribution
                λ_old[i, j, k] += λv

                # ── Right flux (face i+1/2): scatter with −fac ──
                # Slope derivatives: ∂s/∂c_L = -1/2, ∂s/∂c_C = 0, ∂s/∂c_R = 1/2
                if ur > 0
                    α = 1 - ur * Δt / Δx_i
                    # flux = ur * (c[i] + α/2 * slope_i)
                    # slope_i = (c[ip] - c[im]) / 2
                    # ⟹ flux = ur * c[i] + ur*α/4 * c[ip] - ur*α/4 * c[im]
                    λ_old[im, j, k] += fac * ur * α / 4 * λv
                    λ_old[i, j, k]  -= fac * ur * λv
                    λ_old[ip, j, k] -= fac * ur * α / 4 * λv
                else
                    Δx_ip = Δx(ip, j, grid)
                    β = 1 + ur * Δt / Δx_ip
                    # flux = ur * (c[ip] - β/2 * slope_{ip})
                    # slope_{ip} = (c[ipp] - c[i]) / 2
                    # ⟹ flux = ur * c[ip] - ur*β/4 * c[ipp] + ur*β/4 * c[i]
                    λ_old[i, j, k]   -= fac * ur * β / 4 * λv
                    λ_old[ip, j, k]  -= fac * ur * λv
                    λ_old[ipp, j, k] += fac * ur * β / 4 * λv
                end

                # ── Left flux (face i−1/2): scatter with +fac ──
                if ul > 0
                    Δx_im = Δx(im, j, grid)
                    γ = 1 - ul * Δt / Δx_im
                    # flux = ul * (c[im] + γ/2 * slope_{im})
                    # slope_{im} = (c[i] - c[imm]) / 2
                    # ⟹ flux = ul * c[im] + ul*γ/4 * c[i] - ul*γ/4 * c[imm]
                    λ_old[imm, j, k] -= fac * ul * γ / 4 * λv
                    λ_old[im, j, k]  += fac * ul * λv
                    λ_old[i, j, k]   += fac * ul * γ / 4 * λv
                else
                    δc = 1 + ul * Δt / Δx_i
                    # flux = ul * (c[i] - δc/2 * slope_i)
                    # slope_i = (c[ip] - c[im]) / 2
                    # ⟹ flux = ul * c[i] - ul*δc/4 * c[ip] + ul*δc/4 * c[im]
                    λ_old[im, j, k] += fac * ul * δc / 4 * λv
                    λ_old[i, j, k]  += fac * ul * λv
                    λ_old[ip, j, k] -= fac * ul * δc / 4 * λv
                end
            end
        end
        copyto!(λ, λ_old)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# _discrete_adjoint_advect_y! — bounded boundaries
# ---------------------------------------------------------------------------

function _discrete_adjoint_advect_y!(adj_tracers::NamedTuple, velocities,
                                      grid::LatitudeLongitudeGrid, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    for (name, λ) in pairs(adj_tracers)
        T = eltype(λ)
        λ_old = zeros(T, size(λ))

        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δy_j = Δy(i, j, grid)
                fac = Δt / Δy_j
                λv = λ[i, j, k]
                vr = v[i, j + 1, k]
                vl = v[i, j, k]

                λ_old[i, j, k] += λv

                # ── Right flux (face j+1/2): scatter with −fac ──
                if j < Ny
                    if vr > 0
                        α = 1 - vr * Δt / Δy_j
                        has_slope = (j > 1 && j < Ny)
                        if has_slope
                            # slope = (c[j+1] - c[j-1])/2 → derivs: ∂s/∂c[j-1]=-1/2, ∂s/∂c[j+1]=1/2
                            λ_old[i, j-1, k] += fac * vr * α / 4 * λv
                        end
                        λ_old[i, j, k]   -= fac * vr * λv
                        if has_slope
                            λ_old[i, j+1, k] -= fac * vr * α / 4 * λv
                        end
                    else
                        Δy_jp = Δy(i, j + 1, grid)
                        β = 1 + vr * Δt / Δy_jp
                        has_slope_next = (j + 1 > 1 && j + 1 < Ny)
                        if has_slope_next
                            λ_old[i, j, k]   -= fac * vr * β / 4 * λv
                        end
                        λ_old[i, j+1, k] -= fac * vr * λv
                        if has_slope_next && j + 2 <= Ny
                            λ_old[i, j+2, k] += fac * vr * β / 4 * λv
                        end
                    end
                else  # j == Ny: boundary
                    if vr > 0
                        λ_old[i, j, k] -= fac * vr * λv
                    end
                end

                # ── Left flux (face j−1/2): scatter with +fac ──
                if j > 1
                    if vl > 0
                        Δy_jm = Δy(i, j - 1, grid)
                        γ = 1 - vl * Δt / Δy_jm
                        has_slope_prev = (j - 1 > 1 && j - 1 < Ny)
                        if has_slope_prev && j - 2 >= 1
                            λ_old[i, j-2, k] -= fac * vl * γ / 4 * λv
                        end
                        λ_old[i, j-1, k] += fac * vl * λv
                        if has_slope_prev
                            λ_old[i, j, k]   += fac * vl * γ / 4 * λv
                        end
                    else
                        δc = 1 + vl * Δt / Δy_j
                        has_slope = (j > 1 && j < Ny)
                        if has_slope
                            λ_old[i, j-1, k] += fac * vl * δc / 4 * λv
                        end
                        λ_old[i, j, k]   += fac * vl * λv
                        if has_slope && j + 1 <= Ny
                            λ_old[i, j+1, k] -= fac * vl * δc / 4 * λv
                        end
                    end
                else  # j == 1: boundary
                    if vl <= 0
                        λ_old[i, j, k] += fac * vl * λv
                    end
                end
            end
        end
        copyto!(λ, λ_old)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# _discrete_adjoint_advect_z! — bounded boundaries
# ---------------------------------------------------------------------------

function _discrete_adjoint_advect_z!(adj_tracers::NamedTuple, velocities,
                                      grid::LatitudeLongitudeGrid, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    for (name, λ) in pairs(adj_tracers)
        T = eltype(λ)
        λ_old = zeros(T, size(λ))

        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δz_k = Δz(k, grid)
                fac = Δt / Δz_k
                λv = λ[i, j, k]
                wt = w[i, j, k]       # top face (interface k)
                wb = w[i, j, k + 1]   # bottom face (interface k+1)

                λ_old[i, j, k] += λv

                # ── Bottom flux (face k+1): scatter with −fac ──
                if k < Nz
                    if wb > 0
                        α = 1 - wb * Δt / Δz_k
                        has_slope = (k > 1 && k < Nz)
                        if has_slope && k > 1
                            λ_old[i, j, k-1] += fac * wb * α / 4 * λv
                        end
                        λ_old[i, j, k]   -= fac * wb * λv
                        if has_slope
                            λ_old[i, j, k+1] -= fac * wb * α / 4 * λv
                        end
                    else
                        Δz_kp = Δz(k + 1, grid)
                        β = 1 + wb * Δt / Δz_kp
                        has_slope_next = (k + 1 > 1 && k + 1 < Nz)
                        if has_slope_next
                            λ_old[i, j, k]   -= fac * wb * β / 4 * λv
                        end
                        λ_old[i, j, k+1] -= fac * wb * λv
                        if has_slope_next && k + 2 <= Nz
                            λ_old[i, j, k+2] += fac * wb * β / 4 * λv
                        end
                    end
                else  # k == Nz: boundary
                    if wb > 0
                        λ_old[i, j, k] -= fac * wb * λv
                    end
                end

                # ── Top flux (face k): scatter with +fac ──
                if k > 1
                    if wt > 0
                        Δz_km = Δz(k - 1, grid)
                        γ = 1 - wt * Δt / Δz_km
                        has_slope_prev = (k - 1 > 1 && k - 1 < Nz)
                        if has_slope_prev && k - 2 >= 1
                            λ_old[i, j, k-2] -= fac * wt * γ / 4 * λv
                        end
                        λ_old[i, j, k-1] += fac * wt * λv
                        if has_slope_prev
                            λ_old[i, j, k]   += fac * wt * γ / 4 * λv
                        end
                    else
                        δc = 1 + wt * Δt / Δz_k
                        has_slope = (k > 1 && k < Nz)
                        if has_slope
                            λ_old[i, j, k-1] += fac * wt * δc / 4 * λv
                        end
                        λ_old[i, j, k]   += fac * wt * λv
                        if has_slope && k + 1 <= Nz
                            λ_old[i, j, k+1] -= fac * wt * δc / 4 * λv
                        end
                    end
                else  # k == 1: boundary
                    if wt <= 0
                        λ_old[i, j, k] += fac * wt * λv
                    end
                end
            end
        end
        copyto!(λ, λ_old)
    end
    return nothing
end
