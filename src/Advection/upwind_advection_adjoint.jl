# ---------------------------------------------------------------------------
# Discrete adjoint of first-order upwind advection
#
# Upwind is linear → adjoint is exact transpose.
# Given λ = gradient w.r.t. output (c_new), compute A'*λ = gradient w.r.t. input (c).
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Discrete adjoint of advect_x!. Overwrites adj_tracers with A' * adj_tracers.
"""
function adjoint_advect_x!(adj_tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, λ) in pairs(adj_tracers)
        FT = eltype(λ)
        λ_old = zeros(FT, size(λ))
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                Δx_ij = Δx(i, j, grid)
                i_prev = i == 1 ? Nx : i - 1
                i_next = i == Nx ? 1 : i + 1
                u_right = u[i + 1, j, k]
                u_left  = u[i, j, k]
                fac_i = Δt / Δx_ij
                λv = λ[i, j, k]

                # Diagonal: ∂c_new[i]/∂c[i]
                diag = 1 - fac_i * (u_right * (u_right > 0) - u_left * (u_left <= 0))
                λ_old[i, j, k] += diag * λv

                # From row i-1: c[i] appears in right flux of cell i-1
                # when u_left <= 0 (u at face i flows leftward into cell i-1)
                fac_prev = Δt / Δx(i_prev, j, grid)
                λ_old[i, j, k] += -fac_prev * (u_left <= 0 ? u_left : zero(FT)) * λ[i_prev, j, k]

                # From row i+1: c[i] appears in left flux of cell i+1
                # when u_right > 0 (u at face i+1 flows rightward into cell i+1)
                fac_next = Δt / Δx(i_next, j, grid)
                λ_old[i, j, k] += fac_next * (u_right > 0 ? u_right : zero(FT)) * λ[i_next, j, k]
            end
        end
        copyto!(λ, λ_old)
    end
    return nothing
end

"""
$(SIGNATURES)

Discrete adjoint of advect_y!. Overwrites adj_tracers with A' * adj_tracers.
Uses the spherical form matching the forward kernel (sin/cos weighting).
"""
function adjoint_advect_y!(adj_tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    FT = eltype(grid.φᶠ)
    R = grid.radius
    ε = FT(1e-30)
    φᶠ = Array(grid.φᶠ)

    for (name, λ) in pairs(adj_tracers)
        λ_old = zeros(FT, size(λ))

        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @inbounds begin
                sin_N = sind(φᶠ[j + 1])
                sin_S = sind(φᶠ[j])
                cos_N = cosd(φᶠ[j + 1])
                cos_S = cosd(φᶠ[j])
                ds_j  = max(abs(sin_N - sin_S), ε)
                fac_j = Δt / (R * ds_j)

                v_N = v[i, j + 1, k]
                v_S = v[i, j, k]
                λv = λ[i, j, k]

                # Diagonal: ∂c_new[j]/∂c[j]
                d_north = j < Ny ? (v_N > 0 ? v_N * cos_N : zero(FT)) :
                                   (v_N > 0 ? v_N * cos_N : zero(FT))
                d_south = j > 1  ? (v_S <= 0 ? v_S * cos_S : zero(FT)) :
                                   (v_S <= 0 ? v_S * cos_S : zero(FT))
                λ_old[i, j, k] += (1 - fac_j * (d_north - d_south)) * λv

                # Off-diagonal from row j-1: c[j] appears in north flux of cell j-1
                if j > 1
                    sin_Nm1 = sin_S
                    sin_Sm1 = sind(φᶠ[j - 1])
                    ds_jm1  = max(abs(sin_Nm1 - sin_Sm1), ε)
                    fac_jm1 = Δt / (R * ds_jm1)
                    λ_old[i, j, k] += -fac_jm1 * (v_S <= 0 ? v_S * cos_S : zero(FT)) * λ[i, j - 1, k]
                end

                # Off-diagonal from row j+1: c[j] appears in south flux of cell j+1
                if j < Ny
                    sin_Np1 = sind(φᶠ[j + 2])
                    sin_Sp1 = sin_N
                    ds_jp1  = max(abs(sin_Np1 - sin_Sp1), ε)
                    fac_jp1 = Δt / (R * ds_jp1)
                    λ_old[i, j, k] += fac_jp1 * (v_N > 0 ? v_N * cos_N : zero(FT)) * λ[i, j + 1, k]
                end
            end
        end
        copyto!(λ, λ_old)
    end
    return nothing
end

"""
$(SIGNATURES)

Discrete adjoint of advect_z!. Overwrites adj_tracers with A' * adj_tracers.
"""
function adjoint_advect_z!(adj_tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    w = velocities.w
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, λ) in pairs(adj_tracers)
        λ_new = similar(λ)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δz_k = Δz(k, grid)
            k_prev = k - 1
            k_next = k + 1
            w_top = w[i, j, k]
            w_bot = w[i, j, k + 1]
            fac = Δt / Δz_k
            diag = 1 - fac * (w_bot * (w_bot > 0) - w_top * (w_top <= 0))
            contrib = diag * λ[i, j, k]
            # A[k+1,k] uses Δz(k+1); A[k-1,k] uses Δz(k-1) and has minus sign
            if k < Nz
                contrib += (Δt / Δz(k_next, grid)) * w_bot * (w_bot > 0) * λ[i, j, k_next]
            end
            if k > 1
                contrib -= (Δt / Δz(k_prev, grid)) * w_top * (w_top <= 0) * λ[i, j, k_prev]
            end
            λ_new[i, j, k] = contrib
        end
        copyto!(λ, λ_new)
    end
    return nothing
end
