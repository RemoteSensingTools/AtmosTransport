# ---------------------------------------------------------------------------
# Discrete adjoint of first-order upwind advection
#
# Upwind is linear → adjoint is exact transpose.
# Given λ = gradient w.r.t. output (c_new), compute A'*λ = gradient w.r.t. input (c).
# ---------------------------------------------------------------------------

"""
    adjoint_advect_x!(adj_tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

Discrete adjoint of advect_x!. Overwrites adj_tracers with A' * adj_tracers.
"""
function adjoint_advect_x!(adj_tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    u = velocities.u
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, λ) in pairs(adj_tracers)
        λ_new = similar(λ)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δx_ij = Δx(i, j, grid)
            i_prev = i == 1 ? Nx : i - 1
            i_next = i == Nx ? 1 : i + 1
            Δx_next = Δx(i_next, j, grid)
            Δx_prev = Δx(i_prev, j, grid)
            u_right = u[i + 1, j, k]
            u_left = u[i, j, k]
            fac = Δt / Δx_ij
            # A[i,i]*λ[i] + A[i,i+1]*λ[i+1] + A[i,i-1]*λ[i-1]
            # Off-diagonals use fac from the neighbor cell
            diag = 1 - fac * (u_right * (u_right > 0) - u_left * (u_left <= 0))
            λ_new[i, j, k] = diag * λ[i, j, k] +
                (Δt / Δx_next) * u_right * (u_right <= 0) * λ[i_next, j, k] +
                (Δt / Δx_prev) * u_left * (u_left > 0) * λ[i_prev, j, k]
        end
        copyto!(λ, λ_new)
    end
    return nothing
end

"""
    adjoint_advect_y!(adj_tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

Discrete adjoint of advect_y!. Overwrites adj_tracers with A' * adj_tracers.
"""
function adjoint_advect_y!(adj_tracers::NamedTuple, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)
    v = velocities.v
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    for (name, λ) in pairs(adj_tracers)
        λ_new = similar(λ)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Δy_ij = Δy(i, j, grid)
            j_prev = j - 1
            j_next = j + 1
            v_right = v[i, j + 1, k]
            v_left = v[i, j, k]
            fac = Δt / Δy_ij
            diag = 1 - fac * (v_right * (v_right > 0) - v_left * (v_left <= 0))
            contrib = diag * λ[i, j, k]
            if j < Ny
                contrib += fac * v_right * (v_right <= 0) * λ[i, j_next, k]
            end
            if j > 1
                contrib += fac * v_left * (v_left > 0) * λ[i, j_prev, k]
            end
            λ_new[i, j, k] = contrib
        end
        copyto!(λ, λ_new)
    end
    return nothing
end

"""
    adjoint_advect_z!(adj_tracers, velocities, grid::LatitudeLongitudeGrid, scheme::UpwindAdvection, Δt)

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
