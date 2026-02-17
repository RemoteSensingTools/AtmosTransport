# ---------------------------------------------------------------------------
# Boundary-layer vertical diffusion — discrete adjoint
#
# Forward solve:  A · c_new = c_old   where  A = (I - Δt·D)
# Adjoint solve:  A^T · λ_old = λ_new
#
# For NON-UNIFORM vertical grids (hybrid sigma-pressure), the diffusion
# matrix D is NOT symmetric because the off-diagonal entries depend on
# the layer thickness Δz_k of the ROW (not the column):
#
#   A[k, k-1] = -Δt · Kz_{k-1/2} / (Δz_k · Δz_int_{k-1/2})     (= a[k])
#   A[k, k+1] = -Δt · Kz_{k+1/2} / (Δz_k · Δz_int_{k+1/2})     (= c[k])
#
# The transpose A^T has:
#   A^T[k, k-1] = A[k-1, k] = c[k-1]    (original super-diagonal, shifted)
#   A^T[k, k]   = A[k, k]   = b[k]
#   A^T[k, k+1] = A[k+1, k] = a[k+1]    (original sub-diagonal, shifted)
#
# Following TM5/NICAM-TM (Niwa et al., 2017), the discrete adjoint is
# used for diffusion.
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid

function adjoint_diffuse!(adj_tracers::NamedTuple, met, grid::LatitudeLongitudeGrid, diff::BoundaryLayerDiffusion, Δt)
    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    Kz_max = diff.Kz_max

    # Original tridiagonal arrays
    a = Vector{FT}(undef, Nz)
    b = Vector{FT}(undef, Nz)
    c = Vector{FT}(undef, Nz)
    # Transposed tridiagonal arrays
    a_T = Vector{FT}(undef, Nz)
    c_T = Vector{FT}(undef, Nz)
    # Thomas algorithm workspace
    w = Vector{FT}(undef, Nz)
    g = Vector{FT}(undef, Nz)
    col = Vector{FT}(undef, Nz)

    for adj_tracer in values(adj_tracers)
        arr = tracer_data(adj_tracer)
        for j in 1:Ny, i in 1:Nx
            @inbounds for k in 1:Nz
                col[k] = arr[i, j, k]
            end

            # Build original tridiagonal coefficients (same as forward)
            @inbounds for k in 1:Nz
                Δz_k = Δz(k, grid)
                D_below = zero(FT)
                D_above = zero(FT)
                if k > 1
                    Δz_int_below = (Δz(k - 1, grid) + Δz_k) / 2
                    Kz_below = default_Kz_interface(k - 1, Nz, Kz_max, FT)
                    D_below = Kz_below / (Δz_k * Δz_int_below)
                end
                if k < Nz
                    Δz_int_above = (Δz_k + Δz(k + 1, grid)) / 2
                    Kz_above = default_Kz_interface(k, Nz, Kz_max, FT)
                    D_above = Kz_above / (Δz_k * Δz_int_above)
                end
                D_kk = -(D_below + D_above)

                a[k] = k > 1 ? -Δt * D_below : zero(FT)
                b[k] = one(FT) - Δt * D_kk
                c[k] = k < Nz ? -Δt * D_above : zero(FT)
            end

            # Build transposed tridiagonal: A^T[k,k-1] = c[k-1], A^T[k,k+1] = a[k+1]
            @inbounds begin
                a_T[1] = zero(FT)
                for k in 2:Nz
                    a_T[k] = c[k - 1]
                end
                for k in 1:Nz - 1
                    c_T[k] = a[k + 1]
                end
                c_T[Nz] = zero(FT)
            end

            # Solve A^T · λ_old = λ_new
            thomas_solve!(a_T, b, c_T, col, col, w, g, Nz)

            @inbounds for k in 1:Nz
                arr[i, j, k] = col[k]
            end
        end
    end
    return nothing
end
