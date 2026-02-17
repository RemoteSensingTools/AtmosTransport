# ---------------------------------------------------------------------------
# Boundary-layer vertical diffusion — forward
#
# Implicit solve: (I - Δt * D) * c^{n+1} = c^n
# where D is a tridiagonal diffusion operator built from Kz profiles.
# Solved column-by-column with the Thomas algorithm.
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid

"""
    tracer_data(t)

Return the modifiable 3D array for a tracer (Field or raw array).
"""
tracer_data(t) = t isa AbstractField ? interior(t) : t

"""
    default_Kz_interface(k, Nz, Kz_max, FT)

Default exponential Kz profile at interface between level k and k+1.
Decreases with height (smaller at top, larger near surface).
Clamped to [0, Kz_max].
"""
function default_Kz_interface(k, Nz, Kz_max, ::Type{FT}) where {FT}
    scale_height = FT(Nz)
    Kz = Kz_max * exp(-(Nz - k - FT(0.5)) / scale_height)
    return clamp(Kz, zero(FT), Kz_max)
end

"""
    thomas_solve!(a, b, c, d, x, w, g, N)

Solve tridiagonal system A*x = d in-place using Thomas algorithm.
a = sub-diagonal, b = main diagonal, c = super-diagonal.
a[1] and c[N] are not used (boundary).
"""
function thomas_solve!(a, b, c, d, x, w, g, N)
    @inbounds begin
        w[1] = c[1] / b[1]
        g[1] = d[1] / b[1]
        for k in 2:(N - 1)
            denom = b[k] - a[k] * w[k - 1]
            w[k] = c[k] / denom
            g[k] = (d[k] - a[k] * g[k - 1]) / denom
        end
        denom = b[N] - a[N] * w[N - 1]
        g[N] = (d[N] - a[N] * g[N - 1]) / denom
        x[N] = g[N]
        for k in (N - 1):-1:1
            x[k] = g[k] - w[k] * x[k + 1]
        end
    end
end

function diffuse!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid, diff::BoundaryLayerDiffusion, Δt)
    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    Kz_max = diff.Kz_max

    # Preallocate workspace for Thomas algorithm (allocation-free per column)
    a = Vector{FT}(undef, Nz)
    b = Vector{FT}(undef, Nz)
    c = Vector{FT}(undef, Nz)
    w = Vector{FT}(undef, Nz)
    g = Vector{FT}(undef, Nz)
    col = Vector{FT}(undef, Nz)

    for tracer in values(tracers)
        arr = tracer_data(tracer)
        for j in 1:Ny, i in 1:Nx
            # Extract column c_old
            @inbounds for k in 1:Nz
                col[k] = arr[i, j, k]
            end

            # Build tridiagonal coefficients for (I - Δt*D)*c_new = c_old
            # D[k,k-1] = Kz[k-1/2] / (Δz[k] * Δz_interface[k-1/2])
            # D[k,k+1] = Kz[k+1/2] / (Δz[k] * Δz_interface[k+1/2])
            # D[k,k] = -(D[k,k-1] + D[k,k+1])
            # No-flux at top (k=1) and bottom (k=Nz)
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

            thomas_solve!(a, b, c, col, col, w, g, Nz)

            # Write back
            @inbounds for k in 1:Nz
                arr[i, j, k] = col[k]
            end
        end
    end
    return nothing
end
