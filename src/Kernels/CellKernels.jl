# ---------------------------------------------------------------------------
# Cell kernels — one thread per (cell, level)
#
# Used for: source injection, tracer mass updates, diagnostics,
# boundary conditions, air mass computation.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    compute_air_mass!(m, ps, q, A_coeffs, B_coeffs, areas, g, Nz)

Compute dry air mass per cell from hybrid pressure coordinates.
m[i,j,k] = (A[k+1] + B[k+1]*ps[i,j] - A[k] - B[k]*ps[i,j]) * areas[j] / g * (1 - q[i,j,k])
"""
@kernel function _compute_air_mass_kernel!(m, @Const(ps), @Const(q),
                                            @Const(A), @Const(B),
                                            @Const(areas), g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        dp = (A[k+1] + B[k+1] * ps[i, j]) - (A[k] + B[k] * ps[i, j])
        m[i, j, k] = dp * areas[j] / g * (one(eltype(q)) - q[i, j, k])
    end
end

function compute_air_mass!(m::AbstractArray{FT,3}, ps::AbstractArray{FT,2},
                           q::AbstractArray{FT,3},
                           A::AbstractVector{FT}, B::AbstractVector{FT},
                           areas::AbstractVector{FT}, g::FT) where FT
    backend = get_backend(m)
    kernel! = _compute_air_mass_kernel!(backend, 256)
    kernel!(m, ps, q, A, B, areas, g; ndrange=size(m))
    synchronize(backend)
    return nothing
end

"""
    inject_source!(rm, source_rate, area, dt)

Add emissions to tracer mass: rm += source_rate × area × dt.
source_rate is in [kg / m² / s].
"""
@kernel function _inject_source_kernel!(rm, @Const(source_rate), @Const(areas), dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Nz = size(rm, 3)
        rm[i, j, Nz] += source_rate[i, j] * areas[j] * dt
    end
end

export compute_air_mass!, _inject_source_kernel!
