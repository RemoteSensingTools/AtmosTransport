# ---------------------------------------------------------------------------
# Column kernels — one thread per column (i, j)
#
# Used for: vertical diffusion (Thomas solver), convection closures,
# pressure-thickness reconstruction, dry mass closure, and any operation
# that sweeps through the vertical.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    diagnose_cm_column!(cm, am, bm, bt, Nz)

Per-column vertical flux diagnosis from horizontal convergence.
Identical to _diagnose_cm_kernel! in Divergence.jl but callable
independently as a column kernel pattern.
"""
@kernel function _diagnose_cm_column_kernel!(cm, @Const(am), @Const(bm), @Const(bt), Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)
    @inbounds begin
        pit = zero(FT)
        for k in 1:Nz
            pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        end
        acc = zero(FT)
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            acc += conv_k - bt[k] * pit
            cm[i, j, k+1] = acc
        end
    end
end

"""
    thomas_solve_column!(x, a, b, c, d, Nz)

Thomas algorithm (tridiagonal solve) per column for implicit vertical diffusion.
a[k] = sub-diagonal, b[k] = diagonal, c[k] = super-diagonal, d[k] = RHS.
Solution written to x[k].
"""
@kernel function _thomas_solve_kernel!(x, @Const(a), b_copy, @Const(c), d_copy, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(x)
    @inbounds begin
        # Forward elimination
        for k in 2:Nz
            w = a[i, j, k] / b_copy[i, j, k-1]
            b_copy[i, j, k] -= w * c[i, j, k-1]
            d_copy[i, j, k] -= w * d_copy[i, j, k-1]
        end
        # Back substitution
        x[i, j, Nz] = d_copy[i, j, Nz] / b_copy[i, j, Nz]
        for k in (Nz-1):-1:1
            x[i, j, k] = (d_copy[i, j, k] - c[i, j, k] * x[i, j, k+1]) / b_copy[i, j, k]
        end
    end
end

export _diagnose_cm_column_kernel!, _thomas_solve_kernel!
