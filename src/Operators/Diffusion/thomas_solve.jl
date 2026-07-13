"""
    solve_tridiagonal!(x, a, b, c, d, w)

Solve the tridiagonal linear system `T x = d` in place, where
`T` has sub-diagonal `a`, main diagonal `b`, super-diagonal `c`.
By convention `a[1]` and `c[Nz]` are ignored (no-neighbor positions).

Argument roles:
- `x::AbstractVector{FT}` — output, overwritten with the solution.
- `a, b, c, d::AbstractVector{FT}` — read only, not mutated.
- `w::AbstractVector{FT}` — caller-supplied workspace, length ≥ Nz.
  Holds the Thomas forward-elimination factors `w[k] = c[k] / denom`
  used during back-substitution.

Implements the standard Thomas algorithm with one extra array (`w`)
so that `b` and `d` can be read-only. For per-column Nz this is
Θ(Nz) arithmetic, no allocation.

# Adjoint note (this routine is forward-only)

The matrix transpose `T^T` is obtained by swapping *shifted* sub-
and super-diagonals:

    a_T[k] = c[k - 1]     # for k ≥ 2
    b_T[k] = b[k]
    c_T[k] = a[k + 1]     # for k ≤ Nz - 1

The CS adjoint kernel `_vertical_diffusion_cs_single_adjoint_kernel!`
in `src/Adjoints/DiffusionAdjoint.jl` builds `(a_T, b_T, c_T)`
inline and runs an inlined Thomas (rather than reusing this routine)
so the kernel stays allocation-free.
"""
function solve_tridiagonal!(x::AbstractVector{FT},
                            a::AbstractVector{FT},
                            b::AbstractVector{FT},
                            c::AbstractVector{FT},
                            d::AbstractVector{FT},
                            w::AbstractVector{FT}) where FT
    Nz = length(x)
    @boundscheck begin
        length(a) == Nz || throw(DimensionMismatch("a has length $(length(a)), need $Nz"))
        length(b) == Nz || throw(DimensionMismatch("b has length $(length(b)), need $Nz"))
        length(c) == Nz || throw(DimensionMismatch("c has length $(length(c)), need $Nz"))
        length(d) == Nz || throw(DimensionMismatch("d has length $(length(d)), need $Nz"))
        length(w) >= Nz || throw(DimensionMismatch("w has length $(length(w)), need ≥ $Nz"))
    end
    @inbounds begin
        # Forward elimination: w[k] = c[k] / denom; x[k] temporarily holds g[k]
        denom = b[1]
        w[1] = c[1] / denom
        x[1] = d[1] / denom
        for k in 2:Nz
            denom = b[k] - a[k] * w[k - 1]
            w[k]  = c[k] / denom
            x[k]  = (d[k] - a[k] * x[k - 1]) / denom
        end
        # Back substitution: x[k] = g[k] - w[k] * x[k+1]
        for k in (Nz - 1):-1:1
            x[k] = x[k] - w[k] * x[k + 1]
        end
    end
    return x
end
