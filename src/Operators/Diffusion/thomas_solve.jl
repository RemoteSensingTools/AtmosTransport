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

# Adjoint note (forward-only in this commit)

The matrix transpose `T^T` is obtained by swapping *shifted* sub-
and super-diagonals:

    a_T[k] = c[k - 1]     # for k ≥ 2
    b_T[k] = b[k]
    c_T[k] = a[k + 1]     # for k ≤ Nz - 1

A future adjoint kernel calls this same `solve_tridiagonal!` after
building `(a_T, b_T, c_T)`. No structural change required. See
[src_legacy/Diffusion/boundary_layer_diffusion_adjoint.jl:74-84](src_legacy/Diffusion/boundary_layer_diffusion_adjoint.jl#L74-L84).
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

"""
    build_diffusion_coefficients(Kz_col, dz_col, dt) -> (a, b, c)

**Reference** Backward-Euler tridiagonal coefficient builder for
vertical diffusion on one column. Returns three Nz-length
`Vector{FT}`s representing `T = I - dt · D`, where `D` is the
discrete second-derivative operator with variable `Kz` at cell
centers and Neumann (zero-flux) boundary conditions.

Interface Kz is the arithmetic mean of adjacent cell-center values;
interface `dz` is the arithmetic mean of adjacent cell thicknesses.
Zero-flux BCs set `a[1] = 0` and `c[Nz] = 0`.

Formulas at interior k:

    dz_above = 0.5 × (dz[k-1] + dz[k])
    dz_below = 0.5 × (dz[k] + dz[k+1])
    Kz_above = 0.5 × (Kz[k-1] + Kz[k])
    Kz_below = 0.5 × (Kz[k] + Kz[k+1])
    D_above  = Kz_above / (dz[k] × dz_above)
    D_below  = Kz_below / (dz[k] × dz_below)
    a[k]     = -dt × D_above
    b[k]     =  1 + dt × (D_above + D_below)
    c[k]     = -dt × D_below

At the boundaries: `D_above = 0` at `k = 1`, `D_below = 0` at `k = Nz`.

# Role: reference vs. production

This function is the **reference** used in tests. The production
kernel [`_vertical_diffusion_kernel!`](@ref) inlines the same
formulas at each level (to avoid allocation and to read Kz through
`field_value`). Tests verify the kernel's output matches the output
of this reference on matched inputs.

If the formulas here are ever changed, the kernel must be kept in
lock-step.

# Adjoint note

As in [`solve_tridiagonal!`](@ref), the transpose of the resulting
tridiagonal is `a_T[k] = c[k-1]`, `b_T[k] = b[k]`, `c_T[k] = a[k+1]`.
The formulas above are symmetric in `Kz` / `dz` (the interface
values), so the adjoint operator built by a future plan differs only
by the shift, not by the coefficient content.
"""
function build_diffusion_coefficients(Kz_col::AbstractVector{FT},
                                      dz_col::AbstractVector{FT},
                                      dt::Real) where FT
    Nz = length(Kz_col)
    length(dz_col) == Nz || throw(DimensionMismatch(
        "Kz_col has length $Nz but dz_col has length $(length(dz_col))"))
    dt_ft = FT(dt)
    a = zeros(FT, Nz)
    b = Vector{FT}(undef, Nz)
    c = zeros(FT, Nz)
    @inbounds for k in 1:Nz
        dz_k = dz_col[k]
        D_above = zero(FT)
        D_below = zero(FT)
        if k > 1
            dz_above = (dz_col[k - 1] + dz_k) / FT(2)
            Kz_above = (Kz_col[k - 1] + Kz_col[k]) / FT(2)
            D_above  = Kz_above / (dz_k * dz_above)
        end
        if k < Nz
            dz_below = (dz_k + dz_col[k + 1]) / FT(2)
            Kz_below = (Kz_col[k] + Kz_col[k + 1]) / FT(2)
            D_below  = Kz_below / (dz_k * dz_below)
        end
        a[k] = (k > 1)  ? -dt_ft * D_above : zero(FT)
        b[k] = one(FT) + dt_ft * (D_above + D_below)
        c[k] = (k < Nz) ? -dt_ft * D_below : zero(FT)
    end
    return a, b, c
end
