"""
    Diffusion

Vertical diffusion operators (plan 16b).

This submodule ships the solver infrastructure for implicit
(Backward-Euler) vertical diffusion. It does **not** yet include
the `ImplicitVerticalDiffusion` operator type — that lands in
Commit 3 — nor integration into `strang_split_mt!` (Commit 4) or
`TransportModel` (Commit 5).

Commit 2 exports:

- [`solve_tridiagonal!`](@ref) — generic per-column Thomas solve.
- [`build_diffusion_coefficients`](@ref) — reference Backward-Euler
  coefficient builder (pure, tested against closed-form corner cases).
- [`_vertical_diffusion_kernel!`](@ref) — the KA kernel the
  `ImplicitVerticalDiffusion.apply!` will launch. Inlines the
  same coefficient formulas as the reference.

The coefficient arithmetic is deliberately **not fused** into a
pre-factored `(w, inv_denom)` form: `(a, b, c)` are named locals
at every level k so a future adjoint kernel can transpose them
mechanically (see docstring in [`thomas_solve.jl`](@ref)).
"""
module Diffusion

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, AbstractTimeVaryingField, field_value, update_field!
using ...MetDrivers: current_time
import ..apply!

export solve_tridiagonal!, build_diffusion_coefficients
export _vertical_diffusion_kernel!
export AbstractDiffusionOperator, NoDiffusion, ImplicitVerticalDiffusion
export apply_vertical_diffusion!

include("thomas_solve.jl")
include("diffusion_kernels.jl")
include("operators.jl")

end # module Diffusion
