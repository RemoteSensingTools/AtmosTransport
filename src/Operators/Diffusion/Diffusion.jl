"""
    Diffusion

Vertical-diffusion operator hierarchy and solver infrastructure.

Public surface:

- [`NoDiffusion`](@ref) — identity no-op; the default when no
  `[diffusion]` config section is present.
- [`ImplicitVerticalDiffusion`](@ref) — Backward-Euler implicit
  diffusion driven by an `AbstractTimeVaryingField` Kz. Wired into
  the Strang palindrome via [`apply_vertical_diffusion!`](@ref) and
  installed into `TransportModel.diffusion` by the runtime recipe
  when `[diffusion] kind = "constant"`.

Both subtype the global `AbstractDiffusion` declared in
`src/Operators/AbstractOperators.jl`; concrete operator structs live in
`operators.jl`. The KA kernel (`_vertical_diffusion_kernel!`) and the
column-level Thomas solve (`solve_tridiagonal!`) are exposed for tests
and downstream variants. The coefficient arithmetic is deliberately
**not fused** into a pre-factored form: `(a, b, c)` are named locals
at every level k so a future adjoint kernel can transpose them
mechanically — see `thomas_solve.jl`.
"""
module Diffusion

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState,
                AbstractTimeVaryingField, AbstractCubedSphereField,
                PrecomputedCSDkgField,
                field_value, update_field!, panel_field, eachtracer
using ...MetDrivers: current_time
import ..apply!
import ..AbstractDiffusion                # global root from src/Operators/AbstractOperators.jl

export ColumnDiffusionWorkspace
export solve_tridiagonal!, build_diffusion_coefficients
export _vertical_diffusion_kernel!
export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion
export AbstractSurfaceFluxCoupling, SplitSurfaceFluxCoupling,
       DiffusiveSurfaceFluxBoundary, uses_diffusive_surface_flux_boundary
export apply_vertical_diffusion!, apply_vertical_diffusion_vmr!
export fill_dz_hydrostatic_constT!, fill_dz_hydrostatic_virtualT!

"""Scratch owned by vertical diffusion: two arrays per column layout, on the state backend."""
struct ColumnDiffusionWorkspace{A}
    w_scratch::A
    dz_scratch::A
end
ColumnDiffusionWorkspace(a::AbstractArray) = ColumnDiffusionWorkspace(similar(a), similar(a))
function ColumnDiffusionWorkspace(a::NTuple{6}, Nc::Integer)
    allocate() = map(p -> similar(p, eltype(p), Nc, Nc, size(p, 3)), a)
    return ColumnDiffusionWorkspace(allocate(), allocate())
end
Adapt.adapt_structure(to, ws::ColumnDiffusionWorkspace) =
    ColumnDiffusionWorkspace(Adapt.adapt(to, ws.w_scratch), Adapt.adapt(to, ws.dz_scratch))

include("thomas_solve.jl")
include("diffusion_kernels.jl")
include("dz_helpers.jl")
include("operators.jl")

end # module Diffusion
