# ---------------------------------------------------------------------------
# Convection operator hierarchy + NoConvection default.
#
# Plan 18 Commit 1 ships only the type hierarchy and the no-op. Concrete
# operators (`CMFMCConvection`, `TM5Convection`) land in Commits 3 and 4.
# ---------------------------------------------------------------------------

# `AbstractConvection` is the global root declared in
# `src/Operators/AbstractOperators.jl`. Concrete subtypes here:
#
# - `NoConvection` — identity, default.
# - `CMFMCConvection` — GCHP-style RAS / Grell-Freitas transport with
#   CMFMC + optional DTRAIN, mandatory CFL sub-cycling, well-mixed
#   sub-cloud layer. See plan 18 v5.1 §2.1-§2.8.
# - `TM5Convection` — TM5-style four-field matrix transport with
#   in-kernel LU solve. See plan 18 v5.1 §2.13.
#
# Every concrete subtype implements
#
#     apply!(state::CellState{B},
#            forcing::ConvectionForcing,
#            grid::AtmosGrid,
#            op,
#            dt::Real;
#            workspace) where {B <: AbstractMassBasis}
#
# mutating `state.tracers_raw` in place and returning `state`. The forcing
# arrives via `TransportModel.convection_forcing`, populated each substep
# by `DrivenSimulation._refresh_forcing!`. No `meteo` kwarg — unlike
# `ImplicitVerticalDiffusion` (which needs time to refresh Kz) or
# `SurfaceFluxOperator` (which needs time to sample `StepwiseField`
# emission rates), convection forcing IS the time information; the
# operator does not call `current_time`.

"""
    NoConvection()

Identity operator — `apply!` is a no-op. Default for configurations
without active convection. Dispatch is a compile-time dead branch in
`TransportModel.step!`, so the convection block collapses to zero
floating-point work when no operator is installed (bit-exact backward-
compatible with pre-plan-18 behavior).
"""
struct NoConvection <: AbstractConvection end

# =========================================================================
# apply!  (state-level, delegates to apply_convection! in Commits 3/4)
# =========================================================================

"""
    apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
           ::NoConvection, dt; workspace=nothing)

No-op. Accepts any `ConvectionForcing` (including the all-nothing
placeholder) and any workspace, including `nothing`. Returns `state`
unchanged.
"""
function apply!(state::CellState, forcing::ConvectionForcing, grid::AtmosGrid,
                ::NoConvection, dt::Real;
                workspace = nothing)
    return state
end

# =========================================================================
# apply_convection!  (array-level; Commits 3/4 add CMFMC/TM5 methods)
# =========================================================================

"""
    apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                       ::NoConvection, dt, workspace, grid) -> nothing

Array-level no-op, parallels the plan 16b / 17 pattern
(`apply_vertical_diffusion!`, `apply_surface_flux!`). Accepts any
shape of `q_raw` / `air_mass` — `NoConvection` doesn't inspect them.
Returns `nothing`.

The plan-18 structured `apply!` flow goes through the state-level
method above. `apply_convection!` is reserved for the future case
where the convection block is called from inside a palindrome or
another composed setting — same signature contract as plan 16b and 17.
"""
apply_convection!(q_raw, air_mass, forcing::ConvectionForcing,
                   ::NoConvection, dt, workspace, grid) = nothing
