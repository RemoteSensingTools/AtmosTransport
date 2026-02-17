# ---------------------------------------------------------------------------
# Operator-splitting time stepper (forward)
#
# Symmetric Strang splitting following TM5:
#   1. advect_x  (Δt/2)
#   2. advect_y  (Δt/2)
#   3. advect_z  (Δt/2)
#   4. convect   (Δt)
#   5. diffuse   (Δt)
#   6. sources   (Δt)
#   7. advect_z  (Δt/2)
#   8. advect_y  (Δt/2)
#   9. advect_x  (Δt/2)
# ---------------------------------------------------------------------------

using ..Advection: AbstractAdvectionScheme, advect_x!, advect_y!, advect_z!
using ..Convection: AbstractConvection, convect!
using ..Diffusion: AbstractDiffusion, diffuse!
using ..Chemistry: AbstractChemistry, apply_chemistry!

"""
    AbstractTimeStepper

Supertype for time-stepping strategies.
"""
abstract type AbstractTimeStepper end

"""
    OperatorSplittingTimeStepper{FT, A, C, D, Ch} <: AbstractTimeStepper

TM5-style symmetric Strang splitting.

# Fields
- `advection  :: A`  — advection scheme
- `convection :: C`  — convection parameterization
- `diffusion  :: D`  — vertical diffusion parameterization
- `chemistry  :: Ch` — chemistry scheme (NoChemistry for inert tracers)
- `Δt_outer   :: FT` — outer time step [seconds] (e.g. 10800 for 3 hours)
"""
struct OperatorSplittingTimeStepper{FT, A, C, D, Ch} <: AbstractTimeStepper
    advection  :: A
    convection :: C
    diffusion  :: D
    chemistry  :: Ch
    Δt_outer   :: FT
end

function OperatorSplittingTimeStepper(;
        advection  :: AbstractAdvectionScheme,
        convection :: AbstractConvection,
        diffusion  :: AbstractDiffusion,
        chemistry  :: AbstractChemistry = Chemistry.NoChemistry(),
        Δt_outer   :: Real = 10800.0)
    FT = typeof(Δt_outer)
    return OperatorSplittingTimeStepper{FT, typeof(advection), typeof(convection),
                                        typeof(diffusion), typeof(chemistry)}(
        advection, convection, diffusion, chemistry, Δt_outer)
end

"""
    _extract_velocities(met)

Extract the velocity NamedTuple `(; u, v, w)` from a met data object.
Works with any object that has `.u`, `.v`, `.w` properties (NamedTuple,
struct, or the result of `prepare_met_for_physics`).
"""
@inline _extract_velocities(met) = (; u = met.u, v = met.v, w = met.w)

"""
    time_step!(model, Δt)

Perform one forward time step using symmetric Strang operator splitting.
`model` must have fields: `tracers`, `met_data`, `grid`, `timestepper`, `clock`.

`model.met_data` should be a NamedTuple (or struct) with at least:
- `u`, `v`, `w` — staggered velocity arrays for advection
- optionally `conv_mass_flux` for convection
- optionally `diffusivity` for diffusion

Use `prepare_met_for_physics(met_source, grid)` to create this from a
`MetDataSource` object.
"""
function time_step!(model, Δt)
    ts  = model.timestepper
    tr  = model.tracers
    met = model.met_data
    g   = model.grid
    half = Δt / 2

    # Extract staggered velocities from met data
    vel = _extract_velocities(met)

    # Forward symmetric splitting
    advect_x!(tr, vel, g, ts.advection, half)
    advect_y!(tr, vel, g, ts.advection, half)
    advect_z!(tr, vel, g, ts.advection, half)

    convect!(tr, met, g, ts.convection, Δt)
    diffuse!(tr, met, g, ts.diffusion, Δt)
    apply_chemistry!(tr, g, ts.chemistry, Δt)

    advect_z!(tr, vel, g, ts.advection, half)
    advect_y!(tr, vel, g, ts.advection, half)
    advect_x!(tr, vel, g, ts.advection, half)

    tick!(model.clock, Δt)
    return nothing
end
