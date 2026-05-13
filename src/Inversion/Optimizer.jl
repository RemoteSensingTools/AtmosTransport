# ---------------------------------------------------------------------------
# Plan 26 P0.C1 — polymorphic optimizer dispatch for CS 4D-Var.
#
# Layout (designed so future backends — L-BFGS via Optim.jl in C2,
# hand-rolled L-BFGS-B, etc. — plug in as new concrete subtypes
# without touching the public entrypoint):
#
#   AbstractCSOptimizer
#       │
#       └── CSGradientDescent          (the shim shipped in P0.4b,
#                                       now wrapped as a concrete
#                                       subtype)
#
# Backends implement `cs_surface_flux_4dvar_solve(opt, cost_fn,
# controls)` where `cost_fn(controls) -> CS4DVarResult` is the
# closure built by `cs_surface_flux_4dvar_optimize`. The public
# entrypoint keeps its existing keyword surface for backward
# compatibility — when no explicit `optimizer` is passed, it
# constructs a `CSGradientDescent` from the legacy descent-policy
# kwargs.
# ---------------------------------------------------------------------------

"""
    AbstractCSOptimizer

Supertype for CS 4D-Var optimization backends. Concrete subtypes
implement `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)`
where `cost_fn(controls) -> CS4DVarResult` evaluates one
cost-and-gradient pass through the 4D-Var driver.

Shipped backends:

- [`CSGradientDescent`](@ref) — the dependency-free
  backtracking-line-search descent loop.

Planned (Phase C continuation):

- `CSLBFGS` — `Optim.jl` L-BFGS wrapper (Plan 26 commit C2).
"""
abstract type AbstractCSOptimizer end

"""
    CSGradientDescent(; iterations, initial_step, min_step,
                        step_shrink, gradient_tolerance, line_search)

Plain backtracking-line-search gradient descent. `iterations` is the
maximum count of accepted descent steps; the loop terminates early
when the gradient norm falls below `gradient_tolerance` or the
line search shrinks `step` below `min_step`. With
`line_search = false` every candidate step is accepted (constant
step size unless the loop exits on tolerance).
"""
struct CSGradientDescent{FT <: AbstractFloat} <: AbstractCSOptimizer
    iterations::Int
    initial_step::FT
    min_step::FT
    step_shrink::FT
    gradient_tolerance::FT
    line_search::Bool

    function CSGradientDescent(iterations::Integer,
                                initial_step::FT,
                                min_step::FT,
                                step_shrink::FT,
                                gradient_tolerance::FT,
                                line_search::Bool) where FT <: AbstractFloat
        iterations >= 0 || throw(ArgumentError(
            "CSGradientDescent iterations must be non-negative, got $iterations"))
        initial_step > 0 || throw(ArgumentError(
            "CSGradientDescent initial_step must be positive, got $initial_step"))
        min_step > 0 || throw(ArgumentError(
            "CSGradientDescent min_step must be positive, got $min_step"))
        0 < step_shrink < 1 || throw(ArgumentError(
            "CSGradientDescent step_shrink must be in (0, 1), got $step_shrink"))
        gradient_tolerance >= 0 || throw(ArgumentError(
            "CSGradientDescent gradient_tolerance must be non-negative, got " *
            "$gradient_tolerance"))
        return new{FT}(Int(iterations), initial_step, min_step, step_shrink,
                       gradient_tolerance, line_search)
    end
end

function CSGradientDescent(; iterations::Integer = 10,
                            initial_step::Real = 1.0,
                            min_step::Real = sqrt(eps(Float64)),
                            step_shrink::Real = 0.5,
                            gradient_tolerance::Real = 0.0,
                            line_search::Bool = true)
    FT = promote_type(typeof(float(initial_step)),
                      typeof(float(min_step)),
                      typeof(float(step_shrink)),
                      typeof(float(gradient_tolerance)))
    return CSGradientDescent(Int(iterations),
                             FT(initial_step), FT(min_step),
                             FT(step_shrink), FT(gradient_tolerance),
                             line_search)
end

# ---------------------------------------------------------------------------
# Polymorphic backend dispatch
# ---------------------------------------------------------------------------

"""
    cs_surface_flux_4dvar_solve(optimizer::AbstractCSOptimizer,
                                 cost_fn,
                                 controls) -> CS4DVarSolveResult

Run `optimizer` against the cost closure `cost_fn(controls) ->
CS4DVarResult` starting from `controls`. Dispatches on the
optimizer's concrete type — additional backends are added by
defining a new method here, not by branching inside an existing one.
"""
function cs_surface_flux_4dvar_solve end

function cs_surface_flux_4dvar_solve(optimizer::CSGradientDescent,
                                      cost_fn, controls)
    current = cost_fn(controls)
    # Derive history / step `FT` from the cost result, not from the
    # optimizer's parametric `FT`. A user passing `optimizer =
    # CSGradientDescent(initial_step = 0.25)` (defaults to Float64)
    # against a Float32 model would otherwise hit the
    # `CS4DVarSolveResult{FT, A2 <: AbstractArray{FT, 2}}` type bound
    # because A2's eltype is Float32 but the optimizer-FT-tagged
    # `CS4DVarSolveResult` would claim Float64. Policy scalars
    # (`initial_step`, `min_step`, `step_shrink`,
    # `gradient_tolerance`) are coerced once here.
    FT = eltype(current.gradients[1][1])
    initial_step = FT(optimizer.initial_step)
    min_step = FT(optimizer.min_step)
    step_shrink = FT(optimizer.step_shrink)
    gradient_tolerance = FT(optimizer.gradient_tolerance)

    cost_history = FT[FT(current.cost)]
    grad_norm = FT(_control_gradient_norm(current.gradients))
    gradient_norm_history = FT[grad_norm]
    step_history = FT[]

    for _ in 1:optimizer.iterations
        grad_norm <= gradient_tolerance && break
        step = initial_step
        accepted = false
        candidate = nothing
        candidate_controls = nothing
        while step >= min_step
            candidate_controls = _gradient_step_controls(
                current.controls, current.gradients, step)
            candidate = cost_fn(candidate_controls)
            if !optimizer.line_search || candidate.cost <= current.cost
                accepted = true
                break
            end
            step *= step_shrink
        end
        accepted || break
        current = candidate
        grad_norm = FT(_control_gradient_norm(current.gradients))
        push!(cost_history, FT(current.cost))
        push!(gradient_norm_history, grad_norm)
        push!(step_history, step)
    end

    A2 = typeof(current.gradients[1][1])
    return CS4DVarSolveResult{FT, A2}(
        current.controls,
        current,
        cost_history,
        gradient_norm_history,
        step_history,
        length(step_history))
end

# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

"""
    cs_surface_flux_4dvar_optimize(..., observations, controls;
                                    optimizer = nothing, kwargs...)
        -> CS4DVarSolveResult

Run an optimization loop around [`cs_surface_flux_4dvar`](@ref).

`optimizer::AbstractCSOptimizer` (kwarg) selects the backend. When
omitted, a [`CSGradientDescent`](@ref) is constructed from the
legacy descent-policy keyword arguments (`iterations`,
`initial_step`, `min_step`, `step_shrink`, `gradient_tolerance`,
`line_search`) so existing call sites keep working unchanged.

Remaining keyword arguments are forwarded to `cs_surface_flux_4dvar`
on every cost evaluation — including `preconditioner = ...` for the
P0.B3 preconditioned-cost path.
"""
function cs_surface_flux_4dvar_optimize(panels_rm0, panels_m0,
                                        panels_am_steps,
                                        panels_bm_steps,
                                        panels_cm_steps,
                                        mesh::CubedSphereMesh,
                                        observations,
                                        controls;
                                        optimizer::Union{Nothing,
                                                          AbstractCSOptimizer} = nothing,
                                        iterations::Integer = 10,
                                        initial_step = one(eltype(panels_rm0[1])),
                                        min_step = sqrt(eps(eltype(panels_rm0[1]))),
                                        step_shrink = 0.5,
                                        gradient_tolerance = zero(eltype(panels_rm0[1])),
                                        line_search::Bool = true,
                                        kwargs...)
    opt = if optimizer === nothing
        FT = eltype(panels_rm0[1])
        CSGradientDescent(Int(iterations), FT(initial_step), FT(min_step),
                          FT(step_shrink), FT(gradient_tolerance), line_search)
    else
        optimizer
    end

    cost_fn = function (active_controls)
        return cs_surface_flux_4dvar(
            panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
            mesh, observations, active_controls; kwargs...)
    end

    return cs_surface_flux_4dvar_solve(opt, cost_fn, controls)
end
