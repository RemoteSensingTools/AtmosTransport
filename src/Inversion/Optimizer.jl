# ---------------------------------------------------------------------------
# Prototype gradient-descent optimizer shim.
#
# `cs_surface_flux_4dvar_optimize` is the simple dependency-free
# gradient-descent loop around `cs_surface_flux_4dvar`. Plan 26 Phase C
# will replace this with an `Optim.jl` L-BFGS path; this shim is the
# fallback / baseline.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 675-750 unchanged
# in Plan 26 P0.4b; no semantic change.
# ---------------------------------------------------------------------------

"""
    cs_surface_flux_4dvar_optimize(..., observations, controls; kwargs...)
        -> CS4DVarSolveResult

Run a small dependency-free gradient-descent solve around
[`cs_surface_flux_4dvar`](@ref). The control arrays keep their existing
CPU/GPU backend; each trial control update is applied by a
KernelAbstractions kernel. Use `iterations`, `initial_step`, `min_step`,
`step_shrink`, `gradient_tolerance`, and `line_search` for the descent
policy. All remaining keyword arguments are passed to
`cs_surface_flux_4dvar`.
"""
function cs_surface_flux_4dvar_optimize(panels_rm0, panels_m0,
                                        panels_am_steps,
                                        panels_bm_steps,
                                        panels_cm_steps,
                                        mesh::CubedSphereMesh,
                                        observations,
                                        controls;
                                        iterations::Integer = 10,
                                        initial_step = one(eltype(panels_rm0[1])),
                                        min_step = sqrt(eps(eltype(panels_rm0[1]))),
                                        step_shrink = 0.5,
                                        gradient_tolerance = zero(eltype(panels_rm0[1])),
                                        line_search::Bool = true,
                                        kwargs...)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    initial_step > 0 || throw(ArgumentError("initial_step must be positive"))
    min_step > 0 || throw(ArgumentError("min_step must be positive"))
    0 < step_shrink < 1 || throw(ArgumentError("step_shrink must be in (0, 1)"))

    FT = eltype(panels_rm0[1])
    current = cs_surface_flux_4dvar(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, observations, controls; kwargs...)
    cost_history = FT[current.cost]
    grad_norm = FT(_control_gradient_norm(current.gradients))
    gradient_norm_history = FT[grad_norm]
    step_history = FT[]

    for _ in 1:iterations
        grad_norm <= FT(gradient_tolerance) && break
        step = FT(initial_step)
        accepted = false
        candidate = nothing
        candidate_controls = nothing
        while step >= FT(min_step)
            candidate_controls = _gradient_step_controls(
                current.controls, current.gradients, step)
            candidate = cs_surface_flux_4dvar(
                panels_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, observations, candidate_controls; kwargs...)
            if !line_search || candidate.cost <= current.cost
                accepted = true
                break
            end
            step *= FT(step_shrink)
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
