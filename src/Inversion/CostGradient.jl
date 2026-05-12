# ---------------------------------------------------------------------------
# 4D-Var cost and gradient evaluation.
#
# Contains:
#
#   * Control-window validation (`_validate_control_windows`) and
#     conversion (`_surface_rates_from_controls`).
#   * Gradient accumulation (`_add_window_gradient!`) that walks a
#     `CSFootprintResult` and sums per-step contributions back into the
#     control's window.
#   * Diagonal background term (`_background_cost_and_gradient!`) and the
#     four kernels it dispatches to (scalar vs array sigma, value
#     vs zero background).
#   * Gradient-descent step helpers (`_gradient_step_control`,
#     `_gradient_step_controls`, `_control_gradient_norm`,
#     `_gradient_step_panel_kernel!`).
#   * Vector flatten helpers (`_observation_vector`, `_control_vector`).
#   * Step-sequence truncation helpers used when an objective lives
#     before the end of the full run (`_truncate_steps`,
#     `_truncate_convection_forcing`, `_truncate_stepwise_arg`).
#   * `cs_surface_flux_4dvar` — the main cost/gradient entry point.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 243-422 and 534-673
# unchanged in Plan 26 P0.4b; no semantic change.
# ---------------------------------------------------------------------------

function _validate_control_windows(controls, nsteps::Int)
    for control in controls
        for step in control.window.steps
            1 <= step <= nsteps || throw(ArgumentError(
                "surface-flux control $(control.window.name) references step $step, " *
                "but the run has $nsteps steps"))
        end
    end
    return nothing
end

function _surface_rates_from_controls(controls, nsteps::Int,
                                      mesh::CubedSphereMesh, prototype)
    rates = [_zero_surface_rates(mesh, prototype) for _ in 1:nsteps]
    @inbounds for control in controls
        window = control.window
        for (idx, step) in enumerate(window.steps)
            weight = eltype(control.value[1])(window.weights[idx])
            for p in 1:6
                backend = get_backend(rates[step][p])
                kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
                kernel!(rates[step][p], control.value[p], weight;
                        ndrange = size(rates[step][p]))
                synchronize(backend)
            end
        end
    end
    return rates
end

function _add_window_gradient!(gradient, result::CSFootprintResult,
                               window::CSSurfaceFluxWindow, scale;
                               ignore_future::Bool = false)
    nsteps = length(result.footprints)
    FT = eltype(gradient[1])
    @inbounds for (idx, step) in enumerate(window.steps)
        if !(1 <= step <= nsteps)
            if ignore_future && step > nsteps
                continue
            end
            throw(ArgumentError(
                "surface-flux window $(window.name) references step $step, " *
                "but the footprint has $nsteps steps"))
        end
        weight = FT(scale) * FT(window.weights[idx])
        for p in 1:6
            backend = get_backend(gradient[p])
            kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
            kernel!(gradient[p], result.footprints[step][p], weight;
                    ndrange = size(gradient[p]))
            synchronize(backend)
        end
    end
    return nothing
end

@kernel function _add_background_gradient_scalar_kernel!(grad, @Const(value),
                                                        @Const(background),
                                                        invvar)
    i, j = @index(Global, NTuple)
    @inbounds grad[i, j] += (value[i, j] - background[i, j]) * invvar
end

@kernel function _add_background_gradient_zero_scalar_kernel!(grad,
                                                             @Const(value),
                                                             invvar)
    i, j = @index(Global, NTuple)
    @inbounds grad[i, j] += value[i, j] * invvar
end

@kernel function _add_background_gradient_array_kernel!(grad, @Const(value),
                                                       @Const(background),
                                                       @Const(sigma))
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = sigma[i, j]
        grad[i, j] += (value[i, j] - background[i, j]) / (s * s)
    end
end

@kernel function _add_background_gradient_zero_array_kernel!(grad,
                                                            @Const(value),
                                                            @Const(sigma))
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = sigma[i, j]
        grad[i, j] += value[i, j] / (s * s)
    end
end

function _background_cost_and_gradient!(gradient, control::CSSurfaceFluxControl)
    control.sigma === nothing && return zero(eltype(control.value[1]))
    FT = eltype(control.value[1])
    cost = zero(FT)
    @inbounds for p in 1:6
        value = control.value[p]
        backend = get_backend(value)
        if control.sigma isa Real
            sigma = FT(control.sigma)
            invvar = inv(sigma * sigma)
            if control.background === nothing
                cost += FT(0.5) * invvar * sum(abs2, value)
                kernel! = _add_background_gradient_zero_scalar_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, invvar; ndrange = size(value))
            else
                background = control.background[p]
                cost += FT(0.5) * invvar * sum(abs2, value .- background)
                kernel! = _add_background_gradient_scalar_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, background, invvar; ndrange = size(value))
            end
        else
            sigma = control.sigma[p]
            if control.background === nothing
                cost += FT(0.5) * sum(abs2, value ./ sigma)
                kernel! = _add_background_gradient_zero_array_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, sigma; ndrange = size(value))
            else
                background = control.background[p]
                cost += FT(0.5) * sum(abs2, (value .- background) ./ sigma)
                kernel! = _add_background_gradient_array_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, background, sigma; ndrange = size(value))
            end
        end
        synchronize(backend)
    end
    return cost
end

@kernel function _gradient_step_panel_kernel!(dst, @Const(value), @Const(gradient),
                                              step)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = value[i, j] - step * gradient[i, j]
end

function _gradient_step_control(control::CSSurfaceFluxControl, gradient, step)
    FT = eltype(control.value[1])
    stepped = ntuple(p -> begin
        dst = similar(control.value[p])
        backend = get_backend(dst)
        kernel! = _gradient_step_panel_kernel!(backend, (16, 16))
        kernel!(dst, control.value[p], gradient[p], FT(step);
                ndrange = size(dst))
        synchronize(backend)
        dst
    end, 6)
    return CSSurfaceFluxControl(control.window, stepped;
                                background = control.background,
                                sigma = control.sigma)
end

function _gradient_step_controls(controls, gradients, step)
    length(controls) == length(gradients) || throw(DimensionMismatch(
        "controls length $(length(controls)) does not match gradients length $(length(gradients))"))
    return Any[_gradient_step_control(controls[i], gradients[i], step)
               for i in eachindex(controls)]
end

function _control_gradient_norm(gradients)
    isempty(gradients) && return 0.0
    FT = eltype(gradients[1][1])
    total = zero(FT)
    @inbounds for grad in gradients, p in 1:6
        total += sum(abs2, grad[p])
    end
    return sqrt(total)
end

function _observation_vector(obs::CSObservation)
    return CSObservation[obs]
end
function _observation_vector(observations)
    return CSObservation[observations...]
end

function _control_vector(control::CSSurfaceFluxControl)
    return CSSurfaceFluxControl[control]
end
function _control_vector(controls)
    return CSSurfaceFluxControl[controls...]
end

# ---------------------------------------------------------------------------
# Step-sequence truncation helpers + 4D-Var entry
# ---------------------------------------------------------------------------

@inline function _truncate_steps(steps, n::Int)
    return steps[1:n]
end

_truncate_convection_forcing(::Nothing, n::Int) = nothing
function _truncate_convection_forcing(convection_forcing::AbstractVector, n::Int)
    return convection_forcing[1:n]
end
_truncate_convection_forcing(convection_forcing, n::Int) = convection_forcing

_truncate_stepwise_arg(::Nothing, n::Int) = nothing
function _truncate_stepwise_arg(value::AbstractVector, n::Int)
    return value[1:n]
end
_truncate_stepwise_arg(value, n::Int) = value

"""
    cs_surface_flux_4dvar(..., observations, controls; kwargs...) -> CS4DVarResult

Evaluate the prototype CS surface-flux 4D-Var cost and gradient. Controls are
named `CSSurfaceFluxControl`s over `CSSurfaceFluxWindow`s. Observations are
scalar `CSObservation`s sampled after model steps. The observation-gradient
term is assembled from the same reverse-mode footprints used by
`cs_surface_emission_footprint`; optional diagonal background terms are added
per control.
"""
function cs_surface_flux_4dvar(panels_rm0, panels_m0,
                               panels_am_steps,
                               panels_bm_steps,
                               panels_cm_steps,
                               mesh::CubedSphereMesh,
                               observations,
                               controls;
                               scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                               dt = one(eltype(panels_rm0[1])),
                               flux_scale = one(eltype(panels_rm0[1])),
                               cfl_limit = 0.95,
                               diffusion_op = NoDiffusion(),
                               diffusion_workspace = nothing,
                               diffusion_meteo = nothing,
                               convection_op = NoConvection(),
                               convection_forcing = nothing,
                               convection_workspace = nothing,
                               tape_storage = :device)
    FT = eltype(panels_rm0[1])
    dt_ft = FT(dt)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    observation_vec = _observation_vector(observations)
    control_vec = _control_vector(controls)
    isempty(observation_vec) && throw(ArgumentError("at least one CSObservation is required"))
    isempty(control_vec) && throw(ArgumentError("at least one CSSurfaceFluxControl is required"))
    _validate_control_windows(control_vec, nsteps)
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    seen = Set{Symbol}()
    for control in control_vec
        name = control.window.name
        name in seen && throw(ArgumentError("duplicate surface-flux control name $name"))
        push!(seen, name)
    end

    emission_rates = _surface_rates_from_controls(control_vec, nsteps, mesh, panels_rm0[1])
    simulated = _run_cs_observations_forward(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, observation_vec;
        scheme = scheme,
        dt = dt_ft,
        cfl_limit = cfl_limit,
        emission_rates = emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace)

    residuals = Vector{FT}(undef, length(observation_vec))
    gradients = [_zero_surface_like(control.value) for control in control_vec]
    observation_cost = zero(FT)

    @inbounds for obs_idx in eachindex(observation_vec)
        obs = observation_vec[obs_idx]
        residual = simulated[obs_idx] - FT(obs.value)
        sigma = FT(obs.sigma)
        residuals[obs_idx] = residual
        observation_cost += FT(0.5) * (residual / sigma)^2
        scale = residual / (sigma * sigma)

        panels_am_obs = _truncate_steps(panels_am_steps, obs.step)
        panels_bm_obs = _truncate_steps(panels_bm_steps, obs.step)
        panels_cm_obs = _truncate_steps(panels_cm_steps, obs.step)
        diffusion_op_obs = _truncate_stepwise_arg(diffusion_op, obs.step)
        diffusion_workspace_obs = _truncate_stepwise_arg(diffusion_workspace, obs.step)
        base_emission_rates_obs = _truncate_stepwise_arg(emission_rates, obs.step)
        convection_forcing_obs = _truncate_convection_forcing(convection_forcing, obs.step)
        footprint = cs_surface_emission_footprint(
            panels_rm0, panels_m0,
            panels_am_obs, panels_bm_obs, panels_cm_obs,
            mesh, obs.objective;
            scheme = scheme,
            dt = dt_ft,
            flux_scale = FT(flux_scale),
            cfl_limit = cfl_limit,
            base_emission_rates = base_emission_rates_obs,
            diffusion_op = diffusion_op_obs,
            diffusion_workspace = diffusion_workspace_obs,
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = convection_forcing_obs,
            convection_workspace = convection_workspace,
            tape_storage = tape_storage)
        for control_idx in eachindex(control_vec)
            _add_window_gradient!(gradients[control_idx], footprint,
                                  control_vec[control_idx].window, scale;
                                  ignore_future = true)
        end
    end

    background_cost = zero(FT)
    @inbounds for idx in eachindex(control_vec)
        background_cost += FT(_background_cost_and_gradient!(gradients[idx], control_vec[idx]))
    end

    gradient_by_name = Dict{Symbol, typeof(gradients[1])}()
    @inbounds for idx in eachindex(control_vec)
        gradient_by_name[control_vec[idx].window.name] = gradients[idx]
    end
    A2 = typeof(gradients[1][1])
    return CS4DVarResult{FT, A2}(
        observation_cost + background_cost,
        observation_cost,
        background_cost,
        simulated,
        residuals,
        gradients,
        gradient_by_name,
        control_vec,
        observation_vec)
end
