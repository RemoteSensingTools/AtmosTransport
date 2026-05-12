"""
    Adjoints

Prototype adjoint and footprint utilities.

The shipped path here is a kernelized CS split-sweep reverse pass for tracer
mass with fixed meteorology/air-mass evolution. It accumulates
surface-emission footprints from an adjoint seed without perturbing each
surface cell. The optional vertical-diffusion slot mirrors the runtime
surface-source palindrome: half diffusion, midpoint emissions, half diffusion.
The optional convection slot transposes the CS `CMFMCConvection` and
`TM5Convection` column operators.
"""
module Adjoints

using KernelAbstractions: @Const, @atomic, @index, @kernel, get_backend, synchronize

using ..Grids: CubedSphereMesh, reciprocal_edge,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ..Operators.Advection: CSAdvectionWorkspace, NoLimiter,
    MonotoneLimiter, PPMScheme, SlopesScheme, UpwindScheme,
    LinRoodPPMScheme,
    fill_panel_halos!, strang_split_cs!, copy_corners!,
    _cs_static_subcycle_count, _sweep_x_panel!, _sweep_y_panel!, _sweep_z_panel!,
    apply_linrood_horizontal_adjoint_single_panel!,
    _init_q_buf_kernel!, _ppm_y_face_kernel!, _ppm_x_face_kernel!,
    _ppm_y_face_from_q_kernel!, _ppm_x_face_from_q_kernel!,
    _pre_advect_y_kernel!, _pre_advect_x_kernel!, _linrood_update_kernel!,
    strang_split_linrood_ppm!, CSLinRoodAdvectionWorkspace, LinRoodWorkspace,
    fv_tp_2d_cs!, _sweep_z!
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion,
    apply_vertical_diffusion_vmr!
using ..Operators.Convection: CMFMCConvection, CMFMCWorkspace,
    NoConvection, TM5Convection, TM5Workspace,
    invalidate_cmfmc_cache!, _get_or_compute_n_sub!,
    _tm5_diagnose_cloud_dims, _tm5_build_conv1!, _tm5_lu!
using ..State: AbstractCubedSphereField, field_value, panel_field, update_field!
using ..MetDrivers: ConvectionForcing, current_time

# Plan 26 P0.1 — tape storage policies + record types live in src/Tape/
# (loaded before Adjoints in src/AtmosTransport.jl). Re-imported here so
# call sites continue to use the unqualified names. No semantic change
# from the previous monolithic definitions in this file.
using ..Tape: AbstractCSTapeStorage,
              DeviceCSTapeStorage, PinnedHostCSTapeStorage,
              CSTapeSlot, PinnedHostCSTapeSlot,
              _tape_storage, _tape_panels,
              _allocate_tape_slot, stage_panels!, _stage_panels,
              _after_tape_stage!, _after_tape_read!,
              _sync_pinned_tape_storage!, _ensure_tape_read_cache!,
              _bytes_per_panel_tuple,
              _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord,
              _CSDiffusionRecord, _CSConvectionRecord, _CSTapeOp

const CSAdjointLinearScheme = Union{UpwindScheme, SlopesScheme{NoLimiter}, PPMScheme{NoLimiter}}
const CSAdjointNonlinearScheme = Union{PPMScheme{MonotoneLimiter}}
# Plan 25 Commit 6: LinRoodPPMScheme is supported via its own
# horizontal tape record (`_CSLinRoodHorizRecord`) and the kernel
# adjoints shipped in `src/Operators/Advection/linrood_adjoint_kernels.jl`.
# The reverse-loop dispatch arm in `_collect_surface_footprints`
# handles the new record type alongside the existing
# `_CSSweepRecord`, `_CSHaloRecord`, `_CSDiffusionRecord`,
# `_CSConvectionRecord`, and `_CSMidpointRecord` cases. ORD=5 only.
const CSAdjointLinRoodScheme = LinRoodPPMScheme
const CSAdjointSupportedScheme = Union{CSAdjointLinearScheme,
                                        CSAdjointNonlinearScheme,
                                        CSAdjointLinRoodScheme}

# Plan 26 P0.2 — footprint-objective types + seeding/eval kernels in a focused file.
include("ObjectiveSeeding.jl")


# Plan 26 P0.3a — CSFootprintResult + CSTapeByteEstimate relocated to a focused file.
include("../Footprint/FootprintResult.jl")


# Plan 26 P0.4a — control / observation / 4D-Var result types relocated to Inversion/.
include("../Inversion/Observations.jl")


"""
    CSAdjointWorkspace(mesh, prototype)

Scratch storage for CS adjoint sweeps. `lambda_A` is one halo-padded panel
used as the per-panel transpose output.
"""
struct CSAdjointWorkspace{FT, A <: AbstractArray{FT, 3}}
    lambda_A::A
end

function CSAdjointWorkspace(mesh::CubedSphereMesh,
                            prototype::AbstractArray{FT, 3}) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    return CSAdjointWorkspace{FT, typeof(prototype)}(similar(prototype, FT, N, N, size(prototype, 3)))
end

struct _SingleSurfaceRatePerturbation{FT}
    step::Int
    panel::Int
    i::Int
    j::Int
    rate_delta::FT
end

_copy_panel_tuple(panels) = ntuple(p -> copy(panels[p]), 6)

function _zero_panel_tuple_like(panels)
    return ntuple(p -> begin
        a = similar(panels[p])
        fill!(a, zero(eltype(a)))
        a
    end, 6)
end

function _zero_surface_rates(mesh::CubedSphereMesh, prototype::AbstractArray{FT}) where {FT}
    return ntuple(_ -> begin
        a = similar(prototype, FT, mesh.Nc, mesh.Nc)
        fill!(a, zero(FT))
        a
    end, 6)
end

function _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    nsteps = length(panels_am_steps)
    nsteps > 0 || throw(ArgumentError("footprint generation requires at least one step"))
    length(panels_bm_steps) == nsteps ||
        throw(ArgumentError("panels_bm_steps length $(length(panels_bm_steps)) does not match panels_am_steps length $nsteps"))
    length(panels_cm_steps) == nsteps ||
        throw(ArgumentError("panels_cm_steps length $(length(panels_cm_steps)) does not match panels_am_steps length $nsteps"))
    return nsteps
end

function _validate_emission_rates(rates, nsteps::Int, mesh::CubedSphereMesh,
                                  name::AbstractString)
    rates === nothing && return nothing
    length(rates) == nsteps || throw(ArgumentError(
        "$name length $(length(rates)) does not match nsteps $nsteps"))
    @inbounds for step in 1:nsteps, p in 1:6
        size(rates[step][p]) == (mesh.Nc, mesh.Nc) || throw(DimensionMismatch(
            "$name step $step panel $p has shape $(size(rates[step][p])); " *
            "expected $((mesh.Nc, mesh.Nc))"))
    end
    return nothing
end


@kernel function _add_surface_rates_kernel!(panel_rm, @Const(panel_rate),
                                            dt, Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds panel_rm[i + Hp, j + Hp, Nz] += panel_rate[i, j] * dt
end

@kernel function _add_surface_perturbation_kernel!(panel_rm, dt_rate,
                                                   i, j, Nz)
    _ = @index(Global)
    @inbounds panel_rm[i, j, Nz] += dt_rate
end

function _add_surface_rates!(panels_rm, rates::NTuple{6}, dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    @inbounds for p in 1:6
        panel_rm = panels_rm[p]
        panel_rate = rates[p]
        size(panel_rate) == (mesh.Nc, mesh.Nc) || throw(DimensionMismatch(
            "surface rate panel $p has shape $(size(panel_rate)); expected $((mesh.Nc, mesh.Nc))"))
        backend = get_backend(panel_rm)
        kernel! = _add_surface_rates_kernel!(backend, (16, 16))
        kernel!(panel_rm, panel_rate, eltype(panel_rm)(dt), Int32(Hp), Int32(Nz);
                ndrange = (mesh.Nc, mesh.Nc))
        synchronize(backend)
    end
    return nothing
end

function _add_surface_perturbation!(panels_rm, perturb::_SingleSurfaceRatePerturbation,
                                    dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    panel_rm = panels_rm[perturb.panel]
    backend = get_backend(panel_rm)
    kernel! = _add_surface_perturbation_kernel!(backend, 1)
    kernel!(panel_rm,
            eltype(panel_rm)(perturb.rate_delta) * eltype(panel_rm)(dt),
            Int32(Hp + perturb.i), Int32(Hp + perturb.j), Int32(Nz);
            ndrange = 1)
    synchronize(backend)
    return nothing
end

# Plan 26 P0.2 — split-sweep advection adjoint kernels relocated to a focused file.
include("AdvectionAdjoint.jl")

# Plan 26 P0.2 — CS halo-exchange adjoint kernels relocated to a focused file.
include("HaloAdjoint.jl")



@kernel function _add_weighted_footprint_kernel!(dst, @Const(src), weight)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] += weight * src[i, j]
end

function _aggregate_surface_window(result::CSFootprintResult,
                                   window::CSSurfaceFluxWindow;
                                   ignore_future::Bool = false)
    nsteps = length(result.footprints)
    FT = eltype(result.footprints[1][1])
    aggregate = ntuple(p -> begin
        a = similar(result.footprints[1][p])
        fill!(a, zero(FT))
        a
    end, 6)
    @inbounds for (idx, step) in enumerate(window.steps)
        if !(1 <= step <= nsteps)
            if ignore_future && step > nsteps
                continue
            end
            throw(ArgumentError(
                "surface-flux window $(window.name) references step $step, " *
                "but the footprint has $nsteps steps"))
        end
        weight = FT(window.weights[idx])
        for p in 1:6
            backend = get_backend(aggregate[p])
            kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
            kernel!(aggregate[p], result.footprints[step][p], weight;
                    ndrange = size(aggregate[p]))
            synchronize(backend)
        end
    end
    return aggregate
end

function _zero_surface_like(values::NTuple{6, A2}) where {FT, A2 <: AbstractArray{FT, 2}}
    return ntuple(p -> begin
        a = similar(values[p])
        fill!(a, zero(FT))
        a
    end, 6)
end

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

# Plan 26 P0.2 — vertical-diffusion adjoint kernels relocated to a focused file.
include("DiffusionAdjoint.jl")

# Plan 26 P0.2 — CMFMC + TM5 convection adjoint kernels relocated to a focused file.
include("ConvectionAdjoint.jl")


function _convection_forcing_at(convection_forcing, step::Int, nsteps::Int)
    convection_forcing_is_vector = convection_forcing isa AbstractVector
    if convection_forcing === nothing
        return nothing
    elseif convection_forcing_is_vector
        length(convection_forcing) == nsteps || throw(ArgumentError(
            "convection_forcing length $(length(convection_forcing)) does not match nsteps $nsteps"))
        return convection_forcing[step]
    else
        return convection_forcing
    end
end

# Plan 26 P0.1: tape storage policies + staging API now live in
# `src/Tape/TapeStorage.jl` and are imported via `using ..Tape` at
# the top of this module.


# Plan 26 P0.1: tape record types (_CSSweepRecord, _CSHaloRecord,
# _CSMidpointRecord, _CSDiffusionRecord, _CSConvectionRecord) and the
# _CSTapeOp union now live in `src/Tape/TapeRecords.jl` and are imported
# via `using ..Tape` at the top of this module. The `S <: CSAdjointSupportedScheme`
# type constraint on `_CSSweepRecord.scheme` was relaxed to plain `S`
# during the relocation (see Plan 26 NOTES for the dependency-order
# rationale); the constraint is now enforced at the
# `_record_sweep!`/`_adjoint_scheme_sweep!` call sites below.

# Plan 25 Commit 6 — LinRoodPPMScheme tape record + forward/reverse
# integration. Defines `_CSLinRoodHorizRecord`,
# `_record_cs_linrood_tape`, `_apply_cs_linrood_horizontal_adjoint!`.
include("LinRoodTape.jl")

# Plan 26 P0.3b — tape-recorder kernels relocated to a focused file.
include("../Footprint/TapeRecording.jl")



# Plan 26 P0.3c — reverse-loop driver + forward-replay helpers relocated.
include("../Footprint/ReverseLoop.jl")


# Plan 26 P0.3c — user-facing footprint API relocated to a focused file.
include("../Footprint/FootprintAPI.jl")


"""
    cs_surface_flux_jacobian(..., objectives, windows; kwargs...)

Compute surface-flux Jacobian maps for several layer/column objectives and
named time windows. Each returned `footprints[obj, window]` entry is an
`NTuple{6}` of `(Nc, Nc)` arrays. Window aggregation is a weighted sum of
per-step emission-rate footprints; use `CSSurfaceFluxWindow(...;
normalize=true)` for average-rate controls or explicit `weights` for a
custom temporal basis.
"""
_objective_vector(obj::AbstractCSFootprintObjective) =
    AbstractCSFootprintObjective[obj]
_objective_vector(objectives) =
    AbstractCSFootprintObjective[objectives...]
_window_vector(window::CSSurfaceFluxWindow) =
    CSSurfaceFluxWindow[window]
_window_vector(windows) =
    CSSurfaceFluxWindow[windows...]

function cs_surface_flux_jacobian(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objectives,
                                  windows;
                                  kwargs...)
    objective_vec = _objective_vector(objectives)
    window_vec = _window_vector(windows)
    isempty(objective_vec) && throw(ArgumentError("at least one objective is required"))
    isempty(window_vec) && throw(ArgumentError("at least one surface-flux window is required"))

    per_step = Vector{CSFootprintResult}(undef, length(objective_vec))
    first_result = cs_surface_emission_footprint(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, objective_vec[1]; kwargs...)
    per_step[1] = first_result
    first_agg = _aggregate_surface_window(first_result, window_vec[1])
    footprints = Matrix{typeof(first_agg)}(undef, length(objective_vec), length(window_vec))
    footprints[1, 1] = first_agg
    for w in 2:length(window_vec)
        footprints[1, w] = _aggregate_surface_window(first_result, window_vec[w])
    end
    for o in 2:length(objective_vec)
        result = cs_surface_emission_footprint(
            panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
            mesh, objective_vec[o]; kwargs...)
        per_step[o] = result
        for w in eachindex(window_vec)
            footprints[o, w] = _aggregate_surface_window(result, window_vec[w])
        end
    end
    A2 = typeof(first_agg[1])
    FT = eltype(first_agg[1])
    return CSSurfaceFluxJacobianResult{FT, A2}(
        objective_vec, window_vec, footprints, per_step, first_result.dt)
end

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

export AbstractCSFootprintObjective
export CSLayerMeanObjective, CSColumnMeanObjective, CSSeedObjective, CSFootprintResult
export CSSurfaceFluxWindow, CSSurfaceFluxJacobianResult
export CSObservation, CSSurfaceFluxControl, CS4DVarResult, CS4DVarSolveResult
export CSAdjointWorkspace, CSTapeSlot, DeviceCSTapeStorage, PinnedHostCSTapeStorage
export PinnedHostCSTapeSlot, CSTapeByteEstimate
export evaluate_objective, run_cs_footprint_forward
export cs_surface_emission_footprint, cs_surface_emission_footprint_from_seed
export cs_tape_byte_estimate
export cs_surface_flux_jacobian
export cs_surface_flux_4dvar, cs_surface_flux_4dvar_optimize

end # module Adjoints
