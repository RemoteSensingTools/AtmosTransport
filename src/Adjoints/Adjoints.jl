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

abstract type AbstractCSFootprintObjective end

"""
    CSLayerMeanObjective(panel, i, j, level)

Scalar objective equal to the final-layer mixing ratio
`rm[panel][i, j, level] / m[panel][i, j, level]` on the physical CS
interior indices. `level = Nz` is the surface layer.
"""
struct CSLayerMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
    level::Int
end

"""
    CSColumnMeanObjective(panel, i, j)

Scalar objective equal to the final air-mass-weighted column mean mixing
ratio at one physical CS interior cell.
"""
struct CSColumnMeanObjective <: AbstractCSFootprintObjective
    panel::Int
    i::Int
    j::Int
end

"""
    CSSeedObjective()

Marker objective used when the caller supplies an explicit final adjoint
seed (`dJ/drm_final`) instead of one of the built-in scalar objectives.
"""
struct CSSeedObjective <: AbstractCSFootprintObjective end

"""
    CSFootprintResult

Reverse-mode footprint result for one scalar objective. `footprints[t]`
is an `NTuple{6}` of `(Nc, Nc)` arrays containing `dJ / dE`, where `E` is
the per-cell surface-emission rate [kg s^-1] applied at the midpoint of
model step `t`. `lag_steps[t] == nsteps - t`.
"""
struct CSFootprintResult{FT, O <: AbstractCSFootprintObjective, A2 <: AbstractArray{FT, 2}}
    objective::O
    footprints::Vector{NTuple{6, A2}}
    lag_steps::Vector{Int}
    dt::FT
    # Compatibility field from the earlier finite-difference prototype;
    # reverse-mode results set it to zero.
    epsilon::FT
    # Not evaluated by the reverse pass. Current built-in objectives only
    # need dJ/drm at final time, independent of final tracer mass.
    base_value::FT
end

"""
    CSTapeByteEstimate

Counts and byte estimate for the CS adjoint tape. `state_bytes` counts full
stored panel states: air-mass states for linear schemes, plus tracer branch
states for nonlinear limited schemes. Halo and midpoint records are scalar
metadata and are counted in `total_records` but not included in
`state_bytes`.
"""
struct CSTapeByteEstimate
    nsteps::Int
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
    state_records::Int
    total_records::Int
    bytes_per_state::Int
    state_bytes::Int
end

"""
    CSSurfaceFluxWindow(name, steps; weights=nothing, normalize=false)

Named surface-flux control window for Jacobian aggregation. `steps` can be
a single step or any iterable of step indices. By default, a window sums
per-step surface-emission footprints, which corresponds to perturbing the
same rate over the whole window. Pass `normalize=true` for an average-rate
control, or explicit `weights` for custom temporal basis functions.
"""
struct CSSurfaceFluxWindow
    name::Symbol
    steps::Vector{Int}
    weights::Vector{Float64}
end

_step_vector(step::Integer) = [Int(step)]
_step_vector(steps) = Int.(collect(steps))

function CSSurfaceFluxWindow(name::Symbol, steps; weights = nothing,
                             normalize::Bool = false)
    step_vec = _step_vector(steps)
    isempty(step_vec) && throw(ArgumentError("surface-flux window $name has no steps"))
    any(<=(0), step_vec) &&
        throw(ArgumentError("surface-flux window $name contains non-positive step indices"))
    weight_vec = if weights === nothing
        ones(Float64, length(step_vec))
    else
        Float64.(collect(weights))
    end
    length(weight_vec) == length(step_vec) || throw(ArgumentError(
        "surface-flux window $name has $(length(step_vec)) steps but " *
        "$(length(weight_vec)) weights"))
    if normalize
        total = sum(weight_vec)
        total != 0 || throw(ArgumentError(
            "surface-flux window $name cannot normalize zero-sum weights"))
        weight_vec ./= total
    end
    return CSSurfaceFluxWindow(name, step_vec, weight_vec)
end

CSSurfaceFluxWindow(name::AbstractString, steps; kwargs...) =
    CSSurfaceFluxWindow(Symbol(name), steps; kwargs...)

struct CSSurfaceFluxJacobianResult{FT, A2 <: AbstractArray{FT, 2}}
    objectives::Vector{AbstractCSFootprintObjective}
    windows::Vector{CSSurfaceFluxWindow}
    footprints::Matrix{NTuple{6, A2}}
    per_step_results::Vector{<:CSFootprintResult}
    dt::FT
end

"""
    CSObservation(step, objective, value, sigma)

Scalar observation used by the prototype CS 4D-Var surface-flux path.
`step` is the model-step index after which the objective is sampled.
`sigma` is the observation-error standard deviation.
"""
struct CSObservation{O <: AbstractCSFootprintObjective, FT <: AbstractFloat}
    step::Int
    objective::O
    value::FT
    sigma::FT
end

function CSObservation(step::Integer, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real)
    step > 0 || throw(ArgumentError("CSObservation step must be positive, got $step"))
    sigma > 0 || throw(ArgumentError("CSObservation sigma must be positive, got $sigma"))
    FT = promote_type(typeof(float(value)), typeof(float(sigma)))
    return CSObservation{typeof(objective), FT}(Int(step), objective, FT(value), FT(sigma))
end

"""
    CSSurfaceFluxControl(window, value; background=nothing, sigma=nothing)

Surface-flux control block for the prototype 4D-Var path. `value` is an
`NTuple{6}` of `(Nc, Nc)` rate arrays tied to `window`. Optional
`background` and `sigma` add the standard diagonal background term
`0.5 * ((value - background) / sigma)^2`; `sigma` can be a scalar or
matching panel arrays.
"""
struct CSSurfaceFluxControl{FT, A2 <: AbstractArray{FT, 2}, B, S}
    window::CSSurfaceFluxWindow
    value::NTuple{6, A2}
    background::B
    sigma::S
end

function CSSurfaceFluxControl(window::CSSurfaceFluxWindow,
                              value::NTuple{6, A2};
                              background = nothing,
                              sigma = nothing) where {FT, A2 <: AbstractArray{FT, 2}}
    background === nothing || sigma !== nothing || throw(ArgumentError(
        "surface-flux control background requires `sigma`"))
    if background !== nothing
        background isa NTuple{6} || throw(ArgumentError(
            "surface-flux control background must be nothing or an NTuple{6}"))
        @inbounds for p in 1:6
            size(background[p]) == size(value[p]) || throw(DimensionMismatch(
                "surface-flux control background panel $p has shape $(size(background[p])); " *
                "expected $(size(value[p]))"))
        end
    end
    if sigma !== nothing && !(sigma isa Real) && !(sigma isa NTuple{6})
        throw(ArgumentError("surface-flux control sigma must be nothing, a scalar, or an NTuple{6}"))
    end
    if sigma isa Real
        sigma > 0 || throw(ArgumentError("surface-flux control scalar sigma must be positive"))
    elseif sigma isa NTuple{6}
        @inbounds for p in 1:6
            size(sigma[p]) == size(value[p]) || throw(DimensionMismatch(
                "surface-flux control sigma panel $p has shape $(size(sigma[p])); " *
                "expected $(size(value[p]))"))
        end
    end
    return CSSurfaceFluxControl{FT, A2, typeof(background), typeof(sigma)}(
        window, value, background, sigma)
end

struct CS4DVarResult{FT, A2 <: AbstractArray{FT, 2}}
    cost::FT
    observation_cost::FT
    background_cost::FT
    simulated::Vector{FT}
    residuals::Vector{FT}
    gradients::Vector{NTuple{6, A2}}
    gradient_by_name::Dict{Symbol, NTuple{6, A2}}
    controls::Vector{<:CSSurfaceFluxControl}
    observations::Vector{<:CSObservation}
end

struct CS4DVarSolveResult{FT, A2 <: AbstractArray{FT, 2}}
    controls::Vector{<:CSSurfaceFluxControl}
    last::CS4DVarResult{FT, A2}
    cost_history::Vector{FT}
    gradient_norm_history::Vector{FT}
    step_history::Vector{FT}
    iterations::Int
end

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

function _validate_objective(obj::CSLayerMeanObjective, mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    1 <= obj.level <= Nz || throw(ArgumentError("level must be in 1:$Nz, got $(obj.level)"))
    return nothing
end

function _validate_objective(obj::CSColumnMeanObjective, mesh::CubedSphereMesh, Nz::Int)
    1 <= obj.panel <= 6 || throw(ArgumentError("panel must be in 1:6, got $(obj.panel)"))
    1 <= obj.i <= mesh.Nc || throw(ArgumentError("i must be in 1:$(mesh.Nc), got $(obj.i)"))
    1 <= obj.j <= mesh.Nc || throw(ArgumentError("j must be in 1:$(mesh.Nc), got $(obj.j)"))
    return nothing
end

function _validate_objective(::CSSeedObjective, mesh::CubedSphereMesh, Nz::Int)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

@kernel function _evaluate_layer_objective_kernel!(out, @Const(rm), @Const(m),
                                                   i, j, k)
    _ = @index(Global)
    FT = eltype(rm)
    @inbounds out[1] = rm[i, j, k] / max(m[i, j, k], eps(FT))
end

@kernel function _evaluate_column_objective_kernel!(out, @Const(rm), @Const(m),
                                                    i, j, Nz::Int)
    _ = @index(Global)
    FT = eltype(rm)
    num = zero(FT)
    den = zero(FT)
    @inbounds for k in 1:Nz
        num += rm[i, j, k]
        den += m[i, j, k]
    end
    @inbounds out[1] = num / max(den, eps(FT))
end

_host_scalar(a) = Array(a)[1]

function evaluate_objective(obj::CSLayerMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    k = obj.level
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_layer_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj), Int32(k);
            ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
end

function evaluate_objective(obj::CSColumnMeanObjective, panels_rm, panels_m,
                            mesh::CubedSphereMesh)
    Hp = mesh.Hp
    p = obj.panel
    ii = Hp + obj.i
    jj = Hp + obj.j
    FT = eltype(panels_rm[p])
    out = similar(panels_rm[p], FT, 1)
    backend = get_backend(panels_rm[p])
    kernel! = _evaluate_column_objective_kernel!(backend, 1)
    kernel!(out, panels_rm[p], panels_m[p], Int32(ii), Int32(jj),
            size(panels_rm[p], 3); ndrange = 1)
    synchronize(backend)
    return _host_scalar(out)
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

# ---------------------------------------------------------------------------
# Kernelized adjoint building blocks
# ---------------------------------------------------------------------------

@inline _wrap_periodic(idx, N) = mod1(idx, N)

@inline function _upwind_face_coeffs(F, m_l, m_r)
    FT = typeof(F)
    if F >= zero(FT)
        return clamp(F / max(m_l, eps(FT)), zero(FT), one(FT)), zero(FT)
    else
        return zero(FT), clamp(F / max(m_r, eps(FT)), -one(FT), zero(FT))
    end
end

@inline function _slopes_no_limiter_face_coeffs(F, m_ll, m_l, m_r, m_rr)
    FT = typeof(F)
    m_floor = eps(FT)
    mll = max(m_ll, m_floor)
    ml = max(m_l, m_floor)
    mr = max(m_r, m_floor)
    mrr = max(m_rr, m_floor)
    if F >= zero(FT)
        α = clamp(F / ml, zero(FT), one(FT))
        β = α * (one(FT) - α) * ml / FT(4)
        return -β / mll, α, β / mr, zero(FT)
    else
        α = clamp(F / mr, -one(FT), zero(FT))
        β = -α * (one(FT) + α) * mr / FT(4)
        return zero(FT), -β / ml, α, β / mrr
    end
end

@inline function _ppm_no_limiter_face_coeffs(F, m_ll, m_l, m_r, m_rr)
    FT = typeof(F)
    tw12 = FT(1) / FT(12)
    m_floor = eps(FT)
    mll = max(m_ll, m_floor)
    ml = max(m_l, m_floor)
    mr = max(m_r, m_floor)
    mrr = max(m_rr, m_floor)
    if F >= zero(FT)
        α = clamp(F / ml, zero(FT), one(FT))
        β = α * (one(FT) - α) * ml
        c_ll = β * (-tw12) / mll
        c_l  = α + β * (FT(-5) * tw12) / ml
        c_r  = β * (FT(7) * tw12) / mr
        c_rr = β * (-tw12) / mrr
        return c_ll, c_l, c_r, c_rr
    else
        α = clamp(F / mr, -one(FT), zero(FT))
        β = -α * (one(FT) + α) * mr
        c_ll = β * tw12 / mll
        c_l  = β * (FT(-7) * tw12) / ml
        c_r  = α + β * (FT(5) * tw12) / mr
        c_rr = β * tw12 / mrr
        return c_ll, c_l, c_r, c_rr
    end
end

@inline _d6_zero(::Type{FT}) where {FT} =
    (zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))

@inline _d6_basis(::Type{FT}, n::Int, scale) where {FT} =
    (n == 1 ? FT(scale) : zero(FT),
     n == 2 ? FT(scale) : zero(FT),
     n == 3 ? FT(scale) : zero(FT),
     n == 4 ? FT(scale) : zero(FT),
     n == 5 ? FT(scale) : zero(FT),
     n == 6 ? FT(scale) : zero(FT))

@inline _d6_add(a, b) =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3],
     a[4] + b[4], a[5] + b[5], a[6] + b[6])

@inline _d6_sub(a, b) =
    (a[1] - b[1], a[2] - b[2], a[3] - b[3],
     a[4] - b[4], a[5] - b[5], a[6] - b[6])

@inline _d6_scale(a, s) =
    (s * a[1], s * a[2], s * a[3],
     s * a[4], s * a[5], s * a[6])

@inline function _ppm_edge_value_ad(c_ll, d_ll, c_l, d_l, c_r, d_r, c_rr, d_rr)
    FT = typeof(c_ll)
    seven_twelfths = FT(7) / FT(12)
    one_twelfth = FT(1) / FT(12)
    value = seven_twelfths * (c_l + c_r) - one_twelfth * (c_ll + c_rr)
    deriv = _d6_sub(_d6_scale(_d6_add(d_l, d_r), seven_twelfths),
                    _d6_scale(_d6_add(d_ll, d_rr), one_twelfth))
    return value, deriv
end

@inline function _ppm_limit_profile_monotone_ad(q_L, dq_L, c_bar, dc_bar, q_R, dq_R)
    FT = typeof(c_bar)
    is_extremum = (q_R - c_bar) * (c_bar - q_L) <= zero(FT)
    dc = q_R - q_L
    c6 = FT(6) * (c_bar - (q_L + q_R) / FT(2))
    needs_left_fix = dc * c6 > dc * dc
    needs_right_fix = -(dc * dc) > dc * c6

    if is_extremum
        return c_bar, dc_bar, c_bar, dc_bar
    end

    q_L_new = q_L
    dq_L_new = dq_L
    if needs_left_fix
        q_L_new = FT(3) * c_bar - FT(2) * q_R
        dq_L_new = _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_R, FT(2)))
    end

    if needs_right_fix
        q_R_new = FT(3) * c_bar - FT(2) * q_L_new
        dq_R_new = _d6_sub(_d6_scale(dc_bar, FT(3)), _d6_scale(dq_L_new, FT(2)))
        return q_L_new, dq_L_new, q_R_new, dq_R_new
    end

    return q_L_new, dq_L_new, q_R, dq_R
end

@inline function _limited_moment_monotone_ad(sx, dsx, rm_cell, drm_cell)
    limited_min = min(sx, rm_cell)
    if limited_min < -rm_cell
        return -rm_cell, _d6_scale(drm_cell, -one(typeof(rm_cell)))
    elseif sx > rm_cell
        return rm_cell, drm_cell
    else
        return sx, dsx
    end
end

@inline function _ppm_monotone_face_coeffs(F,
                                           m_3, m_2, m_1, m_0, m_p, m_pp,
                                           rm_3, rm_2, rm_1, rm_0, rm_p, rm_pp,
                                           interior_l::Bool, interior_r::Bool)
    FT = typeof(F)
    m_floor = eps(FT)
    m3 = max(m_3, m_floor)
    m2 = max(m_2, m_floor)
    m1 = max(m_1, m_floor)
    m0 = max(m_0, m_floor)
    mp = max(m_p, m_floor)
    mpp = max(m_pp, m_floor)

    c_3 = rm_3 / m3
    c_2 = rm_2 / m2
    c_1 = rm_1 / m1
    c_0 = rm_0 / m0
    c_p = rm_p / mp
    c_pp = rm_pp / mpp

    dc_3 = _d6_basis(FT, 1, inv(m3))
    dc_2 = _d6_basis(FT, 2, inv(m2))
    dc_1 = _d6_basis(FT, 3, inv(m1))
    dc_0 = _d6_basis(FT, 4, inv(m0))
    dc_p = _d6_basis(FT, 5, inv(mp))
    dc_pp = _d6_basis(FT, 6, inv(mpp))

    e_left, de_left = _ppm_edge_value_ad(c_3, dc_3, c_2, dc_2, c_1, dc_1, c_0, dc_0)
    e_face, de_face = _ppm_edge_value_ad(c_2, dc_2, c_1, dc_1, c_0, dc_0, c_p, dc_p)
    e_right, de_right = _ppm_edge_value_ad(c_1, dc_1, c_0, dc_0, c_p, dc_p, c_pp, dc_pp)

    _qLl, _dqLl, qRl, dqRl =
        _ppm_limit_profile_monotone_ad(e_left, de_left, c_1, dc_1, e_face, de_face)
    qLr, dqLr, _qRr, _dqRr =
        _ppm_limit_profile_monotone_ad(e_face, de_face, c_0, dc_0, e_right, de_right)

    sx_l_raw = m1 * (qRl - c_1)
    dsx_l_raw = _d6_scale(_d6_sub(dqRl, dc_1), m1)
    sx_l, dsx_l = interior_l ?
        _limited_moment_monotone_ad(sx_l_raw, dsx_l_raw, rm_1, _d6_basis(FT, 3, one(FT))) :
        (zero(FT), _d6_zero(FT))

    sx_r_raw = m0 * (c_0 - qLr)
    dsx_r_raw = _d6_scale(_d6_sub(dc_0, dqLr), m0)
    sx_r, dsx_r = interior_r ?
        _limited_moment_monotone_ad(sx_r_raw, dsx_r_raw, rm_0, _d6_basis(FT, 4, one(FT))) :
        (zero(FT), _d6_zero(FT))

    if F >= zero(FT)
        α = clamp(F / m1, zero(FT), one(FT))
        drm_l = _d6_basis(FT, 3, one(FT))
        return _d6_add(_d6_scale(drm_l, α),
                       _d6_scale(dsx_l, α * (one(FT) - α)))
    else
        α = clamp(F / m0, -one(FT), zero(FT))
        drm_r = _d6_basis(FT, 4, one(FT))
        return _d6_sub(_d6_scale(drm_r, α),
                       _d6_scale(dsx_r, α * (one(FT) + α)))
    end
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::UpwindScheme, Nx)
    i_l = _wrap_periodic(face_i - Int32(1), Nx)
    i_r = _wrap_periodic(face_i, Nx)
    c_l, c_r = _upwind_face_coeffs(F, m[i_l, j, k], m[i_r, j, k])
    @atomic lambda_in[i_l, j, k] += scale * c_l
    @atomic lambda_in[i_r, j, k] += scale * c_r
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Nx)
    i_ll = _wrap_periodic(face_i - Int32(2), Nx)
    i_l  = _wrap_periodic(face_i - Int32(1), Nx)
    i_r  = _wrap_periodic(face_i, Nx)
    i_rr = _wrap_periodic(face_i + Int32(1), Nx)
    c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
        F, m[i_ll, j, k], m[i_l, j, k], m[i_r, j, k], m[i_rr, j, k])
    @atomic lambda_in[i_ll, j, k] += scale * c_ll
    @atomic lambda_in[i_l,  j, k] += scale * c_l
    @atomic lambda_in[i_r,  j, k] += scale * c_r
    @atomic lambda_in[i_rr, j, k] += scale * c_rr
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, face_i, j, k, F, scale,
                                      ::PPMScheme{NoLimiter}, Nx)
    i_ll = _wrap_periodic(face_i - Int32(2), Nx)
    i_l  = _wrap_periodic(face_i - Int32(1), Nx)
    i_r  = _wrap_periodic(face_i, Nx)
    i_rr = _wrap_periodic(face_i + Int32(1), Nx)
    c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
        F, m[i_ll, j, k], m[i_l, j, k], m[i_r, j, k], m[i_rr, j, k])
    @atomic lambda_in[i_ll, j, k] += scale * c_ll
    @atomic lambda_in[i_l,  j, k] += scale * c_l
    @atomic lambda_in[i_r,  j, k] += scale * c_r
    @atomic lambda_in[i_rr, j, k] += scale * c_rr
    return nothing
end

@inline function _add_x_face_adjoint!(lambda_in, m, rm, face_i, j, k, F, scale,
                                      ::PPMScheme{MonotoneLimiter}, Nx)
    i_3  = _wrap_periodic(face_i - Int32(3), Nx)
    i_2  = _wrap_periodic(face_i - Int32(2), Nx)
    i_1  = _wrap_periodic(face_i - Int32(1), Nx)
    i_0  = _wrap_periodic(face_i, Nx)
    i_p  = _wrap_periodic(face_i + Int32(1), Nx)
    i_pp = _wrap_periodic(face_i + Int32(2), Nx)
    c = _ppm_monotone_face_coeffs(
        F,
        m[i_3, j, k], m[i_2, j, k], m[i_1, j, k],
        m[i_0, j, k], m[i_p, j, k], m[i_pp, j, k],
        rm[i_3, j, k], rm[i_2, j, k], rm[i_1, j, k],
        rm[i_0, j, k], rm[i_p, j, k], rm[i_pp, j, k],
        true, true)
    @atomic lambda_in[i_3,  j, k] += scale * c[1]
    @atomic lambda_in[i_2,  j, k] += scale * c[2]
    @atomic lambda_in[i_1,  j, k] += scale * c[3]
    @atomic lambda_in[i_0,  j, k] += scale * c[4]
    @atomic lambda_in[i_p,  j, k] += scale * c[5]
    @atomic lambda_in[i_pp, j, k] += scale * c[6]
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::UpwindScheme, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jl = max(face_j - Int32(1), Int32(1))
    jr = min(face_j, Ny)
    c_l, c_r = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
    @atomic lambda_in[i, jl, k] += scale * c_l
    @atomic lambda_in[i, jr, k] += scale * c_r
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    interior_l = (jl > Int32(1)) & (jl < Ny)
    interior_r = (jr > Int32(1)) & (jr < Ny)
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l, _ = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            _, c_r = _upwind_face_coeffs(F, m[i, jl, k], m[i, jr, k])
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, jll, k] += scale * c_ll
    @atomic lambda_in[i, jl,  k] += scale * c_l
    @atomic lambda_in[i, jr,  k] += scale * c_r
    @atomic lambda_in[i, jrr, k] += scale * c_rr
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, i, face_j, k, F, scale,
                                      ::PPMScheme{NoLimiter}, Ny)
    FT = typeof(F)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    interior_l = (jl > Int32(2)) & (jl < Ny - Int32(1))
    interior_r = (jr > Int32(2)) & (jr < Ny - Int32(1))
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = clamp(F / max(m[i, jl, k], eps(FT)), zero(FT), one(FT))
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, jll, k], m[i, jl, k], m[i, jr, k], m[i, jrr, k])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            c_r = clamp(F / max(m[i, jr, k], eps(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, jll, k] += scale * c_ll
    @atomic lambda_in[i, jl,  k] += scale * c_l
    @atomic lambda_in[i, jr,  k] += scale * c_r
    @atomic lambda_in[i, jrr, k] += scale * c_rr
    return nothing
end

@inline function _add_y_face_adjoint!(lambda_in, m, rm, i, face_j, k, F, scale,
                                      ::PPMScheme{MonotoneLimiter}, Ny)
    at_boundary = (face_j <= Int32(1)) | (face_j > Ny)
    at_boundary && return nothing
    j3l = max(face_j - Int32(3), Int32(1))
    jll = max(face_j - Int32(2), Int32(1))
    jl  = max(face_j - Int32(1), Int32(1))
    jr  = min(face_j, Ny)
    jrr = min(face_j + Int32(1), Ny)
    j3r = min(face_j + Int32(2), Ny)
    interior_l = (jl > Int32(2)) & (jl < Ny - Int32(1))
    interior_r = (jr > Int32(2)) & (jr < Ny - Int32(1))
    c = _ppm_monotone_face_coeffs(
        F,
        m[i, j3l, k], m[i, jll, k], m[i, jl, k],
        m[i, jr, k], m[i, jrr, k], m[i, j3r, k],
        rm[i, j3l, k], rm[i, jll, k], rm[i, jl, k],
        rm[i, jr, k], rm[i, jrr, k], rm[i, j3r, k],
        interior_l, interior_r)
    @atomic lambda_in[i, j3l, k] += scale * c[1]
    @atomic lambda_in[i, jll, k] += scale * c[2]
    @atomic lambda_in[i, jl,  k] += scale * c[3]
    @atomic lambda_in[i, jr,  k] += scale * c[4]
    @atomic lambda_in[i, jrr, k] += scale * c[5]
    @atomic lambda_in[i, j3r, k] += scale * c[6]
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::UpwindScheme, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kl = max(face_k - Int32(1), Int32(1))
    kr = min(face_k, Nz)
    c_l, c_r = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
    @atomic lambda_in[i, j, kl] += scale * c_l
    @atomic lambda_in[i, j, kr] += scale * c_r
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::SlopesScheme{NoLimiter}, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    interior_l = (kl > Int32(1)) & (kl < Nz)
    interior_r = (kr > Int32(1)) & (kr < Nz)
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l, _ = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _slopes_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            _, c_r = _upwind_face_coeffs(F, m[i, j, kl], m[i, j, kr])
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, j, kll] += scale * c_ll
    @atomic lambda_in[i, j, kl]  += scale * c_l
    @atomic lambda_in[i, j, kr]  += scale * c_r
    @atomic lambda_in[i, j, krr] += scale * c_rr
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, i, j, face_k, F, scale,
                                      ::PPMScheme{NoLimiter}, Nz)
    FT = typeof(F)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    interior_l = (kl > Int32(2)) & (kl < Nz - Int32(1))
    interior_r = (kr > Int32(2)) & (kr < Nz - Int32(1))
    if F >= zero(FT)
        if interior_l
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = clamp(F / max(m[i, j, kl], eps(FT)), zero(FT), one(FT))
            c_r = zero(FT)
            c_rr = zero(FT)
        end
    else
        if interior_r
            c_ll, c_l, c_r, c_rr = _ppm_no_limiter_face_coeffs(
                F, m[i, j, kll], m[i, j, kl], m[i, j, kr], m[i, j, krr])
        else
            c_ll = zero(FT)
            c_l = zero(FT)
            c_r = clamp(F / max(m[i, j, kr], eps(FT)), -one(FT), zero(FT))
            c_rr = zero(FT)
        end
    end
    @atomic lambda_in[i, j, kll] += scale * c_ll
    @atomic lambda_in[i, j, kl]  += scale * c_l
    @atomic lambda_in[i, j, kr]  += scale * c_r
    @atomic lambda_in[i, j, krr] += scale * c_rr
    return nothing
end

@inline function _add_z_face_adjoint!(lambda_in, m, rm, i, j, face_k, F, scale,
                                      ::PPMScheme{MonotoneLimiter}, Nz)
    at_boundary = (face_k <= Int32(1)) | (face_k > Nz)
    at_boundary && return nothing
    k3l = max(face_k - Int32(3), Int32(1))
    kll = max(face_k - Int32(2), Int32(1))
    kl  = max(face_k - Int32(1), Int32(1))
    kr  = min(face_k, Nz)
    krr = min(face_k + Int32(1), Nz)
    k3r = min(face_k + Int32(2), Nz)
    interior_l = (kl > Int32(2)) & (kl < Nz - Int32(1))
    interior_r = (kr > Int32(2)) & (kr < Nz - Int32(1))
    c = _ppm_monotone_face_coeffs(
        F,
        m[i, j, k3l], m[i, j, kll], m[i, j, kl],
        m[i, j, kr], m[i, j, krr], m[i, j, k3r],
        rm[i, j, k3l], rm[i, j, kll], rm[i, j, kl],
        rm[i, j, kr], rm[i, j, krr], rm[i, j, k3r],
        interior_l, interior_r)
    @atomic lambda_in[i, j, k3l] += scale * c[1]
    @atomic lambda_in[i, j, kll] += scale * c[2]
    @atomic lambda_in[i, j, kl]  += scale * c[3]
    @atomic lambda_in[i, j, kr]  += scale * c[4]
    @atomic lambda_in[i, j, krr] += scale * c[5]
    @atomic lambda_in[i, j, k3r] += scale * c[6]
    return nothing
end

@kernel function _cs_xsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(am),
                                            scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Nx = Int32(Nc + 2 * Hp)
        _add_x_face_adjoint!(lambda_in, m, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, Nx)
        _add_x_face_adjoint!(lambda_in, m, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, Nx)
    end
end

@kernel function _cs_ysweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(bm),
                                            scheme, Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Ny = Int32(Nc + 2 * Hp)
        _add_y_face_adjoint!(lambda_in, m, i, Int32(j),     k, flux_scale * bm[i, j,     k],  bar, scheme, Ny)
        _add_y_face_adjoint!(lambda_in, m, i, Int32(j) + 1, k, flux_scale * bm[i, j + 1, k], -bar, scheme, Ny)
    end
end

@kernel function _cs_zsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(cm),
                                            scheme, Nc, Hp, Nz, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_z_face_adjoint!(lambda_in, m, i, j, Int32(k),     flux_scale * cm[i, j, k],     bar, scheme, Int32(Nz))
        _add_z_face_adjoint!(lambda_in, m, i, j, Int32(k) + 1, flux_scale * cm[i, j, k + 1], -bar, scheme, Int32(Nz))
    end
end

@kernel function _cs_xsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(am),
                                            scheme::PPMScheme{MonotoneLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Nx = Int32(Nc + 2 * Hp)
        _add_x_face_adjoint!(lambda_in, m, rm, Int32(i),     j, k, flux_scale * am[i,     j, k],  bar, scheme, Nx)
        _add_x_face_adjoint!(lambda_in, m, rm, Int32(i) + 1, j, k, flux_scale * am[i + 1, j, k], -bar, scheme, Nx)
    end
end

@kernel function _cs_ysweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(bm),
                                            scheme::PPMScheme{MonotoneLimiter},
                                            Nc, Hp, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        Ny = Int32(Nc + 2 * Hp)
        _add_y_face_adjoint!(lambda_in, m, rm, i, Int32(j),     k, flux_scale * bm[i, j,     k],  bar, scheme, Ny)
        _add_y_face_adjoint!(lambda_in, m, rm, i, Int32(j) + 1, k, flux_scale * bm[i, j + 1, k], -bar, scheme, Ny)
    end
end

@kernel function _cs_zsweep_adjoint_kernel!(lambda_in, @Const(lambda_out),
                                            @Const(m), @Const(rm), @Const(cm),
                                            scheme::PPMScheme{MonotoneLimiter},
                                            Nc, Hp, Nz, flux_scale)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        bar = lambda_out[i, j, k]
        @atomic lambda_in[i, j, k] += bar
        _add_z_face_adjoint!(lambda_in, m, rm, i, j, Int32(k),     flux_scale * cm[i, j, k],     bar, scheme, Int32(Nz))
        _add_z_face_adjoint!(lambda_in, m, rm, i, j, Int32(k) + 1, flux_scale * cm[i, j, k + 1], -bar, scheme, Int32(Nz))
    end
end

function _adjoint_scheme_sweep!(lambda_panels, m_before, flux_panels,
                                direction::Symbol, scheme::CSAdjointLinearScheme,
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    @inbounds for p in 1:6
        fill!(ws.lambda_A, zero(eltype(ws.lambda_A)))
        backend = get_backend(lambda_panels[p])
        if direction === :x
            kernel! = _cs_xsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], flux_panels[p],
                    scheme, Int32(Nc), Int32(Hp), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :y
            kernel! = _cs_ysweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], flux_panels[p],
                    scheme, Int32(Nc), Int32(Hp), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :z
            kernel! = _cs_zsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], flux_panels[p],
                    scheme, Int32(Nc), Int32(Hp), Int32(Nz), eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        else
            throw(ArgumentError("unknown CS adjoint sweep direction $direction"))
        end
        synchronize(backend)
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    return nothing
end

function _adjoint_scheme_sweep!(lambda_panels, m_before, rm_before, flux_panels,
                                direction::Symbol, scheme::PPMScheme{MonotoneLimiter},
                                mesh::CubedSphereMesh, ws::CSAdjointWorkspace,
                                flux_scale)
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(lambda_panels[1], 3)
    @inbounds for p in 1:6
        fill!(ws.lambda_A, zero(eltype(ws.lambda_A)))
        backend = get_backend(lambda_panels[p])
        if direction === :x
            kernel! = _cs_xsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    flux_panels[p], scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :y
            kernel! = _cs_ysweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    flux_panels[p], scheme, Int32(Nc), Int32(Hp),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        elseif direction === :z
            kernel! = _cs_zsweep_adjoint_kernel!(backend, 256)
            kernel!(ws.lambda_A, lambda_panels[p], m_before[p], rm_before[p],
                    flux_panels[p], scheme, Int32(Nc), Int32(Hp), Int32(Nz),
                    eltype(ws.lambda_A)(flux_scale);
                    ndrange=(Nc, Nc, Nz))
        else
            throw(ArgumentError("unknown CS adjoint sweep direction $direction"))
        end
        synchronize(backend)
        copyto!(lambda_panels[p], ws.lambda_A)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Adjoint of CS halo exchange
# ---------------------------------------------------------------------------

@inline function _edge_interior_ij(q_e, d, s, Nc, Hp)
    if q_e == EDGE_NORTH
        return (Hp + s, Hp + Nc + 1 - d)
    elseif q_e == EDGE_SOUTH
        return (Hp + s, Hp + d)
    elseif q_e == EDGE_EAST
        return (Hp + Nc + 1 - d, Hp + s)
    else
        return (Hp + d, Hp + s)
    end
end

@inline function _edge_halo_ij(e, d, s, Nc, Hp)
    if e == EDGE_NORTH
        return (Hp + s, Hp + Nc + d)
    elseif e == EDGE_SOUTH
        return (Hp + s, Hp + 1 - d)
    elseif e == EDGE_EAST
        return (Hp + Nc + d, Hp + s)
    else
        return (Hp + 1 - d, Hp + s)
    end
end

@inline function _corner_source_ij(i_dst, j_dst, Nc, Hp, N, dir)
    if dir == 1
        if i_dst <= Hp && j_dst <= Hp
            return (j_dst, 2 * Hp + 1 - i_dst)
        elseif i_dst > Hp + Nc && j_dst <= Hp
            return (N + 1 - j_dst, i_dst - Nc)
        elseif i_dst > Hp + Nc && j_dst > Hp + Nc
            return (j_dst, 2 * (Nc + Hp) + 1 - i_dst)
        else
            return (N + 1 - j_dst, i_dst + Nc)
        end
    else
        if i_dst <= Hp && j_dst <= Hp
            return (2 * Hp + 1 - j_dst, i_dst)
        elseif i_dst > Hp + Nc && j_dst <= Hp
            return (Nc + j_dst, N + 1 - i_dst)
        elseif i_dst > Hp + Nc && j_dst > Hp + Nc
            return (2 * (Nc + Hp) + 1 - j_dst, i_dst)
        else
            return (j_dst - Nc, N + 1 - i_dst)
        end
    end
end

@kernel function _adjoint_corner_halo_kernel!(lambda, Nc, Hp, N, dir)
    di, dj, k = @index(Global, NTuple)
    @inbounds begin
        i_sw = Hp + 1 - di;  j_sw = Hp + 1 - dj
        i_se = Hp + Nc + di; j_se = Hp + 1 - dj
        i_ne = Hp + Nc + di; j_ne = Hp + Nc + dj
        i_nw = Hp + 1 - di;  j_nw = Hp + Nc + dj

        si, sj = _corner_source_ij(i_sw, j_sw, Nc, Hp, N, dir)
        val = lambda[i_sw, j_sw, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_sw, j_sw, k] = zero(val)

        si, sj = _corner_source_ij(i_se, j_se, Nc, Hp, N, dir)
        val = lambda[i_se, j_se, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_se, j_se, k] = zero(val)

        si, sj = _corner_source_ij(i_ne, j_ne, Nc, Hp, N, dir)
        val = lambda[i_ne, j_ne, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_ne, j_ne, k] = zero(val)

        si, sj = _corner_source_ij(i_nw, j_nw, Nc, Hp, N, dir)
        val = lambda[i_nw, j_nw, k]
        @atomic lambda[si, sj, k] += val
        lambda[i_nw, j_nw, k] = zero(val)
    end
end

@kernel function _adjoint_edge_halo_kernel!(dst, src, e, q_e, flip, Nc, Hp)
    s, d, k = @index(Global, NTuple)
    @inbounds begin
        s_src = flip ? (Nc + 1 - s) : s
        i_src, j_src = _edge_interior_ij(q_e, d, s_src, Nc, Hp)
        i_dst, j_dst = _edge_halo_ij(e, d, s, Nc, Hp)
        val = dst[i_dst, j_dst, k]
        @atomic src[i_src, j_src, k] += val
        dst[i_dst, j_dst, k] = zero(val)
    end
end

function _adjoint_fill_panel_halos!(lambda_panels::NTuple{6},
                                    mesh::CubedSphereMesh; dir::Int=0)
    Nc, Hp = mesh.Nc, mesh.Hp
    Hp == 0 && return nothing
    if dir in (1, 2)
        N = Nc + 2 * Hp
        @inbounds for p in 1:6
            q = lambda_panels[p]
            backend = get_backend(q)
            kernel! = _adjoint_corner_halo_kernel!(backend, 256)
            kernel!(q, Int32(Nc), Int32(Hp), Int32(N), Int32(dir);
                    ndrange=(Hp, Hp, size(q, 3)))
            synchronize(backend)
        end
    end

    conn = mesh.connectivity
    @inbounds for p in 1:6
        for e in 1:4
            nb = conn.neighbors[p][e]
            q_e = reciprocal_edge(conn, p, e)
            dst = lambda_panels[p]
            src = lambda_panels[nb.panel]
            backend = get_backend(dst)
            kernel! = _adjoint_edge_halo_kernel!(backend, 256)
            kernel!(dst, src, Int32(e), Int32(q_e), nb.orientation >= 2,
                    Int32(Nc), Int32(Hp);
                    ndrange=(Nc, Hp, size(dst, 3)))
            synchronize(backend)
        end
    end
    return nothing
end

@kernel function _seed_layer_objective_kernel!(lambda, @Const(m), value, i, j, k)
    _ = @index(Global)
    @inbounds lambda[i, j, k] = value / max(m[i, j, k], eps(eltype(lambda)))
end

@kernel function _seed_column_objective_kernel!(lambda, @Const(m), denom, i, j, Hp)
    k = @index(Global, Linear)
    @inbounds lambda[i, j, k] = one(eltype(lambda)) / denom
end

function _seed_objective!(lambda_panels, obj::CSLayerMeanObjective, final_m,
                          mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    p = obj.panel
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    backend = get_backend(lambda_panels[p])
    kernel! = _seed_layer_objective_kernel!(backend, 1)
    kernel!(lambda_panels[p], final_m[p], one(FT), Int32(ii), Int32(jj), Int32(obj.level);
            ndrange=1)
    synchronize(backend)
    return nothing
end

function _seed_objective!(lambda_panels, obj::CSColumnMeanObjective, final_m,
                          mesh::CubedSphereMesh)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        fill!(lambda_panels[p], zero(FT))
    end
    p = obj.panel
    ii = mesh.Hp + obj.i
    jj = mesh.Hp + obj.j
    denom = sum(@view final_m[p][ii, jj, :])
    backend = get_backend(lambda_panels[p])
    kernel! = _seed_column_objective_kernel!(backend, 256)
    kernel!(lambda_panels[p], final_m[p], FT(denom), Int32(ii), Int32(jj), Int32(mesh.Hp);
            ndrange=size(final_m[p], 3))
    synchronize(backend)
    return nothing
end

function _seed_objective!(lambda_panels, ::CSSeedObjective, final_m,
                          mesh::CubedSphereMesh)
    throw(ArgumentError(
        "`CSSeedObjective` is reserved for explicit final adjoint seeds; " *
        "use `cs_surface_emission_footprint_from_seed(final_adjoint_rm, ...)`"))
end

@kernel function _accumulate_surface_footprint_kernel!(footprint, @Const(lambda), dt, Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds footprint[i, j] = dt * lambda[i + Hp, j + Hp, Nz]
end

function _accumulate_surface_footprint!(footprint, lambda_panels, dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    @inbounds for p in 1:6
        backend = get_backend(lambda_panels[p])
        kernel! = _accumulate_surface_footprint_kernel!(backend, (16, 16))
        kernel!(footprint[p], lambda_panels[p], eltype(lambda_panels[p])(dt), Int32(Hp), Int32(Nz);
                ndrange=(mesh.Nc, mesh.Nc))
        synchronize(backend)
    end
    return nothing
end

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

@inline function _adjoint_diffusion_time(::Type{FT}, meteo) where FT
    return meteo === nothing ? zero(FT) : FT(current_time(meteo))
end

@kernel function _vertical_diffusion_cs_single_adjoint_kernel!(
    lambda, @Const(air_mass), kz_field, @Const(dz), w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        w_prev = zero(FT)
        g_prev = zero(FT)

        for k in 1:Nz
            Kz_k = field_value(kz_field, (ii, jj, k))
            dz_k = dz[ii, jj, k]

            D_above = zero(FT)
            D_below = zero(FT)
            a_T = zero(FT)
            c_T = zero(FT)

            if k > 1
                Kz_prev = field_value(kz_field, (ii, jj, k - 1))
                dz_prev = dz[ii, jj, k - 1]
                Kz_above = (Kz_prev + Kz_k) / FT(2)
                dz_above = (dz_prev + dz_k) / FT(2)
                D_above = Kz_above / (dz_k * dz_above)
                a_T = -dt_ft * Kz_above / (dz_prev * dz_above)
            end

            if k < Nz
                Kz_next = field_value(kz_field, (ii, jj, k + 1))
                dz_next = dz[ii, jj, k + 1]
                Kz_below = (Kz_k + Kz_next) / FT(2)
                dz_below = (dz_k + dz_next) / FT(2)
                D_below = Kz_below / (dz_k * dz_below)
                c_T = -dt_ft * Kz_below / (dz_next * dz_below)
            end

            b_T = one(FT) + dt_ft * (D_above + D_below)
            m_k = air_mass[i, j, k]
            d_k = m_k > zero(FT) ? m_k * lambda[i, j, k] : zero(FT)

            if k == 1
                denom = b_T
                w_k = c_T / denom
                g_k = d_k / denom
            else
                denom = b_T - a_T * w_prev
                w_k = c_T / denom
                g_k = (d_k - a_T * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            lambda[i, j, k] = g_k

            if k < Nz
                w_prev = w_k
                g_prev = g_k
            end
        end

        for k in (Nz - 1):-1:1
            lambda[i, j, k] = lambda[i, j, k] -
                              w_scratch[ii, jj, k] * lambda[i, j, k + 1]
        end

        for k in 1:Nz
            m_k = air_mass[i, j, k]
            lambda[i, j, k] = m_k > zero(FT) ? lambda[i, j, k] / m_k : zero(FT)
        end
    end
end

function _require_cs_diffusion_workspace(workspace)
    workspace === nothing && throw(ArgumentError(
        "CS adjoint diffusion requires a workspace with panel-native " *
        "`w_scratch` and `dz_scratch`; pass the transport CSAdvectionWorkspace"))
    hasproperty(workspace, :w_scratch) && hasproperty(workspace, :dz_scratch) ||
        throw(ArgumentError(
            "CS adjoint diffusion requires a workspace with panel-native " *
            "`w_scratch` and `dz_scratch` tuples"))
    w_scratch = getproperty(workspace, :w_scratch)
    dz_scratch = getproperty(workspace, :dz_scratch)
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("CS adjoint diffusion workspace must provide 6 panel scratch arrays"))
    return w_scratch, dz_scratch
end

function _diffusion_sequence_at(value, step::Int, nsteps::Int,
                                name::AbstractString)
    if value isa AbstractVector
        length(value) == nsteps || throw(ArgumentError(
            "$name length $(length(value)) does not match nsteps $nsteps"))
        return value[step]
    else
        return value
    end
end

function _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace,
                                       nsteps::Int)
    for step in 1:nsteps
        op = _diffusion_sequence_at(diffusion_op, step, nsteps, "diffusion_op")
        if !(op isa NoDiffusion)
            ws = _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                        "diffusion_workspace")
            _require_cs_diffusion_workspace(ws)
        end
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m, ::NoDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCubedSphereField{FT}}
    w_scratch, dz_scratch = _require_cs_diffusion_workspace(workspace)
    update_field!(op.kz_field, _adjoint_diffusion_time(FT, meteo))

    Hp = mesh.Hp
    Nc = mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        size(panel_lambda) == size(panel_m) || throw(DimensionMismatch(
            "adjoint tracer panel $p shape $(size(panel_lambda)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nz = size(panel_lambda, 3)
        expected = (Nc, Nc, Nz)
        size(w_scratch[p]) == size(dz_scratch[p]) ||
            throw(DimensionMismatch("CS adjoint diffusion w_scratch and dz_scratch sizes differ on panel $p"))
        size(w_scratch[p]) == expected ||
            throw(DimensionMismatch(
                "CS adjoint diffusion workspace panel $p has shape $(size(w_scratch[p])); " *
                "expected $expected"))

        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
                FT(dt), Nz, Hp; ndrange = (Nc, Nc))
        synchronize(backend)
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m,
                                      op::ImplicitVerticalDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    throw(ArgumentError(
        "CS adjoint diffusion requires `ImplicitVerticalDiffusion` with a " *
        "`CubedSphereField` Kz field; got $(typeof(op.kz_field))"))
end

function _tm5_solve_vector!(rm_col, conv1, pivots, Nz::Integer;
                            icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        for k in k_lo:Nz
            piv = pivots[k]
            if piv != k
                tmp = rm_col[k]
                rm_col[k] = rm_col[piv]
                rm_col[piv] = tmp
            end
        end
        for k in k_lo:Nz
            s = rm_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s
        end
        for k in Nz:-1:k_lo
            s = rm_col[k]
            for j in (k + 1):Nz
                s -= conv1[k, j] * rm_col[j]
            end
            rm_col[k] = s / conv1[k, k]
        end
    end
    return nothing
end

function _tm5_solve_vector_transpose!(lambda_col, conv1, pivots, Nz::Integer;
                                      icltop_eff::Integer = 1)
    Nz == 0 && return nothing
    k_lo = max(Int(icltop_eff), 1)
    @inbounds begin
        # U' z = lambda, where U is stored in the upper triangle.
        for k in k_lo:Nz
            s = lambda_col[k]
            for j in k_lo:(k - 1)
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s / conv1[k, k]
        end
        # L' y = z, where L is unit diagonal and stored below the diagonal.
        for k in Nz:-1:k_lo
            s = lambda_col[k]
            for j in (k + 1):Nz
                s -= conv1[j, k] * lambda_col[j]
            end
            lambda_col[k] = s
        end
        # Forward solve applies pivots in ascending order; the transpose
        # applies the inverse permutation, so replay swaps in reverse.
        for k in Nz:-1:k_lo
            piv = pivots[k]
            if piv != k
                tmp = lambda_col[k]
                lambda_col[k] = lambda_col[piv]
                lambda_col[piv] = tmp
            end
        end
    end
    return nothing
end

@inline function _tm5_effective_cloud_top(icltop, icllfs)
    return min(Int(icllfs), max(Int(icltop), 2) - 1)
end

function _tm5_solve_column_vector!(rm_col, m_col,
                                   entu_col, detu_col, entd_col, detd_col,
                                   conv1_buf, pivots_buf, cloud_dims, dt;
                                   cell_area = one(eltype(rm_col)),
                                   f_buf = conv1_buf,
                                   amu_buf,
                                   amd_buf)
    FT = eltype(rm_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_lu!(conv1_buf, pivots_buf, Nz; icltop_eff = icltop_eff)
    _tm5_solve_vector!(rm_col, conv1_buf, pivots_buf, Nz;
                       icltop_eff = icltop_eff)
    return nothing
end

function _tm5_solve_column_vector_adjoint!(lambda_col, m_col,
                                           entu_col, detu_col, entd_col, detd_col,
                                           conv1_buf, pivots_buf, cloud_dims, dt;
                                           cell_area = one(eltype(lambda_col)),
                                           f_buf = conv1_buf,
                                           amu_buf,
                                           amd_buf)
    FT = eltype(lambda_col)
    Nz = length(m_col)
    Nz == 0 && return nothing
    icltop, iclbas, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
    cloud_dims[1] = icltop
    cloud_dims[2] = iclbas
    cloud_dims[3] = icllfs
    icltop > Nz && return nothing

    icltop_eff = _tm5_effective_cloud_top(icltop, icllfs)
    _tm5_build_conv1!(conv1_buf,
                      entu_col, detu_col, entd_col, detd_col, m_col,
                      icltop, icllfs, FT(dt), Nz;
                      cell_area = FT(cell_area),
                      f = f_buf, amu = amu_buf, amd = amd_buf)
    _tm5_lu!(conv1_buf, pivots_buf, Nz; icltop_eff = icltop_eff)
    _tm5_solve_vector_transpose!(lambda_col, conv1_buf, pivots_buf, Nz;
                                 icltop_eff = icltop_eff)
    return nothing
end

@kernel function _tm5_cs_panel_column_single_kernel!(
    q_raw_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        rm_col = @view q_raw_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector!(
            rm_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

@kernel function _tm5_cs_panel_column_adjoint_kernel!(
    lambda_panel, @Const(air_mass_panel),
    @Const(entu_panel), @Const(detu_panel),
    @Const(entd_panel), @Const(detd_panel),
    @Const(cell_areas_panel),
    conv1_panel, pivots_panel, cloud_panel,
    f_panel, amu_panel, amd_panel,
    Hp::Int, tile_offset::Int, Nc::Int, dt)
    t = @index(Global)
    c_global = tile_offset + t
    c1 = ((c_global - 1) % Nc) + 1
    c2 = ((c_global - 1) ÷ Nc) + 1
    i = c1 + Hp
    j = c2 + Hp
    @inbounds begin
        lambda_col = @view lambda_panel[i, j, :]
        m_col = @view air_mass_panel[i, j, :]
        entu_col = @view entu_panel[c1, c2, :]
        detu_col = @view detu_panel[c1, c2, :]
        entd_col = @view entd_panel[c1, c2, :]
        detd_col = @view detd_panel[c1, c2, :]
        conv1_col = @view conv1_panel[:, :, t]
        pivots_col = @view pivots_panel[:, t]
        cloud_col = @view cloud_panel[:, t]
        f_col = @view f_panel[:, :, t]
        amu_col = @view amu_panel[:, t]
        amd_col = @view amd_panel[:, t]
        _tm5_solve_column_vector_adjoint!(
            lambda_col, m_col, entu_col, detu_col, entd_col, detd_col,
            conv1_col, pivots_col, cloud_col, dt;
            cell_area = cell_areas_panel[c1, c2],
            f_buf = f_col, amu_buf = amu_col, amd_buf = amd_col)
    end
end

@inline function _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                     i, j, k, ::Val{true})
    return dtrain_panel[i, j, k]
end

@inline function _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                     i, j, k, ::Val{false})
    FT = eltype(cmfmc_panel)
    return max(zero(FT), cmfmc_panel[i, j, k + 1] - cmfmc_panel[i, j, k])
end

@inline function _cmfmc_cloud_base(cmfmc_panel, i, j, Nz::Int, tiny)
    cldbase_k = 0
    @inbounds for k in 1:Nz
        cmfmc_bot_k = cmfmc_panel[i, j, k + 1]
        if abs(cmfmc_bot_k) > tiny
            cldbase_k = k
            break
        end
    end
    return cldbase_k
end

@kernel function _cmfmc_cs_panel_column_single_kernel!(
    rm_panel,
    @Const(air_mass_panel),
    @Const(cmfmc_panel),
    @Const(dtrain_panel),
    @Const(cell_areas_panel),
    qc_scratch_panel,
    Nz::Int,
    dt,
    Hp::Int,
    ::Val{has_dtrain}) where has_dtrain
    i, j = @index(Global, NTuple)
    FT = eltype(rm_panel)
    tiny = FT(1e-30)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas_panel[i, j])
    dt_ft = FT(dt)

    @inbounds begin
        cldbase_k = _cmfmc_cloud_base(cmfmc_panel, i, j, Nz, tiny)
        if cldbase_k != 0
            if cldbase_k < Nz
                m_cb = air_mass_panel[ii, jj, cldbase_k]
                q_cldbase = m_cb > tiny ? rm_panel[ii, jj, cldbase_k] / m_cb : zero(FT)
                cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
                if cmfmc_at_cldbase > tiny
                    qb_num = zero(FT)
                    mb = zero(FT)
                    for k in (cldbase_k + 1):Nz
                        m_k = air_mass_panel[ii, jj, k]
                        q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                        qb_num += q_k * m_k
                        mb += m_k
                    end
                    if mb > zero(FT)
                        qb = qb_num / mb
                        qc_mixed = (mb * qb + cmfmc_at_cldbase * q_cldbase * dt_ft) /
                                   (mb + cmfmc_at_cldbase * dt_ft)
                        for k in (cldbase_k + 1):Nz
                            rm_panel[ii, jj, k] = qc_mixed * air_mass_panel[ii, jj, k]
                        end
                    end
                end
            end

            qc_below = zero(FT)
            for k in Nz:-1:1
                m_k = air_mass_panel[ii, jj, k]
                q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                cmfmc_bot = k < Nz ? cmfmc_panel[i, j, k + 1] : zero(FT)
                cmfmc_top = cmfmc_panel[i, j, k]
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                cmout = cmfmc_top + dtrain_k
                cmfmc_bot_eff = min(cmfmc_bot, cmout)
                entrn = cmout - cmfmc_bot_eff
                qc = cmout > tiny ?
                     (cmfmc_bot_eff * qc_below + entrn * q_k) / cmout :
                     q_k
                qc_scratch_panel[ii, jj, k] = qc
                qc_below = qc
            end

            q_env_prev = zero(FT)
            for k in 1:Nz
                m_k = air_mass_panel[ii, jj, k]
                q_k = m_k > tiny ? rm_panel[ii, jj, k] / m_k : zero(FT)
                bmass = m_k / cell_area
                cmfmc_top = cmfmc_panel[i, j, k]
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                qc_post = qc_scratch_panel[ii, jj, k]
                q_new = if k > 1 && bmass > tiny
                    q_k + (dt_ft / bmass) *
                          (cmfmc_top * (q_env_prev - q_k) +
                           dtrain_k * (qc_post - q_k))
                elseif bmass > tiny
                    q_k + (dt_ft / bmass) * dtrain_k * (qc_post - q_k)
                else
                    q_k
                end
                q_env_prev = q_k
                rm_panel[ii, jj, k] = q_new * m_k
            end
        end
    end
end

@kernel function _cmfmc_cs_panel_column_single_adjoint_kernel!(
    lambda_panel,
    @Const(air_mass_panel),
    @Const(cmfmc_panel),
    @Const(dtrain_panel),
    @Const(cell_areas_panel),
    lambda_qc_panel,
    Nz::Int,
    dt,
    Hp::Int,
    ::Val{has_dtrain}) where has_dtrain
    i, j = @index(Global, NTuple)
    FT = eltype(lambda_panel)
    tiny = FT(1e-30)
    ii = i + Hp
    jj = j + Hp
    cell_area = FT(cell_areas_panel[i, j])
    dt_ft = FT(dt)

    @inbounds begin
        for k in 1:Nz
            lambda_qc_panel[ii, jj, k] = zero(FT)
        end

        # Transpose the top-to-bottom environment tendency pass.
        for k in 1:Nz
            m_k = air_mass_panel[ii, jj, k]
            lambda_out = lambda_panel[ii, jj, k]
            lambda_panel[ii, jj, k] = zero(FT)
            if m_k > tiny
                lambda_qnew = lambda_out * m_k
                bmass = m_k / cell_area
                dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                               i, j, k, Val(has_dtrain))
                if bmass > tiny
                    alpha = dt_ft / bmass
                    if k > 1
                        cmfmc_top = cmfmc_panel[i, j, k]
                        lambda_panel[ii, jj, k] +=
                            lambda_qnew *
                            (one(FT) - alpha * (cmfmc_top + dtrain_k)) / m_k
                        m_prev = air_mass_panel[ii, jj, k - 1]
                        if m_prev > tiny
                            lambda_panel[ii, jj, k - 1] +=
                                lambda_qnew * alpha * cmfmc_top / m_prev
                        end
                        lambda_qc_panel[ii, jj, k] =
                            lambda_qnew * alpha * dtrain_k
                    else
                        lambda_panel[ii, jj, k] +=
                            lambda_qnew * (one(FT) - alpha * dtrain_k) / m_k
                        lambda_qc_panel[ii, jj, k] =
                            lambda_qnew * alpha * dtrain_k
                    end
                else
                    lambda_panel[ii, jj, k] += lambda_qnew / m_k
                end
            end
        end

        # Transpose the bottom-to-top updraft recurrence.
        for k in 1:Nz
            lambda_qc = lambda_qc_panel[ii, jj, k]
            cmfmc_bot = k < Nz ? cmfmc_panel[i, j, k + 1] : zero(FT)
            cmfmc_top = cmfmc_panel[i, j, k]
            dtrain_k = _cmfmc_panel_dtrain(cmfmc_panel, dtrain_panel,
                                           i, j, k, Val(has_dtrain))
            cmout = cmfmc_top + dtrain_k
            cmfmc_bot_eff = min(cmfmc_bot, cmout)
            entrn = cmout - cmfmc_bot_eff
            coeff_below = cmout > tiny ? cmfmc_bot_eff / cmout : zero(FT)
            coeff_q = cmout > tiny ? entrn / cmout : one(FT)
            m_k = air_mass_panel[ii, jj, k]
            if m_k > tiny
                lambda_panel[ii, jj, k] += lambda_qc * coeff_q / m_k
            end
            if k < Nz
                lambda_qc_panel[ii, jj, k + 1] += lambda_qc * coeff_below
            end
        end

        # Transpose the optional well-mixed sub-cloud preprocessing.
        cldbase_k = _cmfmc_cloud_base(cmfmc_panel, i, j, Nz, tiny)
        if cldbase_k != 0 && cldbase_k < Nz
            cmfmc_at_cldbase = cmfmc_panel[i, j, cldbase_k + 1]
            if cmfmc_at_cldbase > tiny
                mb = zero(FT)
                lambda_mixed = zero(FT)
                for k in (cldbase_k + 1):Nz
                    m_k = air_mass_panel[ii, jj, k]
                    mb += m_k
                    lambda_mixed += lambda_panel[ii, jj, k] * m_k
                end
                if mb > zero(FT)
                    gamma = cmfmc_at_cldbase * dt_ft
                    denom = mb + gamma
                    coeff_sub = lambda_mixed / denom
                    for k in (cldbase_k + 1):Nz
                        m_k = air_mass_panel[ii, jj, k]
                        lambda_panel[ii, jj, k] = m_k > tiny ? coeff_sub : zero(FT)
                    end
                    m_cb = air_mass_panel[ii, jj, cldbase_k]
                    if m_cb > tiny
                        lambda_panel[ii, jj, cldbase_k] +=
                            lambda_mixed * gamma / denom / m_cb
                    end
                end
            end
        end
    end
end

function _require_cmfmc_convection_workspace(workspace)
    workspace isa CMFMCWorkspace || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `CMFMCWorkspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_cmfmc_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS CMFMC adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.cmfmc === nothing && throw(ArgumentError(
        "CS CMFMC adjoint convection requires `forcing.cmfmc` panel fields"))
    return forcing.cmfmc, forcing.dtrain
end

function _require_tm5_convection_workspace(workspace)
    workspace isa TM5Workspace || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `TM5Workspace`; got $(typeof(workspace))"))
    workspace.cell_metrics === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `workspace.cell_metrics` with per-panel cell areas"))
    return workspace
end

function _assert_tm5_adjoint_forcing(forcing)
    forcing isa ConvectionForcing || throw(ArgumentError(
        "CS TM5 adjoint convection requires a `ConvectionForcing`; got $(typeof(forcing))"))
    forcing.tm5_fields === nothing && throw(ArgumentError(
        "CS TM5 adjoint convection requires `forcing.tm5_fields` (:entu, :detu, :entd, :detd)"))
    return forcing.tm5_fields
end

_require_cs_convection_workspace(::NoConvection, workspace) = nothing
_require_cs_convection_workspace(::CMFMCConvection, workspace) =
    _require_cmfmc_convection_workspace(workspace)
_require_cs_convection_workspace(::TM5Convection, workspace) =
    _require_tm5_convection_workspace(workspace)
function _require_cs_convection_workspace(op, workspace)
    throw(ArgumentError("CS adjoint footprint supports `NoConvection`, " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
    has_dtrain = dtrain !== nothing
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    backend = get_backend(panels_rm[1])
    kernel! = _cmfmc_cs_panel_column_single_kernel!(backend, (16, 16))
    FT = eltype(panels_rm[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            dtrain_panel = has_dtrain ? dtrain[p] : cmfmc[p]
            kernel!(panels_rm[p], panels_m[p], cmfmc[p], dtrain_panel,
                    cell_areas[p], workspace.qc_scratch[p],
                    Nz, sdt, Hp, Val(has_dtrain);
                    ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       ::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    B = size(workspace.conv1, 3)
    backend = get_backend(panels_rm[1])
    kernel! = _tm5_cs_panel_column_single_kernel!(backend)
    FT = eltype(panels_rm[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(panels_rm[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_forward!(panels_rm, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint forward helper supports `NoConvection` " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::NoConvection, dt, workspace,
                                       mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::CMFMCConvection, dt,
                                       workspace::CMFMCWorkspace,
                                       mesh::CubedSphereMesh)
    cmfmc, dtrain = _assert_cmfmc_adjoint_forcing(forcing)
    _require_cmfmc_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    invalidate_cmfmc_cache!(workspace)
    n_sub = _get_or_compute_n_sub!(workspace, cmfmc, panels_m, cell_areas, dt)
    has_dtrain = dtrain !== nothing
    Nc = mesh.Nc
    Hp = mesh.Hp
    Nz = size(lambda_panels[1], 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _cmfmc_cs_panel_column_single_adjoint_kernel!(backend, (16, 16))
    FT = eltype(lambda_panels[1])
    sdt = FT(dt) / FT(n_sub)
    @inbounds for _ in 1:n_sub
        for p in 1:6
            dtrain_panel = has_dtrain ? dtrain[p] : cmfmc[p]
            kernel!(lambda_panels[p], panels_m[p], cmfmc[p], dtrain_panel,
                    cell_areas[p], workspace.qc_scratch[p],
                    Nz, sdt, Hp, Val(has_dtrain);
                    ndrange = (Nc, Nc))
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       ::TM5Convection, dt,
                                       workspace::TM5Workspace,
                                       mesh::CubedSphereMesh)
    tm5 = _assert_tm5_adjoint_forcing(forcing)
    _require_tm5_convection_workspace(workspace)
    cell_areas = workspace.cell_metrics
    Nc = mesh.Nc
    Hp = mesh.Hp
    N_total = Nc * Nc
    B = size(workspace.conv1, 3)
    backend = get_backend(lambda_panels[1])
    kernel! = _tm5_cs_panel_column_adjoint_kernel!(backend)
    FT = eltype(lambda_panels[1])
    @inbounds for p in 1:6
        for tile_off in 0:B:(N_total - 1)
            n = min(B, N_total - tile_off)
            kernel!(lambda_panels[p], panels_m[p],
                    tm5.entu[p], tm5.detu[p], tm5.entd[p], tm5.detd[p],
                    cell_areas[p],
                    workspace.conv1, workspace.pivots, workspace.cloud_dims,
                    workspace.f_scratch,
                    workspace.amu_scratch, workspace.amd_scratch,
                    Hp, Int(tile_off), Nc, FT(dt);
                    ndrange = n)
        end
    end
    synchronize(backend)
    return nothing
end

function _apply_cs_convection_adjoint!(lambda_panels, panels_m, forcing,
                                       op, dt, workspace,
                                       mesh::CubedSphereMesh)
    throw(ArgumentError("CS adjoint footprint reverse helper supports `NoConvection` " *
                        "`CMFMCConvection`, and `TM5Convection`; got $(typeof(op))"))
end

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

struct _CSTapeCounts
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
end

function _tape_byte_estimate(panels_m0,
                             panels_am_steps,
                             panels_bm_steps,
                             panels_cm_steps,
                             mesh::CubedSphereMesh,
                             scheme::CSAdjointSupportedScheme;
                             cfl_limit = 0.95,
                             diffusion_op = NoDiffusion(),
                             convection_op = NoConvection())
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)

    sweep_records = 0
    halo_records = 0
    midpoint_records = 0
    diffusion_records = 0
    diffusion_state_records = 0
    convection_records = 0

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)

        sweep_records += 2n_x + 2n_y + 2n_z
        halo_records += 2n_x + 2n_y + 2
        midpoint_records += 1
        if !(_diffusion_sequence_at(diffusion_op, step, nsteps,
                                    "diffusion_op") isa NoDiffusion)
            diffusion_records += 2
            diffusion_state_records += 1
        end
        convection_op isa NoConvection || (convection_records += 1)

        fs_x = one(FT) / FT(n_x)
        fs_y = one(FT) / FT(n_y)
        fs_z = one(FT) / FT(n_z)

        for _ in 1:n_x
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
        end
        for _ in 1:n_y
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
        end
        for _ in 1:n_z
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end
        for _ in 1:n_z
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end
        fill_panel_halos!(panels_m, mesh; dir=2)
        for _ in 1:n_y
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
        end
        fill_panel_halos!(panels_m, mesh; dir=1)
        for _ in 1:n_x
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
        end
    end

    sweep_state_records = scheme isa CSAdjointNonlinearScheme ? 2sweep_records : sweep_records
    state_records = sweep_state_records + diffusion_state_records + convection_records
    total_records = state_records + halo_records + midpoint_records
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    return CSTapeByteEstimate(nsteps, sweep_records, halo_records,
                              midpoint_records, diffusion_records,
                              convection_records, state_records,
                              total_records, bytes_per_state,
                              state_records * bytes_per_state)
end

cs_tape_byte_estimate(args...; kwargs...) =
    _tape_byte_estimate(args...; kwargs...)

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

function _record_sweep!(ops, direction::Symbol, scheme::CSAdjointLinearScheme,
                        panels_m, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              nothing,
                              panels_flux, flux_scale))
    return nothing
end

function _record_sweep!(ops, direction::Symbol, scheme::CSAdjointNonlinearScheme,
                        panels_m, panels_rm, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              _stage_panels(tape_storage, panels_rm),
                              panels_flux, flux_scale))
    return nothing
end

function _record_cs_mass_tape(panels_m0,
                              panels_am_steps,
                              panels_bm_steps,
                              panels_cm_steps,
                              mesh::CubedSphereMesh,
                              scheme::CSAdjointLinearScheme;
                              cfl_limit,
                              flux_scale,
                              dt,
                              diffusion_op = NoDiffusion(),
                              diffusion_workspace = nothing,
                              convection_op = NoConvection(),
                              convection_forcing = nothing,
                              tape_storage = :device)
    FT = eltype(panels_m0[1])
    storage = _tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)
        fs_x = fs / FT(n_x)
        fs_y = fs / FT(n_y)
        fs_z = fs / FT(n_z)

        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            push!(ops, _CSMidpointRecord(step))
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            panels_m_midpoint = _stage_panels(storage, panels_m)
            half_dt = FT(dt) / FT(2)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            push!(ops, _CSMidpointRecord(step))
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        fill_panel_halos!(panels_m, mesh; dir=2)
        push!(ops, _CSHaloRecord(2))
        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        fill_panel_halos!(panels_m, mesh; dir=1)
        push!(ops, _CSHaloRecord(1))
        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                           _stage_panels(storage, panels_m),
                                           FT(dt)))
        end
    end

    return ops, panels_m
end

function _record_cs_tracer_tape(panels_rm0,
                                panels_m0,
                                panels_am_steps,
                                panels_bm_steps,
                                panels_cm_steps,
                                mesh::CubedSphereMesh,
                                scheme::CSAdjointNonlinearScheme;
                                cfl_limit,
                                flux_scale,
                                dt,
                                base_emission_rates = nothing,
                                diffusion_op = NoDiffusion(),
                                diffusion_workspace = nothing,
                                diffusion_meteo = nothing,
                                convection_op = NoConvection(),
                                convection_forcing = nothing,
                                convection_workspace = nothing,
                                tape_storage = :device)
    storage = _tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir=0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    FT = eltype(panels_m[1])
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)
    dt_ft = FT(dt)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)
        fs_x = fs / FT(n_x)
        fs_y = fs / FT(n_y)
        fs_z = fs / FT(n_z)

        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_rm, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            push!(ops, _CSMidpointRecord(step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            panels_m_midpoint = _stage_panels(storage, panels_m)
            half_dt = dt_ft / FT(2)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
            push!(ops, _CSMidpointRecord(step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_rm, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        fill_panel_halos!(panels_rm, mesh; dir=2)
        fill_panel_halos!(panels_m, mesh; dir=2)
        push!(ops, _CSHaloRecord(2))
        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
        push!(ops, _CSHaloRecord(1))
        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                           _stage_panels(storage, panels_m),
                                           dt_ft))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt_ft,
                                          convection_workspace, mesh)
        end
    end

    return ops, panels_m
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinearScheme;
                                 base_emission_rates = nothing,
                                 diffusion_meteo = nothing,
                                 convection_workspace = nothing,
                                 kwargs...)
    return _record_cs_mass_tape(panels_m0,
                                panels_am_steps, panels_bm_steps, panels_cm_steps,
                                mesh, scheme; kwargs...)
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinRoodScheme; kwargs...)
    return _record_cs_linrood_tape(panels_rm0, panels_m0,
                                    panels_am_steps, panels_bm_steps,
                                    panels_cm_steps, mesh, scheme; kwargs...)
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointNonlinearScheme; kwargs...)
    return _record_cs_tracer_tape(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps, panels_cm_steps,
                                  mesh, scheme; kwargs...)
end

function _collect_surface_footprints(lambda_panels, ops, panels_m0,
                                     mesh::CubedSphereMesh,
                                     objective::AbstractCSFootprintObjective,
                                     dt;
                                     diffusion_workspace = nothing,
                                     diffusion_meteo = nothing,
                                     convection_workspace = nothing)
    FT = eltype(lambda_panels[1])
    nsteps = count(op -> op isa _CSMidpointRecord, ops)
    footprints = Vector{typeof(_zero_surface_rates(mesh, panels_m0[1]))}(undef, nsteps)
    @inbounds for step in 1:nsteps
        footprints[step] = _zero_surface_rates(mesh, panels_m0[1])
    end

    ws = CSAdjointWorkspace(mesh, lambda_panels[1])
    for op in Iterators.reverse(ops)
        if op isa _CSSweepRecord
            if op.panels_rm === nothing
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m),
                                       op.panels_flux, op.direction, op.scheme,
                                       mesh, ws, op.flux_scale)
            else
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m),
                                       _tape_panels(op.panels_rm), op.panels_flux,
                                       op.direction, op.scheme, mesh, ws,
                                       op.flux_scale)
            end
        elseif op isa _CSHaloRecord
            _adjoint_fill_panel_halos!(lambda_panels, mesh; dir=op.dir)
        elseif op isa _CSMidpointRecord
            _accumulate_surface_footprint!(footprints[op.step], lambda_panels, dt, mesh)
        elseif op isa _CSDiffusionRecord
            _apply_cs_diffusion_adjoint!(lambda_panels, _tape_panels(op.panels_m), op.op,
                                         op.workspace, op.dt,
                                         diffusion_meteo, mesh)
        elseif op isa _CSConvectionRecord
            _apply_cs_convection_adjoint!(lambda_panels, _tape_panels(op.panels_m),
                                          op.forcing, op.op, op.dt,
                                          convection_workspace, mesh)
        elseif op isa _CSLinRoodHorizRecord
            # Plan 25 Commit 6: LinRood horizontal substep reverse.
            # `lambda_panels` holds the tracer-rm adjoint propagated
            # through the tape. The substep produces an internal
            # `lambda_m_panels` (via the `c = rm / m` chain rule and
            # the donor-cell α denominator), but the
            # cs_surface_emission_footprint design — like the existing
            # `_record_cs_mass_tape` / `_record_cs_tracer_tape` paths —
            # treats meteorology (air mass evolution) as a FIXED
            # tape input. Mass-flux Jacobians are read off the per-
            # record `panels_m` / `panels_am` / `panels_bm` snapshots
            # rather than propagated via dynamic `lambda_m`, so we seed
            # `lambda_m_new = 0` for each substep and discard its
            # output. This matches the existing PPM reverse-loop
            # contract where there is no `lambda_panels_m` at all.
            lambda_m_panels = ntuple(p -> begin
                a = similar(lambda_panels[p]); fill!(a, zero(eltype(a))); a
            end, Val(6))
            _apply_cs_linrood_horizontal_adjoint!(lambda_panels,
                                                  lambda_m_panels, op, mesh)
        else
            throw(ArgumentError("unknown CS adjoint tape operation $(typeof(op))"))
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

function _run_cs_footprint_forward(panels_rm0, panels_m0,
                                   panels_am_steps,
                                   panels_bm_steps,
                                   panels_cm_steps,
                                   mesh::CubedSphereMesh,
                                   objective::AbstractCSFootprintObjective;
                                   scheme = PPMScheme(NoLimiter()),
                                   dt,
                                   cfl_limit = 0.95,
                                   emission_rates = nothing,
                                   perturbation = nothing,
                                   diffusion_op = NoDiffusion(),
                                   diffusion_workspace = nothing,
                                   diffusion_meteo = nothing,
                                   convection_op = NoConvection(),
                                   convection_forcing = nothing,
                                   convection_workspace = nothing)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)

    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])

    if emission_rates !== nothing && length(emission_rates) != nsteps
        throw(ArgumentError("emission_rates length $(length(emission_rates)) does not match nsteps $nsteps"))
    end
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    @inbounds for step in 1:nsteps
        midpoint! = nothing
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        needs_midpoint = !(diffusion_op_step isa NoDiffusion) ||
                         emission_rates !== nothing ||
                         (perturbation !== nothing && perturbation.step == step)
        if needs_midpoint
            midpoint! = () -> begin
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                emission_rates !== nothing &&
                    _add_surface_rates!(panels_rm, emission_rates[step], dt, mesh)
                perturbation !== nothing && perturbation.step == step &&
                    _add_surface_perturbation!(panels_rm, perturbation, dt, mesh)
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                nothing
            end
        end
        if scheme isa LinRoodPPMScheme
            # LinRood uses a 3-phase unsplit horizontal + Z sweep
            # composition; the standard split-sweep `strang_split_cs!`
            # doesn't have face kernels for LinRoodPPMScheme.
            _linrood_run_forward_step!(panels_rm, panels_m,
                panels_am_steps[step], panels_bm_steps[step],
                panels_cm_steps[step], mesh, scheme, ws, midpoint!)
        else
            strang_split_cs!(panels_rm, panels_m,
                             panels_am_steps[step], panels_bm_steps[step], panels_cm_steps[step],
                             mesh, scheme, ws;
                             flux_scale = one(eltype(panels_m[1])),
                             cfl_limit = cfl_limit,
                             midpoint! = midpoint!)
        end
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt,
                                          convection_workspace, mesh)
        end
    end

    return evaluate_objective(objective, panels_rm, panels_m, mesh)
end

function _run_cs_observations_forward(panels_rm0, panels_m0,
                                      panels_am_steps,
                                      panels_bm_steps,
                                      panels_cm_steps,
                                      mesh::CubedSphereMesh,
                                      observations;
                                      scheme = PPMScheme(NoLimiter()),
                                      dt,
                                      cfl_limit = 0.95,
                                      emission_rates = nothing,
                                      diffusion_op = NoDiffusion(),
                                      diffusion_workspace = nothing,
                                      diffusion_meteo = nothing,
                                      convection_op = NoConvection(),
                                      convection_forcing = nothing,
                                      convection_workspace = nothing)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    @inbounds for obs in observations
        1 <= obs.step <= nsteps || throw(ArgumentError(
            "CSObservation step $(obs.step) is outside 1:$nsteps"))
        _validate_objective(obs.objective, mesh, Nz)
    end

    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])

    if emission_rates !== nothing && length(emission_rates) != nsteps
        throw(ArgumentError("emission_rates length $(length(emission_rates)) does not match nsteps $nsteps"))
    end
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    obs_by_step = [Int[] for _ in 1:nsteps]
    @inbounds for idx in eachindex(observations)
        push!(obs_by_step[observations[idx].step], idx)
    end
    FT = eltype(panels_rm0[1])
    simulated = fill(FT(NaN), length(observations))

    @inbounds for step in 1:nsteps
        midpoint! = nothing
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        needs_midpoint = !(diffusion_op_step isa NoDiffusion) ||
                         emission_rates !== nothing
        if needs_midpoint
            midpoint! = () -> begin
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                emission_rates !== nothing &&
                    _add_surface_rates!(panels_rm, emission_rates[step], dt, mesh)
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                nothing
            end
        end
        if scheme isa LinRoodPPMScheme
            # LinRood uses a 3-phase unsplit horizontal + Z sweep
            # composition; the standard split-sweep `strang_split_cs!`
            # doesn't have face kernels for LinRoodPPMScheme.
            _linrood_run_forward_step!(panels_rm, panels_m,
                panels_am_steps[step], panels_bm_steps[step],
                panels_cm_steps[step], mesh, scheme, ws, midpoint!)
        else
            strang_split_cs!(panels_rm, panels_m,
                             panels_am_steps[step], panels_bm_steps[step], panels_cm_steps[step],
                             mesh, scheme, ws;
                             flux_scale = one(eltype(panels_m[1])),
                             cfl_limit = cfl_limit,
                             midpoint! = midpoint!)
        end
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt,
                                          convection_workspace, mesh)
        end
        for obs_idx in obs_by_step[step]
            simulated[obs_idx] = evaluate_objective(
                observations[obs_idx].objective, panels_rm, panels_m, mesh)
        end
    end

    return simulated
end

"""
    run_cs_footprint_forward(..., objective; kwargs...) -> scalar

Run the CS PPM transport path forward and return `objective` at the final
time. Optional `emission_rates[t][panel][i, j]` entries are midpoint surface
emission rates in kg s^-1. If `diffusion_op` is supplied, the helper applies
`V(dt/2) -> emissions -> V(dt/2)` at the control midpoint and requires a
panel-native `diffusion_workspace` with filled `dz_scratch`. If
`convection_op=CMFMCConvection()` or `TM5Convection()` is supplied, the
helper applies the corresponding CS convection column operator after each
transport step.
"""
function run_cs_footprint_forward(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objective::AbstractCSFootprintObjective;
                                  scheme = PPMScheme(NoLimiter()),
                                  dt = one(eltype(panels_rm0[1])),
                                  cfl_limit = 0.95,
                                  emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing)
    FT = eltype(panels_rm0[1])
    return _run_cs_footprint_forward(panels_rm0, panels_m0,
                                     panels_am_steps, panels_bm_steps, panels_cm_steps,
                                     mesh, objective;
                                     scheme = scheme,
                                     dt = FT(dt),
                                     cfl_limit = cfl_limit,
                                     emission_rates = emission_rates,
                                     diffusion_op = diffusion_op,
                                     diffusion_workspace = diffusion_workspace,
                                     diffusion_meteo = diffusion_meteo,
                                     convection_op = convection_op,
                                     convection_forcing = convection_forcing,
                                     convection_workspace = convection_workspace)
end

"""
    cs_surface_emission_footprint(..., objective; kwargs...) -> CSFootprintResult

Generate reverse-mode footprints for a scalar final-time objective with
respect to surface-emission rates at each prior model step.

This is a kernelized prototype VJP generator for tests and diagnostics.
Supported CS split-sweep schemes are `UpwindScheme()`,
`SlopesScheme(NoLimiter())`, `PPMScheme(NoLimiter())`, and monotone
`PPMScheme()`. The limited PPM path stores tracer branch states from the
base trajectory; pass `base_emission_rates` when differentiating around
nonzero surface emissions.
Optional `ImplicitVerticalDiffusion` support transposes the Backward-Euler
column solve in kernels on CPU/GPU and uses the same midpoint placement as
surface-flux runtime transport. Optional `CMFMCConvection` support transposes
the well-mixed sub-cloud, updraft, and tendency passes; optional
`TM5Convection` support replays the same column matrix and applies the
transposed LU solve after each reverse transport step.
"""
function cs_surface_emission_footprint(panels_rm0, panels_m0,
                                       panels_am_steps,
                                       panels_bm_steps,
                                       panels_cm_steps,
                                       mesh::CubedSphereMesh,
                                       objective::AbstractCSFootprintObjective;
                                       scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                       dt = one(eltype(panels_rm0[1])),
                                       epsilon = nothing,
                                       flux_scale = one(eltype(panels_rm0[1])),
                                       cfl_limit = 0.95,
                                       base_emission_rates = nothing,
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
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    ops, final_m = _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                           panels_am_steps, panels_bm_steps,
                                           panels_cm_steps, mesh, scheme;
                                           cfl_limit = cfl_limit,
                                           flux_scale = FT(flux_scale),
                                           dt = dt_ft,
                                           base_emission_rates = base_emission_rates,
                                           diffusion_op = diffusion_op,
                                           diffusion_workspace = diffusion_workspace,
                                           diffusion_meteo = diffusion_meteo,
                                           convection_op = convection_op,
                                           convection_forcing = convection_forcing,
                                           convection_workspace = convection_workspace,
                                           tape_storage = tape_storage)
    lambda_panels = ntuple(p -> begin
        a = similar(final_m[p])
        fill!(a, zero(FT))
        a
    end, 6)
    _seed_objective!(lambda_panels, objective, final_m, mesh)

    return _collect_surface_footprints(lambda_panels, ops, panels_m0, mesh, objective, dt_ft;
                                       diffusion_workspace = diffusion_workspace,
                                       diffusion_meteo = diffusion_meteo,
                                       convection_workspace = convection_workspace)
end

"""
    cs_surface_emission_footprint_from_seed(final_adjoint_rm, panels_m0,
                                            panels_am_steps, panels_bm_steps,
                                            panels_cm_steps, mesh; kwargs...)

General surface-emission footprint entry point. `final_adjoint_rm` is an
`NTuple{6}` of halo-padded adjoint tracer-mass arrays containing
`dJ/drm_final` for any scalar objective or observation operator. The reverse
pass and surface-gradient accumulation use the same CPU/GPU kernels as
`cs_surface_emission_footprint`.
"""
function cs_surface_emission_footprint_from_seed(final_adjoint_rm::NTuple{6},
                                                 panels_m0,
                                                 panels_am_steps,
                                                 panels_bm_steps,
                                                 panels_cm_steps,
                                                 mesh::CubedSphereMesh;
                                                 scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                                 dt = one(eltype(panels_m0[1])),
                                                 flux_scale = one(eltype(panels_m0[1])),
                                                 cfl_limit = 0.95,
                                                 base_panels_rm0 = nothing,
                                                 base_emission_rates = nothing,
                                                 diffusion_op = NoDiffusion(),
                                                 diffusion_workspace = nothing,
                                                 diffusion_meteo = nothing,
                                                 convection_op = NoConvection(),
                                                 convection_forcing = nothing,
                                                 convection_workspace = nothing,
                                                 tape_storage = :device)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_rm0 = base_panels_rm0 === nothing ?
        _zero_panel_tuple_like(panels_m0) :
        base_panels_rm0
    ops, _ = _record_cs_adjoint_tape(tape_rm0, panels_m0,
                                     panels_am_steps, panels_bm_steps,
                                     panels_cm_steps, mesh, scheme;
                                     cfl_limit = cfl_limit,
                                     flux_scale = FT(flux_scale),
                                     dt = FT(dt),
                                     base_emission_rates = base_emission_rates,
                                     diffusion_op = diffusion_op,
                                     diffusion_workspace = diffusion_workspace,
                                     diffusion_meteo = diffusion_meteo,
                                     convection_op = convection_op,
                                     convection_forcing = convection_forcing,
                                     convection_workspace = convection_workspace,
                                     tape_storage = tape_storage)
    lambda_panels = _copy_panel_tuple(final_adjoint_rm)
    return _collect_surface_footprints(lambda_panels, ops, panels_m0, mesh,
                                       CSSeedObjective(), FT(dt);
                                       diffusion_workspace = diffusion_workspace,
                                       diffusion_meteo = diffusion_meteo,
                                       convection_workspace = convection_workspace)
end

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
