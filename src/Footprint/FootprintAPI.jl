# ---------------------------------------------------------------------------
# User-facing surface-emission footprint API.
#
#   * `run_cs_footprint_forward` — forward-only entry returning the
#     scalar value of an objective at final time.
#   * `cs_surface_emission_footprint` — main reverse-mode entry point.
#     Builds the tape via `_record_cs_adjoint_tape`, seeds the adjoint
#     from the objective, walks `_collect_surface_footprints`.
#   * `cs_surface_emission_footprint_from_seed` — variant that takes an
#     explicit final-time adjoint seed (`dJ/drm_final`) instead of
#     constructing it from one of the built-in objectives.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 867-1046 unchanged in
# Plan 26 P0.3c; no semantic change.
# ---------------------------------------------------------------------------

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
                                       tape_storage = :device,
                                       checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    FT = eltype(panels_rm0[1])
    dt_ft = FT(dt)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    _require_checkpoint_supported(scheme, checkpoint)

    if checkpoint isa StrideCheckpoint
        if scheme isa CSAdjointLinearScheme
            return _collect_surface_footprints_stride(
                panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage)
        else
            # CSAdjointNonlinearScheme or CSAdjointLinRoodScheme —
            # both stride drivers take `panels_rm0` and
            # `base_emission_rates` (meaningless to the linear-mass
            # driver) and dispatch on the scheme type at the stride-
            # driver method level. LinRood additionally requires
            # `tape_storage = :device`; the LinRood method validates
            # that up front.
            return _collect_surface_footprints_stride(
                panels_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                base_emission_rates = base_emission_rates,
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage)
        end
    end

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
                                                 tape_storage = :device,
                                                 checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    # Strided checkpointing for the from-seed entry needs a separate
    # driver: it skips the objective-driven seeding and takes the
    # lambda directly. That driver is not in this commit; refuse
    # non-FullCheckpoint until it lands. When that driver lands, also
    # route through `_require_checkpoint_supported` (see
    # `cs_surface_emission_footprint` above) so the scheme-vs-schedule
    # gate stays in one place.
    checkpoint isa FullCheckpoint || throw(ArgumentError(
        "cs_surface_emission_footprint_from_seed does not yet support " *
        "checkpoint=$(checkpoint); the from-seed stride driver is " *
        "deferred to a follow-up commit. Pass checkpoint=FullCheckpoint() " *
        "or use cs_surface_emission_footprint with an objective."))
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
