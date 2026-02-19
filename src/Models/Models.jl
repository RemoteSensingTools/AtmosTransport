"""
    Models

Top-level model type that assembles grid, fields, physics operators,
met data, I/O, and time stepping into a single runnable object.
"""
module Models

using ..Architectures: AbstractArchitecture, architecture
using ..Grids: AbstractGrid
using ..Fields: Field, Center, TracerFields
using ..Advection: AbstractAdvectionScheme
using ..Convection: AbstractConvection
using ..Diffusion: AbstractDiffusion
using ..Chemistry: AbstractChemistry, NoChemistry
using ..TimeSteppers: AbstractTimeStepper, OperatorSplittingTimeStepper, Clock
using Dates
using ..IO: AbstractMetData, AbstractOutputWriter, MetDataSource
using ..IO: read_met!, prepare_met_for_physics
using ..Callbacks: AbstractCallback
using DocStringExtensions

export AbstractModel, TransportModel, run!, update_met_data!

"""
$(TYPEDEF)

Supertype for all model types.
"""
abstract type AbstractModel{Arch} end

"""
$(TYPEDEF)

The main atmospheric transport model.

$(FIELDS)
"""
struct TransportModel{Arch, G, Tr, ATr, M, TS, OW, CB} <: AbstractModel{Arch}
    "CPU or GPU"
    architecture   :: Arch
    "the computational grid"
    grid           :: G
    "NamedTuple of tracer Fields"
    tracers        :: Tr
    "NamedTuple of adjoint tracer Fields (nothing if not doing adjoint)"
    adj_tracers    :: ATr
    "meteorological data reader"
    met_data       :: M
    "simulation clock"
    clock          :: Clock
    "time-stepping strategy"
    timestepper    :: TS
    "vector of output writers"
    output_writers :: OW
    "vector of callbacks"
    callbacks      :: CB
end

"""
$(TYPEDSIGNATURES)

Construct a `TransportModel` from components.
"""
function TransportModel(;
        grid           :: AbstractGrid{FT},
        tracers        :: NTuple{N, Symbol} where N,
        met_data       :: AbstractMetData = nothing,
        advection      :: AbstractAdvectionScheme,
        convection     :: AbstractConvection,
        diffusion      :: AbstractDiffusion,
        chemistry      :: AbstractChemistry = NoChemistry(),
        Δt             :: Real = 10800.0,
        output_writers :: Vector = AbstractOutputWriter[],
        callbacks      :: Vector = AbstractCallback[],
        adjoint        :: Bool = false) where {FT}

    arch = architecture(grid)
    tracer_fields = TracerFields(tracers, grid)

    adj_tracer_fields = if adjoint
        TracerFields(tracers, grid)
    else
        nothing
    end

    Δt_ft = FT(Δt)
    ts = OperatorSplittingTimeStepper(;
        advection  = advection,
        convection = convection,
        diffusion  = diffusion,
        chemistry  = chemistry,
        Δt_outer   = Δt_ft)

    clock = Clock(FT; Δt = Δt_ft)

    return TransportModel(arch, grid, tracer_fields, adj_tracer_fields,
                          met_data, clock, ts, output_writers, callbacks)
end

"""
$(SIGNATURES)

Read meteorological data for `time` from `met_source` and prepare the
staggered velocity fields for the physics operators. Updates `model.met_data`.

Returns the prepared met fields NamedTuple.
"""
function update_met_data!(model::TransportModel, met_source::MetDataSource, time)
    read_met!(met_source, time)
    physics_fields = prepare_met_for_physics(met_source, model.grid)
    # Store in model (requires mutable field — use a Ref or rebuild)
    return physics_fields
end

"""
$(TYPEDSIGNATURES)

Run the forward model from `t_start` to `t_end`.

At each `met_update_interval`, reads and prepares new met data from
`met_source`. Between met updates, steps the model forward with `Δt`.
"""
function run!(model::TransportModel, met_source::MetDataSource,
              t_start, t_end;
              Δt = model.timestepper.Δt_outer,
              met_update_interval = Δt,
              callback = nothing,
              verbose::Bool = true)

    using_datetime = t_start isa Dates.DateTime
    t_current = using_datetime ? t_start : Float64(t_start)
    t_final   = using_datetime ? t_end : Float64(t_end)

    step = 0
    next_met_update = t_current

    while t_current < t_final
        # Read new met data if needed
        if t_current >= next_met_update
            if verbose
                @info "Reading met data at t = $t_current"
            end
            read_met!(met_source, t_current)
            physics_fields = prepare_met_for_physics(met_source, model.grid)

            # Rebuild the model with updated met_data
            # (TransportModel is immutable, so we update via a local binding)
            model = TransportModel(
                model.architecture, model.grid, model.tracers,
                model.adj_tracers, physics_fields, model.clock,
                model.timestepper, model.output_writers, model.callbacks)

            next_met_update = if using_datetime
                t_current + Dates.Second(round(Int, met_update_interval))
            else
                t_current + met_update_interval
            end
        end

        # Determine actual Δt for this step (don't overshoot)
        dt_actual = if using_datetime
            remaining = Dates.value(t_final - t_current) / 1000.0
            min(Float64(Δt), remaining)
        else
            min(Float64(Δt), Float64(t_final - t_current))
        end

        TimeSteppers.time_step!(model, dt_actual)
        step += 1

        t_current = if using_datetime
            t_current + Dates.Second(round(Int, dt_actual))
        else
            t_current + dt_actual
        end

        if callback !== nothing
            callback(model, step)
        end

        if verbose && step % 10 == 0
            @info "Step $step, t = $t_current"
        end
    end

    if verbose
        @info "Simulation complete: $step steps"
    end

    return model
end

end # module Models
