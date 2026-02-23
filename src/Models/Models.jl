"""
    Models

Top-level model type that assembles grid, fields, physics operators,
met data, I/O, and time stepping into a single runnable object.

Supports both the legacy `run!(model, met_source, t_start, t_end)` path
and the new driver-based `run!(model)` path with dispatch on
`(grid type, buffering strategy)`.
"""
module Models

using ..Architectures: AbstractArchitecture, architecture, array_type
using ..Grids: AbstractGrid
using ..Fields: Field, Center, TracerFields
using ..Advection: AbstractAdvectionScheme
using ..Convection: AbstractConvection
using ..Diffusion: AbstractDiffusion
using ..Chemistry: AbstractChemistry, NoChemistry
using ..TimeSteppers: AbstractTimeStepper, OperatorSplittingTimeStepper, Clock
using Dates
using ..IO: AbstractMetData, AbstractOutputWriter, MetDataSource
using ..IO: AbstractMetDriver, read_met!, prepare_met_for_physics
using ..Sources: AbstractSource
using ..Callbacks: AbstractCallback
using DocStringExtensions

export AbstractModel, TransportModel, run!, update_met_data!
export AbstractBufferingStrategy, SingleBuffer, DoubleBuffer

# Buffering strategy types
include("buffering.jl")

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
struct TransportModel{Arch, G, Tr, ATr, M, TS, Src, OW, CB, Buf, Chem} <: AbstractModel{Arch}
    "CPU or GPU"
    architecture   :: Arch
    "the computational grid"
    grid           :: G
    "NamedTuple of tracer Fields (or raw arrays for direct GPU runs)"
    tracers        :: Tr
    "NamedTuple of adjoint tracer Fields (nothing if not doing adjoint)"
    adj_tracers    :: ATr
    "meteorological data (AbstractMetData or AbstractMetDriver)"
    met_data       :: M
    "simulation clock"
    clock          :: Clock
    "time-stepping strategy"
    timestepper    :: TS
    "vector of emission sources"
    sources        :: Src
    "vector of output writers"
    output_writers :: OW
    "vector of callbacks"
    callbacks      :: CB
    "buffering strategy (SingleBuffer or DoubleBuffer)"
    buffering      :: Buf
    "chemistry scheme (NoChemistry, RadioactiveDecay, CompositeChemistry, ...)"
    chemistry      :: Chem
end

"""
$(TYPEDSIGNATURES)

Construct a `TransportModel` with the new driver-based API.

This constructor supports both the legacy physics-operator style and the new
met-driver + sources + buffering style.
"""
function TransportModel(;
        grid           :: AbstractGrid{FT},
        tracers,
        met_data       = nothing,
        advection      :: Union{AbstractAdvectionScheme, Nothing} = nothing,
        convection     :: Union{AbstractConvection, Nothing} = nothing,
        diffusion      :: Union{AbstractDiffusion, Nothing} = nothing,
        chemistry      :: AbstractChemistry = NoChemistry(),
        Δt             :: Real = 10800.0,
        sources        :: Vector = AbstractSource[],
        output_writers :: Vector = AbstractOutputWriter[],
        callbacks      :: Vector = AbstractCallback[],
        buffering      :: AbstractBufferingStrategy = SingleBuffer(),
        adjoint        :: Bool = false,
        adj_tracers    = nothing) where {FT}

    arch = architecture(grid)

    # Build tracer fields:
    #   (:co2, :ch4)           → allocate via TracerFields (legacy path)
    #   (;co2=nothing)         → allocate raw zero arrays on the architecture
    #   (;co2=array)           → use as-is (pre-allocated GPU/CPU arrays)
    tracer_fields = if tracers isa NTuple{N, Symbol} where N
        TracerFields(tracers, grid)
    elseif tracers isa NamedTuple && any(v -> v === nothing, values(tracers))
        if hasproperty(grid, :Nx)
            # Lat-lon: allocate 3D arrays on device
            AT = array_type(arch)
            names = keys(tracers)
            arrays = ntuple(length(names)) do idx
                if tracers[idx] === nothing
                    AT(zeros(FT, grid.Nx, grid.Ny, grid.Nz))
                else
                    tracers[idx]
                end
            end
            NamedTuple{names}(arrays)
        else
            # Cubed-sphere: run loop allocates its own panel arrays
            tracers
        end
    else
        tracers  # already constructed (e.g. NamedTuple of GPU arrays)
    end

    adj_tracer_fields = if adjoint && adj_tracers === nothing
        tracer_fields isa NamedTuple ? TracerFields(keys(tracer_fields), grid) : nothing
    else
        adj_tracers
    end

    Δt_ft = FT(Δt)

    # Build timestepper only if physics operators are provided (legacy path)
    ts = if advection !== nothing && convection !== nothing && diffusion !== nothing
        OperatorSplittingTimeStepper(;
            advection  = advection,
            convection = convection,
            diffusion  = diffusion,
            chemistry  = chemistry,
            Δt_outer   = Δt_ft)
    else
        nothing
    end

    clock = Clock(FT; Δt = Δt_ft)

    return TransportModel(arch, grid, tracer_fields, adj_tracer_fields,
                          met_data, clock, ts, sources, output_writers,
                          callbacks, buffering, chemistry)
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
    return physics_fields
end

"""
$(TYPEDSIGNATURES)

Run the forward model from `t_start` to `t_end` using the legacy
MetDataSource-based API.

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

            model = TransportModel(
                model.architecture, model.grid, model.tracers,
                model.adj_tracers, physics_fields, model.clock,
                model.timestepper, model.sources, model.output_writers,
                model.callbacks, model.buffering, model.chemistry)

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

# New driver-based run implementations (dispatch on grid + buffering)
include("run_implementations.jl")

end # module Models
