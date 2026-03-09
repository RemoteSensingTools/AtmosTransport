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
using ..Sources: AbstractSurfaceFlux
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
struct TransportModel{Arch, G, Tr, ATr, M, TS, Src, OW, CB, Buf, Adv, Chem, Diff, Conv} <: AbstractModel{Arch}
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
    "advection scheme (SlopesAdvection or PPMAdvection{ORD}, default SlopesAdvection)"
    advection_scheme :: Adv
    "chemistry scheme (NoChemistry, RadioactiveDecay, CompositeChemistry, ...)"
    chemistry      :: Chem
    "vertical diffusion scheme (nothing or BoundaryLayerDiffusion or PBLDiffusion)"
    diffusion      :: Diff
    "convection scheme (nothing or TiedtkeConvection)"
    convection     :: Conv
    "run metadata: config, user, hostname, timestamps"
    metadata       :: Dict{String, Any}
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
        sources        :: Vector = AbstractSurfaceFlux[],
        output_writers :: Vector = AbstractOutputWriter[],
        callbacks      :: Vector = AbstractCallback[],
        buffering      :: AbstractBufferingStrategy = SingleBuffer(),
        adjoint        :: Bool = false,
        adj_tracers    = nothing,
        metadata       :: Dict{String, Any} = Dict{String, Any}()) where {FT}

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

    # Default to SlopesAdvection if no advection scheme specified
    adv_scheme = advection !== nothing ? advection : SlopesAdvection()

    return TransportModel(arch, grid, tracer_fields, adj_tracer_fields,
                          met_data, clock, ts, sources, output_writers,
                          callbacks, buffering, adv_scheme, chemistry, diffusion, convection,
                          metadata)
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
                model.callbacks, model.buffering, model.chemistry,
                model.diffusion)

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

# =====================================================================
# Pretty-print tree for TransportModel
# =====================================================================

function _tree_line(io, prefix, connector, text)
    print(io, prefix, connector, text, "\n")
end

function _grid_summary(g)
    if hasproperty(g, :Nc)
        "CubedSphereGrid C$(g.Nc) (6 × $(g.Nc) × $(g.Nc) × $(g.Nz))"
    elseif hasproperty(g, :Nx)
        "LatLonGrid $(g.Nx) × $(g.Ny) × $(g.Nz)"
    else
        string(typeof(g))
    end
end

function _met_summary(io, m, prefix)
    if m === nothing
        _tree_line(io, prefix, "├── ", "Met data: none")
        return
    end
    tname = string(nameof(typeof(m)))
    _tree_line(io, prefix, "├── ", "Met data: $tname")
    sub = prefix * "│   "
    if hasproperty(m, :mode)
        nw = hasproperty(m, :n_windows) ? m.n_windows : "?"
        _tree_line(io, sub, "├── ", "Mode: $(m.mode) ($nw windows)")
    end
    if hasproperty(m, :Nc)
        nz = hasproperty(m, :Nz) ? m.Nz : "?"
        _tree_line(io, sub, "├── ", "Resolution: C$(m.Nc) × $nz levels")
    end
    if hasproperty(m, :met_interval)
        mfdt = hasproperty(m, :mass_flux_dt) ? ", mass_flux_dt=$(m.mass_flux_dt)s" : ""
        _tree_line(io, sub, "├── ", "Interval: $(m.met_interval)s$mfdt")
    end
    if hasproperty(m, :_start_date)
        sd = m._start_date
        nw = hasproperty(m, :n_windows) ? m.n_windows : 0
        mi = hasproperty(m, :met_interval) ? m.met_interval : 3600
        ed = sd + Day(max(0, round(Int, nw * mi / 86400) - 1))
        _tree_line(io, sub, "└── ", "Period: $sd → $ed")
    end
end

function _advection_str(a)
    if a === nothing
        "none"
    else
        sprint(show, a)
    end
end

function _conv_str(c)
    c === nothing ? "none" : string(nameof(typeof(c)))
end

function _diff_summary(io, d, prefix)
    if d === nothing
        _tree_line(io, prefix, "├── ", "Diffusion: none")
        return
    end
    tname = string(nameof(typeof(d)))
    params = String[]
    for f in (:β_h, :Kz_bg, :Kz_max, :fak, :sffrac)
        hasproperty(d, f) && push!(params, "$f=$(getproperty(d, f))")
    end
    pstr = isempty(params) ? "" : " ($(join(params, ", ")))"
    _tree_line(io, prefix, "├── ", "Diffusion: $tname$pstr")
end

function _chem_summary(io, c, prefix)
    tname = string(nameof(typeof(c)))
    if c isa NoChemistry
        _tree_line(io, prefix, "├── ", "Chemistry: none")
    elseif hasproperty(c, :schemes)
        _tree_line(io, prefix, "├── ", "Chemistry: $tname ($(length(c.schemes)) schemes)")
        sub = prefix * "│   "
        for (i, s) in enumerate(c.schemes)
            conn = i < length(c.schemes) ? "├── " : "└── "
            sname = string(nameof(typeof(s)))
            detail = hasproperty(s, :species) ? "$(s.species) → $sname" : sname
            if hasproperty(s, :half_life)
                days = round(s.half_life / 86400, digits=2)
                detail *= " (t½=$(days)d)"
            end
            _tree_line(io, sub, conn, detail)
        end
    else
        detail = hasproperty(c, :species) ? " ($(c.species))" : ""
        _tree_line(io, prefix, "├── ", "Chemistry: $tname$detail")
    end
end

function _source_label(s)
    tname = string(nameof(typeof(s)))
    sp = hasproperty(s, :species) ? string(s.species) : "?"
    lab = hasproperty(s, :label) ? " \"$(s.label)\"" : ""
    # snapshot count
    nt = if hasproperty(s, :flux_data) && s.flux_data isa AbstractArray && ndims(s.flux_data) == 3
        " ($(size(s.flux_data, 3)) snapshots)"
    elseif hasproperty(s, :flux_data) && s.flux_data isa AbstractVector
        " ($(length(s.flux_data)) snapshots)"
    elseif hasproperty(s, :time_hours)
        " ($(length(s.time_hours)) snapshots)"
    else
        ""
    end
    "$sp: $tname$lab$nt"
end

function _sources_summary(io, sources, prefix)
    if isempty(sources)
        _tree_line(io, prefix, "├── ", "Sources: none")
        return
    end
    # Flatten CombinedFlux to show individual components
    flat = Pair{String, Any}[]
    for s in sources
        if hasproperty(s, :components)
            for c in s.components
                push!(flat, _source_label(c) => c)
            end
        else
            push!(flat, _source_label(s) => s)
        end
    end
    _tree_line(io, prefix, "├── ", "Sources ($(length(flat))):")
    sub = prefix * "│   "
    for (i, (lab, _)) in enumerate(flat)
        conn = i < length(flat) ? "├── " : "└── "
        _tree_line(io, sub, conn, lab)
    end
end

function _output_summary(io, writers, prefix)
    if isempty(writers)
        _tree_line(io, prefix, "├── ", "Output: none")
        return
    end
    for (wi, w) in enumerate(writers)
        is_last_writer = wi == length(writers)
        tname = string(nameof(typeof(w)))
        _tree_line(io, prefix, "├── ", "Output: $tname")
        sub = prefix * "│   "
        if hasproperty(w, :filepath)
            p = w.filepath
            # Shorten home directory
            home = get(ENV, "HOME", "")
            if !isempty(home) && startswith(p, home)
                p = "~" * p[length(home)+1:end]
            end
            _tree_line(io, sub, "├── ", "Path: $p")
        end
        if hasproperty(w, :schedule)
            sched = w.schedule
            interval = hasproperty(sched, :interval) ? "every $(sched.interval)s" : string(sched)
            split_str = hasproperty(w, :split) && w.split != :none ? ", split=$(w.split)" : ""
            _tree_line(io, sub, "├── ", "Schedule: $interval$split_str")
        end
        if hasproperty(w, :fields)
            nf = length(w.fields)
            names = collect(keys(w.fields))
            shown = min(nf, 4)
            flist = join(string.(names[1:shown]), ", ")
            if nf > shown
                flist *= ", …"
            end
            _tree_line(io, sub, "└── ", "Fields ($nf): $flist")
        end
    end
end

_float_type(::AbstractGrid{FT}) where FT = FT
_float_type(g) = eltype(g)

function Base.show(io::IO, ::MIME"text/plain", model::TransportModel)
    FT = _float_type(model.grid)
    arch = string(nameof(typeof(model.architecture)))
    println(io, "TransportModel ($FT, $arch)")

    pre = ""

    # Grid
    gs = _grid_summary(model.grid)
    _tree_line(io, pre, "├── ", "Grid: $gs")
    g = model.grid
    if hasproperty(g, :vertical)
        vz = string(nameof(typeof(g.vertical)))
        nz = g.Nz
        _tree_line(io, pre, "│   ", "└── Vertical: $vz ($nz levels)")
    end

    # Clock
    _tree_line(io, pre, "├── ", "Clock: Δt=$(model.clock.Δt)s")

    # Tracers
    if model.tracers isa NamedTuple
        tnames = join(string.(keys(model.tracers)), ", ")
        _tree_line(io, pre, "├── ", "Tracers: $tnames")
    else
        _tree_line(io, pre, "├── ", "Tracers: $(typeof(model.tracers))")
    end

    # Met data
    _met_summary(io, model.met_data, pre)

    # Advection
    _tree_line(io, pre, "├── ", "Advection: $(_advection_str(model.advection_scheme))")

    # Convection
    _tree_line(io, pre, "├── ", "Convection: $(_conv_str(model.convection))")

    # Diffusion
    _diff_summary(io, model.diffusion, pre)

    # Chemistry
    _chem_summary(io, model.chemistry, pre)

    # Sources
    _sources_summary(io, model.sources, pre)

    # Output
    _output_summary(io, model.output_writers, pre)

    # Buffering (last item → └──)
    buf = string(nameof(typeof(model.buffering)))
    _tree_line(io, pre, "└── ", "Buffering: $buf")
end

# Compact single-line show
function Base.show(io::IO, model::TransportModel)
    FT = _float_type(model.grid)
    arch = string(nameof(typeof(model.architecture)))
    gs = _grid_summary(model.grid)
    nt = model.tracers isa NamedTuple ? length(keys(model.tracers)) : "?"
    print(io, "TransportModel($FT, $arch, $gs, $(nt) tracers)")
end

end # module Models
