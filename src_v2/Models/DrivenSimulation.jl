"""
    DrivenSimulation

Window-driven standalone runtime for `src_v2` transport models.

A `DrivenSimulation` keeps transport-window timing and forcing in the driver,
while the model retains ownership of prognostic tracer and air-mass state.
The runtime interpolates forcing within each met window and advances the model
with the same `step!(model, Δt)` entry point used by the fixed-flux smoke
harness.
"""
mutable struct DrivenSimulation{ModelT, DriverT, WindowT, AT, QT, FT, CB}
    model                 :: ModelT
    driver                :: DriverT
    window                :: WindowT
    expected_air_mass     :: AT
    qv_buffer             :: QT
    Δt                    :: FT
    window_dt             :: FT
    steps_per_window      :: Int
    time                  :: FT
    iteration             :: Int
    current_window_index  :: Int
    stop_window           :: Int
    final_iteration       :: Int
    callbacks                   :: CB
    initialize_air_mass         :: Bool
    use_midpoint_forcing        :: Bool
    reset_air_mass_each_window  :: Bool
    interpolate_fluxes_within_window :: Bool
end

@inline _basis_symbol(::DryBasis) = :dry
@inline _basis_symbol(::MoistBasis) = :moist

function _check_grid_compatibility(model_grid::AtmosGrid, driver_grid_ref::AtmosGrid)
    typeof(model_grid.horizontal) === typeof(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model grid $(typeof(model_grid.horizontal)) does not match driver grid $(typeof(driver_grid_ref.horizontal))"))
    nlevels(model_grid) == nlevels(driver_grid_ref) ||
        throw(ArgumentError("model and driver vertical levels do not match"))
    ncells(model_grid.horizontal) == ncells(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model and driver horizontal cell counts do not match"))
    nfaces(model_grid.horizontal) == nfaces(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model and driver horizontal face counts do not match"))
    return nothing
end

function _check_basis_compatibility(model::TransportModel, driver::D) where {D <: AbstractMetDriver}
    basis_sym = air_mass_basis(driver)
    _basis_symbol(mass_basis(model.state)) == basis_sym ||
        throw(ArgumentError("model state basis $(_basis_symbol(mass_basis(model.state))) does not match driver basis $(basis_sym)"))
    _basis_symbol(mass_basis(model.fluxes)) == basis_sym ||
        throw(ArgumentError("model flux basis $(_basis_symbol(mass_basis(model.fluxes))) does not match driver basis $(basis_sym)"))
    return nothing
end

@inline function _substep_fraction(substep::Int, steps_per_window::Int, ::Type{FT}, use_midpoint::Bool) where FT
    if steps_per_window == 1
        return zero(FT)
    elseif use_midpoint
        return FT((substep - 0.5) / steps_per_window)
    else
        return FT((substep - 1) / steps_per_window)
    end
end

@inline _active_substep(iteration::Int, steps_per_window::Int) = mod(iteration, steps_per_window) + 1

@inline function _allocate_qv_buffer(window)
    has_humidity_endpoints(window) || return nothing
    return similar(window.qv_start)
end

@inline function _window_backend_adapter(reference_array)
    if isdefined(Main, :CUDA)
        CUDA = getfield(Main, :CUDA)
        if reference_array isa CUDA.AbstractGPUArray
            return CUDA.CuArray
        end
    end
    return Array
end

@inline function _adapt_window_to_model_backend(window, model_air_mass)
    adaptor = _window_backend_adapter(model_air_mass)
    return adaptor === Array ? window : Adapt.adapt(adaptor, window)
end

function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
    else
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    end
    expected_air_mass!(sim.expected_air_mass, sim.window, λ)
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end
    return λ
end

function _load_window(driver::D, win::Int) where {D <: AbstractMetDriver}
    return load_transport_window(driver, win)
end

function _maybe_advance_window!(sim::DrivenSimulation, substep::Int)
    if sim.iteration > 0 && substep == 1
        next_window = sim.current_window_index + 1
        next_window <= sim.stop_window ||
            throw(ArgumentError("DrivenSimulation attempted to step past stop_window=$(sim.stop_window)"))
        sim.window = _adapt_window_to_model_backend(_load_window(sim.driver, next_window), sim.model.state.air_mass)
        sim.current_window_index = next_window
        if sim.qv_buffer !== nothing && !has_humidity_endpoints(sim.window)
            throw(ArgumentError("driver humidity endpoint support changed between windows"))
        end
        if sim.reset_air_mass_each_window
            copyto!(sim.model.state.air_mass, sim.window.air_mass)
        end
    end
    return nothing
end

"""
    DrivenSimulation(model, driver; kwargs...)

Construct a window-driven `src_v2` runtime.

Keyword arguments:
- `start_window=1`
- `stop_window=total_windows(driver)`
- `initialize_air_mass=true`
- `use_midpoint_forcing=true`
- `reset_air_mass_each_window=true`
- `interpolate_fluxes_within_window=nothing` (derive from driver)
- `callbacks=NamedTuple()`
"""
function DrivenSimulation(model::TransportModel,
                          driver::D;
                          start_window::Integer = 1,
                          stop_window::Integer = total_windows(driver),
                          initialize_air_mass::Bool = true,
                          use_midpoint_forcing::Bool = true,
                          reset_air_mass_each_window::Bool = true,
                          interpolate_fluxes_within_window = nothing,
                          callbacks = NamedTuple()) where {D <: AbstractMetDriver}
    1 <= start_window <= stop_window <= total_windows(driver) ||
        throw(ArgumentError("invalid window range: start_window=$(start_window), stop_window=$(stop_window), total_windows=$(total_windows(driver))"))
    supports_native_vertical_flux(driver) ||
        throw(ArgumentError("DrivenSimulation requires native vertical mass fluxes in the met-driver contract"))

    _check_grid_compatibility(model.grid, driver_grid(driver))
    _check_basis_compatibility(model, driver)

    window = _adapt_window_to_model_backend(_load_window(driver, start_window), model.state.air_mass)
    expected_air_mass = similar(model.state.air_mass)
    qv_buffer = _allocate_qv_buffer(window)
    FT = promote_type(eltype(model.state.air_mass), typeof(window_dt(driver)))
    Δt = FT(window_dt(driver) / steps_per_window(driver))
    nsteps_total = (stop_window - start_window + 1) * steps_per_window(driver)

    flux_interp = interpolate_fluxes_within_window === nothing ?
                  (flux_interpolation_mode(driver) === :interpolate) : Bool(interpolate_fluxes_within_window)

    sim = DrivenSimulation{typeof(model), typeof(driver), typeof(window), typeof(expected_air_mass), typeof(qv_buffer), FT, typeof(callbacks)}(
        model,
        driver,
        window,
        expected_air_mass,
        qv_buffer,
        Δt,
        FT(window_dt(driver)),
        steps_per_window(driver),
        zero(FT),
        0,
        Int(start_window),
        Int(stop_window),
        Int(nsteps_total),
        callbacks,
        initialize_air_mass,
        use_midpoint_forcing,
        reset_air_mass_each_window,
        flux_interp,
    )

    if initialize_air_mass
        copyto!(sim.model.state.air_mass, sim.window.air_mass)
    end
    copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    copyto!(sim.expected_air_mass, sim.window.air_mass)
    if sim.qv_buffer !== nothing
        copyto!(sim.qv_buffer, sim.window.qv_start)
    end
    return sim
end

window_index(sim::DrivenSimulation) = sim.current_window_index
substep_index(sim::DrivenSimulation) = _active_substep(sim.iteration, sim.steps_per_window)
current_qv(sim::DrivenSimulation) = sim.qv_buffer

function step!(sim::DrivenSimulation)
    sim.iteration < sim.final_iteration ||
        throw(ArgumentError("DrivenSimulation has already completed all scheduled steps"))

    substep = _active_substep(sim.iteration, sim.steps_per_window)
    _maybe_advance_window!(sim, substep)
    _refresh_forcing!(sim, substep)

    step!(sim.model, sim.Δt)
    sim.time += sim.Δt
    sim.iteration += 1
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

function run_window!(sim::DrivenSimulation)
    target_iteration = min(sim.final_iteration,
                           ((div(sim.iteration, sim.steps_per_window) + 1) * sim.steps_per_window))
    while sim.iteration < target_iteration
        step!(sim)
    end
    return sim
end

function run!(sim::DrivenSimulation)
    while sim.iteration < sim.final_iteration
        step!(sim)
    end
    return sim
end

export DrivenSimulation, run_window!, window_index, substep_index, current_qv
