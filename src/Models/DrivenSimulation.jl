"""
    DrivenSimulation

Window-driven standalone runtime for `src` transport models.

A `DrivenSimulation` keeps transport-window timing and forcing in the driver,
while the model retains ownership of prognostic tracer and air-mass state.
The runtime interpolates forcing within each met window and advances the model
with the same `step!(model, Δt)` entry point used by the fixed-flux smoke
harness.

`SurfaceFluxSource` (previously defined here) was migrated to
`src/Operators/SurfaceFlux/` in plan 17 Commit 2; it is still re-exported
below for backward compatibility with external callers that imported it
via `AtmosTransport.SurfaceFluxSource`.
"""
mutable struct DrivenSimulation{ModelT, DriverT, WindowT, AT, QT, FT, CB, SS, CT, PT}
    model                 :: ModelT
    driver                :: DriverT
    window                :: WindowT
    prefetch_window       :: WindowT
    prefetch_task         :: PT
    prefetch_window_index :: Int
    expected_air_mass     :: AT
    qv_buffer             :: QT
    Δt                    :: FT
    window_dt             :: FT
    steps_per_window      :: Int
    steps_per_window_schedule :: Vector{Int}
    time                  :: FT
    iteration             :: Int
    start_window          :: Int
    current_window_index  :: Int
    current_window_start_iteration :: Int
    current_window_end_iteration   :: Int
    stop_window           :: Int
    final_iteration       :: Int
    callbacks                   :: CB
    surface_sources             :: SS
    chemistry                   :: CT
    initialize_air_mass         :: Bool
    use_midpoint_forcing        :: Bool
    interpolate_fluxes_within_window :: Bool
    reset_air_mass_each_window  :: Bool
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
        return (FT(substep) - FT(0.5)) / FT(steps_per_window)
    else
        return (FT(substep) - one(FT)) / FT(steps_per_window)
    end
end

@inline _active_substep(iteration::Int, steps_per_window::Int) = mod(iteration, steps_per_window) + 1

function _driver_step_schedule(driver::AbstractMetDriver)
    schedule = Int.(steps_per_window_schedule(driver))
    length(schedule) == total_windows(driver) ||
        throw(ArgumentError("driver steps_per_window_schedule length $(length(schedule)) " *
                            "does not match total_windows=$(total_windows(driver))"))
    all(>=(1), schedule) ||
        throw(ArgumentError("driver steps_per_window_schedule must contain only positive integers"))
    return schedule
end

@inline _allocate_storage_like(reference) = Base.invokelatest(similar, reference)
@inline _allocate_storage_like(reference::NTuple{6}) =
    ntuple(p -> Base.invokelatest(similar, reference[p]), 6)

@inline _copy_storage!(dest, src) = copyto!(dest, src)
@inline function _copy_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline _storage_eltype(reference) = eltype(reference)
@inline _storage_eltype(reference::NTuple{6}) = eltype(reference[1])

@inline _empty_prefetch_task() = Task(() -> nothing)

_refresh_state_halos!(state, _mesh) = state

function _refresh_state_halos!(state::CubedSphereState, mesh::CubedSphereMesh)
    fill_panel_halos!(state.air_mass, mesh; dir = 1)
    fill_panel_halos!(state.tracers_raw, mesh; dir = 1)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CellState, new_air_mass, _mesh)
    old_air_mass = state.air_mass
    FT = eltype(old_air_mass)
    floor_m = eps(FT)
    for (_name, rm) in eachtracer(state)
        @. rm = ifelse(old_air_mass > floor_m,
                       rm / old_air_mass * new_air_mass,
                       zero(FT))
    end
    copyto!(old_air_mass, new_air_mass)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CubedSphereState,
                                       new_air_mass::NTuple{6},
                                       mesh::CubedSphereMesh)
    fill_panel_halos!(new_air_mass, mesh; dir = 1)
    FT = eltype(state.air_mass[1])
    floor_m = eps(FT)
    for p in 1:6
        old_p = state.air_mass[p]
        new_p = new_air_mass[p]
        raw_p = state.tracers_raw[p]
        for idx in 1:length(tracer_names(state))
            rm = selectdim(raw_p, ndims(raw_p), idx)
            @. rm = ifelse(old_p > floor_m,
                           rm / old_p * new_p,
                           zero(FT))
        end
        copyto!(old_p, new_p)
    end
    return _refresh_state_halos!(state, mesh)
end

@inline function _allocate_qv_buffer(window)
    has_humidity_endpoints(window) || return nothing
    return _allocate_storage_like(window.qv_start)
end

@inline function _window_backend_adapter(reference_array)
    return array_adapter_for(reference_array)
end

@inline _window_backend_adapter(reference_array::NTuple{6}) = _window_backend_adapter(reference_array[1])

@inline function _adapt_window_to_model_backend(window, model_air_mass)
    adaptor = _window_backend_adapter(model_air_mass)
    return adaptor === Array ? window : Base.invokelatest(Adapt.adapt, adaptor, window)
end

@inline function _copy_optional_storage!(dest, src, field::Symbol)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window capability for `$(field)` changed between windows"))
        return dest
    end
    return _copy_storage!(dest, src)
end

@inline function _copy_optional_convection!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window convection capability changed between windows"))
        return dest
    end
    return copy_convection_forcing!(dest, src)
end

@inline function _copy_optional_surface!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window surface-forcing capability changed between windows"))
        return dest
    end
    return _copy_surface_forcing!(dest, src)
end

@inline _copy_surface_forcing!(dest, src) =
    throw(ArgumentError("unsupported surface-forcing refresh from $(typeof(src)) to $(typeof(dest))"))

@inline function _copy_surface_forcing!(dest::PBLSurfaceForcing, src::PBLSurfaceForcing)
    _copy_storage!(dest.pblh, src.pblh)
    _copy_storage!(dest.ustar, src.ustar)
    _copy_storage!(dest.hflux, src.hflux)
    _copy_storage!(dest.t2m, src.t2m)
    return dest
end

@inline function _copy_flux_deltas!(dest::StructuredFluxDeltas, src::StructuredFluxDeltas)
    _copy_storage!(dest.dam, src.dam)
    _copy_storage!(dest.dbm, src.dbm)
    _copy_storage!(dest.dcm, src.dcm)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_flux_deltas!(dest::FaceIndexedFluxDeltas, src::FaceIndexedFluxDeltas)
    _copy_storage!(dest.dhflux, src.dhflux)
    _copy_storage!(dest.dcm, src.dcm)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_flux_deltas!(dest::CubedSphereFluxDeltas, src::CubedSphereFluxDeltas)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_optional_deltas!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window flux-delta capability changed between windows"))
        return dest
    end
    return _copy_flux_deltas!(dest, src)
end

function _copy_window_payload!(dest::StructuredTransportWindow{B},
                               src::StructuredTransportWindow{B}) where {B <: AbstractMassBasis}
    _copy_storage!(dest.air_mass, src.air_mass)
    _copy_storage!(dest.surface_pressure, src.surface_pressure)
    copy_fluxes!(dest.fluxes, src.fluxes)
    _copy_optional_storage!(dest.qv_start, src.qv_start, :qv_start)
    _copy_optional_storage!(dest.qv_end, src.qv_end, :qv_end)
    _copy_optional_deltas!(dest.deltas, src.deltas)
    _copy_optional_convection!(dest.convection, src.convection)
    return dest
end

function _copy_window_payload!(dest::FaceIndexedTransportWindow{B},
                               src::FaceIndexedTransportWindow{B}) where {B <: AbstractMassBasis}
    _copy_storage!(dest.air_mass, src.air_mass)
    _copy_storage!(dest.surface_pressure, src.surface_pressure)
    copy_fluxes!(dest.fluxes, src.fluxes)
    _copy_optional_storage!(dest.qv_start, src.qv_start, :qv_start)
    _copy_optional_storage!(dest.qv_end, src.qv_end, :qv_end)
    _copy_optional_deltas!(dest.deltas, src.deltas)
    _copy_optional_convection!(dest.convection, src.convection)
    return dest
end

function _copy_window_payload!(dest::CubedSphereTransportWindow{B},
                               src::CubedSphereTransportWindow{B}) where {B <: AbstractMassBasis}
    _copy_storage!(dest.air_mass, src.air_mass)
    _copy_storage!(dest.surface_pressure, src.surface_pressure)
    copy_fluxes!(dest.fluxes, src.fluxes)
    _copy_optional_storage!(dest.qv_start, src.qv_start, :qv_start)
    _copy_optional_storage!(dest.qv_end, src.qv_end, :qv_end)
    _copy_optional_deltas!(dest.deltas, src.deltas)
    _copy_optional_convection!(dest.convection, src.convection)
    _copy_optional_surface!(dest.surface, src.surface)
    return dest
end

function _load_window_into_existing_backend!(existing_window,
                                             driver::AbstractMetDriver,
                                             win::Int,
                                             model_air_mass)
    loaded = _load_window(driver, win)
    adaptor = _window_backend_adapter(model_air_mass)
    if adaptor === Array
        return loaded
    end
    _copy_window_payload!(existing_window, loaded)
    return existing_window
end

@inline _prefetch_enabled(model_air_mass) =
    _window_backend_adapter(model_air_mass) !== Array && Threads.nthreads() > 1

function _start_window_prefetch!(sim::DrivenSimulation, target_window::Int)
    if target_window > sim.stop_window || !_prefetch_enabled(sim.model.state.air_mass)
        sim.prefetch_window_index = 0
        sim.prefetch_task = _empty_prefetch_task()
        return nothing
    end
    target_slot = sim.prefetch_window
    driver = sim.driver
    model_air_mass = sim.model.state.air_mass
    sim.prefetch_window_index = target_window
    sim.prefetch_task = Threads.@spawn _load_window_into_existing_backend!(
        target_slot, driver, target_window, model_air_mass)
    return nothing
end

function _take_prefetched_window!(sim::DrivenSimulation, next_window::Int)
    if _prefetch_enabled(sim.model.state.air_mass) &&
       sim.prefetch_window_index == next_window
        fetched = fetch(sim.prefetch_task)
        fetched === sim.prefetch_window ||
            throw(ArgumentError("prefetched transport window identity changed unexpectedly"))
        old_current = sim.window
        sim.window = sim.prefetch_window
        sim.prefetch_window = old_current
        sim.prefetch_task = _empty_prefetch_task()
        sim.prefetch_window_index = 0
        return nothing
    end
    sim.window = _load_window_into_existing_backend!(sim.window, sim.driver,
                                                     next_window,
                                                     sim.model.state.air_mass)
    return nothing
end

function _reclaim_backend_pool_after_startup!(model_air_mass)
    _window_backend_adapter(model_air_mass) === Array && return nothing
    # Startup can allocate large transient CuArrays while adapting initial
    # conditions and first-window forcing. They are dead before the run loop,
    # but CUDA.jl's pool keeps them reserved unless we explicitly trim it.
    GC.gc(false)
    if isdefined(Main, :CUDA)
        CUDA = getproperty(Main, :CUDA)
        if isdefined(CUDA, :synchronize)
            Base.invokelatest(getproperty(CUDA, :synchronize))
        end
        if isdefined(CUDA, :reclaim)
            Base.invokelatest(getproperty(CUDA, :reclaim))
        end
    elseif isdefined(Main, :Metal)
        Metal = getproperty(Main, :Metal)
        isdefined(Metal, :synchronize) &&
            Base.invokelatest(getproperty(Metal, :synchronize))
    end
    return nothing
end

@inline function _adapt_sources_to_model_backend(surface_sources, model_air_mass)
    adaptor = _window_backend_adapter(model_air_mass)
    return adaptor === Array ? surface_sources :
           map(source -> Base.invokelatest(Adapt.adapt, adaptor, source), surface_sources)
end

# Surface-source helpers (`_surface_shape`, `_check_surface_source_compatibility`,
# `_apply_surface_source!`) migrated to `src/Operators/SurfaceFlux/sources.jl`
# in plan 17 Commit 2. Imported here from the SurfaceFlux submodule so the
# sim-level application path (`_apply_surface_sources!` below) keeps working
# unchanged until plan 17 Commit 6 moves the call site into the palindrome.
using ..Operators.SurfaceFlux: _surface_shape,
                                _check_surface_source_compatibility,
                                _apply_surface_source!
using ..Operators.Diffusion: NoDiffusion, fill_dz_hydrostatic_constT!

# ---------------------------------------------------------------------------
# Diffusion `dz_scratch` populator.
#
# `apply_vertical_diffusion!` divides by `dz` per cell — if the workspace's
# `dz_scratch` array is left at its default zeros (the allocator initializes
# it that way), every diffusion step nukes the tracer field to NaN starting
# from frame 2.  We refresh `dz_scratch` from the just-loaded window's
# surface pressure + the grid's hybrid-σp coefficients each time the
# simulation advances to a new met window.
#
# `dz` only depends on (ps, ak, bk) (constant-T_ref hydrostatic); within a
# window ps is fixed, so one fill per window is correct and cheap.
# ---------------------------------------------------------------------------
@inline function _refresh_dz_for_window!(sim::DrivenSimulation)
    sim.model.diffusion isa NoDiffusion && return nothing
    workspace = sim.model.workspace
    hasproperty(workspace, :dz_scratch) || return nothing
    dz_scratch = workspace.dz_scratch
    vertical = sim.model.grid.vertical
    ps = sim.window.surface_pressure
    fill_dz_hydrostatic_constT!(dz_scratch, ps, vertical.A, vertical.B)
    return nothing
end

@inline _refresh_pbl_kz_for_window!(_field, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(field::WindowPBLKzField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_pbl_kz_cache!(field, sim.window.surface, sim.window.air_mass,
                           mesh.cell_areas; halo_width = mesh.Hp)
    return nothing
end

@inline _refresh_pbl_kz_for_window!(::NoDiffusion, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(op::ImplicitVerticalDiffusion,
                                     sim::DrivenSimulation)
    _refresh_pbl_kz_for_window!(op.kz_field, sim)
    return nothing
end

function _apply_surface_sources!(sim::DrivenSimulation)
    isempty(sim.surface_sources) && return nothing
    for source in sim.surface_sources
        rm = get_tracer(sim.model.state, source.tracer_name)
        _apply_surface_source!(rm, source, sim.Δt)
    end
    return nothing
end

"""
    _validate_convection_window!(op, window, driver) -> nothing

Per-operator validation of a loaded transport window. Operator
authors add a method for their concrete type; the fallback method
throws `ArgumentError` naming the operator and pointing at this
function as the place to add a method (plan 23 principle 10).

Plan 23 Commit 1 refactors the former `if/elseif op isa …` chain
in `_validate_convection_runtime` into this dispatch pattern so
adding `TM5Convection` (or any future operator) does not require
editing the old runtime block, only adding a method here.
"""
_validate_convection_window!(::NoConvection, _window, _driver) = nothing

function _validate_convection_window!(::CMFMCConvection,
                                       window::AbstractTransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.cmfmc === nothing &&
        throw(ArgumentError(
            "CMFMCConvection requires `window.convection.cmfmc` to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without CMFMC."))
    return nothing
end

function _validate_convection_window!(::TM5Convection,
                                       window::AbstractTransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.tm5_fields === nothing &&
        throw(ArgumentError(
            "TM5Convection requires `window.convection.tm5_fields` " *
            "(NamedTuple with :entu, :detu, :entd, :detd) to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without TM5 fields. " *
            "Preprocess the binary with `scripts/preprocessing/preprocess_spectral_v4_binary.jl` " *
            "and `tm5_convection = true` in the run config, or fall back to " *
            "`CMFMCConvection()` if you have GEOS-FP CMFMC data instead."))
    return nothing
end

function _validate_convection_window!(op::AbstractConvection,
                                       ::AbstractTransportWindow,
                                       ::AbstractMetDriver)
    throw(ArgumentError(
        "DrivenSimulation does not support convection operator $(typeof(op)) yet. " *
        "Add a `_validate_convection_window!(::$(typeof(op)), window, driver)` " *
        "method in `src/Models/DrivenSimulation.jl` that checks its forcing " *
        "requirements."))
end

function _validate_convection_runtime(model::TransportModel,
                                      driver::AbstractMetDriver,
                                      window::AbstractTransportWindow)
    op = model.convection
    op isa NoConvection && return nothing

    window.convection === nothing &&
        throw(ArgumentError(
            "DrivenSimulation loaded a transport window without convection forcing, " *
            "but model.convection = $(typeof(op)) is active. " *
            "Install a driver/window path that populates `window.convection` " *
            "for this operator."))

    _validate_convection_window!(op, window, driver)
    return nothing
end

function _install_convection_forcing(model::TransportModel,
                                     driver::AbstractMetDriver,
                                     window::AbstractTransportWindow)
    _validate_convection_runtime(model, driver, window)
    model.convection isa NoConvection && return model

    forcing = allocate_convection_forcing_like(window.convection, model.state.air_mass)
    copy_convection_forcing!(forcing, window.convection)
    return with_convection_forcing(model, forcing)
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
    if !(sim.model.convection isa NoConvection)
        copy_convection_forcing!(sim.model.convection_forcing, sim.window.convection)
    end
    return λ
end

function _load_window(driver::D, win::Int) where {D <: AbstractMetDriver}
    return load_transport_window(driver, win)
end

function _maybe_advance_window!(sim::DrivenSimulation)
    if sim.iteration > 0 && sim.iteration == sim.current_window_end_iteration
        next_window = sim.current_window_index + 1
        next_window <= sim.stop_window ||
            throw(ArgumentError("DrivenSimulation attempted to step past stop_window=$(sim.stop_window)"))
        sim.current_window_start_iteration = sim.iteration
        sim.current_window_index = next_window
        sim.steps_per_window = sim.steps_per_window_schedule[next_window]
        sim.current_window_end_iteration = sim.iteration + sim.steps_per_window
        sim.Δt = sim.window_dt / typeof(sim.Δt)(sim.steps_per_window)
        _take_prefetched_window!(sim, next_window)
        if sim.reset_air_mass_each_window
            _reset_air_mass_preserve_vmr!(sim.model.state, sim.window.air_mass,
                                          sim.model.grid.horizontal)
        end
        if sim.qv_buffer !== nothing && !has_humidity_endpoints(sim.window)
            throw(ArgumentError("driver humidity endpoint support changed between windows"))
        end
        _validate_convection_runtime(sim.model, sim.driver, sim.window)
        _refresh_dz_for_window!(sim)
        _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
        # Plan 39 Commit G: the `reset_air_mass_each_window` flag has been
        # removed. Under the canonical `:window_constant` contract, the
        # runtime's own flux divergence integrates to `(m_next - m)` over
        # each window, so `state.air_mass` naturally tracks `window.air_mass`
        # at window boundaries without an explicit reset. The reset used to
        # inject the 2nd-order ps-acceleration mismatch that caused the
        # upwind monotonicity-violating window-edge jump (~0.87% on uniform
        # IC) diagnosed in plan-24 post-mortem (memo 37 + this plan).
        invalidate_cmfmc_cache!(sim.model.workspace.convection_ws)
        _start_window_prefetch!(sim, next_window + 1)
    end
    return nothing
end

function _maybe_reset_to_window_endpoint!(sim::DrivenSimulation)
    (sim.reset_air_mass_each_window && _uses_binary_transport_schedule(sim)) ||
        return nothing
    expected_air_mass!(sim.expected_air_mass, sim.window, one(typeof(sim.Δt)))
    _reset_air_mass_preserve_vmr!(sim.model.state, sim.expected_air_mass,
                                  sim.model.grid.horizontal)
    return nothing
end

"""
    DrivenSimulation(model, driver; kwargs...)

Construct a window-driven `src` runtime.

Keyword arguments:
- `start_window=1`
- `stop_window=total_windows(driver)`
- `initialize_air_mass=true`
- `use_midpoint_forcing=true`
- `interpolate_fluxes_within_window=nothing` (derive from driver)
- `reset_air_mass_each_window=false` — when true, each newly loaded
  window replaces prognostic air mass while preserving tracer VMR.
  For binary-scheduled runs, the same endpoint reset is applied before
  the once-per-window convection/chemistry block so physics sees the
  binary's authoritative window-end mass.
- `surface_sources=()`
- `chemistry=NoChemistry()` — applied after advection + surface sources each step
- `callbacks=NamedTuple()`
"""
function DrivenSimulation(model::TransportModel,
                          driver::D;
                          start_window::Integer = 1,
                          stop_window::Integer = total_windows(driver),
                          initialize_air_mass::Bool = true,
                          use_midpoint_forcing::Bool = true,
                          interpolate_fluxes_within_window = nothing,
                          reset_air_mass_each_window::Bool = false,
                          surface_sources = (),
                          chemistry::AbstractChemistryOperator = NoChemistry(),
                          callbacks = NamedTuple()) where {D <: AbstractMetDriver}
    1 <= start_window <= stop_window <= total_windows(driver) ||
        throw(ArgumentError("invalid window range: start_window=$(start_window), stop_window=$(stop_window), total_windows=$(total_windows(driver))"))
    supports_native_vertical_flux(driver) ||
        throw(ArgumentError("DrivenSimulation requires native vertical mass fluxes in the met-driver contract"))

    _check_grid_compatibility(model.grid, driver_grid(driver))
    _check_basis_compatibility(model, driver)

    window = _adapt_window_to_model_backend(_load_window(driver, start_window), model.state.air_mass)
    prefetch_window = _prefetch_enabled(model.state.air_mass) && start_window < stop_window ?
                      _adapt_window_to_model_backend(_load_window(driver, start_window), model.state.air_mass) :
                      window
    prefetch_task = _empty_prefetch_task()
    expected_air_mass = _allocate_storage_like(model.state.air_mass)
    qv_buffer = _allocate_qv_buffer(window)
    surface_sources_adapted = _adapt_sources_to_model_backend(Tuple(surface_sources), model.state.air_mass)
    foreach(source -> _check_surface_source_compatibility(model.state, source), surface_sources_adapted)

    # Plan 17 Commit 6: move chemistry + emissions from sim-level post-
    # step application into the model's transport block. `with_emissions`
    # installs the user-supplied surface sources as a `SurfaceFluxOperator`
    # inside the wrapped model so the palindrome's S slot runs at the
    # correct center-of-transport position. `with_chemistry` installs the
    # user's chemistry in the model; `step!(model)` runs
    # `advection → emissions → diffusion → chemistry` as ONE composed
    # call. The sim's `_apply_surface_sources!` helper and post-step
    # `chemistry_block!` are no longer called at sim level — they are
    # retained on the sim struct for adaptive reconfiguration via
    # future helpers but the step loop no longer invokes them directly.
    #
    # Pre-plan-17 the sim held chemistry at the sim level as a plan-15
    # workaround to preserve TM5's `advection → emissions → chemistry`
    # order while emissions still lived outside the palindrome. That
    # workaround is now resolved by the palindrome integration (plan 17
    # Commit 5), so the sim delegates entirely to `step!(model)`.
    model = with_chemistry(model, chemistry)
    if !isempty(surface_sources_adapted)
        emissions_op = SurfaceFluxOperator(PerTracerFluxMap(surface_sources_adapted))
        model = with_emissions(model, emissions_op)
    end
    model = _install_convection_forcing(model, driver, window)
    FT = _storage_eltype(model.state.air_mass)
    step_schedule = _driver_step_schedule(driver)
    steps_current = step_schedule[Int(start_window)]
    Δt = FT(window_dt(driver)) / FT(steps_current)
    nsteps_total = sum(@view step_schedule[Int(start_window):Int(stop_window)])

    flux_interp = interpolate_fluxes_within_window === nothing ?
                  (flux_interpolation_mode(driver) === :interpolate) : Bool(interpolate_fluxes_within_window)

    sim = DrivenSimulation{typeof(model), typeof(driver), typeof(window),
                           typeof(expected_air_mass), typeof(qv_buffer), FT,
                           typeof(callbacks), typeof(surface_sources_adapted),
                           typeof(chemistry), typeof(prefetch_task)}(
        model,
        driver,
        window,
        prefetch_window,
        prefetch_task,
        0,
        expected_air_mass,
        qv_buffer,
        Δt,
        FT(window_dt(driver)),
        steps_current,
        step_schedule,
        zero(FT),
        0,
        Int(start_window),
        Int(start_window),
        0,
        steps_current,
        Int(stop_window),
        Int(nsteps_total),
        callbacks,
        surface_sources_adapted,
        chemistry,
        initialize_air_mass,
        use_midpoint_forcing,
        flux_interp,
        Bool(reset_air_mass_each_window),
    )

    if initialize_air_mass
        _copy_storage!(sim.model.state.air_mass, sim.window.air_mass)
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    elseif sim.reset_air_mass_each_window
        _reset_air_mass_preserve_vmr!(sim.model.state, sim.window.air_mass,
                                      sim.model.grid.horizontal)
    else
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    end
    copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    _copy_storage!(sim.expected_air_mass, sim.window.air_mass)
    if sim.qv_buffer !== nothing
        _copy_storage!(sim.qv_buffer, sim.window.qv_start)
    end
    _refresh_dz_for_window!(sim)
    _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
    _reclaim_backend_pool_after_startup!(sim.model.state.air_mass)
    _start_window_prefetch!(sim, Int(start_window) + 1)
    return sim
end

window_index(sim::DrivenSimulation) = sim.current_window_index
function substep_index(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        return 1
    end
    return min(sim.steps_per_window,
               sim.iteration - sim.current_window_start_iteration + 1)
end
current_qv(sim::DrivenSimulation) = sim.qv_buffer

"""
    current_time(sim::DrivenSimulation) -> FT

Simulation time [s] at the start of the next step. Returns
`sim.time`, which is initialized to `zero(FT)` at sim construction
and advanced by `sim.time += sim.Δt` at the end of each `step!(sim)`.

Plan 18 A3 threads `sim` through operators via the `meteo` kwarg:

    step!(sim.model, sim.Δt; meteo = sim)   # not sim.driver

so operators that need time (`StepwiseField` emission rates,
time-varying Kz, future convection DerivedConvMassFluxField) read
`current_time(meteo)` and get `sim.time`. `meteo.driver` remains
accessible for operator code that needs driver-level capabilities
(e.g. `supports_cmfmc(meteo.driver)`).

The legacy `current_time(::AbstractMetDriver) = 0.0` stub is kept
for backward compatibility — the driver is stateless and cannot
provide real time information on its own.
"""
MetDrivers.current_time(sim::DrivenSimulation) = sim.time

@inline _uses_binary_transport_schedule(sim::DrivenSimulation) =
    uses_binary_substep_contract(sim.driver)

function step!(sim::DrivenSimulation)
    sim.iteration < sim.final_iteration ||
        throw(ArgumentError("DrivenSimulation has already completed all scheduled steps"))

    SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    substep = substep_index(sim)
    SectionTimer.@section :forcing_refresh _refresh_forcing!(sim, substep)

    # Plan 17 Commit 6 + plan 18 A3: the default path keeps the live
    # operator suite in one call. Plan 41 v3 transport binaries carry an
    # advection substep contract, not a physics cadence contract, so driven
    # binary-scheduled runs apply only the transport block at each stored
    # substep and defer convection + chemistry to the end of the met window.
    if _uses_binary_transport_schedule(sim)
        transport_step!(sim.model, sim.Δt; meteo = sim)
    else
        step!(sim.model, sim.Δt; meteo = sim)
    end
    sim.time += sim.Δt
    sim.iteration += 1
    if _uses_binary_transport_schedule(sim) &&
       sim.iteration == sim.current_window_end_iteration
        _maybe_reset_to_window_endpoint!(sim)
        convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
    end
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

function run_window!(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    end
    target_iteration = min(sim.final_iteration, sim.current_window_end_iteration)
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

# `SurfaceFluxSource` re-exported for backward compat with external callers.
# The symbol resolves to `Operators.SurfaceFlux.SurfaceFluxSource` — its
# canonical location post plan 17 Commit 2.
export SurfaceFluxSource, DrivenSimulation, run_window!, window_index, substep_index, current_qv
