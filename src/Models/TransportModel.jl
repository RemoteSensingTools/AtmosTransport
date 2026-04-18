"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src` transport runs.

Post plan 15: carries both transport (advection) and chemistry operators.
`step!(model, dt)` composes the two per `OPERATOR_COMPOSITION.md` §3.1:

    transport_block(dt)   →   chemistry_block(dt)

Default `chemistry = NoChemistry()` makes the chemistry block a no-op and
keeps the pre-plan-15 behaviour for callers that don't opt in.
"""
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT, ChemT}
    state     :: StateT
    fluxes    :: FluxT
    grid      :: GridT
    advection :: SchemeT
    workspace :: WorkspaceT
    chemistry :: ChemT
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace), typeof(chemistry)}(
        state, fluxes, grid, advection, workspace, chemistry)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace), typeof(chemistry)}(
        state, fluxes, grid, advection, workspace, chemistry)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry()) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh is metadata-only in src; structured transport models are only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

"""
    with_chemistry(model::TransportModel, chemistry)

Return a copy of `model` with its chemistry operator replaced. All other
fields share storage with the original. Used by `DrivenSimulation` to
install the simulation-level chemistry operator into the model.
"""
function with_chemistry(model::TransportModel, chemistry::AbstractChemistryOperator)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(chemistry)}(
        model.state, model.fluxes, model.grid, model.advection, model.workspace,
        chemistry)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state     = Adapt.adapt(to, model.state)
    fluxes    = Adapt.adapt(to, model.fluxes)
    workspace = Adapt.adapt(to, model.workspace)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid),
                          typeof(model.advection), typeof(workspace), typeof(model.chemistry)}(
        state, fluxes, model.grid, model.advection, workspace, model.chemistry)
end

"""
    step!(model::TransportModel, dt)

Advance `model.state` by one step: transport block (advection) → chemistry
block. Chemistry defaults to `NoChemistry()` (no-op) unless the model was
built with an explicit chemistry operator. Mirrors
`OPERATOR_COMPOSITION.md` §3.1 with `N_chem_substeps = 1`.
"""
function step!(model::TransportModel, dt)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace)
    chemistry_block!(model.state, nothing, model.grid, model.chemistry, dt)
    return nothing
end

export TransportModel, step!, with_chemistry
