"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src_v2` transport runs.
"""
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT}
    state     :: StateT
    fluxes    :: FluxT
    grid      :: GridT
    advection :: SchemeT
    workspace :: WorkspaceT
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::Union{AbstractAdvection, AbstractAdvectionScheme};
                        workspace = AdvectionWorkspace(state.air_mass)) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid), typeof(advection), typeof(workspace)}(
        state, fluxes, grid, advection, workspace)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::Union{AbstractAdvection, AbstractAdvectionScheme};
                        workspace = AdvectionWorkspace(state.air_mass)) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid), typeof(advection), typeof(workspace)}(
        state, fluxes, grid, advection, workspace)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::Union{AbstractAdvection, AbstractAdvectionScheme};
                        workspace = AdvectionWorkspace(state.air_mass)) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh is metadata-only in src_v2; structured transport models are only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

function Adapt.adapt_structure(to, model::TransportModel)
    state = Adapt.adapt(to, model.state)
    fluxes = Adapt.adapt(to, model.fluxes)
    workspace = Adapt.adapt(to, model.workspace)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid), typeof(model.advection), typeof(workspace)}(
        state, fluxes, model.grid, model.advection, workspace)
end

function step!(model::TransportModel, dt)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace=model.workspace)
    return nothing
end

export TransportModel, step!
