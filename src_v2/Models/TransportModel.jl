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
                        grid::AtmosGrid{<:AbstractStructuredMesh},
                        advection::AbstractAdvection;
                        workspace = AdvectionWorkspace(state.air_mass)) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid), typeof(advection), typeof(workspace)}(
        state, fluxes, grid, advection, workspace)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvection;
                        workspace = AdvectionWorkspace(state.air_mass)) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid), typeof(advection), typeof(workspace)}(
        state, fluxes, grid, advection, workspace)
end

function step!(model::TransportModel, dt)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace=model.workspace)
    return nothing
end

export TransportModel, step!
