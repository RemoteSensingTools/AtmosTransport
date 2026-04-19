"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src` transport runs.

Post plan 15: carries both transport (advection) and chemistry operators.
Post plan 16b Commit 5: also carries the vertical diffusion operator.
`step!(model, dt)` composes the three per `OPERATOR_COMPOSITION.md` §3.1:

    transport_block(dt)   →   chemistry_block(dt)

where diffusion rides inside the transport block at the palindrome
center (plan 16b Commit 4):

    X → Y → Z → V(dt) → Z → Y → X

Default `chemistry = NoChemistry()` and `diffusion = NoDiffusion()`
make both blocks no-ops and keep pre-16b behaviour for callers that
don't opt in.
"""
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT, ChemT, DiffT}
    state     :: StateT
    fluxes    :: FluxT
    grid      :: GridT
    advection :: SchemeT
    workspace :: WorkspaceT
    chemistry :: ChemT
    diffusion :: DiffT
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion)}(
        state, fluxes, grid, advection, workspace, chemistry, diffusion)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion)}(
        state, fluxes, grid, advection, workspace, chemistry, diffusion)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion()) where {B <: AbstractMassBasis}
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
                          typeof(model.workspace), typeof(chemistry),
                          typeof(model.diffusion)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, chemistry, model.diffusion)
end

"""
    with_diffusion(model::TransportModel, diffusion)

Return a copy of `model` with its diffusion operator replaced. All other
fields share storage with the original. Parallel to [`with_chemistry`](@ref);
useful for installing a diffusion operator into a model that was
constructed with the default `NoDiffusion()`.
"""
function with_diffusion(model::TransportModel, diffusion::AbstractDiffusionOperator)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(diffusion)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, diffusion)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state     = Adapt.adapt(to, model.state)
    fluxes    = Adapt.adapt(to, model.fluxes)
    workspace = Adapt.adapt(to, model.workspace)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid),
                          typeof(model.advection), typeof(workspace),
                          typeof(model.chemistry), typeof(model.diffusion)}(
        state, fluxes, model.grid, model.advection, workspace,
        model.chemistry, model.diffusion)
end

"""
    step!(model::TransportModel, dt)

Advance `model.state` by one step: transport block (advection with
vertical diffusion at the palindrome center) → chemistry block.

With default `diffusion = NoDiffusion()` the advection apply! call
dispatches to the dead `NoDiffusion` branch, matching pre-16b
behavior bit-exactly. With default `chemistry = NoChemistry()` the
chemistry block is a no-op. Mirrors `OPERATOR_COMPOSITION.md` §3.1
with `N_chem_substeps = 1`.
"""
function step!(model::TransportModel, dt)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion)
    chemistry_block!(model.state, nothing, model.grid, model.chemistry, dt)
    return nothing
end

export TransportModel, step!, with_chemistry, with_diffusion
