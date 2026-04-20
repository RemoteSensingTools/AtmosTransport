"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src` transport runs.

Carries advection, chemistry, vertical diffusion, and surface emissions
operators. `step!(model, dt)` composes them per
`OPERATOR_COMPOSITION.md` §3.1:

    transport_block(dt)   →   chemistry_block(dt)

where `transport_block` runs the full palindrome with diffusion and
emissions at the center (plan 16b Commit 4 + plan 17 Commit 5):

    X → Y → Z → V(dt/2) → S(dt) → V(dt/2) → Z → Y → X      (emissions active)
    X → Y → Z → V(dt) → Z → Y → X                          (no emissions; bit-exact pre-17)

Defaults `chemistry = NoChemistry()`, `diffusion = NoDiffusion()`,
`emissions = NoSurfaceFlux()` make all three blocks no-ops and keep
pre-refactor behaviour for callers that don't opt in.

Plan 17 Commit 6 added the `emissions` field. The corresponding
`with_emissions` helper, paired with the `DrivenSimulation` cleanup
that removes the plan-15 `with_chemistry(model, NoChemistry())`
workaround, puts `advection → emissions → chemistry` on the natural
TM5 ordering without sim-level sleight-of-hand.
"""
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT, ChemT, DiffT, EmT}
    state     :: StateT
    fluxes    :: FluxT
    grid      :: GridT
    advection :: SchemeT
    workspace :: WorkspaceT
    chemistry :: ChemT
    diffusion :: DiffT
    emissions :: EmT
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion), typeof(emissions)}(
        state, fluxes, grid, advection, workspace, chemistry, diffusion, emissions)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion), typeof(emissions)}(
        state, fluxes, grid, advection, workspace, chemistry, diffusion, emissions)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux()) where {B <: AbstractMassBasis}
    throw(ArgumentError("CubedSphereMesh is metadata-only in src; structured transport models are only supported on LatLonMesh until cubed-sphere geometry/connectivity are implemented"))
end

"""
    with_chemistry(model::TransportModel, chemistry)

Return a copy of `model` with its chemistry operator replaced. All other
fields share storage with the original. Used by `DrivenSimulation` up
through plan 15 to install the sim-level chemistry operator into the
model; plan 17 Commit 6 removed the sim-level workaround so this helper
is now primarily useful for tests that want to swap chemistry on a
constructed model.
"""
function with_chemistry(model::TransportModel, chemistry::AbstractChemistryOperator)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(chemistry),
                          typeof(model.diffusion), typeof(model.emissions)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, chemistry, model.diffusion, model.emissions)
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
                          typeof(diffusion), typeof(model.emissions)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, diffusion, model.emissions)
end

"""
    with_emissions(model::TransportModel, emissions)

Return a copy of `model` with its surface-emissions operator replaced.
All other fields share storage with the original. Parallel to
[`with_chemistry`](@ref) and [`with_diffusion`](@ref); used by
`DrivenSimulation` (plan 17 Commit 6) to install the sim-level
`surface_sources` tuple as a `SurfaceFluxOperator` inside the wrapped
model, so the palindrome's S slot runs at the right place in the
transport block without sim-level post-step hacks.
"""
function with_emissions(model::TransportModel, emissions::AbstractSurfaceFluxOperator)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(emissions)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, emissions)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state     = Adapt.adapt(to, model.state)
    fluxes    = Adapt.adapt(to, model.fluxes)
    workspace = Adapt.adapt(to, model.workspace)
    emissions = Adapt.adapt(to, model.emissions)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid),
                          typeof(model.advection), typeof(workspace),
                          typeof(model.chemistry), typeof(model.diffusion),
                          typeof(emissions)}(
        state, fluxes, model.grid, model.advection, workspace,
        model.chemistry, model.diffusion, emissions)
end

"""
    step!(model::TransportModel, dt; meteo = nothing)

Advance `model.state` by one step: transport block (advection with
vertical diffusion at the palindrome center, surface emissions
wrapped by the two V half-steps when active) → chemistry block.

With defaults `diffusion = NoDiffusion()`, `emissions = NoSurfaceFlux()`,
`chemistry = NoChemistry()`, every component is a dead branch and the
call is bit-exact equivalent to pre-refactor advection.

`meteo` is optional and defaults to `nothing`; pass a real meteorology
object (`AbstractMetDriver`) to thread `current_time(meteo)` through
operators that consume time-varying fields (plan 17 Commit 4).
"""
function step!(model::TransportModel, dt; meteo = nothing)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)
    chemistry_block!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end

export TransportModel, step!, with_chemistry, with_diffusion, with_emissions
