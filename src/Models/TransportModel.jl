"""
    TransportModel

Minimal Oceanigans-style model object for standalone `src` transport runs.

Carries advection, chemistry, vertical diffusion, surface emissions,
and convection operators. `step!(model, dt)` composes them per
`OPERATOR_COMPOSITION.md` §3.1 and plan 18 v5.1 §2.2 Decision 1:

    transport_block(dt)   →   convection_block(dt)   →   chemistry_block(dt)

where `transport_block` runs the full palindrome with diffusion and
emissions at the center (plan 16b Commit 4 + plan 17 Commit 5):

    X → Y → Z → V(dt/2) → S(dt) → V(dt/2) → Z → Y → X      (emissions active)
    X → Y → Z → V(dt) → Z → Y → X                          (no emissions; bit-exact pre-17)

Defaults `chemistry = NoChemistry()`, `diffusion = NoDiffusion()`,
`emissions = NoSurfaceFlux()`, `convection = NoConvection()` make all
four blocks no-ops and keep pre-refactor behaviour for callers that
don't opt in.

# Plan 18 Commit 2 additions

Two new fields beyond plan 17:

- `convection :: ConvT` — operator type, defaults to `NoConvection()`.
  Concrete subtypes (`CMFMCConvection`, `TM5Convection`) land in plan
  18 Commits 3 and 4. `NoConvection` is a compile-time dead branch
  in `step!` (Commit 6 wires the block).
- `convection_forcing :: CF` — per-step forcing container (plan 18
  v5.1 §2.17 Decision 23). Defaults to `ConvectionForcing()` (all-
  nothing placeholder). `DrivenSimulation` construction allocates
  real buffers via `allocate_convection_forcing_like`
  (§2.20 Decision 26); `_refresh_forcing!` populates them from
  `sim.window.convection` each substep.

Helpers `with_convection(model, op)` and
`with_convection_forcing(model, forcing)` parallel
`with_chemistry` / `with_diffusion` / `with_emissions`.
"""
struct TransportModel{StateT, FluxT, GridT, SchemeT, WorkspaceT,
                       ChemT, DiffT, EmT, ConvT, CF}
    state              :: StateT
    fluxes             :: FluxT
    grid               :: GridT
    advection          :: SchemeT
    workspace          :: WorkspaceT
    chemistry          :: ChemT
    diffusion          :: DiffT
    emissions          :: EmT
    convection         :: ConvT     # plan 18 Commit 2 — default NoConvection()
    convection_forcing :: CF        # plan 18 Commit 2 — default ConvectionForcing() placeholder
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:LatLonMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvectionOperator = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion), typeof(emissions),
                          typeof(convection), typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace,
        chemistry, diffusion, emissions, convection, convection_forcing)
end

function TransportModel(state::CellState{B},
                        fluxes::FaceIndexedFluxState{B},
                        grid::AtmosGrid,
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state; mesh=grid.horizontal),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvectionOperator = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
    return TransportModel{typeof(state), typeof(fluxes), typeof(grid),
                          typeof(advection), typeof(workspace),
                          typeof(chemistry), typeof(diffusion), typeof(emissions),
                          typeof(convection), typeof(convection_forcing)}(
        state, fluxes, grid, advection, workspace,
        chemistry, diffusion, emissions, convection, convection_forcing)
end

function TransportModel(state::CellState{B},
                        fluxes::StructuredFaceFluxState{B},
                        grid::AtmosGrid{<:CubedSphereMesh},
                        advection::AbstractAdvectionScheme;
                        workspace = AdvectionWorkspace(state),
                        chemistry::AbstractChemistryOperator = NoChemistry(),
                        diffusion::AbstractDiffusionOperator = NoDiffusion(),
                        emissions::AbstractSurfaceFluxOperator = NoSurfaceFlux(),
                        convection::AbstractConvectionOperator = NoConvection(),
                        convection_forcing::ConvectionForcing = ConvectionForcing()) where {B <: AbstractMassBasis}
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
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, chemistry, model.diffusion, model.emissions,
        model.convection, model.convection_forcing)
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
                          typeof(diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, diffusion, model.emissions,
        model.convection, model.convection_forcing)
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
                          typeof(model.diffusion), typeof(emissions),
                          typeof(model.convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, emissions,
        model.convection, model.convection_forcing)
end

"""
    with_convection(model::TransportModel, convection)

Return a copy of `model` with its convection operator replaced
(plan 18 Commit 2). All other fields — including
`convection_forcing` — share storage with the original.

Note: `with_convection` does NOT allocate convection-forcing
buffers. The model-side `ConvectionForcing()` placeholder stays
as-is. `DrivenSimulation` construction (plan 18 Commit 8) is
responsible for allocating real buffers via
`allocate_convection_forcing_like` after the first window loads
(plan 18 v5.1 §2.20 Decision 26). For tests that bypass the sim
layer, use `with_convection_forcing(model, forcing)` to inject
allocated buffers directly.
"""
function with_convection(model::TransportModel, convection::AbstractConvectionOperator)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(convection), typeof(model.convection_forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, model.emissions,
        convection, model.convection_forcing)
end

"""
    with_convection_forcing(model::TransportModel, forcing::ConvectionForcing)

Return a copy of `model` with its per-step convection-forcing
container replaced (plan 18 Commit 2). All other fields — including
the `convection` operator — share storage with the original.

Used by `DrivenSimulation` construction (plan 18 Commit 8) to
install the allocated forcing buffers after the first window loads.
Also useful for tests that inject forcing directly without going
through the sim's `_refresh_forcing!` path.
"""
function with_convection_forcing(model::TransportModel, forcing::ConvectionForcing)
    return TransportModel{typeof(model.state), typeof(model.fluxes),
                          typeof(model.grid), typeof(model.advection),
                          typeof(model.workspace), typeof(model.chemistry),
                          typeof(model.diffusion), typeof(model.emissions),
                          typeof(model.convection), typeof(forcing)}(
        model.state, model.fluxes, model.grid, model.advection,
        model.workspace, model.chemistry, model.diffusion, model.emissions,
        model.convection, forcing)
end

function Adapt.adapt_structure(to, model::TransportModel)
    state              = Adapt.adapt(to, model.state)
    fluxes             = Adapt.adapt(to, model.fluxes)
    workspace          = Adapt.adapt(to, model.workspace)
    emissions          = Adapt.adapt(to, model.emissions)
    convection_forcing = Adapt.adapt(to, model.convection_forcing)
    return TransportModel{typeof(state), typeof(fluxes), typeof(model.grid),
                          typeof(model.advection), typeof(workspace),
                          typeof(model.chemistry), typeof(model.diffusion),
                          typeof(emissions), typeof(model.convection),
                          typeof(convection_forcing)}(
        state, fluxes, model.grid, model.advection, workspace,
        model.chemistry, model.diffusion, emissions,
        model.convection, convection_forcing)
end

"""
    step!(model::TransportModel, dt; meteo = nothing)

Advance `model.state` by one step: transport block (advection with
vertical diffusion at the palindrome center, surface emissions
wrapped by the two V half-steps when active) → chemistry block.

With defaults `diffusion = NoDiffusion()`, `emissions = NoSurfaceFlux()`,
`chemistry = NoChemistry()`, `convection = NoConvection()`, every
component is a dead branch and the call is bit-exact equivalent to
pre-refactor advection.

The plan 18 convection block is wired into `step!` in plan 18
Commit 6. Until then, `convection` and `convection_forcing` are
carried on the struct but not read by `step!` — installing a
non-default convection operator has no effect in Commit 2.

`meteo` is optional and defaults to `nothing`; pass a real
meteorology object (`AbstractMetDriver`) or a `DrivenSimulation`
(plan 18 A3) to thread `current_time(meteo)` through operators
that consume time-varying fields.
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

export TransportModel, step!
export with_chemistry, with_diffusion, with_emissions
export with_convection, with_convection_forcing
