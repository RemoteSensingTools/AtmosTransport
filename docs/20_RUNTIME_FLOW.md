# Runtime Flow

End-to-end walkthrough of how a tracer step is executed, from the
`DrivenSimulation` entry point down to the topology-dispatched
operator kernels.

## 30-second summary

```
DrivenSimulation.step!(sim)
  └─ _active_substep
  └─ _maybe_advance_window!(sim, substep)  ← load next transport window if needed
  └─ _refresh_forcing!(sim, substep)       ← driver window → model forcing + convection_forcing
  └─ default drivers: step!(sim.model, sim.Δt; meteo = sim)
     binary-scheduled drivers:
       ├─ transport_step!(sim.model, sim.Δt; meteo = sim)
       └─ at met-window boundary:
          convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
  └─ sim.time += sim.Δt; sim.iteration += 1; callbacks
```

All dispatch is multiple-dispatch on `typeof(grid.horizontal)` (the
mesh) and `typeof(operator)`. `NoDiffusion` / `NoSurfaceFlux` /
`NoConvection` / `NoChemistry` are compile-time dead branches — the
default paths are bit-exact to pre-operator behavior.

## Ownership

| What | Owner | Where it lives at runtime |
|---|---|---|
| File I/O, window timing, humidity endpoints, interpolation | Met driver | `sim.driver` (e.g. `TransportBinaryDriver`, `CubedSphereBinaryReader`) |
| Air mass, tracer masses, advection workspace, operator config | Model | `sim.model :: TransportModel` |
| Current simulation time | Simulation | `sim.time`, exposed via `current_time(sim)` |
| Per-window forcing payload | Driver-owned, model-consumed | `sim.window` (from driver) → `model.fluxes` + `model.convection_forcing` |

This separation is load-bearing: tracer state must not be entangled
with I/O policy.

## Step-by-step trace

### 1. `DrivenSimulation.step!(sim)`

Defined at [`src/Models/DrivenSimulation.jl:349`](../src/Models/DrivenSimulation.jl#L349). The entry point called by `run!(sim)` and by all driven-simulation scripts under `scripts/`.

```julia
substep = _active_substep(sim.iteration, sim.steps_per_window)
_maybe_advance_window!(sim, substep)
_refresh_forcing!(sim, substep)
if uses_binary_substep_contract(sim.driver)
    transport_step!(sim.model, sim.Δt; meteo = sim)
else
    step!(sim.model, sim.Δt; meteo = sim)
end
sim.time += sim.Δt
sim.iteration += 1
if uses_binary_substep_contract(sim.driver) &&
   sim.iteration == sim.current_window_end_iteration
    convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
end
for callback in values(sim.callbacks)
    callback(sim)
end
```

### 2. `_maybe_advance_window!`

Defined at [`src/Models/DrivenSimulation.jl:193`](../src/Models/DrivenSimulation.jl#L193). When the substep counter rolls past `steps_per_window`, the driver loads the next transport window from disk and replaces `sim.window`. No work on substeps within the current window.

### 3. `_refresh_forcing!`

Defined at [`src/Models/DrivenSimulation.jl:172`](../src/Models/DrivenSimulation.jl#L172). Populates the model's runtime forcing containers from the driver's window object:

- `model.fluxes` — interpolated mass fluxes `am`, `bm`, `cm` for the current substep (time-linear interpolation between window endpoints)
- `model.convection_forcing` — `cmfmc`, `dtrain`, surface pressure slices sliced to the substep; populated only when the driver supports CMFMC (`supports_cmfmc(sim.driver)`)

### 4. `step!(sim.model, sim.Δt; meteo = sim)`

Defined at [`src/Models/TransportModel.jl:335`](../src/Models/TransportModel.jl#L335). Three blocks, in order:

```julia
function step!(model::TransportModel, dt; meteo = nothing)
    transport_step!(model, dt; meteo = meteo)
    convection_chemistry_step!(model, dt; meteo = meteo)
    return nothing
end
```

For Plan-41 v3 transport binaries that declare
`runtime_substep_contract = "binary_schedule"`, `DrivenSimulation` calls
`transport_step!` on every stored binary substep and calls
`convection_chemistry_step!` once at the met-window boundary with
`dt = sim.window_dt`. The stored schedule is therefore the advection/transport
cadence, not a request to run convection and chemistry dozens of times per
hour.

The block helpers are:

```julia
function transport_step!(model::TransportModel, dt; meteo = nothing)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace.advection_ws,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo = meteo)
    return nothing
end

function convection_chemistry_step!(model::TransportModel, dt; meteo = nothing)
    if !(model.convection isa NoConvection)
        apply!(model.state, model.convection_forcing, model.grid,
               model.convection, dt;
               workspace = model.workspace.convection_ws)
    end
    apply!(model.state, meteo, model.grid, model.chemistry, dt)
    return nothing
end
```

Note that `meteo = sim` is passed, not `sim.driver`. Operators can reach the driver via `meteo.driver` when needed (e.g. for `supports_cmfmc(meteo.driver)`), and can ask `current_time(meteo)` which resolves to `sim.time`. Drivers are stateless and do not provide a clock.

### 5. Transport block — per-topology dispatch

The transport block's outer `apply!` dispatches on the mesh type:

- **`AtmosGrid{<:LatLonMesh}`** → rank-4 Strang palindrome in [`src/Operators/Advection/StrangSplitting.jl`](../src/Operators/Advection/StrangSplitting.jl). Sequence: `X → Y → Z → V(dt) → Z → Y → X`. Double-buffered ping-pong (Invariant 4) in `rm_A` / `m_A` and `rm_B` / `m_B`.
- **`AtmosGrid{<:ReducedGaussianMesh}`** → face-indexed `H → V(dt) → H` in the same `StrangSplitting.jl`; `@atomic` scatter kernel on rank-3 `tracers_raw`.
- **`AtmosGrid{<:CubedSphereMesh}`** → panel-oriented `strang_split_cs!` in [`src/Operators/Advection/CubedSphereStrang.jl`](../src/Operators/Advection/CubedSphereStrang.jl); halo exchanges between panels before each horizontal sweep.

Diffusion and surface flux are embedded at the Strang midpoint via the `V(dt)` call. `diffusion_op = NoDiffusion()` and `emissions_op = NoSurfaceFlux()` collapse to a bit-exact no-op path.

### 6. Convection block

Runs only when `model.convection !isa NoConvection`. `CMFMCConvection.apply!` has three topology dispatches in [`src/Operators/Convection/CMFMCConvection.jl`](../src/Operators/Convection/CMFMCConvection.jl):

- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:LatLonMesh}, ::CMFMCConvection, dt)` — rank-4 `tracers_raw`
- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:ReducedGaussianMesh}, ::CMFMCConvection, dt)` — rank-3 face-indexed `tracers_raw`
- `apply!(::CubedSphereState, ::ConvectionForcing, ::AtmosGrid{<:CubedSphereMesh}, ::CMFMCConvection, dt)` — `NTuple{6}` panel storage

A fourth dispatch rejects face-indexed state on non-RG grids to catch configuration mistakes.

Forcing refresh is upstream in `_refresh_forcing!`. The kernel consumes `convection_forcing.cmfmc` / `.dtrain` / `.ps` and accumulates tendencies into `state.tracers`.

### 7. Chemistry block

`apply!(state, meteo, grid, chemistry, dt)` dispatches on `typeof(chemistry)`. Operators: `ExponentialDecay` and `CompositeChemistry` support `CellState` (LatLon and RG) and `CubedSphereState`.

### 8. Callbacks

After `step!(sim.model, sim.Δt)` returns, each callback in `sim.callbacks` sees the post-step state. Common callbacks: output writers, mass diagnostics, CFL reporters.

## Time interpolation

Time interpolation belongs upstream of the kernels. The kernels
themselves are time-agnostic: they see a snapshot forcing state per
substep and produce a tendency.

Substep interpolation is done in `_refresh_forcing!` by linearly
blending between window endpoints using the fraction
`(substep - 0.5) / steps_per_window`. `AbstractTimeVaryingField`
implementations (`ConstantField`, `ProfileKzField`, `StepwiseField`,
`DerivedKzField`, `PreComputedKzField`) all honor this convention via
their `update_field!` hooks when they implement it.

## Closure policy

All carrier-mass conversion (wet → dry) and continuity closure
(`cm` diagnosis from horizontal divergence) happens in **preprocessing**,
not at runtime. The transport binary ships dry-basis, mass-balanced
fluxes ready to consume (Invariants 13 and 14). See
[`30_BINARY_AND_DRIVERS.md`](30_BINARY_AND_DRIVERS.md) for the binary
contract.

The `DryFluxBuilder` runtime converter (`src/MetDrivers/ERA5/DryFluxBuilder.jl`)
is retained for backward compatibility with old moist-basis binaries
only.

## Related docs

- [`10_CORE_CONTRACTS.md`](10_CORE_CONTRACTS.md) — State / flux / driver contracts
- [`30_BINARY_AND_DRIVERS.md`](30_BINARY_AND_DRIVERS.md) — Transport binary format
- [`35_RUNTIME_STABILITY_AND_SUBCYCLING.md`](35_RUNTIME_STABILITY_AND_SUBCYCLING.md) — CFL pilots and subcycling
- [`../src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md) — Per-operator dispatch matrix
- [`../src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md) — Per-operator dispatch matrix
- [`reference/ARCHITECTURE.md`](reference/ARCHITECTURE.md) — Architecture overview
