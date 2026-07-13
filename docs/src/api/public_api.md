# Curated public API

This page is the small top-level surface exposed by `using AtmosTransport`.
Everything else remains reachable through its owning submodule, for example
`AtmosTransport.Preprocessing.process_day` or
`AtmosTransport.Adjoints.cs_surface_emission_footprint`.

## Run a configured simulation

| Symbol | Use |
|---|---|
| `run_driven_simulation(cfg)` | Canonical high-level run entry point for a TOML-derived config dictionary. |
| `validate_config(cfg)` | Pre-flight common config mistakes before model allocation. |
| `expand_data_path(path)` | Resolve `~/...`, `$ATMOSTRANSPORT_DATA_ROOT/...`, and quickstart data-root paths. |
| `expand_binary_paths(input_cfg)` | Resolve `[input]` `binary_paths` or `folder + start_date + end_date`. |

```julia
using TOML, AtmosTransport

cfg = TOML.parsefile("config/runs/quickstart/ll72x37_advonly.toml")
ok, errors = validate_config(cfg)
ok || error(join(errors, "\n"))
model = run_driven_simulation(cfg)
```

## Inspect transport binaries

| Symbol | Use |
|---|---|
| `inspect_binary(path)` | Print header/capability information and return capability flags. |
| `binary_capabilities(reader)` | Summarize operator payload support for an open reader. |
| `TransportBinaryReader`, `TransportBinaryHeader` | Read lat-lon and reduced-Gaussian transport binaries. |
| `TransportBinaryDriver` | Runtime met driver for LL/RG transport binaries. |
| `load_transport_window(driver, i)` | Load a runtime forcing window. |
| `write_transport_binary(...)` | Write synthetic or preprocessed LL/RG transport binaries. |
| `total_windows`, `window_dt`, `steps_per_window`, `steps_per_window_schedule` | Inspect driver timing. |
| `grid_type`, `horizontal_topology` | Inspect binary topology metadata. |
| `supports_diffusion`, `supports_convection` | Check whether a driver can support requested physics. |

Cubed-sphere readers and drivers are also available at the top level.
Streaming writer helpers live there too because they are preprocessing internals.

## Work with grids and state

| Symbol | Use |
|---|---|
| `CPU`, `GPU` | Architecture tags for grids and drivers. |
| `earth_parameters` | Default planetary constants. |
| `AtmosGrid` | Combine horizontal mesh, vertical coordinate, and architecture. |
| `LatLonMesh`, `ReducedGaussianMesh`, `CubedSphereMesh` | Horizontal mesh constructors. |
| `HybridSigmaPressure` | Hybrid sigma-pressure vertical coordinate. |
| `nx`, `ny`, `ncells`, `nfaces`, `nlevels`, `nrings`, `cell_index` | Grid dimensions and indexing. |
| `cell_area`, `cell_faces`, `floattype` | Grid geometry and scalar type helpers. |
| `radius`, `gravity`, `reference_pressure` | Planet constants stored on a grid. |
| `pressure_at_interface`, `pressure_at_level`, `level_thickness` | Vertical coordinate helpers. |
| `CellState`, `CubedSphereState` | Store air mass and tracer masses. |
| `DryBasis`, `MoistBasis` | Make the air-mass basis explicit in state containers. |
| `allocate_face_fluxes`, `allocate_tracers` | Allocate state-compatible transport storage. |
| `get_tracer`, `mixing_ratio`, `total_mass`, `total_air_mass` | Common state diagnostics. |
| `tracer_names`, `ntracers` | Inspect tracer layout. |

## Configure physics and custom loops

| Symbol | Use |
|---|---|
| `AdvectionWorkspace`, `DiffusionWorkspace` | Independent scratch storage for low-level advection and diffusion calls. Model constructors allocate them automatically. |
| `UpwindScheme`, `SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme` | Common advection schemes. |
| `NoDiffusion`, `ImplicitVerticalDiffusion` | Diffusion operator choices. |
| `NoSurfaceFlux`, `SurfaceFluxOperator`, `SurfaceFluxSource` | Surface-flux operator stack. |
| `PerTracerFluxMap`, `flux_for` | Map emissions to named tracers. |
| `NoConvection`, `CMFMCConvection`, `TM5Convection` | Convection operator choices. |
| `NoChemistry`, `ExponentialDecay`, `CompositeChemistry` | Chemistry operator choices. |
| `ConstantField`, `ProfileKzField` | Simple time-varying fields used by physics operators. |
| `apply!` | Generic low-level operator application hook. |
| `TransportModel` | Low-level bundle of state, fluxes, grid, and operators. |
| `Simulation`, `DrivenSimulation` | Low-level harnesses for custom in-memory and window-driven loops. |
| `step!`, `run!`, `run_window!` | Low-level loop hooks for custom experiments. |
| `with_chemistry`, `with_diffusion`, `with_emissions` | Functional model-update helpers. |
| `build_runtime_physics_recipe`, `validate_runtime_physics_recipe` | Build and check physics recipes from config. |
| `build_initial_mixing_ratio`, `pack_initial_tracer_mass` | Initial-condition helpers. |
| `TransportTracerSpec` | Runtime tracer specification record. |

### Manual `DrivenSimulation` Assembly

Use this pattern when you want the runtime loop but need to own the model
construction. It is the advanced path behind `run_driven_simulation`, without
the wrapper's multi-file output, GPU adaptation, and progress plumbing.

```julia
using TOML, AtmosTransport
using AtmosTransport.MetDrivers: air_mass_basis, driver_grid

cfg = TOML.parsefile("config/runs/quickstart/ll72x37_advonly.toml")
paths = expand_binary_paths(cfg["input"])
FT = Float64

driver = TransportBinaryDriver(first(paths); FT = FT, arch = CPU())
recipe = build_runtime_physics_recipe(cfg, driver, FT)
validate_runtime_physics_recipe(recipe, driver)

grid = driver_grid(driver)
window1 = load_transport_window(driver, 1)
Basis = air_mass_basis(driver) === :dry ? DryBasis : MoistBasis
air = copy(window1.air_mass)

vmr = build_initial_mixing_ratio(
    air, grid, Dict("kind" => "uniform", "background" => 400e-6);
    surface_pressure = window1.surface_pressure,
)
co2 = pack_initial_tracer_mass(grid, air, vmr; mass_basis = Basis())

state = CellState(Basis, air; CO2 = co2)
fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid);
                              FT = FT, basis = Basis)
model = TransportModel(state, fluxes, grid, recipe.advection;
                       diffusion = recipe.diffusion,
                       convection = recipe.convection)

sim = DrivenSimulation(model, driver;
                       stop_window = min(total_windows(driver), 24),
                       chemistry = recipe.chemistry)
run_window!(sim)
```

The constructor validates grid and mass-basis compatibility before stepping.
`DrivenSimulation.step!` then refreshes forcing from the driver and delegates
the actual operator ordering to `TransportModel.step!`.

## Output, regridding, and visualization

| Symbol | Use |
|---|---|
| `capture_snapshot`, `write_snapshot_netcdf` | Capture and write diagnostic frames. |
| `build_regridder`, `apply_regridder!` | Build and apply offline conservative regridders. |
| `open_snapshot` | Open a written snapshot NetCDF. |
| `fieldview` | Extract a plottable field view from a snapshot. |
| `mapplot`, `movie` | Plot or animate snapshot fields when a Makie backend is loaded. |

## Advanced names

Internal tape records, checkpoint schedules, payload-section loaders, kernel
variants, and preprocessing drivers are intentionally not exported at the top
level. They are still part of the package namespace through their owning
modules, so existing advanced code can use explicit qualification:

```julia
AtmosTransport.MetDrivers.load_qv_window!(...)
AtmosTransport.Preprocessing.process_day(...)
AtmosTransport.Adjoints.cs_surface_emission_footprint(...)
```
