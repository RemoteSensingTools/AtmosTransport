# MetDrivers

Meteorological driver and transport-window infrastructure.

This folder is the bridge between external data and the runtime model.
It owns the abstract driver contract, binary readers, driver adapters,
mass-closure helpers, convection-forcing containers, and the ERA5- and
cubed-sphere-specific data paths.

## Entry Points

- Driver contract:
  [`AbstractMetDriver.jl`](AbstractMetDriver.jl)
  defines `AbstractMetDriver`, capability traits, and `current_time`
- Module assembly and exports:
  [`MetDrivers.jl`](MetDrivers.jl)
- Generic transport-binary reader:
  [`TransportBinary.jl`](TransportBinary.jl)
- Generic runtime driver over transport-binary windows:
  [`TransportBinaryDriver.jl`](TransportBinaryDriver.jl)
- Cubed-sphere binary reader:
  [`CubedSphereBinaryReader.jl`](CubedSphereBinaryReader.jl)
- Cubed-sphere runtime driver:
  [`CubedSphereTransportDriver.jl`](CubedSphereTransportDriver.jl)
- Convection forcing container:
  [`ConvectionForcing.jl`](ConvectionForcing.jl)
- ERA5-specific stack:
  [`ERA5/README.md`](ERA5/README.md)

## What This Folder Owns

- The abstract met-driver capability contract used by `DrivenSimulation`
- Binary/window readers and adapters that return runtime-friendly
  transport windows
- Mass-closure and dry-flux helper logic
- Driver-level capability traits such as `supports_diffusion` and
  `supports_convection`
- Simulation time plumbing via `current_time(meteo)`

## File Map

- [`MetDrivers.jl`](MetDrivers.jl) — submodule assembly and public exports
- [`AbstractMetDriver.jl`](AbstractMetDriver.jl) — required driver
  methods, traits, and default `current_time`
- [`MassClosure.jl`](MassClosure.jl) — vertical-closure strategy types
- [`DryFluxBuilder.jl`](DryFluxBuilder.jl) — generic dry-flux and air-mass
  assembly helpers
- [`ConvectionForcing.jl`](ConvectionForcing.jl) — convection forcing
  storage, adaptation, and copy helpers
- [`TransportBinary.jl`](TransportBinary.jl) — binary reader/writer and
  transport-window loading helpers
- [`TransportBinaryDriver.jl`](TransportBinaryDriver.jl) — structured and
  reduced-Gaussian runtime driver
- [`CubedSphereBinaryReader.jl`](CubedSphereBinaryReader.jl) — CS window reader
- [`CubedSphereTransportDriver.jl`](CubedSphereTransportDriver.jl) — CS runtime driver
- [`ERA5/`](ERA5/README.md) — ERA5-specific readers, geometry, closure,
  and dry-flux building

## Common Tasks

- Adding a new driver:
  start with [`AbstractMetDriver.jl`](AbstractMetDriver.jl), then decide
  whether the path is reader-first, driver-first, or both
- Debugging time-dependent operator inputs:
  check whether `current_time(meteo)` is coming from the sim or from the
  fallback driver stub
- Debugging a window-driven runtime issue:
  read `load_transport_window`, `driver_grid`, and the driver capability
  traits before touching `DrivenSimulation`
- Extending topology support:
  keep binary-reader semantics separate from runtime-driver semantics
  rather than overloading one path to do both

## Cross-Dependencies

- [`../Models/README.md`](../Models/README.md) consumes these drivers in
  `DrivenSimulation`
- [`../State/README.md`](../State/README.md) provides the state and
  flux-storage types the windows populate
- [`../Grids/README.md`](../Grids/README.md) provides mesh construction
  for `load_grid`
- [`../Operators/Convection/README.md`](../Operators/Convection/README.md)
  consumes `ConvectionForcing`

## Related Docs And Tests

- Binary/runtime walkthrough:
  [`../../docs/30_BINARY_AND_DRIVERS.md`](../../docs/30_BINARY_AND_DRIVERS.md)
- Runtime flow:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../test/test_transport_binary_reader.jl`](../../test/test_transport_binary_reader.jl)
  - [`../../test/test_transport_binary_v2_dispatch.jl`](../../test/test_transport_binary_v2_dispatch.jl)
  - [`../../test/test_run_transport_binary_v2.jl`](../../test/test_run_transport_binary_v2.jl)
  - [`../../test/test_dry_flux_interface.jl`](../../test/test_dry_flux_interface.jl)
  - [`../../test/test_current_time.jl`](../../test/test_current_time.jl)
  - [`../../test/test_cubed_sphere_runtime.jl`](../../test/test_cubed_sphere_runtime.jl)
