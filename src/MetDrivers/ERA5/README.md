# ERA5

ERA5-specific reader, geometry, closure, and dry-flux-building code.

This folder is the ERA5 specialization inside `MetDrivers`. It handles
preprocessed ERA5 binaries, native reduced-Gaussian geometry helpers,
vertical-flux diagnosis, and ERA5-specific dry-flux preparation.

## Entry Points

- ERA5 submodule assembly:
  [`ERA5.jl`](ERA5.jl)
- Preprocessed ERA5 binary reader:
  [`BinaryReader.jl`](BinaryReader.jl)
  defines `ERA5BinaryReader` and window-loading helpers
- Native reduced-Gaussian geometry helpers:
  [`NativeGRIBGeometry.jl`](NativeGRIBGeometry.jl)
- Vertical closure and diagnosed `cm` builders:
  [`VerticalClosure.jl`](VerticalClosure.jl)
- ERA5-specific dry-flux driver:
  [`DryFluxBuilder.jl`](DryFluxBuilder.jl)
  defines `PreprocessedERA5Driver`

## File Map

- [`ERA5.jl`](ERA5.jl) — submodule assembly and imports from the generic
  met-driver layer
- [`BinaryReader.jl`](BinaryReader.jl) — ERA5 binary header/window
  loading, feature flags, and per-window data access
- [`NativeGRIBGeometry.jl`](NativeGRIBGeometry.jl) — ERA5 reduced-Gaussian
  geometry recovery from native sources
- [`VerticalClosure.jl`](VerticalClosure.jl) — continuity-based vertical
  mass-flux diagnosis utilities
- [`DryFluxBuilder.jl`](DryFluxBuilder.jl) — ERA5-specific driver path
  that turns preprocessed fields into runtime fluxes

## Common Tasks

- Debugging an ERA5 window-content issue:
  start in [`BinaryReader.jl`](BinaryReader.jl)
- Debugging reduced-Gaussian geometry:
  read [`NativeGRIBGeometry.jl`](NativeGRIBGeometry.jl) before touching
  the mesh layer
- Debugging diagnosed vertical fluxes:
  start with [`VerticalClosure.jl`](VerticalClosure.jl)
- Following the ERA5 runtime path into a simulation:
  start with `PreprocessedERA5Driver` in [`DryFluxBuilder.jl`](DryFluxBuilder.jl),
  then jump to [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)

## Cross-Dependencies

- [`../README.md`](../README.md) provides the abstract driver contract and
  shared transport-binary infrastructure
- [`../../Grids/README.md`](../../Grids/README.md) provides `LatLonMesh`
  and `ReducedGaussianMesh`
- [`../../State/README.md`](../../State/README.md) provides `MetState`
  and flux-state types
- [`../../../docs/reference/METEO_PREPROCESSING.md`](../../../docs/reference/METEO_PREPROCESSING.md)
  and related reference docs explain the upstream data-prep assumptions

## Related Docs And Tests

- Reference docs:
  - [`../../../docs/reference/METEO_PREPROCESSING.md`](../../../docs/reference/METEO_PREPROCESSING.md)
  - [`../../../docs/reference/BINARY_FORMAT_V5.md`](../../../docs/reference/BINARY_FORMAT_V5.md)
  - [`../../../docs/reference/GRID_TYPES.md`](../../../docs/reference/GRID_TYPES.md)
- Tests:
  - [`../../../test/test_transport_binary_reader.jl`](../../../test/test_transport_binary_reader.jl)
  - [`../../../test/test_dry_flux_interface.jl`](../../../test/test_dry_flux_interface.jl)
  - [`../../../test/test_era5_latlon_e2e.jl`](../../../test/test_era5_latlon_e2e.jl)
  - [`../../../test/test_real_era5_direct_dry_binary.jl`](../../../test/test_real_era5_direct_dry_binary.jl)
  - [`../../../test/test_real_era5_dry_conversion.jl`](../../../test/test_real_era5_dry_conversion.jl)
