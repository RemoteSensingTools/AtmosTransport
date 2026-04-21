# Advection

Tracer advection via finite-volume Strang splitting.

This folder owns the transport core: reconstruction, limiter logic,
structured and face-indexed sweeps, cubed-sphere panel transport, and
the model-facing `apply!` entrypoints that the transport block calls.

## Entry Points

- Scheme hierarchy:
  [`schemes.jl`](schemes.jl)
  defines `AbstractAdvectionScheme`, `UpwindScheme`, `SlopesScheme`,
  and `PPMScheme`
- Structured and face-indexed runtime orchestrators:
  [`StrangSplitting.jl`](StrangSplitting.jl)
  provides `strang_split!`, `strang_split_mt!`, and `apply!`
- Cubed-sphere runtime orchestrator:
  [`CubedSphereStrang.jl`](CubedSphereStrang.jl)
  provides `strang_split_cs!` and `CSAdvectionWorkspace`
- Cubed-sphere halo support:
  [`HaloExchange.jl`](HaloExchange.jl)
  provides `fill_panel_halos!` and `copy_corners!`
- Vertical mass-flux diagnosis:
  [`Divergence.jl`](Divergence.jl)
  provides `diagnose_cm!`

## Runtime Shape

- LatLon and reduced-Gaussian transport run through
  [`StrangSplitting.jl`](StrangSplitting.jl)
- Cubed-sphere transport runs through
  [`CubedSphereStrang.jl`](CubedSphereStrang.jl)
- Diffusion and surface flux are not separate outer blocks here; they are
  threaded into the palindrome midpoint through imports from
  `../Diffusion` and `../SurfaceFlux`
- This folder is the main place where topology-specific transport
  execution diverges while the public operator API stays uniform

## File Map

- [`Advection.jl`](Advection.jl) — submodule assembly, imports, include order
- [`schemes.jl`](schemes.jl) — scheme and limiter type hierarchy
- [`limiters.jl`](limiters.jl) — limiter formulas and slope controls
- [`reconstruction.jl`](reconstruction.jl) — face-value reconstruction
  logic shared by the sweep kernels
- [`structured_kernels.jl`](structured_kernels.jl) — structured-grid
  sweep kernels and helpers
- [`multitracer_kernels.jl`](multitracer_kernels.jl) — fused multi-tracer
  transport kernels and `TracerView`
- [`StrangSplitting.jl`](StrangSplitting.jl) — structured and
  face-indexed transport palindromes, model-facing `apply!`
- [`HaloExchange.jl`](HaloExchange.jl) — cubed-sphere panel-edge halo
  exchange and corner fill
- [`CubedSphereStrang.jl`](CubedSphereStrang.jl) — panel-native
  cubed-sphere palindrome
- [`ppm_subgrid_distributions.jl`](ppm_subgrid_distributions.jl) — PPM
  subcell distributions shared by CS-specific code
- [`LinRood.jl`](LinRood.jl) — Lin-Rood style cubed-sphere horizontal
  transport utilities
- [`VerticalRemap.jl`](VerticalRemap.jl) — conservative vertical remap
  helpers used by CS/FV3-style paths
- [`Divergence.jl`](Divergence.jl) — divergence and vertical-flux
  diagnosis utilities

## Common Tasks

- Adding a new advection scheme:
  start in [`schemes.jl`](schemes.jl), then wire reconstruction behavior
  in [`reconstruction.jl`](reconstruction.jl)
- Debugging mass conservation:
  read the palindrome structure in [`StrangSplitting.jl`](StrangSplitting.jl)
  or [`CubedSphereStrang.jl`](CubedSphereStrang.jl) before changing kernels
- Extending cubed-sphere transport:
  start with [`CubedSphereStrang.jl`](CubedSphereStrang.jl) and
  [`HaloExchange.jl`](HaloExchange.jl); keep the panel-native boundary honest
- Performance tuning:
  compare [`multitracer_kernels.jl`](multitracer_kernels.jl) against
  [`structured_kernels.jl`](structured_kernels.jl) before adding another
  launch path
- Debugging topology dispatch:
  read the `apply!` methods at the end of
  [`StrangSplitting.jl`](StrangSplitting.jl)

## Cross-Dependencies

- `../Diffusion` and `../SurfaceFlux` are imported into the palindrome
  midpoint; changes there can change advection runtime behavior
- [`../../State/`](../../State/) provides tracer slices, flux-state
  types, and the panel-native cubed-sphere containers
- [`../../Grids/`](../../Grids/) provides mesh geometry and panel
  connectivity
- [`../../MetDrivers/`](../../MetDrivers/) provides
  `diagnose_cm_from_continuity!`
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  chooses when this folder's `apply!` methods are executed

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Runtime/block ordering:
  [`../../../docs/plans/OPERATOR_COMPOSITION.md`](../../../docs/plans/OPERATOR_COMPOSITION.md)
- Runtime walkthrough:
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../../test/test_advection_kernels.jl`](../../../test/test_advection_kernels.jl)
  - [`../../../test/test_cubed_sphere_advection.jl`](../../../test/test_cubed_sphere_advection.jl)
  - [`../../../test/test_basis_explicit_core.jl`](../../../test/test_basis_explicit_core.jl)
  - [`../../../test/test_emissions_palindrome.jl`](../../../test/test_emissions_palindrome.jl)
  - [`../../../test/test_diffusion_palindrome.jl`](../../../test/test_diffusion_palindrome.jl)
